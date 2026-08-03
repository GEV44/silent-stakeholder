"""Canonical, ID-grounded data contracts for every pipeline stage.

The models in this module deliberately reject unknown fields. A cached artifact
that no longer matches the code should fail at its boundary instead of silently
dropping evidence or changing the meaning of a score.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ArtifactId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[SRNG](?:[0-9a-f]{12}|\d{4,})$",
    ),
]
SignalId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^S(?:[0-9a-f]{12}|\d{4,})$",
    ),
]
RoadmapId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^R(?:[0-9a-f]{12}|\d{4,})$",
    ),
]
NeedId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^N(?:[0-9a-f]{12}|\d{4,})$",
    ),
]
GapId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^G(?:[0-9a-f]{12}|\d{4,})$",
    ),
]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    """Shared settings for durable on-disk artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        validate_default=True,
    )


class RoadmapItemType(StrEnum):
    ISSUE = "issue"
    MILESTONE = "milestone"


class RoadmapState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class PriorityTier(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKLOG = "backlog"
    UNSPECIFIED = "unspecified"


class KanoClass(StrEnum):
    BASIC = "basic"
    PERFORMANCE = "performance"
    EXCITEMENT = "excitement"
    INDIFFERENT = "indifferent"
    REVERSE = "reverse"
    UNKNOWN = "unknown"


class GapVerdict(StrEnum):
    IGNORED = "IGNORED"
    UNDER_PRIORITIZED = "UNDER-PRIORITIZED"
    MISUNDERSTOOD = "MISUNDERSTOOD"
    COVERED = "COVERED"


class CritiqueVerdict(StrEnum):
    DEFENSIBLE = "DEFENSIBLE"
    WEAK = "WEAK"
    UNSUPPORTED = "UNSUPPORTED"


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class Signal(StrictModel):
    """One atomic user-signal record.

    ``id`` is assigned before analysis and is the only identifier later stages
    may cite. ``source_id`` preserves the upstream provider's identifier when
    one exists but is never used as an evidence ID.
    """

    id: SignalId
    source: NonEmptyText = "app_review"
    source_id: str | None = None
    text: NonEmptyText
    timestamp: datetime | None = None
    rating: float | None = Field(default=None, ge=1.0, le=5.0)
    package_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    _normalize_timestamp = field_validator("timestamp")(_as_utc)

    @field_validator("source_id", "package_name", mode="before")
    @classmethod
    def empty_string_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class PriorityMetadata(StrictModel):
    """Inspectible metadata behind a roadmap priority judgment."""

    tier: PriorityTier = PriorityTier.UNSPECIFIED
    score: UnitFloat = 0.5
    is_low_priority: bool = False
    has_explicit_priority: bool = False
    reasons: list[str] = Field(default_factory=list)
    matched_labels: list[str] = Field(default_factory=list)

    @field_validator("reasons", "matched_labels")
    @classmethod
    def unique_nonempty_strings(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = " ".join(value.split())
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                result.append(normalized)
        return result

    @model_validator(mode="before")
    @classmethod
    def low_priority_agrees_with_tier(cls, data: Any) -> Any:
        if isinstance(data, dict):
            tier = data.get("tier", PriorityTier.UNSPECIFIED)
            if tier in {
                PriorityTier.LOW,
                PriorityTier.BACKLOG,
                PriorityTier.LOW.value,
                PriorityTier.BACKLOG.value,
            }:
                data = {**data, "is_low_priority": True}
        return data


class RoadmapItem(StrictModel):
    """A normalized GitHub issue or milestone."""

    id: RoadmapId
    type: RoadmapItemType
    repository: NonEmptyText
    number: int = Field(ge=1)
    title: NonEmptyText
    body: str = ""
    state: RoadmapState
    state_reason: str | None = None
    labels: list[str] = Field(default_factory=list)
    milestone: str | None = None
    milestone_number: int | None = Field(default=None, ge=1)
    milestone_due: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    closed_at: datetime | None = None
    html_url: str | None = None
    priority: PriorityMetadata = Field(default_factory=PriorityMetadata)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _normalize_milestone_due = field_validator("milestone_due")(_as_utc)
    _normalize_created_at = field_validator("created_at")(_as_utc)
    _normalize_updated_at = field_validator("updated_at")(_as_utc)
    _normalize_closed_at = field_validator("closed_at")(_as_utc)

    @field_validator("repository")
    @classmethod
    def canonical_repository(cls, value: str) -> str:
        value = value.strip().strip("/")
        if not re.fullmatch(r"[^/\s]+/[^/\s]+", value):
            raise ValueError("repository must use the 'owner/name' form")
        return value

    @field_validator("labels")
    @classmethod
    def unique_labels(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = " ".join(value.split())
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                result.append(normalized)
        return result

    @field_validator("milestone", "html_url", "state_reason", mode="before")
    @classmethod
    def blank_optional_string_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def issue_and_milestone_fields_are_consistent(self) -> RoadmapItem:
        if self.type == RoadmapItemType.MILESTONE and (
            self.milestone is not None or self.milestone_number is not None
        ):
            raise ValueError("milestone items cannot themselves belong to a milestone")
        return self

    @property
    def priority_is_low(self) -> bool:
        """Compatibility helper used by the deterministic verdict gate."""

        return self.priority.is_low_priority


class LatentNeed(StrictModel):
    """A cluster-level, second-order user need grounded in signal IDs."""

    id: NeedId
    latent_need: NonEmptyText
    jtbd_statement: NonEmptyText
    kano_class: KanoClass = KanoClass.UNKNOWN
    root_cause_hypothesis: str = ""
    symptom: str = ""
    supporting_signal_ids: list[SignalId] = Field(min_length=1)
    cluster_id: int | str | None = None
    importance: float | None = Field(default=None, ge=0.0, le=10.0)
    satisfaction: float | None = Field(default=None, ge=0.0, le=10.0)
    opportunity_score: float | None = Field(default=None, ge=0.0, le=20.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("supporting_signal_ids")
    @classmethod
    def unique_signal_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class EvidenceQuote(StrictModel):
    id: SignalId
    span: NonEmptyText
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def offsets_are_ordered(self) -> EvidenceQuote:
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("quote end must be greater than or equal to quote start")
        return self


class RoadmapQuote(StrictModel):
    id: RoadmapId
    span: NonEmptyText
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def offsets_are_ordered(self) -> RoadmapQuote:
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("quote end must be greater than or equal to quote start")
        return self


class GapEvidence(StrictModel):
    signal_ids: list[SignalId] = Field(default_factory=list)
    quotes: list[EvidenceQuote] = Field(default_factory=list)
    roadmap_quote: RoadmapQuote | None = None

    @field_validator("signal_ids")
    @classmethod
    def unique_signal_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def quote_ids_are_declared_evidence(self) -> GapEvidence:
        undeclared = {quote.id for quote in self.quotes} - set(self.signal_ids)
        if undeclared:
            raise ValueError(
                "quote IDs must also appear in signal_ids: " + ", ".join(sorted(undeclared))
            )
        return self


class Gap(StrictModel):
    """A ranked roadmap gap with all judge-facing numbers made explicit."""

    id: GapId
    need_id: NeedId
    latent_need: NonEmptyText
    jtbd: NonEmptyText
    kano_class: KanoClass = KanoClass.UNKNOWN
    verdict: GapVerdict
    matched_roadmap_id: RoadmapId | None = None
    similarity: UnitFloat | None = None
    symptom_similarity: UnitFloat | None = None
    latent_similarity: UnitFloat | None = None
    calibrated_confidence: UnitFloat | None = None
    opportunity_score: float | None = Field(default=None, ge=0.0, le=20.0)
    rank_score: float | None = Field(default=None, ge=0.0)
    rank: int | None = Field(default=None, ge=1)
    evidence: GapEvidence = Field(default_factory=GapEvidence)
    critique: CritiqueVerdict | None = None
    why_rank: str | None = None
    features: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("features")
    @classmethod
    def feature_values_are_finite_unit_scores(cls, values: dict[str, float]) -> dict[str, float]:
        invalid = [
            name
            for name, value in values.items()
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0
        ]
        if invalid:
            raise ValueError(
                "features must contain numeric values in [0, 1]: " + ", ".join(sorted(invalid))
            )
        return {name: float(value) for name, value in values.items()}
