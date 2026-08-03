from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.io_utils import ArtifactIOError, atomic_write_json, make_stable_id, read_json
from src.schema import (
    EvidenceQuote,
    Gap,
    GapEvidence,
    GapVerdict,
    LatentNeed,
    PriorityMetadata,
    RoadmapItem,
    RoadmapItemType,
    RoadmapState,
    Signal,
)


def test_stable_id_is_order_independent_and_unicode_canonical() -> None:
    first = make_stable_id("S", " app_review ", "Ｃafé   export")
    second = make_stable_id("S", "APP_REVIEW", "Café export")

    assert first == second
    assert first.startswith("S")
    assert len(first) == 13


def test_stable_id_rejects_bad_identity() -> None:
    with pytest.raises(ValueError, match="prefix"):
        make_stable_id("X", "identity")
    with pytest.raises(ValueError, match="identity"):
        make_stable_id("S", "  ")


def test_signal_normalizes_naive_timestamp_to_utc() -> None:
    signal = Signal(
        id=make_stable_id("S", "review", 1),
        text="Export is slow",
        timestamp=datetime(2017, 1, 2, 3, 4, 5),
        rating=2,
    )

    assert signal.timestamp is not None
    assert signal.timestamp.tzinfo == UTC


def test_signal_rejects_unknown_fields_and_out_of_range_rating() -> None:
    with pytest.raises(ValidationError):
        Signal(
            id=make_stable_id("S", "review", 1),
            text="hello",
            rating=6,
            surprise=True,
        )


def test_roadmap_milestone_cannot_belong_to_itself() -> None:
    with pytest.raises(ValidationError, match="cannot themselves belong"):
        RoadmapItem(
            id=make_stable_id("R", "repo", "milestone", 1),
            type=RoadmapItemType.MILESTONE,
            repository="owner/repo",
            number=1,
            title="v1",
            state=RoadmapState.OPEN,
            milestone="v0",
            priority=PriorityMetadata(),
        )


def test_gap_evidence_rejects_undeclared_quote_id() -> None:
    declared = make_stable_id("S", "declared")
    undeclared = make_stable_id("S", "undeclared")

    with pytest.raises(ValidationError, match="quote IDs"):
        GapEvidence(
            signal_ids=[declared],
            quotes=[EvidenceQuote(id=undeclared, span="quote")],
        )


def test_end_to_end_analysis_models_accept_contract() -> None:
    signal_id = make_stable_id("S", "review")
    need_id = make_stable_id("N", "need")
    need = LatentNeed(
        id=need_id,
        latent_need="Reliable offline editing",
        jtbd_statement="Edit drafts without a dependable connection",
        supporting_signal_ids=[signal_id],
    )
    gap = Gap(
        id=make_stable_id("G", need.id),
        need_id=need.id,
        latent_need=need.latent_need,
        jtbd=need.jtbd_statement,
        verdict=GapVerdict.IGNORED,
        evidence={
            "signal_ids": [signal_id],
            "quotes": [{"id": signal_id, "span": "offline"}],
        },
        features={"vol": 0.5, "cons": 1.0},
        rank=1,
    )

    assert gap.evidence.quotes[0].id == signal_id
    assert gap.rank == 1


def test_atomic_json_write_serializes_models_and_replaces_existing(tmp_path) -> None:
    target = tmp_path / "nested" / "artifact.json"
    target.parent.mkdir()
    target.write_text('{"old": true}\n', encoding="utf-8")
    signal = Signal(id=make_stable_id("S", "one"), text="Hello")

    returned = atomic_write_json(target, {"signals": [signal]})

    assert returned == target
    assert read_json(target)["signals"][0]["text"] == "Hello"
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_read_json_reports_parse_location(tmp_path) -> None:
    target = tmp_path / "bad.json"
    target.write_text('{"broken": }\n', encoding="utf-8")

    with pytest.raises(ArtifactIOError, match=r"line 1, column"):
        read_json(target)


def test_atomic_json_rejects_non_finite_values(tmp_path) -> None:
    with pytest.raises(ArtifactIOError, match="NaN"):
        atomic_write_json(tmp_path / "bad.json", {"value": float("nan")})
