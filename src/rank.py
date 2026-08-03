"""Stage 5: stable evidence-strength ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"expected mapping or model, got {type(value).__name__}")


def _nested_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return _mapping(value)


def rank_score(
    calibrated_confidence: float,
    opportunity_score: float,
) -> float:
    """The documented rank key: calibrated confidence × ODI opportunity."""

    confidence = float(calibrated_confidence)
    opportunity = float(opportunity_score)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("calibrated_confidence must be in [0, 1]")
    if not 0.0 <= opportunity <= 20.0:
        raise ValueError("opportunity_score must be in [0, 20]")
    return confidence * opportunity


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def rank_gaps(
    gaps: Sequence[Mapping[str, Any] | Any],
    *,
    top_k: int = 5,
    min_confidence: float = 0.0,
    include_covered: bool = False,
    include_unsupported: bool = False,
    deduplicate_titles: bool = True,
) -> list[dict[str, Any]]:
    """Rank gaps without mutating inputs.

    Ties break on confidence, opportunity, evidence count, then stable gap ID.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    candidates: list[dict[str, Any]] = []
    for gap_value in gaps:
        gap = _mapping(gap_value)
        verdict = _enum_value(gap.get("verdict") or "")
        critique = _enum_value(gap.get("critique") or "")
        if verdict == "COVERED" and not include_covered:
            continue
        if critique == "UNSUPPORTED" and not include_unsupported:
            continue
        confidence_value = gap.get("calibrated_confidence")
        opportunity_value = gap.get("opportunity_score")
        if confidence_value is None or opportunity_value is None:
            continue
        confidence = float(confidence_value)
        opportunity = float(opportunity_value)
        if confidence < min_confidence:
            continue
        score = rank_score(confidence, opportunity)
        gap["rank_score"] = round(score, 6)
        candidates.append(gap)

    def ordering(gap: dict[str, Any]) -> tuple[float, float, float, int, str]:
        evidence = _nested_mapping(gap.get("evidence"))
        evidence_count = len(set(map(str, evidence.get("signal_ids") or [])))
        return (
            -float(gap["rank_score"]),
            -float(gap["calibrated_confidence"]),
            -float(gap["opportunity_score"]),
            -evidence_count,
            str(gap.get("id") or ""),
        )

    candidates.sort(key=ordering)
    if deduplicate_titles:
        unique: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for gap in candidates:
            key = " ".join(str(gap.get("latent_need") or "").casefold().split())
            if key and key in seen_titles:
                continue
            seen_titles.add(key)
            unique.append(gap)
        candidates = unique
    ranked = candidates[:top_k]
    for index, gap in enumerate(ranked, start=1):
        confidence = float(gap["calibrated_confidence"])
        opportunity = float(gap["opportunity_score"])
        score = float(gap["rank_score"])
        gap["rank"] = index
        gap["why_rank"] = (
            f"opportunity {opportunity:.2f} × calibrated confidence "
            f"{confidence:.3f} = {score:.3f}"
        )
        metadata = _nested_mapping(gap.get("metadata"))
        metadata["rank"] = index
        metadata["tie_break"] = (
            "rank_score, calibrated_confidence, opportunity_score, "
            "evidence_count, gap_id"
        )
        gap["metadata"] = metadata
    return ranked
