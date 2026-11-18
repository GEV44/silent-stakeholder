"""Plain-language briefing built from a validated artifact directory."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.io_utils import ArtifactIOError, read_json

_VERDICT_TEXT = {
    "IGNORED": "no roadmap item addresses this need",
    "MISUNDERSTOOD": "a roadmap item exists but frames the problem differently",
    "UNDER-PRIORITIZED": "planned, but ranked below what users report",
    "COVERED": "the roadmap already covers this need",
}


def _load(directory: Path, name: str) -> Mapping[str, Any]:
    path = directory / name
    if not path.is_file():
        raise ArtifactIOError(f"{name} not found in {directory}; run the pipeline or demo first")
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ArtifactIOError(f"{name} must contain a JSON object")
    return payload


def load_briefing(artifacts: str | Path) -> dict[str, Any]:
    """Join ranked gaps with roadmap titles into a flat, presentation-ready structure."""

    directory = Path(artifacts)
    gaps = _load(directory, "top_gaps.json").get("gaps")
    if not isinstance(gaps, list):
        raise ArtifactIOError("top_gaps.json: 'gaps' must be a list")
    roadmap_payload = _load(directory, "roadmap.json").get("roadmap")
    if not isinstance(roadmap_payload, list):
        raise ArtifactIOError("roadmap.json: 'roadmap' must be a list")
    titles = {
        str(item.get("id")): item.get("title")
        for item in roadmap_payload
        if isinstance(item, Mapping)
    }
    findings: list[dict[str, Any]] = []
    for gap in gaps:
        if not isinstance(gap, Mapping):
            raise ArtifactIOError("top_gaps.json: every gap must be an object")
        evidence = gap.get("evidence") or {}
        quotes = evidence.get("quotes") or []
        findings.append(
            {
                "rank": gap.get("rank"),
                "need": gap.get("latent_need"),
                "jtbd": gap.get("jtbd"),
                "verdict": gap.get("verdict"),
                "rank_score": float(gap.get("rank_score") or 0.0),
                "roadmap_id": gap.get("matched_roadmap_id"),
                "roadmap_title": titles.get(str(gap.get("matched_roadmap_id"))),
                "evidence_count": len(evidence.get("signal_ids") or quotes),
                "quote": quotes[0].get("span") if quotes else None,
                "calibrated": gap.get("calibrated_confidence") is not None,
            }
        )
    return {"findings": findings, "calibrated": any(f["calibrated"] for f in findings)}


def render_markdown(briefing: Mapping[str, Any]) -> str:
    lines = ["# Silent Stakeholder briefing", ""]
    note = (
        "Scores are calibrated against adjudicated labels."
        if briefing["calibrated"]
        else "Scores are uncalibrated evidence scores, not probabilities."
    )
    lines += [note, ""]
    for f in briefing["findings"]:
        lines += [
            f"## {f['rank']}. {f['need']}",
            "",
            f"- Verdict: **{f['verdict']}** ({_VERDICT_TEXT.get(str(f['verdict']), 'see report')})",
            f"- Score: {f['rank_score']:.2f} from {f['evidence_count']} verified user signals",
            f"- Closest roadmap item: {f['roadmap_id']} {f['roadmap_title'] or '(not found)'}",
            f"- Job to be done: {f['jtbd']}",
        ]
        if f["quote"]:
            lines.append(f'- Voice of the user: "{f["quote"]}"')
        lines.append("")
    return "\n".join(lines)
