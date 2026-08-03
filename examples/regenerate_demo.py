"""Regenerate the synthetic demo fixture with real pipeline code.

`examples/CLAUDE.md` forbids hand-editing `examples/demo/`: a hand-edited fixture
drifts from the schema and starts claiming verdicts the code cannot emit. This
script is the sanctioned alternative. It holds the synthetic **inputs** only --
ExamplePress signals and roadmap records -- and every downstream claim (needs,
verdicts, coverage numbers, confidence, ranking) is produced by `src.run.analyze`,
the same function a real run uses.

    python -m examples.regenerate_demo

The corpus is engineered so the deterministic framing gate genuinely fires:
R000001 covers the media symptom's distinctive vocabulary (`fail`, `stall`,
`block`, `publish`, `workflow`) while no roadmap record covers the job's
(`background`, `recover`, `individual`, `failure`, `restart`). That asymmetry --
not a hand-typed string -- is what makes MISUNDERSTOOD appear.

Nothing here names a real company; the fixture product is ExamplePress.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.io_utils import atomic_write_json
from src.run import (
    _artifact_declarations,
    _artifact_input_hashes,
    _as_demo_manifest,
    analyze,
)
from src.schema import (
    Gap,
    PriorityMetadata,
    RoadmapItem,
    RoadmapItemType,
    RoadmapState,
    Signal,
)

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "examples" / "demo"

# Pinned so recency-sensitive scores never drift between regenerations.
AS_OF = datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC)
SIGNAL_TIME = datetime(2026, 6, 15, 9, 0, 0, tzinfo=UTC)

PACKAGE = "com.examplepress.android"
REPOSITORY = "ExamplePress/ExamplePress-Android"

# Each block's vocabulary steers the offline need frame (`_OFFLINE_DOMAINS` in
# src/needs.py) toward one domain. Six per block keeps every cluster above the
# scaled min_cluster_size (len(signals)//8 == 3).
_REVIEW_TEXT: list[tuple[str, float]] = [
    # -- media: upload/photo/image/gallery vocabulary ONLY.  Any drafting word
    # here (editor, publish, post, save) pulls the cluster into the drafting
    # domain and the media need never forms.
    ("Every photo upload stalls halfway and I must start the whole gallery upload again.", 2.0),
    ("Uploading images fails silently and the media gallery stays completely empty.", 1.0),
    ("The media upload queue freezes and I cannot upload any more photos at all.", 2.0),
    ("If one picture in a gallery upload fails, the entire batch of images is lost.", 1.0),
    ("Photo uploads stall on hotel wifi with no way to resume the media upload.", 2.0),
    ("Large image uploads freeze the whole app while the gallery uploads photos.", 2.0),
    ("The upload queue for photos and images stalls at ninety percent every time.", 1.0),
    ("Media uploads from my camera roll fail and the gallery shows broken images.", 1.0),
    # -- drafting ----------------------------------------------------------
    ("My draft disappeared after the editor tried to sync a page I had just saved.", 1.0),
    ("Edits to a long post do not save and the draft reverts to an older version.", 1.0),
    ("The editor lost a page when it tried to publish and then failed to sync.", 1.0),
    ("Drafts saved on the train never sync, so the post is gone when I open the editor.", 2.0),
    ("I publish a post and the editor silently drops the last paragraph I saved.", 2.0),
    ("Every long draft I edit fails to save and I retype the whole page again.", 1.0),
    # -- sites -------------------------------------------------------------
    ("I run multiple blogs and the dashboard constantly forgets which site I picked.", 2.0),
    ("Switching between sites is confusing; the stats shown belong to another blog.", 2.0),
    ("The site switcher resets to the wrong blog every time I open the dashboard.", 2.0),
    ("Managing multiple sites is painful because the dashboard stats never match.", 1.0),
    ("I manage six blogs and the dashboard shows one site's stats under another site.", 2.0),
    ("The dashboard for multiple sites mixes up which blog the stats belong to.", 2.0),
    # -- engagement --------------------------------------------------------
    (
        "A delayed comment notification opens the reply too late; notifications for "
        "comments need a timely reply.",
        2.0,
    ),
    ("A notification for a deleted comment appears with no reply option at all.", 2.0),
    ("Notifications for new comments never open the right comment thread on reply.", 2.0),
    (
        "A comment notification does not group comment replies, so the reply "
        "conversation is slow.",
        2.0,
    ),
    ("A reply notification opens the wrong comment and I lose the whole conversation.", 1.0),
    ("Comment replies I send from a notification silently vanish without warning.", 1.0),
]


def _signals() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (text, rating) in enumerate(_REVIEW_TEXT, start=1):
        rows.append(
            Signal(
                id=f"S{index:06d}",
                source="app_review",
                text=text,
                timestamp=SIGNAL_TIME,
                rating=rating,
                package_name=PACKAGE,
                metadata={"synthetic": True},
            ).model_dump(mode="json")
        )
    return rows


def _roadmap_item(
    number: int,
    title: str,
    body: str,
    labels: list[str],
    milestone: str | None,
    priority: dict[str, Any],
) -> dict[str, Any]:
    return RoadmapItem(
        id=f"R{number:06d}",
        type=RoadmapItemType.ISSUE,
        repository=REPOSITORY,
        number=100 + number,
        title=title,
        body=body,
        state=RoadmapState.OPEN,
        labels=labels,
        milestone=milestone,
        created_at=datetime(2025, 11, 4, tzinfo=UTC),
        updated_at=datetime(2026, 5, 20, tzinfo=UTC),
        html_url=f"https://example.invalid/{REPOSITORY}/issues/{100 + number}",
        priority=PriorityMetadata.model_validate(priority),
        metadata={"synthetic": True},
    ).model_dump(mode="json")


def _roadmap() -> list[dict[str, Any]]:
    return [
        # Covers the media SYMPTOM vocabulary (fail / stall / block / publish /
        # workflow) and none of the job's. This asymmetry is what the framing
        # gate detects; the words below are load-bearing, not decorative.
        _roadmap_item(
            1,
            "Warn the author when a media upload fails or stalls",
            "Show an error banner when uploads fail or stall and block the "
            "publishing workflow. Scope is the warning only.",
            ["media", "enhancement"],
            "Next release",
            {
                "tier": "unspecified",
                "score": 0.55,
                "has_explicit_priority": False,
                "reasons": ["assigned to a dated release milestone"],
                "matched_labels": ["enhancement"],
            },
        ),
        # Echoes the drafting need's job framing almost verbatim, so job
        # coverage is high and MISUNDERSTOOD cannot fire -- but it sits in the
        # backlog, which is what UNDER-PRIORITIZED means.
        _roadmap_item(
            2,
            "Lossless drafting and publishing",
            "Preserve edits across drafting, syncing, and publishing "
            "transitions so no saved page or post is dropped.",
            ["editor", "reliability"],
            "Future",
            {
                "tier": "backlog",
                "score": 0.15,
                "has_explicit_priority": True,
                "reasons": ["milestone 'Future' is an undated backlog bucket"],
                "matched_labels": ["reliability"],
            },
        ),
        _roadmap_item(
            3,
            "Refresh the onboarding illustrations",
            "Update the artwork shown on first launch to match the new brand palette.",
            ["design"],
            None,
            {
                "tier": "backlog",
                "score": 0.1,
                "has_explicit_priority": True,
                "reasons": ["no milestone assigned"],
                "matched_labels": [],
            },
        ),
        _roadmap_item(
            4,
            "Adopt the new typography scale",
            "Apply the updated heading and caption sizes across settings screens.",
            ["design"],
            "Next release",
            {
                "tier": "unspecified",
                "score": 0.5,
                "has_explicit_priority": False,
                "reasons": ["assigned to a dated release milestone"],
                "matched_labels": [],
            },
        ),
        _roadmap_item(
            5,
            "Reduce cold start time on low-end devices",
            "Profile and trim the startup path so first paint happens sooner.",
            ["performance"],
            "Next release",
            {
                "tier": "unspecified",
                "score": 0.6,
                "has_explicit_priority": False,
                "reasons": ["assigned to a dated release milestone"],
                "matched_labels": [],
            },
        ),
    ]


def main() -> int:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    signals, roadmap = _signals(), _roadmap()
    atomic_write_json(
        DEMO_DIR / "signals.json",
        {"schema_version": "1.0", "mode": "demo_fixture", "signals": signals},
    )
    atomic_write_json(
        DEMO_DIR / "roadmap.json",
        {"schema_version": "1.0", "mode": "demo_fixture", "roadmap": roadmap},
    )

    manifest = analyze(
        DEMO_DIR,
        DEMO_DIR,
        embedding_backend="hashing",
        use_llm=False,
        label_file=None,
        min_calibration_labels=100,
        top_k=5,
        as_of=AS_OF,
        analysis_mode="exploratory_snapshot",
        include_covered=False,
    )

    # `analyze` writes each artifact as a bare JSON list. The fixture wraps them
    # so `mode: "demo_fixture"` travels *with* the data: app.py must be able to
    # prove provenance from the file it loaded, not infer it from a sibling.
    # This is packaging only -- every verdict, score and ID inside is untouched
    # pipeline output.
    wrappers = {
        "signals.json": "signals",
        "roadmap.json": "roadmap",
        "needs.json": "needs",
        "gaps.json": "gaps",
        "top_gaps.json": "gaps",
        "verification.json": "verification",
    }
    for name, key in wrappers.items():
        path = DEMO_DIR / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get(key, payload)
        atomic_write_json(
            path,
            {"schema_version": "1.0", "mode": "demo_fixture", key: rows},
        )

    manifest_path = DEMO_DIR / "run_manifest.json"
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored = _as_demo_manifest(stored)
    stored["inputs"] = _artifact_input_hashes(DEMO_DIR)
    stored["artifacts"] = _artifact_declarations(DEMO_DIR)
    atomic_write_json(manifest_path, stored)

    # Rule 1 of examples/CLAUDE.md: validate before committing.
    top = json.loads((DEMO_DIR / "top_gaps.json").read_text(encoding="utf-8"))
    rows = top["gaps"] if isinstance(top, dict) else top
    known = {row["id"] for row in signals}
    for row in rows:
        Gap.model_validate(row)
        cited = set(row.get("evidence", {}).get("signal_ids", []))
        unknown = cited - known
        if unknown:
            raise SystemExit(f"gap {row['id']} cites signals not in the fixture: {sorted(unknown)}")

    verdicts = [row.get("verdict") for row in rows]
    print(f"regenerated {len(rows)} ranked gaps; all validate as Gap")
    print(f"verdicts: {verdicts}")
    print(f"counts: {manifest['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
