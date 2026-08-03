from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.run import (
    ROOT,
    PipelineError,
    _artifact_scope,
    _as_demo_manifest,
    _calibration_labels,
    _observed_llm,
    _rank_uncalibrated,
    doctor,
    main,
)
from src.schema import Gap, GapVerdict


def test_demo_copies_self_describing_artifacts(tmp_path: Path) -> None:
    assert main(["demo", "--out-dir", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    gaps = json.loads((tmp_path / "top_gaps.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "demo_fixture"
    assert gaps["mode"] == "demo_fixture"
    assert manifest["scope"] | {
        "product": "ExamplePress for Android (synthetic)",
        "signal_window": "synthetic fixture",
        "roadmap_snapshot": "synthetic fixture",
    } == manifest["scope"]


def test_demo_manifest_scope_is_merged_without_mutating_input() -> None:
    original = {"mode": "production", "scope": {"analysis_mode": "exploratory_snapshot"}}
    stamped = _as_demo_manifest(original)
    assert original == {
        "mode": "production",
        "scope": {"analysis_mode": "exploratory_snapshot"},
    }
    assert stamped["mode"] == "demo_fixture"
    assert stamped["scope"] == {
        "analysis_mode": "exploratory_snapshot",
        "product": "ExamplePress for Android (synthetic)",
        "signal_window": "synthetic fixture",
        "roadmap_snapshot": "synthetic fixture",
    }
    assert stamped["limitations"] == [
        "Synthetic fixture dates are pinned for reproducibility; they are not historical "
        "user evidence or current product findings."
    ]


def test_demo_preserves_generator_commit_and_is_deterministic(tmp_path, monkeypatch) -> None:
    import src.run as run_module

    expected_commit = "a" * 40
    monkeypatch.setattr(run_module, "_git_commit", lambda: expected_commit)
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert main(["demo", "--out-dir", str(first)]) == 0
    assert main(["demo", "--out-dir", str(second)]) == 0
    first_manifest = json.loads((first / "run_manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["code_version"] == expected_commit
    assert json.loads((first / "top_gaps.json").read_text(encoding="utf-8")) == json.loads(
        (second / "top_gaps.json").read_text(encoding="utf-8")
    )


def test_demo_regenerates_through_the_pipeline_not_a_copy(tmp_path: Path) -> None:
    """`demo` must run the pipeline, or reproducibility passes vacuously.

    When the command only copied the fixture, two "runs" were byte-identical
    because neither ran anything, so stale code and config hashes survived every
    check. The manifest's hashes must describe the checked-out code.
    """

    assert main(["demo", "--out-dir", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))

    repro = manifest["reproducibility"]
    expected_config = hashlib.sha256((ROOT / "config" / "pipeline.json").read_bytes()).hexdigest()
    assert repro["pipeline_config_sha256"] == expected_config

    digest = hashlib.sha256()
    for name in sorted(repro["inference_contract_files"]):
        # _combined_sha256 hashes the ROOT-relative path, not the absolute one.
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / name).read_bytes())
        digest.update(b"\0")
    assert repro["inference_contract_sha256"] == digest.hexdigest()

    assert manifest["inputs"]["signals_sha256"] == hashlib.sha256(
        (tmp_path / "signals.json").read_bytes()
    ).hexdigest()
    assert manifest["inputs"]["roadmap_sha256"] == hashlib.sha256(
        (tmp_path / "roadmap.json").read_bytes()
    ).hexdigest()

    # Derived artifacts only exist if a pipeline actually ran.
    assert (tmp_path / "needs.json").exists()
    assert (tmp_path / "verification.json").exists()


def test_manifest_records_observed_llm_outcome_not_just_intent() -> None:
    """A configured run whose every call fails must not read as model-backed."""

    class _Stats:
        def __init__(self, **counts: int) -> None:
            self._counts = counts

        def as_dict(self) -> dict[str, int]:
            return dict(self._counts)

    class _Client:
        def __init__(self, **counts: int) -> None:
            self.stats = _Stats(**counts)

    assert _observed_llm(None, use_llm=False)["status"] == "not_requested"
    # The silent-fallback case: Gemini requested, nothing succeeded, needs came
    # from the offline frames instead.
    assert _observed_llm(_Client(calls=0, failures=0), use_llm=True)["status"] == (
        "requested_but_no_calls"
    )
    assert _observed_llm(_Client(calls=9, failures=2), use_llm=True)["status"] == "partial"
    observed = _observed_llm(_Client(calls=9, failures=0, cache_hits=3), use_llm=True)
    assert observed["status"] == "ok"
    assert observed["cache_hits"] == 3


def test_shipped_demo_fixture_validates_against_the_gap_model() -> None:
    """The fixture must be something the pipeline could actually emit.

    A hand-edited fixture once shipped claiming a MISUNDERSTOOD verdict with an
    obsolete field set, and the whole suite stayed green because nothing
    validated it. Regenerate with ``python -m examples.regenerate_demo``; never
    hand-edit it back into passing.
    """

    fixture = ROOT / "examples" / "demo" / "top_gaps.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    rows = payload["gaps"] if isinstance(payload, dict) else payload
    assert rows, "demo fixture has no ranked gaps"
    for row in rows:
        Gap.model_validate(row)
        assert row["verdict"] in {v.value for v in GapVerdict}


def test_demo_fixture_evidence_ids_resolve_to_shipped_signals() -> None:
    """Every cited ID must exist in the fixture's own signals.json."""

    demo = ROOT / "examples" / "demo"
    signals = json.loads((demo / "signals.json").read_text(encoding="utf-8"))
    known = {row["id"] for row in signals["signals"]}
    payload = json.loads((demo / "top_gaps.json").read_text(encoding="utf-8"))
    rows = payload["gaps"] if isinstance(payload, dict) else payload
    for row in rows:
        cited = set(row.get("evidence", {}).get("signal_ids", []))
        assert cited, f"gap {row['id']} cites no signals"
        assert cited <= known, f"gap {row['id']} cites unknown signals: {sorted(cited - known)}"


def test_uncalibrated_ranking_is_explicit() -> None:
    gap = {
        "id": "G0001",
        "verdict": "IGNORED",
        "critique": "DEFENSIBLE",
        "opportunity_score": 10.0,
        "metadata": {"evidence_score": 0.7, "raw_confidence": 0.7},
    }
    ranked = _rank_uncalibrated([gap], top_k=5)
    assert ranked[0]["rank_score"] == 7.0
    assert "not a probability" in ranked[0]["why_rank"]
    assert (
        ranked[0]["metadata"]["rank_basis"]
        == "uncalibrated_evidence_score_not_probability"
    )


def test_uncalibrated_near_ties_share_a_priority_band() -> None:
    gaps = [
        {
            "id": "G0001",
            "latent_need": "First need",
            "verdict": "IGNORED",
            "critique": "DEFENSIBLE",
            "opportunity_score": 10.0,
            "metadata": {"evidence_score": 0.8},
        },
        {
            "id": "G0002",
            "latent_need": "Second need",
            "verdict": "IGNORED",
            "critique": "DEFENSIBLE",
            "opportunity_score": 10.0,
            "metadata": {"evidence_score": 0.796},
        },
    ]

    ranked = _rank_uncalibrated(gaps, top_k=5)

    assert [row["metadata"]["priority_band"] for row in ranked] == [1, 1]
    assert all(row["metadata"]["deterministic_order_only"] for row in ranked)
    assert all(
        row["metadata"]["rank_separation"]
        == "not_established_within_1_percent_band"
        for row in ranked
    )
    assert all("not evidence of meaningful separation" in row["why_rank"] for row in ranked)


def test_uncalibrated_clear_score_gap_starts_a_new_priority_band() -> None:
    gaps = [
        {
            "id": "G0001",
            "latent_need": "First need",
            "verdict": "IGNORED",
            "critique": "DEFENSIBLE",
            "opportunity_score": 10.0,
            "metadata": {"evidence_score": 0.8},
        },
        {
            "id": "G0002",
            "latent_need": "Second need",
            "verdict": "IGNORED",
            "critique": "DEFENSIBLE",
            "opportunity_score": 10.0,
            "metadata": {"evidence_score": 0.76},
        },
    ]

    ranked = _rank_uncalibrated(gaps, top_k=5)

    assert [row["metadata"]["priority_band"] for row in ranked] == [1, 2]
    assert not any(row["metadata"]["deterministic_order_only"] for row in ranked)
    assert all("not validated priority separation" in row["why_rank"] for row in ranked)
    assert all(
        row["metadata"]["rank_separation"]
        == "outside_adjacent_1_percent_display_bands_not_validated"
        for row in ranked
    )


def test_example_labels_cannot_calibrate(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.json"
    label_path.write_text(
        json.dumps({"status": "EXAMPLE_ONLY_NOT_FOR_CALIBRATION", "labels": []}),
        encoding="utf-8",
    )
    with pytest.raises(PipelineError, match="example label"):
        _calibration_labels(label_path, [], minimum=1)


def test_doctor_is_safe_without_live_network() -> None:
    result = doctor(live=False)
    assert result["live_smoke_test"] is None
    assert "project" not in result["llm"]


def test_time_modes_reject_invalid_claim_scope() -> None:
    old_signals = [{"timestamp": "2017-05-02T00:00:00Z"}]
    with pytest.raises(PipelineError, match="older than two years"):
        _artifact_scope(
            mode="current_opportunity",
            signals=old_signals,
            ingest_scope={"github": {"state_scope": "all"}},
            as_of=datetime(2026, 7, 31, tzinfo=UTC),
        )
    with pytest.raises(PipelineError, match="all-state"):
        _artifact_scope(
            mode="historical_archive_check",
            signals=old_signals,
            ingest_scope={"github": {"state_scope": "open"}},
            as_of=datetime(2026, 7, 31, tzinfo=UTC),
        )
