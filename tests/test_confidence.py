from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from src.confidence import (
    FEATURE_NAMES,
    ConfidenceConfig,
    ProbabilityCalibrator,
    bootstrap_stability,
    brier_score,
    calibration_metrics,
    confidence_features,
    cross_fitted_calibration,
    expected_calibration_error,
    plot_reliability,
    raw_confidence,
    score_gaps,
)


def _gap():
    return {
        "id": "G0001",
        "need_id": "N0001",
        "evidence": {"signal_ids": ["S0001", "S0002"], "quotes": []},
        "metadata": {},
    }


def _signals():
    return [
        {
            "id": "S0001",
            "source": "review",
            "rating": 1,
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "id": "S0002",
            "source": "ticket",
            "rating": 2,
            "timestamp": "2025-01-01T00:00:00Z",
        },
    ]


def _need():
    return {
        "id": "N0001",
        "metadata": {
            "mean_sentiment_intensity": 0.8,
            "self_consistency": 0.75,
            "llm_samples": 3,
        },
    }


def test_all_confidence_features_are_mechanistic_unit_values():
    features = confidence_features(
        _gap(),
        signals=_signals(),
        need=_need(),
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert set(features) == set(FEATURE_NAMES)
    assert all(0 <= value <= 1 for value in features.values())
    assert features["div"] > 0
    assert features["sent"] == 0.8
    assert features["selfc"] == 0.75
    assert features["cons"] == 0.75


def test_harmonic_fusion_penalizes_low_self_consistency():
    base = {name: 0.9 for name in FEATURE_NAMES}
    high = raw_confidence(base, config=ConfidenceConfig(fusion="harmonic"))
    base["selfc"] = 0.1
    low = raw_confidence(base, config=ConfidenceConfig(fusion="harmonic"))
    assert low < high


def test_platt_and_isotonic_calibrators_return_probabilities():
    scores = np.linspace(0.05, 0.95, 40)
    labels = (scores > 0.55).astype(int)
    isotonic = ProbabilityCalibrator(method="auto", isotonic_min_samples=30).fit(
        scores, labels
    )
    assert isotonic.method_ == "isotonic"
    predicted = isotonic.predict([0.1, 0.9])
    assert predicted[0] <= predicted[1]
    assert np.all((predicted >= 0) & (predicted <= 1))

    platt = ProbabilityCalibrator(method="auto", isotonic_min_samples=100).fit(
        scores[10:30], labels[10:30]
    )
    assert platt.method_ == "platt"


def test_calibration_metrics_match_known_values_and_plot(tmp_path):
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]
    assert brier_score(labels, probabilities) == np.mean([0.01, 0.04, 0.04, 0.01])
    assert expected_calibration_error(labels, probabilities, n_bins=2) == pytest.approx(0.15)
    metrics = calibration_metrics(labels, probabilities, n_bins=2)
    assert metrics["n"] == 4
    target = plot_reliability(labels, probabilities, tmp_path / "calibration.png", n_bins=2)
    assert target.exists()
    assert target.stat().st_size > 0


def test_reported_calibration_metrics_are_cross_fitted():
    scores = np.linspace(0.05, 0.95, 20)
    labels = (scores > 0.5).astype(int)
    probabilities, metrics, final_calibrator = cross_fitted_calibration(
        scores,
        labels,
        method="platt",
        n_splits=4,
        n_bins=4,
    )
    assert len(probabilities) == len(labels)
    assert metrics["evaluation"] == "cross-fitted"
    assert metrics["folds"] == 4
    assert final_calibrator.method_ == "platt"


def test_score_gaps_attaches_features_provenance_and_calibrated_value():
    gap = {
        **_gap(),
        "latent_need": "Reliable export",
        "jtbd": "Export safely",
        "kano_class": "basic",
        "verdict": "IGNORED",
        "matched_roadmap_id": "R0001",
        "similarity": 0.1,
        "symptom_similarity": 0.1,
        "latent_similarity": 0.1,
        "calibrated_confidence": None,
        "opportunity_score": 12,
        "rank_score": None,
        "critique": None,
        "why_rank": None,
    }
    scored = score_gaps([gap], signals=_signals(), needs=[_need()])
    assert len(scored) == 1
    assert scored[0]["calibrated_confidence"] is None
    assert set(scored[0]["features"]) == set(FEATURE_NAMES)
    assert "raw_confidence" in scored[0]["metadata"]
    assert (
        scored[0]["metadata"]["confidence_status"]
        == "uncalibrated_evidence_score_not_probability"
    )
    assert scored[0]["metadata"]["evidence_score"] == scored[0]["metadata"]["raw_confidence"]


def test_feature_diagnostics_name_the_features_that_did_no_work():
    """The evidence score must disclose its own dynamic range.

    On a single-source offline set several features are constant, so they carry
    no discriminating information regardless of weight. That has to be visible
    in the artifact rather than something a reader discovers by inspection.
    """

    signals = [
        {"id": "S0001", "source": "review", "rating": 1, "timestamp": "2026-01-01T00:00:00Z"},
        {"id": "S0002", "source": "review", "rating": 2, "timestamp": "2026-01-02T00:00:00Z"},
        {"id": "S0003", "source": "review", "rating": 1, "timestamp": "2026-01-03T00:00:00Z"},
    ]
    gaps = [
        {
            "id": "G0001",
            "need_id": "N0001",
            "evidence": {"signal_ids": ["S0001"], "quotes": []},
            "metadata": {},
        },
        {
            "id": "G0002",
            "need_id": "N0002",
            "evidence": {"signal_ids": ["S0001", "S0002", "S0003"], "quotes": []},
            "metadata": {},
        },
    ]
    scored = score_gaps(gaps, signals=signals, needs=[])
    diagnostics = scored[0]["metadata"]["feature_diagnostics"]

    # Every feature is classified exactly once, either constant or varying.
    classified = set(diagnostics["constant_features"]) | set(
        diagnostics["discriminating_features"]
    )
    assert classified == set(scored[0]["features"])
    assert not (
        set(diagnostics["constant_features"]) & set(diagnostics["discriminating_features"])
    )
    # Single-source corpus: source diversity cannot discriminate.
    assert "div" in diagnostics["constant_features"]
    # The share of live weight spent on constants is reported, not hidden.
    assert 0.0 <= diagnostics["weight_on_constant_features"] <= 1.0
    assert diagnostics["evidence_score_range"] >= 0.0
    # All gaps carry the same run-level diagnostic.
    assert scored[1]["metadata"]["feature_diagnostics"] == diagnostics


def test_feature_diagnostics_skipped_for_a_single_gap():
    """Variance is undefined for one row; do not claim a diagnostic we cannot make."""

    scored = score_gaps([_gap()], signals=_signals(), needs=[])
    assert "feature_diagnostics" not in scored[0]["metadata"]


def test_cohesion_separates_tight_evidence_from_scattered_evidence():
    """`coh` must move with how related a gap's cited signals actually are.

    It exists because div/selfc/agree are constant on a single-source offline
    run, leaving the evidence score with almost no dynamic range.
    """

    tight = [
        {"id": "S0001", "source": "review", "text": "media upload fails halfway every time"},
        {"id": "S0002", "source": "review", "text": "media upload fails halfway again"},
        {"id": "S0003", "source": "review", "text": "media upload keeps failing halfway"},
    ]
    scattered = [
        {"id": "S0001", "source": "review", "text": "media upload fails halfway every time"},
        {"id": "S0002", "source": "review", "text": "the font picker has too few typefaces"},
        {"id": "S0003", "source": "review", "text": "billing renewed without any warning"},
    ]

    def coh(signals):
        gap = {
            "id": "G0001",
            "need_id": "N0001",
            "evidence": {"signal_ids": [s["id"] for s in signals], "quotes": []},
            "metadata": {},
        }
        return confidence_features(gap, signals=signals)["coh"]

    assert coh(tight) > coh(scattered)
    assert 0.0 <= coh(scattered) <= coh(tight) <= 1.0


def test_cohesion_is_neutral_for_a_single_signal():
    """Pairwise similarity is undefined for one item; do not score it as incoherent."""

    signals = [{"id": "S0001", "source": "review", "text": "upload fails"}]
    gap = {
        "id": "G0001",
        "need_id": "N0001",
        "evidence": {"signal_ids": ["S0001"], "quotes": []},
        "metadata": {},
    }
    assert confidence_features(gap, signals=signals)["coh"] == 0.5


def test_score_breakdown_reconstructs_the_evidence_score_exactly():
    """The rendered arithmetic must equal the pipeline's number, not approximate it.

    If the UI could drift from the score, showing the arithmetic would be worse
    than showing nothing.
    """

    from src.confidence import score_breakdown

    features = {name: 0.6 for name in FEATURE_NAMES}
    breakdown = score_breakdown(features, generative_observed=False)

    # Each term is weight x value, and the terms plus intercept give the logit.
    for term in breakdown["terms"]:
        assert term["contribution"] == pytest.approx(term["weight"] * term["value"])
    assert breakdown["logit"] == pytest.approx(
        breakdown["intercept"] + sum(t["contribution"] for t in breakdown["terms"])
    )
    # And the published score matches raw_confidence to the digit.
    assert breakdown["evidence_score"] == pytest.approx(
        raw_confidence(features, generative_observed=False)
    )
    # selfc is fused, never summed as a term - listing it twice would double-count.
    assert "selfc" not in {t["feature"] for t in breakdown["terms"]}
    assert "not a probability" in breakdown["not_a_probability"].lower()


def test_score_gaps_publishes_the_breakdown_for_the_ui():
    scored = score_gaps([_gap()], signals=_signals(), needs=[_need()])
    breakdown = scored[0]["metadata"]["score_breakdown"]
    assert breakdown["evidence_score"] == scored[0]["metadata"]["evidence_score"]
    assert {t["feature"] for t in breakdown["terms"]} == set(FEATURE_NAMES) - {"selfc"}


def test_counterevidence_names_the_signals_that_argue_against_the_gap():
    """Counting dissent is not the same as showing it.

    A reader cannot verify we did not cherry-pick unless the contradicting
    signals are named, and the list must agree with what `cons` scored.
    """

    from src.confidence import counterevidence

    signals = [
        {"id": "S0001", "source": "review", "rating": 1},  # supports (negative)
        {"id": "S0002", "source": "review", "rating": 1},  # supports
        {"id": "S0003", "source": "review", "rating": 5},  # contradicts (positive)
    ]
    gap = {
        "id": "G0001",
        "need_id": "N0001",
        "evidence": {"signal_ids": ["S0001", "S0002", "S0003"], "quotes": []},
        "metadata": {},
    }
    report = counterevidence(gap, signals)

    assert report["signal_ids"] == ["S0003"]
    assert report["count"] == 1
    assert report["share_of_evidence"] == pytest.approx(1 / 3)
    # Dissent lowers consistency, so the surfaced list and the score agree.
    with_dissent = confidence_features(gap, signals=signals)["cons"]
    no_dissent = confidence_features(
        {**gap, "evidence": {"signal_ids": ["S0001", "S0002"], "quotes": []}},
        signals=signals,
    )["cons"]
    assert with_dissent < no_dissent


def test_counterevidence_is_empty_and_honest_when_evidence_is_unanimous():
    from src.confidence import counterevidence

    signals = [{"id": "S0001", "source": "review", "rating": 1}]
    gap = {
        "id": "G0001",
        "need_id": "N0001",
        "evidence": {"signal_ids": ["S0001"], "quotes": []},
        "metadata": {},
    }
    report = counterevidence(gap, signals)
    assert report["signal_ids"] == []
    assert report["count"] == 0
    assert report["share_of_evidence"] == 0.0


def test_default_weights_cover_every_declared_feature():
    """A feature in FEATURE_NAMES but missing from the defaults scores at 0 silently."""

    assert set(ConfidenceConfig().weights) == set(FEATURE_NAMES)


def test_weakest_assumption_picks_the_most_severe_applicable_risk():
    """A near-gate verdict outranks scattered evidence: it is the one that flips."""

    from src.confidence import weakest_assumption

    gap = {
        "id": "G0001",
        "similarity": 0.39,
        "metadata": {"thresholds": {"low": 0.38}},
    }
    features = {name: 0.9 for name in FEATURE_NAMES}
    features["coh"] = 0.02  # also scattered, but less severe than a flippable verdict
    report = weakest_assumption(gap, features, dissent_share=0.30)

    assert "similarity gate" in report["factor"]
    assert report["value"] == pytest.approx(0.01, abs=1e-6)
    assert report["what_would_change_it"]
    # Lesser risks are still listed, not discarded.
    assert len(report["all_applicable"]) >= 3


def test_weakest_assumption_never_contradicts_verdict_stability():
    """The two judge-facing fields must name the same governing gate.

    On the shipped flagship MISUNDERSTOOD gap these disagreed: verdict_stability
    said `framing_coverage` ... "robust to any `low` value", while
    weakest_assumption reported, as the single most fragile thing the gap rests
    on, "a gate shift of 0.005 would change this verdict". Both cannot be true,
    and the false one carried the louder label. weakest_assumption now reads the
    governing gate rather than inferring its own.
    """

    from src.confidence import weakest_assumption

    # Similarity sits 0.005 from `low` -- the old code's severity-4 trigger --
    # but the verdict is decided by framing coverage, so `low` is inert.
    gap = {
        "id": "G0001",
        "similarity": 0.375,
        "metadata": {
            "thresholds": {"low": 0.38},
            "framing": {"misunderstood": True, "coverage_threshold": 0.5},
            "verdict_stability": {
                "governing_gate": "framing_coverage",
                "margin_to_flip": 0.5,
                "flips_to": None,
            },
        },
    }
    features = {name: 0.9 for name in FEATURE_NAMES}
    report = weakest_assumption(gap, features)

    factors = [report["factor"], *report["all_applicable"]]
    assert not any("similarity gate" in f for f in factors), (
        f"similarity gate claimed on a framing-decided verdict: {factors}"
    )

    # And when framing IS close to its own cutoff, that is what gets named.
    gap["metadata"]["verdict_stability"]["margin_to_flip"] = 0.017
    tight = weakest_assumption(gap, features)
    assert "framing-coverage cutoff" in tight["factor"]
    assert tight["value"] == pytest.approx(0.017, abs=1e-6)


def test_weakest_assumption_falls_back_to_the_calibration_caveat():
    """A gap with no specific weakness still must not present as unqualified."""

    from src.confidence import weakest_assumption

    gap = {"id": "G0001", "similarity": 0.9, "metadata": {"thresholds": {"low": 0.38}}}
    features = {name: 0.9 for name in FEATURE_NAMES}
    report = weakest_assumption(gap, features, dissent_share=0.0)

    assert "uncalibrated" in report["factor"]
    assert report["all_applicable"] == ["confidence is uncalibrated"]


def test_score_gaps_publishes_the_weakest_assumption():
    scored = score_gaps([_gap()], signals=_signals(), needs=[_need()])
    report = scored[0]["metadata"]["weakest_assumption"]
    assert report["factor"] and report["why"] and report["what_would_change_it"]


def test_null_embedding_model_in_config_resolves_to_the_documented_default():
    """Regression for REQ-main-3.

    `dict.get(key, default)` returns the STORED value when the key exists, so a
    `null` in pipeline.json defeated the default and produced model_name=None.
    SentenceTransformer(None) then builds an object without raising and the run
    died stages later inside encode(). Null and blank must mean "absent".
    """

    import json as _json
    from unittest.mock import patch

    from src.config import PipelineConfig, load_json_config

    base = _json.loads(_json.dumps(load_json_config("pipeline.json")))
    for stored in (None, "", "   "):
        variant = {**base, "embedding": {**base.get("embedding", {}), "model": stored}}
        with patch("src.config.load_json_config", return_value=variant):
            resolved = PipelineConfig.load().embedding_model
        assert resolved, f"stored {stored!r} resolved to a falsy model name"
        assert resolved.strip() == resolved


# ---------------------------------------------------------------------------
# Resampling stability
#
# The measure that lets us say one finding is better evidenced than another.
# Everything here runs against a fake `infer` so it stays offline and fast; the
# real one is exercised end to end by the pipeline.
# ---------------------------------------------------------------------------


def _stability_signals(count: int) -> list[dict[str, object]]:
    return [{"id": f"S{i:06d}", "text": f"signal {i}"} for i in range(count)]


def _stability_need(title: str, ids: list[str]) -> dict[str, object]:
    return {"latent_need": title, "supporting_signal_ids": ids}


def test_stability_separates_a_stable_need_from_a_fragile_one() -> None:
    """The whole point: survival must tell these two apart.

    "always" is present in every draw; "fragile" only when one specific signal
    survives the subsample. A measure that scored them alike would be no better
    than the evidence score it is meant to supplement.
    """

    signals = _stability_signals(50)

    def infer(rows):
        ids = [str(row["id"]) for row in rows]
        needs = [_stability_need("always", ids)]
        if "S000000" in ids:
            needs.append(_stability_need("fragile", ["S000000"]))
        return needs

    report = bootstrap_stability(signals, infer, iterations=20, fraction=0.5, seed=7)

    assert report["shipped"]["always"]["survival"] == 1.0
    fragile = report["shipped"]["fragile"]["survival"]
    assert 0.0 < fragile < 1.0, "a sample-dependent need must not score 0 or 1"


def test_stability_is_deterministic_for_a_seed_and_moves_with_it() -> None:
    """A number quoted on a slide has to be reproducible, and it is seeded."""

    signals = _stability_signals(40)

    def infer(rows):
        ids = [str(row["id"]) for row in rows]
        return [_stability_need("here", ids)] if "S000003" in ids else []

    first = bootstrap_stability(signals, infer, iterations=12, fraction=0.5, seed=1)
    again = bootstrap_stability(signals, infer, iterations=12, fraction=0.5, seed=1)
    assert first == again

    other = bootstrap_stability(signals, infer, iterations=12, fraction=0.5, seed=99)
    assert other["seed"] == 99


def test_survival_and_jaccard_answer_different_questions() -> None:
    """A need can come back every time on entirely different evidence.

    Reporting survival alone would call that perfectly stable. It is not, and
    the deck is required to quote both numbers.
    """

    signals = _stability_signals(30)

    def infer(rows):
        # Always returns the need, but supported by whichever signals it drew,
        # so membership churns while the title never disappears.
        return [_stability_need("churns", [str(row["id"]) for row in rows[:5]])]

    report = bootstrap_stability(signals, infer, iterations=15, fraction=0.5, seed=3)
    entry = report["shipped"]["churns"]

    assert entry["survival"] == 1.0
    assert entry["mean_jaccard"] < 1.0, "identical membership would defeat the point"


def test_needs_absent_from_the_baseline_are_reported_not_dropped() -> None:
    """The needs a different draw would have shipped are part of the finding.

    Silently discarding them would report instability only in the direction that
    flatters us.
    """

    signals = _stability_signals(30)

    def infer(rows):
        ids = [str(row["id"]) for row in rows]
        needs = [_stability_need("baseline", ids)]
        if "S000029" not in ids:
            needs.append(_stability_need("only-in-subsamples", ids[:3]))
        return needs

    report = bootstrap_stability(signals, infer, iterations=10, fraction=0.5, seed=5)

    assert "baseline" in report["shipped"]
    assert "only-in-subsamples" in report["unshipped"]
    assert report["unshipped"]["only-in-subsamples"]["survival"] > 0


def test_subsamples_never_duplicate_a_signal() -> None:
    """Sampling is without replacement because infer_needs demands unique IDs.

    A textbook bootstrap would duplicate review text and inflate cluster
    density, which is the very quantity being measured.
    """

    signals = _stability_signals(40)
    seen: list[int] = []

    def infer(rows):
        ids = [str(row["id"]) for row in rows]
        assert len(ids) == len(set(ids)), "a subsample repeated a signal"
        seen.append(len(ids))
        return [_stability_need("n", ids)]

    report = bootstrap_stability(signals, infer, iterations=6, fraction=0.75, seed=11)
    assert report["subsample_size"] == 30
    assert seen[1:] == [30] * 6


def test_stability_rejects_arguments_that_would_produce_a_meaningless_number() -> None:
    signals = _stability_signals(5)
    def infer(rows):
        return [_stability_need("n", [str(r["id"]) for r in rows])]

    with pytest.raises(ValueError):
        bootstrap_stability(signals, infer, iterations=0)
    with pytest.raises(ValueError):
        bootstrap_stability(signals, infer, fraction=0.0)
    with pytest.raises(ValueError):
        bootstrap_stability(signals, infer, fraction=1.5)
    with pytest.raises(ValueError):
        bootstrap_stability([], infer)


def test_stability_never_presents_itself_as_a_probability() -> None:
    """Invariant 5. A label-free frequency is not a calibrated confidence."""

    signals = _stability_signals(20)
    report = bootstrap_stability(
        signals,
        lambda rows: [_stability_need("n", [str(r["id"]) for r in rows])],
        iterations=4,
        seed=2,
    )
    assert "not a calibrated probability" in report["note"]
