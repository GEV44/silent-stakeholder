from __future__ import annotations

import pytest

from src.rank import rank_gaps, rank_score


def _gap(gap_id, confidence, opportunity, *, verdict="IGNORED", critique="DEFENSIBLE"):
    return {
        "id": gap_id,
        "need_id": "N0001",
        "latent_need": f"Need {gap_id}",
        "jtbd": "Do work",
        "kano_class": "basic",
        "verdict": verdict,
        "matched_roadmap_id": "R0001",
        "similarity": 0.1,
        "symptom_similarity": 0.1,
        "latent_similarity": 0.1,
        "calibrated_confidence": confidence,
        "opportunity_score": opportunity,
        "rank_score": None,
        "evidence": {"signal_ids": ["S0001", "S0002"], "quotes": []},
        "critique": critique,
        "why_rank": None,
        "features": {},
        "metadata": {},
    }


def test_rank_formula_and_stable_tie_break():
    gaps = [
        _gap("G0002", 0.5, 10),
        _gap("G0001", 0.5, 10),
        _gap("G0003", 0.7, 10),
    ]
    ranked = rank_gaps(gaps, top_k=3)
    assert [gap["id"] for gap in ranked] == ["G0003", "G0001", "G0002"]
    assert [gap["rank"] for gap in ranked] == [1, 2, 3]
    assert ranked[0]["rank_score"] == 7
    assert "opportunity 10.00" in ranked[0]["why_rank"]
    assert gaps[0]["rank_score"] is None  # inputs are not mutated


def test_covered_and_unsupported_are_excluded_by_default():
    gaps = [
        _gap("G0001", 0.9, 20, verdict="COVERED"),
        _gap("G0002", 0.9, 20, critique="UNSUPPORTED"),
        _gap("G0003", 0.4, 10),
    ]
    assert [gap["id"] for gap in rank_gaps(gaps)] == ["G0003"]


def test_rank_score_rejects_undefendable_ranges():
    assert rank_score(0.7, 15) == pytest.approx(10.5)
    with pytest.raises(ValueError):
        rank_score(1.1, 10)
    with pytest.raises(ValueError):
        rank_score(0.5, 21)


def test_rank_keeps_only_strongest_duplicate_need_title():
    weaker = _gap("G0001", 0.4, 10)
    stronger = _gap("G0002", 0.8, 10)
    weaker["latent_need"] = "Reliable uploads"
    stronger["latent_need"] = "  reliable   uploads "
    ranked = rank_gaps([weaker, stronger], top_k=5)
    assert [gap["id"] for gap in ranked] == ["G0002"]
