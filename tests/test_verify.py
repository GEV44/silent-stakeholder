from __future__ import annotations

from src.verify import verify_gap, verify_gaps, verify_quote

SIGNALS = [
    {"id": "S0001", "text": "Exporting a draft always crashes on my phone."},
    {"id": "S0002", "text": "I lose the post whenever export fails."},
]
ROADMAP = [
    {
        "id": "R0001",
        "title": "Export crash fix",
        "body": "Prevent a crash after tapping export.",
        "labels": ["bug"],
        "milestone": "Backlog",
    }
]
NEEDS = [{"id": "N0001"}]


def _gap():
    return {
        "id": "G0001",
        "need_id": "N0001",
        "latent_need": "Reliable export",
        "jtbd": "Save and move drafts safely",
        "kano_class": "basic",
        "verdict": "UNDER-PRIORITIZED",
        "matched_roadmap_id": "R0001",
        "similarity": 0.8,
        "symptom_similarity": 0.8,
        "latent_similarity": 0.8,
        "calibrated_confidence": 0.7,
        "opportunity_score": 15,
        "rank_score": None,
        "evidence": {
            "signal_ids": ["S0001", "S0002"],
            "quotes": [
                {"id": "S0001", "span": "Exporting a draft always crashes", "start": 0, "end": 32},
                {"id": "S0002", "span": "I lose the post whenever export fails."},
            ],
        },
        "critique": None,
        "why_rank": None,
        "features": {},
        "metadata": {},
    }


def test_quote_verification_exact_and_fuzzy():
    exact = verify_quote(SIGNALS[0]["text"], "always crashes")
    assert exact.valid and exact.exact and exact.score == 1.0
    fuzzy = verify_quote(
        SIGNALS[0]["text"],
        "Exporting the draft always crashes",
        fuzzy_threshold=0.80,
    )
    assert fuzzy.valid
    assert not fuzzy.exact
    assert fuzzy.matched_span in SIGNALS[0]["text"]
    rejected = verify_quote(SIGNALS[0]["text"], "calendar reminders are missing")
    assert not rejected.valid


def test_verify_gap_rejects_unknown_ids_and_bad_quotes():
    gap = _gap()
    gap["evidence"]["signal_ids"].append("S9999")
    gap["evidence"]["quotes"][0]["span"] = "This was never said"
    report = verify_gap(gap, signals=SIGNALS, roadmap=ROADMAP, needs=NEEDS)
    assert not report["valid"]
    assert report["critique"] == "UNSUPPORTED"
    assert any("unknown signal" in issue for issue in report["issues"])
    assert any("fuzzy threshold" in issue for issue in report["issues"])


def test_verify_gaps_canonicalizes_fuzzy_quote_and_can_only_be_downgraded():
    gap = _gap()
    gap["evidence"]["quotes"][0] = {
        "id": "S0001",
        "span": "Exporting the draft always crashes",
    }

    class Skeptic:
        def critique(self, **_kwargs):
            return {
                "verdict": "WEAK",
                "rationale": "The roadmap evidence addresses only one failure mode.",
                "signal_ids": ["S0001"],
                "roadmap_ids": ["R0001"],
            }

    verified, reports = verify_gaps(
        [gap],
        signals=SIGNALS,
        roadmap=ROADMAP,
        needs=NEEDS,
        fuzzy_threshold=0.80,
        critic=Skeptic(),
    )
    assert reports[0]["valid"]
    assert verified[0]["critique"] == "WEAK"
    canonical = verified[0]["evidence"]["quotes"][0]
    assert canonical["span"] in SIGNALS[0]["text"]
    assert canonical["start"] is not None
