from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from src.gaps import (
    GapThresholds,
    _validate_adjudication,
    detect_gaps,
    deterministic_verdict,
    evaluate_framing,
    framing_probes,
    priority_is_low,
    roadmap_text,
    tune_thresholds,
)


class KeywordEmbedder:
    vocabulary = ("export", "crash", "notification", "search")

    def encode(self, texts):
        rows = []
        for text in texts:
            lowered = text.lower()
            row = np.asarray([lowered.count(token) for token in self.vocabulary], dtype=float)
            norm = np.linalg.norm(row)
            rows.append(row / norm if norm else row)
        return np.asarray(rows, dtype=np.float32)


def _roadmap():
    return [
        {
            "id": "R0001",
            "type": "issue",
            "title": "Export crash fix",
            "body": "Stop a crash in export.",
            "state": "open",
            "labels": ["bug"],
            "milestone": "Backlog",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "priority": {
                "tier": "backlog",
                "score": 0.2,
                "is_low_priority": True,
                "has_explicit_priority": False,
            },
        },
        {
            "id": "R0002",
            "type": "issue",
            "title": "Notification settings",
            "body": "Committed notification controls.",
            "state": "open",
            "labels": ["high priority"],
            "milestone": "7.0",
            "created_at": "2025-12-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
            "priority": {
                "tier": "high",
                "score": 0.9,
                "is_low_priority": False,
                "has_explicit_priority": True,
            },
        },
    ]


# ---------------------------------------------------------------------------
# Framing fixtures: a need whose symptom and job use disjoint vocabulary, one
# roadmap item speaking only the symptom's language, one only the job's.
# ---------------------------------------------------------------------------

_COMMITTED = {
    "tier": "high",
    "score": 0.9,
    "is_low_priority": False,
    "has_explicit_priority": True,
}


def _framed_need(**overrides):
    need = {
        "id": "N0001",
        "latent_need": "Trust that a post makes it out intact",
        "jtbd_statement": (
            "publish a finished draft dependably even when the connection drops midway"
        ),
        "symptom": "export screen crashes and the upload window freezes",
        "supporting_signal_ids": ["S0001"],
        "opportunity_score": 14,
    }
    need.update(overrides)
    return need


def _symptom_item():
    return {
        "id": "R1001",
        "type": "issue",
        "title": "Fix export screen crash",
        "body": (
            "The export screen crashes and the upload window freezes on large posts. "
            "Crash logs attached."
        ),
        "state": "open",
        "labels": ["bug"],
        "milestone": "7.0",
        "created_at": "2025-12-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "priority": dict(_COMMITTED),
    }


def _job_item():
    return {
        "id": "R1002",
        "type": "issue",
        "title": "Preserve drafts when the connection drops",
        "body": "Publish a finished draft even if the connection drops midway.",
        "state": "open",
        "labels": ["enhancement"],
        "milestone": "7.0",
        "created_at": "2025-12-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "priority": dict(_COMMITTED),
    }


def _filler_item():
    return {
        "id": "R1003",
        "type": "issue",
        "title": "Notification settings",
        "body": "Committed notification controls.",
        "state": "open",
        "labels": [],
        "milestone": "7.0",
        "created_at": "2025-12-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "priority": dict(_COMMITTED),
    }


def _detect(needs, roadmap):
    return detect_gaps(
        needs,
        roadmap,
        signals=[{"id": "S0001", "text": "The export screen crashed again."}],
        thresholds=GapThresholds(low=0.05, high=0.95),
        include_covered=True,
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )


def test_misunderstood_fires_on_symptom_only_coverage_and_cites_the_proof():
    gaps = _detect([_framed_need()], [_symptom_item(), _filler_item()])
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap["verdict"] == "MISUNDERSTOOD"
    # The cited item is the symptom-covering one: the proof, not the retrieval argmax.
    assert gap["matched_roadmap_id"] == "R1001"
    framing = gap["metadata"]["framing"]
    assert framing["eligible"] and framing["misunderstood"]
    assert framing["symptom_coverage"] >= 0.5 > framing["job_coverage"]
    # The roadmap quote must be an exact substring so src/verify.py accepts it.
    quote = gap["evidence"].get("roadmap_quote")
    assert quote and quote["span"] in roadmap_text(_symptom_item())


def test_misunderstood_flips_off_when_the_job_is_also_addressed():
    """V1 flip: reachability alone proves nothing; discrimination is the claim."""

    gaps = _detect([_framed_need()], [_symptom_item(), _job_item()])
    assert len(gaps) == 1
    assert gaps[0]["verdict"] != "MISUNDERSTOOD"
    framing = gaps[0]["metadata"]["framing"]
    assert framing["job_coverage"] >= 0.5


def test_misunderstood_does_not_fire_when_framings_are_swapped():
    """V2 swap: the gate measures asymmetry, not magnitude."""

    swapped = _framed_need(
        symptom=(
            "publish a finished draft dependably even when the connection drops midway"
        ),
        jtbd_statement="export screen crashes and the upload window freezes",
    )
    gaps = _detect([swapped], [_symptom_item(), _filler_item()])
    assert len(gaps) == 1
    assert gaps[0]["verdict"] != "MISUNDERSTOOD"


def test_one_sided_majority_below_the_margin_floor_does_not_fire():
    """REQ-E-03: two independent reviewers rejected verdicts decided this way.

    A one-sided majority (symptom_coverage >= tau > job_coverage) is not
    sufficient on its own -- the two numbers must also be separated by at
    least ``min_coverage_margin``, or the "majority" is indistinguishable
    from complaint-generic vocabulary carrying the symptom probe past the
    cutoff by a few points.
    """

    need = _framed_need()
    # Partially covers both probes: symptom clears 50%, job does not, and the
    # two are close enough together that a low floor rejects it while a
    # default-strength floor accepts it -- isolating the margin mechanism
    # itself rather than depending on hand-tuned exact percentages.
    item = {
        "id": "R2001",
        "type": "issue",
        "title": "Crash export freezes on screen window",
        "body": (
            "The crash export freezes on screen window. Connection drops "
            "during draft dependably."
        ),
        "state": "open",
        "labels": [],
        "milestone": "7.0",
        "created_at": "2025-12-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "priority": dict(_COMMITTED),
    }

    lenient = detect_gaps(
        [need],
        [item, _filler_item()],
        signals=[{"id": "S0001", "text": "The export screen crashed again."}],
        thresholds=GapThresholds(low=0.05, high=0.95, min_coverage_margin=0.10),
        include_covered=True,
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )
    lenient_framing = lenient[0]["metadata"]["framing"]
    assert lenient_framing["symptom_coverage"] >= 0.5 > lenient_framing["job_coverage"]
    margin = lenient_framing["coverage_margin"]
    assert lenient_framing["misunderstood"] is True

    strict = detect_gaps(
        [need],
        [item, _filler_item()],
        signals=[{"id": "S0001", "text": "The export screen crashed again."}],
        thresholds=GapThresholds(low=0.05, high=0.95, min_coverage_margin=margin + 0.05),
        include_covered=True,
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )
    strict_framing = strict[0]["metadata"]["framing"]
    # Same coverage numbers, same one-sided majority -- only the floor moved.
    assert strict_framing["symptom_coverage"] == lenient_framing["symptom_coverage"]
    assert strict_framing["job_coverage"] == lenient_framing["job_coverage"]
    assert strict_framing["misunderstood"] is False
    assert "margin" in strict_framing["rationale"] and "REQ-E-03" in strict_framing["rationale"]
    assert strict[0]["verdict"] != "MISUNDERSTOOD"


def test_inseparable_framings_are_ineligible_not_guessed():
    """A need whose symptom and job share vocabulary cannot be MISUNDERSTOOD."""

    degenerate = _framed_need(
        symptom="the export crashes",
        jtbd_statement="export without crashing",
    )
    gaps = _detect([degenerate], [_symptom_item(), _filler_item()])
    assert len(gaps) == 1
    framing = gaps[0]["metadata"]["framing"]
    assert not framing["eligible"]
    assert gaps[0]["verdict"] != "MISUNDERSTOOD"


def test_probes_are_disjoint_equal_size_and_deterministic():
    need = _framed_need()
    df: dict[str, int] = {}
    first = framing_probes(need, df=df, total=0)
    second = framing_probes(need, df=df, total=0)
    assert first == second
    symptom_probe, job_probe = first
    assert len(symptom_probe) == len(job_probe) > 0
    assert not set(symptom_probe) & set(job_probe)


def test_framing_verdict_outranks_the_low_similarity_gate():
    """A symptom-covering item can sit below ``low``; the citation still wins."""

    fired = {"misunderstood": True, "rationale": "covers symptom terms only"}
    verdict, rationale = deterministic_verdict(
        similarity=0.01,
        symptom_similarity=0.0,
        latent_similarity=0.0,
        low_priority=False,
        thresholds=GapThresholds(low=0.3, high=0.7),
        framing=fired,
    )
    assert verdict == "MISUNDERSTOOD"
    assert "symptom" in rationale


def test_adjudicator_cannot_create_a_misunderstood_the_framing_gate_declined():
    """REQ-D-03: the guarantee must be symmetric, not just one-directional.

    ``deterministic_verdict`` already refuses to let an adjudicator override a
    framing-backed MISUNDERSTOOD. The gap this closes is the reverse: nothing
    stopped the adjudicator from *inventing* a MISUNDERSTOOD when framing
    explicitly declined one. Observed on the real WordPress corpus
    (gap G4208b45c1d4e): framing reported "misunderstood": false at 18% symptom
    coverage, yet the shipped verdict was MISUNDERSTOOD via adjudication="llm".
    """

    declined_framing = {
        "misunderstood": False,
        "rationale": "no one-sided majority at the 50% cutoff",
    }
    item = _filler_item()

    with pytest.raises(ValueError, match="framing gate declined"):
        _validate_adjudication(
            {
                "verdict": "MISUNDERSTOOD",
                "roadmap_id": item["id"],
                "roadmap_quote": "Committed notification controls.",
                "rationale": "looks related",
            },
            item,
            item["id"],
            framing=declined_framing,
        )


class _AlwaysMisunderstoodAdjudicator:
    """Stub reproducing the bug: fires MISUNDERSTOOD regardless of framing."""

    def adjudicate(self, *, roadmap_item, allowed_roadmap_id, **_kwargs):
        return {
            "verdict": "MISUNDERSTOOD",
            "roadmap_id": allowed_roadmap_id,
            "roadmap_quote": roadmap_text(roadmap_item)[:20],
            "rationale": "adjudicator wrongly asserts a match",
        }


def test_detect_gaps_rejects_an_adjudicator_invented_misunderstood():
    """End-to-end: an unrelated item in the ambiguous band must not become
    MISUNDERSTOOD just because the adjudicator says so when framing declined it."""

    gaps = detect_gaps(
        [_framed_need()],
        [_filler_item()],
        signals=[{"id": "S0001", "text": "The export screen crashed again."}],
        adjudicator=_AlwaysMisunderstoodAdjudicator(),
        thresholds=GapThresholds(low=0.0, high=1.0),
        include_covered=True,
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert len(gaps) == 1
    framing = gaps[0]["metadata"]["framing"]
    assert not framing["misunderstood"]
    assert gaps[0]["verdict"] != "MISUNDERSTOOD"


def _coverage_fixture(*, state="open", state_reason=None):
    need = {
        "id": "NCLOSED",
        "latent_need": "Export crash fix",
        "jtbd_statement": "Stop a crash in export",
        "symptom": "Stop a crash in export",
        "supporting_signal_ids": ["S0001"],
        "opportunity_score": 10,
    }
    item = {
        "id": "RCLOSED",
        "type": "issue",
        "title": "Export crash fix",
        "body": "Stop a crash in export.",
        "state": state,
        "state_reason": state_reason,
        "labels": ["high priority"],
        "milestone": "7.0",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "priority": dict(_COMMITTED),
    }
    return need, item


@pytest.mark.parametrize("include_covered", [False, True])
def test_closed_completed_history_is_disclosure_only(include_covered):
    need, item = _coverage_fixture(state="closed", state_reason="completed")
    kwargs = {
        "signals": [{"id": "S0001", "text": "Export crashes"}],
        "embedder": KeywordEmbedder(),
        "thresholds": GapThresholds(low=0.25, high=0.5),
        "include_covered": include_covered,
        "as_of": datetime(2026, 1, 2, tzinfo=UTC),
    }
    first = detect_gaps([need], [item], **kwargs)
    second = detect_gaps([need], [item], **kwargs)
    assert len(first) == 1
    assert first[0]["verdict"] == "IGNORED"
    assert first[0]["id"] == second[0]["id"]
    history = first[0]["metadata"]["closed_history"]
    assert history == {
        "state": "closed",
        "state_reason": "completed",
        "treatment": "disclosure_only",
        "verified_merged_change_and_release": False,
        "coverage_guard_applied": True,
    }
    assert first[0]["metadata"]["verdict_stability"]["governing_gate"] == (
        "closed_history_policy"
    )


def test_closed_not_planned_history_cannot_become_coverage():
    need, item = _coverage_fixture(state="closed", state_reason="not_planned")
    gaps = detect_gaps(
        [need],
        [item],
        embedder=KeywordEmbedder(),
        thresholds=GapThresholds(low=0.25, high=0.5),
        include_covered=True,
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert gaps[0]["verdict"] == "IGNORED"
    assert gaps[0]["metadata"]["closed_history"]["state_reason"] == "not_planned"


def test_open_high_similarity_match_keeps_existing_coverage_behavior():
    need, item = _coverage_fixture(state="open")
    assert detect_gaps(
        [need],
        [item],
        embedder=KeywordEmbedder(),
        thresholds=GapThresholds(low=0.25, high=0.5),
        include_covered=False,
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    ) == []
    included = detect_gaps(
        [need],
        [item],
        embedder=KeywordEmbedder(),
        thresholds=GapThresholds(low=0.25, high=0.5),
        include_covered=True,
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert included[0]["verdict"] == "COVERED"


class _AmbiguousEmbedder:
    def encode(self, texts):
        rows = []
        for text in texts:
            rows.append([1.0, 0.0] if "ROADMAP" in text else [0.5, 0.8660254])
        return np.asarray(rows, dtype=np.float32)


class _ClosedAdjudicator:
    def __init__(self, verdict):
        self.verdict = verdict

    def adjudicate(self, *, roadmap_item, allowed_roadmap_id, **_kwargs):
        return {
            "verdict": self.verdict,
            "roadmap_id": allowed_roadmap_id,
            "roadmap_quote": roadmap_text(roadmap_item)[:8],
            "rationale": "valid structured test adjudication",
        }


def test_closed_item_rejects_covered_adjudication_but_allows_noncoverage():
    need, item = _coverage_fixture(state="closed", state_reason="completed")
    need.update(
        latent_need="NEED",
        jtbd_statement="NEED",
        symptom="NEED",
    )
    item.update(title="ROADMAP", body="ROADMAP body")
    common = {
        "embedder": _AmbiguousEmbedder(),
        "thresholds": GapThresholds(low=0.25, high=0.75),
        "include_covered": True,
        "as_of": datetime(2026, 1, 2, tzinfo=UTC),
    }
    rejected = detect_gaps(
        [need], [item], adjudicator=_ClosedAdjudicator("COVERED"), **common
    )
    assert rejected[0]["verdict"] == "IGNORED"
    assert rejected[0]["metadata"]["adjudication"] == "deterministic"

    accepted = detect_gaps(
        [need], [item], adjudicator=_ClosedAdjudicator("UNDER-PRIORITIZED"), **common
    )
    assert accepted[0]["verdict"] == "UNDER-PRIORITIZED"
    assert accepted[0]["metadata"]["adjudication"] == "llm"


def test_verdict_tree_has_deterministic_boundaries():
    thresholds = GapThresholds(low=0.3, high=0.7)
    assert deterministic_verdict(
        similarity=0.29,
        symptom_similarity=0.9,
        latent_similarity=0.1,
        low_priority=True,
        thresholds=thresholds,
    )[0] == "IGNORED"
    assert deterministic_verdict(
        similarity=0.8,
        symptom_similarity=0.8,
        latent_similarity=0.8,
        low_priority=True,
        thresholds=thresholds,
    )[0] == "UNDER-PRIORITIZED"
    assert deterministic_verdict(
        similarity=0.8,
        symptom_similarity=0.8,
        latent_similarity=0.8,
        low_priority=False,
        thresholds=thresholds,
    )[0] == "COVERED"
    # Without a framing result the old similarity delta must NOT resurrect
    # MISUNDERSTOOD: it was unreachable by construction on real data.
    assert deterministic_verdict(
        similarity=0.8,
        symptom_similarity=0.99,
        latent_similarity=0.1,
        low_priority=False,
        thresholds=thresholds,
    )[0] == "COVERED"


def test_evaluate_framing_reports_an_auditable_trail():
    from src.gaps import _framing_terms, build_document_frequencies

    roadmap = [_symptom_item(), _filler_item()]
    term_sets = [frozenset(_framing_terms(roadmap_text(item))) for item in roadmap]
    df = build_document_frequencies(term_sets)
    result = evaluate_framing(
        _framed_need(),
        candidate_indices=[0, 1],
        roadmap_rows=roadmap,
        roadmap_term_sets=term_sets,
        df=df,
        total=len(term_sets),
        thresholds=GapThresholds(),
    )
    assert result["symptom_probe"] and result["job_probe"]
    assert result["symptom_item_id"] == "R1001"
    assert "cutoff" in result["rationale"]


def test_all_three_verdicts_are_reachable_in_one_corpus():
    """The regression BLOCK-C demands: no future tweak may silently collapse
    the verdict distribution back to one branch."""

    needs = [
        _framed_need(),  # -> MISUNDERSTOOD (symptom item covers, job item absent)
        {
            "id": "N0002",
            "latent_need": "Notification controls that respect focus",
            "jtbd_statement": "silence notification noise during focused work",
            "symptom": "too many notification banners",
            "supporting_signal_ids": ["S0001"],
            "opportunity_score": 9,
        },
        {
            "id": "N0003",
            "latent_need": "Plan a podcast tour itinerary",
            "jtbd_statement": "schedule interview travel across cities",
            "symptom": "itinerary spreadsheet chaos",
            "supporting_signal_ids": ["S0001"],
            "opportunity_score": 5,
        },
    ]
    backlog_notifications = {
        "id": "R2001",
        "type": "issue",
        "title": "Notification banner settings",
        "body": "Add notification banner controls someday.",
        "state": "open",
        "labels": [],
        "milestone": "Backlog",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "priority": {
            "tier": "backlog",
            "score": 0.2,
            "is_low_priority": True,
            "has_explicit_priority": False,
        },
    }
    gaps = detect_gaps(
        needs,
        [_symptom_item(), backlog_notifications],
        signals=[{"id": "S0001", "text": "The export screen crashed again."}],
        thresholds=GapThresholds(low=0.3, high=0.9),
        include_covered=True,
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )
    verdicts = {gap["need_id"]: gap["verdict"] for gap in gaps}
    assert verdicts["N0001"] == "MISUNDERSTOOD"
    assert verdicts["N0002"] == "UNDER-PRIORITIZED"
    assert verdicts["N0003"] == "IGNORED"
    fired = sum(1 for verdict in verdicts.values() if verdict == "MISUNDERSTOOD")
    assert fired == 1  # populated, but never dominating a mixed corpus


def test_misunderstood_set_is_invariant_to_the_embedding_backend():
    """Given the same candidates, the lexical decision does not depend on the backend.

    Scoped claim only: this fixture's roadmap (2 items) is smaller than
    candidate_pool, so both embedders retrieve the same candidate set and only
    the deterministic lexical rule is exercised here. It does NOT show the
    verdict is backend-invariant end-to-end -- on a roadmap larger than
    candidate_pool, the embedding's *retrieval* stage runs first and can drop
    a MISUNDERSTOOD-supporting item from the pool before the lexical rule ever
    sees it. artifact-red-team demonstrated exactly that on the shipped
    WordPress run (REQ-main-4): swapping only the embedder moved both
    MISUNDERSTOOD verdicts to IGNORED. See config/pipeline.json's
    embedding.note for the corrected, backend-sensitive claim.
    """

    needs = [_framed_need()]
    roadmap = [_symptom_item(), _filler_item()]

    def _fired(embedder):
        gaps = detect_gaps(
            needs,
            roadmap,
            signals=[{"id": "S0001", "text": "The export screen crashed again."}],
            embedder=embedder,
            thresholds=GapThresholds(low=0.05, high=0.95),
            include_covered=True,
            as_of=datetime(2026, 1, 2, tzinfo=UTC),
        )
        return {g["need_id"] for g in gaps if g["verdict"] == "MISUNDERSTOOD"}

    assert _fired(None) == _fired(KeywordEmbedder()) == {"N0001"}


def test_framing_records_the_missing_terms_exhibit():
    gaps = _detect([_framed_need()], [_symptom_item(), _filler_item()])
    framing = gaps[0]["metadata"]["framing"]
    # Every probe term is either matched or missing, with no overlap.
    for side in ("symptom", "job"):
        probe = set(framing[f"{side}_probe"])
        matched = set(framing[f"{side}_matched_terms"])
        missing = set(framing[f"{side}_missing_terms"])
        assert matched | missing == probe
        assert not matched & missing


def test_detect_gaps_matches_ids_and_preserves_exact_signal_quotes():
    needs = [
        {
            "id": "N0001",
            "latent_need": "Reliable export",
            "jtbd_statement": "export without a crash",
            "symptom": "export crash",
            "kano_class": "basic",
            "supporting_signal_ids": ["S0001", "S0002"],
            "opportunity_score": 15,
        }
    ]
    signals = [
        {"id": "S0001", "text": "Export crashed and lost my post."},
        {"id": "S0002", "text": "Export crashes every time."},
    ]
    gaps = detect_gaps(
        needs,
        _roadmap(),
        signals=signals,
        embedder=KeywordEmbedder(),
        thresholds=GapThresholds(low=0.1, high=0.6),
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap["matched_roadmap_id"] == "R0001"
    assert gap["verdict"] == "UNDER-PRIORITIZED"
    assert gap["evidence"]["signal_ids"] == ["S0001", "S0002"]
    assert gap["evidence"]["quotes"][0]["span"] == signals[0]["text"]
    assert gap["id"].startswith("G")


def test_string_priority_fails_loud_with_the_canonical_shape_named():
    """Lead decision for Block D: the PriorityMetadata mapping is canonical.

    A bare-string priority marks an obsolete-schema input; it must fail with an
    actionable message, never be silently coerced and never crash opaquely.
    """

    legacy = dict(_roadmap()[0], priority="backlog")
    with pytest.raises(ValueError, match="PriorityMetadata"):
        priority_is_low(legacy, as_of=datetime(2026, 1, 2, tzinfo=UTC))


def test_priority_gate_reports_reasons():
    low, reasons = priority_is_low(
        _roadmap()[0],
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
        thresholds=GapThresholds(),
    )
    assert low
    assert any("tier" in reason for reason in reasons)


def test_explicit_high_priority_outranks_every_structural_proxy():
    """An issue the maintainers labelled High is not UNDER-PRIORITIZED.

    `priority_is_low` was a bare OR over reasons, so "no committed release
    milestone" alone marked an item low. On the real roadmap 770 of 774 open
    items satisfy at least one structural proxy, so without this veto the
    verdict carries almost no information -- and it produces a false claim
    about the reader's own board for the 11 issues WordPress marked High.
    """

    item = {
        "id": "R2001",
        "type": "issue",
        "title": "Editor: post contains local changes even when the user did not edit",
        "body": "Reported repeatedly.",
        "state": "open",
        "labels": ["[Pri] High", "[Type] Bug"],
        "milestone": None,  # no milestone, and open for years
        "created_at": "2019-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "priority": {
            "tier": "high",
            "score": 0.85,
            "is_low_priority": False,
            "has_explicit_priority": True,
            "matched_labels": ["[Pri] High"],
            "reasons": [],
        },
    }
    low, reasons = priority_is_low(item, as_of=datetime(2026, 8, 1, tzinfo=UTC))

    assert low is False
    assert any("labelled this high" in reason for reason in reasons)

    # An explicitly LOW label still reads low -- the veto is one-directional.
    item["labels"] = ["[Pri] Low"]
    item["priority"] = {
        "tier": "low",
        "score": 0.35,
        "is_low_priority": True,
        "has_explicit_priority": True,
        "matched_labels": ["[Pri] Low"],
        "reasons": [],
    }
    low_again, _ = priority_is_low(item, as_of=datetime(2026, 8, 1, tzinfo=UTC))
    assert low_again is True


def test_closed_class_function_words_never_become_probe_terms():
    """REQ-E-08: `across` reached a real job probe as the stem `acros`.

    It matched roadmap items about content splitting across screens and toasts
    not centering -- a preposition is never a distinctive term for what a user
    is trying to do. The filter must run on the surface form, because no
    stopword list catches `acros` after stemming.
    """

    from src.gaps import _framing_terms

    terms = _framing_terms("preserve edits across the transition between screens")
    assert "acros" not in terms and "across" not in terms
    assert "between" not in terms
    # Content words on the same line survive.
    assert "preserve" in terms and "transition" in terms

    # Negation and desiderative language is signal, not noise, and must stay.
    kept = _framing_terms("cannot publish without losing work, never again")
    for token in ("cannot", "without", "never", "again"):
        assert token in kept, f"{token} was dropped; negation carries meaning"


def test_job_coverage_is_also_reported_over_the_whole_roadmap():
    """REQ-E-08: "no candidate covers the job" is a claim about the pool.

    A job probe inflated by generic vocabulary SUPPRESSES a MISUNDERSTOOD
    verdict, and a suppressed verdict emits no artifact for anyone to review --
    the asymmetry that makes this worse than the symptom-side defect. Scoring
    the job probe over every roadmap item makes the negative claim checkable
    from the output.
    """

    gaps = _detect([_framed_need()], [_symptom_item(), _job_item(), _filler_item()])
    framing = gaps[0]["metadata"]["framing"]

    assert framing["job_corpus_scope"] == 3
    # The corpus scan is a superset of the pool, so it can only find more.
    assert framing["job_coverage_corpus"] >= framing["job_coverage"]
    # The job-covering item is named, so the claim is traceable to a row.
    assert framing["job_item_id_corpus"] == "R1002"


def test_priority_reason_diagnostics_expose_a_non_discriminating_rule():
    """A reason that fires on nearly every item is not evidence about one item.

    On the real WordPress roadmap only 18 of 774 open issues carry a milestone,
    so "no committed release milestone" marks 98% of the corpus low and
    separates almost nothing. We publish the firing rate rather than dropping
    the reason, because dropping it after seeing which split it produced would
    be choosing the rule by its outcome.
    """

    from src.gaps import priority_reason_diagnostics

    committed = dict(_COMMITTED)
    # Nine items with no milestone (the vacuous reason) and one committed item.
    roadmap = [
        {
            "id": f"R{i:04d}",
            "type": "issue",
            "title": f"item {i}",
            "body": "b",
            "state": "open",
            "labels": [],
            "milestone": None,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "priority": {
                "tier": "unspecified",
                "score": 0.5,
                "is_low_priority": False,
                "has_explicit_priority": False,
            },
        }
        for i in range(9)
    ]
    roadmap.append({
        "id": "R0009",
        "type": "issue",
        "title": "committed",
        "body": "b",
        "state": "open",
        "labels": ["[Pri] High"],
        "milestone": "7.0",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "priority": dict(committed),
    })

    diagnostics = priority_reason_diagnostics(
        roadmap, as_of=datetime(2026, 1, 2, tzinfo=UTC)
    )

    assert diagnostics["roadmap_items"] == 10
    assert diagnostics["low_priority_items"] == 9
    assert diagnostics["reason_firing_rate"]["no committed release milestone"] == 0.9
    assert "no committed release milestone" in diagnostics["non_discriminating_reasons"]
    assert "weak signal" in diagnostics["note"]


def test_detect_gaps_publishes_priority_reason_diagnostics():
    gaps = _detect([_framed_need()], [_symptom_item(), _filler_item()])
    diagnostics = gaps[0]["metadata"]["priority_reason_diagnostics"]
    assert diagnostics["roadmap_items"] == 2
    assert "reason_firing_rate" in diagnostics


def test_thresholds_are_tunable_from_labeled_examples():
    examples = [
        {"similarity": 0.1, "label": "IGNORED"},
        {
            "similarity": 0.8,
            "priority_is_low": True,
            "label": "UNDER-PRIORITIZED",
        },
        {"similarity": 0.8, "label": "COVERED"},
    ]
    tuned, metrics = tune_thresholds(
        examples,
        low_grid=[0.2],
        high_grid=[0.6],
    )
    assert tuned.low == 0.2
    assert metrics["accuracy"] == 1.0


def test_tuning_reports_out_of_fold_accuracy_not_just_in_sample():
    """The number we quote must not be scored on the rows it was fitted to.

    docs/EVALUATION_PROTOCOL.md forbids tuning and reporting on the same
    examples. With a grid this size, in-sample accuracy largely measures how
    many rows the search could memorize.
    """

    # Deliberately noisy: two rows at the same similarity carry opposite labels,
    # so no threshold can separate them and honest held-out accuracy must fall
    # below the in-sample figure.
    examples = [
        {"similarity": 0.10, "label": "IGNORED"},
        {"similarity": 0.20, "label": "IGNORED"},
        {"similarity": 0.55, "priority_is_low": True, "label": "UNDER-PRIORITIZED"},
        {"similarity": 0.55, "priority_is_low": True, "label": "IGNORED"},
        {"similarity": 0.80, "label": "COVERED"},
        {"similarity": 0.90, "label": "COVERED"},
    ]
    _, metrics = tune_thresholds(examples, folds=3)

    assert "cv_accuracy" in metrics
    assert metrics["folds"] == 3.0
    assert metrics["examples"] == 6.0
    assert 0.0 <= metrics["cv_accuracy"] <= 1.0
    # The optimism gap must be visible, not hidden.
    assert metrics["cv_accuracy"] <= metrics["accuracy"]


def test_tuning_still_returns_thresholds_when_folds_exceed_rows():
    """Two labels is not enough to cross-validate meaningfully, but it must not crash."""

    tuned, metrics = tune_thresholds(
        [
            {"similarity": 0.1, "label": "IGNORED"},
            {"similarity": 0.9, "label": "COVERED"},
        ],
        low_grid=[0.2],
        high_grid=[0.6],
        folds=10,
    )
    assert tuned.low == 0.2
    assert metrics["examples"] == 2.0


def test_verdict_stability_reports_the_flip_point_and_direction():
    """A verdict near the gate must say so, and say what it would become."""

    from src.gaps import verdict_stability

    thresholds = GapThresholds(low=0.38, high=0.62)
    # Just above the gate, low-priority match -> UNDER-PRIORITIZED, but barely.
    near = verdict_stability(
        similarity=0.39,
        low_priority=True,
        verdict="UNDER-PRIORITIZED",
        framing=None,
        thresholds=thresholds,
    )
    assert near["borderline"] is True
    assert near["margin_to_flip"] == pytest.approx(0.01)
    assert near["flips_to"] == "IGNORED"
    assert near["governing_gate"] == "low"

    # Comfortably below the gate -> IGNORED, and not borderline.
    far = verdict_stability(
        similarity=0.10,
        low_priority=True,
        verdict="IGNORED",
        framing=None,
        thresholds=thresholds,
    )
    assert far["borderline"] is False
    assert far["flips_to"] == "UNDER-PRIORITIZED"


def test_ignored_never_claims_it_would_flip_to_an_unreachable_covered():
    """A non-low-priority IGNORED gap must not promise a transition it cannot make.

    Lowering `low` past `similarity` does not hand the gap to COVERED -- the
    tree falls through to the midpoint branch, which keeps returning IGNORED
    until `low <= 2*similarity - high`. The shipped artifact previously
    advertised `flips_to: "COVERED"` on a gap where no admissible `low`
    reaches it, which is a false counterfactual in a judge-facing field.
    """

    from src.gaps import verdict_stability

    thresholds = GapThresholds(low=0.38, high=0.62)
    report = verdict_stability(
        similarity=0.193,  # far below the 0.50 midpoint
        low_priority=False,
        verdict="IGNORED",
        framing=None,
        thresholds=thresholds,
    )
    assert report["flips_to"] is None
    assert report["margin_to_flip"] is None
    assert report["borderline"] is False
    assert "structurally unreachable" in report["note"]

    # Cross-check the claim against the tree itself: no admissible `low` flips it.
    for candidate_low in (0.30, 0.19, 0.10, 0.0):
        verdict, _ = deterministic_verdict(
            similarity=0.193,
            symptom_similarity=0.0,
            latent_similarity=0.0,
            low_priority=False,
            thresholds=GapThresholds(low=candidate_low, high=0.62),
            framing=None,
        )
        assert verdict == "IGNORED", f"low={candidate_low} unexpectedly moved the verdict"

    # A gap above the midpoint genuinely can reach COVERED, and still says so.
    reachable = verdict_stability(
        similarity=0.55,
        low_priority=False,
        verdict="IGNORED",
        framing=None,
        thresholds=thresholds,
    )
    assert reachable["flips_to"] == "COVERED"


def test_framing_backed_verdicts_are_immune_to_the_similarity_gate():
    """MISUNDERSTOOD is decided by coverage, so no `low` value can move it."""

    from src.gaps import verdict_stability

    thresholds = GapThresholds(low=0.38, high=0.62)
    # Comfortably one-sided: neither coverage number is near the 0.5 cutoff.
    comfortable = verdict_stability(
        similarity=0.01,
        low_priority=False,
        verdict="MISUNDERSTOOD",
        framing={"misunderstood": True, "symptom_coverage": 0.9, "job_coverage": 0.05},
        thresholds=thresholds,
    )
    assert comfortable["governing_gate"] == "framing_coverage"
    assert comfortable["borderline"] is False
    assert comfortable["flips_to"] is None
    assert comfortable["margin_to_flip"] == pytest.approx(0.4)  # min(0.4, 0.45)

    # `similarity` plays no part in either case -- only the coverage numbers do.
    assert (
        verdict_stability(
            similarity=0.99,
            low_priority=False,
            verdict="MISUNDERSTOOD",
            framing={"misunderstood": True, "symptom_coverage": 0.9, "job_coverage": 0.05},
            thresholds=thresholds,
        )["margin_to_flip"]
        == comfortable["margin_to_flip"]
    )


def test_framing_backed_verdict_reports_the_real_coverage_margin():
    """REQ-D-04: a MISUNDERSTOOD one term from flipping must say so, not `borderline: false`."""

    from src.gaps import verdict_stability

    thresholds = GapThresholds(low=0.38, high=0.62)
    # symptom_coverage 0.517 vs job_coverage 0.469 at tau=0.5: the tighter side
    # is symptom (0.017 above the cutoff), closer than job (0.031 below it).
    report = verdict_stability(
        similarity=0.20,  # below `low` -- proves similarity is irrelevant here
        low_priority=False,
        verdict="MISUNDERSTOOD",
        framing={"misunderstood": True, "symptom_coverage": 0.517, "job_coverage": 0.469},
        thresholds=thresholds,
    )
    assert report["governing_gate"] == "framing_coverage"
    assert report["margin_to_flip"] == pytest.approx(0.017)
    assert report["borderline"] is True
    assert report["flips_to"] == "IGNORED"  # similarity=0.20 < low=0.38


def test_detect_gaps_publishes_verdict_stability():
    gaps = _detect([_framed_need()], [_symptom_item(), _filler_item()])
    stability = gaps[0]["metadata"]["verdict_stability"]
    assert "borderline" in stability and "note" in stability
