"""Friction gating and redundant-need merging.

These protect two judge-facing claims: that a cluster of praise never becomes an
"unmet need", and that merging redundant needs cannot lose a single evidence ID.
Lives beside ``test_needs.py`` rather than inside it so the block-b owner's file
stays untouched.
"""

from __future__ import annotations

import pytest

from src.needs import (
    NeedConfig,
    _validate_merge_groups,
    has_friction_language,
    infer_needs,
    is_friction_signal,
    merge_redundant_needs,
    partition_friction_signals,
)
from src.schema import LatentNeed


def _config(**overrides):
    base = {"cluster_similarity": 0.26, "min_cluster_size": 2, "llm_samples": 1}
    base.update(overrides)
    return NeedConfig(**base)


def _praise_signals():
    return [
        {"id": "S0101", "source": "review", "text": "Great app!", "rating": 5},
        {"id": "S0102", "source": "review", "text": "Amazing, love it", "rating": 5},
        {"id": "S0103", "source": "review", "text": "Perfect and reliable", "rating": 4},
    ]


def _friction_signals():
    return [
        {"id": "S0201", "source": "review", "text": "Photo upload fails every time", "rating": 1},
        {"id": "S0202", "source": "review", "text": "Photo upload keeps failing", "rating": 1},
        {"id": "S0203", "source": "review", "text": "Media gallery attach is broken", "rating": 2},
        {"id": "S0204", "source": "review", "text": "Media gallery attach broke", "rating": 2},
    ]


# --------------------------------------------------------------------------
# Friction gate
# --------------------------------------------------------------------------


def test_pure_praise_never_becomes_a_need():
    assert infer_needs(_praise_signals(), config=_config()) == []


def test_high_rating_survives_when_it_reports_a_struggle():
    # The review a rating filter alone would throw away, and the most articulate
    # evidence in the corpus.
    kept, dropped = partition_friction_signals(
        [
            {"id": "S0301", "text": "Great app, but I wish it had offline drafts", "rating": 5},
            {"id": "S0302", "text": "Love this app so much", "rating": 5},
        ]
    )
    assert [row["id"] for row in kept] == ["S0301"]
    assert [row["id"] for row in dropped] == ["S0302"]


@pytest.mark.parametrize(
    "text",
    [
        "Great app but uploads fail",
        "Nice, however the editor lags",
        "Solid, wish it had dark mode",
        "Good app, needs a better editor",
        "Fine except there is no way to reorder",
        "Would be nice to have offline mode",
    ],
)
def test_friction_language_is_detected_in_positive_reviews(text):
    assert has_friction_language(text)


@pytest.mark.parametrize(
    "text", ["Great app!", "Amazing, love it", "Perfect", "Works well and fast"]
)
def test_plain_praise_carries_no_friction_language(text):
    assert not has_friction_language(text)


def test_unrated_signals_are_kept_because_silence_is_not_praise():
    assert is_friction_signal({"id": "S0401", "text": "Great app!", "rating": None})
    assert is_friction_signal({"id": "S0402", "text": "Great app!"})


def test_boundary_rating_is_governed_by_the_configured_floor():
    four_star = {"id": "S0501", "text": "Nice app", "rating": 4}
    assert not is_friction_signal(four_star, praise_rating_floor=4)
    assert is_friction_signal(four_star, praise_rating_floor=5)


def test_filter_can_be_disabled_without_touching_pipeline_config():
    # Near-identical text so the praise genuinely clusters; this isolates the
    # toggle rather than accidentally testing min_cluster_size.
    praise = [
        {"id": "S0601", "text": "Great app, love the design", "rating": 5},
        {"id": "S0602", "text": "Great app, love the design!", "rating": 5},
    ]
    assert infer_needs(praise, config=_config()) == []
    assert infer_needs(praise, config=_config(friction_filter=False))


def test_dropped_praise_count_is_reported_for_the_defense():
    signals = _friction_signals() + _praise_signals()
    needs = infer_needs(signals, config=_config())
    assert needs
    assert needs[0]["metadata"]["praise_signals_dropped"] == 3
    assert needs[0]["metadata"]["friction_filter"] is True


# --------------------------------------------------------------------------
# Redundant-need merging
# --------------------------------------------------------------------------


def test_redundant_needs_merge_without_losing_one_evidence_id():
    signals = _friction_signals()
    unmerged = infer_needs(signals, config=_config(merge_similar_needs=False))
    merged = infer_needs(signals, config=_config(merge_similar_needs=True))

    assert len(unmerged) > len(merged), "expected redundant needs to collapse"
    before = {sid for need in unmerged for sid in need["supporting_signal_ids"]}
    after = {sid for need in merged for sid in need["supporting_signal_ids"]}
    assert before == after, "merging must not drop evidence"
    assert after == {"S0201", "S0202", "S0203", "S0204"}


def test_merged_need_records_its_provenance():
    merged = infer_needs(_friction_signals(), config=_config())
    survivor = next(
        need for need in merged if need["metadata"].get("merged_need_count", 1) > 1
    )
    assert survivor["metadata"]["merged_need_count"] >= 2
    assert len(survivor["metadata"]["merged_from"]) >= 2
    assert survivor["metadata"]["support_count"] == len(survivor["supporting_signal_ids"])


def test_merged_needs_still_satisfy_the_artifact_schema():
    for need in infer_needs(_friction_signals(), config=_config()):
        LatentNeed.model_validate(need)


def test_merge_is_order_independent():
    signals = _friction_signals()
    forward = infer_needs(signals, config=_config())
    backward = infer_needs(list(reversed(signals)), config=_config())
    assert forward == backward


def test_semantically_distinct_needs_are_left_alone():
    signals = {row["id"]: row for row in _friction_signals()}
    needs = [
        {
            "id": "Naaaaaaaaaaaa",
            "latent_need": "Recoverable media uploads",
            "jtbd_statement": "When posting, users need uploads to recover.",
            "symptom": "Uploads fail.",
            "opportunity_score": 12.0,
            "supporting_signal_ids": ["S0201", "S0202"],
            "metadata": {"aspects": ["upload"], "self_consistency": 1.0},
        },
        {
            "id": "Nbbbbbbbbbbbb",
            "latent_need": "Dependable self-hosted account access",
            "jtbd_statement": "When signing in, users need every site to open.",
            "symptom": "Login fails repeatedly.",
            "opportunity_score": 11.0,
            "supporting_signal_ids": ["S0203", "S0204"],
            "metadata": {"aspects": ["login"], "self_consistency": 1.0},
        },
    ]
    merged = merge_redundant_needs(needs, signal_by_id=signals, total=4)
    assert len(merged) == 2, "different jobs must keep separate identities"


def test_identical_offline_titles_do_collapse():
    """Documents a known limitation, so it cannot regress silently.

    ``_offline_need_frame`` labels from a fixed menu of eight domains, so two
    genuinely distinct clusters can receive byte-identical titles and will then
    merge. The cure is a more discriminating labeller (c-TF-IDF over
    cluster-distinguishing terms), not a higher merge threshold — at cosine 1.0
    there is no threshold that separates them.
    """

    signals = _friction_signals()
    unmerged = infer_needs(signals, config=_config(merge_similar_needs=False))
    assert len({need["latent_need"] for need in unmerged}) == 1
    merged = infer_needs(signals, config=_config(merge_similarity=0.999999))
    assert len(merged) == 1


# --------------------------------------------------------------------------
# The fabrication test — the executable form of the evidence contract
# --------------------------------------------------------------------------


def test_merger_proposing_unknown_need_ids_is_rejected():
    with pytest.raises(ValueError, match="unknown need IDs"):
        _validate_merge_groups(
            {"groups": [{"canonical_need_id": "Naaaaaaaaaaaa",
                         "member_need_ids": ["Naaaaaaaaaaaa", "Nffffffffffff"]}]},
            ["Naaaaaaaaaaaa"],
        )


def test_merger_reusing_a_need_in_two_groups_is_rejected():
    with pytest.raises(ValueError, match="reused need IDs"):
        _validate_merge_groups(
            {
                "groups": [
                    {"canonical_need_id": "N1", "member_need_ids": ["N1", "N2"]},
                    {"canonical_need_id": "N2", "member_need_ids": ["N2", "N3"]},
                ]
            },
            ["N1", "N2", "N3"],
        )


def test_fabricating_merger_cannot_inject_evidence_into_the_output():
    class FabricatingMerger:
        def propose_groups(self, *, needs, allowed_need_ids, temperature):  # pylint: disable=unused-argument
            return {
                "groups": [
                    {
                        "canonical_need_id": allowed_need_ids[0],
                        "member_need_ids": [allowed_need_ids[0], "Ndeadbeef0000"],
                    }
                ]
            }

    signals = _friction_signals()
    needs = infer_needs(signals, merger=FabricatingMerger(), config=_config())

    assert needs, "a bad merger must not empty the pipeline"
    cited = {sid for need in needs for sid in need["supporting_signal_ids"]}
    assert cited <= {row["id"] for row in signals}
    assert "Ndeadbeef0000" not in {need["id"] for need in needs}


def test_merger_failure_falls_back_to_the_deterministic_merge():
    class ExplodingMerger:
        def propose_groups(self, **kwargs):
            raise RuntimeError("no credentials")

    signals = _friction_signals()
    with_merger = infer_needs(signals, merger=ExplodingMerger(), config=_config())
    without = infer_needs(signals, config=_config())
    assert with_merger == without


def test_merge_preserves_evidence_when_called_directly():
    signals = {row["id"]: row for row in _friction_signals()}
    needs = [
        {
            "id": "Naaaaaaaaaaaa",
            "latent_need": "Recoverable media uploads",
            "jtbd_statement": "When posting, users need uploads to recover.",
            "symptom": "Uploads fail.",
            "opportunity_score": 12.0,
            "supporting_signal_ids": ["S0201", "S0202"],
            "metadata": {"aspects": ["upload"], "self_consistency": 1.0},
        },
        {
            "id": "Nbbbbbbbbbbbb",
            "latent_need": "Recoverable media uploads",
            "jtbd_statement": "When posting, users need uploads to recover.",
            "symptom": "Uploads fail.",
            "opportunity_score": 11.0,
            "supporting_signal_ids": ["S0203", "S0204"],
            "metadata": {"aspects": ["gallery"], "self_consistency": 1.0},
        },
    ]
    merged = merge_redundant_needs(needs, signal_by_id=signals, total=4)
    assert len(merged) == 1
    assert merged[0]["supporting_signal_ids"] == ["S0201", "S0202", "S0203", "S0204"]
    assert merged[0]["metadata"]["merged_from"] == ["Naaaaaaaaaaaa", "Nbbbbbbbbbbbb"]
