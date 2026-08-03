"""Structural guarantees of the clustering step, independent of cluster quality.

`cluster_quality` (the skill) governs whether clusters are *good*. These tests
cover something narrower and more load-bearing: whether the partition is
**sound**. Every downstream count a judge sees — signals behind a need, evidence
IDs in a trace, the praise-filter tally — assumes each signal lands in exactly
one cluster and that the assignment does not depend on input order.

The rule these protect most directly is the skill's "noise stays noise": never
force-assign a point to its nearest cluster to tidy the numbers, because
manufactured themes do not survive a judge reading the evidence trace.

Offline: the default hashing backend needs no model.
"""

from __future__ import annotations

from src.needs import cluster_signals

_EXPORT_A = "Export fails every time and loses my draft"
_EXPORT_B = "Export fails when I save a draft"
_UNRELATED = "Dark mode colours look great at night"


def _signals(*texts: str) -> list[dict[str, object]]:
    return [
        {"id": f"S{index:04d}", "source": "review", "text": text, "rating": 2}
        for index, text in enumerate(texts, start=1)
    ]


def _ids(clusters) -> list[list[str]]:
    return [[str(signal["id"]) for signal in cluster] for cluster in clusters]


def test_no_signals_produce_no_clusters():
    assert cluster_signals([]) == []


def test_every_signal_lands_in_exactly_one_cluster():
    """The partition property: no signal duplicated, none dropped.

    A duplicated signal inflates the evidence count behind a need; a dropped one
    means a review we ingested is silently missing from the analysis.
    """

    signals = _signals(_EXPORT_A, _EXPORT_B, _UNRELATED, "Login keeps signing me out")
    assigned = [identifier for cluster in _ids(cluster_signals(signals)) for identifier in cluster]

    assert sorted(assigned) == [str(signal["id"]) for signal in signals]
    assert len(assigned) == len(set(assigned))


def test_clustering_does_not_depend_on_input_order():
    """Re-ingesting in a different order must not move the needs.

    `infer_needs` is already asserted stable end to end; this pins the stage that
    actually makes it stable, so a regression names the guilty step.
    """

    signals = _signals(_EXPORT_A, _EXPORT_B, _UNRELATED)
    assert _ids(cluster_signals(signals)) == _ids(cluster_signals(list(reversed(signals))))


def test_an_unrelated_signal_is_not_absorbed_by_the_nearest_cluster():
    """Noise stays noise. This is the invariant most tempting to break.

    Absorbing outliers makes clusters look larger and the deck look better, and
    it is exactly how a theme gets manufactured out of unrelated reviews.
    """

    clusters = cluster_signals(_signals(_EXPORT_A, _EXPORT_B, _UNRELATED))
    unrelated = [cluster for cluster in clusters if any(s["text"] == _UNRELATED for s in cluster)]

    assert len(unrelated) == 1
    assert len(unrelated[0]) == 1, "an unrelated review was force-assigned to a cluster"


def test_threshold_zero_collapses_everything_into_one_cluster():
    """The permissive boundary, not a comfortable middle value."""

    clusters = cluster_signals(_signals(_EXPORT_A, _UNRELATED), similarity_threshold=0.0)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_threshold_above_one_leaves_every_signal_alone():
    """The strict boundary: cosine cannot exceed 1, so nothing may ever merge."""

    signals = _signals(_EXPORT_A, _EXPORT_B, _UNRELATED)
    clusters = cluster_signals(signals, similarity_threshold=1.01)
    assert len(clusters) == len(signals)
    assert all(len(cluster) == 1 for cluster in clusters)


def test_near_duplicate_wording_does_group_at_the_default_threshold():
    """The counterweight: the guard above must not pass by clustering nothing."""

    clusters = cluster_signals(_signals(_EXPORT_A, _EXPORT_B))
    assert _ids(clusters) == [["S0001", "S0002"]]
