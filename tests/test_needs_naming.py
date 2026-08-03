"""Titles must name a product need, or the cluster must not become a need.

On the real WordPress corpus the offline labeller emitted "Reliable its",
"Reliable more" and "Reliable kadang" — titles built from whatever token was
most frequent after stopword filtering. The clusters behind them were residue:
one-word verdicts, stray names, and unfocused praise. A title is an assertion
about users, so inventing one for evidence that supports none is the failure
these tests exist to prevent.
"""

from __future__ import annotations

import src.needs as needs_module
from src.needs import (
    NeedConfig,
    _cluster_aspects,
    _offline_need_frame,
    infer_needs,
    is_unnameable_cluster,
)


def _signals(texts, rating=2, start=1):
    return [
        {"id": f"S{start + i:06d}", "source": "review", "text": text, "rating": rating}
        for i, text in enumerate(texts)
    ]


RESIDUE = [
    "Bs",
    "Best Best",
    "Very bad Very bad",
    "Nyz Ryt",
    "Usefull",
    "It OK New in it but love it and expect better performance from it",
]

MEDIA = [
    "Photo upload fails every time",
    "Photo upload keeps failing on my phone",
    "Media gallery attach is broken",
    "Image upload stalls forever",
]


def test_residue_cluster_is_not_nameable():
    cluster = _signals(RESIDUE)
    assert is_unnameable_cluster(cluster, _cluster_aspects(cluster))


def test_product_cluster_is_nameable():
    cluster = _signals(MEDIA)
    assert not is_unnameable_cluster(cluster, _cluster_aspects(cluster))


def test_residue_never_reaches_the_output():
    needs = infer_needs(
        _signals(RESIDUE),
        config=NeedConfig(cluster_similarity=0.0, min_cluster_size=2, llm_samples=1),
    )
    assert needs == []


def test_dropped_count_is_reported_on_every_need():
    # One nameable cluster and one cohesive residue cluster. The surviving need
    # must carry the *final* count, not the value the counter happened to hold
    # when that need was built.
    signals = _signals(
        ["Photo upload fails often", "Photo upload fails again", "Photo upload fails badly"]
    ) + _signals(
        ["Nice more some better", "Nice more some worse", "Nice more some okay"], start=500
    )
    needs = infer_needs(
        signals,
        config=NeedConfig(cluster_similarity=0.2, min_cluster_size=2, llm_samples=1),
    )
    assert [need["latent_need"] for need in needs] == ["Recoverable media uploads"]
    assert needs[0]["metadata"]["unnameable_clusters_dropped"] == 1


def test_gate_can_be_disabled():
    needs = infer_needs(
        _signals(RESIDUE),
        config=NeedConfig(
            cluster_similarity=0.0,
            min_cluster_size=2,
            llm_samples=1,
            drop_unnameable_clusters=False,
        ),
    )
    assert needs, "residue should survive when the gate is off"


def test_title_never_uses_an_unrecognised_token():
    # "kadang" is a real cluster aspect from the corpus; a length check cannot
    # tell it from a product noun, so only known domain nouns may name a need.
    cluster = _signals(["kadang error", "kadang nice", "kadang some"])
    title, _job, _symptom = _offline_need_frame(cluster, ["kadang", "nice", "some"])
    assert "kadang" not in title
    assert title == "Reliable product workflow"


def test_title_prefers_a_recognised_product_noun():
    # Cluster text matches no domain strongly enough to be framed directly, so
    # the fallback runs and must pick the product noun over the frequent filler.
    cluster = _signals(["nice more some", "more some nice"])
    title, _job, _symptom = _offline_need_frame(cluster, ["more", "theme", "nice"])
    assert title == "Reliable theme"


def test_weak_tokens_never_become_titles():
    for weak in ("its", "more", "nice", "better", "some"):
        cluster = _signals([f"{weak} thing", f"{weak} stuff"])
        title, _job, _symptom = _offline_need_frame(cluster, [weak, "thing"])
        assert weak not in title.split(), f"{weak!r} leaked into {title!r}"


def test_off_domain_cluster_reaches_successful_extractor():
    class Extractor:
        calls = 0

        def extract(self, *, allowed_signal_ids, **_kwargs):
            self.calls += 1
            return {
                "latent_need": "Recoverable game saves",
                "jtbd_statement": "Resume a game from a durable checkpoint.",
                "kano_class": "basic",
                "root_cause_hypothesis": "Save-state persistence may be incomplete.",
                "symptom": "Game checkpoints disappear.",
                "supporting_signal_ids": list(allowed_signal_ids),
            }

    extractor = Extractor()
    needs = infer_needs(
        _signals(["savestate checkpoint vanished", "game checkpoint was erased"]),
        extractor=extractor,
        config=NeedConfig(
            cluster_similarity=0.0,
            min_cluster_size=2,
            llm_samples=1,
            merge_similar_needs=False,
        ),
    )
    assert extractor.calls == 1
    assert [need["latent_need"] for need in needs] == ["Recoverable game saves"]
    assert needs[0]["metadata"]["inference"] == "gemini"


def test_failed_extractor_does_not_fabricate_off_domain_fallback(monkeypatch):
    off_domain = _signals(["savestate vanished", "checkpoint erased"], start=1)
    media = _signals(["photo upload fails", "image upload stalls"], start=100)
    monkeypatch.setattr(
        needs_module,
        "cluster_signals",
        lambda *_args, **_kwargs: [off_domain, media],
    )

    class FailingExtractor:
        def extract(self, **_kwargs):
            raise RuntimeError("offline test failure")

    needs = infer_needs(
        [*off_domain, *media],
        extractor=FailingExtractor(),
        config=NeedConfig(
            min_cluster_size=2,
            llm_samples=1,
            merge_similar_needs=False,
        ),
    )
    assert [need["latent_need"] for need in needs] == ["Recoverable media uploads"]
    assert needs[0]["metadata"]["unnameable_clusters_dropped"] == 1


def test_unnameable_cluster_does_not_consume_max_needs(monkeypatch):
    junk = _signals(["nonsense token"] * 3, start=1)
    media = _signals(["photo upload fails", "image upload stalls"], start=100)
    monkeypatch.setattr(needs_module, "cluster_signals", lambda *_args, **_kwargs: [junk, media])
    needs = infer_needs(
        [*junk, *media],
        config=NeedConfig(
            min_cluster_size=2,
            llm_samples=1,
            max_needs=1,
            merge_similar_needs=False,
        ),
    )
    assert [need["latent_need"] for need in needs] == ["Recoverable media uploads"]
    assert needs[0]["metadata"]["unnameable_clusters_dropped"] == 1


def test_cluster_aspects_exclude_intensifiers():
    cluster = _signals(
        ["absolutely terrible really broken", "totally bad extremely broken"]
    )
    assert _cluster_aspects(cluster) == ["product use"]


def test_offline_evidence_excludes_cross_domain_bridge_signal(monkeypatch):
    drafting = _signals(
        ["draft editor save fails", "post draft publish fails"], start=1
    )
    site_bridge = _signals(["multiple sites dashboard picks wrong blog"], start=100)
    cluster = [*drafting, *site_bridge]
    monkeypatch.setattr(
        needs_module, "cluster_signals", lambda *_args, **_kwargs: [cluster]
    )
    needs = infer_needs(
        cluster,
        config=NeedConfig(
            min_cluster_size=2,
            llm_samples=1,
            merge_similar_needs=False,
        ),
    )
    assert len(needs) == 1
    assert needs[0]["latent_need"] == "Lossless drafting and publishing"
    assert set(needs[0]["supporting_signal_ids"]) == {"S000001", "S000002"}
    assert needs[0]["metadata"]["semantic_signals_excluded"] == 1


def test_small_domain_fragment_attaches_to_one_existing_offline_need(monkeypatch):
    main_cluster = _signals(["photo upload fails", "image upload stalls"], start=1)
    small_fragment = _signals(["media gallery upload is broken"], start=100)
    monkeypatch.setattr(
        needs_module,
        "cluster_signals",
        lambda *_args, **_kwargs: [main_cluster, small_fragment],
    )
    needs = infer_needs(
        [*main_cluster, *small_fragment],
        config=NeedConfig(
            min_cluster_size=2,
            llm_samples=1,
            merge_similar_needs=False,
        ),
    )
    assert len(needs) == 1
    assert set(needs[0]["supporting_signal_ids"]) == {
        "S000001",
        "S000002",
        "S000100",
    }
    assert needs[0]["metadata"]["domain_signals_attached"] == 1
