"""The arithmetic contract every downstream stage assumes about embeddings.

`gaps.py` ranks roadmap candidates by cosine similarity and `needs.py` clusters
and merges on it. Both treat the numbers as a metric: bounded, symmetric, and
never NaN. A single NaN row silently poisons an `argmax` into picking element
zero, which would surface as a *wrong roadmap match* — a judge-facing error with
no traceback to follow back here.

`test_embedding_resolution.py` covers model-name resolution and load failure for
the sentence-transformers path. These cover the maths and backend selection,
which run on every path including the offline demo.

All offline: the hashing backend needs no model and no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.embedding import (
    HashingEmbedder,
    cosine_similarity_matrix,
    embed_texts,
    get_embedder,
    normalize_rows,
)


def test_empty_text_scores_zero_similarity_rather_than_nan():
    """A blank review must not become a maximally-similar match.

    Dividing a zero vector by its zero norm yields NaN, and `np.argmax` over a
    NaN row returns index 0 — so an empty signal would silently "match" whatever
    roadmap item happens to sort first.
    """

    vectors = HashingEmbedder(dimensions=64).encode(["", "export fails"])
    assert np.linalg.norm(vectors[0]) == 0.0
    similarity = cosine_similarity_matrix(vectors[0], vectors[1])
    assert similarity[0, 0] == 0.0
    assert not np.isnan(similarity).any()


def test_normalize_rows_leaves_a_zero_vector_at_zero():
    normalized = normalize_rows(np.zeros(5, dtype=np.float32))
    assert normalized.shape == (5,)
    assert not np.isnan(normalized).any()
    assert np.array_equal(normalized, np.zeros(5, dtype=np.float32))


def test_identical_text_is_perfectly_similar_to_itself():
    vectors = HashingEmbedder(dimensions=128).encode(["media upload stalls"])
    assert cosine_similarity_matrix(vectors, vectors)[0, 0] == pytest.approx(1.0)


def test_mismatched_dimensions_are_rejected_instead_of_broadcast():
    """Two backends with different widths must fail loudly, not silently align."""

    with pytest.raises(ValueError, match="dimensions do not match"):
        cosine_similarity_matrix(np.zeros((1, 4)), np.zeros((1, 5)))


def test_similarity_rejects_arrays_that_are_not_vectors_or_matrices():
    with pytest.raises(ValueError, match="vectors or 2-D matrices"):
        cosine_similarity_matrix(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))


def test_normalize_rows_rejects_arrays_that_are_not_vectors_or_matrices():
    with pytest.raises(ValueError, match="vector or a 2-D matrix"):
        normalize_rows(np.zeros((2, 2, 2)))


def test_hashing_is_deterministic_across_separate_instances():
    """Determinism must survive process and object boundaries, not just calls.

    Two stages construct their own embedder; if the two disagreed, artifacts
    would stop being byte-reproducible between runs.
    """

    first = HashingEmbedder(dimensions=64).encode(["export fails", "login drops"])
    second = HashingEmbedder(dimensions=64).encode(["export fails", "login drops"])
    assert np.array_equal(first, second)


def test_hashing_honours_the_configured_width():
    assert HashingEmbedder(dimensions=32).encode(["anything"]).shape == (1, 32)


def test_degenerate_hashing_configuration_is_rejected_at_construction():
    """Fail at construction, not thousands of encoded rows later."""

    with pytest.raises(ValueError, match="at least 32"):
        HashingEmbedder(dimensions=16)
    with pytest.raises(ValueError, match="n-gram family"):
        HashingEmbedder(word_ngrams=(), char_ngrams=())


@pytest.mark.parametrize("alias", ["hash", "hashing", "offline", "HASHING", "Offline"])
def test_offline_backend_aliases_all_resolve_to_hashing(alias):
    """`--embedding offline` is what a teammate types when the network is down."""

    assert isinstance(get_embedder(alias), HashingEmbedder)


def test_unknown_backend_names_its_own_mistake():
    with pytest.raises(ValueError, match="unknown embedding backend"):
        get_embedder("sentence-transformer")  # singular: a real typo


def test_embed_texts_consumes_a_one_shot_iterable():
    """Callers pass generators; materializing them is the wrapper's job."""

    vectors = embed_texts(text for text in ["alpha", "beta"])
    assert vectors.shape[0] == 2
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0])
