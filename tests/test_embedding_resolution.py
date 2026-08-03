"""Model-name resolution and load validation for the sentence-transformers path.

Protects against a failure that cost a full pipeline run: ``pipeline.json`` ships
``"embedding": {"model": null}``, that null reaches the adapter as ``None``, and
``SentenceTransformer(None)`` returns an *empty* model instead of raising. The
run then died with ``'NoneType' object has no attribute 'tokenize'`` inside
encode(), several stages after the real mistake.

These run offline — no model is downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.embedding import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEmbedder,
    _require_loaded,
    get_embedder,
)


class _FakeModel:
    """Minimal stand-in with a non-empty body, so no download is needed."""

    def __init__(self, body: object = "transformer") -> None:
        self._body = body

    def __getitem__(self, index: int) -> object:
        if index != 0:
            raise IndexError(index)
        return self._body

    def encode(self, texts, **_kwargs):
        return np.ones((len(texts), 4), dtype=np.float32)


def test_null_model_name_falls_back_to_the_default():
    # config/pipeline.json ships `"model": null`; config.get returns None because
    # the key exists, so the documented default never applies upstream.
    embedder = SentenceTransformerEmbedder(model_name=None, model=_FakeModel())
    assert embedder.model_name == DEFAULT_MODEL_NAME


def test_blank_model_name_falls_back_to_the_default():
    embedder = SentenceTransformerEmbedder(model_name="   ", model=_FakeModel())
    assert embedder.model_name == DEFAULT_MODEL_NAME


def test_explicit_model_name_is_preserved():
    embedder = SentenceTransformerEmbedder(
        model_name="BAAI/bge-small-en-v1.5", model=_FakeModel()
    )
    assert embedder.model_name == "BAAI/bge-small-en-v1.5"


def test_get_embedder_tolerates_a_null_model_name():
    # Must not raise while selecting the backend; hashing needs no model at all.
    assert get_embedder("hashing", model_name=None) is not None


def test_empty_transformer_body_is_rejected_with_a_named_cause():
    with pytest.raises(RuntimeError, match="did not load"):
        _require_loaded(_FakeModel(body=None), "some/model")


def test_unindexable_model_is_rejected():
    class NoIndex:
        pass

    with pytest.raises(RuntimeError, match="did not load"):
        _require_loaded(NoIndex(), "some/model")


def test_loaded_model_passes_validation():
    _require_loaded(_FakeModel(), "some/model")  # must not raise


def test_encode_normalizes_rows():
    embedder = SentenceTransformerEmbedder(model_name=None, model=_FakeModel())
    vectors = embedder.encode(["a", "b"])
    assert vectors.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
