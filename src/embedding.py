"""Small, deterministic text embeddings with an optional SBERT backend.

The hashing backend is deliberately dependency-light and network-free.  It is
not intended to beat a trained embedding model; it is the reliable fallback
that keeps the complete pipeline runnable during a demo.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)?", re.UNICODE)


class Embedder(Protocol):
    """The only embedding interface used by the analysis stages."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a two-dimensional, row-normalized float array."""


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text or "")]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """L2-normalize a vector or matrix without producing NaNs for empty rows."""

    array = np.asarray(values, dtype=np.float32)
    was_vector = array.ndim == 1
    if was_vector:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("embeddings must be a vector or a 2-D matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)
    return normalized[0] if was_vector else normalized


@dataclass(slots=True)
class HashingEmbedder:
    """Deterministic feature-hashing baseline using word and character n-grams."""

    dimensions: int = 768
    word_ngrams: tuple[int, ...] = (1, 2)
    char_ngrams: tuple[int, ...] = (3, 4)

    def __post_init__(self) -> None:
        if self.dimensions < 32:
            raise ValueError("dimensions must be at least 32")
        if not self.word_ngrams and not self.char_ngrams:
            raise ValueError("at least one n-gram family is required")

    def _add_feature(self, row: np.ndarray, feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, byteorder="little", signed=False)
        index = raw % self.dimensions
        sign = 1.0 if (raw >> 63) == 0 else -1.0
        row[index] += sign * weight

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in zip(rows, texts, strict=True):
            tokens = _tokens(str(text))
            for n in self.word_ngrams:
                if n <= 0:
                    continue
                weight = 1.0 / math.sqrt(n)
                for start in range(0, len(tokens) - n + 1):
                    self._add_feature(
                        row, "w:" + "\x1f".join(tokens[start : start + n]), weight
                    )

            # Character features make spelling variants less brittle.  Boundary
            # markers prevent unrelated substrings from receiving full credit.
            collapsed = " ".join(tokens)
            for token in tokens:
                bounded = f"^{token}$"
                for n in self.char_ngrams:
                    if n <= 0:
                        continue
                    weight = 0.35 / math.sqrt(n)
                    for start in range(0, len(bounded) - n + 1):
                        self._add_feature(
                            row, f"c{n}:{bounded[start:start + n]}", weight
                        )
            if not tokens and collapsed == "":
                # Empty strings intentionally remain zero vectors.
                continue
        return normalize_rows(rows)


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _require_loaded(model: object, model_name: str) -> None:
    """Fail loudly if the transformer body did not actually load.

    ``SentenceTransformer`` returns a usable-looking object whose first module is
    ``None`` when the name is empty or a download did not complete. Encoding then
    dies with an opaque ``'NoneType' has no attribute 'tokenize'`` several stages
    downstream. A teammate on a cold cache hits this, so it must name its cause.
    """

    try:
        body = model[0]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        body = None
    if body is None:
        raise RuntimeError(
            f"sentence-transformers model {model_name!r} did not load "
            "(the transformer body is empty). This usually means the model "
            "download was interrupted or the name is wrong. Re-run to finish "
            "the download, or use --embedding hashing to stay offline."
        )


class SentenceTransformerEmbedder:
    """Lazy sentence-transformers adapter.

    Import and model loading happen only when this backend is explicitly
    selected, so importing the project never downloads a model.
    """

    def __init__(
        self,
        model_name: str | None = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        model: object | None = None,
    ) -> None:
        # ``config/pipeline.json`` ships ``"embedding": {"model": null}`` because
        # shipped runs use hashing. A null reaches us as None, and
        # ``SentenceTransformer(None)`` returns an *empty* model without raising
        # — the failure then surfaces as an AttributeError inside encode(), long
        # after ingest. Resolve it here instead of crashing mid-pipeline.
        resolved = (model_name or "").strip() or DEFAULT_MODEL_NAME
        if model is None:
            try:
                from sentence_transformers import (  # type: ignore[import-not-found]
                    SentenceTransformer,
                )
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "sentence-transformers is not installed; use backend='hashing' "
                    "or install the optional embedding dependency"
                ) from exc
            model = SentenceTransformer(resolved, device=device)
            _require_loaded(model, resolved)
        self.model_name = resolved
        self._model: Any = model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return normalize_rows(np.asarray(values, dtype=np.float32))


def get_embedder(
    backend: str = "hashing",
    *,
    model_name: str | None = DEFAULT_MODEL_NAME,
    dimensions: int = 768,
    device: str | None = None,
) -> Embedder:
    """Construct a configured embedding backend.

    ``auto`` prefers a locally available sentence-transformers installation,
    but safely falls back to hashing if the import or model load fails.
    """

    selected = backend.casefold().replace("_", "-")
    if selected in {"hash", "hashing", "offline"}:
        return HashingEmbedder(dimensions=dimensions)
    if selected in {"sentence-transformers", "sbert"}:
        return SentenceTransformerEmbedder(model_name=model_name, device=device)
    if selected == "auto":
        try:
            return SentenceTransformerEmbedder(model_name=model_name, device=device)
        except (RuntimeError, OSError, ValueError):
            return HashingEmbedder(dimensions=dimensions)
    raise ValueError(f"unknown embedding backend: {backend!r}")


def cosine_similarity_matrix(
    left: np.ndarray | Sequence[Sequence[float]],
    right: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Return pairwise cosine similarities, with empty vectors scoring zero."""

    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("left and right must be vectors or 2-D matrices")
    if a.shape[1] != b.shape[1]:
        raise ValueError("embedding dimensions do not match")
    return normalize_rows(a) @ normalize_rows(b).T


def embed_texts(
    texts: Iterable[str], embedder: Embedder | None = None
) -> np.ndarray:
    """Convenience wrapper used by small scripts and tests."""

    materialized = [str(text) for text in texts]
    return (embedder or HashingEmbedder()).encode(materialized)
