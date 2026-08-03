"""Gemini access layer: credentials, resilience, caching, and an offline stub.

The analysis stages (:mod:`src.needs`, :mod:`src.gaps`, :mod:`src.verify`) take an
*injected* client and only ever call::

    client.models.generate_content(model=..., contents=..., config={...})

This module supplies that object.  Everything that is not deterministic
analysis -- authentication, backend selection, retry/backoff, rate limiting,
response caching, and call accounting -- lives here so the analysis core stays
readable and testable.

Three backends, chosen automatically from the environment:

``vertex``
    Vertex AI.  Uses Application Default Credentials, or an express-mode API
    key when one is supplied.  Set ``GOOGLE_GENAI_USE_VERTEXAI=true`` plus
    ``GOOGLE_CLOUD_PROJECT``.
``developer_api``
    The Gemini Developer API, keyed by ``GOOGLE_API_KEY``.
``offline``
    A deterministic stub that synthesizes a schema-valid response without any
    network access.  Selected automatically when no credentials are present, or
    forced with ``FIRECODE_OFFLINE=true``.

The offline backend is not a toy: it keeps the full pipeline, the test suite,
and a conference-wifi demo runnable, and it makes every LLM-dependent stage
reproducible byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from .config import CACHE_DIR, LLMConfig

logger = logging.getLogger(__name__)

# Error fragments that mean "retry later" rather than "your request is wrong".
_RETRYABLE_MARKERS = (
    "429",
    "resource_exhausted",
    "rate limit",
    "quota",
    "500",
    "502",
    "503",
    "504",
    "unavailable",
    "deadline",
    "timeout",
    "internal error",
)


class LLMError(RuntimeError):
    """Raised when a generation cannot be completed after all retries."""


@dataclass(slots=True)
class CallStats:
    """Cheap accounting so cost and cache behaviour are demonstrable."""

    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }


class _Response:
    """Minimal stand-in for a genai response (offline and cache replay)."""

    __slots__ = ("text", "parsed", "usage_metadata", "from_cache")

    def __init__(self, text: str, *, from_cache: bool = False) -> None:
        self.text = text
        self.from_cache = from_cache
        self.usage_metadata = None
        try:
            self.parsed: Any = json.loads(text)
        except (TypeError, ValueError):
            self.parsed = None


def _is_retryable(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    if isinstance(code, int) and (code == 429 or 500 <= code < 600):
        return True
    message = f"{type(exc).__name__}: {exc}".casefold()
    return any(marker in message for marker in _RETRYABLE_MARKERS)


def _canonical(value: Any) -> Any:
    """Deterministically normalize a config/contents object for hashing."""

    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _canonical(dump())
    return repr(value)


def cache_key(model: str, contents: Any, config: Any, attempt: int = 0) -> str:
    """Stable digest over everything that can change a response."""

    payload = json.dumps(
        {
            "model": model,
            "contents": _canonical(contents),
            "config": _canonical(config),
            "attempt": attempt,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Offline synthesis
# ---------------------------------------------------------------------------

# Field names we can answer sensibly, so offline output reads like real output
# instead of lorem ipsum.  Anything unlisted falls back to a generic sentence.
_OFFLINE_TEXT: dict[str, str] = {
    "latent_need": "Reliably resume interrupted work without losing edits",
    "jtbd_statement": (
        "When I am interrupted mid-task on a flaky connection, I want my work "
        "preserved automatically so I do not have to redo it"
    ),
    "root_cause_hypothesis": (
        "Hypothesis: state is committed only on an explicit successful sync, so "
        "any interruption discards work"
    ),
    "symptom": "Users report losing drafts and edits after connection errors",
    "rationale": "Offline stub rationale: deterministic placeholder, not a model judgment",
    "roadmap_quote": "",
    "reason": "Offline stub reason: deterministic placeholder",
}


def _seeded_rng(*parts: str) -> random.Random:
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "little"))


def synthesize_from_schema(schema: Mapping[str, Any], rng: random.Random, name: str = "") -> Any:
    """Build a deterministic value satisfying a (subset of) JSON Schema.

    Supports the constructs this project actually emits: objects with
    ``properties``/``required``, arrays with ``items``, strings with ``enum``,
    plus numbers, integers, and booleans.  Enum values are always honoured,
    which is what makes offline evidence IDs valid rather than invented.
    """

    if not isinstance(schema, Mapping):
        return None

    enum = schema.get("enum")
    if isinstance(enum, Sequence) and not isinstance(enum, (str, bytes)) and enum:
        return enum[rng.randrange(len(enum))]

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t != "null"), "string")

    if schema_type == "object":
        properties = schema.get("properties") or {}
        required = schema.get("required") or list(properties)
        result: dict[str, Any] = {}
        for key in required:
            sub = properties.get(key, {"type": "string"})
            result[key] = synthesize_from_schema(sub, rng, key)
        return result

    if schema_type == "array":
        items = schema.get("items") or {"type": "string"}
        item_enum = items.get("enum") if isinstance(items, Mapping) else None
        is_seq = isinstance(item_enum, Sequence) and not isinstance(item_enum, (str, bytes))
        if is_seq and item_enum:
            # Evidence-ID arrays: return a deterministic, non-empty subset of the
            # allowed values, never an invented identifier.
            pool = list(item_enum)
            take = max(1, min(len(pool), schema.get("minItems", 0) or min(3, len(pool))))
            rng.shuffle(pool)
            return sorted(pool[:take])
        count = max(1, int(schema.get("minItems", 1) or 1))
        return [synthesize_from_schema(items, rng, name) for _ in range(count)]

    if schema_type == "integer":
        return int(rng.randrange(1, 6))
    if schema_type == "number":
        return round(rng.uniform(0.0, 1.0), 4)
    if schema_type == "boolean":
        return bool(rng.getrandbits(1))

    return _OFFLINE_TEXT.get(name, f"offline placeholder for {name or 'value'}")


def _extract_schema(config: Any) -> Mapping[str, Any] | None:
    if not isinstance(config, Mapping):
        return None
    for key in ("response_json_schema", "response_schema"):
        candidate = config.get(key)
        if isinstance(candidate, Mapping):
            return candidate
        if candidate is not None:
            dumped = getattr(candidate, "model_json_schema", None)
            if callable(dumped):
                return dumped()
    return None


class _OfflineModels:
    def __init__(self, owner: OfflineClient) -> None:
        self._owner = owner

    def generate_content(self, *, model: str, contents: Any, config: Any = None) -> _Response:
        self._owner.stats.calls += 1
        schema = _extract_schema(config)
        temperature = 0.0
        if isinstance(config, Mapping):
            temperature = float(config.get("temperature", 0.0) or 0.0)
        # Temperature participates in the seed so self-consistency sampling
        # produces genuine variation offline instead of N identical answers.
        rng = _seeded_rng(model, json.dumps(_canonical(contents))[:4000], f"{temperature:.3f}")
        if schema is None:
            return _Response(json.dumps({"text": "offline placeholder response"}))
        return _Response(json.dumps(synthesize_from_schema(schema, rng), ensure_ascii=False))


class OfflineClient:
    """Deterministic, network-free stand-in with the genai client surface."""

    backend = "offline"

    def __init__(self) -> None:
        self.stats = CallStats()
        self.models = _OfflineModels(self)


class _RateLimiter:
    """Minimum-interval limiter; keeps bursts under a per-minute ceiling."""

    def __init__(self, requests_per_minute: int) -> None:
        self._interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            self._next_at = max(now, self._next_at) + self._interval
        if wait > 0:
            time.sleep(wait)


class _ResilientModels:
    def __init__(self, owner: ResilientClient) -> None:
        self._owner = owner

    def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
        owner = self._owner
        key = cache_key(model, contents, config)

        cached = owner.read_cache(key)
        if cached is not None:
            owner.stats.cache_hits += 1
            return _Response(cached, from_cache=True)

        delay = 1.0
        last_exc: BaseException | None = None
        for attempt in range(owner.cfg.max_retries):
            owner.limiter.acquire()
            with owner.semaphore:
                try:
                    response = owner.call_upstream(model=model, contents=contents, config=config)
                except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                    last_exc = exc
                    if not _is_retryable(exc) or attempt == owner.cfg.max_retries - 1:
                        break
                    owner.stats.retries += 1
                    logger.warning(
                        "gemini call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        owner.cfg.max_retries,
                        delay,
                        exc,
                    )
                else:
                    owner.stats.calls += 1
                    owner.record_usage(response)
                    text = getattr(response, "text", None)
                    if text:
                        owner.write_cache(key, text)
                    return response
            # Full jitter backoff: spreads a team's concurrent retries apart.
            time.sleep(delay * (0.5 + random.random()))
            delay = min(delay * 2.0, 30.0)

        owner.stats.failures += 1
        raise LLMError(f"gemini generate_content failed after retries: {last_exc}") from last_exc


class ResilientClient:
    """Wraps a real ``genai.Client`` with cache, retry, limiting, accounting."""

    def __init__(self, inner: Any, cfg: LLMConfig, cache_dir: Path | None = None) -> None:
        self.inner = inner
        self.cfg = cfg
        self.backend = cfg.backend
        self.stats = CallStats()
        self.models = _ResilientModels(self)
        self.semaphore = threading.Semaphore(max(1, cfg.max_concurrency))
        self.limiter = _RateLimiter(_requests_per_minute(cfg))
        self.cache_dir = cache_dir or (CACHE_DIR / cfg.model.replace("/", "_"))
        if cfg.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- cache -------------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def read_cache(self, key: str) -> str | None:
        if not self.cfg.cache_enabled:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        except (OSError, ValueError, KeyError):
            return None

    def write_cache(self, key: str, text: str) -> None:
        if not self.cfg.cache_enabled:
            return
        tmp = self._cache_path(key).with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._cache_path(key))
        except OSError:  # pragma: no cover - cache is best-effort
            logger.debug("could not write llm cache entry %s", key)

    # -- upstream ----------------------------------------------------------
    def call_upstream(self, *, model: str, contents: Any, config: Any) -> Any:
        """Call the SDK, tolerating both response-schema field spellings.

        ``response_json_schema`` is the newer field; older SDK builds only accept
        ``response_schema``.  Pinning either one would make the repo fragile
        across the four laptops this runs on, so we translate on failure.
        """

        try:
            return self.inner.models.generate_content(
                model=model, contents=contents, config=config
            )
        except TypeError as exc:
            alt = _swap_schema_field(config)
            if alt is None:
                raise
            logger.debug("retrying with alternate schema field: %s", exc)
            return self.inner.models.generate_content(model=model, contents=contents, config=alt)
        except Exception as exc:  # noqa: BLE001
            if "response_json_schema" in str(exc) or "response_schema" in str(exc):
                alt = _swap_schema_field(config)
                if alt is not None:
                    return self.inner.models.generate_content(
                        model=model, contents=contents, config=alt
                    )
            raise

    def record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        self.stats.prompt_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
        self.stats.output_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)


def _swap_schema_field(config: Any) -> dict[str, Any] | None:
    if not isinstance(config, Mapping):
        return None
    updated = dict(config)
    if "response_json_schema" in updated:
        updated["response_schema"] = updated.pop("response_json_schema")
        return updated
    if "response_schema" in updated:
        updated["response_json_schema"] = updated.pop("response_schema")
        return updated
    return None


def _requests_per_minute(cfg: LLMConfig) -> int:  # pylint: disable=unused-argument
    """Return the operator-set throttle; actual quota depends on the account.

    ``cfg`` stays in the signature for call-site symmetry with the other
    per-config helpers, but the throttle is an env knob, not a config field.
    """

    from .config import env_int

    return env_int("FIRECODE_LLM_RPM", 60)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_client(cfg: LLMConfig | None = None, *, strict: bool = False) -> Any:
    """Return a client exposing ``.models.generate_content``.

    Never raises for missing credentials: an unconfigured environment falls back
    to the offline stub with a warning, so a teammate without a key can still
    run every stage end to end.
    """

    cfg = cfg or LLMConfig.load()

    if cfg.is_offline:
        if strict:
            raise LLMError(
                "live LLM inference was requested, but no Google credentials are configured"
            )
        logger.info("LLM backend: offline (deterministic stub, no network)")
        return OfflineClient()

    try:
        from google import genai  # type: ignore[import-untyped]
    except ImportError:
        if strict:
            raise LLMError("google-genai is required for live LLM inference") from None
        logger.warning("google-genai is not installed; falling back to the offline stub")
        return OfflineClient()

    try:
        from google.genai import types  # type: ignore[import-untyped]

        http_options = types.HttpOptions(api_version=cfg.api_version)
        if cfg.backend == "vertex":
            kwargs: dict[str, Any] = {
                "vertexai": True,
                "project": cfg.project,
                "location": cfg.location or "global",
                "http_options": http_options,
            }
            if cfg.api_key:
                # Vertex express mode: an API key instead of ADC.
                kwargs["api_key"] = cfg.api_key
            inner = genai.Client(**kwargs)
        elif cfg.backend == "developer_api":
            inner = genai.Client(api_key=cfg.api_key, http_options=http_options)
        else:
            raise ValueError(f"unknown LLM backend {cfg.backend!r}")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        if strict:
            raise LLMError(f"could not build configured Gemini client: {exc}") from exc
        logger.warning("could not build Gemini client (%s); using offline stub", exc)
        return OfflineClient()

    logger.info("LLM backend: %s, model: %s", cfg.backend, cfg.model)
    return ResilientClient(inner, cfg)


def describe_backend(cfg: LLMConfig | None = None) -> dict[str, Any]:
    """Human-readable status, printed by ``python -m src.run doctor``."""

    cfg = cfg or LLMConfig.load()
    try:
        sdk_version = importlib_metadata.version("google-genai")
    except importlib_metadata.PackageNotFoundError:
        sdk_version = None
    return {
        "backend": cfg.backend,
        "model": cfg.model,
        "project_configured": bool(cfg.project),
        "location": cfg.location,
        "api_version": cfg.api_version,
        "sdk": "google-genai",
        "sdk_version": sdk_version,
        "api_key_present": bool(cfg.api_key),
        "cache_enabled": cfg.cache_enabled,
        "self_consistency_samples": cfg.self_consistency_samples,
        "max_concurrency": cfg.max_concurrency,
    }


# ---------------------------------------------------------------------------
# Self-consistency
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ConsistencyResult:
    """Outcome of N independent samples of the same prompt."""

    consensus: dict[str, Any] | None
    agreement: float
    samples: list[dict[str, Any]] = field(default_factory=list)
    failures: int = 0

    @property
    def n_valid(self) -> int:
        return len(self.samples)


def self_consistency(
    call: Any,
    *,
    samples: int,
    signature: Any,
    temperature: float = 0.7,
) -> ConsistencyResult:
    """Run ``call(temperature, index)`` N times and measure agreement.

    Agreement -- how often independent samples land on the same answer -- is the
    uncertainty signal we trust.  A model's *stated* confidence is not used
    anywhere in this pipeline, because verbalized confidence is known to be
    systematically overconfident.

    ``signature`` maps a sample to the hashable identity that must match for two
    samples to count as agreeing (e.g. the verdict, or the need's key phrase).
    """

    results: list[dict[str, Any]] = []
    failures = 0
    for index in range(max(1, samples)):
        try:
            value = call(temperature, index)
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            failures += 1
            logger.debug("self-consistency sample %d failed: %s", index, exc)
            continue
        if isinstance(value, Mapping):
            results.append(dict(value))

    if not results:
        return ConsistencyResult(consensus=None, agreement=0.0, samples=[], failures=failures)

    counts = Counter(str(signature(item)) for item in results)
    winner, votes = counts.most_common(1)[0]
    consensus = next(item for item in results if str(signature(item)) == winner)
    # Denominator is the requested sample count, so failures reduce confidence
    # rather than being silently forgiven.
    agreement = votes / max(1, samples)
    return ConsistencyResult(
        consensus=consensus,
        agreement=agreement,
        samples=results,
        failures=failures,
    )
