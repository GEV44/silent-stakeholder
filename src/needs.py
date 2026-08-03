"""Stage 1: deterministic latent-need inference plus an optional Gemini lift.

The offline path is intentionally modest but complete: stable clustering,
lexicon/rating sentiment, an explicit JTBD reframe, Kano classification, and
ODI opportunity scoring.  A structured LLM extractor can replace only the
second-order reframe; cluster membership and citation validation remain code.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np

try:  # Support both ``python -m src.needs`` and direct module imports.
    from .embedding import Embedder, HashingEmbedder, cosine_similarity_matrix
except ImportError:  # pragma: no cover - direct script compatibility
    from embedding import (  # type: ignore[import-not-found,no-redef]
        Embedder,
        HashingEmbedder,
        cosine_similarity_matrix,
    )


_TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)?", re.UNICODE)
_NEGATIVE = {
    "bad",
    "broken",
    "bug",
    "buggy",
    "cannot",
    "can't",
    "crash",
    "crashes",
    "difficult",
    "error",
    "fail",
    "fails",
    "failed",
    "hate",
    "impossible",
    "missing",
    "never",
    "not",
    "problem",
    "slow",
    "terrible",
    "unusable",
    "worse",
    "worst",
}
_POSITIVE = {
    "amazing",
    "excellent",
    "fast",
    "good",
    "great",
    "helpful",
    "love",
    "perfect",
    "reliable",
    "smooth",
    "useful",
    "works",
}
_INTENSIFIERS = {"absolutely", "always", "extremely", "really", "totally", "very"}
# Friction markers rescue a high-rated review that still reports a struggle.
# "Great app but uploads fail" is a 5-star review carrying a real need; dropping
# it on rating alone would discard some of the most articulate evidence we have.
# Contrast and desiderative words carry that signal more reliably than polarity,
# so these are kept separate from ``_NEGATIVE`` rather than folded into it.
_FRICTION_MARKERS = {
    "annoying",
    "although",
    "but",
    "confusing",
    "except",
    "hard",
    "however",
    "hoping",
    "issue",
    "issues",
    "lack",
    "lacks",
    "unless",
    "unfortunately",
    "wish",
    "yet",
}
_FRICTION_PHRASES = re.compile(
    r"\b(?:"
    r"would (?:be )?(?:nice|love|prefer|like)"
    r"|(?:it|that)'?s a shame"
    r"|only (?:complaint|downside|issue|problem)"
    r"|needs? (?:to|a|an|more|better)"
    r"|should (?:be|have|support|allow)"
    r"|no way to"
    r"|can'?t (?:seem to|find|get|figure)"
    r"|if only"
    r"|other than that"
    r"|the only thing"
    r")\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "am",
    "an",
    "and",
    "app",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "get",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "just",
    "me",
    "my",
    "need",
    "no",
    "not",
    "of",
    "on",
    "or",
    "please",
    "so",
    "that",
    "the",
    "their",
    "this",
    "to",
    "too",
    "update",
    "use",
    "very",
    "want",
    "was",
    "we",
    "when",
    "will",
    "with",
    "work",
    "works",
    "would",
    "you",
}


class StructuredNeedExtractor(Protocol):
    def extract(
        self,
        *,
        cluster_texts: Sequence[str],
        allowed_signal_ids: Sequence[str],
        aspects: Sequence[str],
        temperature: float,
    ) -> Mapping[str, Any]:
        """Return a schema-like latent need grounded in allowed IDs."""


@dataclass(slots=True)
class NeedConfig:
    cluster_similarity: float = 0.26
    min_cluster_size: int = 2
    include_singletons: bool = False
    max_signals_per_cluster: int = 40
    max_needs: int | None = None
    llm_samples: int = 3
    llm_temperature: float = 0.7
    random_seed: int = 42
    # Defaults live here rather than in ``config/pipeline.json`` only because that
    # file is outside this change's lane; see
    # docs/contracts/REQ-main-2-friction-and-merge-thresholds.md for the keys to
    # promote once the lead applies the contract.
    friction_filter: bool = True
    praise_rating_floor: int = 4
    merge_similar_needs: bool = True
    merge_similarity: float = 0.9
    drop_unnameable_clusters: bool = True

    def __post_init__(self) -> None:
        if not -1.0 <= self.cluster_similarity <= 1.0:
            raise ValueError("cluster_similarity must be in [-1, 1]")
        if self.min_cluster_size < 1:
            raise ValueError("min_cluster_size must be positive")
        if self.max_signals_per_cluster < 1:
            raise ValueError("max_signals_per_cluster must be positive")
        if self.llm_samples < 1:
            raise ValueError("llm_samples must be positive")
        if not 1 <= self.praise_rating_floor <= 5:
            raise ValueError("praise_rating_floor must be in [1, 5]")
        if not -1.0 <= self.merge_similarity <= 1.0:
            raise ValueError("merge_similarity must be in [-1, 1]")


class GeminiNeedExtractor:
    """Thin adapter for native Gemini JSON-schema output.

    The client is injected, which keeps credentials, networking, retries, and
    SDK version choices outside the deterministic core.
    """

    def __init__(self, client: Any, model: str, *, max_output_tokens: int):
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens

    def extract(
        self,
        *,
        cluster_texts: Sequence[str],
        allowed_signal_ids: Sequence[str],
        aspects: Sequence[str],
        temperature: float,
    ) -> Mapping[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "latent_need": {"type": "string"},
                "jtbd_statement": {"type": "string"},
                "kano_class": {
                    "type": "string",
                    "enum": ["basic", "performance", "excitement"],
                },
                "root_cause_hypothesis": {"type": "string"},
                "symptom": {"type": "string"},
                "supporting_signal_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(allowed_signal_ids)},
                },
            },
            "required": [
                "latent_need",
                "jtbd_statement",
                "kano_class",
                "root_cause_hypothesis",
                "symptom",
                "supporting_signal_ids",
            ],
            "additionalProperties": False,
        }
        evidence = "\n".join(
            f"{signal_id}: {text}"
            for signal_id, text in zip(
                allowed_signal_ids, cluster_texts, strict=True
            )
        )
        prompt = (
            "Infer one latent user need from this evidence. Reframe the surface "
            "complaint as a job-to-be-done. Cite only the supplied IDs, avoid "
            "inventing product facts, and make the root cause explicitly a "
            "hypothesis.\n"
            f"Candidate aspects: {', '.join(aspects)}\nEvidence:\n{evidence}"
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": self.max_output_tokens,
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )
        return json.loads(response.text)


class StructuredNeedMerger(Protocol):
    def propose_groups(
        self,
        *,
        needs: Sequence[Mapping[str, Any]],
        allowed_need_ids: Sequence[str],
        temperature: float,
    ) -> Mapping[str, Any]:
        """Return groupings of redundant need IDs. Evidence is never returned."""


class GeminiNeedMerger:
    """Proposes which needs are redundant. It never carries the evidence itself.

    The model sees need *titles* and returns only need IDs; the union of
    ``supporting_signal_ids`` is computed in code afterwards. That is deliberate:
    asking a model to "combine the evidence IDs so no evidence is lost" makes
    evidence integrity a matter of model compliance, when it can instead be a
    property of the merge function. A dropped ID then becomes impossible rather
    than unlikely.
    """

    def __init__(self, client: Any, model: str, *, max_output_tokens: int):
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens

    def propose_groups(
        self,
        *,
        needs: Sequence[Mapping[str, Any]],
        allowed_need_ids: Sequence[str],
        temperature: float,
    ) -> Mapping[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "canonical_need_id": {
                                "type": "string",
                                "enum": list(allowed_need_ids),
                            },
                            "member_need_ids": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": list(allowed_need_ids),
                                },
                            },
                        },
                        "required": ["canonical_need_id", "member_need_ids"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["groups"],
            "additionalProperties": False,
        }
        catalog = "\n".join(
            f"{need['id']}: {need.get('latent_need', '')} — {need.get('jtbd_statement', '')}"
            for need in needs
        )
        prompt = (
            "Review this list of unmet user needs. Identify duplicates or highly "
            "overlapping needs and group them so each group becomes one stronger "
            "need. Two needs belong together only if they describe the same "
            "underlying job; a shared topic is not enough. Leave a need out of "
            "every group if it is distinct.\n"
            "Return only need IDs. Supporting evidence is combined automatically "
            "and must not be listed.\n"
            f"Needs:\n{catalog}"
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": self.max_output_tokens,
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )
        return json.loads(response.text)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"expected mapping or model, got {type(value).__name__}")


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text or "")]


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "\x1e".join(parts).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()[:12]


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def analyze_sentiment(signal: Mapping[str, Any] | Any) -> dict[str, float | str]:
    """Return deterministic aspect sentiment in [-1,1] and intensity in [0,1]."""

    row = _mapping(signal)
    text = str(row.get("text") or "")
    tokens = _tokens(text)
    negative = sum(token in _NEGATIVE for token in tokens)
    positive = sum(token in _POSITIVE for token in tokens)
    intensifiers = sum(token in _INTENSIFIERS for token in tokens)

    lexical_count = negative + positive
    lexical = (positive - negative) / lexical_count if lexical_count else 0.0
    rating = row.get("rating")
    rating_sentiment: float | None = None
    try:
        if rating is not None:
            rating_sentiment = max(-1.0, min(1.0, (float(rating) - 3.0) / 2.0))
    except (TypeError, ValueError):
        rating_sentiment = None

    if lexical_count and rating_sentiment is not None:
        sentiment = 0.7 * lexical + 0.3 * rating_sentiment
    elif lexical_count:
        sentiment = lexical
    elif rating_sentiment is not None:
        sentiment = rating_sentiment
    else:
        sentiment = 0.0
    intensity = min(
        1.0,
        abs(sentiment) * 0.65
        + min(1.0, lexical_count / 3.0) * 0.25
        + min(1.0, intensifiers / 2.0) * 0.1,
    )

    candidates = [
        token
        for token in tokens
        if token not in _STOPWORDS
        and token not in _NEGATIVE
        and token not in _POSITIVE
        and token not in _INTENSIFIERS
        and len(token) > 2
    ]
    aspect = Counter(candidates).most_common(1)[0][0] if candidates else "product use"
    return {
        "aspect": aspect,
        "sentiment": round(max(-1.0, min(1.0, sentiment)), 6),
        "intensity": round(intensity, 6),
    }


def cluster_signals(
    signals: Sequence[Mapping[str, Any] | Any],
    *,
    embedder: Embedder | None = None,
    similarity_threshold: float = 0.26,
    random_seed: int = 42,
) -> list[list[dict[str, Any]]]:
    """Cluster signals with a deterministic large-corpus baseline.

    Large corpora use MiniBatchKMeans only as a credential-free fallback. The
    production path should use the configured density-based topic pipeline and
    report its outlier/stability metrics.
    """

    rows = sorted((_mapping(signal) for signal in signals), key=lambda item: str(item["id"]))
    if not rows:
        return []
    texts = [str(row.get("text") or "") for row in rows]
    vectors = (embedder or HashingEmbedder()).encode(texts)
    if len(rows) >= 100:
        try:
            from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-untyped]
        except ImportError:
            pass
        else:
            cluster_count = max(8, min(80, round(math.sqrt(len(rows) / 2))))
            model = MiniBatchKMeans(
                n_clusters=cluster_count,
                random_state=random_seed,
                batch_size=min(1024, len(rows)),
                n_init=5,
            )
            labels = model.fit_predict(vectors)
            members: dict[int, list[int]] = {}
            for index, label in enumerate(labels):
                members.setdefault(int(label), []).append(index)
            ordered = sorted(members.values(), key=lambda group: str(rows[group[0]]["id"]))
            return [[rows[index] for index in group] for group in ordered]

    clusters: list[list[int]] = []
    centroids: list[np.ndarray] = []

    for index, vector in enumerate(vectors):
        if not clusters:
            clusters.append([index])
            centroids.append(vector.copy())
            continue
        similarities = cosine_similarity_matrix(vector, np.vstack(centroids))[0]
        best = int(np.argmax(similarities))
        if float(similarities[best]) >= similarity_threshold:
            clusters[best].append(index)
            centroid = np.mean(vectors[clusters[best]], axis=0)
            norm = float(np.linalg.norm(centroid))
            centroids[best] = centroid / norm if norm else centroid
        else:
            clusters.append([index])
            centroids.append(vector.copy())
    return [[rows[index] for index in members] for members in clusters]


def _cluster_aspects(cluster: Sequence[dict[str, Any]], limit: int = 3) -> list[str]:
    term_counts: Counter[str] = Counter()
    document_counts: Counter[str] = Counter()
    for signal in cluster:
        tokens = {
            token
            for token in _tokens(str(signal.get("text") or ""))
            if token not in _STOPWORDS
            and token not in _NEGATIVE
            and token not in _POSITIVE
            and token not in _INTENSIFIERS
            and len(token) > 2
        }
        document_counts.update(tokens)
        term_counts.update(
            token
            for token in _tokens(str(signal.get("text") or ""))
            if token in tokens
        )
    count = max(1, len(cluster))
    scored = [
        (
            term_counts[token] * (1.0 + math.log((1.0 + count) / (1.0 + docs))),
            token,
        )
        for token, docs in document_counts.items()
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [token for _, token in scored[:limit]] or ["product use"]


def opportunity_score(importance: float, satisfaction: float) -> float:
    """Review-derived priority proxy using the ODI-shaped 0–20 formula.

    The inputs are inferred from reviews, not measured by an ODI customer
    survey, so downstream artifacts must not describe this as validated ODI.
    """

    importance = max(0.0, min(10.0, float(importance)))
    satisfaction = max(0.0, min(10.0, float(satisfaction)))
    return importance + max(importance - satisfaction, 0.0)


_OFFLINE_DOMAINS: dict[str, tuple[set[str], str, str, str]] = {
    "media": (
        {"upload", "uploads", "photo", "photos", "image", "images", "picture", "gallery", "media"},
        "Recoverable media uploads",
        "upload media in the background and recover individual failures without restarting",
        "Media uploads fail, stall, or block the publishing workflow.",
    ),
    "access": (
        {"login", "log", "password", "account", "username", "access", "signin", "hosted"},
        "Dependable account and self-hosted access",
        "open every authorized site without repeated authentication failures",
        "Users cannot reliably sign in or connect self-hosted sites.",
    ),
    "drafting": (
        {
            "draft",
            "drafts",
            "edit",
            "edits",
            "editor",
            "post",
            "posts",
            "page",
            "pages",
            "sync",
            "save",
            "saved",
            "publish",
        },
        "Lossless drafting and publishing",
        "preserve edits across drafting, syncing, and publishing transitions",
        "Draft or page changes fail to save, sync, or publish reliably.",
    ),
    "engagement": (
        {"notification", "notifications", "comment", "comments", "reply", "replies"},
        "Actionable engagement notifications",
        "review and respond to audience activity without losing context",
        "Comment and notification workflows are incomplete or hard to act on.",
    ),
    "sites": (
        {"site", "sites", "blog", "blogs", "dashboard", "stats", "multiple"},
        "Unambiguous multi-site management",
        "identify the correct site and keep dashboard analytics scoped to it",
        "Site selection and management are unreliable or ambiguous.",
    ),
    "performance": (
        {"crash", "crashes", "slow", "freeze", "freezes", "loading", "open", "opening"},
        "Responsive, crash-free mobile work",
        "complete time-sensitive work without waiting for or restarting the app",
        "The app crashes, freezes, loads slowly, or fails to open.",
    ),
    "customization": (
        {
            "theme",
            "themes",
            "custom",
            "customize",
            "design",
            "html",
            "format",
            "formatting",
            "font",
            "layout",
        },
        "Flexible mobile formatting and customization",
        "shape content and site presentation without leaving the mobile workflow",
        "Mobile editing lacks needed formatting, theme, or layout controls.",
    ),
    "analytics": (
        {"stats", "statistics", "analytics", "views", "traffic", "visitors"},
        "Trustworthy mobile performance insights",
        "understand site activity and content performance from the mobile dashboard",
        "Site statistics are missing, unclear, or unreliable on mobile.",
    ),
}


_DOMAIN_KEYWORDS: frozenset[str] = frozenset(
    keyword for keywords, *_ in _OFFLINE_DOMAINS.values() for keyword in keywords
)
# Tokens that survive stopword filtering but cannot name a need. Ranking them
# by frequency is what produced titles like "Reliable its" and "Reliable more"
# on the real corpus: judgements, quantifiers and fillers are common precisely
# because they carry no product content.
_WEAK_ASPECTS: frozenset[str] = frozenset(
    {
        "its", "it's", "some", "any", "such", "other", "another", "same",
        "more", "most", "less", "least", "much", "many", "few", "lot", "lots",
        "better", "best", "worse", "worst", "nice", "good", "great", "bad",
        "awesome", "cool", "fine", "okay", "sure", "well", "easy", "hard",
        "thing", "things", "stuff", "way", "ways", "time", "times", "bit",
        "kind", "sort", "part", "everything", "something", "nothing",
        "working", "using", "used", "usefull", "useful", "improve",
        "improvement", "improved", "new", "old", "now", "still", "even",
        "back", "first", "last", "next", "one", "two", "day", "days",
    }
)


def _informative_token_count(text: str) -> int:
    """Content tokens left after stripping stopwords and pure sentiment words."""

    return sum(
        1
        for token in _tokens(text)
        if token not in _STOPWORDS
        and token not in _WEAK_ASPECTS
        and token not in _POSITIVE
        and token not in _NEGATIVE
        and len(token) > 2
    )


def _match_offline_domain(
    cluster: Sequence[dict[str, Any]],
) -> tuple[str, str, str] | None:
    """Return a domain frame, or None when no domain is clearly indicated."""

    tokens = Counter(
        token
        for signal in cluster
        for token in _tokens(str(signal.get("text") or ""))
    )
    candidates: list[tuple[int, str, str, str, str]] = []
    for domain, (keywords, title, job, symptom) in _OFFLINE_DOMAINS.items():
        score = sum(tokens[keyword] for keyword in keywords)
        candidates.append((score, domain, title, job, symptom))
    score, _, title, job, symptom = max(candidates, key=lambda item: (item[0], item[1]))
    if score >= max(2, len(cluster) // 8):
        return title, job, symptom
    return None


def is_unnameable_cluster(
    cluster: Sequence[dict[str, Any]], aspects: Sequence[str]
) -> bool:
    """True when the offline labeller cannot name this cluster in product terms.

    On the real corpus these are the residue: one-word verdicts ("Bs", "Best
    Best"), stray names, and unfocused praise that groups together precisely
    because none of it says anything specific.

    The test is deliberately not a token-count threshold. Both residue clusters
    in the WordPress run scored 0.55 on informative-token share, so any cutoff
    separating them would have been fitted to those two clusters rather than
    derived. The honest criterion is vocabulary coverage: the offline labeller
    can only name the eight domains in ``_OFFLINE_DOMAINS``, so a cluster that
    touches none of them — not in aggregate, not in its top aspects — cannot be
    named, and inventing a title for it asserts a need the evidence never
    supports. Same rule the evidence contract applies to gaps: no evidence, no
    need. Dropped counts surface in need metadata, so this is auditable rather
    than silent.
    """

    if _match_offline_domain(cluster) is not None:
        return False
    return not any(
        aspect not in _WEAK_ASPECTS and aspect in _DOMAIN_KEYWORDS for aspect in aspects
    )


def _offline_need_frame(
    cluster: Sequence[dict[str, Any]], aspects: Sequence[str]
) -> tuple[str, str, str]:
    matched = _match_offline_domain(cluster)
    if matched is not None:
        return matched
    # Prefer a real product noun over the most frequent token; frequency is what
    # promoted "its" and "more" to titles.
    # Only a recognised product noun may name a need. Falling back to "the most
    # frequent remaining token" is what produced "Reliable its", "Reliable more"
    # and "Reliable kadang" — a length check cannot tell a product noun from an
    # Indonesian adverb. Two unnameable clusters both landing on the generic
    # title is fine: they carry identical text and the merge step folds them.
    primary = next(
        (
            aspect.replace("_", " ")
            for aspect in aspects
            if aspect in _DOMAIN_KEYWORDS and aspect not in _WEAK_ASPECTS
        ),
        "product workflow",
    )
    return (
        f"Reliable {primary}",
        f"complete {primary} reliably so they can finish their work without interruption",
        f"Users report problems with {primary}.",
    )


def _grounded_offline_signal_ids(
    cluster: Sequence[dict[str, Any]], title: str
) -> list[str]:
    """Keep only signals carrying vocabulary for the deterministic need frame.

    Embedding clusters are candidate neighborhoods, not evidence labels. A
    connected component can contain a bridge signal from another product area;
    exact quote verification would prove that quote exists, but not that it
    supports the inferred need. The deterministic baseline can make the
    stronger support decision because its domain vocabulary is explicit.
    """

    keywords = next(
        (
            domain_keywords
            for domain_keywords, domain_title, _job, _symptom in _OFFLINE_DOMAINS.values()
            if domain_title == title
        ),
        None,
    )
    if keywords is None:
        return [str(signal["id"]) for signal in cluster]
    grounded = [
        str(signal["id"])
        for signal in cluster
        if set(_tokens(str(signal.get("text") or ""))) & keywords
    ]
    return grounded or [str(signal["id"]) for signal in cluster]


def _offline_extract(
    cluster: Sequence[dict[str, Any]],
    aspects: Sequence[str],
    analyses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    title, job, symptom = _offline_need_frame(cluster, aspects)
    primary = aspects[0].replace("_", " ")
    mean_sentiment = float(np.mean([float(row["sentiment"]) for row in analyses]))
    mean_intensity = float(np.mean([float(row["intensity"]) for row in analyses]))
    negative_share = float(
        np.mean([1.0 if float(row["sentiment"]) < -0.05 else 0.0 for row in analyses])
    )
    if negative_share >= 0.6 or mean_intensity >= 0.65:
        kano = "basic"
    elif negative_share >= 0.25:
        kano = "performance"
    else:
        kano = "excitement"
    return {
        "latent_need": title,
        "jtbd_statement": (
            f"When using the product, users need to {job}."
        ),
        "kano_class": kano,
        "root_cause_hypothesis": (
            f"Repeated evidence around {primary} suggests an unreliable or "
            "incomplete workflow; product telemetry is needed to confirm the cause."
        ),
        "symptom": symptom,
        "supporting_signal_ids": [str(signal["id"]) for signal in cluster],
        "_mean_sentiment": mean_sentiment,
        "_mean_intensity": mean_intensity,
        "_negative_share": negative_share,
    }


def _validate_extraction(
    result: Mapping[str, Any], allowed_ids: Sequence[str]
) -> dict[str, Any]:
    required = {
        "latent_need",
        "jtbd_statement",
        "kano_class",
        "root_cause_hypothesis",
        "symptom",
        "supporting_signal_ids",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"need extraction missing fields: {', '.join(missing)}")
    allowed = set(allowed_ids)
    cited = list(dict.fromkeys(map(str, result["supporting_signal_ids"])))
    if not cited:
        raise ValueError("need extraction cited no supporting signals")
    unknown = sorted(set(cited).difference(allowed))
    if unknown:
        raise ValueError(f"need extraction cited unknown IDs: {', '.join(unknown)}")
    kano = str(result["kano_class"]).casefold()
    if kano not in {"basic", "performance", "excitement"}:
        raise ValueError(f"invalid Kano class: {kano!r}")
    cleaned = dict(result)
    cleaned["kano_class"] = kano
    cleaned["supporting_signal_ids"] = cited
    for key in required.difference({"supporting_signal_ids"}):
        cleaned[key] = str(cleaned[key]).strip()
        if not cleaned[key]:
            raise ValueError(f"need extraction field {key!r} is empty")
    return cleaned


def _signature(result: Mapping[str, Any]) -> str:
    text = f"{result.get('latent_need', '')} {result.get('jtbd_statement', '')}"
    return " ".join(_tokens(text))


def _select_consensus(results: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    signatures = [_signature(result) for result in results]
    counts = Counter(signatures)
    best_signature, best_count = sorted(
        counts.items(), key=lambda pair: (-pair[1], pair[0])
    )[0]
    chosen = next(
        result
        for result, signature in zip(results, signatures, strict=True)
        if signature == best_signature
    )
    return chosen, best_count / len(results)


def has_friction_language(text: str) -> bool:
    """True when the text reports a struggle, not only a verdict."""

    tokens = set(_tokens(text))
    if tokens & _FRICTION_MARKERS or tokens & _NEGATIVE:
        return True
    return bool(_FRICTION_PHRASES.search(text or ""))


def is_friction_signal(
    signal: Mapping[str, Any] | Any, *, praise_rating_floor: int = 4
) -> bool:
    """Keep a signal only if it can plausibly evidence an unmet need.

    A high rating alone is not proof of satisfaction — it is proof of a verdict.
    Reviews at or above the praise floor survive only when they also carry
    friction language. Anything unrated is kept: absence of a rating is not
    evidence of praise, and silently discarding it would bias the corpus.
    """

    row = _mapping(signal)
    rating = row.get("rating")
    try:
        value = float(rating) if rating is not None else None
    except (TypeError, ValueError):
        value = None
    if value is None or value < praise_rating_floor:
        return True
    return has_friction_language(str(row.get("text") or ""))


def partition_friction_signals(
    signals: Sequence[Mapping[str, Any] | Any], *, praise_rating_floor: int = 4
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split signals into (friction-bearing, praise-only), preserving order."""

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for signal in signals:
        row = _mapping(signal)
        target = (
            kept
            if is_friction_signal(row, praise_rating_floor=praise_rating_floor)
            else dropped
        )
        target.append(row)
    return kept, dropped


def _score_need(
    analyses: Sequence[Mapping[str, Any]], support_count: int, total: int
) -> dict[str, float]:
    """Derive the ODI-shaped scores. Shared so merged needs cannot drift."""

    mean_sentiment = float(np.mean([float(row["sentiment"]) for row in analyses]))
    mean_intensity = float(np.mean([float(row["intensity"]) for row in analyses]))
    negative_share = float(
        np.mean([1.0 if float(row["sentiment"]) < -0.05 else 0.0 for row in analyses])
    )
    importance = min(
        10.0,
        3.5
        + 4.0 * (math.log1p(support_count) / math.log1p(max(1, total)))
        + 2.5 * mean_intensity,
    )
    satisfaction = max(0.0, min(10.0, 5.0 * (mean_sentiment + 1.0)))
    return {
        "importance": importance,
        "satisfaction": satisfaction,
        "opportunity_score": opportunity_score(importance, satisfaction),
        "mean_sentiment": mean_sentiment,
        "mean_intensity": mean_intensity,
        "negative_share": negative_share,
    }


def _attach_unassigned_domain_signals(
    needs: list[dict[str, Any]],
    signals: Sequence[dict[str, Any]],
    *,
    total: int,
) -> None:
    """Attach domain-grounded stragglers to one unambiguous offline need.

    Graph clustering can leave a small, coherent fragment below the cluster-size
    floor. If that fragment carries the exact deterministic domain vocabulary
    of one already-emitted offline need, dropping it would understate support.
    Ambiguous titles, model-derived needs, and unrecognized vocabulary remain
    untouched and therefore visible in the manifest's unassigned-signal count.
    """

    targets: dict[str, list[dict[str, Any]]] = {}
    for need in needs:
        metadata = _mapping(need.get("metadata") or {})
        if metadata.get("inference") == "offline":
            targets.setdefault(str(need.get("latent_need") or ""), []).append(need)

    assigned = {
        str(signal_id)
        for need in needs
        for signal_id in need.get("supporting_signal_ids", [])
    }
    attached: dict[int, list[str]] = {}
    for signal in signals:
        signal_id = str(signal["id"])
        if signal_id in assigned:
            continue
        matched = _match_offline_domain([signal])
        if matched is None:
            continue
        candidates = targets.get(matched[0], [])
        if len(candidates) != 1:
            continue
        attached.setdefault(id(candidates[0]), []).append(signal_id)

    signal_by_id = {str(signal["id"]): signal for signal in signals}
    for need in needs:
        extra_ids = attached.get(id(need), [])
        if not extra_ids:
            continue
        support_ids = sorted(
            {
                *(str(value) for value in need.get("supporting_signal_ids", [])),
                *extra_ids,
            }
        )
        analyses = [analyze_sentiment(signal_by_id[value]) for value in support_ids]
        scores = _score_need(analyses, len(support_ids), total)
        timestamps = [
            parsed
            for parsed in (
                _parse_timestamp(signal_by_id[value].get("timestamp"))
                for value in support_ids
            )
            if parsed is not None
        ]
        metadata = _mapping(need.get("metadata") or {})
        need["supporting_signal_ids"] = support_ids
        need["cluster_id"] = _stable_id("C", *support_ids)
        need["id"] = _stable_id(
            "N", str(need.get("latent_need") or "").casefold(), *support_ids
        )
        need["importance"] = round(scores["importance"], 6)
        need["satisfaction"] = round(scores["satisfaction"], 6)
        need["opportunity_score"] = round(scores["opportunity_score"], 6)
        metadata["support_count"] = len(support_ids)
        metadata["mean_sentiment"] = round(scores["mean_sentiment"], 6)
        metadata["mean_sentiment_intensity"] = round(scores["mean_intensity"], 6)
        metadata["negative_share"] = round(scores["negative_share"], 6)
        metadata["domain_signals_attached"] = len(extra_ids)
        metadata["temporal_scope"] = {
            "evidence_start": min(timestamps).isoformat() if timestamps else None,
            "evidence_end": max(timestamps).isoformat() if timestamps else None,
        }
        need["metadata"] = metadata


def _need_text(need: Mapping[str, Any]) -> str:
    return " ".join(
        str(need.get(key) or "")
        for key in ("latent_need", "jtbd_statement", "symptom")
    )


def _validate_merge_groups(
    result: Mapping[str, Any], allowed_need_ids: Sequence[str]
) -> list[list[str]]:
    """Re-check the model's groupings in code (evidence-contract defense 2)."""

    if not isinstance(result, Mapping) or "groups" not in result:
        raise ValueError("merge proposal missing 'groups'")
    allowed = set(allowed_need_ids)
    groups: list[list[str]] = []
    seen: set[str] = set()
    for group in result["groups"]:
        members = list(dict.fromkeys(map(str, _mapping(group).get("member_need_ids", []))))
        unknown = sorted(set(members).difference(allowed))
        if unknown:
            raise ValueError(f"merge proposal cited unknown need IDs: {', '.join(unknown)}")
        if len(members) < 2:
            continue
        repeated = sorted(seen.intersection(members))
        if repeated:
            raise ValueError(f"merge proposal reused need IDs: {', '.join(repeated)}")
        seen.update(members)
        groups.append(members)
    return groups


def _merge_partition(
    need_ids: Sequence[str], pairs: Sequence[tuple[str, str]]
) -> list[list[str]]:
    """Partition needs by transitive closure of the redundancy pairs.

    Union-find rather than greedy sweeping: the resulting partition depends only
    on the set of pairs, not on the order they were discovered, which is what
    keeps the merge deterministic across runs.
    """

    parent = {need_id: need_id for need_id in need_ids}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in pairs:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            first, second = sorted((root_left, root_right))
            parent[second] = first

    grouped: dict[str, list[str]] = {}
    for need_id in need_ids:
        grouped.setdefault(find(need_id), []).append(need_id)
    return list(grouped.values())


def _merge_need_group(
    group: Sequence[Mapping[str, Any]],
    signal_by_id: Mapping[str, Mapping[str, Any]],
    total: int,
) -> dict[str, Any]:
    """Fold redundant needs into one, keeping every supporting signal ID."""

    ordered = sorted(
        group, key=lambda need: (-float(need.get("opportunity_score") or 0.0), str(need["id"]))
    )
    primary = ordered[0]
    union_ids = sorted({
        str(signal_id)
        for need in ordered
        for signal_id in need.get("supporting_signal_ids", [])
    })
    if not union_ids:
        raise ValueError("merged need would carry no evidence")

    analyses = [
        analyze_sentiment(signal_by_id[signal_id])
        for signal_id in union_ids
        if signal_id in signal_by_id
    ]
    scores = _score_need(analyses, len(union_ids), total) if analyses else None

    aspects: list[str] = []
    for need in ordered:
        for aspect in _mapping(need.get("metadata") or {}).get("aspects", []):
            if aspect not in aspects:
                aspects.append(aspect)

    timestamps = [
        parsed
        for parsed in (
            _parse_timestamp(signal_by_id[signal_id].get("timestamp"))
            for signal_id in union_ids
            if signal_id in signal_by_id
        )
        if parsed is not None
    ]

    merged = dict(primary)
    metadata = dict(_mapping(primary.get("metadata") or {}))
    merged["supporting_signal_ids"] = union_ids
    merged["cluster_id"] = _stable_id("C", *union_ids)
    merged["id"] = _stable_id("N", str(primary["latent_need"]).casefold(), *union_ids)
    if scores is not None:
        merged["importance"] = round(scores["importance"], 6)
        merged["satisfaction"] = round(scores["satisfaction"], 6)
        merged["opportunity_score"] = round(scores["opportunity_score"], 6)
        metadata["mean_sentiment"] = round(scores["mean_sentiment"], 6)
        metadata["mean_sentiment_intensity"] = round(scores["mean_intensity"], 6)
        metadata["negative_share"] = round(scores["negative_share"], 6)
    metadata["aspects"] = aspects[:6]
    metadata["support_count"] = len(union_ids)
    metadata["merged_from"] = [str(need["id"]) for need in ordered]
    metadata["merged_need_count"] = len(ordered)
    metadata["self_consistency"] = round(
        min(
            float(_mapping(need.get("metadata") or {}).get("self_consistency", 1.0))
            for need in ordered
        ),
        6,
    )
    metadata["llm_failures"] = sum(
        int(_mapping(need.get("metadata") or {}).get("llm_failures", 0)) for need in ordered
    )
    metadata["temporal_scope"] = {
        "evidence_start": min(timestamps).isoformat() if timestamps else None,
        "evidence_end": max(timestamps).isoformat() if timestamps else None,
    }
    merged["metadata"] = metadata
    return merged


def merge_redundant_needs(
    needs: Sequence[Mapping[str, Any]],
    *,
    signal_by_id: Mapping[str, Mapping[str, Any]],
    total: int,
    embedder: Embedder | None = None,
    similarity_threshold: float = 0.9,
    merger: StructuredNeedMerger | None = None,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """Collapse needs that describe the same job, losing no evidence IDs.

    The union of supporting signal IDs is computed here rather than requested
    from a model, so the "no evidence is lost" property holds by construction
    and is testable. An optional ``merger`` may only *propose* groupings.
    """

    rows = [dict(need) for need in needs]
    if len(rows) < 2:
        return rows
    need_ids = [str(row["id"]) for row in rows]

    vectors = (embedder or HashingEmbedder()).encode([_need_text(row) for row in rows])
    similarity = cosine_similarity_matrix(vectors, vectors)
    pairs: list[tuple[str, str]] = [
        (need_ids[left], need_ids[right])
        for left in range(len(rows))
        for right in range(left + 1, len(rows))
        if float(similarity[left, right]) >= similarity_threshold
    ]

    llm_groups = 0
    if merger is not None:
        try:
            proposal = merger.propose_groups(
                needs=[
                    {
                        "id": row["id"],
                        "latent_need": row.get("latent_need", ""),
                        "jtbd_statement": row.get("jtbd_statement", ""),
                    }
                    for row in rows
                ],
                allowed_need_ids=need_ids,
                temperature=temperature,
            )
            for members in _validate_merge_groups(proposal, need_ids):
                llm_groups += 1
                for member in members[1:]:
                    pairs.append((members[0], member))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError):
            llm_groups = 0

    by_id = {str(row["id"]): row for row in rows}
    merged: list[dict[str, Any]] = []
    for group in _merge_partition(need_ids, pairs):
        if len(group) == 1:
            merged.append(by_id[group[0]])
            continue
        merged.append(
            _merge_need_group([by_id[member] for member in group], signal_by_id, total)
        )
    merged.sort(key=lambda need: (-float(need["opportunity_score"]), str(need["id"])))
    return merged


def infer_needs(
    signals: Sequence[Mapping[str, Any] | Any],
    *,
    embedder: Embedder | None = None,
    extractor: StructuredNeedExtractor | None = None,
    merger: StructuredNeedMerger | None = None,
    config: NeedConfig | None = None,
) -> list[dict[str, Any]]:
    """Infer stable, ID-grounded latent needs.

    LLM failures never make the offline pipeline fail. A grounded deterministic
    frame is used when available; otherwise the unnameable cluster is dropped.
    """

    settings = config or NeedConfig()
    signal_rows = [_mapping(signal) for signal in signals]
    ids = [str(row.get("id") or "") for row in signal_rows]
    if any(not signal_id for signal_id in ids):
        raise ValueError("every signal must have a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("signal ids must be unique")

    # Scores stay normalized against the *whole* corpus, not the friction subset.
    # Dividing by the smaller post-filter count would raise every importance
    # score purely because satisfied users were removed, which is an artifact a
    # judge would rightly attack.
    total = max(1, len(signal_rows))
    dropped_praise: list[dict[str, Any]] = []
    if settings.friction_filter:
        signal_rows, dropped_praise = partition_friction_signals(
            signal_rows, praise_rating_floor=settings.praise_rating_floor
        )
    if not signal_rows:
        return []

    clusters = cluster_signals(
        signal_rows,
        embedder=embedder,
        similarity_threshold=settings.cluster_similarity,
        random_seed=settings.random_seed,
    )

    eligible = [
        cluster
        for cluster in clusters
        if len(cluster) >= settings.min_cluster_size
        or (settings.include_singletons and len(cluster) == 1)
    ]
    eligible.sort(key=lambda cluster: (-len(cluster), str(cluster[0]["id"])))
    needs: list[dict[str, Any]] = []
    unnameable_clusters = 0
    for cluster in eligible:
        if settings.max_needs is not None and len(needs) >= settings.max_needs:
            break
        cluster = cluster[: settings.max_signals_per_cluster]
        analyses = [analyze_sentiment(signal) for signal in cluster]
        aspects = _cluster_aspects(cluster)
        allowed_ids = [str(signal["id"]) for signal in cluster]
        chosen: dict[str, Any] | None = None
        model_derived = False
        self_consistency = 0.0
        failures = 0
        if extractor is not None:
            candidates: list[dict[str, Any]] = []
            for _ in range(settings.llm_samples):
                try:
                    candidate = extractor.extract(
                        cluster_texts=[str(signal.get("text") or "") for signal in cluster],
                        allowed_signal_ids=allowed_ids,
                        aspects=aspects,
                        temperature=settings.llm_temperature,
                    )
                    candidates.append(_validate_extraction(candidate, allowed_ids))
                except (ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError):
                    failures += 1
            if candidates:
                chosen, agreement = _select_consensus(candidates)
                model_derived = True
                self_consistency = agreement * (len(candidates) / settings.llm_samples)

        if chosen is None:
            if settings.drop_unnameable_clusters and is_unnameable_cluster(cluster, aspects):
                unnameable_clusters += 1
                continue
            chosen = _offline_extract(cluster, aspects, analyses)
            chosen["supporting_signal_ids"] = _grounded_offline_signal_ids(
                cluster, str(chosen["latent_need"])
            )
            self_consistency = 1.0 if extractor is None else 0.0

        chosen = _validate_extraction(chosen, allowed_ids)
        supporting_ids = chosen["supporting_signal_ids"]
        selected_analyses = [
            analysis
            for signal, analysis in zip(cluster, analyses, strict=True)
            if str(signal["id"]) in set(supporting_ids)
        ]
        scores = _score_need(selected_analyses, len(supporting_ids), total)
        mean_sentiment = scores["mean_sentiment"]
        mean_intensity = scores["mean_intensity"]
        importance = scores["importance"]
        satisfaction = scores["satisfaction"]
        opportunity = scores["opportunity_score"]
        cluster_id = _stable_id("C", *sorted(supporting_ids))
        need_id = _stable_id(
            "N",
            str(chosen["latent_need"]).casefold(),
            *sorted(supporting_ids),
        )
        selected_timestamps = [
            parsed
            for parsed in (
                _parse_timestamp(signal.get("timestamp"))
                for signal in cluster
                if str(signal["id"]) in set(supporting_ids)
            )
            if parsed is not None
        ]
        needs.append(
            {
                "id": need_id,
                "latent_need": chosen["latent_need"],
                "jtbd_statement": chosen["jtbd_statement"],
                "kano_class": chosen["kano_class"],
                "root_cause_hypothesis": chosen["root_cause_hypothesis"],
                "symptom": chosen["symptom"],
                "supporting_signal_ids": supporting_ids,
                "cluster_id": cluster_id,
                "importance": round(importance, 6),
                "satisfaction": round(satisfaction, 6),
                "opportunity_score": round(opportunity, 6),
                "metadata": {
                    "aspects": aspects,
                    "support_count": len(supporting_ids),
                    "mean_sentiment": round(mean_sentiment, 6),
                    "mean_sentiment_intensity": round(mean_intensity, 6),
                    "negative_share": round(scores["negative_share"], 6),
                    "friction_filter": settings.friction_filter,
                    "praise_signals_dropped": len(dropped_praise),
                    "unnameable_clusters_dropped": unnameable_clusters,
                    "self_consistency": round(self_consistency, 6),
                    "llm_samples": settings.llm_samples if extractor else 0,
                    "llm_failures": failures,
                    "inference": "gemini" if model_derived else "offline",
                    "semantic_support_filter": (
                        "model_cited_ids"
                        if model_derived
                        else "deterministic_domain_vocabulary"
                    ),
                    "semantic_signals_excluded": len(cluster) - len(supporting_ids),
                    "priority_score_semantics": (
                        "review-derived priority proxy; not survey-validated ODI"
                    ),
                    "temporal_scope": {
                        "evidence_start": (
                            min(selected_timestamps).isoformat()
                            if selected_timestamps
                            else None
                        ),
                        "evidence_end": (
                            max(selected_timestamps).isoformat()
                            if selected_timestamps
                            else None
                        ),
                    },
                },
            }
        )
    _attach_unassigned_domain_signals(needs, signal_rows, total=total)
    for need in needs:
        metadata = dict(_mapping(need.get("metadata") or {}))
        metadata["unnameable_clusters_dropped"] = unnameable_clusters
        need["metadata"] = metadata
    if settings.merge_similar_needs:
        needs = merge_redundant_needs(
            needs,
            signal_by_id={str(row["id"]): row for row in signal_rows},
            total=total,
            embedder=embedder,
            similarity_threshold=settings.merge_similarity,
            merger=merger,
            temperature=settings.llm_temperature,
        )
    needs.sort(key=lambda need: (-float(need["opportunity_score"]), str(need["id"])))
    return needs
