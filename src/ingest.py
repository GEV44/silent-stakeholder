"""Normalize app reviews and a GitHub roadmap into canonical artifacts.

Examples
--------
Online GitHub pull with a local review export::

    python -m src.ingest --app wordpress --reviews-csv data/reviews.csv

Fully offline, reproducible fixture run::

    python -m src.ingest --app wordpress --reviews-csv tests/fixtures/reviews.csv \
        --github-fixture tests/fixtures/github_wordpress.json --offline
"""

from __future__ import annotations

import argparse
import csv
import io
import ipaddress
import json
import os
import re
import socket
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from pydantic import ValidationError

from .io_utils import (
    ArtifactIOError,
    assert_unique_ids,
    atomic_write_json,
    canonical_text,
    make_stable_id,
    read_json,
)
from .schema import (
    PriorityMetadata,
    PriorityTier,
    RoadmapItem,
    RoadmapItemType,
    RoadmapState,
    Signal,
)

DEFAULT_GITHUB_API = "https://api.github.com"
GITHUB_API_VERSION = "2026-03-10"
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_REVIEW_DOWNLOAD_BYTES = 64 * 1024 * 1024
MAX_LOCAL_REVIEW_BYTES = 256 * 1024 * 1024
MAX_REVIEW_ROWS = 1_000_000
MAX_REVIEW_TEXT_CHARS = 100_000
MAX_TOTAL_REVIEW_TEXT_CHARS = 128 * 1024 * 1024
MAX_REVIEW_REDIRECTS = 5
MAX_GITHUB_ITEMS = 100_000
MAX_GITHUB_PAGE_BYTES = 16 * 1024 * 1024
MAX_PARQUET_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
REVIEW_SOURCE_HOST_SUFFIXES = (
    "hf.co",
    "huggingface.co",
    "raw.githubusercontent.com",
)


class IngestError(RuntimeError):
    """A local input is missing or structurally unsuitable for ingestion."""


class GitHubAPIError(IngestError):
    """A GitHub response could not be used safely."""


@dataclass(frozen=True, slots=True)
class AppConfig:
    slug: str
    package_name: str
    repository: str


@dataclass(frozen=True, slots=True)
class GitHubFixtureData:
    milestones: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    repository: str | None = None
    retrieved_at: str | None = None
    state_scope: str | None = None
    api_version: str | None = None


APP_CONFIGS: dict[str, AppConfig] = {
    "wordpress": AppConfig(
        slug="wordpress",
        package_name="org.wordpress.android",
        repository="wordpress-mobile/WordPress-Android",
    ),
    "ppsspp": AppConfig(
        slug="ppsspp",
        package_name="org.ppsspp.ppsspp",
        repository="hrydgard/ppsspp",
    ),
    "antennapod": AppConfig(
        slug="antennapod",
        package_name="de.danoeh.antennapod",
        repository="AntennaPod/AntennaPod",
    ),
}


@dataclass(frozen=True, slots=True)
class ReviewIngestStats:
    total_rows: int
    matching_rows: int
    blank_text_rows: int
    invalid_rows: int
    duplicates_removed: int
    emitted_signals: int


@dataclass(frozen=True, slots=True)
class ReviewIngestResult:
    signals: list[Signal]
    stats: ReviewIngestStats


@dataclass(frozen=True, slots=True)
class IngestArtifacts:
    signals: list[Signal]
    roadmap: list[RoadmapItem]
    review_stats: ReviewIngestStats
    scope_metadata: dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_datetime(value: Any, *, context: str) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    candidates = (normalized, normalized.replace(" ", "T", 1))
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            pass
    for date_format in (
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%B %d %Y",
        "%b %d %Y",
    ):
        try:
            return datetime.strptime(text, date_format).replace(tzinfo=UTC)
        except ValueError:
            pass
    raise IngestError(f"invalid datetime {text!r} in {context}")


def _parse_rating(value: Any, *, context: str) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        rating = float(str(value).strip())
    except ValueError as exc:
        raise IngestError(f"invalid rating {value!r} in {context}") from exc
    if not 1.0 <= rating <= 5.0:
        raise IngestError(f"rating must be in [1, 5] in {context}; got {rating}")
    return rating


def _normalize_review_text(value: Any) -> str:
    """Normalize whitespace while preserving punctuation and original casing."""

    return canonical_text(value)


def _field_name(
    fieldnames: Sequence[str] | None,
    aliases: Sequence[str],
    *,
    required: bool,
) -> str | None:
    lookup = {
        canonical_text(name).casefold(): name for name in (fieldnames or []) if name is not None
    }
    for alias in aliases:
        if alias.casefold() in lookup:
            return lookup[alias.casefold()]
    if required:
        raise IngestError(
            "review CSV is missing a required column; expected one of " + ", ".join(aliases)
        )
    return None


def _url_origin(value: str, *, context: str) -> tuple[str, str, int]:
    """Return a strict HTTPS origin suitable for credential-bearing requests."""

    parsed = urlparse(value)
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise IngestError(f"{context} must use an absolute HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise IngestError(f"{context} must not contain URL credentials")
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise IngestError(f"{context} has an invalid port") from exc
    return "https", parsed.hostname.casefold().rstrip("."), port


def _public_review_url(value: str) -> str:
    """Reject review URLs that could reach local or private infrastructure.

    DNS is checked before each request and redirect. Production deployments
    should also enforce an outbound network policy because DNS can change after
    validation and before the socket is opened.
    """

    _scheme, hostname, port = _url_origin(value, context="review CSV URL")
    if port != 443:
        raise IngestError("review CSV URL must use the standard HTTPS port 443")
    if not any(
        hostname == suffix or hostname.endswith(f".{suffix}")
        for suffix in REVIEW_SOURCE_HOST_SUFFIXES
    ):
        raise IngestError("review CSV URL host is not in the approved source allowlist")
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise IngestError("review CSV URL resolves to a local/private host")
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise IngestError("review CSV URL host could not be resolved") from exc
    if not addresses:
        raise IngestError("review CSV URL host resolved to no addresses")
    for address in addresses:
        try:
            candidate = ipaddress.ip_address(str(address[4][0]).split("%", 1)[0])
        except ValueError as exc:
            raise IngestError("review CSV URL resolved to an invalid address") from exc
        if not candidate.is_global:
            raise IngestError("review CSV URL resolves to a non-public address")
    return value


def _safe_source_reference(source: str | os.PathLike[str]) -> str:
    """Keep useful provenance without persisting credentials or private paths."""

    value = str(source)
    parsed = urlparse(value)
    if parsed.scheme.casefold() in {"http", "https"}:
        _scheme, hostname, port = _url_origin(value, context="review CSV URL")
        netloc = hostname if port == 443 else f"{hostname}:{port}"
        return urlunparse(("https", netloc, "", "", "", ""))
    return "local-review-input"


def _bounded_response_bytes(response: requests.Response, *, limit: int, context: str) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            announced = int(content_length)
        except ValueError as exc:
            raise IngestError(f"{context} returned an invalid Content-Length") from exc
        if announced < 0 or announced > limit:
            raise IngestError(f"{context} exceeds the {limit}-byte safety limit")

    chunks: list[bytes] = []
    total = 0
    try:
        iterator = response.iter_content(chunk_size=1024 * 1024)
        for chunk in iterator:
            if not chunk:
                continue
            total += len(chunk)
            if total > limit:
                raise IngestError(f"{context} exceeds the {limit}-byte safety limit")
            chunks.append(chunk)
    except requests.RequestException as exc:
        raise IngestError(f"could not read {context}: {exc}") from exc
    return b"".join(chunks)


def _assert_clean_review_session(session: requests.Session | None) -> None:
    """Do not repurpose a credential-bearing session for an arbitrary data URL."""

    if session is None:
        return
    if getattr(session, "auth", None):
        raise IngestError("review CSV download requires a session without authentication")
    headers = getattr(session, "headers", {})
    default_headers = {"accept", "accept-encoding", "connection", "user-agent"}
    if isinstance(headers, Mapping) and any(
        str(key).casefold() not in default_headers for key in headers
    ):
        raise IngestError("review CSV download requires a session without custom headers")
    cookies = getattr(session, "cookies", None)
    if cookies and len(cookies):
        raise IngestError("review CSV download requires a session without cookies")
    if getattr(session, "proxies", None):
        raise IngestError("review CSV download requires a session without configured proxies")


def _review_http_session(
    session: requests.Session | None,
) -> tuple[requests.Session | Any, bool]:
    _assert_clean_review_session(session)
    if session is not None and not isinstance(session, requests.Session):
        # Test doubles never reach production network code.
        return session, False
    clean = requests.Session()
    clean.auth = None
    clean.cookies.clear()
    clean.headers.clear()
    clean.proxies.clear()
    clean.trust_env = False
    clean.hooks = {"response": []}
    return clean, True


def _reject_unsafe_local_source(source: str | os.PathLike[str]) -> Path:
    value = os.fspath(source)
    normalized = value.replace("\\", "/")
    if normalized.startswith("//"):
        raise IngestError("network, UNC, and device review paths are forbidden")
    parsed = urlparse(value)
    if parsed.scheme and not re.fullmatch(r"[A-Za-z]", parsed.scheme):
        raise IngestError("review input uses an unsupported URL/path scheme")
    return Path(value)


def _read_local_review_text(path: Path) -> str:
    display_name = path.name or "review input"
    try:
        with path.open("rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise IngestError("review CSV must be a regular file")
            if metadata.st_size > MAX_LOCAL_REVIEW_BYTES:
                raise IngestError(
                    f"review input exceeds the {MAX_LOCAL_REVIEW_BYTES}-byte safety limit"
                )
            payload = handle.read(MAX_LOCAL_REVIEW_BYTES + 1)
    except FileNotFoundError as exc:
        raise IngestError(f"review CSV does not exist: {display_name}") from exc
    except OSError as exc:
        raise IngestError(f"could not read review CSV: {display_name}") from exc
    if len(payload) > MAX_LOCAL_REVIEW_BYTES:
        raise IngestError(
            f"review input exceeds the {MAX_LOCAL_REVIEW_BYTES}-byte safety limit"
        )
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestError(f"review CSV is not valid UTF-8: {display_name}") from exc


def _csv_content(
    source: str | os.PathLike[str],
    *,
    session: requests.Session | None,
    offline: bool,
    timeout: float,
) -> str:
    source_text = str(source)
    parsed = urlparse(source_text)
    if parsed.scheme.casefold() in {"http", "https"}:
        if offline:
            raise IngestError("offline mode forbids downloading a review CSV")
        http, owns_http = _review_http_session(session)
        current_url = _public_review_url(source_text)
        try:
            for redirect_count in range(MAX_REVIEW_REDIRECTS + 1):
                try:
                    response = http.get(
                        current_url,
                        timeout=timeout,
                        allow_redirects=False,
                        stream=True,
                        headers={"Accept": "text/csv, text/plain;q=0.9, */*;q=0.1"},
                    )
                except requests.RequestException as exc:
                    raise IngestError("could not download review CSV") from exc
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        response.close()
                        raise IngestError("review CSV redirect omitted Location")
                    if redirect_count >= MAX_REVIEW_REDIRECTS:
                        response.close()
                        raise IngestError("review CSV exceeded the redirect safety limit")
                    response.close()
                    current_url = _public_review_url(urljoin(current_url, location))
                    continue
                if not 200 <= response.status_code < 300:
                    response.close()
                    raise IngestError(
                        f"review CSV download failed with HTTP {response.status_code}"
                    )
                try:
                    payload = _bounded_response_bytes(
                        response,
                        limit=MAX_REVIEW_DOWNLOAD_BYTES,
                        context="review CSV download",
                    )
                finally:
                    response.close()
                try:
                    return payload.decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    raise IngestError("downloaded review CSV is not valid UTF-8") from exc
            raise IngestError("review CSV exceeded the redirect safety limit")
        finally:
            if owns_http:
                http.close()

    return _read_local_review_text(_reject_unsafe_local_source(source))


def _review_rows(
    source: str | os.PathLike[str],
    *,
    session: requests.Session | None,
    offline: bool,
    timeout: float,
) -> tuple[list[str], Iterable[Mapping[str, Any]]]:
    """Read CSV or the Hugging Face parquet snapshot behind one interface."""

    source_text = str(source)
    parsed = urlparse(source_text)
    local_path = (
        None
        if parsed.scheme.casefold() in {"http", "https"}
        else _reject_unsafe_local_source(source)
    )
    if local_path is not None and local_path.suffix.casefold() in {
        ".parquet",
        ".pq",
    }:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise IngestError(
                "parquet review input requires pyarrow"
            ) from exc
        handle: Any = None
        try:
            handle = local_path.open("rb")
            file_stats = os.fstat(handle.fileno())
            if not stat.S_ISREG(file_stats.st_mode):
                handle.close()
                raise IngestError("review parquet must be a regular file")
            if file_stats.st_size > MAX_LOCAL_REVIEW_BYTES:
                handle.close()
                raise IngestError(
                    f"review input exceeds the {MAX_LOCAL_REVIEW_BYTES}-byte safety limit"
                )
            parquet = pq.ParquetFile(handle)
            metadata = parquet.metadata
            uncompressed_bytes = sum(
                metadata.row_group(index).total_byte_size
                for index in range(metadata.num_row_groups)
            )
            if metadata.num_rows > MAX_REVIEW_ROWS:
                handle.close()
                raise IngestError(
                    f"review input exceeds the {MAX_REVIEW_ROWS}-row safety limit"
                )
            if uncompressed_bytes > MAX_PARQUET_UNCOMPRESSED_BYTES:
                handle.close()
                raise IngestError(
                    "review parquet exceeds the uncompressed-byte safety limit"
                )
        except FileNotFoundError as exc:
            raise IngestError(f"review parquet does not exist: {local_path.name}") from exc
        except (OSError, ValueError, pa.ArrowException) as exc:
            if handle is not None and not handle.closed:
                handle.close()
            raise IngestError(f"could not read review parquet: {local_path.name}") from exc

        def records() -> Iterable[Mapping[str, Any]]:
            try:
                for batch in parquet.iter_batches(batch_size=10_000):
                    yield from batch.to_pylist()
            except pa.ArrowException as exc:
                raise IngestError(
                    f"could not read review parquet: {local_path.name}"
                ) from exc
            finally:
                handle.close()

        return [str(column) for column in parquet.schema.names], records()

    content = _csv_content(
        source,
        session=session,
        offline=offline,
        timeout=timeout,
    )
    reader = csv.DictReader(io.StringIO(content, newline=""))
    if reader.fieldnames is None:
        raise IngestError("review CSV has no header row")
    return list(reader.fieldnames), reader


def ingest_reviews_csv(
    source: str | os.PathLike[str],
    package_name: str,
    *,
    source_name: str = "app_review",
    session: requests.Session | None = None,
    offline: bool = False,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    require_matches: bool = True,
) -> ReviewIngestResult:
    """Filter and exactly deduplicate a review CSV for one package.

    Exact duplicate text with the same rating is collapsed. The newest record
    becomes the representative, and ``duplicate_count`` remains in metadata so
    later confidence logic can decide whether repeated text is organic signal
    or spam. Different ratings are retained as contradictory evidence.
    """

    package_name = canonical_text(package_name)
    if not package_name:
        raise IngestError("package_name cannot be empty")

    fieldnames, rows = _review_rows(
        source,
        session=session,
        offline=offline,
        timeout=timeout,
    )

    package_field = _field_name(
        fieldnames,
        ("package_name", "package", "app_id"),
        required=True,
    )
    text_field = _field_name(
        fieldnames,
        ("review", "review_text", "text", "content"),
        required=True,
    )
    if package_field is None or text_field is None:  # pragma: no cover - required=True
        raise IngestError("review CSV required columns could not be resolved")
    rating_field = _field_name(
        fieldnames,
        ("star", "stars", "rating", "score"),
        required=False,
    )
    timestamp_field = _field_name(
        fieldnames,
        ("date", "timestamp", "created_at", "review_date"),
        required=False,
    )
    source_id_field = _field_name(
        fieldnames,
        ("id", "review_id"),
        required=False,
    )
    version_field = _field_name(
        fieldnames,
        ("version_id", "version", "app_version"),
        required=False,
    )

    total_rows = 0
    matching_rows = 0
    blank_text_rows = 0
    invalid_rows = 0
    total_text_chars = 0
    # key -> (duplicate count, deterministic best representative). Retaining
    # every duplicate row made a repeated-text input an avoidable memory DoS.
    buckets: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}

    def representative_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        timestamp = candidate["timestamp"]
        timestamp_key = timestamp.isoformat() if isinstance(timestamp, datetime) else ""
        return (
            timestamp_key,
            candidate["source_id"] or "",
            candidate["version"] or "",
            candidate["text"],
        )

    for row_number, row in enumerate(rows, start=2):
        total_rows += 1
        if total_rows > MAX_REVIEW_ROWS:
            raise IngestError(f"review input exceeds the {MAX_REVIEW_ROWS}-row safety limit")
        row_package = canonical_text(row.get(package_field, ""))
        if row_package.casefold() != package_name.casefold():
            continue
        matching_rows += 1
        text = _normalize_review_text(row.get(text_field, ""))
        if not text:
            blank_text_rows += 1
            continue
        if len(text) > MAX_REVIEW_TEXT_CHARS:
            raise IngestError(
                f"review CSV row {row_number} exceeds the "
                f"{MAX_REVIEW_TEXT_CHARS}-character text safety limit"
            )
        total_text_chars += len(text)
        if total_text_chars > MAX_TOTAL_REVIEW_TEXT_CHARS:
            raise IngestError("review input exceeds the aggregate text safety limit")
        try:
            rating = _parse_rating(
                row.get(rating_field) if rating_field else None,
                context=f"review CSV row {row_number}",
            )
            timestamp = _parse_datetime(
                row.get(timestamp_field) if timestamp_field else None,
                context=f"review CSV row {row_number}",
            )
        except IngestError:
            invalid_rows += 1
            continue

        source_id_value = row.get(source_id_field) if source_id_field else None
        source_id = canonical_text(source_id_value) if source_id_value else None
        version_value = row.get(version_field) if version_field else None
        version = canonical_text(version_value) if version_value else None
        rating_key = "" if rating is None else f"{rating:g}"
        dedup_key = (text.casefold(), rating_key)
        candidate = {
            "text": text,
            "rating": rating,
            "timestamp": timestamp,
            "source_id": source_id,
            "version": version,
        }
        prior = buckets.get(dedup_key)
        if prior is None:
            buckets[dedup_key] = (1, candidate)
        else:
            count, representative = prior
            if representative_key(candidate) > representative_key(representative):
                representative = candidate
            buckets[dedup_key] = (count + 1, representative)

    if require_matches and matching_rows == 0:
        raise IngestError(f"review CSV contains no rows for package {package_name!r}")

    signals: list[Signal] = []
    duplicates_removed = 0
    for (text_key, rating_key), (count, representative) in buckets.items():
        duplicates_removed += count - 1
        metadata: dict[str, Any] = {
            "duplicate_count": count,
            "deduplication": "exact_normalized_text_and_rating",
        }
        if representative["version"]:
            metadata["version_id"] = representative["version"]
        signal_id = make_stable_id(
            "S",
            source_name,
            package_name,
            text_key,
            rating_key,
        )
        try:
            signals.append(
                Signal(
                    id=signal_id,
                    source=source_name,
                    source_id=representative["source_id"],
                    text=representative["text"],
                    timestamp=representative["timestamp"],
                    rating=representative["rating"],
                    package_name=package_name,
                    metadata=metadata,
                )
            )
        except ValidationError as exc:
            raise IngestError(f"normalized review failed schema validation: {exc}") from exc

    signals.sort(key=lambda signal: signal.id)
    assert_unique_ids(signals)
    stats = ReviewIngestStats(
        total_rows=total_rows,
        matching_rows=matching_rows,
        blank_text_rows=blank_text_rows,
        invalid_rows=invalid_rows,
        duplicates_removed=duplicates_removed,
        emitted_signals=len(signals),
    )
    return ReviewIngestResult(signals=signals, stats=stats)


def load_reviews_csv(
    source: str | os.PathLike[str],
    package_name: str,
    **kwargs: Any,
) -> list[Signal]:
    """Convenience wrapper returning only normalized signals."""

    return ingest_reviews_csv(source, package_name, **kwargs).signals


def _repository_parts(repository: str) -> tuple[str, str]:
    repository = canonical_text(repository).strip("/")
    match = re.fullmatch(r"([^/\s]+)/([^/\s]+)", repository)
    if not match:
        raise IngestError("repository must use the 'owner/name' form")
    return match.group(1), match.group(2)


def _next_link(response: requests.Response) -> str | None:
    response_links = getattr(response, "links", None)
    if response_links and isinstance(response_links, Mapping):
        next_entry = response_links.get("next")
        if isinstance(next_entry, Mapping) and next_entry.get("url"):
            return str(next_entry["url"])

    link_header = response.headers.get("Link", "")
    # GitHub Link values are URI references in angle brackets followed by
    # semicolon parameters. URLs themselves do not contain unescaped commas.
    for part in link_header.split(","):
        match = re.match(r'\s*<([^>]+)>\s*;\s*rel=["\']?([^"\';]+)', part)
        if match and match.group(2).strip() == "next":
            return match.group(1).strip()
    return None


def _github_json_payload(response: requests.Response | Any) -> Any:
    if isinstance(response, requests.Response):
        try:
            payload = _bounded_response_bytes(
                response,
                limit=MAX_GITHUB_PAGE_BYTES,
                context="GitHub API page",
            )
            return json.loads(payload.decode("utf-8"))
        except (IngestError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GitHubAPIError("GitHub returned invalid or oversized JSON") from exc
    try:
        return response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise GitHubAPIError("GitHub returned invalid JSON") from exc


def _close_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


class GitHubClient:
    """Small transparent GitHub REST client with Link-header pagination."""

    def __init__(
        self,
        *,
        token: str | None = None,
        base_url: str = DEFAULT_GITHUB_API,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_pages: int = 1_000,
        max_items: int = MAX_GITHUB_ITEMS,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.origin = _url_origin(self.base_url, context="GitHub API base URL")
        if token and self.origin != _url_origin(
            DEFAULT_GITHUB_API,
            context="default GitHub API URL",
        ):
            raise GitHubAPIError(
                "refusing to send a GitHub token to a non-default API origin"
            )
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_pages = max_pages
        self.max_items = max_items
        if self.max_pages < 1 or self.max_items < 1:
            raise IngestError("GitHub pagination limits must be positive")
        self.authenticated = bool(token)
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": "firecode-silent-stakeholder/1.0",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def paginate(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch every page by following GitHub's explicit ``rel=next`` link."""

        parsed_endpoint = urlparse(endpoint)
        if (
            endpoint.startswith("//")
            or parsed_endpoint.scheme
            or parsed_endpoint.netloc
            or parsed_endpoint.username is not None
            or parsed_endpoint.password is not None
        ):
            raise GitHubAPIError("GitHub endpoint must be a relative API path")
        next_url = urljoin(self.base_url, endpoint.lstrip("/"))
        try:
            initial_origin = _url_origin(next_url, context="GitHub API request")
        except IngestError as exc:
            raise GitHubAPIError(str(exc)) from exc
        if initial_origin != self.origin:
            raise GitHubAPIError("GitHub endpoint resolved outside the configured API origin")
        visited: set[str] = set()
        items: list[dict[str, Any]] = []
        page_params: Mapping[str, Any] | None = dict(params or {})

        for _ in range(self.max_pages):
            if next_url in visited:
                raise GitHubAPIError(f"GitHub pagination loop detected at {next_url}")
            visited.add(next_url)
            try:
                response = self.session.get(
                    next_url,
                    headers=self.headers,
                    params=page_params,
                    timeout=self.timeout,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException as exc:
                raise GitHubAPIError("GitHub request failed") from exc
            page_params = None
            try:
                status_code = response.status_code
                headers = response.headers
                candidate = _next_link(response)
                payload = (
                    None
                    if 300 <= status_code < 400
                    else _github_json_payload(response)
                )
            finally:
                _close_response(response)

            if 300 <= status_code < 400:
                raise GitHubAPIError(
                    "refusing an HTTP redirect from the credential-bearing GitHub API request"
                )

            if status_code >= 400:
                remaining = headers.get("X-RateLimit-Remaining")
                if status_code in {403, 429} and remaining == "0":
                    reset = headers.get("X-RateLimit-Reset", "unknown")
                    raise GitHubAPIError(
                        "GitHub API rate limit exhausted; set GITHUB_TOKEN "
                        f"or retry after Unix time {reset}"
                    )
                message = (
                    canonical_text(payload.get("message", ""))
                    if isinstance(payload, Mapping)
                    else ""
                )
                suffix = f": {message}" if message else ""
                raise GitHubAPIError(
                    f"GitHub returned HTTP {status_code}{suffix}"
                )
            if not isinstance(payload, list):
                raise GitHubAPIError(
                    "GitHub pagination expected a list; "
                    f"got {type(payload).__name__}"
                )
            for index, item in enumerate(payload):
                if not isinstance(item, dict):
                    raise GitHubAPIError(f"GitHub item {index} is not an object")
                items.append(item)
                if len(items) > self.max_items:
                    raise GitHubAPIError(
                        f"GitHub pagination exceeded the safety limit of {self.max_items} items"
                    )

            if not candidate:
                return items
            candidate_url = urljoin(next_url, candidate)
            try:
                candidate_origin = _url_origin(
                    candidate_url,
                    context="GitHub pagination link",
                )
            except IngestError as exc:
                raise GitHubAPIError(str(exc)) from exc
            if candidate_origin != self.origin:
                raise GitHubAPIError(
                    "refusing pagination link to a different host or HTTPS origin"
                )
            next_url = candidate_url

        raise GitHubAPIError(
            f"GitHub pagination exceeded the safety limit of {self.max_pages} pages"
        )

    def fetch_repository(
        self,
        repository: str,
        *,
        state: str = "open",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if state not in {"open", "all"}:
            raise IngestError("GitHub state must be 'open' or 'all'")
        if state == "all" and not self.authenticated:
            raise GitHubAPIError(
                "full all-state GitHub history requires GITHUB_TOKEN; "
                "without a token use the open-only scope"
            )
        owner, name = _repository_parts(repository)
        prefix = f"/repos/{owner}/{name}"
        milestones = self.paginate(
            f"{prefix}/milestones",
            params={"state": state, "per_page": 100},
        )
        issues = self.paginate(
            f"{prefix}/issues",
            params={"state": state, "per_page": 100},
        )
        return milestones, issues


# `pri <tier>` covers the bracketed `[Pri] High` convention that
# wordpress-mobile/WordPress-Android actually uses.  Without it we read 234 of
# 774 open issues as having no stated priority when the maintainers had stated
# one, and reported issues they marked High as low priority -- a false claim
# about the reader's own board, which is the worst kind this project can make.
_LABEL_RULES: tuple[tuple[PriorityTier, tuple[str, ...]], ...] = (
    (
        PriorityTier.CRITICAL,
        (
            "p0",
            "critical",
            "blocker",
            "priority: critical",
            "severity: critical",
            "pri critical",
        ),
    ),
    (
        PriorityTier.HIGH,
        ("p1", "high priority", "priority: high", "severity: high", "pri high"),
    ),
    (
        PriorityTier.MEDIUM,
        (
            "p2",
            "medium priority",
            "priority: medium",
            "severity: medium",
            "pri medium",
        ),
    ),
    (
        PriorityTier.LOW,
        (
            "p3",
            "p4",
            "low priority",
            "priority: low",
            "severity: low",
            "pri low",
        ),
    ),
    (
        PriorityTier.BACKLOG,
        ("backlog", "icebox", "wontfix", "won't fix", "deferred"),
    ),
)
_TIER_SCORE = {
    PriorityTier.CRITICAL: 1.0,
    PriorityTier.HIGH: 0.85,
    PriorityTier.MEDIUM: 0.65,
    PriorityTier.LOW: 0.35,
    PriorityTier.BACKLOG: 0.15,
    PriorityTier.UNSPECIFIED: 0.5,
}
_BACKLOG_MILESTONE_WORDS = ("future", "backlog", "icebox", "someday", "later")


def _label_matches(label: str, rule: str) -> bool:
    # Brackets are separators, not characters.  GitHub label taxonomies
    # routinely namespace with them (`[Pri] High`, `[Type] Bug`), and leaving
    # them in defeats the `\b` boundary below: `\bpri high\b` cannot match
    # `[pri] high` because the `]` sits between the two words.
    normalized = re.sub(r"[_\-/\[\]]+", " ", label.casefold())
    normalized = " ".join(normalized.split())
    rule_normalized = re.sub(r"[_\-/\[\]]+", " ", rule.casefold())
    rule_normalized = " ".join(rule_normalized.split())
    if len(rule_normalized) <= 2:
        return normalized == rule_normalized
    return normalized == rule_normalized or bool(
        re.search(rf"\b{re.escape(rule_normalized)}\b", normalized)
    )


def derive_priority_metadata(
    *,
    labels: Sequence[str],
    milestone_title: str | None,
    milestone_due: datetime | None,
    created_at: datetime | None,
    state: str | RoadmapState,
    as_of: datetime | None = None,
) -> PriorityMetadata:
    """Derive an auditable priority tier from explicit and structural clues."""

    now = as_of or _utc_now()
    now = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
    normalized_labels = [canonical_text(label) for label in labels if canonical_text(label)]
    tier = PriorityTier.UNSPECIFIED
    matched_labels: list[str] = []
    for candidate_tier, rules in _LABEL_RULES:
        matches = [
            label
            for label in normalized_labels
            if any(_label_matches(label, rule) for rule in rules)
        ]
        if matches:
            tier = candidate_tier
            matched_labels = matches
            break

    has_explicit = bool(matched_labels)
    reasons: list[str] = []
    if has_explicit:
        reasons.append(
            f"explicit {tier.value} priority label: "
            + ", ".join(sorted(matched_labels, key=str.casefold))
        )

    structural_low = False
    milestone_key = canonical_text(milestone_title or "").casefold()
    if milestone_key and any(word in milestone_key for word in _BACKLOG_MILESTONE_WORDS):
        structural_low = True
        reasons.append(f"backlog-style milestone: {milestone_title}")
        if not has_explicit:
            tier = PriorityTier.BACKLOG
    elif milestone_title is None:
        structural_low = True
        reasons.append("not assigned to a milestone")
        if not has_explicit:
            tier = PriorityTier.LOW

    if milestone_due is not None:
        due = (
            milestone_due.replace(tzinfo=UTC)
            if milestone_due.tzinfo is None
            else milestone_due.astimezone(UTC)
        )
        if due > now + timedelta(days=183):
            structural_low = True
            reasons.append("milestone due more than two quarters out")
            if not has_explicit and tier == PriorityTier.UNSPECIFIED:
                tier = PriorityTier.LOW

    state_value = state.value if isinstance(state, RoadmapState) else str(state)
    if created_at is not None and state_value.casefold() == RoadmapState.OPEN.value:
        created = (
            created_at.replace(tzinfo=UTC)
            if created_at.tzinfo is None
            else created_at.astimezone(UTC)
        )
        if created < now - timedelta(days=365):
            structural_low = True
            reasons.append("open for more than one year")
            if not has_explicit and tier == PriorityTier.UNSPECIFIED:
                tier = PriorityTier.LOW

    if not reasons:
        reasons.append("no explicit or structural priority signal")
    is_low = tier in {PriorityTier.LOW, PriorityTier.BACKLOG} or (
        structural_low and tier not in {PriorityTier.CRITICAL, PriorityTier.HIGH}
    )
    return PriorityMetadata(
        tier=tier,
        score=_TIER_SCORE[tier],
        is_low_priority=is_low,
        has_explicit_priority=has_explicit,
        reasons=reasons,
        matched_labels=matched_labels,
    )


def _github_labels(value: Any, *, context: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise IngestError(f"{context} labels must be a list")
    labels: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            label = canonical_text(item.get("name", ""))
        else:
            label = canonical_text(item)
        if label:
            labels.append(label)
    # GitHub label order has no semantic meaning.
    return sorted(set(labels), key=str.casefold)


def _required_int(payload: Mapping[str, Any], field: str, *, context: str) -> int:
    try:
        value = int(payload[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise IngestError(f"{context} has no valid integer {field!r}") from exc
    if value < 1:
        raise IngestError(f"{context} {field!r} must be positive")
    return value


def _required_text(payload: Mapping[str, Any], field: str, *, context: str) -> str:
    value = canonical_text(payload.get(field, ""))
    if not value:
        raise IngestError(f"{context} has no non-empty {field!r}")
    return value


def normalize_github_roadmap(
    repository: str,
    milestones: Iterable[Mapping[str, Any]],
    issues: Iterable[Mapping[str, Any]],
    *,
    as_of: datetime | None = None,
) -> list[RoadmapItem]:
    """Normalize GitHub payloads, dropping pull requests from the issue feed."""

    owner, name = _repository_parts(repository)
    canonical_repository = f"{owner}/{name}"
    milestone_payloads: dict[int, Mapping[str, Any]] = {}
    for raw in milestones:
        if not isinstance(raw, Mapping):
            raise IngestError("GitHub milestone fixture contains a non-object item")
        number = _required_int(raw, "number", context="GitHub milestone")
        if number in milestone_payloads:
            raise IngestError(f"duplicate GitHub milestone number {number}")
        milestone_payloads[number] = raw

    normalized: list[RoadmapItem] = []
    for number, raw in milestone_payloads.items():
        context = f"GitHub milestone #{number}"
        title = _required_text(raw, "title", context=context)
        state = canonical_text(raw.get("state", "")).casefold()
        try:
            state_enum = RoadmapState(state)
        except ValueError as exc:
            raise IngestError(f"{context} has invalid state {state!r}") from exc
        due = _parse_datetime(raw.get("due_on"), context=f"{context} due_on")
        created = _parse_datetime(raw.get("created_at"), context=f"{context} created_at")
        priority = derive_priority_metadata(
            labels=[],
            milestone_title=title,
            milestone_due=due,
            created_at=created,
            state=state_enum,
            as_of=as_of,
        )
        try:
            normalized.append(
                RoadmapItem(
                    id=make_stable_id("R", "github", canonical_repository, "milestone", number),
                    type=RoadmapItemType.MILESTONE,
                    repository=canonical_repository,
                    number=number,
                    title=title,
                    body=str(raw.get("description") or "").strip(),
                    state=state_enum,
                    state_reason=raw.get("state_reason"),
                    labels=[],
                    milestone_due=due,
                    created_at=created,
                    updated_at=_parse_datetime(
                        raw.get("updated_at"), context=f"{context} updated_at"
                    ),
                    closed_at=_parse_datetime(raw.get("closed_at"), context=f"{context} closed_at"),
                    html_url=raw.get("html_url"),
                    priority=priority,
                    metadata={"provider": "github", "api_version": GITHUB_API_VERSION},
                )
            )
        except ValidationError as exc:
            raise IngestError(f"{context} failed schema validation: {exc}") from exc

    seen_issue_numbers: set[int] = set()
    for raw in issues:
        if not isinstance(raw, Mapping):
            raise IngestError("GitHub issue fixture contains a non-object item")
        if "pull_request" in raw:
            continue
        number = _required_int(raw, "number", context="GitHub issue")
        context = f"GitHub issue #{number}"
        if number in seen_issue_numbers:
            raise IngestError(f"duplicate GitHub issue number {number}")
        seen_issue_numbers.add(number)
        title = _required_text(raw, "title", context=context)
        state = canonical_text(raw.get("state", "")).casefold()
        try:
            state_enum = RoadmapState(state)
        except ValueError as exc:
            raise IngestError(f"{context} has invalid state {state!r}") from exc

        milestone_ref = raw.get("milestone")
        milestone_title: str | None = None
        milestone_number: int | None = None
        milestone_due: datetime | None = None
        if milestone_ref is not None:
            if not isinstance(milestone_ref, Mapping):
                raise IngestError(f"{context} milestone must be an object or null")
            milestone_number = _required_int(
                milestone_ref, "number", context=f"{context} milestone"
            )
            milestone_raw = milestone_payloads.get(milestone_number, milestone_ref)
            milestone_title = _required_text(milestone_raw, "title", context=f"{context} milestone")
            milestone_due = _parse_datetime(
                milestone_raw.get("due_on"),
                context=f"{context} milestone due_on",
            )
        labels = _github_labels(raw.get("labels"), context=context)
        created = _parse_datetime(raw.get("created_at"), context=f"{context} created_at")
        priority = derive_priority_metadata(
            labels=labels,
            milestone_title=milestone_title,
            milestone_due=milestone_due,
            created_at=created,
            state=state_enum,
            as_of=as_of,
        )
        try:
            normalized.append(
                RoadmapItem(
                    id=make_stable_id("R", "github", canonical_repository, "issue", number),
                    type=RoadmapItemType.ISSUE,
                    repository=canonical_repository,
                    number=number,
                    title=title,
                    body=str(raw.get("body") or "").strip(),
                    state=state_enum,
                    state_reason=raw.get("state_reason"),
                    labels=labels,
                    milestone=milestone_title,
                    milestone_number=milestone_number,
                    milestone_due=milestone_due,
                    created_at=created,
                    updated_at=_parse_datetime(
                        raw.get("updated_at"), context=f"{context} updated_at"
                    ),
                    closed_at=_parse_datetime(raw.get("closed_at"), context=f"{context} closed_at"),
                    html_url=raw.get("html_url"),
                    priority=priority,
                    metadata={"provider": "github", "api_version": GITHUB_API_VERSION},
                )
            )
        except ValidationError as exc:
            raise IngestError(f"{context} failed schema validation: {exc}") from exc

    normalized.sort(
        key=lambda item: (
            0 if item.type == RoadmapItemType.MILESTONE else 1,
            item.number,
        )
    )
    assert_unique_ids(normalized)
    return normalized


def _fixture_collection(payload: Any, key: str, *, source: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise IngestError(f"GitHub fixture must be a JSON object: {source}")
    collection = payload.get(key)
    if not isinstance(collection, list):
        raise IngestError(f"GitHub fixture {source} must contain a {key!r} list")
    if not all(isinstance(item, dict) for item in collection):
        raise IngestError(f"GitHub fixture {source} {key!r} contains a non-object item")
    return collection


def _load_github_fixture_data(path: str | os.PathLike[str]) -> GitHubFixtureData:
    """Load fixture records and retain provenance carried by a raw snapshot.

    A combined file has ``{"milestones": [...], "issues": [...]}``. A fixture
    directory can instead contain ``milestones*.json`` and ``issues*.json`` page
    files, each either a raw list or a combined-object fragment.
    """

    source = Path(path)
    if source.is_file():
        try:
            payload = read_json(source)
        except ArtifactIOError as exc:
            raise IngestError(str(exc)) from exc
        if not isinstance(payload, Mapping):
            raise IngestError(f"GitHub fixture must be a JSON object: {source}")
        state_scope = canonical_text(payload.get("state_scope", "")).casefold() or None
        if state_scope not in {None, "open", "all"}:
            raise IngestError(f"GitHub fixture {source} has invalid state_scope {state_scope!r}")
        return GitHubFixtureData(
            milestones=_fixture_collection(payload, "milestones", source=source),
            issues=_fixture_collection(payload, "issues", source=source),
            repository=canonical_text(payload.get("repository", "")) or None,
            retrieved_at=canonical_text(payload.get("retrieved_at", "")) or None,
            state_scope=state_scope,
            api_version=canonical_text(payload.get("api_version", "")) or None,
        )
    if not source.exists():
        raise IngestError(f"GitHub fixture does not exist: {source}")
    if not source.is_dir():
        raise IngestError(f"GitHub fixture is not a file or directory: {source}")

    def load_pages(prefix: str) -> list[dict[str, Any]]:
        files = sorted(source.glob(f"{prefix}*.json"), key=lambda item: item.name)
        if not files:
            raise IngestError(f"GitHub fixture directory has no {prefix}*.json files: {source}")
        result: list[dict[str, Any]] = []
        for fixture_file in files:
            try:
                payload = read_json(fixture_file)
            except ArtifactIOError as exc:
                raise IngestError(str(exc)) from exc
            if isinstance(payload, Mapping):
                payload = payload.get(prefix)
            if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
                raise IngestError(f"fixture page must be a list of objects: {fixture_file}")
            result.extend(payload)
        return result

    return GitHubFixtureData(
        milestones=load_pages("milestones"),
        issues=load_pages("issues"),
    )


def load_github_fixture(
    path: str | os.PathLike[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a combined fixture JSON or paginated fixture files."""

    fixture = _load_github_fixture_data(path)
    return fixture.milestones, fixture.issues


def fetch_github_roadmap(
    repository: str,
    *,
    client: GitHubClient,
    state: str = "open",
    as_of: datetime | None = None,
) -> list[RoadmapItem]:
    milestones, issues = client.fetch_repository(repository, state=state)
    return normalize_github_roadmap(
        repository,
        milestones,
        issues,
        as_of=as_of,
    )


def run_ingestion(
    *,
    reviews_csv: str | os.PathLike[str],
    package_name: str,
    repository: str,
    out_dir: str | os.PathLike[str],
    github_fixture: str | os.PathLike[str] | None = None,
    offline: bool = False,
    github_token: str | None = None,
    github_state: str | None = None,
    as_of: datetime | None = None,
    session: requests.Session | None = None,
) -> IngestArtifacts:
    """Run both ingesters and atomically commit canonical JSON arrays."""

    reviews = ingest_reviews_csv(
        reviews_csv,
        package_name,
        session=session,
        offline=offline,
    )
    if github_fixture is not None:
        fixture = _load_github_fixture_data(github_fixture)
        milestones, issues = fixture.milestones, fixture.issues
        if fixture.repository and fixture.repository.casefold() != repository.casefold():
            raise IngestError(
                "GitHub fixture repository does not match the requested repository: "
                f"{fixture.repository!r} != {repository!r}"
            )
        roadmap = normalize_github_roadmap(
            repository,
            milestones,
            issues,
            as_of=as_of,
        )
        effective_state = fixture.state_scope or "fixture-provided"
        github_mode = "fixture"
        source_retrieved_at = fixture.retrieved_at
        source_api_version = fixture.api_version or GITHUB_API_VERSION
        raw_snapshot: dict[str, Any] | None = None
    else:
        if offline:
            raise IngestError("offline mode requires --github-fixture; no network request was made")
        effective_state = github_state or ("all" if github_token else "open")
        client = GitHubClient(token=github_token, session=session)
        milestones, issues = client.fetch_repository(repository, state=effective_state)
        source_retrieved_at = _utc_now().isoformat()
        source_api_version = GITHUB_API_VERSION
        roadmap = normalize_github_roadmap(
            repository,
            milestones,
            issues,
            as_of=as_of,
        )
        github_mode = "live"
        raw_snapshot = {
            "schema_version": 1,
            "repository": repository,
            "retrieved_at": source_retrieved_at,
            "api_version": GITHUB_API_VERSION,
            "state_scope": effective_state,
            "milestones": milestones,
            "issues": issues,
        }

    destination = Path(out_dir)
    processed_at = _utc_now().isoformat()
    scope_metadata = {
        "schema_version": 1,
        "package_name": package_name,
        "repository": repository,
        "reviews": {
            "input": _safe_source_reference(reviews_csv),
            "matching_rows": reviews.stats.matching_rows,
            "emitted_signals": reviews.stats.emitted_signals,
            "duplicates_removed": reviews.stats.duplicates_removed,
        },
        "github": {
            "mode": github_mode,
            "state_scope": effective_state,
            "authenticated": bool(github_token) if github_mode == "live" else None,
            "api_version": source_api_version,
            "retrieved_at": source_retrieved_at,
            "processed_at": processed_at,
            "pull_requests_dropped": True,
            "pull_requests_dropped_count": sum(
                1 for issue in issues if isinstance(issue, Mapping) and "pull_request" in issue
            ),
            "roadmap_items": len(roadmap),
            "raw_snapshot": "github_raw.json" if raw_snapshot is not None else None,
        },
        "priority_as_of": as_of.isoformat() if as_of is not None else None,
    }
    atomic_write_json(destination / "signals.json", reviews.signals)
    atomic_write_json(destination / "roadmap.json", roadmap)
    atomic_write_json(destination / "ingest_scope.json", scope_metadata)
    if raw_snapshot is not None:
        atomic_write_json(destination / "github_raw.json", raw_snapshot)
    return IngestArtifacts(
        signals=reviews.signals,
        roadmap=roadmap,
        review_stats=reviews.stats,
        scope_metadata=scope_metadata,
    )


def _resolve_app(args: argparse.Namespace) -> tuple[str, str]:
    config = APP_CONFIGS.get(args.app) if args.app else None
    package_name = args.package_name or (config.package_name if config else None)
    repository = args.repo or (config.repository if config else None)
    if not package_name or not repository:
        raise IngestError("choose --app or provide both --package-name and --repo")
    return package_name, repository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build canonical signals.json and roadmap.json artifacts."
    )
    parser.add_argument("--app", choices=sorted(APP_CONFIGS))
    parser.add_argument("--package-name")
    parser.add_argument("--repo", help="GitHub repository in owner/name form")
    parser.add_argument("--reviews-csv", required=True, help="Local path or HTTPS URL")
    parser.add_argument("--out-dir", default="out")
    parser.add_argument(
        "--github-fixture",
        help="Combined fixture JSON or fixture-page directory",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid all network access; requires --github-fixture and local reviews",
    )
    parser.add_argument(
        "--github-token-env",
        default="GITHUB_TOKEN",
        help="Environment variable containing a GitHub token",
    )
    parser.add_argument(
        "--github-state",
        choices=("open", "all"),
        help=(
            "GitHub history scope (default: all with a token, open without one; "
            "all explicitly requires authentication)"
        ),
    )
    parser.add_argument(
        "--as-of",
        help="UTC/ISO timestamp for reproducible age/due priority rules",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        package_name, repository = _resolve_app(args)
        as_of = _parse_datetime(args.as_of, context="--as-of") if args.as_of else None
        artifacts = run_ingestion(
            reviews_csv=args.reviews_csv,
            package_name=package_name,
            repository=repository,
            out_dir=args.out_dir,
            github_fixture=args.github_fixture,
            offline=args.offline,
            github_token=os.environ.get(args.github_token_env),
            github_state=args.github_state,
            as_of=as_of,
        )
    except (IngestError, ArtifactIOError) as exc:
        parser.exit(2, f"ingest error: {exc}\n")

    stats = artifacts.review_stats
    print(
        f"wrote {len(artifacts.signals)} signals and "
        f"{len(artifacts.roadmap)} roadmap items to {Path(args.out_dir)}"
    )
    print(
        "reviews: "
        f"{stats.matching_rows}/{stats.total_rows} package rows, "
        f"{stats.duplicates_removed} duplicates removed, "
        f"{stats.blank_text_rows} blank and {stats.invalid_rows} invalid skipped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
