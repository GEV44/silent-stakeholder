"""Deterministic IDs and crash-safe JSON artifact I/O."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unicodedata
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ArtifactIOError(RuntimeError):
    """Raised when a pipeline artifact cannot be read or written safely."""


def canonical_text(value: Any) -> str:
    """Return a stable Unicode/whitespace representation for hashing and dedup."""

    normalized = unicodedata.normalize("NFKC", str(value))
    return " ".join(normalized.split()).strip()


def make_stable_id(prefix: str, *identity_parts: Any) -> str:
    """Build a compact stable ID from an item's immutable logical identity.

    Hash-derived IDs do not change when input rows are reordered or when new
    records are added. The prefix keeps IDs human-auditable at evidence
    boundaries (``S`` signal, ``R`` roadmap, ``N`` need, ``G`` gap).
    """

    if prefix not in {"S", "R", "N", "G"}:
        raise ValueError("prefix must be one of S, R, N, or G")
    if not identity_parts:
        raise ValueError("at least one identity part is required")

    canonical_parts = [canonical_text(part).casefold() for part in identity_parts]
    if not any(canonical_parts):
        raise ValueError("identity parts cannot all be empty")
    payload = "\x1f".join(canonical_parts).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()[:12]


def assert_unique_ids(items: list[Any]) -> None:
    """Raise a clear error if separately-normalized records collide."""

    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        item_id = item.id if hasattr(item, "id") else item["id"]
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)
    if duplicates:
        raise ArtifactIOError("duplicate stable IDs detected: " + ", ".join(sorted(duplicates)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactIOError("JSON artifacts cannot contain NaN or Infinity")
    return value


def atomic_write_json(
    path: str | os.PathLike[str],
    data: Any,
    *,
    indent: int = 2,
) -> Path:
    """Serialize JSON to a same-directory temp file, then atomically replace.

    A process interruption can leave the old artifact or the complete new
    artifact, never a half-written JSON file.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        payload = json.dumps(
            _jsonable(data),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactIOError(f"could not write JSON artifact {destination}: {exc}") from exc
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read a UTF-8 JSON artifact with path and parse context in failures."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ArtifactIOError(f"JSON artifact does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactIOError(
            f"invalid JSON in {source} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ArtifactIOError(f"could not read JSON artifact {source}: {exc}") from exc
