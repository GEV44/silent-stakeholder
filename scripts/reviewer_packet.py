"""Export a blind reviewer packet and validate adjudicated labels on return."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GIT_EXECUTABLE = shutil.which("git")
WORKLIST = ROOT / "eval" / "labeling_worklist.json"
BLIND = ROOT / "eval" / "misunderstood_adjudication_blind.json"
INSTRUCTIONS = ROOT / "docs" / "LABELING_INSTRUCTIONS.md"
PACKET_FILES = {
    "labeling_worklist.json",
    "misunderstood_adjudication_blind.json",
    "LABELING_INSTRUCTIONS.md",
    "PACKET.txt",
}
VERDICTS = {"IGNORED", "UNDER-PRIORITIZED", "MISUNDERSTOOD", "COVERED"}
MATCHES = {"none", "partial", "material"}
FORBIDDEN_KEYS = {
    "_model_verdict",
    "model_verdict",
    "_model_confidence",
    "model_confidence",
    "calibrated_confidence",
    "confidence",
    "similarity",
    "symptom_similarity",
    "latent_similarity",
    "_pair_kind",
    "evidence",
    "quotes",
    "signal_text",
    "review_text",
    "signals",
}
FORBIDDEN_KEY_PARTS = {"signal", "quote", "evidence", "raw_review", "confidence", "similarity"}
V1_TOP_KEYS = {"schema_version", "instructions", "pairs"}
V2_TOP_KEYS = {"schema_version", "instructions", "distractor_policy", "run", "pairs"}
V1_PAIR_KEYS = {
    "pair_id",
    "need_id",
    "latent_need",
    "jtbd",
    "symptom",
    "roadmap_id",
    "roadmap_title",
    "roadmap_body",
    "roadmap_labels",
    "roadmap_milestone",
    "roadmap_priority",
    "reviewer1",
    "reviewer2",
    "adjudicated_verdict",
    "adjudicated_notes",
}
V2_PAIR_KEYS = {
    "pair_id",
    "need_id",
    "latent_need",
    "jtbd",
    "symptom",
    "roadmap_id",
    "roadmap_title",
    "roadmap_body",
    "roadmap_priority",
    "roadmap_milestone",
    "reviewer1",
    "reviewer2",
    "adjudicated_verdict",
    "adjudicated_notes",
}
V1_REVIEW_KEYS = {
    "need_supported",
    "public_artifact_match",
    "public_claim_defensible",
    "verdict",
    "notes",
}
V2_REVIEW_KEYS = {"coverage", "verdict", "notes"}
LABEL_KEYS = {
    "gap_id",
    "need_id",
    "roadmap_id",
    "need_supported",
    "public_artifact_match",
    "public_claim_defensible",
    "verdict",
    "adjudicated",
    "notes",
}
PROTECTED_NAMES = {
    "wordpress_misunderstood_review.json",
    "demo_fixture_misunderstood_review.json",
    "misunderstood_review.md",
}
MAX_JSON_FILE_BYTES = 64 * 1024 * 1024
MAX_TEXT_FILE_BYTES = 4 * 1024 * 1024
MAX_JSON_NODES = 1_000_000
MAX_JSON_DEPTH = 64
MAX_JSON_STRING_CHARS = 1_000_000


class PacketError(Exception):
    pass


def _warn_v2_calibration_limit() -> None:
    print(
        "WARNING: WORKLIST SCHEMA 2.0 DOES NOT COLLECT need_supported OR "
        "public_claim_defensible; THIS PACKET ALONE CANNOT SUBSTANTIATE "
        "CALIBRATION READINESS, REGARDLESS OF LABEL COUNT.",
        file=sys.stderr,
    )


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PacketError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    _reject_protected_path(path)
    try:
        document = json.loads(
            _bounded_text(path, limit=MAX_JSON_FILE_BYTES),
            object_pairs_hook=_reject_duplicates,
        )
        _validate_json_complexity(document)
        return document
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PacketError(f"cannot read JSON {path}: {exc}") from exc


def _bounded_text(path: Path, *, limit: int) -> str:
    if path.is_symlink():
        raise PacketError(f"refusing symlink input: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PacketError(f"cannot inspect {path}: {exc}") from exc
    if size > limit:
        raise PacketError(f"input exceeds the {limit}-byte safety limit: {path.name}")
    return path.read_text(encoding="utf-8")


def _validate_json_complexity(document: Any) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(document, 1)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise PacketError("JSON exceeds the node-count safety limit")
        if depth > MAX_JSON_DEPTH:
            raise PacketError("JSON exceeds the nesting-depth safety limit")
        if isinstance(value, str) and len(value) > MAX_JSON_STRING_CHARS:
            raise PacketError("JSON contains an oversized string")
        if isinstance(value, dict):
            for key, item in value.items():
                if len(str(key)) > MAX_JSON_STRING_CHARS:
                    raise PacketError("JSON contains an oversized object key")
                stack.append((item, depth + 1))
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)


def _reject_protected_path(path: Path) -> None:
    lowered = {part.casefold() for part in path.resolve().parts}
    if any("_answer_key" in part for part in lowered) or lowered & PROTECTED_NAMES:
        raise PacketError(f"refusing protected answer-key/pre-review path: {path}")


def sha256(path: Path) -> str:
    if path.is_symlink():
        raise PacketError(f"refusing symlink input: {path}")
    try:
        if path.stat().st_size > MAX_JSON_FILE_BYTES:
            raise PacketError(
                f"input exceeds the {MAX_JSON_FILE_BYTES}-byte safety limit: {path.name}"
            )
    except OSError as exc:
        raise PacketError(f"cannot inspect {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PacketError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _immutable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _immutable(item)
            for key, item in value.items()
            if key not in {"reviewer1", "reviewer2"} and not key.startswith("adjudicated_")
        }
    if isinstance(value, list):
        return [_immutable(item) for item in value]
    return value


def _check_forbidden(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            folded = key.casefold()
            if folded in FORBIDDEN_KEYS or any(part in folded for part in FORBIDDEN_KEY_PARTS):
                raise PacketError(f"prohibited field {location}.{key}")
            _check_forbidden(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_forbidden(item, f"{location}[{index}]")


def validate_worklist(doc: Any, *, require_blank: bool) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(doc, dict):
        raise PacketError("worklist must be an object")
    schema = doc.get("schema_version")
    if schema == "1.0":
        top_keys = V1_TOP_KEYS
        pair_keys = V1_PAIR_KEYS
        review_keys = V1_REVIEW_KEYS
    elif schema == "2.0":
        top_keys = V2_TOP_KEYS
        pair_keys = V2_PAIR_KEYS
        review_keys = V2_REVIEW_KEYS
    else:
        raise PacketError("worklist schema_version must be 1.0 or 2.0")
    if set(doc) != top_keys:
        raise PacketError(f"worklist schema {schema} has missing or unexpected top-level fields")
    if not isinstance(doc["pairs"], list) or not doc["pairs"]:
        raise PacketError("worklist pairs must be a non-empty list")
    seen_pair_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for index, pair in enumerate(doc["pairs"]):
        where = f"pairs[{index}]"
        if not isinstance(pair, dict) or set(pair) != pair_keys:
            raise PacketError(f"{where} has missing or unexpected fields")
        identity = (str(pair["pair_id"]), str(pair["need_id"]), str(pair["roadmap_id"]))
        if (
            not re.fullmatch(r"P\d+", identity[0])
            or not re.fullmatch(r"N\w+", identity[1])
            or not re.fullmatch(r"R\w+", identity[2])
        ):
            raise PacketError(f"{where} has invalid pair/need/roadmap IDs")
        pair_key = (identity[1], identity[2])
        if identity[0] in seen_pair_ids:
            raise PacketError(f"duplicate worklist pair_id: {identity[0]}")
        if pair_key in seen_pairs:
            raise PacketError(f"duplicate worklist need/roadmap pair: {pair_key}")
        seen_pair_ids.add(identity[0])
        seen_pairs.add(pair_key)
        for reviewer in ("reviewer1", "reviewer2"):
            row = pair[reviewer]
            if not isinstance(row, dict) or set(row) != review_keys:
                raise PacketError(f"{where}.{reviewer} has missing or unexpected fields")
            if require_blank and any(value not in (None, "") for value in row.values()):
                raise PacketError(
                    f"{where}.{reviewer} is pre-filled; export requires a blind sheet"
                )
            if schema == "1.0":
                if any(
                    row[field] is not None and type(row[field]) is not bool
                    for field in ("need_supported", "public_claim_defensible")
                ):
                    raise PacketError(f"{where}.{reviewer} has an invalid boolean")
                if row["public_artifact_match"] not in MATCHES | {None}:
                    raise PacketError(f"{where}.{reviewer} has an invalid public_artifact_match")
            elif row["coverage"] not in MATCHES | {None}:
                raise PacketError(f"{where}.{reviewer} has an invalid coverage")
            if row["verdict"] not in VERDICTS | {None} or not isinstance(row["notes"], str):
                raise PacketError(f"{where}.{reviewer} has an invalid verdict or notes value")
        if pair["adjudicated_verdict"] not in VERDICTS | {None} or not isinstance(
            pair["adjudicated_notes"], str
        ):
            raise PacketError(f"{where} has an invalid adjudication value")
        if require_blank and any(
            pair[key] not in (None, "") for key in ("adjudicated_verdict", "adjudicated_notes")
        ):
            raise PacketError(f"{where} contains pre-filled adjudication")
    _check_forbidden(doc)
    return schema, doc["pairs"]


def _validate_blind(value: Any, *, require_blank: bool, location: str = "$") -> None:
    if isinstance(value, dict):
        if value.get("adjudicated_answer") is not None:
            answers = [
                value.get(reviewer, {}).get("answer")
                if isinstance(value.get(reviewer), dict)
                else None
                for reviewer in ("reviewer1", "reviewer2")
            ]
            if any(answer not in ("job", "symptom_only", "neither") for answer in answers):
                raise PacketError(f"{location}.adjudicated_answer requires both reviewer answers")
        for key, item in value.items():
            if key in {"reviewer1", "reviewer2"} and isinstance(item, dict):
                answer = item.get("answer")
                if (
                    set(item) != {"answer", "notes"}
                    or answer not in (None, "job", "symptom_only", "neither")
                    or not isinstance(item.get("notes"), str)
                ):
                    raise PacketError(f"{location}.{key} has invalid fields")
                if require_blank and any(field not in (None, "") for field in item.values()):
                    raise PacketError(f"{location}.{key} is pre-filled")
            elif key == "adjudicated_answer" and item not in (
                None,
                "job",
                "symptom_only",
                "neither",
            ):
                raise PacketError(f"{location}.{key} has an invalid answer")
            elif key == "adjudicated_notes" and not isinstance(item, str):
                raise PacketError(f"{location}.{key} must be a string")
            if require_blank and key.startswith("adjudicated_") and item not in (None, ""):
                raise PacketError(f"{location}.{key} is pre-filled")
            _validate_blind(item, require_blank=require_blank, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_blind(item, require_blank=require_blank, location=f"{location}[{index}]")


def blind_counts(value: Any) -> tuple[int, int, int]:
    total = reviewer_complete = adjudicated = 0
    if isinstance(value, dict):
        if "reviewer1" in value and "reviewer2" in value:
            total = 1
            answers = [
                value.get(reviewer, {}).get("answer")
                if isinstance(value.get(reviewer), dict)
                else None
                for reviewer in ("reviewer1", "reviewer2")
            ]
            reviewer_complete = int(
                all(answer in ("job", "symptom_only", "neither") for answer in answers)
            )
            adjudicated = int(value.get("adjudicated_answer") in ("job", "symptom_only", "neither"))
        for item in value.values():
            child = blind_counts(item)
            total += child[0]
            reviewer_complete += child[1]
            adjudicated += child[2]
    elif isinstance(value, list):
        for item in value:
            child = blind_counts(item)
            total += child[0]
            reviewer_complete += child[1]
            adjudicated += child[2]
    return total, reviewer_complete, adjudicated


def git(*args: str) -> str:
    # Arguments come only from fixed call sites below; no shell is involved.
    if GIT_EXECUTABLE is None:
        raise PacketError("git executable is unavailable")
    result = subprocess.run(  # noqa: S603 -- resolved executable, internal argv
        [GIT_EXECUTABLE, *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise PacketError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def safe_destination(path: Path) -> Path:
    destination = path.resolve()
    if destination.exists() and not destination.is_dir():
        raise PacketError(f"destination is not a directory: {destination}")
    try:
        relative = destination.relative_to(ROOT)
    except ValueError:
        return destination
    if GIT_EXECUTABLE is None:
        raise PacketError("git executable is unavailable")
    ignored = subprocess.run(  # noqa: S603 -- resolved executable, fixed argv
        [GIT_EXECUTABLE, "check-ignore", "-q", "--", str(relative)],
        cwd=ROOT,
        check=False,
    )
    if ignored.returncode != 0:
        raise PacketError("destination inside the repository must be demonstrably gitignored")
    return destination


def _run_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    paths = tuple(
        getattr(args, name).resolve() for name in ("gaps", "needs", "roadmap", "run_manifest")
    )
    if len({path.parent for path in paths}) != 1:
        raise PacketError("gaps, needs, roadmap, and run-manifest must come from one run directory")
    return paths


def run_identity(gaps: Path, needs: Path, roadmap: Path, manifest: Path) -> dict[str, str]:
    return {
        "gaps_sha256": sha256(gaps),
        "needs_sha256": sha256(needs),
        "roadmap_sha256": sha256(roadmap),
        "run_manifest_sha256": sha256(manifest),
    }


def packet_identity(
    worklist: Any,
    blind: Any,
    gaps: Path,
    needs: Path,
    roadmap: Path,
    manifest: Path,
    instructions: Path,
) -> dict[str, str]:
    return {
        "source_commit": git("rev-parse", "HEAD"),
        **run_identity(gaps, needs, roadmap, manifest),
        "worklist_identity": _canonical_hash(_immutable(worklist)),
        "blind_identity": _canonical_hash(_immutable(blind)),
        "instructions_sha256": sha256(instructions),
    }


def parse_packet(path: Path) -> dict[str, str]:
    try:
        lines = _bounded_text(path, limit=MAX_TEXT_FILE_BYTES).splitlines()
    except OSError as exc:
        raise PacketError(f"cannot read {path}: {exc}") from exc
    if len(lines) != 1:
        raise PacketError("PACKET.txt must contain exactly one line")
    parts = lines[0].split()
    if not parts or parts[0] != "FIRECODE_REVIEW_PACKET_V1":
        raise PacketError("unsupported PACKET.txt format")
    values: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise PacketError("malformed PACKET.txt field")
        key, value = part.split("=", 1)
        if key in values:
            raise PacketError(f"duplicate PACKET.txt field: {key}")
        values[key] = value
    expected = {
        "source_commit",
        "gaps_sha256",
        "needs_sha256",
        "roadmap_sha256",
        "run_manifest_sha256",
        "worklist_identity",
        "blind_identity",
        "instructions_sha256",
    }
    if set(values) != expected or any(
        not re.fullmatch(r"[0-9a-f]{40,64}", value) for value in values.values()
    ):
        raise PacketError("PACKET.txt has missing or invalid identity fields")
    return values


def validate_labels(doc: Any) -> list[dict[str, Any]]:
    if not isinstance(doc, dict) or set(doc) != {"schema_version", "reviewers", "labels"}:
        raise PacketError("labels must contain only schema_version, reviewers, and labels")
    reviewers, labels = doc["reviewers"], doc["labels"]
    if (
        doc["schema_version"] != "1.0"
        or not isinstance(reviewers, list)
        or not all(isinstance(name, str) and name.strip() for name in reviewers)
        or len(set(reviewers)) != len(reviewers)
        or len(reviewers) < 2
    ):
        raise PacketError("labels require schema_version 1.0 and at least two unique reviewers")
    if not labels or not isinstance(labels, list):
        raise PacketError("labels must be a non-empty array")
    for index, row in enumerate(labels):
        required = LABEL_KEYS - {"gap_id", "verdict", "notes"}
        if not isinstance(row, dict) or not required <= set(row) <= LABEL_KEYS:
            raise PacketError(f"labels[{index}] has missing or unexpected fields")
        if not re.fullmatch(r"N\w+", str(row["need_id"])) or not re.fullmatch(
            r"R\w+", str(row["roadmap_id"])
        ):
            raise PacketError(f"labels[{index}] has invalid IDs")
        if "gap_id" in row and not re.fullmatch(r"G\w+", str(row["gap_id"])):
            raise PacketError(f"labels[{index}].gap_id is invalid")
        if (
            type(row["need_supported"]) is not bool
            or type(row["public_claim_defensible"]) is not bool
            or type(row["adjudicated"]) is not bool
        ):
            raise PacketError(f"labels[{index}] boolean fields must be true or false")
        if row["public_artifact_match"] not in MATCHES or (
            "verdict" in row and row["verdict"] not in VERDICTS
        ):
            raise PacketError(f"labels[{index}] has an invalid enum value")
        if "notes" in row and not isinstance(row["notes"], str):
            raise PacketError(f"labels[{index}].notes must be a string")
    return labels


def artifact_rows(path: Path, *keys: str) -> list[dict[str, Any]]:
    doc = load_json(path)
    rows: Any = doc
    if isinstance(doc, dict):
        rows = next((doc[key] for key in keys if isinstance(doc.get(key), list)), None)
    if not isinstance(rows, list) or not rows:
        raise PacketError(f"{path.name} must be an array or contain one of: {', '.join(keys)}")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PacketError(f"{path.name}[{index}] must be an object")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise PacketError(f"{path.name}[{index}] has a missing or invalid id")
        if identifier in seen:
            raise PacketError(f"{path.name} contains duplicate id: {identifier}")
        seen.add(identifier)
    return rows


def _index(rows: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get("id") or "")
        if not identifier or identifier in result:
            raise PacketError(f"{kind} artifact contains a missing or duplicate ID: {identifier!r}")
        result[identifier] = row
    return result


def _roadmap_priority(item: dict[str, Any]) -> str:
    priority = item.get("priority")
    if isinstance(priority, dict):
        return str(priority.get("tier") or "unspecified")
    return str(priority or "unspecified")


def _roadmap_milestone(item: dict[str, Any]) -> str:
    milestone = item.get("milestone")
    if isinstance(milestone, dict):
        milestone = milestone.get("title")
    return str(milestone or "(none)")


def worklist_binding(
    pairs: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    needs: list[dict[str, Any]],
    roadmap: list[dict[str, Any]],
) -> tuple[set[tuple[str, str]], int]:
    needs_by_id, roadmap_by_id = _index(needs, "needs"), _index(roadmap, "roadmap")
    need_ids = {str(row.get("need_id")) for row in gaps}
    gap_pairs = {(str(row.get("need_id")), str(row.get("matched_roadmap_id"))) for row in gaps}
    stale = sorted({str(pair["need_id"]) for pair in pairs} - need_ids)
    if stale:
        raise PacketError("worklist need IDs do not belong to this run: " + ", ".join(stale))
    for pair in pairs:
        need_id, roadmap_id = str(pair["need_id"]), str(pair["roadmap_id"])
        need, item = needs_by_id.get(need_id), roadmap_by_id.get(roadmap_id)
        if need is None:
            raise PacketError(f"worklist need_id is absent from needs.json: {need_id}")
        if item is None:
            raise PacketError(f"worklist roadmap_id is absent from roadmap.json: {roadmap_id}")
        expected_need = {
            "latent_need": str(need.get("latent_need") or ""),
            "jtbd": str(need.get("jtbd_statement") or need.get("jtbd") or ""),
            "symptom": str(need.get("symptom") or ""),
        }
        for field, expected in expected_need.items():
            if str(pair[field]) != expected:
                raise PacketError(f"worklist {need_id} field {field} drifted from needs.json")
        body, embedded_body = str(item.get("body") or ""), str(pair["roadmap_body"])
        expected_roadmap: dict[str, Any] = {
            "roadmap_title": str(item.get("title") or ""),
            "roadmap_labels": item.get("labels") or [],
            "roadmap_milestone": _roadmap_milestone(item),
            "roadmap_priority": _roadmap_priority(item),
        }
        for field, expected in expected_roadmap.items():
            if field in pair and pair[field] != expected:
                raise PacketError(f"worklist {roadmap_id} field {field} drifted from roadmap.json")
        if embedded_body != body[:400]:
            raise PacketError(f"worklist {roadmap_id} roadmap_body drifted from roadmap.json")
    worklist_pairs = {(str(pair["need_id"]), str(pair["roadmap_id"])) for pair in pairs}
    matched = worklist_pairs & gap_pairs
    if not matched:
        raise PacketError("worklist has zero matched pairs in this run; refusing stale export")
    return matched, len(pairs) - len(matched)


def command_export(args: argparse.Namespace) -> int:
    gaps_path, needs_path, roadmap_path, manifest_path = _run_paths(args)
    worklist, blind = load_json(args.worklist), load_json(args.blind)
    _reject_protected_path(args.instructions)
    schema, pairs = validate_worklist(worklist, require_blank=True)
    _check_forbidden(blind)
    _validate_blind(blind, require_blank=True)
    gaps = artifact_rows(gaps_path, "gaps")
    needs = artifact_rows(needs_path, "needs", "latent_needs")
    roadmap = artifact_rows(roadmap_path, "roadmap")
    matched_pairs, unmatched = worklist_binding(pairs, gaps, needs, roadmap)
    identity = packet_identity(
        worklist,
        blind,
        gaps_path,
        needs_path,
        roadmap_path,
        manifest_path,
        args.instructions,
    )
    destination = safe_destination(args.destination)
    if destination.exists():
        raise PacketError(f"atomic export requires a new destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{destination.name}.tmp-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=destination.parent) as temporary:
        staging = Path(temporary)
        shutil.copyfile(args.worklist, staging / "labeling_worklist.json")
        shutil.copyfile(args.blind, staging / "misunderstood_adjudication_blind.json")
        shutil.copyfile(args.instructions, staging / "LABELING_INSTRUCTIONS.md")
        line = (
            "FIRECODE_REVIEW_PACKET_V1 "
            + " ".join(f"{key}={value}" for key, value in identity.items())
            + "\n"
        )
        (staging / "PACKET.txt").write_text(line, encoding="utf-8", newline="\n")
        if {path.name for path in staging.iterdir()} != PACKET_FILES:
            raise PacketError("packet staging file set is not exact")
        staging.rename(destination)
    print(
        f"exported blind reviewer packet to {destination}; "
        f"worklist matched={len(matched_pairs)} unmatched={unmatched}"
    )
    if schema == "2.0":
        _warn_v2_calibration_limit()
    return 0


def command_validate(args: argparse.Namespace) -> int:
    gaps_path, needs_path, roadmap_path, manifest_path = _run_paths(args)
    packet = args.packet.resolve()
    if packet.is_symlink() or not packet.is_dir():
        raise PacketError(f"returned packet directory does not exist: {packet}")
    entries = list(packet.iterdir())
    if {path.name for path in entries} != PACKET_FILES:
        raise PacketError("returned packet must contain exactly the four exported files")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise PacketError("returned packet entries must be regular files, not symlinks")
    returned_worklist = load_json(packet / "labeling_worklist.json")
    returned_blind = load_json(packet / "misunderstood_adjudication_blind.json")
    schema, _ = validate_worklist(returned_worklist, require_blank=False)
    _check_forbidden(returned_blind)
    _validate_blind(returned_blind, require_blank=False)
    blind_total, blind_ready, blind_adjudicated = blind_counts(returned_blind)
    print(
        f"blind_rows={blind_total} reviewer_complete={blind_ready} adjudicated={blind_adjudicated}"
    )
    recorded = parse_packet(packet / "PACKET.txt")
    current = run_identity(gaps_path, needs_path, roadmap_path, manifest_path)
    returned = {
        "worklist_identity": _canonical_hash(_immutable(returned_worklist)),
        "blind_identity": _canonical_hash(_immutable(returned_blind)),
        "instructions_sha256": sha256(packet / "LABELING_INSTRUCTIONS.md"),
    }
    mismatches = [key for key, value in current.items() if recorded.get(key) != value]
    mismatches += [key for key, value in returned.items() if recorded.get(key) != value]
    current_commit = git("rev-parse", "HEAD")
    if current_commit != recorded["source_commit"] and not mismatches:
        print(
            f"WARNING: packet source_commit={recorded['source_commit']} differs from "
            f"current HEAD={current_commit}; exact packet/run hashes still govern validation",
            file=sys.stderr,
        )
    if mismatches and not args.salvage_report:
        raise PacketError("identity mismatch: " + ", ".join(sorted(set(mismatches))))
    labels = validate_labels(load_json(args.labels))
    gaps = artifact_rows(gaps_path, "gaps")
    needs = artifact_rows(needs_path, "needs", "latent_needs")
    roadmap = artifact_rows(roadmap_path, "roadmap")
    expected_pairs: set[tuple[str, str]] = set()
    if not mismatches:
        expected_pairs, _ = worklist_binding(returned_worklist["pairs"], gaps, needs, roadmap)
    by_gap = {str(row.get("id")): row for row in gaps}
    by_pair = {(str(row.get("need_id")), str(row.get("matched_roadmap_id"))): row for row in gaps}
    packet_pairs = {
        (str(row["need_id"]), str(row["roadmap_id"])): row for row in returned_worklist["pairs"]
    }
    labels_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    matched: list[tuple[str, str]] = []
    unmatched: list[tuple[str, str]] = []
    alien: list[tuple[str, str]] = []
    unadjudicated: list[tuple[str, str]] = []
    invalid_gap_ids: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in labels:
        pair = (str(row["need_id"]), str(row["roadmap_id"]))
        if pair in seen:
            raise PacketError(f"duplicate returned label pair: {pair}")
        seen.add(pair)
        labels_by_pair[pair] = row
        if pair not in packet_pairs:
            alien.append(pair)
        if "gap_id" in row:
            match = by_gap.get(str(row["gap_id"]))
            if (
                match is None
                or (str(match.get("need_id")), str(match.get("matched_roadmap_id"))) != pair
            ):
                invalid_gap_ids.append(str(row["gap_id"]))
                match = None
        else:
            match = by_pair.get(pair)
        (matched if match else unmatched).append(pair)
        if row["adjudicated"] is not True:
            unadjudicated.append(pair)
    matched_pairs = set(matched)
    missing_expected = expected_pairs - matched_pairs
    unexpected_pairs = seen - expected_pairs if not mismatches else set()
    protocol_errors: list[str] = []
    decision_fields = (
        ("need_supported", "public_artifact_match", "public_claim_defensible", "verdict")
        if schema == "1.0"
        else ("coverage", "verdict")
    )
    for pair in sorted(expected_pairs | seen):
        worklist_row = packet_pairs.get(pair)
        if worklist_row is None:
            continue
        for reviewer in ("reviewer1", "reviewer2"):
            missing = [field for field in decision_fields if worklist_row[reviewer][field] is None]
            if missing:
                protocol_errors.append(f"{pair} {reviewer} missing {','.join(missing)}")
        adjudicated_verdict = worklist_row["adjudicated_verdict"]
        if adjudicated_verdict is None:
            protocol_errors.append(f"{pair} missing adjudicated_verdict")
        label = labels_by_pair.get(pair)
        if (
            label is not None
            and label.get("verdict") is not None
            and label["verdict"] != adjudicated_verdict
        ):
            protocol_errors.append(
                f"{pair} label verdict {label['verdict']} != {adjudicated_verdict}"
            )
        if schema == "2.0" and label is not None:
            coverage1 = worklist_row["reviewer1"]["coverage"]
            coverage2 = worklist_row["reviewer2"]["coverage"]
            if (
                coverage1 is not None
                and coverage1 == coverage2
                and label["public_artifact_match"] != coverage1
            ):
                protocol_errors.append(
                    f"{pair} label public_artifact_match "
                    f"{label['public_artifact_match']} != agreed reviewer coverage {coverage1}"
                )
    print(
        f"labels={len(labels)} expected={len(expected_pairs)} "
        f"expected_covered={len(expected_pairs & matched_pairs)} "
        f"missing={len(missing_expected)} matched={len(matched)} "
        f"unmatched={len(unmatched)} alien={len(alien)} "
        f"unadjudicated={len(unadjudicated)}"
    )
    outcomes = {
        bool(row["public_claim_defensible"])
        for pair, row in labels_by_pair.items()
        if pair in matched_pairs
    }
    if schema == "2.0":
        _warn_v2_calibration_limit()
    else:
        if len(matched_pairs) < 100:
            print(
                f"WARNING: calibration is not ready: {len(matched_pairs)} matched labels < 100",
                file=sys.stderr,
            )
        if len(outcomes) < 2:
            print(
                "WARNING: calibration is not ready: matched labels do not contain both "
                "public_claim_defensible outcomes",
                file=sys.stderr,
            )
    if args.salvage_report:
        current_titles: dict[tuple[str, str], list[str]] = {}
        for gap in gaps:
            current_titles.setdefault(
                (str(gap.get("latent_need", "")), str(gap.get("matched_roadmap_id", ""))), []
            ).append(str(gap.get("id", "")))
        report = []
        for row in labels:
            source = packet_pairs.get((str(row["need_id"]), str(row["roadmap_id"])))
            key = (str(source.get("latent_need", "")) if source else "", str(row["roadmap_id"]))
            report.append(
                {
                    "need_id": row["need_id"],
                    "roadmap_id": row["roadmap_id"],
                    "candidate_gap_ids": current_titles.get(key, []),
                }
            )
        print(json.dumps({"salvage_report_only": True, "rows": report}, indent=2, sort_keys=True))
    if (
        mismatches
        or missing_expected
        or unexpected_pairs
        or unmatched
        or alien
        or unadjudicated
        or invalid_gap_ids
        or protocol_errors
    ):
        details = []
        if mismatches:
            details.append("identity mismatch: " + ", ".join(sorted(set(mismatches))))
        if unmatched:
            details.append(f"{len(unmatched)} labels do not bind to the current run")
        if alien:
            details.append(f"{len(alien)} labels were not in the exported worklist")
        if unadjudicated:
            details.append(f"{len(unadjudicated)} labels are not human-adjudicated")
        if invalid_gap_ids:
            details.append(
                "supplied gap_id does not exist or bind exactly: " + ", ".join(invalid_gap_ids)
            )
        if missing_expected:
            details.append(f"{len(missing_expected)} expected worklist labels are missing")
        if unexpected_pairs:
            details.append(f"{len(unexpected_pairs)} labels are outside the expected matched set")
        if protocol_errors:
            details.append("review protocol incomplete: " + "; ".join(protocol_errors))
        raise PacketError("; ".join(details))
    print("return validation passed; no files were modified")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="create a four-file blind packet")
    export.add_argument("--destination", type=Path, required=True)
    export.add_argument("--gaps", type=Path, required=True)
    export.add_argument("--needs", type=Path, required=True)
    export.add_argument("--roadmap", type=Path, required=True)
    export.add_argument("--run-manifest", type=Path, required=True)
    export.add_argument("--worklist", type=Path, default=WORKLIST)
    export.add_argument("--blind", type=Path, default=BLIND)
    export.add_argument("--instructions", type=Path, default=INSTRUCTIONS)
    export.set_defaults(func=command_export)
    validate = subparsers.add_parser(
        "validate-return", help="validate labels against an unchanged packet and run"
    )
    validate.add_argument("--packet", type=Path, required=True)
    validate.add_argument("--labels", type=Path, required=True)
    validate.add_argument("--gaps", type=Path, required=True)
    validate.add_argument("--needs", type=Path, required=True)
    validate.add_argument("--roadmap", type=Path, required=True)
    validate.add_argument("--run-manifest", type=Path, required=True)
    validate.add_argument(
        "--salvage-report",
        action="store_true",
        help="print title/roadmap re-key candidates; never mutate",
    )
    validate.set_defaults(func=command_validate)
    return result


def main() -> int:
    try:
        arguments = parser().parse_args()
        return int(arguments.func(arguments))
    except PacketError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
