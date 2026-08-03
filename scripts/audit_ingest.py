#!/usr/bin/env python3
"""Report exact, non-sensitive counts from an ingestion artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


class AuditError(RuntimeError):
    """The artifact set is incomplete or internally inconsistent."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditError(f"missing required artifact: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid JSON in {path.name}: {exc}") from exc


def _object_list(value: Any, *, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise AuditError(f"{name} must be a JSON list of objects")
    return value


def _unique_prefixed_ids(records: list[dict[str, Any]], *, prefix: str, name: str) -> None:
    ids = [item.get("id") for item in records]
    pattern = re.compile(rf"{re.escape(prefix)}(?:[0-9a-f]{{12}}|\d{{4,}})")
    if not all(isinstance(item_id, str) and pattern.fullmatch(item_id) for item_id in ids):
        raise AuditError(f"{name} contains a missing or noncanonical {prefix} ID")
    if len(ids) != len(set(ids)):
        raise AuditError(f"{name} contains duplicate IDs")


def _counter(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(item.get(key, "<missing>")) for item in records)
    return dict(sorted(counts.items()))


def _canonical_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    return " ".join(normalized.split()).strip()


def _record_key(record_type: str, record: dict[str, Any], *, name: str) -> tuple[str, int]:
    try:
        number = int(record["number"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError(f"{name} contains a record with no valid number") from exc
    if number < 1:
        raise AuditError(f"{name} contains non-positive record number {number}")
    return record_type, number


def _raw_projection(record_type: str, record: dict[str, Any], *, name: str) -> dict[str, Any]:
    state_reason = record.get("state_reason")
    return {
        "key": _record_key(record_type, record, name=name),
        "title": _canonical_text(record.get("title", "")),
        "state": _canonical_text(record.get("state", "")).casefold(),
        "state_reason": _canonical_text(state_reason) if state_reason is not None else None,
    }


def audit_ingest(artifact_dir: Path) -> dict[str, Any]:
    signals = _object_list(_read_json(artifact_dir / "signals.json"), name="signals.json")
    roadmap = _object_list(_read_json(artifact_dir / "roadmap.json"), name="roadmap.json")
    scope = _read_json(artifact_dir / "ingest_scope.json")
    if not isinstance(scope, dict):
        raise AuditError("ingest_scope.json must be a JSON object")

    reviews_scope = scope.get("reviews")
    github_scope = scope.get("github")
    if not isinstance(reviews_scope, dict) or not isinstance(github_scope, dict):
        raise AuditError("ingest_scope.json must contain reviews and github objects")

    _unique_prefixed_ids(signals, prefix="S", name="signals.json")
    _unique_prefixed_ids(roadmap, prefix="R", name="roadmap.json")

    if reviews_scope.get("emitted_signals") != len(signals):
        raise AuditError("reviews.emitted_signals does not match signals.json")
    if github_scope.get("roadmap_items") != len(roadmap):
        raise AuditError("github.roadmap_items does not match roadmap.json")

    matching_rows = reviews_scope.get("matching_rows")
    duplicates_removed = reviews_scope.get("duplicates_removed")
    if not isinstance(matching_rows, int) or not isinstance(duplicates_removed, int):
        raise AuditError("review matching and duplicate counts must be integers")
    review_residual = matching_rows - duplicates_removed - len(signals)
    if matching_rows < 0 or duplicates_removed < 0 or review_residual < 0:
        raise AuditError("review counts are arithmetically inconsistent")

    repository = scope.get("repository")
    if not isinstance(repository, str) or not repository.strip():
        raise AuditError("ingest_scope.json has no repository")
    roadmap_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for item in roadmap:
        record_type = item.get("type")
        if record_type not in {"issue", "milestone"}:
            raise AuditError(f"roadmap.json contains invalid type {record_type!r}")
        if item.get("repository") != repository:
            raise AuditError("roadmap.json contains a record from a different repository")
        key = _record_key(record_type, item, name="roadmap.json")
        if key in roadmap_by_key:
            raise AuditError(f"roadmap.json contains duplicate {key[0]} number {key[1]}")
        roadmap_by_key[key] = item

    roadmap_states = {str(item.get("state")) for item in roadmap}
    if not roadmap_states <= {"open", "closed"}:
        raise AuditError("roadmap.json contains a state other than open or closed")
    state_scope = github_scope.get("state_scope")
    if state_scope == "open" and roadmap_states != {"open"}:
        raise AuditError("github.state_scope is open but roadmap.json contains another state")

    missing_priority = 0
    missing_reasons = 0
    low_priority = 0
    explicit_priority = 0
    tiers: Counter[str] = Counter()
    for item in roadmap:
        priority = item.get("priority")
        if not isinstance(priority, dict):
            missing_priority += 1
            continue
        tiers[str(priority.get("tier", "<missing>"))] += 1
        low_priority += int(priority.get("is_low_priority") is True)
        explicit_priority += int(priority.get("has_explicit_priority") is True)
        reasons = priority.get("reasons")
        if not isinstance(reasons, list) or not any(
            isinstance(reason, str) and reason.strip() for reason in reasons
        ):
            missing_reasons += 1
    if missing_priority:
        raise AuditError(f"{missing_priority} roadmap items have no priority object")
    if missing_reasons:
        raise AuditError(f"{missing_reasons} roadmap items have no auditable priority reason")

    raw_snapshot_name = github_scope.get("raw_snapshot")
    raw_check = "not available for fixture/reprocessed input"
    if raw_snapshot_name:
        raw_snapshot_file = Path(str(raw_snapshot_name))
        if raw_snapshot_file.is_absolute() or len(raw_snapshot_file.parts) != 1:
            raise AuditError("github.raw_snapshot must be a filename within the artifact directory")
        raw_path = artifact_dir / raw_snapshot_file
        raw = _read_json(raw_path)
        if not isinstance(raw, dict):
            raise AuditError(f"{raw_path.name} must be a JSON object")
        if raw.get("repository") != repository:
            raise AuditError("raw snapshot repository does not match ingest scope")
        if raw.get("state_scope") != state_scope:
            raise AuditError("raw snapshot state scope does not match ingest scope")
        raw_milestones = _object_list(raw.get("milestones"), name="raw milestones")
        raw_issues = _object_list(raw.get("issues"), name="raw issues")
        raw_prs = sum("pull_request" in issue for issue in raw_issues)
        expected_roadmap = len(raw_milestones) + len(raw_issues) - raw_prs
        if expected_roadmap != len(roadmap):
            raise AuditError(
                "raw milestone + non-PR issue count does not match roadmap.json "
                f"({expected_roadmap} != {len(roadmap)})"
            )
        if github_scope.get("pull_requests_dropped_count") != raw_prs:
            raise AuditError("github.pull_requests_dropped_count does not match raw snapshot")

        raw_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        raw_records = [
            *(_raw_projection("milestone", item, name="raw milestones") for item in raw_milestones),
            *(
                _raw_projection("issue", item, name="raw issues")
                for item in raw_issues
                if "pull_request" not in item
            ),
        ]
        for projection in raw_records:
            key = projection["key"]
            if key in raw_by_key:
                raise AuditError(f"raw snapshot contains duplicate {key[0]} number {key[1]}")
            raw_by_key[key] = projection
        if raw_by_key.keys() != roadmap_by_key.keys():
            missing = sorted(raw_by_key.keys() - roadmap_by_key.keys())[:3]
            extra = sorted(roadmap_by_key.keys() - raw_by_key.keys())[:3]
            raise AuditError(
                "raw and normalized roadmap identities differ; "
                f"missing={missing!r}, extra={extra!r}"
            )
        mismatches: list[tuple[str, int, str]] = []
        for key, expected in raw_by_key.items():
            observed = roadmap_by_key[key]
            observed_reason = observed.get("state_reason")
            observed_fields = {
                "title": _canonical_text(observed.get("title", "")),
                "state": _canonical_text(observed.get("state", "")).casefold(),
                "state_reason": (
                    _canonical_text(observed_reason) if observed_reason is not None else None
                ),
            }
            for field in ("title", "state", "state_reason"):
                if observed_fields[field] != expected[field]:
                    mismatches.append((key[0], key[1], field))
        if mismatches:
            raise AuditError(
                f"{len(mismatches)} raw-to-roadmap field mismatches; first={mismatches[:3]!r}"
            )
        raw_check = f"verified identities and state/title fields against {raw_path.name}"

    return {
        "artifact_dir": str(artifact_dir),
        "reviews": {
            "matching_rows": reviews_scope.get("matching_rows"),
            "signals": len(signals),
            "duplicates_removed": reviews_scope.get("duplicates_removed"),
            "other_rows_not_emitted": review_residual,
        },
        "github": {
            "mode": github_scope.get("mode"),
            "state_scope": github_scope.get("state_scope"),
            "authenticated": github_scope.get("authenticated"),
            "retrieved_at": github_scope.get("retrieved_at"),
            "pull_requests_dropped": github_scope.get("pull_requests_dropped_count"),
            "raw_snapshot_check": raw_check,
        },
        "roadmap": {
            "items": len(roadmap),
            "types": _counter(roadmap, "type"),
            "states": _counter(roadmap, "state"),
            "priority_tiers": dict(sorted(tiers.items())),
            "low_priority_items": low_priority,
            "explicit_priority_items": explicit_priority,
            "items_missing_priority_reasons": missing_reasons,
        },
        "checks": "PASS",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_reprocessed(
    source_dir: Path,
    reprocessed_dir: Path,
    source_report: dict[str, Any],
    reprocessed_report: dict[str, Any],
) -> dict[str, Any]:
    """Prove a reprocessed handoff is byte-identical to its authenticated source."""

    source_scope = _read_json(source_dir / "ingest_scope.json")
    reprocessed_scope = _read_json(reprocessed_dir / "ingest_scope.json")
    if not isinstance(source_scope, dict) or not isinstance(reprocessed_scope, dict):
        raise AuditError("comparison scopes must be JSON objects")
    source_github = source_scope.get("github")
    reprocessed_github = reprocessed_scope.get("github")
    if not isinstance(source_github, dict) or not isinstance(reprocessed_github, dict):
        raise AuditError("comparison scopes must contain github objects")
    if source_github.get("authenticated") is not True or source_github.get("mode") != "live":
        raise AuditError("comparison source is not recorded as an authenticated live ingest")
    if not str(source_report["github"]["raw_snapshot_check"]).startswith("verified identities"):
        raise AuditError("comparison source was not reconciled to its raw snapshot")
    if reprocessed_github.get("mode") != "fixture":
        raise AuditError("comparison target is not recorded as fixture reprocessing")
    processed_at = reprocessed_github.get("processed_at")
    if not isinstance(processed_at, str) or not processed_at.strip():
        raise AuditError("comparison target has no processed_at timestamp")

    for field in ("package_name", "repository", "priority_as_of"):
        if source_scope.get(field) != reprocessed_scope.get(field):
            raise AuditError(f"source and reprocessed scope differ at {field}")
    for field in (
        "state_scope",
        "api_version",
        "retrieved_at",
        "pull_requests_dropped_count",
        "roadmap_items",
    ):
        if source_github.get(field) != reprocessed_github.get(field):
            raise AuditError(f"source and reprocessed GitHub scope differ at {field}")
    if source_report["reviews"] != reprocessed_report["reviews"]:
        raise AuditError("source and reprocessed review counts differ")
    if source_report["roadmap"] != reprocessed_report["roadmap"]:
        raise AuditError("source and reprocessed roadmap counts differ")

    content_hashes: dict[str, str] = {}
    for name in ("signals.json", "roadmap.json"):
        source_hash = _sha256(source_dir / name)
        reprocessed_hash = _sha256(reprocessed_dir / name)
        if source_hash != reprocessed_hash:
            raise AuditError(f"source and reprocessed {name} are not byte-identical")
        content_hashes[name] = source_hash

    return {
        "checks": "PASS",
        "authenticated_source": True,
        "raw_source_reconciled": True,
        "state_scope": source_github.get("state_scope"),
        "retrieved_at": source_github.get("retrieved_at"),
        "processed_at": processed_at,
        "byte_identical_sha256": content_hashes,
    }


def _display(value: Any) -> str:
    if value is None:
        return "not-recorded"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _counts(values: dict[str, int]) -> str:
    return ", ".join(f"{key}={count}" for key, count in values.items())


def print_report(report: dict[str, Any]) -> None:
    reviews = report["reviews"]
    github = report["github"]
    roadmap = report["roadmap"]
    print(f"artifact: {report['artifact_dir']}")
    print(
        "reviews: "
        f"matching_rows={_display(reviews['matching_rows'])}, "
        f"signals={reviews['signals']}, "
        f"duplicates_removed={_display(reviews['duplicates_removed'])}, "
        f"other_rows_not_emitted={reviews['other_rows_not_emitted']}"
    )
    print(
        "github: "
        f"mode={_display(github['mode'])}, "
        f"state_scope={_display(github['state_scope'])}, "
        f"authenticated={_display(github['authenticated'])}, "
        f"retrieved_at={_display(github['retrieved_at'])}, "
        f"pull_requests_dropped={_display(github['pull_requests_dropped'])}"
    )
    print(f"roadmap: items={roadmap['items']}; types: {_counts(roadmap['types'])}")
    print(f"states: {_counts(roadmap['states'])}")
    print(f"priority_tiers: {_counts(roadmap['priority_tiers'])}")
    print(
        "priority: "
        f"low={roadmap['low_priority_items']}, "
        f"explicit={roadmap['explicit_priority_items']}, "
        f"missing_reasons={roadmap['items_missing_priority_reasons']}"
    )
    print(f"raw_snapshot: {github['raw_snapshot_check']}")
    print(f"checks: {report['checks']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit exact, non-sensitive counts in an ingestion artifact directory."
    )
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument(
        "--compare-reprocessed",
        type=Path,
        metavar="DIR",
        help=(
            "audit a reprocessed directory and prove it is byte-identical to this "
            "authenticated raw-source directory"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        source_report = audit_ingest(args.artifact_dir)
        reprocessed_report = None
        comparison = None
        if args.compare_reprocessed is not None:
            reprocessed_report = audit_ingest(args.compare_reprocessed)
            comparison = compare_reprocessed(
                args.artifact_dir,
                args.compare_reprocessed,
                source_report,
                reprocessed_report,
            )
    except AuditError as exc:
        print(f"ingest audit: FAIL: {exc}", file=sys.stderr)
        return 2
    if args.json:
        payload = {"source": source_report}
        if reprocessed_report is not None and comparison is not None:
            payload["reprocessed"] = reprocessed_report
            payload["comparison"] = comparison
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_report(source_report)
        if reprocessed_report is not None and comparison is not None:
            print()
            print_report(reprocessed_report)
            print()
            print(
                "comparison: PASS; authenticated raw source reconciled; "
                "signals.json and roadmap.json byte-identical; "
                f"state_scope={comparison['state_scope']}; "
                f"retrieved_at={comparison['retrieved_at']}; "
                f"processed_at={comparison['processed_at']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
