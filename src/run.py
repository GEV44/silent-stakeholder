"""End-to-end orchestration for cached Silent Stakeholder artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .confidence import (
    ConfidenceConfig,
    bootstrap_stability,
    cross_fitted_calibration,
    plot_reliability,
    score_gaps,
)
from .config import LLMConfig, PipelineConfig
from .embedding import get_embedder
from .gaps import GapThresholds, GeminiGapAdjudicator, detect_gaps
from .io_utils import ArtifactIOError, atomic_write_json, read_json
from .llm import LLMError, build_client, describe_backend
from .needs import GeminiNeedExtractor, NeedConfig, infer_needs
from .rank import rank_gaps
from .schema import Gap, LatentNeed
from .verify import GeminiGapCritic, verify_gaps

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "examples" / "demo"
UNCALIBRATED_TIE_RELATIVE_MARGIN = 0.01


class PipelineError(RuntimeError):
    """A user-actionable pipeline failure."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PipelineError(f"invalid ISO timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _records(payload: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if not all(isinstance(item, Mapping) for item in payload):
            raise PipelineError("artifact list contains a non-object value")
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                if not all(isinstance(item, Mapping) for item in value):
                    raise PipelineError(f"artifact field {key!r} contains a non-object value")
                return [dict(item) for item in value]
    raise PipelineError(f"expected a JSON list or one of fields: {', '.join(keys)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    """The commit an artifact was produced at, so a claim can be reproduced."""

    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else None


def _observed_llm(client: Any, *, use_llm: bool) -> dict[str, Any]:
    """Measured call accounting, not configuration.

    ``status`` answers the only question a reader actually has: did model
    inference happen? ``requested_but_no_calls`` means the run asked for Gemini
    and got nothing, so any need text in the artifact came from the offline
    fallback frames rather than a model.
    """

    if not use_llm:
        return {"status": "not_requested", "calls": 0}
    stats = getattr(client, "stats", None)
    as_dict = getattr(stats, "as_dict", None)
    counts = as_dict() if callable(as_dict) else {}
    calls = int(counts.get("calls", 0) or 0)
    failures = int(counts.get("failures", 0) or 0)
    if calls == 0:
        status = "requested_but_no_calls"
    elif failures:
        status = "partial"
    else:
        status = "ok"
    return {"status": status, **counts}


def _combined_sha256(paths: Sequence[Path]) -> str:
    """Hash named files as one versioned inference contract."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


REPORT_ARTIFACTS = (
    "signals.json",
    "roadmap.json",
    "needs.json",
    "gaps.json",
    "top_gaps.json",
    "verification.json",
)


def _artifact_declarations(output_dir: Path) -> dict[str, dict[str, str]]:
    """Bind every report input to its exact bytes in the run manifest."""

    names = [*REPORT_ARTIFACTS]
    if (output_dir / "stability.json").exists():
        names.append("stability.json")
    return {
        name: {"sha256": _sha256(output_dir / name), "schema_version": "1.0"}
        for name in names
    }


def _artifact_input_hashes(output_dir: Path) -> dict[str, str]:
    """Hash the exact signal and roadmap bytes packaged with a run."""

    return {
        "signals_sha256": _sha256(output_dir / "signals.json"),
        "roadmap_sha256": _sha256(output_dir / "roadmap.json"),
    }


def _artifact_scope(
    *,
    mode: str,
    signals: Sequence[Mapping[str, Any]],
    ingest_scope: Mapping[str, Any],
    as_of: datetime | None,
) -> dict[str, Any]:
    timestamps = [
        parsed
        for parsed in (_parse_time(str(signal.get("timestamp") or "")) for signal in signals)
        if parsed is not None
    ]
    evidence_start = min(timestamps) if timestamps else None
    evidence_end = max(timestamps) if timestamps else None
    reference = as_of or _utc_now()
    github_scope = ingest_scope.get("github")
    github_state = (
        str(github_scope.get("state_scope") or "")
        if isinstance(github_scope, Mapping)
        else ""
    )
    if mode == "current_opportunity":
        if evidence_end is None:
            raise PipelineError("current_opportunity mode requires timestamped user evidence")
        age_days = (reference - evidence_end).total_seconds() / 86400.0
        if age_days > 730:
            raise PipelineError(
                "current_opportunity mode rejects evidence older than two years; "
                "use current signals or choose exploratory_snapshot"
            )
    if mode == "historical_archive_check" and github_state != "all":
        raise PipelineError(
            "historical_archive_check requires an all-state GitHub ingest; "
            "open-only data cannot rule out a past public match"
        )
    return {
        **dict(ingest_scope),
        "analysis_mode": mode,
        "public_planning_proxy": True,
        "evidence_window": {
            "start": evidence_start.isoformat() if evidence_start else None,
            "end": evidence_end.isoformat() if evidence_end else None,
        },
        "analysis_as_of": reference.isoformat(),
        "claim": {
            "absence_means": "no matching artifact in the inspected public scope",
            "private_roadmap_observed": False,
        },
    }


def _attach_public_states(gaps: Sequence[dict[str, Any]]) -> None:
    mapping = {
        "IGNORED": "NO_PUBLIC_MATCH",
        "UNDER-PRIORITIZED": "ACKNOWLEDGED_UNSCHEDULED",
        "MISUNDERSTOOD": "PARTIAL_COVERAGE_HYPOTHESIS",
        "COVERED": "PUBLIC_MATCH",
    }
    for gap in gaps:
        metadata = dict(gap.get("metadata") or {})
        metadata["public_planning_state"] = mapping.get(
            str(gap.get("verdict") or ""), "UNKNOWN"
        )
        metadata["priority_score_semantics"] = (
            "review-derived priority proxy; not survey-validated ODI"
        )
        gap["metadata"] = metadata


DEMO_AS_OF = datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC)
DEMO_SCOPE = {
    "product": "ExamplePress for Android (synthetic)",
    "signal_window": "synthetic fixture",
    "roadmap_snapshot": "synthetic fixture",
}


def _as_demo_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a self-describing synthetic manifest without mutating the input."""

    stamped = dict(manifest)
    stamped["mode"] = "demo_fixture"
    scope = dict(stamped.get("scope") or {})
    scope.update(DEMO_SCOPE)
    stamped["scope"] = scope
    limitations = [
        str(item)
        for item in (stamped.get("limitations") or [])
        if not str(item).startswith("Historical reviews require")
    ]
    limitations.append(
        "Synthetic fixture dates are pinned for reproducibility; they are not historical "
        "user evidence or current product findings."
    )
    stamped["limitations"] = limitations
    return stamped


def _copy_demo(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in DEMO_DIR.glob("*.json"):
        shutil.copyfile(source, output_dir / source.name)


def _regenerate_demo(output_dir: Path) -> dict[str, Any]:
    """Run the real pipeline over the committed synthetic inputs.

    Copying the fixture made reproducibility pass vacuously: two "runs" were
    byte-identical because neither ran anything, so stale code and config hashes
    survived every check. This regenerates through :func:`analyze`, so the
    manifest's hashes describe the code that is actually checked out.

    Inputs are read from ``examples/demo`` and only ``signals.json`` and
    ``roadmap.json`` are used; everything else is derived.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.resolve() != DEMO_DIR.resolve():
        for name in ("signals.json", "roadmap.json"):
            shutil.copyfile(DEMO_DIR / name, output_dir / name)

    manifest = analyze(
        output_dir,
        output_dir,
        embedding_backend="hashing",
        use_llm=False,
        label_file=None,
        min_calibration_labels=100,
        top_k=5,
        as_of=DEMO_AS_OF,
        analysis_mode="exploratory_snapshot",
        include_covered=False,
    )
    _stamp_demo_mode(output_dir, manifest)
    return manifest


def _stamp_demo_mode(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Make every artifact self-describing as synthetic.

    ``analyze`` writes bare JSON lists. The fixture wraps them so
    ``mode: "demo_fixture"`` travels with the data itself -- the UI must be able
    to prove provenance from the file it loaded rather than infer it from a
    sibling file that may not have been copied alongside.
    """

    wrappers = {
        "signals.json": "signals",
        "roadmap.json": "roadmap",
        "needs.json": "needs",
        "gaps.json": "gaps",
        "top_gaps.json": "gaps",
        "verification.json": "verification",
    }
    for name, key in wrappers.items():
        path = output_dir / name
        if not path.exists():
            continue
        payload = read_json(path)
        rows = payload if isinstance(payload, list) else payload.get(key, payload)
        atomic_write_json(path, {"schema_version": "1.0", "mode": "demo_fixture", key: rows})

    manifest_path = output_dir / "run_manifest.json"
    stored = dict(read_json(manifest_path)) if manifest_path.exists() else dict(manifest)
    stamped = _as_demo_manifest(stored)
    stamped["inputs"] = _artifact_input_hashes(output_dir)
    stamped["artifacts"] = _artifact_declarations(output_dir)
    atomic_write_json(manifest_path, stamped)


def doctor(*, live: bool = False) -> dict[str, Any]:
    """Inspect safe runtime configuration and optionally smoke-test Gemini."""

    cfg = LLMConfig.load()
    result = {
        "status": "configured" if not cfg.is_offline else "offline",
        "llm": describe_backend(cfg),
        "live_smoke_test": None,
    }
    if not live:
        return result
    client = build_client(cfg, strict=True)
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["OK"]},
        },
        "required": ["status"],
        "additionalProperties": False,
    }
    started = _utc_now()
    response = client.models.generate_content(
        model=cfg.model,
        contents="Return the single allowed status value.",
        config={
            "temperature": 0.0,
            "max_output_tokens": 32,
            "response_mime_type": "application/json",
            "response_json_schema": schema,
        },
    )
    try:
        payload = json.loads(response.text)
    except (TypeError, ValueError) as exc:
        raise PipelineError("Gemini smoke test returned invalid JSON") from exc
    if payload != {"status": "OK"}:
        raise PipelineError("Gemini smoke test did not honor the output schema")
    result["status"] = "ready"
    result["live_smoke_test"] = {
        "ok": True,
        "elapsed_ms": round((_utc_now() - started).total_seconds() * 1000),
        "schema_sha256": hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    return result


def _label_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("gap_id") or ""),
        str(row.get("need_id") or ""),
        str(row.get("roadmap_id") or row.get("matched_roadmap_id") or ""),
    )


def _calibration_labels(
    label_path: Path,
    scored_gaps: Sequence[Mapping[str, Any]],
    *,
    minimum: int,
) -> tuple[list[float], list[int], list[str]]:
    payload = read_json(label_path)
    if isinstance(payload, Mapping) and str(payload.get("status", "")).startswith("EXAMPLE"):
        raise PipelineError("the example label file cannot be used for calibration")
    labels = _records(payload, "labels")
    by_gap: dict[str, Mapping[str, Any]] = {}
    by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    for label in labels:
        if label.get("adjudicated") is False:
            continue
        gap_id, need_id, roadmap_id = _label_key(label)
        if gap_id:
            by_gap[gap_id] = label
        if need_id and roadmap_id:
            by_pair[(need_id, roadmap_id)] = label

    scores: list[float] = []
    outcomes: list[int] = []
    matched_ids: list[str] = []
    for gap in scored_gaps:
        matched_label = by_gap.get(str(gap.get("id") or "")) or by_pair.get(
            (
                str(gap.get("need_id") or ""),
                str(gap.get("matched_roadmap_id") or ""),
            )
        )
        if matched_label is None or matched_label.get("public_claim_defensible") is None:
            continue
        metadata_value = gap.get("metadata")
        metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        raw_score = metadata.get("evidence_score", metadata.get("raw_confidence", 0.0))
        scores.append(float(raw_score or 0.0))
        outcomes.append(1 if bool(matched_label["public_claim_defensible"]) else 0)
        matched_ids.append(str(gap.get("id") or ""))

    if len(scores) < minimum:
        raise PipelineError(
            f"calibration requires at least {minimum} matched, adjudicated labels; "
            f"found {len(scores)}"
        )
    if len(set(outcomes)) < 2:
        raise PipelineError(
            "calibration labels must contain both defensible and non-defensible gaps"
        )
    return scores, outcomes, matched_ids


def _rank_uncalibrated(
    gaps: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Create an explicitly non-probabilistic demo/research ordering."""

    candidates: list[dict[str, Any]] = []
    for value in gaps:
        gap = dict(value)
        if str(gap.get("verdict")) == "COVERED" or str(gap.get("critique")) == "UNSUPPORTED":
            continue
        metadata = dict(gap.get("metadata") or {})
        raw = metadata.get("evidence_score", metadata.get("raw_confidence"))
        opportunity = gap.get("opportunity_score")
        if raw is None or opportunity is None:
            continue
        score = float(raw) * float(opportunity)
        gap["rank_score"] = round(score, 6)
        metadata["rank_basis"] = "uncalibrated_evidence_score_not_probability"
        gap["metadata"] = metadata
        candidates.append(gap)
    candidates.sort(
        key=lambda gap: (
            -float(gap["rank_score"]),
            -float(
                gap["metadata"].get(
                    "evidence_score", gap["metadata"].get("raw_confidence", 0.0)
                )
            ),
            -float(gap.get("opportunity_score") or 0.0),
            str(gap.get("id") or ""),
        )
    )
    unique: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for gap in candidates:
        title_key = " ".join(str(gap.get("latent_need") or "").casefold().split())
        if title_key and title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        unique.append(gap)
    framed = [
        gap
        for gap in unique
        if not str(gap.get("latent_need") or "").casefold().startswith("reliable ")
    ]
    if len(framed) >= 3:
        unique = framed
    ranked = unique[:top_k]

    # Scores this close are reproducibly ordered, but the arithmetic does not
    # establish a meaningful priority difference. A member is compared with
    # the highest score in its band so transitive chaining cannot widen it.
    priority_band = 0
    band_high_score: float | None = None
    for gap in ranked:
        score = float(gap["rank_score"])
        relative_margin = (
            0.0
            if band_high_score is None or band_high_score == 0.0
            else (band_high_score - score) / band_high_score
        )
        if band_high_score is None or relative_margin > UNCALIBRATED_TIE_RELATIVE_MARGIN:
            priority_band += 1
            band_high_score = score
        gap["metadata"]["priority_band"] = priority_band

    band_sizes: dict[int, int] = {}
    for gap in ranked:
        band = int(gap["metadata"]["priority_band"])
        band_sizes[band] = band_sizes.get(band, 0) + 1

    for index, gap in enumerate(ranked, start=1):
        raw = float(
            gap["metadata"].get(
                "evidence_score", gap["metadata"].get("raw_confidence", 0.0)
            )
        )
        opportunity = float(gap["opportunity_score"])
        metadata = gap["metadata"]
        band = int(metadata["priority_band"])
        within_tie_band = band_sizes[band] > 1
        previous_score = float(ranked[index - 2]["rank_score"]) if index > 1 else None
        metadata.update(
            {
                "priority_band_policy": "within_1_percent_of_band_high_score",
                "rank_separation": (
                    "not_established_within_1_percent_band"
                    if within_tie_band
                    else "outside_adjacent_1_percent_display_bands_not_validated"
                ),
                "deterministic_order_only": within_tie_band,
                "score_margin_to_previous": (
                    None
                    if previous_score is None
                    else round((previous_score - float(gap["rank_score"])) / previous_score, 6)
                ),
            }
        )
        gap["rank"] = index
        gap["why_rank"] = (
            f"priority proxy {opportunity:.2f} × evidence score {raw:.3f} = "
            f"{float(gap['rank_score']):.3f}; not a probability. Display band {band}; "
            + (
                "order within this 1% score band is a deterministic tie-break, not "
                "evidence of meaningful separation."
                if within_tie_band
                else (
                    "arithmetically outside the adjacent 1% display band; without labels or "
                    "stability analysis, this is not validated priority separation."
                )
            )
        )
    return ranked


def analyze(
    input_dir: Path,
    output_dir: Path,
    *,
    embedding_backend: str,
    use_llm: bool,
    label_file: Path | None,
    min_calibration_labels: int,
    top_k: int,
    as_of: datetime | None,
    analysis_mode: str,
    include_covered: bool = False,
    stability_iterations: int = 0,
) -> dict[str, Any]:
    """Run stages 1–5 from cached canonical inputs."""

    signals_path = input_dir / "signals.json"
    roadmap_path = input_dir / "roadmap.json"
    signals = _records(read_json(signals_path), "signals", "items")
    roadmap = _records(read_json(roadmap_path), "roadmap", "items")
    if not signals:
        raise PipelineError("signals.json is empty")
    if not roadmap:
        raise PipelineError("roadmap.json is empty")
    ingest_scope_path = input_dir / "ingest_scope.json"
    ingest_scope = read_json(ingest_scope_path) if ingest_scope_path.exists() else {}
    scope = _artifact_scope(
        mode=analysis_mode,
        signals=signals,
        ingest_scope=ingest_scope if isinstance(ingest_scope, Mapping) else {},
        as_of=as_of,
    )

    pipeline_cfg = PipelineConfig.load()
    embedder = get_embedder(
        embedding_backend,
        model_name=pipeline_cfg.embedding_model,
    )
    extractor = None
    adjudicator = None
    critic = None
    llm_info: dict[str, Any] = {"backend": "disabled"}
    llm_client: Any = None
    if use_llm:
        llm_cfg = LLMConfig.load()
        client = build_client(llm_cfg, strict=True)
        llm_client = client
        # The configured budget is mandatory: too-small responses can truncate
        # structured extraction mid-JSON and silently force deterministic fallback.
        extractor = GeminiNeedExtractor(
            client, llm_cfg.model, max_output_tokens=llm_cfg.max_output_tokens
        )
        adjudicator = GeminiGapAdjudicator(client, llm_cfg.model)
        critic = GeminiGapCritic(client, llm_cfg.model)
        llm_info = describe_backend(llm_cfg)

    # ``min_cluster_size`` is tuned for the full review corpus (thousands of
    # signals).  Applying it unchanged to a small input makes clustering return
    # nothing and the run hard-fail, which would break the live "re-run on a gap
    # you named" demo.  Scale it down for small corpora and leave the configured
    # value untouched once the corpus is large enough to support it.
    need_config = NeedConfig(
        min_cluster_size=max(2, min(pipeline_cfg.min_cluster_size, len(signals) // 8)),
        max_needs=40,
        llm_samples=LLMConfig.load().self_consistency_samples if use_llm else 3,
        random_seed=pipeline_cfg.random_seed,
    )
    needs = infer_needs(
        signals,
        embedder=embedder,
        extractor=extractor,
        config=need_config,
    )
    for need in needs:
        LatentNeed.model_validate(need)
    if not needs:
        raise PipelineError(
            "need extraction produced no clusters; lower min_cluster_size or inspect signal quality"
        )

    thresholds = GapThresholds(
        low=pipeline_cfg.thresholds.low,
        high=pipeline_cfg.thresholds.high,
        misunderstood_delta=pipeline_cfg.thresholds.symptom_delta,
    )
    gaps = detect_gaps(
        needs,
        roadmap,
        signals=signals,
        embedder=embedder,
        thresholds=thresholds,
        adjudicator=adjudicator,
        include_covered=include_covered,
        as_of=as_of,
    )
    _attach_public_states(gaps)
    confidence_config = ConfidenceConfig(
        weights=pipeline_cfg.confidence_weights or ConfidenceConfig().weights
    )
    preliminary = score_gaps(
        gaps,
        signals=signals,
        needs=needs,
        config=confidence_config,
        as_of=as_of,
    )

    calibrator = None
    calibration: dict[str, Any] = {
        "calibrated": False,
        "label_count": 0,
        "status": "uncalibrated_evidence_score_not_probability",
    }
    if label_file is not None:
        raw_scores, labels, matched_ids = _calibration_labels(
            label_file,
            preliminary,
            minimum=min_calibration_labels,
        )
        oof, metrics, calibrator = cross_fitted_calibration(
            raw_scores,
            labels,
            method=pipeline_cfg.calibrator,
            isotonic_min_samples=pipeline_cfg.min_isotonic_labels,
            random_state=pipeline_cfg.random_seed,
        )
        plot_reliability(
            labels,
            oof.tolist(),
            output_dir / "calibration.png",
            title="Gap defensibility reliability",
        )
        calibration = {
            "calibrated": True,
            "label_count": len(labels),
            "matched_gap_ids": matched_ids,
            **metrics,
        }

    scored = score_gaps(
        gaps,
        signals=signals,
        needs=needs,
        calibrator=calibrator,
        config=confidence_config,
        as_of=as_of,
    )
    verified, verification_reports = verify_gaps(
        scored,
        signals=signals,
        roadmap=roadmap,
        needs=needs,
        fuzzy_threshold=pipeline_cfg.fuzzy_quote_threshold,
        critic=critic,
        drop_unsupported=True,
    )
    for gap in verified:
        Gap.model_validate(gap)
    top_gaps = (
        rank_gaps(verified, top_k=top_k)
        if calibrator is not None
        else _rank_uncalibrated(verified, top_k=top_k)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "signals.json", signals)
    atomic_write_json(output_dir / "roadmap.json", roadmap)
    atomic_write_json(output_dir / "needs.json", needs)
    atomic_write_json(output_dir / "gaps.json", verified)
    atomic_write_json(output_dir / "verification.json", verification_reports)
    atomic_write_json(output_dir / "top_gaps.json", top_gaps)

    # Opt-in: this re-runs need inference once per iteration, so it costs
    # roughly `iterations` times a normal run. Kept off the default path so the
    #13-second reproduction stays 13 seconds, and written as its own artifact so
    # the explorer can show it when present and simply omit the panel when not.
    if stability_iterations:
        stability = bootstrap_stability(
            signals,
            lambda subset: infer_needs(subset, embedder=embedder, config=need_config),
            iterations=stability_iterations,
            seed=pipeline_cfg.random_seed,
        )
        atomic_write_json(output_dir / "stability.json", stability)

    assigned_signal_ids = {
        str(signal_id)
        for need in needs
        for signal_id in need.get("supporting_signal_ids", [])
    }
    top_k_shortfall = max(0, top_k - len(top_gaps))
    manifest = {
        "schema_version": "1.0",
        "mode": "production" if use_llm else "deterministic_baseline",
        "generated_at": _utc_now().isoformat(),
        "scope": scope,
        "embedding": {
            "requested_backend": embedding_backend,
            "backend": type(embedder).__name__,
            "model": pipeline_cfg.embedding_model if embedding_backend != "hashing" else None,
        },
        "llm": llm_info,
        # What the run *configured* is not what it *did*. A run can request
        # Gemini, have every call fail, silently fall back to offline need
        # frames, and still look model-backed if only `llm` is recorded.
        # `llm_observed` is measured after the fact and is the field to trust.
        "llm_observed": _observed_llm(llm_client, use_llm=use_llm),
        "code_version": _git_commit(),
        "calibration": calibration,
        "counts": {
            "signals": len(signals),
            "roadmap": len(roadmap),
            "needs": len(needs),
            "candidate_gaps": len(gaps),
            "verified_gaps": len(verified),
            "top_gaps": len(top_gaps),
            "signals_assigned_to_needs": len(assigned_signal_ids),
            "signals_unassigned": len(signals) - len(assigned_signal_ids),
            "top_k_requested": top_k,
            "top_k_shortfall": top_k_shortfall,
            "top_k_shortfall_reason": (
                "fewer verified, non-covered, uniquely titled candidates than requested"
                if top_k_shortfall
                else None
            ),
        },
        "inputs": _artifact_input_hashes(output_dir),
        "artifacts": _artifact_declarations(output_dir),
        "reproducibility": {
            "random_seed": pipeline_cfg.random_seed,
            "pipeline_config_sha256": _sha256(ROOT / "config" / "pipeline.json"),
            "inference_contract_sha256": _combined_sha256(
                [
                    ROOT / "src" / "embedding.py",
                    ROOT / "src" / "needs.py",
                    ROOT / "src" / "gaps.py",
                    ROOT / "src" / "confidence.py",
                    ROOT / "src" / "verify.py",
                    ROOT / "src" / "rank.py",
                    ROOT / "src" / "run.py",
                ]
            ),
            "inference_contract_files": [
                "src/embedding.py",
                "src/needs.py",
                "src/gaps.py",
                "src/confidence.py",
                "src/verify.py",
                "src/rank.py",
                "src/run.py",
            ],
        },
        "limitations": [
            "Public GitHub artifacts are not a complete internal roadmap.",
            "Historical reviews require an explicit temporal interpretation.",
            (
                "Need boundaries are one deterministic clustering lens; closely related "
                "symptoms may be grouped, and no alternate-clustering omission claim is made."
            ),
            (
                "Uncalibrated scores within 1% of a band's highest score share a display band. "
                "Bands expose near-ties; neither within-band order nor cross-band position proves "
                "validated priority separation without labels or stability analysis."
            ),
            (
                "Confidence values are calibrated probabilities."
                if calibrator is not None
                else "Evidence score is uncalibrated and is not a probability."
            ),
        ],
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="firecode",
        description="Evidence-grounded latent need vs roadmap analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="regenerate the credential-free synthetic demo through the pipeline"
    )
    demo.add_argument("--out-dir", type=Path, default=ROOT / "out")
    demo.add_argument(
        "--copy-only",
        action="store_true",
        help="copy the committed fixture verbatim instead of regenerating "
        "(reproducibility checks are meaningless against a copy)",
    )

    doctor_parser = subparsers.add_parser(
        "doctor", help="show safe configuration and optionally test live Gemini access"
    )
    doctor_parser.add_argument("--live", action="store_true")

    analyze_parser = subparsers.add_parser(
        "analyze", help="run all analytical stages from cached inputs"
    )
    analyze_parser.add_argument("--input-dir", type=Path, default=ROOT / "out")
    analyze_parser.add_argument("--out-dir", type=Path, default=ROOT / "out")
    analyze_parser.add_argument(
        "--embedding",
        choices=("hashing", "auto", "sentence-transformers"),
        default="hashing",
    )
    analyze_parser.add_argument(
        "--llm", action="store_true", help="enable configured Gemini inference"
    )
    analyze_parser.add_argument("--labels", type=Path)
    analyze_parser.add_argument("--min-calibration-labels", type=int, default=100)
    analyze_parser.add_argument("--top-k", type=int, choices=range(3, 6), default=5)
    analyze_parser.add_argument("--as-of", help="ISO timestamp for reproducible recency/priority")
    analyze_parser.add_argument(
        "--mode",
        choices=("exploratory_snapshot", "historical_archive_check", "current_opportunity"),
        default="exploratory_snapshot",
        help="claim/time contract enforced in the run manifest",
    )
    analyze_parser.add_argument("--include-covered", action="store_true")
    analyze_parser.add_argument(
        "--stability",
        type=int,
        default=0,
        metavar="N",
        help=(
            "resample the corpus N times and write stability.json; off by default "
            "because it re-runs need inference once per iteration"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            if args.copy_only:
                _copy_demo(args.out_dir)
                print(f"demo artifacts copied verbatim to {args.out_dir.resolve()}")
                return 0
            manifest = _regenerate_demo(args.out_dir)
            counts = manifest["counts"]
            print(
                f"demo regenerated through the pipeline: {counts['top_gaps']} top gaps from "
                f"{counts['signals']} synthetic signals -> {args.out_dir.resolve()}"
            )
            return 0
        if args.command == "doctor":
            print(json.dumps(doctor(live=args.live), indent=2, sort_keys=True))
            return 0
        manifest = analyze(
            args.input_dir,
            args.out_dir,
            embedding_backend=args.embedding,
            use_llm=args.llm,
            label_file=args.labels,
            min_calibration_labels=args.min_calibration_labels,
            top_k=args.top_k,
            as_of=_parse_time(args.as_of),
            analysis_mode=args.mode,
            include_covered=args.include_covered,
            stability_iterations=args.stability,
        )
        counts = manifest["counts"]
        print(
            f"wrote {counts['top_gaps']} top gaps from {counts['signals']} signals and "
            f"{counts['roadmap']} roadmap records to {args.out_dir.resolve()}"
        )
        print(manifest["calibration"].get("status", "calibrated"))
        return 0
    except (PipelineError, ArtifactIOError, LLMError, OSError, ValueError) as exc:
        sys.stderr.write(f"pipeline error: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
