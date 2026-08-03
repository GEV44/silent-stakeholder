"""Render a run's artifacts as a single self-contained HTML dashboard.

Why this exists alongside ``app.py``: Streamlit is an excellent analyst tool and
a poor product surface. Its own chrome, widgets and type scale show through any
styling laid on top, so the explorer will always read as a notebook. The people
this is built for -- engineering and product teams deciding whether a finding is
worth acting on -- judge a tool partly on whether it looks like it was built for
them. This produces that surface, from exactly the same artifacts, with no
second source of truth: every number here is read from the run directory.

Output is one file with no external requests. That keeps it usable with the
network unplugged, which the live demo requires, and makes it safe to open from
disk.

Security note. Review text and issue titles are attacker-controlled: the corpus
is public and anyone can publish a review. No artifact value is ever interpolated
into markup here. The data travels as JSON inside a script tag and is written
into the page with ``textContent``, so a review containing a link, an image tag
or a script cannot render as anything but characters.

Usage:
    python report.py --artifacts out/wordpress-open --out report.html
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html as html_stdlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
REQUIRED_REPORT_ARTIFACTS = (
    "signals.json",
    "roadmap.json",
    "needs.json",
    "gaps.json",
    "top_gaps.json",
    "verification.json",
)


class ArtifactValidationError(ValueError):
    """A report input cannot be proven to belong to one complete run."""


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json(path: Path, *, root: Path, limit: int = MAX_ARTIFACT_BYTES) -> Any:
    if path.is_symlink():
        raise ArtifactValidationError(f"symlinked artifact is not allowed: {path.name}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ArtifactValidationError(f"missing or unreadable artifact: {path.name}") from exc
    if not resolved.is_relative_to(root.resolve()):
        raise ArtifactValidationError(f"artifact escapes the run directory: {path.name}")
    if resolved.stat().st_size > limit:
        raise ArtifactValidationError(f"artifact exceeds {limit} bytes: {path.name}")
    try:
        raw = resolved.read_bytes()
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"malformed artifact: {path.name}") from exc


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_run_artifacts(artifacts: Path) -> dict[str, Any]:
    """Validate hashes, declarations, shapes, and IDs before rendering anything."""

    root = artifacts.resolve(strict=True)
    if not root.is_dir():
        raise ArtifactValidationError("artifact path is not a directory")
    manifest = _strict_json(root / "run_manifest.json", root=root, limit=2 * 1024 * 1024)
    if not isinstance(manifest, dict):
        raise ArtifactValidationError("run_manifest.json must contain an object")
    declarations = manifest.get("artifacts")
    if not isinstance(declarations, dict):
        raise ArtifactValidationError("manifest has no artifact hash declarations")

    loaded: dict[str, Any] = {"run_manifest.json": manifest}
    names = [*REQUIRED_REPORT_ARTIFACTS]
    stability_path = root / "stability.json"
    if stability_path.exists():
        if "stability.json" not in declarations:
            raise ArtifactValidationError("undeclared stability.json would mix run state")
        names.append("stability.json")

    for name in names:
        declaration = declarations.get(name)
        if not isinstance(declaration, Mapping):
            raise ArtifactValidationError(f"manifest does not declare {name}")
        expected = str(declaration.get("sha256") or "")
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected.casefold()):
            raise ArtifactValidationError(f"manifest has an invalid hash for {name}")
        path = root / name
        payload = _strict_json(path, root=root)
        if _sha256_bytes(path) != expected.casefold():
            raise ArtifactValidationError(f"artifact hash mismatch: {name}")
        if str(declaration.get("schema_version") or "") != "1.0":
            raise ArtifactValidationError(f"unsupported artifact schema: {name}")
        loaded[name] = payload

    for name, keys in {
        "signals.json": ("signals", "items"),
        "roadmap.json": ("roadmap", "items"),
        "needs.json": ("needs", "items"),
        "gaps.json": ("gaps", "items"),
        "top_gaps.json": ("gaps", "top_gaps", "items"),
    }.items():
        ids = [str(row.get("id") or "") for row in records(loaded[name], *keys)]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ArtifactValidationError(f"{name} contains missing or duplicate IDs")
    return loaded


def load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default


def records(payload: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def safe_repo_url(url: Any, repository: str) -> str:
    """Admit only a plain https github.com URL inside the verified repository.

    Roadmap records are artifact data, and an artifact can be malicious: a
    `javascript:` or credential-bearing URL assigned to an anchor becomes an
    active navigation on a page a judge trusts. Anything that fails this check
    renders as inert text instead (REQ-A-08).
    """

    value = str(url or "")
    prefix = "https://github.com/"
    if not value.startswith(prefix):
        return ""
    if any(ord(ch) < 33 for ch in value) or any(ch in value for ch in ('@', chr(92), '"', "'")):
        return ""
    if repository and not value[len(prefix):].lower().startswith(repository.lower() + "/"):
        return ""
    return value


def first(item: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if item.get(name) is not None:
            return item[name]
    return default


# The verdict names are our internal vocabulary. What a team acts on is a
# statement about their own board, so that is what the page shows.
PLANNING = {
    "IGNORED": ("Not on the board", "No artifact in the inspected public scope addresses this."),
    "UNDER-PRIORITIZED": ("On the board, parked", "An artifact exists but carries low priority signal."),
    "MISUNDERSTOOD": ("Aimed at the wrong thing", "An artifact covers the symptom, not the job."),
    "COVERED": ("Covered", "A public artifact addresses this need."),
    "REVIEW_REQUIRED": ("Needs review", "No verdict was recorded."),
    "NO_PUBLIC_MATCH": ("Not on the board", "No artifact in the inspected public scope addresses this."),
    "ACKNOWLEDGED_UNSCHEDULED": ("On the board, parked", "Acknowledged publicly but not scheduled."),
    "PARTIAL_COVERAGE_HYPOTHESIS": ("Aimed at the wrong thing", "Touches the area, misses the job."),
    "PUBLIC_MATCH": ("Covered", "A public artifact addresses this need."),
}

TONE = {
    "Not on the board": "critical",
    "On the board, parked": "warn",
    "Aimed at the wrong thing": "info",
    "Covered": "ok",
    "Needs review": "muted",
}


def mapping(value: Any) -> dict[str, Any]:
    """Return a concrete dictionary so optional artifact fields stay type-safe."""

    return value if isinstance(value, dict) else {}


def build_payload(
    artifacts: Path,
    profile: str = "public",
    *,
    validated: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read a run directory into the shape the page renders.

    ``profile`` is the licence boundary (REQ-A-08). The review corpus has no
    confirmed redistribution licence, so a single easy-to-attach HTML file
    containing real review text is exactly the accident to prevent. ``public``
    (the default) withholds verbatim spans on any run that is not the synthetic
    demo fixture — IDs, counts and verification totals remain. ``internal``
    keeps the spans and says so on the page.
    """

    def artifact(name: str, default: Any) -> Any:
        return validated.get(name, default) if validated is not None else load(artifacts / name, default)

    top = artifact("top_gaps.json", [])
    gaps = records(top, "gaps", "top_gaps", "items")
    all_gaps = records(artifact("gaps.json", top), "gaps", "items") or gaps
    signals = records(artifact("signals.json", []), "signals", "items")
    roadmap = records(artifact("roadmap.json", []), "roadmap", "items")
    verification = records(artifact("verification.json", []), "verification", "items")
    manifest = artifact("run_manifest.json", {})
    manifest = manifest if isinstance(manifest, dict) else {}
    stability = artifact("stability.json", {})
    stability = stability if isinstance(stability, dict) else {}

    demo_run = (not manifest) or manifest.get("mode") == "demo_fixture"
    allow_spans = profile in {"internal", "private-evidence"} or demo_run
    scope_map = mapping(manifest.get("scope"))
    repository = str(scope_map.get("repository") or "")
    withheld = 0

    signals_by_id = {str(first(row, "id", "signal_id", default="")): row for row in signals}
    roadmap_by_id = {str(first(row, "id", "roadmap_id", default="")): row for row in roadmap}
    verification_by_gap = {str(row.get("gap_id") or ""): row for row in verification}
    stability_shipped = mapping(stability.get("shipped"))

    quotes_valid = quotes_checked = 0
    for record in verification:
        for report in record.get("quote_reports") or []:
            if isinstance(report, dict):
                quotes_checked += 1
                quotes_valid += bool(report.get("valid"))

    findings: list[dict[str, Any]] = []
    for index, gap in enumerate(gaps):
        gap_id = str(gap.get("id") or "")
        metadata = mapping(gap.get("metadata"))
        state = str(metadata.get("public_planning_state") or first(gap, "verdict", default="")).upper()
        headline, explain = PLANNING.get(state, (state.replace("_", " ").title(), ""))
        title = str(first(gap, "latent_need", "need", "title", default="Untitled need"))

        record = verification_by_gap.get(gap_id) or {}
        spans = {}
        for report in record.get("quote_reports") or []:
            if isinstance(report, dict) and report.get("valid"):
                sid, span = str(report.get("id") or ""), str(report.get("matched_span") or "")
                if sid and span:
                    spans[sid] = span

        evidence = mapping(gap.get("evidence"))
        cited = [str(i) for i in (evidence.get("signal_ids") or gap.get("supporting_signal_ids") or [])]
        if not cited:
            cited = [str(q.get("id")) for q in (evidence.get("quotes") or []) if isinstance(q, dict)]

        proof = []
        for sid in cited:
            verified_span = spans.get(sid)
            if not verified_span:
                continue
            source = signals_by_id.get(sid, {})
            if not allow_spans:
                withheld += 1
            proof.append(
                {
                    "id": sid,
                    "span": verified_span if allow_spans else "",
                    "rating": first(source, "rating", "star", default=None),
                    "date": str(first(source, "timestamp", default=""))[:10],
                }
            )

        roadmap_id = str(first(gap, "matched_roadmap_id", "roadmap_id", default=""))
        match = roadmap_by_id.get(roadmap_id, {})
        stability_entry = stability_shipped.get(title) if isinstance(stability_shipped, dict) else None

        weakest = metadata.get("weakest_assumption")
        counter = metadata.get("counterevidence")
        findings.append(
            {
                "rank": int(first(gap, "rank", default=index + 1)),
                "id": gap_id,
                "title": title,
                "job": str(first(gap, "jtbd", "jtbd_statement", default="")).strip(),
                "state": headline,
                "tone": TONE.get(headline, "muted"),
                "explain": explain,
                "verified": len(proof),
                "cited": len(cited),
                "whyRank": str(first(gap, "why_rank", default="") or ""),
                "priorityBand": int(metadata.get("priority_band") or index + 1),
                "deterministicOrderOnly": bool(metadata.get("deterministic_order_only")),
                "opportunity": float(first(gap, "opportunity_score", default=0.0)),
                "kano": str(first(gap, "kano_class", default="") or ""),
                "roadmap": (
                    {
                        "id": roadmap_id,
                        "number": match.get("number"),
                        "title": str(first(match, "title", default="")),
                        "state": str(first(match, "state", default="")),
                        "url": safe_repo_url(first(match, "html_url", default=""), repository),
                        "milestone": str(first(match, "milestone", default="") or ""),
                        "labels": [str(x) for x in (match.get("labels") or [])][:6],
                    }
                    if roadmap_id
                    else None
                ),
                "proof": proof,
                "stability": (
                    {
                        "survival": float(stability_entry.get("survival") or 0.0),
                        "jaccard": float(stability_entry.get("mean_jaccard") or 0.0),
                    }
                    if isinstance(stability_entry, dict)
                    else None
                ),
                "weakest": (
                    {
                        "factor": str(weakest.get("factor") or ""),
                        "why": str(weakest.get("why") or ""),
                        "change": str(weakest.get("what_would_change_it") or ""),
                    }
                    if isinstance(weakest, dict) and weakest.get("factor")
                    else None
                ),
                "counter": (
                    {
                        "count": int(counter.get("count") or 0),
                        "share": float(counter.get("share_of_evidence") or 0.0),
                    }
                    if isinstance(counter, dict)
                    else None
                ),
                "borderline": bool(mapping(metadata.get("verdict_stability")).get("borderline")),
            }
        )

    calibration = mapping(manifest.get("calibration"))
    scope = mapping(manifest.get("scope"))
    manifest_counts = mapping(manifest.get("counts"))
    limitations = [str(item) for item in (manifest.get("limitations") or [])]
    inventory: list[dict[str, Any]] = []
    top_ids = {f["id"] for f in findings}
    for item in all_gaps:
        item_meta = mapping(item.get("metadata"))
        state = str(item_meta.get("public_planning_state") or first(item, "verdict", default="")).upper()
        ev = mapping(item.get("evidence"))
        inventory.append(
            {
                "title": str(first(item, "latent_need", "need", "title", default="Untitled")),
                "state": PLANNING.get(state, (state.title(), ""))[0],
                "cited": len(ev.get("signal_ids") or item.get("supporting_signal_ids") or []),
                "shipped": str(item.get("id") or "") in top_ids,
            }
        )

    issues_new = (
        f"https://github.com/{repository}/issues/new"
        if repository and safe_repo_url(f"https://github.com/{repository}/issues/new", repository)
        else ""
    )
    return {
        "profile": profile,
        "issuesNew": issues_new,
        "withheldQuotes": withheld,
        "mode": str(manifest.get("mode") or "unknown"),
        "demo": (not manifest) or manifest.get("mode") == "demo_fixture",
        "generated": str(manifest.get("generated_at") or ""),
        "product": str(scope.get("product") or scope.get("repository") or scope.get("package_name") or ""),
        "analysisMode": str(scope.get("analysis_mode") or ""),
        "counts": {
            "findings": len(findings),
            "candidates": len(all_gaps),
            "signals": len(signals),
            "roadmap": len(roadmap),
            "quotesValid": quotes_valid,
            "quotesChecked": quotes_checked,
            "signalsUnassigned": int(manifest_counts.get("signals_unassigned") or 0),
            "topKRequested": int(manifest_counts.get("top_k_requested") or len(findings)),
            "topKShortfall": int(manifest_counts.get("top_k_shortfall") or 0),
            "topKShortfallReason": str(manifest_counts.get("top_k_shortfall_reason") or ""),
        },
        "calibrated": bool(calibration.get("calibrated")),
        "labels": int(calibration.get("label_count") or 0),
        "llm": str(mapping(manifest.get("llm")).get("backend") or ""),
        "findings": findings,
        "inventory": inventory,
        "limitations": limitations,
        "stability": {
            "iterations": int(stability.get("iterations") or 0),
            "fraction": float(stability.get("fraction") or 0.0),
        }
        if stability_shipped
        else None,
    }


TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src '__SCRIPT_HASH__'; img-src 'none'; connect-src 'none'; object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'">
<title>__TITLE__</title>
<style>
/* ============================================================
   Design tokens. Ground: cool-biased neutrals; accent: teal-slate,
   deliberately clear of the four semantic verdict hues. Serif for
   the two title levels, grotesque for interface, monospace for
   every identifier, count and state. Tuned for a notebook screen:
   1080px column, panel instead of inline expansion, motion that
   directs attention and never blocks it.
   ============================================================ */
:root {
  color-scheme: light dark;

  --serif: ui-serif, Georgia, "Iowan Old Style", "Times New Roman", serif;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", "Segoe UI Mono", Consolas, monospace;

  --fs-xs: .72rem; --fs-sm: .81rem; --fs-base: .925rem; --fs-md: 1.02rem;
  --fs-lg: clamp(1.15rem, 1.05rem + .4vw, 1.35rem);
  --fs-xl: clamp(1.6rem, 1.25rem + 1.1vw, 2.3rem);

  --sp-1: .25rem; --sp-2: .5rem; --sp-3: .75rem; --sp-4: 1rem;
  --sp-5: 1.4rem; --sp-6: 2rem; --sp-7: 2.9rem;

  --r-sm: 6px; --r-md: 10px; --r-lg: 14px; --r-full: 999px;

  --dur-fast: 120ms; --dur: 200ms; --dur-slow: 380ms;
  --ease: cubic-bezier(.4, 0, .2, 1);
  --ease-panel: cubic-bezier(.32, .72, .28, 1);

  --bg: oklch(98.6% .004 210);
  --surface: oklch(100% 0 0);
  --surface-2: oklch(97% .005 210);
  --line: oklch(90% .008 210);
  --line-soft: oklch(94% .006 210);
  --text: oklch(24% .017 215);
  --text-2: oklch(48% .015 213);
  --text-3: oklch(60% .013 212);
  --accent: oklch(48% .095 205);
  --accent-soft: oklch(95.5% .022 205);
  --scrim: oklch(24% .017 215 / .38);

  --critical: oklch(53% .18 25);   --critical-soft: oklch(96% .028 25);
  --warn: oklch(58% .12 72);       --warn-soft: oklch(96% .035 72);
  --info: oklch(52% .15 278);      --info-soft: oklch(96% .028 278);
  --ok: oklch(52% .11 155);        --ok-soft: oklch(96% .032 155);

  --elev-1: 0 1px 2px oklch(24% .017 215 / .05);
  --elev-2: 0 3px 10px oklch(24% .017 215 / .07), 0 1px 2px oklch(24% .017 215 / .05);
  --elev-panel: -18px 0 48px oklch(24% .017 215 / .16);
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: oklch(17.5% .014 220); --surface: oklch(21% .016 220); --surface-2: oklch(24.5% .017 220);
    --line: oklch(31% .018 220); --line-soft: oklch(27% .016 220);
    --text: oklch(95% .005 220); --text-2: oklch(73% .013 218); --text-3: oklch(59% .015 216);
    --accent: oklch(76% .10 200); --accent-soft: oklch(28% .045 203);
    --scrim: oklch(5% 0 0 / .55);
    --critical: oklch(70% .15 25); --critical-soft: oklch(29% .065 25);
    --warn: oklch(76% .12 72); --warn-soft: oklch(29% .055 72);
    --info: oklch(73% .13 278); --info-soft: oklch(29% .06 278);
    --ok: oklch(72% .11 155); --ok-soft: oklch(28% .055 155);
    --elev-1: 0 1px 2px oklch(0% 0 0 / .32);
    --elev-2: 0 3px 12px oklch(0% 0 0 / .4), 0 1px 2px oklch(0% 0 0 / .3);
    --elev-panel: -18px 0 56px oklch(0% 0 0 / .5);
  }
}
/* The viewer's toggle must beat the media query in both directions. */
:root[data-theme="light"] {
  --bg: oklch(98.6% .004 210); --surface: oklch(100% 0 0); --surface-2: oklch(97% .005 210);
  --line: oklch(90% .008 210); --line-soft: oklch(94% .006 210);
  --text: oklch(24% .017 215); --text-2: oklch(48% .015 213); --text-3: oklch(60% .013 212);
  --accent: oklch(48% .095 205); --accent-soft: oklch(95.5% .022 205);
  --scrim: oklch(24% .017 215 / .38);
  --critical: oklch(53% .18 25); --critical-soft: oklch(96% .028 25);
  --warn: oklch(58% .12 72); --warn-soft: oklch(96% .035 72);
  --info: oklch(52% .15 278); --info-soft: oklch(96% .028 278);
  --ok: oklch(52% .11 155); --ok-soft: oklch(96% .032 155);
  --elev-1: 0 1px 2px oklch(24% .017 215 / .05);
  --elev-2: 0 3px 10px oklch(24% .017 215 / .07), 0 1px 2px oklch(24% .017 215 / .05);
  --elev-panel: -18px 0 48px oklch(24% .017 215 / .16);
}
:root[data-theme="dark"] {
  --bg: oklch(17.5% .014 220); --surface: oklch(21% .016 220); --surface-2: oklch(24.5% .017 220);
  --line: oklch(31% .018 220); --line-soft: oklch(27% .016 220);
  --text: oklch(95% .005 220); --text-2: oklch(73% .013 218); --text-3: oklch(59% .015 216);
  --accent: oklch(76% .10 200); --accent-soft: oklch(28% .045 203);
  --scrim: oklch(5% 0 0 / .55);
  --critical: oklch(70% .15 25); --critical-soft: oklch(29% .065 25);
  --warn: oklch(76% .12 72); --warn-soft: oklch(29% .055 72);
  --info: oklch(73% .13 278); --info-soft: oklch(29% .06 278);
  --ok: oklch(72% .11 155); --ok-soft: oklch(28% .055 155);
  --elev-1: 0 1px 2px oklch(0% 0 0 / .32);
  --elev-2: 0 3px 12px oklch(0% 0 0 / .4), 0 1px 2px oklch(0% 0 0 / .3);
  --elev-panel: -18px 0 56px oklch(0% 0 0 / .5);
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: var(--sans); font-size: var(--fs-base); line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
body.panel-open { overflow: hidden; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 0 var(--sp-5); }
h1, h2, h3 { margin: 0; }
a { color: var(--accent); text-underline-offset: 2px; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 4px; }
:focus:not(:focus-visible) { outline: none; }

/* ---- top bar ---- */
.bar { position: sticky; top: 0; z-index: 30;
       background: color-mix(in oklab, var(--bg) 86%, transparent);
       backdrop-filter: blur(12px) saturate(150%);
       border-bottom: 1px solid var(--line-soft); }
.bar-in { display: flex; align-items: center; gap: var(--sp-3); height: 54px; }
.mark { font-family: var(--serif); font-size: var(--fs-md); font-weight: 600;
        letter-spacing: -.01em; display: flex; align-items: center; gap: .5rem;
        white-space: nowrap; }
.mark svg { width: 23px; height: 20px; flex: 0 0 auto; color: var(--accent);
            transition: opacity var(--dur-slow) var(--ease); }
.mark:hover svg { opacity: .7; }
.spacer { flex: 1; }
.pill { font-family: var(--mono); font-size: var(--fs-xs); font-weight: 600;
        padding: .26rem .6rem; border-radius: var(--r-sm); border: 1px solid var(--line);
        color: var(--text-2); background: var(--surface); white-space: nowrap; }
.pill.live { color: var(--ok); border-color: color-mix(in oklab, var(--ok) 42%, var(--line));
             background: var(--ok-soft); }
.pill.demo { color: var(--warn); border-color: color-mix(in oklab, var(--warn) 48%, var(--line));
             background: var(--warn-soft); }
.iconbtn { display: grid; place-items: center; width: 32px; height: 32px; padding: 0;
           border-radius: var(--r-sm); border: 1px solid var(--line);
           background: var(--surface); color: var(--text-2); cursor: pointer; font: inherit;
           transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease),
                       transform var(--dur-fast) var(--ease); }
.iconbtn:hover { background: var(--surface-2); color: var(--text); }
.iconbtn:active { transform: scale(.92); }

/* ---- hero ---- */
.run { padding: var(--sp-7) 0 0; }
.meta { font-family: var(--mono); font-size: var(--fs-xs); color: var(--text-3);
        text-transform: uppercase; letter-spacing: .07em; margin-bottom: var(--sp-3); }
.run h1 { font-family: var(--serif); font-size: var(--fs-xl); font-weight: 600;
          letter-spacing: -.018em; text-wrap: balance; max-width: 26ch; }
.run h1 em { font-style: normal; color: var(--accent); }
.run > .wrap > p { color: var(--text-2); max-width: 66ch; margin: var(--sp-3) 0 0; }

.rail { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1px; margin-top: var(--sp-6); background: var(--line-soft);
        border: 1px solid var(--line-soft); border-radius: var(--r-md); overflow: hidden; }
.cell { background: var(--surface); padding: var(--sp-4); }
.cell dt { font-size: var(--fs-xs); text-transform: uppercase; letter-spacing: .08em;
           color: var(--text-3); font-weight: 600; margin: 0 0 .3rem; }
.cell dd { margin: 0; font-family: var(--mono); font-variant-numeric: tabular-nums;
           font-size: 1.5rem; font-weight: 600; letter-spacing: -.02em; line-height: 1.1; }
.cell small { display: block; font-family: var(--sans); font-size: var(--fs-xs);
              color: var(--text-3); font-weight: 400; margin-top: .3rem; line-height: 1.4; }

.banner { display: flex; gap: var(--sp-3); padding: var(--sp-3) var(--sp-4);
          border-radius: var(--r-md); border: 1px solid; font-size: var(--fs-sm);
          margin-top: var(--sp-4); }
.banner.warn { background: var(--warn-soft);
               border-color: color-mix(in oklab, var(--warn) 42%, transparent); }
.banner b { font-weight: 650; }

section { padding: var(--sp-7) 0 0; }
.sec-head { display: flex; align-items: baseline; gap: var(--sp-3);
            margin-bottom: var(--sp-4); flex-wrap: wrap;
            border-bottom: 1px solid var(--line-soft); padding-bottom: var(--sp-3); }
.sec-head h2 { font-family: var(--serif); font-size: var(--fs-lg); font-weight: 600; }
.sec-head p { margin: 0; color: var(--text-3); font-size: var(--fs-sm); }

/* ---- verdict filter chips ---- */
.filters { display: flex; gap: var(--sp-2); flex-wrap: wrap; margin-bottom: var(--sp-4); }
.fbtn { font: inherit; font-size: var(--fs-xs); font-weight: 650; cursor: pointer;
        padding: .32rem .7rem; border-radius: var(--r-full);
        border: 1px solid var(--line); background: var(--surface); color: var(--text-2);
        transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease),
                    border-color var(--dur-fast) var(--ease), transform var(--dur-fast) var(--ease); }
.fbtn:hover { border-color: color-mix(in oklab, var(--accent) 40%, var(--line)); }
.fbtn:active { transform: scale(.96); }
.fbtn[aria-pressed="true"] { background: var(--accent-soft); color: var(--accent);
                             border-color: color-mix(in oklab, var(--accent) 45%, var(--line)); }
.fbtn .n { font-family: var(--mono); opacity: .75; margin-left: .3rem; }

/* ---- finding card: stripe encodes the verdict ---- */
.card { position: relative; background: var(--surface); border: 1px solid var(--line);
        border-radius: var(--r-lg); margin-bottom: var(--sp-3); overflow: hidden;
        box-shadow: var(--elev-1);
        opacity: 0; transform: translateY(10px);
        transition: box-shadow var(--dur) var(--ease), border-color var(--dur) var(--ease),
                    transform var(--dur) var(--ease), opacity var(--dur-slow) var(--ease); }
.card.in { opacity: 1; transform: none; }
.card.hidden-by-filter { display: none; }
.card:hover { box-shadow: var(--elev-2); transform: translateY(-1px);
              border-color: color-mix(in oklab, var(--accent) 22%, var(--line)); }
.card::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 3px;
                background: var(--text-3); }
.card.t-critical::before { background: var(--critical); }
.card.t-warn::before { background: var(--warn); }
.card.t-info::before { background: var(--info); }
.card.t-ok::before { background: var(--ok); }

.head { display: grid; grid-template-columns: auto 1fr auto; gap: var(--sp-4);
        padding: var(--sp-4) var(--sp-5) var(--sp-3) calc(var(--sp-5) - 3px);
        align-items: start; }
.rk { font-family: var(--mono); font-size: var(--fs-sm); font-weight: 600;
      color: var(--text-3); padding-top: .18rem; font-variant-numeric: tabular-nums; }
.card h3 { font-size: var(--fs-md); font-weight: 620; letter-spacing: -.011em; }
.job { color: var(--text-2); font-size: var(--fs-sm); margin: .3rem 0 0; max-width: 70ch; }
.chip { font-family: var(--mono); font-size: var(--fs-xs); font-weight: 600;
        white-space: nowrap; padding: .24rem .55rem; border-radius: var(--r-sm);
        border: 1px solid; }
.chip.critical { color: var(--critical); background: var(--critical-soft);
                 border-color: color-mix(in oklab, var(--critical) 34%, transparent); }
.chip.warn { color: var(--warn); background: var(--warn-soft);
             border-color: color-mix(in oklab, var(--warn) 34%, transparent); }
.chip.info { color: var(--info); background: var(--info-soft);
             border-color: color-mix(in oklab, var(--info) 34%, transparent); }
.chip.ok { color: var(--ok); background: var(--ok-soft);
           border-color: color-mix(in oklab, var(--ok) 34%, transparent); }
.chip.muted { color: var(--text-3); background: var(--surface-2); border-color: var(--line); }

.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(142px, 1fr));
         gap: var(--sp-4); padding: 0 var(--sp-5) var(--sp-4) calc(var(--sp-5) - 3px); }
.fact dt { font-size: var(--fs-xs); text-transform: uppercase; letter-spacing: .075em;
           color: var(--text-3); font-weight: 600; margin: 0 0 .2rem; }
.fact dd { margin: 0; font-size: var(--fs-sm); font-weight: 600; }
.fact dd em { font-style: normal; font-family: var(--mono); font-variant-numeric: tabular-nums; }
.fact dd span { display: block; font-weight: 400; color: var(--text-3);
                font-size: var(--fs-xs); margin-top: .15rem; line-height: 1.4; }
.meter { height: 3px; border-radius: var(--r-full); background: var(--line);
         overflow: hidden; margin-top: .4rem; }
.meter i { display: block; height: 100%; background: var(--accent);
           transform-origin: left; transform: scaleX(0);
           transition: transform var(--dur-slow) var(--ease); }
.card.in .meter i { transform: scaleX(var(--v, 0)); }

.foot { border-top: 1px solid var(--line-soft);
        padding: var(--sp-3) var(--sp-5) var(--sp-3) calc(var(--sp-5) - 3px);
        display: flex; gap: var(--sp-3); align-items: center; flex-wrap: wrap;
        background: var(--surface-2); }
.btn { font: inherit; font-size: var(--fs-sm); font-weight: 620; cursor: pointer;
       padding: .38rem .85rem; border-radius: var(--r-sm); border: 1px solid var(--line);
       background: var(--surface); color: var(--text);
       transition: background var(--dur-fast) var(--ease), transform var(--dur-fast) var(--ease),
                   border-color var(--dur-fast) var(--ease); }
.btn:hover { background: var(--accent-soft);
             border-color: color-mix(in oklab, var(--accent) 40%, var(--line)); }
.btn:active { transform: scale(.97); }
.btn.primary { background: var(--accent); border-color: var(--accent);
               color: oklch(99% 0 0); }
.btn.primary:hover { filter: brightness(1.08); }
.why { font-size: var(--fs-xs); color: var(--text-3); }

/* ---- side panel ---- */
.scrim { position: fixed; inset: 0; z-index: 40; background: var(--scrim);
         opacity: 0; pointer-events: none; transition: opacity var(--dur-slow) var(--ease); }
.scrim.show { opacity: 1; pointer-events: auto; }
.panel { position: fixed; top: 0; right: 0; bottom: 0; z-index: 50;
         width: min(600px, 94vw); background: var(--bg);
         border-left: 1px solid var(--line); box-shadow: var(--elev-panel);
         transform: translateX(105%);
         transition: transform var(--dur-slow) var(--ease-panel);
         display: flex; flex-direction: column; }
.panel.open { transform: translateX(0); }
.panel-head { display: flex; align-items: flex-start; gap: var(--sp-3);
              padding: var(--sp-5); border-bottom: 1px solid var(--line-soft); }
.panel-head .rk { font-size: var(--fs-md); }
.panel-title { font-family: var(--serif); font-size: var(--fs-lg); font-weight: 600;
               letter-spacing: -.014em; flex: 1; }
.panel-body { overflow-y: auto; padding: var(--sp-5); flex: 1; }
.h4 { font-size: var(--fs-xs); text-transform: uppercase; letter-spacing: .08em;
      font-weight: 600; color: var(--text-3); margin: var(--sp-5) 0 var(--sp-3); }
.h4:first-child { margin-top: 0; }
.quote { background: var(--surface); border: 1px solid var(--line-soft);
         border-left: 3px solid var(--accent); border-radius: var(--r-sm);
         padding: var(--sp-3) var(--sp-4); margin-bottom: var(--sp-3);
         font-size: var(--fs-sm); white-space: pre-wrap; overflow-wrap: anywhere; }
.qmeta { display: flex; gap: var(--sp-3); align-items: center; flex-wrap: wrap;
         font-family: var(--mono); font-size: var(--fs-xs); color: var(--text-3);
         margin-bottom: .3rem; }
.stars { color: var(--warn); letter-spacing: .1em; }
.redacted { border-left-color: var(--line); color: var(--text-3); font-style: italic; }
.issue { display: block; background: var(--surface); border: 1px solid var(--line);
         border-radius: var(--r-sm); padding: var(--sp-3) var(--sp-4);
         text-decoration: none; color: inherit;
         transition: border-color var(--dur-fast) var(--ease); }
.issue:hover { border-color: color-mix(in oklab, var(--accent) 45%, var(--line)); }
.issue-id { font-family: var(--mono); font-size: var(--fs-xs); color: var(--text-3);
            margin-bottom: .25rem; }
.tags { display: flex; gap: var(--sp-2); flex-wrap: wrap; margin-top: var(--sp-2); }
.tag { font-family: var(--mono); font-size: var(--fs-xs); padding: .12rem .45rem;
       border-radius: var(--r-full); background: var(--surface-2);
       border: 1px solid var(--line); color: var(--text-3); }
.note { font-size: var(--fs-sm); color: var(--text-2); margin: 0 0 var(--sp-3);
        max-width: 70ch; }

/* ---- tables / details ---- */
table { width: 100%; border-collapse: collapse; font-size: var(--fs-sm); }
th, td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid var(--line-soft); }
th { font-size: var(--fs-xs); text-transform: uppercase; letter-spacing: .07em;
     color: var(--text-3); font-weight: 600; }
td.n, th.n { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }
tbody tr { transition: background var(--dur-fast) var(--ease); }
tbody tr:hover { background: var(--surface-2); }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: var(--r-sm);
          background: var(--surface); }
details { border: 1px solid var(--line); border-radius: var(--r-md);
          background: var(--surface); padding: var(--sp-4) var(--sp-5);
          margin-bottom: var(--sp-3); }
summary { cursor: pointer; font-weight: 620; font-size: var(--fs-sm); }
details[open] summary { margin-bottom: var(--sp-3); }
footer { margin-top: var(--sp-7); border-top: 1px solid var(--line-soft);
         padding: var(--sp-5) 0 var(--sp-7); color: var(--text-3);
         font-size: var(--fs-xs); }
footer .wrap { max-width: 78ch; margin-left: auto; margin-right: auto; }

@media (max-width: 640px) {
  .head { grid-template-columns: auto 1fr; }
  .head .chip { grid-column: 2; justify-self: start; }
}
@media print {
  .bar, .scrim, .panel, .filters, .foot .btn { display: none !important; }
  .card { break-inside: avoid; opacity: 1 !important; transform: none !important; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  *, *::before, *::after { animation-duration: .01ms !important;
                           transition-duration: .01ms !important; }
  .card { opacity: 1; transform: none; }
  .meter i { transform: scaleX(var(--v, 0)); }
}
</style>

<header class="bar"><div class="wrap bar-in">
  <div class="mark"><svg viewBox="0 0 100 88" role="img" aria-label="The Silent Stakeholder"><path fill="currentColor" fill-rule="evenodd" d="M8 12 h84 a6 6 0 0 1 6 6 v40 a6 6 0 0 1 -6 6 h-46 l-16 16 v-16 h-22 a6 6 0 0 1 -6 -6 v-40 a6 6 0 0 1 6 -6 z M34 28 h34 a4 4 0 0 1 4 4 v14 a4 4 0 0 1 -4 4 h-14 l-9 9 v-9 h-11 a4 4 0 0 1 -4 -4 v-14 a4 4 0 0 1 4 -4 z"/></svg> The Silent Stakeholder</div>
  <div class="spacer"></div>
  <span class="pill" id="provenance"></span>
  <button class="iconbtn" id="theme" aria-label="Toggle colour theme" title="Toggle colour theme">&#9681;</button>
</div></header>

<main>
  <div class="run">
    <div class="wrap">
      <div class="meta" id="scope"></div>
      <h1 id="verdict-line"></h1>
      <p id="lede"></p>
      <dl class="rail" id="rail"></dl>
      <div id="banners"></div>
    </div>
  </div>

  <section aria-label="Findings">
    <div class="wrap">
      <div class="sec-head">
        <h2>Findings</h2>
        <p id="findings-sub"></p>
      </div>
      <div class="filters" id="filters" role="group" aria-label="Filter findings by verdict"></div>
      <div id="findings"></div>
    </div>
  </section>

  <section aria-label="Method and provenance">
    <div class="wrap">
      <div class="sec-head">
        <h2>How to check us</h2>
        <p>Every number above, and what qualifies it.</p>
      </div>
      <div id="method"></div>
    </div>
  </section>

  <footer><div class="wrap" id="footer"></div></footer>
</main>

<div class="scrim" id="scrim"></div>
<aside class="panel" id="panel" role="dialog" aria-modal="true" aria-labelledby="panel-title">
  <div class="panel-head">
    <div class="rk" id="panel-rk"></div>
    <div class="panel-title" id="panel-title"></div>
    <button class="iconbtn" id="panel-close" aria-label="Close evidence panel">&#10005;</button>
  </div>
  <div class="panel-body" id="panel-body"></div>
</aside>

<script id="payload" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("payload").textContent);

  // Every value below reaches the page through textContent. Review text and
  // issue titles are attacker-controlled -- the corpus is public and anyone
  // can publish a review -- so nothing is ever assigned to innerHTML, and
  // roadmap URLs were validated server-side (https + verified repo only).
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }
  function pct(x) { return Math.round(x * 100) + "%"; }
  function num(x) { return Number(x).toLocaleString(); }
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var chr10 = String.fromCharCode(10);

  /* ---- theme ---- */
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem("ss-theme"); } catch (e) { saved = null; }
  if (saved) root.setAttribute("data-theme", saved);
  document.getElementById("theme").addEventListener("click", function () {
    var now = root.getAttribute("data-theme");
    var sysDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    var next = now ? (now === "dark" ? "light" : "dark") : (sysDark ? "light" : "dark");
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("ss-theme", next); } catch (e) {}
  });

  /* ---- header ---- */
  var prov = document.getElementById("provenance");
  prov.textContent = D.demo ? "DEMO \\u00b7 synthetic" : "REAL \\u00b7 " + (D.mode || "unknown");
  prov.className = "pill " + (D.demo ? "demo" : "live");

  var scopeBits = [];
  if (D.product) scopeBits.push(D.product);
  if (D.analysisMode) scopeBits.push(D.analysisMode.replace(/_/g, " "));
  if (D.generated) scopeBits.push(D.generated.slice(0, 10));
  if (D.profile === "private-evidence" && !D.demo) {
    scopeBits.push("PRIVATE — LICENSE RESTRICTED");
  }
  document.getElementById("scope").textContent = scopeBits.join("  \\u00b7  ");

  var C = D.counts;
  var h1 = document.getElementById("verdict-line");
  h1.appendChild(el("em", null, C.findings));
  h1.appendChild(document.createTextNode(
    (C.findings === 1 ? " need" : " needs") + " surfaced for roadmap review."));

  var byState = {};
  D.findings.forEach(function (f) { byState[f.state] = (byState[f.state] || 0) + 1; });
  var parts = Object.keys(byState).map(function (k) { return byState[k] + " " + k.toLowerCase(); });
  document.getElementById("lede").textContent =
    (parts.length ? parts.join(", ") + ". " : "") +
    "Inferred from what users wrote, checked against the roadmap records declared by this " +
    "run, and traceable to the evidence IDs behind each one.";

  /* ---- stat rail with count-up ---- */
  var railDefs = [
    ["Findings", C.findings, "from " + num(C.candidates) + " candidate needs"],
    ["Quotes verified", C.quotesValid,
     C.quotesChecked ? "of " + num(C.quotesChecked) + " re-matched against source text"
                     : "no verification record on this run"],
    ["User signals read", C.signals, "deduplicated reviews"],
    ["Roadmap compared", C.roadmap, "records in the declared snapshot"]
  ];
  var counters = [];
  railDefs.forEach(function (row) {
    var d = el("div", "cell");
    d.appendChild(el("dt", null, row[0]));
    var dd = el("dd");
    var value = el("span", null, reduced ? num(row[1]) : "0");
    dd.appendChild(value);
    dd.appendChild(el("small", null, row[2]));
    d.appendChild(dd);
    document.getElementById("rail").appendChild(d);
    counters.push({ node: value, target: Number(row[1]) });
  });
  function runCounters() {
    if (reduced) return;
    var t0 = null, dur = 700;
    function step(ts) {
      if (!t0) t0 = ts;
      var k = Math.min((ts - t0) / dur, 1);
      var easeOut = 1 - Math.pow(1 - k, 3);
      counters.forEach(function (c) {
        c.node.textContent = num(Math.round(c.target * easeOut));
      });
      if (k < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  runCounters();

  /* ---- banners ---- */
  var banners = document.getElementById("banners");
  function banner(boldText, rest) {
    var b = el("div", "banner warn"), t = el("div");
    t.appendChild(el("b", null, boldText));
    t.appendChild(document.createTextNode(rest));
    b.appendChild(t); banners.appendChild(b);
  }
  if (D.demo) {
    banner("Synthetic demo data. ",
      "These are fixtures for interface and contract testing, not product findings.");
  }
  if (D.profile === "private-evidence" && !D.demo) {
    banner("PRIVATE — LICENSE RESTRICTED. ",
      "This export contains short verified evidence spans and must not be published.");
  }
  if (C.quotesChecked && C.quotesValid < C.quotesChecked) {
    banner((C.quotesChecked - C.quotesValid) + " quote checks failed. ",
      "Those quotes are withheld from every screen here and from exported drafts.");
  }
  if (C.signalsUnassigned > 0) {
    banner(C.signalsUnassigned + " signals did not support an emitted need. ",
      "They remain counted in the run manifest instead of disappearing from the audit trail.");
  }
  if (C.topKShortfall > 0) {
    banner("The requested top " + C.topKRequested + " contains " + C.findings + ". ",
      "Only verified, non-covered, uniquely titled candidates emitted by this clustering lens " +
      "are shown. This shortfall is not evidence that no fifth need exists.");
  }
  var tied = D.findings.filter(function (f) { return f.deterministicOrderOnly; });
  if (tied.length > 1) {
    var tiedBands = Array.from(new Set(tied.map(function (f) { return f.priorityBand; })));
    banner("Near-tied priorities share band " + tiedBands.join(", ") + ". ",
      "Bands are display groups. Order inside a 1% band is a deterministic tie-break, not validated priority separation.");
  }
  if (D.withheldQuotes > 0) {
    banner("Public export: " + num(D.withheldQuotes) + " verified quotes withheld. ",
      "The review corpus has no confirmed redistribution licence, so verbatim text stays in " +
      "the run artifact. Signal IDs and verification counts remain. Regenerate with " +
      "--profile private-evidence with explicit acknowledgement for a private walkthrough.");
  }

  document.getElementById("findings-sub").textContent =
    "Ordered by evidence \\u00d7 opportunity; near ties share a display band. Band position is not validated priority separation.";

  /* ---- side panel ---- */
  var panel = document.getElementById("panel");
  var scrim = document.getElementById("scrim");
  var panelBody = document.getElementById("panel-body");
  var panelTitle = document.getElementById("panel-title");
  var panelRk = document.getElementById("panel-rk");
  var closeBtn = document.getElementById("panel-close");
  var lastOpener = null;

  function closePanel() {
    panel.classList.remove("open");
    scrim.classList.remove("show");
    document.body.classList.remove("panel-open");
    if (lastOpener) { lastOpener.focus(); lastOpener = null; }
  }
  scrim.addEventListener("click", closePanel);
  closeBtn.addEventListener("click", closePanel);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && panel.classList.contains("open")) closePanel();
  });

  function openPanel(f, opener) {
    lastOpener = opener || null;
    panelRk.textContent = "B" + f.priorityBand + " \u00b7 " + String(f.rank).padStart(2, "0");
    panelTitle.textContent = f.title;
    panelBody.textContent = "";

    var chip = el("span", "chip " + f.tone, f.state);
    panelBody.appendChild(chip);
    if (f.job) panelBody.appendChild(el("p", "note", f.job)).style.marginTop = "0.8rem";
    if (f.explain) panelBody.appendChild(el("p", "note", f.explain));

    if (f.whyRank) {
      panelBody.appendChild(el("p", "h4", "Why this order"));
      panelBody.appendChild(el("p", "note", f.whyRank));
    }

    panelBody.appendChild(el("p", "h4",
      "What users wrote \\u00b7 " + f.verified + " of " + f.cited + " cited signals verified"));
    if (!f.proof.length) {
      panelBody.appendChild(el("p", "note",
        "No verified quote is available for this finding. A quote appears only after its span " +
        "is matched against the immutable source text."));
    } else {
      // Quotes live in their own box so "show more" can append to the END of the
      // list rather than after the note and button that follow it.
      var quoteBox = el("div");
      panelBody.appendChild(quoteBox);
      var renderQuote = function (p) {
        var meta = el("div", "qmeta");
        meta.appendChild(el("span", null, p.id));
        if (p.rating) {
          meta.appendChild(el("span", "stars",
            "\\u2605".repeat(Math.max(0, Math.min(5, p.rating)))));
        }
        if (p.date) meta.appendChild(el("span", null, p.date));
        meta.appendChild(el("span", null, "verified"));
        quoteBox.appendChild(meta);
        if (p.span) {
          quoteBox.appendChild(el("div", "quote", p.span));
        } else {
          quoteBox.appendChild(el("div", "quote redacted",
            "Quote withheld in the public export (unresolved corpus licence). The signal ID " +
            "above resolves to the verified span in the run artifact."));
        }
      };
      f.proof.slice(0, 12).forEach(renderQuote);
      if (f.proof.length > 12) {
        var qnote = el("p", "note",
          "Showing 12 of " + f.proof.length + " verified quotes. Every cited ID is in the run " +
          (D.demo
            ? "artifact; the initial view is capped for readability."
            : "artifact; excerpts are capped because the review corpus has an unresolved licence."));
        var qmore = el("button", "btn", "Show all " + f.proof.length + " quotes");
        qmore.type = "button";
        qmore.addEventListener("click", function () {
          // Same renderer, so the redaction branch still fires on revealed rows:
          // a public export must keep withholding text after expansion.
          f.proof.slice(12).forEach(renderQuote);
          qnote.textContent = "Showing all " + f.proof.length + " verified quotes.";
          qmore.remove();
        });
        panelBody.appendChild(qnote);
        panelBody.appendChild(qmore);
      }
    }

    panelBody.appendChild(el("p", "h4", "Closest record on the roadmap"));
    if (f.roadmap) {
      var a = el(f.roadmap.url ? "a" : "div", "issue");
      if (f.roadmap.url) {
        a.href = f.roadmap.url; a.target = "_blank"; a.rel = "noopener noreferrer";
      }
      a.appendChild(el("div", "issue-id",
        (f.roadmap.number ? "#" + f.roadmap.number + "  \\u00b7  " : "") + f.roadmap.state));
      a.appendChild(el("div", null, f.roadmap.title || "Record not present in artifact"));
      if (f.roadmap.labels.length) {
        var tags = el("div", "tags");
        f.roadmap.labels.forEach(function (l) { tags.appendChild(el("span", "tag", l)); });
        a.appendChild(tags);
      }
      panelBody.appendChild(a);
    } else {
      panelBody.appendChild(el("p", "note", "No roadmap record was close enough to cite."));
    }

    panelBody.appendChild(el("p", "h4", "What would change our mind"));
    var said = false;
    if (f.weakest) {
      panelBody.appendChild(el("p", "note", f.weakest.factor + " \\u2014 " + f.weakest.why));
      if (f.weakest.change) {
        panelBody.appendChild(el("p", "note", "What would change it: " + f.weakest.change));
      }
      said = true;
    }
    if (f.counter) {
      panelBody.appendChild(el("p", "note", f.counter.count
        ? f.counter.count + " cited signal(s) read as positive about this area ("
          + pct(f.counter.share) + " of the evidence)."
        : "No cited signal contradicts this need."));
      said = true;
    }
    if (!said) panelBody.appendChild(el("p", "note", "Not recorded for this finding."));

    panel.classList.add("open");
    scrim.classList.add("show");
    document.body.classList.add("panel-open");
    panelBody.scrollTop = 0;
    closeBtn.focus();
  }

  /* ---- findings ---- */
  var host = document.getElementById("findings");
  D.findings.forEach(function (f) {
    var card = el("article", "card t-" + f.tone);
    card.setAttribute("data-state", f.state);

    var head = el("div", "head");
    head.appendChild(el("div", "rk",
      "B" + f.priorityBand + " \u00b7 " + String(f.rank).padStart(2, "0")));
    var mid = el("div");
    mid.appendChild(el("h3", null, f.title));
    if (f.job) mid.appendChild(el("p", "job", f.job));
    head.appendChild(mid);
    head.appendChild(el("span", "chip " + f.tone, f.state));
    card.appendChild(head);

    var facts = el("dl", "facts");
    function fact(label, value, sub, meter) {
      var d = el("div", "fact");
      d.appendChild(el("dt", null, label));
      var dd = el("dd");
      dd.appendChild(el("em", null, value));
      if (sub) dd.appendChild(el("span", null, sub));
      if (meter !== undefined) {
        var m = el("div", "meter"), i = el("i");
        i.style.setProperty("--v", meter);
        m.appendChild(i); dd.appendChild(m);
      }
      d.appendChild(dd); facts.appendChild(d);
    }

    var band = f.verified === 0 ? "Unverified"
             : (f.verified >= 50 && f.verified === f.cited) ? "Strong"
             : f.verified >= 10 ? "Moderate" : "Limited";
    fact("Evidence", band, f.verified + " of " + f.cited + " cited signals verified",
         f.cited ? Math.min(f.verified / f.cited, 1) : 0);

    if (f.stability) {
      var sBand = f.stability.survival >= 0.9 ? "Reproducible"
                : f.stability.survival >= 0.6 ? "Mostly stable" : "Fragile";
      fact("Resampling", sBand + " " + pct(f.stability.survival),
           "reappears when the corpus is resampled", f.stability.survival);
    }

    fact("Opportunity",
         f.opportunity >= 12 ? "High" : f.opportunity >= 8 ? "Moderate" : "Low",
         f.kano ? f.kano + " need" : "unmet-need score", Math.min(f.opportunity / 20, 1));

    if (f.roadmap) {
      fact("Roadmap record",
           f.roadmap.number ? "#" + f.roadmap.number : f.roadmap.id,
           f.roadmap.state
             ? f.roadmap.state + (f.roadmap.milestone ? " \\u00b7 " + f.roadmap.milestone
                                                      : " \\u00b7 no milestone")
             : "");
    } else {
      fact("Roadmap record", "none", "nothing in the inspected scope");
    }
    card.appendChild(facts);

    var foot = el("div", "foot");
    var btn = el("button", "btn primary", "Open the evidence");
    btn.addEventListener("click", function () { openPanel(f, btn); });
    foot.appendChild(btn);
    if (D.issuesNew) {
      // Human-reviewed prefill only: no API write and no verbatim review text.
      var lines = [
        "Reported by The Silent Stakeholder -- DRAFT, review before filing.",
        "",
        "Job to be done: " + (f.job || "not recorded"),
        "Roadmap reading: " + f.state + " -- " + f.explain,
        f.roadmap && f.roadmap.number
          ? "Closest existing record: #" + f.roadmap.number + " (" + f.roadmap.state + ")"
          : "Closest existing record: none in the inspected public scope",
        "Evidence: " + f.verified + " of " + f.cited + " cited signals verified against source text",
        f.stability ? "Survives " + pct(f.stability.survival) + " of corpus resamples" : "",
        "",
        "Signal IDs: " + f.proof.map(function (q) { return q.id; }).join(", "),
        "",
        D.demo
          ? "Quotes are synthetic fixture text; resolve the IDs in the run artifact."
          : "Quotes are licence-restricted and withheld here; resolve the IDs in the run artifact."
      ].filter(Boolean);
      var draft = el("a", "btn", "Draft the ticket");
      draft.href = D.issuesNew + "?title=" + encodeURIComponent(f.title) +
                   "&body=" + encodeURIComponent(lines.join(chr10));
      draft.target = "_blank";
      draft.rel = "noopener noreferrer";
      foot.appendChild(draft);
    }
    foot.appendChild(el("span", "why", f.explain));
    if (f.borderline) foot.appendChild(el("span", "chip warn", "borderline"));
    card.appendChild(foot);

    host.appendChild(card);
  });

  /* ---- verdict filter chips ---- */
  var filterHost = document.getElementById("filters");
  var states = Object.keys(byState);
  var active = "";
  function applyFilter() {
    Array.prototype.forEach.call(document.querySelectorAll(".card"), function (c) {
      var show = !active || c.getAttribute("data-state") === active;
      c.classList.toggle("hidden-by-filter", !show);
    });
    Array.prototype.forEach.call(filterHost.children, function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-state") === active));
    });
  }
  function filterButton(label, state, count) {
    var b = el("button", "fbtn", label);
    b.setAttribute("data-state", state);
    b.setAttribute("aria-pressed", "false");
    b.appendChild(el("span", "n", String(count)));
    b.addEventListener("click", function () {
      active = (active === state) ? "" : state;
      applyFilter();
    });
    filterHost.appendChild(b);
  }
  if (states.length > 1) {
    filterButton("All", "", D.findings.length);
    states.forEach(function (s) { filterButton(s, s, byState[s]); });
    applyFilter();
  }

  /* ---- staggered reveal: walks the eye down the ranking ---- */
  var cards = document.querySelectorAll(".card");
  if (!("IntersectionObserver" in window) || reduced) {
    Array.prototype.forEach.call(cards, function (c) { c.classList.add("in"); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e, i) {
        if (!e.isIntersecting) return;
        var t = e.target;
        setTimeout(function () { t.classList.add("in"); }, i * 50);
        io.unobserve(t);
      });
    }, { rootMargin: "0px 0px -6% 0px", threshold: 0.05 });
    Array.prototype.forEach.call(cards, function (c) { io.observe(c); });
  }

  /* ---- method ---- */
  var method = document.getElementById("method");
  function detail(summaryText, build) {
    var d = el("details");
    d.appendChild(el("summary", null, summaryText));
    build(d);
    method.appendChild(d);
  }

  detail("Is the confidence number a probability?", function (d) {
    if (D.calibrated) {
      d.appendChild(el("p", "note",
        "This run reports cross-fitted calibrated probabilities from " + D.labels +
        " human labels."));
    } else {
      d.appendChild(el("p", "note",
        "No \\u2014 and we do not print one. This run carries " + D.labels +
        " human labels, so nothing on this page appears as a confidence percentage. " +
        "Findings are ranked by an evidence score, and each card reports what was " +
        "actually checked instead."));
      d.appendChild(el("p", "note",
        "To make it a probability: label at least 100 need/roadmap pairs blind, then fit " +
        "cross-fitted Platt scaling. Isotonic is not selected below 1,000 labels. Labels " +
        "must come from a human who has not seen the verdicts; the pipeline refuses to " +
        "fabricate them, because a made-up probability is worse than an honest score."));
    }
  });

  detail("How a verdict is decided", function (d) {
    [["Aimed at the wrong thing",
      "Decided first. Each need is split into two equal-size, non-overlapping word probes " +
      "\\u2014 words unique to the reported symptom, words unique to the job. It fires only " +
      "when a roadmap record covers most of the symptom words while nothing covers most of " +
      "the job words."],
     ["Not on the board",
      "The closest roadmap record's similarity falls below the configured floor."],
     ["On the board, parked", "A record matches, but carries low priority signal."]
    ].forEach(function (row) {
      var p = el("p", "note");
      p.appendChild(el("b", null, row[0] + " \\u2014 "));
      p.appendChild(document.createTextNode(row[1]));
      d.appendChild(p);
    });
    d.appendChild(el("p", "note",
      "Only the ambiguous middle goes to a language model, and it must cite a roadmap ID " +
      "and an exact quote to be accepted. No model assigns rank or severity."));
  });

  function table(container, headers, rows) {
    var wrap = el("div", "scroll"), t = el("table");
    var thead = el("thead"), hr = el("tr");
    headers.forEach(function (h, i) { hr.appendChild(el("th", i ? "n" : null, h)); });
    thead.appendChild(hr); t.appendChild(thead);
    var tb = el("tbody");
    rows.forEach(function (r) {
      var tr = el("tr");
      r.forEach(function (c, i) { tr.appendChild(el("td", i ? "n" : null, c)); });
      tb.appendChild(tr);
    });
    t.appendChild(tb); wrap.appendChild(t); container.appendChild(wrap);
  }

  if (D.stability) {
    detail("Does each finding survive resampling?", function (d) {
      d.appendChild(el("p", "note",
        D.stability.iterations + " subsamples at " + pct(D.stability.fraction) +
        " of the corpus, without replacement, re-running need inference each time. " +
        "This is the one measure here that genuinely separates findings, and it uses " +
        "no human labels."));
      table(d, ["Finding", "Reappears", "Same signals"],
        D.findings.filter(function (f) { return f.stability; }).map(function (f) {
          return [f.title, pct(f.stability.survival), pct(f.stability.jaccard)];
        }));
      d.appendChild(el("p", "note",
        "The columns answer different questions. Reappears asks whether the need is a " +
        "property of the corpus or of one draw. Same signals asks whether the evidence " +
        "behind it is the same evidence \\u2014 a need can come back every time on a " +
        "different set of reviews."));
    });
  }

  detail("Every candidate emitted by this clustering run (" + D.inventory.length + ")", function (d) {
    d.appendChild(el("p", "note",
      "This inventories one configured clustering lens, including candidates that did not make " +
      "the cut. Another defensible split may emit different needs; this is not an exhaustiveness claim."));
    table(d, ["Need", "Reads as", "Signals", "In findings"],
      D.inventory.slice().sort(function (a, b) {
        return (b.shipped - a.shipped) || (b.cited - a.cited);
      }).map(function (r) {
        return [r.title, r.state, num(r.cited), r.shipped ? "yes" : "\\u2014"];
      }));
  });

  detail("Scope, and what we are not claiming", function (d) {
    d.appendChild(el("p", "note",
      "Public GitHub issues are a planning proxy, not proof of intent. Absence from the " +
      "inspected public scope is reported as \\u201cno matching artifact in the inspected " +
      "public scope\\u201d \\u2014 never as \\u201cthe team ignored this\\u201d. " +
      (D.demo
        ? "This corpus is synthetic fixture text with pinned dates, not historical user evidence or current product findings."
        : "Reviews in this corpus may be historical; a current roadmap snapshot cannot prove a historical need was ignored today.")));
    d.appendChild(el("p", "note",
      "Every signal here is one source type -- app-store reviews. Cross-source " +
      "corroboration (support tickets, churn notes) is not modelled, so the diversity " +
      "feature is constant on this run and cannot reward independent channels."));
    if (D.llm) d.appendChild(el("p", "note",
      "Language model backend on this run: " + D.llm + "."));
  });

  if (D.limitations.length) {
    detail("Recorded limitations (" + D.limitations.length + ")", function (d) {
      D.limitations.forEach(function (item) {
        d.appendChild(el("p", "note", "\u2022 " + item));
      });
    });
  }

  document.getElementById("footer").textContent =
    "Generated from run artifacts" + (D.generated ? " of " + D.generated : "") +
    ". Every figure resolves to a file in the run directory. " +
    (D.demo
      ? "All displayed review text is synthetic fixture content."
      : "Review text is licence-restricted \\u2014 cite signal IDs rather than redistributing quotes.");
})();
</script>
"""


def render(payload: dict[str, Any], title: str) -> str:
    # `</script>` inside the JSON would close the tag early, so the escape is on
    # `<` itself; json.dumps already handles quoting and control characters.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    page = TEMPLATE.replace("__DATA__", data).replace(
        "__TITLE__", html_stdlib.escape(str(title))
    )
    script = page.rsplit("<script>", 1)[1].split("</script>", 1)[0]
    digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
    return page.replace("__SCRIPT_HASH__", f"sha256-{digest}")


def _atomic_write_html(destination: Path, content: str, *, private: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ArtifactValidationError("refusing to replace a symlink destination")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600 if private else 0o644)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "examples" / "demo")
    parser.add_argument("--out", type=Path, default=ROOT / "report.html")
    parser.add_argument("--title", default="The Silent Stakeholder — roadmap gap report")
    parser.add_argument(
        "--profile",
        choices=("public", "private-evidence"),
        default="public",
        help=(
            "public (default) withholds verbatim review text on real runs -- the corpus "
            "licence is unresolved; private-evidence requires acknowledgement and an "
            "outside-repository destination"
        ),
    )
    parser.add_argument(
        "--acknowledge-private-export",
        action="store_true",
        help="confirm that a private-evidence export is licence-restricted",
    )
    args = parser.parse_args(argv)

    private = args.profile == "private-evidence"
    if private and not args.acknowledge_private_export:
        parser.error("private-evidence requires --acknowledge-private-export")
    if private and args.out.resolve().is_relative_to(ROOT.resolve()):
        parser.error("private-evidence output must be outside the repository")

    try:
        validated = validate_run_artifacts(args.artifacts)
        payload = build_payload(args.artifacts, profile=args.profile, validated=validated)
        _atomic_write_html(args.out, render(payload, args.title), private=private)
    except (ArtifactValidationError, OSError) as exc:
        parser.error(str(exc))
    print(
        f"wrote {args.out} — {payload['counts']['findings']} findings, "
        f"{payload['counts']['quotesValid']} verified quotes, "
        f"{'DEMO' if payload['demo'] else payload['mode']} [{args.profile}"
        f"{', ' + str(payload['withheldQuotes']) + ' quotes withheld' if payload['withheldQuotes'] else ''}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
