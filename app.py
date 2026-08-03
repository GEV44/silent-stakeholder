"""Read-only Streamlit explorer for cached Silent Stakeholder artifacts."""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "out"
DEMO_OUTPUT = ROOT / "examples" / "demo"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def records(payload: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def mapping(value: Any) -> dict[str, Any]:
    """Return a typed object mapping for optional JSON subdocuments."""

    return value if isinstance(value, dict) else {}


def load_manifest(directory: Path) -> dict[str, Any]:
    """Read a run manifest, treating unreadable as *unknown* rather than raising.

    Provenance is a safety signal, so it has to degrade instead of crashing: a
    live re-run interrupted mid-write leaves truncated JSON, and the explorer
    must then show the DEMO banner rather than a traceback in front of judges.

    Deliberately NOT applied to data artifacts. A corrupt `gaps.json` must fail
    loudly — silently rendering "no gaps" would be a false statement about the
    run, which is worse than an error page.

    Returns `{}` for missing, unparseable, or wrongly-shaped manifests; callers
    read an empty manifest as unknown provenance, which resolves to demo.
    """

    try:
        payload = load_json(directory / "run_manifest.json", {})
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def is_demo_manifest(directory: Path) -> bool:
    """Unknown provenance resolves to demo, so the banner shows rather than hides."""

    manifest = load_manifest(directory)
    return not manifest or manifest.get("mode") == "demo_fixture"


def artifact_dir() -> tuple[Path, bool]:
    """Pick which cached run to read. Provenance (real vs. demo) is decided ONLY by
    reading each candidate's run_manifest.json — never inferred from a file merely
    existing in a directory, which previously let stale fixtures pass as real."""
    configured_value = os.getenv("FIRECODE_OUTPUT_DIR")
    if configured_value:
        configured = Path(configured_value)
        if (configured / "top_gaps.json").exists():
            return configured, is_demo_manifest(configured)
        return DEMO_OUTPUT, True
    candidates = [DEFAULT_OUTPUT]
    if DEFAULT_OUTPUT.exists():
        candidates.extend(
            sorted(
                path
                for path in DEFAULT_OUTPUT.iterdir()
                if path.is_dir() and (path / "top_gaps.json").exists()
            )
        )
    candidates = [path for path in candidates if (path / "top_gaps.json").exists()]
    if len(candidates) > 1:
        configured = st.sidebar.selectbox(
            "Cached run",
            candidates,
            format_func=lambda path: "Demo fixture" if path == DEFAULT_OUTPUT else path.name,
        )
    else:
        configured = candidates[0] if candidates else DEFAULT_OUTPUT
    if (configured / "top_gaps.json").exists():
        return configured, is_demo_manifest(configured)
    return DEMO_OUTPUT, True


# Two parallel vocabularies reach this function and BOTH must be handled.
# `verdict` is src/schema.py's GapVerdict. `public_planning_state` is the
# cautious public-scope wording the cards actually display, and it is what the
# overview tab passes in. Dropping the planning-state keys once made every row
# on every artifact render grey, because real gaps always carry that metadata.
VERDICT_COLORS = {
    # GapVerdict
    "IGNORED": "#f05252",
    "UNDER-PRIORITIZED": "#f5a524",
    "MISUNDERSTOOD": "#8b5cf6",
    "COVERED": "#10b981",
    "REVIEW_REQUIRED": "#64748b",
    # public_planning_state, aligned with the verdict it narrates
    "NO_PUBLIC_MATCH": "#f05252",
    "ACKNOWLEDGED_UNSCHEDULED": "#f5a524",
    "PARTIAL_COVERAGE_HYPOTHESIS": "#8b5cf6",
    "PUBLIC_MATCH": "#10b981",
}


def verdict_color(verdict: str) -> str:
    return VERDICT_COLORS.get(str(verdict).upper(), "#64748b")


def first(item: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if item.get(name) is not None:
            return item[name]
    return default


def confidence_value(gap: dict[str, Any]) -> tuple[float, str]:
    calibrated = gap.get("calibrated_confidence")
    if calibrated is not None:
        return float(calibrated), "calibrated probability"
    metadata = mapping(gap.get("metadata"))
    raw = first(gap, "confidence", "raw_confidence", default=metadata.get("raw_confidence", 0.0))
    status = str(
        first(
            gap,
            "confidence_kind",
            default=metadata.get("confidence_status", "uncalibrated_score_not_probability"),
        )
    ).replace("_", " ")
    return float(raw), status


st.set_page_config(
    page_title="The Silent Stakeholder",
    page_icon="◌",
    layout="wide",
)
st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 1.6rem;}

      /* Product header ------------------------------------------------ */
      .eyebrow {letter-spacing:.13em; text-transform:uppercase; color:#64748b;
                font-size:.7rem; font-weight:700;}
      .hero {font-size:1.9rem; line-height:1.15; letter-spacing:-.025em;
             font-weight:700; margin:.15rem 0 .35rem;}
      .subtle {color:#64748b; max-width:760px; font-size:.94rem; line-height:1.5;}
      /* The mark is a single evenodd path whose inner bubble is a HOLE, so it
         inherits the text colour and needs no accent. See examples/brand/. */
      .brand {display:flex; align-items:flex-start; gap:1rem;}
      .brand svg {flex:0 0 auto; margin-top:.15rem; color:currentColor;}

      /* Banners ------------------------------------------------------- */
      .demo {background:#fff7ed; color:#9a3412; border:1px solid #fdba74;
             padding:.6rem .85rem; border-radius:8px; font-weight:650;
             margin:.75rem 0;}

      /* Finding card -------------------------------------------------- */
      .rank-no {font-size:.72rem; font-weight:800; letter-spacing:.08em;
                color:#94a3b8; text-transform:uppercase;}
      .finding-title {font-size:1.16rem; font-weight:680; line-height:1.3;
                      letter-spacing:-.012em; margin:.1rem 0 .3rem;}
      .job {color:#475569; font-size:.93rem; line-height:1.5; margin-bottom:.15rem;}

      /* Chips: the verdict in the team's own triage vocabulary --------- */
      .chip {display:inline-block; padding:.2rem .6rem; border-radius:999px;
             font-size:.74rem; font-weight:700; letter-spacing:.01em;
             border:1px solid currentColor; white-space:nowrap;}

      /* Labelled fact, used instead of a bare float ------------------- */
      .factlabel {font-size:.68rem; letter-spacing:.09em; text-transform:uppercase;
                  color:#94a3b8; font-weight:700; margin-bottom:.1rem;}
      .factvalue {font-size:1rem; font-weight:650; line-height:1.25;}
      .factnote {font-size:.78rem; color:#64748b; line-height:1.35;}

      /* Verified review quote. Rendered from escaped HTML, never markdown,
         so review text cannot inject a link, an image beacon, or markup. */
      .quote {border-left:3px solid #cbd5e1; padding:.4rem .8rem; margin:.3rem 0 .6rem;
              color:#334155; font-size:.92rem; line-height:1.55;
              background:rgba(148,163,184,.07); border-radius:0 6px 6px 0;
              white-space:pre-wrap; word-break:break-word;}
      .sigid {font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
              font-size:.74rem; color:#64748b;}

      @media (prefers-color-scheme: dark) {
        .job {color:#cbd5e1;}
        .quote {color:#e2e8f0; border-left-color:#475569;}
        .subtle, .factnote, .sigid {color:#94a3b8;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

base, demo_mode = artifact_dir()
top_payload = load_json(base / "top_gaps.json", [])
gap_payload = load_json(base / "gaps.json", top_payload)
signal_payload = load_json(base / "signals.json", [])
roadmap_payload = load_json(base / "roadmap.json", [])
manifest = load_manifest(base)
verification_payload = load_json(base / "verification.json", [])
# Optional: only runs produced with `--stability N` carry this. Absent is the
# normal case and the panel simply does not appear.
stability_payload = load_json(base / "stability.json", {})
demo_mode = demo_mode or manifest.get("mode") == "demo_fixture"

gaps = records(top_payload, ("gaps", "top_gaps", "items"))
all_gaps = records(gap_payload, ("gaps", "items")) or gaps
signals = records(signal_payload, ("signals", "items"))
roadmap = records(roadmap_payload, ("roadmap", "items"))
verifications = records(verification_payload, ("verification", "items"))
signals_by_id = {str(first(row, "id", "signal_id", default="")): row for row in signals}
roadmap_by_id = {str(first(row, "id", "roadmap_id", default="")): row for row in roadmap}
verification_by_gap = {str(row.get("gap_id") or ""): row for row in verifications}


def stale_artifact_reasons(manifest: dict[str, Any]) -> list[str]:
    """Compare the manifest's recorded hashes against the checked-out code.

    A cached artifact can outlive the code that produced it. Rendering it as
    current is how a demo ends up defending numbers the pipeline no longer
    produces, so the mismatch is surfaced rather than silently tolerated.
    """
    repro = manifest.get("reproducibility")
    if not isinstance(repro, dict):
        return ["run manifest has no reproducibility contract"]
    reasons: list[str] = []

    # This is a safety gate, so it fails CLOSED: anything that prevents the
    # comparison is reported, never silently treated as a pass. A contract file
    # that cannot be read is itself evidence the artifact predates the current
    # tree -- a renamed or deleted stage is exactly the case that must warn.
    config_path = ROOT / "config" / "pipeline.json"
    recorded_config = repro.get("pipeline_config_sha256")
    if not recorded_config:
        reasons.append("run manifest does not declare the pipeline configuration hash")
    else:
        try:
            current = hashlib.sha256(config_path.read_bytes()).hexdigest()
        except OSError:
            reasons.append("`config/pipeline.json` is unreadable, so this run cannot be verified")
        else:
            if current != recorded_config:
                reasons.append("`config/pipeline.json` changed since this run")

    files = repro.get("inference_contract_files")
    recorded_contract = repro.get("inference_contract_sha256")
    if not recorded_contract:
        reasons.append("run manifest does not declare the inference-contract hash")
    if not isinstance(files, list) or not files:
        reasons.append("run manifest does not declare its inference-contract files")
    elif recorded_contract:
        digest = hashlib.sha256()
        unreadable: list[str] = []
        for name in sorted(str(item) for item in files):
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            try:
                digest.update((ROOT / name).read_bytes())
            except OSError:
                unreadable.append(name)
            digest.update(b"\0")
        if unreadable:
            reasons.append(
                "pipeline source is missing or unreadable, so this run cannot be verified: "
                + ", ".join(f"`{name}`" for name in unreadable)
            )
        elif digest.hexdigest() != recorded_contract:
            reasons.append("pipeline code (`needs`/`gaps`/`verify`) changed since this run")
    return reasons


def evidence_identity(gap: dict[str, Any]) -> str:
    """A dedupe key derived from the evidence, not the wording.

    Titles get reworded, and title-matching then files a second copy of the same
    finding into someone's real repository. The set of cited signal IDs is the
    stable identity of a finding: rephrase the need all you like, it is the same
    gap if it rests on the same evidence.
    """
    evidence_value = gap.get("evidence")
    evidence: dict[str, Any] = evidence_value if isinstance(evidence_value, dict) else {}
    ids = evidence.get("signal_ids") or gap.get("supporting_signal_ids") or []
    canonical = ",".join(sorted(str(item) for item in ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def ticket_draft(gap: dict[str, Any]) -> str:
    """Render a gap as a human-reviewable issue draft.

    Deliberately a draft and never a filed issue: nothing here writes to an
    external repository. Only verification-valid spans are quoted, so a draft
    cannot carry text the pipeline could not prove came from a real review.
    """
    metadata = mapping(gap.get("metadata"))
    need = str(first(gap, "latent_need", "need", "title", default="Untitled need"))
    jtbd = str(first(gap, "jtbd", "jtbd_statement", default="")).strip()
    verdict = str(first(gap, "verdict", default="REVIEW_REQUIRED"))
    roadmap_id = str(first(gap, "matched_roadmap_id", "roadmap_id", default=""))
    spans, has_record = verified_spans(str(gap.get("id") or ""))

    lines = [
        f"# {need}",
        "",
        "> **DRAFT — not filed.** Generated by The Silent Stakeholder from user-signal "
        "evidence. Requires human review before it becomes an issue.",
        "",
        "## The job users are trying to do",
        jtbd or "_Not recorded for this gap._",
        "",
        "## Why this is a gap",
        f"Verdict: **{verdict}**"
        + (f" against the closest public roadmap record `{roadmap_id}`." if roadmap_id else "."),
    ]

    match = roadmap_by_id.get(roadmap_id, {})
    if match:
        lines += ["", f"Closest existing record: {first(match, 'title', default=roadmap_id)}"]

    lines += ["", "## Acceptance criteria"]
    if jtbd:
        lines.append(f"- A user can complete this without a workaround: {jtbd}")
    lines += [
        "- The behaviour holds for the cited scenarios below, not only the reported symptom.",
        "- Regression coverage references at least one cited signal ID.",
    ]

    # Two constraints shape this section. A gap can cite hundreds of signals
    # (398 on the strongest real one), which is unusable in an issue tracker;
    # and the review corpus has an unresolved redistribution licence, so a
    # downloadable file is a distribution vector in a way an on-screen view is
    # not. IDs are the durable reference and travel freely; the verbatim text
    # stays in the restricted artifact.
    quote_budget = 8
    lines += ["", "## Evidence (verified spans only)"]
    if not has_record:
        lines.append(
            "_No verification record for this gap. Quotes withheld — re-run the pipeline._"
        )
    elif not spans:
        lines.append("_No span passed verification._")
    else:
        ordered = sorted(spans.items())
        for signal_id, span in ordered[:quote_budget]:
            excerpt = " ".join(span.split())
            if len(excerpt) > 220:
                excerpt = excerpt[:217].rstrip() + "…"
            lines.append(f'- `{signal_id}` — "{excerpt}"')
        remaining = len(ordered) - quote_budget
        if remaining > 0:
            lines.append(
                f"- _…and {remaining} further verified signal(s). All {len(ordered)} IDs are "
                "recorded in the run artifact; excerpts are capped here because the review "
                "corpus has an unresolved redistribution licence._"
            )

    weakest = metadata.get("weakest_assumption")
    if isinstance(weakest, dict) and weakest.get("factor"):
        lines += [
            "",
            "## Risk / weakest assumption",
            f"{weakest.get('factor')} — {weakest.get('why', '')}",
        ]
        if weakest.get("what_would_change_it"):
            lines.append(f"What would change it: {weakest['what_would_change_it']}")

    counter = metadata.get("counterevidence")
    if isinstance(counter, dict) and int(counter.get("count") or 0):
        lines += [
            "",
            "## Counterevidence",
            f"{counter['count']} cited signal(s) read as positive about this area "
            f"({counter.get('basis', 'n/a')}).",
        ]

    scope = mapping(manifest.get("scope"))
    lines += [
        "",
        "## Provenance",
        f"- Gap ID: `{gap.get('id', '?')}`",
        f"- Evidence identity (dedupe key): `{evidence_identity(gap)}`",
        f"- Product/scope: {scope.get('product', 'unrecorded')}",
        f"- Analysis as-of: {scope.get('analysis_as_of', 'unrecorded')}",
        f"- Produced at commit: `{manifest.get('code_version', 'unrecorded')}`",
        "- Confidence: "
        + (
            "uncalibrated evidence score, not a probability"
            if not mapping(manifest.get("calibration")).get("calibrated")
            else "calibrated probability"
        ),
        "",
        "_Public roadmap records are a planning proxy. Absence from them is "
        '"no matching artifact in the inspected public scope", not evidence of intent._',
    ]
    if not demo_mode:
        lines += [
            "",
            "_Quoted review text is licence-restricted and excerpt-capped. Do not "
            "redistribute this draft outside the team; cite signal IDs instead._",
        ]
    return "\n".join(lines)


def verified_spans(gap_id: str) -> tuple[dict[str, str], bool]:
    """Return {signal_id: verified span} and whether a verification record exists.

    Only spans the verifier confirmed against immutable source text are
    returned. The UI must never fall back to raw signal text: an unverified
    quote on screen undoes the entire anti-hallucination argument, and the
    fallback is indistinguishable from a verified quote once rendered.
    """
    record = verification_by_gap.get(str(gap_id))
    if not record:
        return {}, False
    spans: dict[str, str] = {}
    for report in record.get("quote_reports") or []:
        if not isinstance(report, dict) or not report.get("valid"):
            continue
        signal_id = str(report.get("id") or "")
        span = str(report.get("matched_span") or "")
        if signal_id and span:
            spans[signal_id] = span
    return spans, True

# --------------------------------------------------------------------------
# Presentation helpers
#
# Design rule for this whole section: a number reaches the primary surface only
# if a product manager can act on it. Raw model floats (0.1324, 13.2754) look
# like precision and carry none -- the evidence scores on a real run span
# 0.064 end to end, so a reader comparing 0.132 with 0.128 is reading noise.
# Bands and counts go on the card; the underlying numbers stay one click away
# in "Method", where they are shown with the diagnostics that qualify them.
# --------------------------------------------------------------------------

# The verdict names in our schema are internal vocabulary. What a team acts on
# is their own triage language, so each verdict is rendered as the statement it
# makes about their board (CLAUDE.md, audience rule 2).
PLANNING_LANGUAGE: dict[str, tuple[str, str]] = {
    "IGNORED": (
        "Not on the board",
        "No artifact in the inspected public scope addresses this need.",
    ),
    "UNDER-PRIORITIZED": (
        "On the board, parked",
        "An artifact exists but carries low priority signal — no milestone, or open beyond a year.",
    ),
    "MISUNDERSTOOD": (
        "On the board, aimed at the wrong thing",
        "An artifact covers the reported symptom but not the job the user is trying to finish.",
    ),
    "COVERED": ("Covered", "A public artifact addresses this need."),
    "REVIEW_REQUIRED": ("Needs review", "No verdict was recorded for this need."),
    "NO_PUBLIC_MATCH": (
        "Not on the board",
        "No artifact in the inspected public scope addresses this need.",
    ),
    "ACKNOWLEDGED_UNSCHEDULED": (
        "On the board, parked",
        "Acknowledged publicly but not scheduled.",
    ),
    "PARTIAL_COVERAGE_HYPOTHESIS": (
        "On the board, partial coverage",
        "An artifact touches this area but does not cover the job.",
    ),
    "PUBLIC_MATCH": ("Covered", "A public artifact addresses this need."),
}


def planning_language(state: str) -> tuple[str, str]:
    key = str(state).upper()
    return PLANNING_LANGUAGE.get(key, (key.replace("_", " ").title(), ""))


def chip_html(label: str, color: str) -> str:
    """A verdict chip. Escapes its label, so artifact text cannot inject markup."""

    return (
        f'<span class="chip" style="color:{verdict_color(color)}">'
        f"{html.escape(str(label))}</span>"
    )


def quote_html(text: str) -> str:
    """Render a review or roadmap span as text, never as markdown.

    Reviews and issue titles are attacker-controlled: the corpus is public and
    anyone can publish one. Passing that through ``st.markdown`` would let a
    review render a clickable link, a remote image beacon, or arbitrary markup
    on a screen a judge is being asked to trust. Escaping into a styled block
    keeps the span readable and inert.
    """

    return f'<div class="quote">{html.escape(str(text))}</div>'


def fact_html(label: str, value: str, note: str = "") -> str:
    """A labelled fact: what it is, what it says, and what qualifies it."""

    parts = [
        f'<div class="factlabel">{html.escape(str(label))}</div>',
        f'<div class="factvalue">{html.escape(str(value))}</div>',
    ]
    if note:
        parts.append(f'<div class="factnote">{html.escape(str(note))}</div>')
    return "".join(parts)


def evidence_strength(gap: dict[str, Any]) -> tuple[str, str, str]:
    """Describe a gap's evidence by what was checked, not by the sigmoid.

    The composite evidence score is unusable as a headline: 58% of its weight
    sits on features that are constant across every gap on this run, so the
    scores barely separate. What genuinely differs, and what a skeptical reader
    can re-check line by line, is how many cited signals carry a quote that was
    matched against the immutable source text. That is what the card shows.

    Returns (band, headline, qualifier).
    """

    evidence = mapping(gap.get("evidence"))
    cited = evidence.get("signal_ids") or gap.get("supporting_signal_ids") or []
    cited_count = len(cited)
    spans, has_record = verified_spans(str(gap.get("id") or ""))
    verified = sum(1 for signal_id in cited if str(signal_id) in spans)

    if not has_record:
        return (
            "Unverified",
            "No verification record",
            f"{cited_count} signals cited, none re-checked against source text. "
            "Re-run the pipeline.",
        )
    if verified == 0:
        return ("Unverified", "0 quotes verified", f"{cited_count} signals cited, none matched.")

    complete = verified == cited_count
    if verified >= 50 and complete:
        band = "Strong"
    elif verified >= 10:
        band = "Moderate"
    else:
        band = "Limited"

    headline = f"{verified} verified quotes"
    if complete:
        qualifier = f"Every one of the {cited_count} cited signals matched its source review."
    else:
        qualifier = (
            f"{verified} of {cited_count} cited signals matched; "
            f"{cited_count - verified} did not and are withheld."
        )
    return band, headline, qualifier


def opportunity_band(score: float) -> str:
    """Bucket the recorded opportunity score for reading, not for deciding.

    Ranking uses the score itself; these bands only decide the word on the card.
    Cut points are display choices and are disclosed as such in Method.
    """

    if score >= 12.0:
        return "High"
    if score >= 8.0:
        return "Moderate"
    return "Low"


def verification_totals() -> tuple[int, int, int]:
    """(gaps carrying a verification record, valid quotes, total quote checks).

    The header once reported "verified top gaps" as simply the number of ranked
    gaps, which asserted verification the artifact had not been consulted for.
    This reads verification.json and reports what it actually says.
    """

    with_record = valid = checked = 0
    for gap in gaps:
        record = verification_by_gap.get(str(gap.get("id") or ""))
        if not record:
            continue
        with_record += 1
        for report in record.get("quote_reports") or []:
            if isinstance(report, dict):
                checked += 1
                valid += bool(report.get("valid"))
    return with_record, valid, checked


def gap_title(gap: dict[str, Any]) -> str:
    return str(first(gap, "latent_need", "need", "title", default="Untitled need"))


STABILITY = stability_payload if isinstance(stability_payload, dict) else {}
STABILITY_SHIPPED = mapping(STABILITY.get("shipped"))


def stability_for(gap: dict[str, Any]) -> tuple[str, str, str] | None:
    """How often this finding reappears when the corpus is resampled.

    Reported first among the numbers on a card when available, because it is the
    only one that meaningfully separates findings: the evidence score spans about
    0.06 across a whole run, while this spans 0.18 to 1.00. Returns None when the
    run was produced without ``--stability``.
    """

    entry = STABILITY_SHIPPED.get(gap_title(gap))
    if not isinstance(entry, dict) or entry.get("survival") is None:
        return None
    survival = float(entry["survival"])
    iterations = int(STABILITY.get("iterations") or 0)
    if survival >= 0.9:
        band = "Reproducible"
    elif survival >= 0.6:
        band = "Mostly stable"
    else:
        band = "Fragile"
    note = f"Reappears in {survival:.0%} of {iterations} resamples of the corpus."
    jaccard = entry.get("mean_jaccard")
    if isinstance(jaccard, (int, float)):
        note += f" Same supporting signals {float(jaccard):.0%} of the time."
    return band, f"{survival:.0%}", note


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

# The mark: a speech bubble with a bubble-shaped hole punched through it —
# what users said is solid, what they meant is the void inside. Inlined rather
# than st.image() so it inherits the theme's text colour. Source of truth and
# usage notes: examples/brand/.
BRAND_MARK = (
    '<svg width="50" height="44" viewBox="0 0 100 88" role="img" '
    'aria-label="The Silent Stakeholder">'
    '<path fill="currentColor" fill-rule="evenodd" '
    'd="M8 12 h84 a6 6 0 0 1 6 6 v40 a6 6 0 0 1 -6 6 h-46 l-16 16 v-16 h-22 '
    "a6 6 0 0 1 -6 -6 v-40 a6 6 0 0 1 6 -6 z "
    "M34 28 h34 a4 4 0 0 1 4 4 v14 a4 4 0 0 1 -4 4 h-14 l-9 9 v-9 h-11 "
    'a4 4 0 0 1 -4 -4 v-14 a4 4 0 0 1 4 -4 z"/>'
    "</svg>"
)

# Concatenated, not interpolated: tests/test_app_safety.py rejects any formatted
# value inside an unsafe_allow_html call, and it is right to — a static constant
# and a gap field are indistinguishable to the AST.
st.markdown(
    '<div class="brand">'
    + BRAND_MARK
    + '<div><div class="eyebrow">Roadmap gap analysis</div>'
    '<div class="hero">The Silent Stakeholder</div></div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtle">What your users need that your roadmap does not cover — inferred from '
    "their reviews, checked against your public issues and milestones, and traceable to the "
    "individual signals that support it.</div>",
    unsafe_allow_html=True,
)

if demo_mode:
    st.markdown(
        '<div class="demo">DEMO DATA — synthetic fixtures for interface and contract testing. '
        "These are not product findings.</div>",
        unsafe_allow_html=True,
    )

_stale = stale_artifact_reasons(manifest if isinstance(manifest, dict) else {})
if _stale:
    st.error(
        "**Stale results.** These were produced by code or configuration that no longer matches "
        "this checkout, so what is on screen may not be what the pipeline would produce now: "
        + "; ".join(_stale)
        + ". Re-run before presenting."
    )

_with_record, _valid_quotes, _checked_quotes = verification_totals()
_head = st.columns(4)
_head[0].metric("Findings", len(gaps), help="Ranked gaps this run is prepared to defend.")
_head[1].metric(
    "Quotes verified",
    f"{_valid_quotes:,}",
    help=(
        f"Quotes re-matched against the immutable source text, out of {_checked_quotes:,} checked. "
        f"{_with_record} of {len(gaps)} findings carry a verification record."
    ),
)
_head[2].metric("User signals analysed", f"{len(signals):,}")
_head[3].metric("Roadmap records compared", f"{len(roadmap):,}")

if _checked_quotes and _valid_quotes < _checked_quotes:
    st.warning(
        f"{_checked_quotes - _valid_quotes} of {_checked_quotes} quote checks failed. "
        "Those quotes are withheld from every screen and from issue drafts."
    )
if gaps and _with_record < len(gaps):
    st.warning(
        f"{len(gaps) - _with_record} of {len(gaps)} findings have no verification record. "
        "Their quotes are hidden until the pipeline is re-run."
    )

scope = mapping(manifest.get("scope"))
scope_values = [
    scope.get("product") or scope.get("repository") or scope.get("package_name"),
    scope.get("analysis_mode") or scope.get("signal_window"),
    scope.get("roadmap_snapshot"),
]
if isinstance(scope.get("evidence_window"), dict):
    window = scope["evidence_window"]
    scope_values.append(f"{window.get('start', '?')} → {window.get('end', '?')}")
if any(scope_values):
    st.caption("Scope: " + " · ".join(str(value) for value in scope_values if value))

def corpus_evidence(terms: set[str]) -> dict[str, Any]:
    """Scan the raw signal corpus for a judge's terms and say where each hit went.

    The ranked list is five needs drawn from 843 of 2,974 signals, so "we missed
    it" is almost never the true answer to a challenge -- the signals usually
    exist and were either absorbed into a need or dropped before one formed.
    This reports which, from immutable source text, so the answer is checkable
    on screen in the seconds a live question allows.
    """

    empty = {
        "matched": 0, "cited": 0, "uncovered": 0, "uncovered_rating": 0.0,
        "by_need": [], "examples": [],
    }
    if not terms:
        return empty

    cited_by: dict[str, str] = {}
    for candidate in all_gaps:
        evidence = mapping(candidate.get("evidence"))
        ids = evidence.get("signal_ids") or candidate.get("supporting_signal_ids") or []
        for signal_id in ids:
            cited_by.setdefault(str(signal_id), gap_title(candidate))

    counts: dict[str, int] = {}
    uncovered: list[dict[str, Any]] = []
    matched = 0
    for row in signals:
        text = str(row.get("text") or "")
        if not any(term in text.lower() for term in terms):
            continue
        matched += 1
        owner = cited_by.get(str(first(row, "id", "signal_id", default="")))
        if owner:
            counts[owner] = counts.get(owner, 0) + 1
        else:
            uncovered.append(row)

    ratings = [float(row.get("rating") or 0.0) for row in uncovered if row.get("rating")]
    # Worst-rated first: if we failed to cluster something, the angriest
    # unclustered signals are the ones a judge is most likely to be holding.
    uncovered.sort(key=lambda row: float(row.get("rating") or 5.0))
    return {
        "matched": matched,
        "cited": matched - len(uncovered),
        "uncovered": len(uncovered),
        "uncovered_rating": (sum(ratings) / len(ratings)) if ratings else 0.0,
        "by_need": sorted(counts.items(), key=lambda pair: -pair[1]),
        "examples": [
            {
                "id": str(first(row, "id", "signal_id", default="")),
                "rating": float(row.get("rating") or 0.0),
                "text": str(row.get("text") or ""),
            }
            for row in uncovered[:3]
        ],
    }


findings_tab, proof_tab, method_tab = st.tabs(["Findings", "Proof", "Method & provenance"])

# --------------------------------------------------------------------------
# Findings — what the roadmap is missing, and what each one implies
# --------------------------------------------------------------------------

with findings_tab:
    if not gaps:
        st.info(
            "No ranked results found. Run `python -m src.run demo` for the synthetic fixture, "
            "or `python -m src.run analyze` against an ingested corpus."
        )

    for index, gap in enumerate(gaps):
        rank = int(first(gap, "rank", default=index + 1))
        metadata = mapping(gap.get("metadata"))
        verdict = str(first(gap, "verdict", default="REVIEW_REQUIRED")).upper()
        state = str(metadata.get("public_planning_state") or verdict).upper()
        headline, verdict_note = planning_language(state)
        opportunity = float(first(gap, "opportunity_score", default=0.0))
        band, evidence_headline, evidence_note = evidence_strength(gap)
        roadmap_id = str(first(gap, "matched_roadmap_id", "roadmap_id", default=""))
        match = roadmap_by_id.get(roadmap_id, {})

        with st.container(border=True):
            left, right = st.columns([7, 3])

            priority_band = int(metadata.get("priority_band") or rank)
            order_note = (
                f"Display band {priority_band} · deterministic order {rank}"
                if metadata.get("deterministic_order_only")
                else f"Display band {priority_band} · order {rank}"
            )
            left.markdown(
                f'<div class="rank-no">{html.escape(order_note)}</div>',
                unsafe_allow_html=True,
            )
            left.markdown(
                f'<div class="finding-title">{html.escape(gap_title(gap))}</div>',
                unsafe_allow_html=True,
            )
            jtbd = str(first(gap, "jtbd", "jtbd_statement", default="")).strip()
            if jtbd:
                left.markdown(
                    f'<div class="job">{html.escape(jtbd)}</div>', unsafe_allow_html=True
                )

            right.markdown(chip_html(headline, state), unsafe_allow_html=True)
            if verdict_note:
                right.markdown(
                    f'<div class="factnote">{html.escape(verdict_note)}</div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            stability = stability_for(gap)
            facts = st.columns(4 if stability else 3)
            # Composed outside the markdown call on purpose: every value reaching
            # raw HTML must be escaped by the helper, and the safety test can only
            # see that when nothing is interpolated at the call site.
            evidence_value = f"{band} · {evidence_headline}"
            facts[0].markdown(
                fact_html("Evidence", evidence_value, evidence_note),
                unsafe_allow_html=True,
            )
            column = 1
            if stability:
                stability_band, stability_value, stability_note = stability
                stability_display = f"{stability_band} · {stability_value}"
                facts[column].markdown(
                    fact_html("Holds up on resampling", stability_display, stability_note),
                    unsafe_allow_html=True,
                )
                column += 1
            facts[column].markdown(
                fact_html(
                    "Opportunity",
                    opportunity_band(opportunity),
                    "Unmet-need score, banded for reading. Exact value in Method.",
                ),
                unsafe_allow_html=True,
            )
            column += 1
            if roadmap_id:
                facts[column].markdown(
                    fact_html(
                        "Closest roadmap record",
                        roadmap_id,
                        str(first(match, "title", default=""))[:110],
                    ),
                    unsafe_allow_html=True,
                )
            else:
                facts[column].markdown(
                    fact_html(
                        "Closest roadmap record", "None found", "Nothing in the inspected scope."
                    ),
                    unsafe_allow_html=True,
                )

            stability = metadata.get("verdict_stability")
            if isinstance(stability, dict) and stability.get("borderline"):
                st.warning(
                    f"**Borderline.** This reads as *{headline}* today, but flips to "
                    f"*{planning_language(str(stability.get('flips_to', ''))) [0]}* under a small "
                    "change to a threshold we have not fitted to labelled data. Treat it as a "
                    "soft call — see Method."
                )

            actions = st.columns([2, 2, 6])
            if actions[0].button("Open the proof", key=f"proof-{index}", width="stretch"):
                st.session_state["proof_gap"] = index
            actions[1].download_button(
                "Issue draft",
                data=ticket_draft(gap),
                file_name=f"gap-{gap.get('id', 'draft')}.md",
                mime="text/markdown",
                key=f"draft-{index}",
                width="stretch",
            )

            with st.expander("Why this reads as a gap"):
                st.caption(verdict_note)
                weakest = metadata.get("weakest_assumption")
                if isinstance(weakest, dict) and weakest.get("factor"):
                    st.markdown("**The weakest part of this finding**, stated by us")
                    st.write(f"{weakest.get('factor')} — {weakest.get('why', '')}")
                    if weakest.get("what_would_change_it"):
                        st.caption(f"What would change it: {weakest['what_would_change_it']}")

                counter = metadata.get("counterevidence")
                if isinstance(counter, dict):
                    count = int(counter.get("count") or 0)
                    st.markdown("**Evidence against this finding**")
                    if count:
                        st.write(
                            f"{count} cited signal(s) are positive about this area — "
                            f"{float(counter.get('share_of_evidence') or 0):.0%} of the evidence."
                        )
                    else:
                        st.caption(
                            "No cited signal contradicts this need "
                            f"({counter.get('basis', 'n/a')})."
                        )

                if isinstance(stability, dict) and stability.get("note"):
                    st.markdown("**Verdict stability**")
                    st.caption(str(stability["note"]))

    if all_gaps:
        st.write("")
        with st.expander(
            f"Every candidate need this run produced ({len(all_gaps)}) — including the "
            f"{max(len(all_gaps) - len(gaps), 0)} that did not make the cut"
        ):
            st.caption(
                "A need below the cut was cut for a recorded reason, not dropped quietly. "
                "Search the same inventory by keyword in the Proof tab."
            )
            top_ids = {str(first(item, "id", default="")) for item in gaps}
            inventory = []
            for item in all_gaps:
                item_id = str(first(item, "id", default=""))
                item_meta = mapping(item.get("metadata"))
                item_state = str(
                    item_meta.get("public_planning_state")
                    or first(item, "verdict", default="REVIEW_REQUIRED")
                ).upper()
                ev = mapping(item.get("evidence"))
                inventory.append(
                    {
                        "Need": gap_title(item),
                        "Reads as": planning_language(item_state)[0],
                        "Signals cited": len(
                            ev.get("signal_ids") or item.get("supporting_signal_ids") or []
                        ),
                        "Opportunity": opportunity_band(
                            float(first(item, "opportunity_score", default=0.0))
                        ),
                        "In top findings": "yes" if item_id in top_ids else "no",
                    }
                )
            inventory.sort(key=lambda row: (row["In top findings"] != "yes", row["Need"]))
            st.dataframe(inventory, hide_index=True, width="stretch")

# --------------------------------------------------------------------------
# Proof — the evidence trace, and the answer to "here is one you missed"
# --------------------------------------------------------------------------

with proof_tab:
    if not gaps:
        st.info("No ranked results to trace.")
    else:
        default_index = int(st.session_state.get("proof_gap", 0))
        default_index = min(max(default_index, 0), len(gaps) - 1)
        selected = st.selectbox(
            "Finding",
            options=list(range(len(gaps))),
            index=default_index,
            format_func=lambda idx: (
                f"Band {int((gaps[idx].get('metadata') or {}).get('priority_band') or idx + 1)}"
                f" · order {first(gaps[idx], 'rank', default=idx + 1)} · {gap_title(gaps[idx])}"
            ),
        )
        gap = gaps[selected]
        evidence = mapping(gap.get("evidence"))
        signal_ids = first(
            evidence, "signal_ids", default=first(gap, "supporting_signal_ids", default=[])
        )
        spans, has_record = verified_spans(str(gap.get("id") or ""))

        band, evidence_headline, evidence_note = evidence_strength(gap)
        st.markdown(f"#### Evidence — {band.lower()}: {evidence_headline}")
        st.caption(evidence_note)

        if not has_record:
            st.error(
                "**No verification record for this finding.** Quotes are hidden. A quote is shown "
                "only after `src/verify.py` matches it against the immutable source text; without "
                "that record there is nothing to prove it was not fabricated. Re-run the pipeline "
                "so `verification.json` is written."
            )
        else:
            shown = hidden = 0
            QUOTE_LIMIT = 25
            for signal_id in signal_ids or []:
                source = signals_by_id.get(str(signal_id), {})
                rating = first(source, "rating", "star", default="—")
                span = spans.get(str(signal_id))
                if span:
                    if shown < QUOTE_LIMIT:
                        st.markdown(
                            f'<div class="sigid">{html.escape(str(signal_id))} · '
                            f"rating {html.escape(str(rating))} · verified</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(quote_html(span), unsafe_allow_html=True)
                    shown += 1
                else:
                    hidden += 1
            if shown > QUOTE_LIMIT:
                st.caption(
                    f"Showing {QUOTE_LIMIT} of {shown} verified quotes. "
                    "Every cited ID is recorded in "
                    "the run artifact; excerpts are capped because the review corpus has an "
                    "unresolved redistribution licence."
                )
            if hidden:
                st.warning(
                    f"{hidden} of {shown + hidden} cited signals have no verified span and are "
                    "not displayed. Treat this finding's evidence as incomplete."
                )

        roadmap_id = str(first(gap, "matched_roadmap_id", "roadmap_id", default=""))
        match = roadmap_by_id.get(roadmap_id, {})
        st.markdown("#### Closest record on the public roadmap")
        if roadmap_id:
            st.markdown(
                f'<div class="sigid">{html.escape(roadmap_id)}</div>', unsafe_allow_html=True
            )
            st.markdown(
                quote_html(first(match, "title", default="Roadmap record not present in artifact")),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No roadmap record was close enough to cite.")
        st.caption(
            "Absence from the inspected public scope is not proof of a team's private intent. "
            "We report what the public artifacts show, and say so in those words."
        )

        st.markdown("#### Turn this into work")
        st.caption(
            "A reviewable issue draft carrying the verified evidence, acceptance criteria, the "
            "weakest assumption and full provenance. Nothing is filed — this app never writes to "
            "an external repository. Duplicates are keyed on the cited-evidence set "
            f"(`{evidence_identity(gap)}`), not the title, so rewording the need does not create "
            "a second issue."
        )
        draft = ticket_draft(gap)
        st.download_button(
            "Download issue draft (.md)",
            data=draft,
            file_name=f"gap-{gap.get('id', 'draft')}.md",
            mime="text/markdown",
            key="draft-proof",
        )
        with st.expander("Preview the draft"):
            st.code(draft, language="markdown")

    st.divider()
    st.markdown("#### “Here is a gap you missed”")
    st.caption(
        f"Searches all {len(all_gaps)} needs this run generated, not just the ranked "
        f"{len(gaps)}. If a proposed need is not in there at all, that is a recall miss and we "
        "record it as one rather than improvising an evidence trace."
    )
    query = st.text_input(
        "Describe the need",
        placeholder="e.g. reliable offline editing",
        label_visibility="collapsed",
    )
    if query:
        terms = {term for term in query.lower().split() if len(term) > 2}
        scored: list[tuple[float, dict[str, Any]]] = []
        for candidate in all_gaps:
            haystack = " ".join(
                str(first(candidate, name, default=""))
                for name in ("latent_need", "need", "jtbd", "jtbd_statement", "symptom")
            ).lower()
            hits = sum(term in haystack for term in terms)
            scored.append((hits / len(terms) if terms else 0.0, candidate))
        scored.sort(key=lambda pair: (-pair[0], str(first(pair[1], "id", default=""))))
        matches = [(score, item) for score, item in scored if score > 0][:5]
        if matches:
            top_k_ids = {str(first(item, "id", default="")) for item in gaps}
            for score, item in matches:
                item_id = str(first(item, "id", default=""))
                in_top = item_id in top_k_ids
                item_meta = mapping(item.get("metadata"))
                item_state = str(
                    item_meta.get("public_planning_state")
                    or first(item, "verdict", default="REVIEW_REQUIRED")
                ).upper()
                with st.container(border=True):
                    st.markdown(f"**{gap_title(item)}**")
                    where = (
                        f"ranked {first(item, 'rank', default='?')} of {len(gaps)} shown"
                        if in_top
                        else "generated, but below the cut"
                    )
                    st.caption(f"{planning_language(item_state)[0]} · {where}")
                    ev = mapping(item.get("evidence"))
                    cited = len(ev.get("signal_ids") or item.get("supporting_signal_ids") or [])
                    st.caption(f"{cited} signals cited · matches {score:.0%} of your terms")
                    if not in_top:
                        weakest = item_meta.get("weakest_assumption")
                        reason = (
                            weakest.get("factor")
                            if isinstance(weakest, dict) and weakest.get("factor")
                            else "ranked below the cut on evidence × opportunity"
                        )
                        st.caption(f"Why it was cut: {reason}")
        else:
            best = scored[0][0] if scored else 0.0
            st.error(
                f"**Recall miss, measured.** No generated need shares this vocabulary — best "
                f"term overlap across all {len(all_gaps)} needs is {best:.0%}. We record this as "
                "a miss rather than improvising an evidence trace for it."
            )

        # A miss against five needs is not the same as a miss against the corpus.
        # Only 843 of 2,974 signals support a shipped need, so the honest answer to
        # "you missed this" is usually "here is what we hold on it, and where it
        # went" -- which is checkable -- rather than "we missed it", which is not.
        corpus = corpus_evidence(terms)
        if corpus["matched"]:
            st.markdown("##### What the corpus actually holds on that")
            st.caption(
                f"{corpus['matched']} of {len(signals)} signals mention these terms. "
                "This is a raw lexical scan of immutable source text, not a generated claim."
            )
            columns = st.columns(3)
            columns[0].metric("Mention the terms", corpus["matched"])
            columns[1].metric("Already cited by a need", corpus["cited"])
            columns[2].metric("Uncovered", corpus["uncovered"])
            if corpus["by_need"]:
                st.caption(
                    "Absorbed into: "
                    + " · ".join(
                        f"{name} ({count})" for name, count in corpus["by_need"]
                    )
                )
            if corpus["uncovered"]:
                st.caption(
                    f"{corpus['uncovered']} uncovered, mean rating "
                    f"{corpus['uncovered_rating']:.1f}/5. A low mean is friction we did not "
                    "cluster into a need; a high mean is praise the friction filter drops by "
                    "design. Either way it is measured, not asserted."
                )
                for row in corpus["examples"]:
                    # Built as one pre-escaped string: nesting an f-string inside
                    # html.escape() would still put a raw interpolation inside an
                    # unsafe_allow_html call, which is exactly what we forbid.
                    attribution = f"{row['id']} · rated {row['rating']:.0f}/5"
                    st.markdown(
                        f"<span class='sigid'>{html.escape(attribution)}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(quote_html(row["text"]), unsafe_allow_html=True)
        elif query:
            st.caption(
                f"No signal in the {len(signals)}-signal corpus contains these terms either. "
                "The evidence for this need is absent from our inputs, not merely unranked."
            )

# --------------------------------------------------------------------------
# Method & provenance — every number, and what qualifies it
# --------------------------------------------------------------------------

with method_tab:
    st.markdown("#### How a finding gets its verdict")
    st.markdown(
        "- **On the board, aimed at the wrong thing** is decided first. Each need is split into "
        "two equal-size, non-overlapping word probes: words unique to the reported symptom, and "
        "words unique to the job. It fires only when a roadmap record covers most of the symptom "
        "words while nothing covers most of the job words — same measure and same cutoff on both "
        "sides, so neither side is favoured by being longer.\n"
        "- **Not on the board** — the closest roadmap record's similarity falls below the "
        "configured floor.\n"
        "- **On the board, parked** — a record matches, but carries low priority signal.\n"
        "- Only the ambiguous middle goes to a language model, and it must cite a roadmap ID and "
        "an exact quote to be accepted."
    )

    st.divider()
    st.markdown("#### Why this order")
    st.caption(
        "Order = evidence score × opportunity score. Both are recorded per finding. Scores "
        "within 1% of the highest score in their band share a display band. Order inside that "
        "band is a deterministic tie-break. Crossing a band boundary is only arithmetic; without "
        "labels or stability analysis, it is not validated priority separation. No model assigns "
        "rank or severity."
    )
    if len(gaps) < 2:
        st.info("At least two findings are needed to compare an ordering.")
    else:
        def _label(index: int) -> str:
            return f"{first(gaps[index], 'rank', default=index + 1)}. {gap_title(gaps[index])[:52]}"

        pick = st.columns(2)
        a_index = pick[0].selectbox("Higher-ranked", range(len(gaps)), 0, format_func=_label)
        b_index = pick[1].selectbox("Compared with", range(len(gaps)), 1, format_func=_label)

        if a_index == b_index:
            st.info("Pick two different findings.")
        else:
            def _parts(gap: dict[str, Any]) -> tuple[float, float, float, list[dict[str, Any]]]:
                metadata = mapping(gap.get("metadata"))
                breakdown = mapping(metadata.get("score_breakdown"))
                evidence = float(breakdown.get("evidence_score") or confidence_value(gap)[0])
                priority = float(first(gap, "opportunity_score", default=0.0))
                total = float(first(gap, "rank_score", "score", default=evidence * priority))
                terms = breakdown.get("terms")
                return evidence, priority, total, terms if isinstance(terms, list) else []

            a_gap, b_gap = gaps[a_index], gaps[b_index]
            a_ev, a_pr, a_total, a_terms = _parts(a_gap)
            b_ev, b_pr, b_total, b_terms = _parts(b_gap)

            a_meta = mapping(a_gap.get("metadata"))
            b_meta = mapping(b_gap.get("metadata"))
            if (
                a_meta.get("priority_band") == b_meta.get("priority_band")
                and a_meta.get("deterministic_order_only")
            ):
                st.warning(
                    "**No meaningful separation is established.** These findings share the "
                    "same 1% display band; the displayed order is only a deterministic tie-break."
                )

            st.dataframe(
                [
                    {
                        "": "Evidence score",
                        "Higher-ranked": round(a_ev, 4),
                        "Compared": round(b_ev, 4),
                        "Δ": round(a_ev - b_ev, 4),
                    },
                    {
                        "": "Opportunity score",
                        "Higher-ranked": round(a_pr, 3),
                        "Compared": round(b_pr, 3),
                        "Δ": round(a_pr - b_pr, 3),
                    },
                    {
                        "": "Rank score (product)",
                        "Higher-ranked": round(a_total, 4),
                        "Compared": round(b_total, 4),
                        "Δ": round(a_total - b_total, 4),
                    },
                ],
                hide_index=True,
                width="stretch",
            )

            evidence_favours_a = a_ev > b_ev
            priority_favours_a = a_pr > b_pr
            if evidence_favours_a and priority_favours_a:
                st.success(
                    "**Both factors agree.** The higher-ranked finding has both the stronger "
                    "evidence and the higher opportunity score — the ordering is not resting on "
                    "one number."
                )
            elif not evidence_favours_a and priority_favours_a:
                st.warning(
                    f"**The factors disagree, and opportunity decided it.** "
                    f"“{gap_title(b_gap)}” actually holds the stronger evidence score; it ranks "
                    "lower because its opportunity score is smaller. Say this before it is found."
                )
            elif evidence_favours_a and not priority_favours_a:
                st.warning(
                    "**The factors disagree, and evidence decided it.** The higher-ranked finding "
                    "wins on evidence despite the lower opportunity score."
                )
            else:
                st.error(
                    "**Both factors favour the lower-ranked finding.** That should be impossible "
                    "under `evidence × opportunity` — treat this artifact as suspect."
                )

            weights = {str(t.get("feature")): t for t in a_terms if isinstance(t, dict)}
            rows_out: list[dict[str, Any]] = []
            for term in b_terms:
                if not isinstance(term, dict):
                    continue
                name = str(term.get("feature"))
                mine = weights.get(name, {})
                a_contrib = float(mine.get("contribution") or 0.0)
                b_contrib = float(term.get("contribution") or 0.0)
                rows_out.append(
                    {
                        "feature": name,
                        "weight": mine.get("weight", term.get("weight")),
                        "higher-ranked": round(a_contrib, 4),
                        "compared": round(b_contrib, 4),
                        "Δ contribution": round(a_contrib - b_contrib, 4),
                    }
                )
            if rows_out:
                rows_out.sort(key=lambda row: -abs(float(row["Δ contribution"])))
                st.markdown("**Which features moved the evidence score** (largest gap first)")
                st.dataframe(rows_out, hide_index=True, width="stretch")
                top = rows_out[0]
                if float(top["Δ contribution"]) == 0.0:
                    st.caption(
                        "No feature separates these two — their evidence scores differ only "
                        "through the sigmoid, so the ordering rests on the opportunity score."
                    )

    if STABILITY_SHIPPED:
        st.divider()
        st.markdown("#### Does each finding survive resampling the corpus?")
        st.caption(
            f"{STABILITY.get('iterations', '?')} subsamples at "
            f"{float(STABILITY.get('fraction') or 0):.0%} of the corpus, without replacement, "
            "re-running need inference each time. This is the one measure here that genuinely "
            "separates findings — the evidence score spans about 0.06 across a whole run and "
            "never changes the order. It uses no human labels, and it is not a probability."
        )
        rows = []
        for gap in gaps:
            entry = STABILITY_SHIPPED.get(gap_title(gap))
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "Finding": gap_title(gap),
                    "Reappears": f"{float(entry.get('survival') or 0):.0%}",
                    "Same supporting signals": f"{float(entry.get('mean_jaccard') or 0):.0%}",
                    "Signals behind it": entry.get("baseline_signals"),
                }
            )
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            "The two columns answer different questions. *Reappears* asks whether the need is a "
            "property of the corpus or of this one draw. *Same supporting signals* asks whether "
            "the evidence behind it is the same evidence — a need can come back every time on a "
            "different set of reviews, and reporting only the first column would hide that."
        )
        weak = [row for row in rows if float(str(row["Reappears"]).rstrip("%")) < 90]
        if weak:
            st.warning(
                "**Least reproducible finding"
                + ("s" if len(weak) > 1 else "")
                + ":** "
                + ", ".join(f"{row['Finding']} ({row['Reappears']})" for row in weak)
                + ". Weaker than the rest of the packet and stated here rather than left to be "
                "found. It is still shown because the ranking is evidence × opportunity, and "
                "this measure is reported alongside rather than folded into it."
            )
        unshipped = STABILITY.get("unshipped")
        if isinstance(unshipped, dict) and unshipped:
            st.caption(
                "Needs that appeared in subsamples but not in the full run — what a different "
                "draw would have surfaced instead: "
                + ", ".join(
                    f"{title} ({float((entry or {}).get('survival') or 0):.0%})"
                    for title, entry in sorted(unshipped.items())
                )
            )

    st.divider()
    st.markdown("#### Is the confidence number a probability?")
    calibration = mapping(manifest.get("calibration"))
    calibrated = bool(calibration.get("calibrated"))
    label_count = calibration.get("label_count", 0) or 0
    if calibrated:
        st.success("This run reports cross-fitted calibrated probabilities.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Human labels", label_count)
        c2.metric("Brier", f"{calibration.get('brier', 0):.3f}")
        c3.metric("ECE", f"{calibration.get('ece', 0):.3f}")
        image_path = base / "calibration.png"
        if image_path.exists():
            st.image(str(image_path), caption="Out-of-fold reliability diagram")
    else:
        st.warning(
            f"**No — and we do not print one.** This run carries {label_count} human labels, so "
            "nothing on any screen is shown as a percentage or a probability. Findings are ranked "
            "by an evidence score, and the cards report what was actually checked (verified quote "
            "counts) instead of a number that would look like a probability and not be one."
        )
        st.caption(
            "To make it a probability: label ~30 need/roadmap pairs blind in "
            "`eval/labeling_worklist.json`, then fit cross-fitted Platt scaling. Isotonic is not "
            "selected below 1,000 labels. Until then the honest statement is the one above."
        )

    diagnostics = next(
        (
            gap["metadata"]["feature_diagnostics"]
            for gap in gaps
            if isinstance(gap.get("metadata"), dict)
            and isinstance(gap["metadata"].get("feature_diagnostics"), dict)
        ),
        None,
    )
    if diagnostics:
        with st.expander("What the evidence score is actually made of"):
            constant = diagnostics.get("constant_features") or {}
            discriminating = diagnostics.get("discriminating_features") or {}
            weight_constant = diagnostics.get("weight_on_constant_features")
            left, right = st.columns(2)
            if isinstance(weight_constant, (int, float)):
                left.metric(
                    "Weight on features that do not vary", f"{float(weight_constant):.0%}"
                )
            score_range = diagnostics.get("evidence_score_range")
            if isinstance(score_range, (int, float)):
                right.metric("Spread across all findings", f"{float(score_range):.3f}")
            if constant:
                st.markdown(
                    f"**Constant on this run ({len(constant)}).** These carry no discriminating "
                    "information here. We report them rather than renormalising them away."
                )
                st.dataframe(
                    [{"feature": k, "value": v} for k, v in sorted(constant.items())],
                    hide_index=True,
                    width="stretch",
                )
            if discriminating:
                st.markdown(f"**Discriminating ({len(discriminating)}).** These separate findings.")
                st.dataframe(
                    [{"feature": k, "spread": v} for k, v in sorted(discriminating.items())],
                    hide_index=True,
                    width="stretch",
                )
            note = diagnostics.get("note")
            if note:
                st.caption(str(note))

    with st.expander("Display bands, stated plainly"):
        st.markdown(
            "Cards show bands, not raw floats, because the underlying spread is too small to read "
            "as a difference. The cut points are display choices and are not used for ranking:\n\n"
            "- **Opportunity** — High ≥ 12, Moderate ≥ 8, Low below, on the recorded 0–20 score.\n"
            "- **Evidence** — Strong = at least 50 verified quotes and every cited signal "
            "verified; Moderate = at least 10 verified; Limited = fewer; Unverified = no "
            "verification record.\n\n"
            "Ranking uses the recorded scores themselves, which are in the comparison table above."
        )

    st.divider()
    st.markdown("#### Provenance")
    st.caption(f"Artifact directory: {base}")
    st.json(manifest or {"status": "No run manifest found", "artifact_directory": str(base)})
