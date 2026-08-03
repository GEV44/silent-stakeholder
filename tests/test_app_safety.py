"""Safety invariants for the judge-facing explorer.

These protect three failures that are worse than a broken page: showing an
unverified quote as if it were verified, showing synthetic data without saying
so, and presenting numbers produced by code that is no longer checked out.

The app module executes Streamlit calls at import time, so the pure helpers are
extracted from the AST and exercised directly. That keeps the tests offline,
fast, and free of a Streamlit runtime dependency.
"""

from __future__ import annotations

import ast
import hashlib
import html
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _load_helper(name: str) -> Any:
    """Compile one top-level function out of app.py without running the page."""

    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace: dict[str, Any] = {"hashlib": hashlib, "html": html, "ROOT": ROOT, "Any": Any}
    exec(  # noqa: S102 - compiling our own source, not user input
        compile(ast.Module(body=[node], type_ignores=[]), "<app-helper>", "exec"),
        namespace,
    )
    return namespace[name]


def _verdict_colors() -> dict[str, str]:
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.Assign)
        and any(getattr(t, "id", None) == "VERDICT_COLORS" for t in item.targets)
    )
    namespace: dict[str, Any] = {}
    exec(  # noqa: S102 - our own source
        compile(ast.Module(body=[node], type_ignores=[]), "<app-colors>", "exec"),
        namespace,
    )
    return namespace["VERDICT_COLORS"]


def test_every_shipped_planning_state_has_a_colour() -> None:
    """Both verdict vocabularies must resolve, or real rows render grey.

    The cards colour by ``public_planning_state``, not ``verdict``. Removing the
    planning-state keys once made 100% of rows grey on every artifact because
    real gaps always carry that metadata.
    """

    colors = _verdict_colors()
    grey = "#64748b"
    payload = json.loads(
        (ROOT / "examples" / "demo" / "top_gaps.json").read_text(encoding="utf-8")
    )
    rows = payload["gaps"] if isinstance(payload, dict) else payload
    assert rows
    for row in rows:
        metadata = row.get("metadata") or {}
        state = str(metadata.get("public_planning_state") or row["verdict"]).upper()
        assert state in colors, f"{state} has no colour and would render grey"
        assert colors[state] != grey, f"{state} falls back to the grey default"


def test_stale_artifact_is_detected_in_both_directions() -> None:
    """A fresh artifact must pass, and a drifted hash must be caught."""

    stale_artifact_reasons = _load_helper("stale_artifact_reasons")
    manifest = json.loads(
        (ROOT / "examples" / "demo" / "run_manifest.json").read_text(encoding="utf-8")
    )

    assert stale_artifact_reasons(manifest) == []

    drifted_config = json.loads(json.dumps(manifest))
    drifted_config["reproducibility"]["pipeline_config_sha256"] = "0" * 64
    assert stale_artifact_reasons(drifted_config)

    drifted_code = json.loads(json.dumps(manifest))
    drifted_code["reproducibility"]["inference_contract_sha256"] = "0" * 64
    assert stale_artifact_reasons(drifted_code)

    # A manifest with no reproducibility block cannot be judged, so the safety
    # gate must warn rather than silently treating unknown provenance as fresh.
    assert stale_artifact_reasons({})


def test_stale_detection_fails_closed_when_it_cannot_verify(tmp_path: Path) -> None:
    """Unverifiable must warn, never pass silently.

    The first version swallowed OSError and returned no reasons, so an artifact
    whose pipeline source had been renamed or deleted rendered with no banner —
    the exact case the gate exists to catch.
    """

    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "stale_artifact_reasons"
    )
    # Point ROOT at an empty tree so every referenced file is missing.
    namespace: dict[str, Any] = {"hashlib": hashlib, "ROOT": tmp_path, "Any": Any}
    exec(  # noqa: S102 - our own source
        compile(ast.Module(body=[node], type_ignores=[]), "<app-helper>", "exec"),
        namespace,
    )
    reasons = namespace["stale_artifact_reasons"](
        {
            "reproducibility": {
                "pipeline_config_sha256": "0" * 64,
                "inference_contract_sha256": "0" * 64,
                "inference_contract_files": ["src/needs.py", "src/gaps.py"],
            }
        }
    )
    assert reasons, "a run that cannot be verified must not be reported as fresh"
    assert any("unreadable" in reason for reason in reasons)


def test_only_verified_spans_are_offered_for_display() -> None:
    """A quote is displayable only when the verifier marked that span valid.

    The app previously fell back to the full raw signal text when a quote was
    missing, which on screen is indistinguishable from a verified quote.
    """

    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "verification.json" in source, "the app must load verification records"
    assert 'first(source, "text", "review"' not in source, (
        "the raw-signal-text fallback must stay removed: it renders unverified "
        "text that looks identical to a verified quote"
    )


def test_rank_score_is_the_product_it_claims_to_be() -> None:
    """The rank comparator decomposes rank into evidence x priority.

    If the shipped artifact's `rank_score` stopped equalling that product, the
    tab would narrate arithmetic the pipeline did not perform — a worse failure
    than showing nothing, because it looks authoritative.
    """

    payload = json.loads(
        (ROOT / "examples" / "demo" / "top_gaps.json").read_text(encoding="utf-8")
    )
    rows = payload["gaps"] if isinstance(payload, dict) else payload
    assert rows
    checked = 0
    for row in rows:
        breakdown = (row.get("metadata") or {}).get("score_breakdown") or {}
        evidence = breakdown.get("evidence_score")
        priority = row.get("opportunity_score")
        total = row.get("rank_score")
        if evidence is None or priority is None or total is None:
            continue
        assert abs(float(evidence) * float(priority) - float(total)) < 1e-3, (
            f"gap {row.get('id')}: rank_score {total} != evidence {evidence} x "
            f"priority {priority}"
        )
        checked += 1
    assert checked, "no gap carried the fields the rank comparator renders"


def test_evidence_identity_survives_rewording_but_not_new_evidence() -> None:
    """Dedupe must key on evidence, not the title.

    A competitor deduped issue drafts by exact title, so any rewording filed a
    second copy of the same finding into a real repository. The identity of a
    finding is the set of signals it rests on.
    """

    evidence_identity = _load_helper("evidence_identity")
    base = {"evidence": {"signal_ids": ["S000002", "S000001"]}, "latent_need": "Original wording"}
    reworded = {**base, "latent_need": "Completely different phrasing of the same need"}
    reordered = {"evidence": {"signal_ids": ["S000001", "S000002"]}, "latent_need": "x"}
    different = {"evidence": {"signal_ids": ["S000003"]}, "latent_need": "Original wording"}

    assert evidence_identity(base) == evidence_identity(reworded)
    assert evidence_identity(base) == evidence_identity(reordered), "order must not matter"
    assert evidence_identity(base) != evidence_identity(different)


def test_ticket_draft_never_files_and_never_leaks_unverified_text() -> None:
    """The draft is an artifact, not an action, and carries only verified spans."""

    source = (ROOT / "app.py").read_text(encoding="utf-8")
    start = source.index("def ticket_draft")
    end = source.index("\ndef ", start + 1)
    body = source[start:end]

    # No network or repository mutation anywhere in the draft path.
    for forbidden in ("requests.", "http", "gh ", "create_issue", "POST"):
        assert forbidden not in body, f"ticket_draft must not reach out: found {forbidden!r}"

    # Quotes come only from the verification-gated helper.
    assert "verified_spans(" in body
    assert "signals_by_id" not in body, (
        "the draft must not read raw signal text; only verification-valid spans"
    )
    assert "DRAFT — not filed" in body


def test_no_artifact_content_reaches_raw_html_unescaped() -> None:
    """Every unsafe_allow_html site must be a literal or an escaped value.

    An f-string interpolating artifact content into markup is the one way
    model- or user-authored text could execute in the page.
    """

    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not any(
            kw.arg == "unsafe_allow_html"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in node.keywords
        ):
            continue
        for arg in node.args:
            for piece in ast.walk(arg):
                # A formatted value is only safe if it is html.escape(...) or a
                # call into the fixed colour lookup.
                if isinstance(piece, ast.FormattedValue):
                    inner = piece.value
                    escaped = (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "escape"
                    )
                    from_lookup = (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Name)
                        and inner.func.id == "verdict_color"
                    )
                    if not (escaped or from_lookup):
                        offenders.append(getattr(node, "lineno", -1))
    assert not offenders, (
        f"unescaped interpolation into unsafe_allow_html at line(s) {sorted(set(offenders))}"
    )


def test_html_builder_helpers_escape_everything_they_interpolate() -> None:
    """The `*_html` builders must escape, because the test above cannot see inside them.

    Markup is composed by helpers (`quote_html`, `chip_html`, `fact_html`) and the
    result is passed to ``st.markdown`` as a plain name. That satisfies the
    call-site check trivially, so without this test the helpers would be an
    unchecked hole straight through it — and they are exactly where review text
    and roadmap titles, both attacker-controlled, reach the page.
    """

    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    builders = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_html")
    ]
    assert builders, "no *_html builders found; did the markup helpers get renamed?"

    offenders: list[str] = []
    for builder in builders:
        for piece in ast.walk(builder):
            if not isinstance(piece, ast.FormattedValue):
                continue
            inner = piece.value
            escaped = (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "escape"
            )
            from_lookup = (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "verdict_color"
            )
            if not (escaped or from_lookup):
                offenders.append(f"{builder.name}:{getattr(piece, 'lineno', -1)}")
    assert not offenders, f"unescaped interpolation inside markup builders: {offenders}"


def test_review_text_cannot_render_as_markdown() -> None:
    """A review containing a link or an image must display as text, not render.

    The corpus is public and anyone can publish a review, so review text is
    attacker-controlled. Rendered as markdown it becomes a clickable link or a
    remote image beacon on a screen a judge is being asked to trust.
    """

    quote_html = _load_helper("quote_html")
    hostile = '[click me](https://evil.example) ![](https://evil.example/x.png) <script>x</script>'
    rendered = quote_html(hostile)

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    # The markdown stays literal: escaped into a div, Streamlit never parses it.
    assert "](https://evil.example)" in rendered
    assert rendered.startswith('<div class="quote">')


def test_demo_fixture_spans_all_verify() -> None:
    """The shipped fixture must not need the gate to hide anything."""

    demo = ROOT / "examples" / "demo"
    verification = json.loads((demo / "verification.json").read_text(encoding="utf-8"))
    records = (
        verification["verification"]
        if isinstance(verification, dict)
        else verification
    )
    assert records
    for record in records:
        reports = record.get("quote_reports") or []
        assert reports, f"gap {record.get('gap_id')} has no quote reports"
        assert all(item.get("valid") for item in reports), (
            f"gap {record.get('gap_id')} ships an unverified span"
        )
