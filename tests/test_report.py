"""Invariants for the standalone HTML report.

The report is the surface most likely to be sent to someone outside the team, so
two things matter more here than anywhere else: it must not become a vector for
the review text it displays, and it must not overstate the run it was built
from.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import report

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "examples" / "demo"


def _valid_empty_run(directory: Path) -> Path:
    directory.mkdir(parents=True)
    declarations = {}
    for name in report.REQUIRED_REPORT_ARTIFACTS:
        path = directory / name
        path.write_text("[]", encoding="utf-8")
        declarations[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema_version": "1.0",
        }
    (directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "mode": "demo_fixture",
                "scope": {"product": "ExamplePress for Android (synthetic)"},
                "artifacts": declarations,
            }
        ),
        encoding="utf-8",
    )
    return directory


@pytest.fixture(scope="module")
def page() -> str:
    return report.render(report.build_payload(DEMO), "test")


def test_report_makes_no_external_requests(page: str) -> None:
    """One file, no network. The live demo runs with the cable out.

    An external font or script would also be silently blocked by the artifact
    host's CSP, which fails as a blank page rather than a visible error.
    """

    markup = page.split('<script id="payload"')[0]
    for pattern in ("http://", "https://", "//cdn", "<link", "@import"):
        assert pattern not in markup, f"report reaches outside itself: {pattern!r}"


def test_no_artifact_text_is_ever_written_as_markup(page: str) -> None:
    """Review text is attacker-controlled, so it must never touch innerHTML.

    Anyone can publish a review in the corpus. Rendered as markup, one could put
    a phishing link or a remote image beacon on a page a judge is being asked to
    trust. The data travels as JSON and is written with textContent, which makes
    that impossible by construction rather than by escaping discipline.
    """

    script = page.split("<script>")[-1].split("</script>")[0]
    assert "innerHTML" not in script.replace(
        "so nothing is ever assigned to innerHTML", ""
    ), "the report must build every node with textContent"
    assert "insertAdjacentHTML" not in script
    assert "document.write" not in script


def test_payload_cannot_break_out_of_its_script_tag(page: str) -> None:
    """A '<' inside the JSON would let review text close the tag and inject.

    Escaping the character itself is what makes a review containing
    '</script><script>...' inert.
    """

    payload = page.split('<script id="payload" type="application/json">')[1]
    payload = payload.split("</script>")[0]
    assert "<" not in payload
    json.loads(payload)


def test_hostile_review_text_survives_as_data_not_markup(tmp_path: Path) -> None:
    """End to end: a malicious review reaches the page as characters."""

    hostile = '</script><script>alert(1)</script><img src=x onerror=alert(2)>'
    (tmp_path / "top_gaps.json").write_text(
        json.dumps(
            [
                {
                    "id": "G1",
                    "rank": 1,
                    "latent_need": hostile,
                    "verdict": "IGNORED",
                    "opportunity_score": 9.0,
                    "evidence": {"signal_ids": ["S000001"]},
                }
            ]
        ),
        encoding="utf-8",
    )
    page = report.render(report.build_payload(tmp_path), "t")

    # The hostile string is present as data, but no executable tag was formed.
    assert "alert(1)" in page, "the value should still round-trip as data"
    assert "<script>alert(1)" not in page
    assert "<img src=x" not in page


def test_only_verified_quotes_reach_the_report() -> None:
    """A quote is shown only if the verifier matched it against source text.

    An unverified quote on screen is indistinguishable from a verified one, and
    would undo the whole anti-fabrication argument.
    """

    payload = report.build_payload(DEMO)
    verification = json.loads((DEMO / "verification.json").read_text(encoding="utf-8"))
    rows = verification["verification"] if isinstance(verification, dict) else verification
    valid_spans = {
        str(item.get("matched_span"))
        for row in rows
        for item in row.get("quote_reports") or []
        if item.get("valid")
    }
    shown = {p["span"] for f in payload["findings"] for p in f["proof"]}
    assert shown, "the fixture should surface at least one verified quote"
    assert shown <= valid_spans, "the report displayed a span the verifier did not confirm"


def test_synthetic_run_is_reported_as_demo() -> None:
    """The fixture must never present itself as a product finding."""

    payload = report.build_payload(DEMO)
    assert payload["demo"] is True
    assert payload["counts"]["findings"] == len(payload["findings"])


def test_absent_run_directory_degrades_instead_of_lying(tmp_path: Path) -> None:
    """No manifest means unknown provenance, which must resolve to demo."""

    (tmp_path / "top_gaps.json").write_text("[]", encoding="utf-8")
    payload = report.build_payload(tmp_path)
    assert payload["demo"] is True
    assert payload["counts"]["findings"] == 0


def test_confidence_is_never_printed_as_a_probability(page: str) -> None:
    """Invariant 5, enforced at the presentation layer too."""

    payload = report.build_payload(DEMO)
    assert payload["calibrated"] is False
    assert payload["labels"] == 0
    assert "we do not print one" in page
    # No finding carries a percentage-shaped confidence field into the page.
    for finding in payload["findings"]:
        assert "confidence" not in finding


def test_every_finding_states_a_verdict_a_team_can_act_on() -> None:
    """Internal verdict names are translated into the reader's triage language."""

    payload = report.build_payload(DEMO)
    allowed = {label for label, _ in report.PLANNING.values()}
    assert payload["findings"]
    for finding in payload["findings"]:
        assert finding["state"] in allowed
        assert finding["tone"] in {"critical", "warn", "info", "ok", "muted"}


def test_both_themes_define_the_same_tokens(page: str) -> None:
    """A token defined in one theme and missed in the other renders unstyled.

    The viewer's toggle stamps data-theme on the root, so both explicit blocks
    have to carry the full palette, not a partial override of the media query.
    """

    def tokens(block: str) -> set[str]:
        return set(re.findall(r"(--[a-z0-9-]+):", block))

    light = page.split(':root[data-theme="light"] {')[1].split("}")[0]
    dark = page.split(':root[data-theme="dark"] {')[1].split("}")[0]
    assert tokens(light) == tokens(dark), "the two themes define different tokens"
    assert "--accent" in tokens(light)


# ---------------------------------------------------------------------------
# REQ-A-08 hardening: injectable title, hostile URLs, licence-safe default
# ---------------------------------------------------------------------------


def test_title_is_data_not_markup() -> None:
    """A CLI title like '</title><script>' must never become executable markup."""

    payload = report.build_payload(DEMO)
    hostile = '</title><script>alert(1)</script>'
    page = report.render(payload, hostile)
    assert "</title><script>" not in page
    assert "&lt;/title&gt;" in page


def test_roadmap_urls_admit_only_the_verified_repository() -> None:
    """A `javascript:` or off-repo URL from an artifact renders as text, not a link.

    Roadmap records are artifact data and an artifact can be malicious; an
    unvalidated href is an active navigation on a page a judge trusts.
    """

    repo = "wordpress-mobile/WordPress-Android"
    good = f"https://github.com/{repo}/issues/3913"
    assert report.safe_repo_url(good, repo) == good
    for bad in (
        "javascript:alert(1)",
        "data:text/html,x",
        "https://evil.example/github.com/x",
        "https://github.com/other-org/other-repo/issues/1",
        "https://user:pw@github.com/wordpress-mobile/WordPress-Android/issues/1",
        f"https://github.com/{repo}/issues/1\x01",
        "",
        None,
    ):
        assert report.safe_repo_url(bad, repo) == "", f"admitted: {bad!r}"


def test_public_profile_withholds_real_run_quotes(tmp_path: Path) -> None:
    """The default export of a REAL run contains no verbatim review text.

    The corpus licence is unresolved; a single easy-to-attach HTML file holding
    real review text is exactly the accident this prevents. IDs and counts stay.
    """

    secret = "VERBATIM-REVIEW-TEXT-THAT-MUST-NOT-LEAK"
    (tmp_path / "top_gaps.json").write_text(
        json.dumps([
            {
                "id": "G1", "rank": 1, "latent_need": "Need", "verdict": "IGNORED",
                "opportunity_score": 9.0, "evidence": {"signal_ids": ["S000001"]},
            }
        ]),
        encoding="utf-8",
    )
    (tmp_path / "verification.json").write_text(
        json.dumps([
            {
                "gap_id": "G1", "valid": True,
                "quote_reports": [
                    {"id": "S000001", "valid": True, "matched_span": secret}
                ],
            }
        ]),
        encoding="utf-8",
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"mode": "production", "scope": {}}), encoding="utf-8"
    )

    public = report.build_payload(tmp_path, profile="public")
    assert public["withheldQuotes"] == 1
    assert public["findings"][0]["proof"][0]["span"] == ""
    assert secret not in report.render(public, "t")

    internal = report.build_payload(tmp_path, profile="internal")
    assert internal["withheldQuotes"] == 0
    assert internal["findings"][0]["proof"][0]["span"] == secret

    # The synthetic demo fixture stays viewable on the public profile: its
    # spans are synthetic, and a blank demo would punish honesty.
    demo = report.build_payload(DEMO, profile="public")
    assert demo["withheldQuotes"] == 0


def test_page_carries_a_deny_by_default_csp(page: str) -> None:
    """Belt over suspenders: even a slipped bug cannot beacon out."""

    assert 'http-equiv="Content-Security-Policy"' in page
    assert "default-src 'none'" in page
    assert "form-action 'none'" in page
    assert "script-src 'sha256-" in page
    assert "script-src 'unsafe-inline'" not in page


def test_manifest_hash_drift_fails_before_replacing_a_good_report(tmp_path: Path) -> None:
    run = _valid_empty_run(tmp_path / "run")
    destination = tmp_path / "report.html"
    destination.write_bytes(b"known-good-report")
    (run / "top_gaps.json").write_text('[{"id":"G-drift"}]', encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        report.main(["--artifacts", str(run), "--out", str(destination)])
    assert caught.value.code == 2
    assert destination.read_bytes() == b"known-good-report"


def test_duplicate_manifest_key_and_undeclared_optional_artifact_fail(tmp_path: Path) -> None:
    duplicate = _valid_empty_run(tmp_path / "duplicate")
    (duplicate / "run_manifest.json").write_text(
        '{"mode":"demo_fixture","mode":"production","artifacts":{}}',
        encoding="utf-8",
    )
    with pytest.raises(report.ArtifactValidationError, match="duplicate JSON key"):
        report.validate_run_artifacts(duplicate)

    undeclared = _valid_empty_run(tmp_path / "undeclared")
    (undeclared / "stability.json").write_text("{}", encoding="utf-8")
    with pytest.raises(report.ArtifactValidationError, match="undeclared stability"):
        report.validate_run_artifacts(undeclared)


def test_private_export_requires_acknowledgement_and_outside_destination(tmp_path: Path) -> None:
    run = _valid_empty_run(tmp_path / "run")
    outside = tmp_path / "private.html"
    with pytest.raises(SystemExit):
        report.main(
            [
                "--artifacts",
                str(run),
                "--out",
                str(outside),
                "--profile",
                "private-evidence",
            ]
        )
    assert not outside.exists()

    inside = ROOT / "private-export-test.html"
    with pytest.raises(SystemExit):
        report.main(
            [
                "--artifacts",
                str(run),
                "--out",
                str(inside),
                "--profile",
                "private-evidence",
                "--acknowledge-private-export",
            ]
        )
    assert not inside.exists()


def test_validated_public_export_replaces_atomically(tmp_path: Path) -> None:
    run = _valid_empty_run(tmp_path / "run")
    destination = tmp_path / "public.html"
    assert report.main(["--artifacts", str(run), "--out", str(destination)]) == 0
    content = destination.read_text(encoding="utf-8")
    assert "Synthetic demo data" in content
    assert "__SCRIPT_HASH__" not in content


def test_demo_report_distinguishes_fixture_text_from_restricted_real_text(page: str) -> None:
    payload = report.build_payload(DEMO)
    assert payload["demo"] is True
    assert payload["withheldQuotes"] == 0
    assert all(proof["span"] for finding in payload["findings"] for proof in finding["proof"])
    assert "This corpus is synthetic fixture text with pinned dates" in page
    assert "All displayed review text is synthetic fixture content" in page
    assert "no alternate-clustering omission claim" in page
    assert "This shortfall is not evidence that no fifth need exists" in page


def test_committed_public_report_matches_current_exporter_and_artifacts() -> None:
    """CI binds the checked-in HTML to current report code and declared JSON.

    The run manifest binds pipeline artifacts, not the HTML that embeds it. A
    deterministic render-equivalence check closes that separate publication
    boundary without pretending the manifest can hash a file containing itself.
    """

    validated = report.validate_run_artifacts(DEMO)
    payload = report.build_payload(DEMO, profile="public", validated=validated)
    expected = report.render(payload, "The Silent Stakeholder — roadmap gap report")
    actual = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    assert actual == expected


def test_ticket_draft_is_a_prefill_that_never_leaks_quotes(tmp_path: Path) -> None:
    """Loop closure stays human-reviewed and inside the public evidence boundary."""

    repo = "wordpress-mobile/WordPress-Android"
    (tmp_path / "top_gaps.json").write_text(
        json.dumps(
            [
                {
                    "id": "G1",
                    "rank": 1,
                    "latent_need": "Need",
                    "verdict": "IGNORED",
                    "opportunity_score": 9.0,
                    "evidence": {"signal_ids": ["S000001"]},
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"mode": "production", "scope": {"repository": repo}}),
        encoding="utf-8",
    )
    payload = report.build_payload(tmp_path)
    assert payload["issuesNew"] == f"https://github.com/{repo}/issues/new"
    assert report.build_payload(DEMO)["issuesNew"] == ""

    script = report.render(payload, "t").split("<script>")[-1]
    assert "Draft the ticket" in script
    assert "q.id" in script, "the body cites signal IDs"
    assert "licence-restricted and withheld" in script
    for banned in ("fetch(", "XMLHttpRequest", "method: 'POST'", 'method: "POST"'):
        assert banned not in script, f"the draft must not write anywhere: {banned}"


def test_single_source_limitation_is_disclosed(page: str) -> None:
    """State the confidence-feature limitation before a reviewer has to ask."""

    assert "one source type" in page
    assert "corroboration (support tickets, churn notes) is not modelled" in page
