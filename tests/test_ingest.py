from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.ingest import (
    GitHubAPIError,
    GitHubClient,
    IngestError,
    derive_priority_metadata,
    ingest_reviews_csv,
    load_github_fixture,
    normalize_github_roadmap,
    run_ingestion,
)
from src.schema import PriorityTier, RoadmapItemType

FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        links: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.links = links or {}

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"unexpected request: {url}")
        return self.responses.pop(0)


def test_review_ingest_filters_deduplicates_and_is_reorder_stable(tmp_path) -> None:
    original = (FIXTURES / "reviews.csv").read_text(encoding="utf-8")
    source = tmp_path / "reviews.csv"
    source.write_text(original, encoding="utf-8")
    first = ingest_reviews_csv(source, "org.wordpress.android", offline=True)

    lines = original.splitlines()
    source.write_text(
        "\n".join([lines[0], *reversed(lines[1:])]) + "\n",
        encoding="utf-8",
    )
    second = ingest_reviews_csv(source, "org.wordpress.android", offline=True)

    assert [signal.id for signal in first.signals] == [signal.id for signal in second.signals]
    assert first.stats.total_rows == 7
    assert first.stats.matching_rows == 6
    assert first.stats.blank_text_rows == 1
    assert first.stats.invalid_rows == 1
    assert first.stats.duplicates_removed == 1
    assert first.stats.emitted_signals == 3
    slow_low_rating = next(
        signal for signal in first.signals if signal.text == "Export is slow" and signal.rating == 2
    )
    assert slow_low_rating.source_id == "102"
    assert slow_low_rating.metadata["duplicate_count"] == 2
    # Same words with a contradictory rating are not erased by dedup.
    assert len([signal for signal in first.signals if signal.text == "Export is slow"]) == 2


def test_review_ingest_has_clear_missing_package_error() -> None:
    with pytest.raises(IngestError, match="contains no rows"):
        ingest_reviews_csv(
            FIXTURES / "reviews.csv",
            "missing.package",
            offline=True,
        )


def test_offline_mode_rejects_review_url_before_network() -> None:
    with pytest.raises(IngestError, match="offline mode forbids"):
        ingest_reviews_csv(
            "https://example.invalid/reviews.csv",
            "org.wordpress.android",
            offline=True,
        )


def test_github_pagination_follows_link_and_only_sends_params_once() -> None:
    second_url = "https://api.github.com/repos/owner/repo/issues?state=all&per_page=100&page=2"
    session = FakeSession(
        [
            FakeResponse(
                [{"number": 1}],
                links={"next": {"url": second_url, "rel": "next"}},
            ),
            FakeResponse([{"number": 2}]),
        ]
    )
    client = GitHubClient(session=session)

    result = client.paginate(
        "/repos/owner/repo/issues",
        params={"state": "all", "per_page": 100},
    )

    assert [item["number"] for item in result] == [1, 2]
    assert session.calls[0]["params"] == {"state": "all", "per_page": 100}
    assert session.calls[1]["params"] is None
    assert session.calls[1]["url"] == second_url


def test_github_pagination_rejects_cross_host_next_link() -> None:
    session = FakeSession(
        [
            FakeResponse(
                [{"number": 1}],
                headers={"Link": '<https://attacker.example/steal>; rel="next"'},
            )
        ]
    )

    with pytest.raises(GitHubAPIError, match="different host"):
        GitHubClient(session=session).paginate("/repos/owner/repo/issues")


def test_github_rate_limit_error_tells_user_how_to_fix() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {"message": "API rate limit exceeded"},
                status_code=403,
                headers={
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": "12345",
                },
            )
        ]
    )

    with pytest.raises(GitHubAPIError, match="GITHUB_TOKEN"):
        GitHubClient(session=session).paginate("/repos/owner/repo/issues")


def test_all_state_history_requires_authentication_before_network() -> None:
    session = FakeSession([])

    with pytest.raises(GitHubAPIError, match="requires GITHUB_TOKEN"):
        GitHubClient(session=session).fetch_repository("owner/repo", state="all")

    assert not session.calls


def test_normalization_drops_prs_and_derives_priority() -> None:
    milestones, issues = load_github_fixture(FIXTURES / "github_wordpress.json")
    items = normalize_github_roadmap(
        "wordpress-mobile/WordPress-Android",
        milestones,
        issues,
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert len(items) == 3
    assert {item.number for item in items if item.type == RoadmapItemType.ISSUE} == {
        10,
        11,
    }
    milestone = next(item for item in items if item.type == RoadmapItemType.MILESTONE)
    assert milestone.priority.tier == PriorityTier.BACKLOG
    p1_issue = next(item for item in items if item.number == 10)
    assert p1_issue.priority.tier == PriorityTier.HIGH
    assert p1_issue.priority.has_explicit_priority is True
    assert p1_issue.priority.is_low_priority is False
    future_issue = next(
        item for item in items if item.type == RoadmapItemType.ISSUE and item.number == 11
    )
    assert future_issue.milestone_due == datetime(2027, 12, 31, tzinfo=UTC)
    assert future_issue.priority.is_low_priority is True
    assert future_issue.priority.tier == PriorityTier.BACKLOG


def test_priority_old_unmilestoned_issue_is_auditable() -> None:
    priority = derive_priority_metadata(
        labels=[],
        milestone_title=None,
        milestone_due=None,
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
        state="open",
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert priority.tier == PriorityTier.LOW
    assert priority.is_low_priority
    assert "not assigned to a milestone" in priority.reasons
    assert "open for more than one year" in priority.reasons


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("[Pri] Critical", PriorityTier.CRITICAL),
        ("[Pri] High", PriorityTier.HIGH),
        ("[Pri] Medium", PriorityTier.MEDIUM),
        ("[Pri] Low", PriorityTier.LOW),
    ],
)
def test_bracketed_priority_labels_are_read(label, expected) -> None:
    """`[Pri] High` is the convention the target repository actually uses.

    The rules matched `high priority` and `priority: high` but nothing
    bracketed, and `_label_matches` did not treat `[`/`]` as separators -- so
    `\\bpri high\\b` could not match `[pri] high`. Measured on the real
    roadmap, 234 of 774 open issues carry one of these labels and every one
    was recorded as having no stated priority.
    """

    priority = derive_priority_metadata(
        labels=[label, "[Type] Bug"],
        milestone_title=None,
        milestone_due=None,
        created_at=None,
        state="open",
        as_of=None,
    )

    assert priority.tier == expected
    assert priority.has_explicit_priority
    assert priority.matched_labels == [label]


def test_non_priority_bracketed_labels_stay_unmatched() -> None:
    """Stripping brackets must not turn every namespaced label into a priority."""

    priority = derive_priority_metadata(
        labels=["[Type] Bug", "[Feature] Editor", "Accessibility"],
        milestone_title="7.0",
        milestone_due=None,
        created_at=None,
        state="open",
        as_of=None,
    )

    assert priority.tier == PriorityTier.UNSPECIFIED
    assert not priority.has_explicit_priority
    assert priority.matched_labels == []


def test_offline_run_writes_complete_artifacts(tmp_path) -> None:
    result = run_ingestion(
        reviews_csv=FIXTURES / "reviews.csv",
        package_name="org.wordpress.android",
        repository="wordpress-mobile/WordPress-Android",
        out_dir=tmp_path / "out",
        github_fixture=FIXTURES / "github_wordpress.json",
        offline=True,
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
    )

    signals_payload = json.loads((tmp_path / "out" / "signals.json").read_text(encoding="utf-8"))
    roadmap_payload = json.loads((tmp_path / "out" / "roadmap.json").read_text(encoding="utf-8"))
    assert len(signals_payload) == len(result.signals) == 3
    assert len(roadmap_payload) == len(result.roadmap) == 3
    scope_payload = json.loads((tmp_path / "out" / "ingest_scope.json").read_text(encoding="utf-8"))
    assert all(item["id"].startswith("S") for item in signals_payload)
    assert all(item["id"].startswith("R") for item in roadmap_payload)
    assert scope_payload["github"]["state_scope"] == "fixture-provided"
    assert scope_payload["github"]["pull_requests_dropped"] is True
    assert scope_payload["github"]["retrieved_at"] is None


def test_fixture_provenance_is_preserved_and_processing_time_is_separate(tmp_path) -> None:
    payload = json.loads((FIXTURES / "github_wordpress.json").read_text(encoding="utf-8"))
    payload.update(
        {
            "repository": "wordpress-mobile/WordPress-Android",
            "retrieved_at": "2026-07-30T12:34:56+00:00",
            "state_scope": "all",
            "api_version": "2026-03-10",
        }
    )
    fixture = tmp_path / "snapshot.json"
    fixture.write_text(json.dumps(payload), encoding="utf-8")

    result = run_ingestion(
        reviews_csv=FIXTURES / "reviews.csv",
        package_name="org.wordpress.android",
        repository="wordpress-mobile/WordPress-Android",
        out_dir=tmp_path / "out",
        github_fixture=fixture,
        offline=True,
        as_of=datetime(2026, 8, 1, tzinfo=UTC),
    )
    github = result.scope_metadata["github"]
    assert github["state_scope"] == "all"
    assert github["retrieved_at"] == "2026-07-30T12:34:56+00:00"
    assert github["api_version"] == "2026-03-10"
    assert github["processed_at"] != github["retrieved_at"]


def test_fixture_repository_mismatch_fails_closed(tmp_path) -> None:
    payload = json.loads((FIXTURES / "github_wordpress.json").read_text(encoding="utf-8"))
    payload["repository"] = "someone/another-repository"
    fixture = tmp_path / "wrong-repository.json"
    fixture.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IngestError, match="does not match"):
        run_ingestion(
            reviews_csv=FIXTURES / "reviews.csv",
            package_name="org.wordpress.android",
            repository="wordpress-mobile/WordPress-Android",
            out_dir=tmp_path / "out",
            github_fixture=fixture,
            offline=True,
        )


def test_offline_run_requires_github_fixture(tmp_path) -> None:
    with pytest.raises(IngestError, match="requires --github-fixture"):
        run_ingestion(
            reviews_csv=FIXTURES / "reviews.csv",
            package_name="org.wordpress.android",
            repository="wordpress-mobile/WordPress-Android",
            out_dir=tmp_path,
            offline=True,
        )
