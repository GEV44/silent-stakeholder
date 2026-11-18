from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.io_utils import ArtifactIOError
from src.run import main
from src.summary import load_briefing, render_markdown

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"


def test_briefing_matches_demo_artifacts():
    briefing = load_briefing(DEMO)
    assert len(briefing["findings"]) == 4
    first = briefing["findings"][0]
    assert first["rank"] == 1
    assert first["roadmap_title"] == "Warn the author when a media upload fails or stalls"
    assert first["evidence_count"] >= 2
    scores = [f["rank_score"] for f in briefing["findings"]]
    assert scores == sorted(scores, reverse=True)


def test_markdown_lists_every_finding_with_verdict():
    text = render_markdown(load_briefing(DEMO))
    assert text.startswith("# ")
    assert text.count("\n## ") == 4
    assert "MISUNDERSTOOD" in text
    assert "uncalibrated" in text.lower()


def test_missing_directory_gives_clear_error(tmp_path):
    with pytest.raises(ArtifactIOError, match="top_gaps.json"):
        load_briefing(tmp_path)


def test_malformed_gaps_payload_rejected(tmp_path):
    shutil.copytree(DEMO, tmp_path / "a")
    (tmp_path / "a" / "top_gaps.json").write_text(json.dumps({"gaps": "nope"}), encoding="utf-8")
    with pytest.raises(ArtifactIOError, match="gaps"):
        load_briefing(tmp_path / "a")


def test_unknown_roadmap_id_is_reported_not_crash(tmp_path):
    shutil.copytree(DEMO, tmp_path / "a")
    payload = json.loads((tmp_path / "a" / "top_gaps.json").read_text(encoding="utf-8"))
    payload["gaps"][0]["matched_roadmap_id"] = "R999999"
    (tmp_path / "a" / "top_gaps.json").write_text(json.dumps(payload), encoding="utf-8")
    assert load_briefing(tmp_path / "a")["findings"][0]["roadmap_title"] is None


def test_cli_summary_prints_and_writes(tmp_path, capsys):
    out = tmp_path / "brief.md"
    assert main(["summary", "--artifacts", str(DEMO), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# ")
    assert "Recoverable media uploads" in capsys.readouterr().out


def test_cli_summary_bad_dir_exit_code_2(tmp_path, capsys):
    assert main(["summary", "--artifacts", str(tmp_path)]) == 2
    assert "pipeline error" in capsys.readouterr().err
