from __future__ import annotations

import numpy as np
import pytest

from src.embedding import HashingEmbedder, cosine_similarity_matrix
from src.needs import (
    GeminiNeedExtractor,
    GeminiNeedMerger,
    NeedConfig,
    infer_needs,
    opportunity_score,
)


def _signals():
    return [
        {
            "id": "S0001",
            "source": "review",
            "text": "Export fails every time and loses my draft",
            "rating": 1,
        },
        {
            "id": "S0002",
            "source": "ticket",
            "text": "Export fails when I save a draft",
            "rating": 2,
        },
        {
            "id": "S0003",
            "source": "review",
            "text": "I love the notification controls",
            "rating": 5,
        },
    ]


def test_hashing_embeddings_are_deterministic_and_normalized():
    embedder = HashingEmbedder(dimensions=128)
    first = embedder.encode(["Export a draft", "", "Notification controls"])
    second = embedder.encode(["Export a draft", "", "Notification controls"])
    assert np.array_equal(first, second)
    assert np.linalg.norm(first[0]) == pytest.approx(1.0)
    assert np.linalg.norm(first[1]) == 0.0
    assert cosine_similarity_matrix(first[:1], second[:1])[0, 0] == pytest.approx(1.0)


def test_infer_needs_is_stable_grounded_and_second_order():
    config = NeedConfig(
        cluster_similarity=0.05,
        min_cluster_size=2,
        include_singletons=False,
        llm_samples=2,
    )
    first = infer_needs(_signals(), config=config)
    second = infer_needs(list(reversed(_signals())), config=config)
    assert first == second
    assert len(first) >= 1
    export_need = next(
        need for need in first if {"S0001", "S0002"} <= set(need["supporting_signal_ids"])
    )
    assert export_need["id"].startswith("N")
    assert export_need["jtbd_statement"].startswith("When using the product")
    assert export_need["kano_class"] in {"basic", "performance", "excitement"}
    assert 0 <= export_need["opportunity_score"] <= 20
    assert export_need["metadata"]["inference"] == "offline"


def test_offline_site_frame_stays_within_dashboard_evidence() -> None:
    signals = [
        {
            "id": "S1001",
            "source": "review",
            "text": "The dashboard shows stats for the wrong site when I switch blogs.",
            "rating": 1,
        },
        {
            "id": "S1002",
            "source": "review",
            "text": "Multiple sites are confusing because each blog gets another site's stats.",
            "rating": 2,
        },
    ]
    needs = infer_needs(
        signals,
        config=NeedConfig(cluster_similarity=0.0, min_cluster_size=2, llm_samples=2),
    )
    assert len(needs) == 1
    job = needs[0]["jtbd_statement"].casefold()
    assert "correct site" in job
    assert "dashboard analytics" in job
    assert "content" not in job
    assert "edit" not in job


def test_hallucinated_extractor_ids_are_rejected_and_offline_fallback_is_used():
    class BadExtractor:
        def extract(self, **_kwargs):
            return {
                "latent_need": "Reliable export",
                "jtbd_statement": "When exporting, save reliably.",
                "kano_class": "basic",
                "root_cause_hypothesis": "Unknown",
                "symptom": "Export fails",
                "supporting_signal_ids": ["S9999"],
            }

    needs = infer_needs(
        _signals()[:2],
        extractor=BadExtractor(),
        config=NeedConfig(cluster_similarity=0.0, min_cluster_size=2, llm_samples=2),
    )
    assert len(needs) == 1
    assert needs[0]["metadata"]["inference"] == "offline"
    assert needs[0]["metadata"]["llm_failures"] == 2
    assert set(needs[0]["supporting_signal_ids"]) == {"S0001", "S0002"}


def test_duplicate_signal_ids_fail_before_analysis():
    duplicate = _signals()[:2]
    duplicate[1]["id"] = duplicate[0]["id"]
    with pytest.raises(ValueError, match="unique"):
        infer_needs(duplicate)


def test_odi_opportunity_formula():
    assert opportunity_score(8, 3) == 13
    assert opportunity_score(8, 9) == 8
    assert opportunity_score(99, -3) == 20


class _Response:
    def __init__(self, payload):
        import json

        self.text = json.dumps(payload)


class _RecordingModels:
    def __init__(self, payload):
        self.payload = payload
        self.config = None

    def generate_content(self, **kwargs):
        self.config = kwargs["config"]
        return _Response(self.payload)


class _RecordingClient:
    def __init__(self, payload):
        self.models = _RecordingModels(payload)


def test_extractor_requires_explicit_output_budget():
    with pytest.raises(TypeError, match="max_output_tokens"):
        GeminiNeedExtractor(object(), "model")  # type: ignore[call-arg]


def test_merger_requires_explicit_output_budget():
    with pytest.raises(TypeError, match="max_output_tokens"):
        GeminiNeedMerger(object(), "model")  # type: ignore[call-arg]


def test_extractor_forwards_explicit_output_budget():
    payload = {
        "latent_need": "Reliable export",
        "jtbd_statement": "Export without data loss.",
        "kano_class": "basic",
        "root_cause_hypothesis": "The export path may be fragile.",
        "symptom": "Export fails.",
        "supporting_signal_ids": ["S1"],
    }
    client = _RecordingClient(payload)
    extractor = GeminiNeedExtractor(client, "model", max_output_tokens=4096)
    extractor.extract(
        cluster_texts=["Export fails"],
        allowed_signal_ids=["S1"],
        aspects=["export"],
        temperature=0.0,
    )
    assert client.models.config["max_output_tokens"] == 4096


def test_merger_forwards_explicit_output_budget():
    client = _RecordingClient({"groups": []})
    merger = GeminiNeedMerger(client, "model", max_output_tokens=2048)
    merger.propose_groups(
        needs=[{"id": "N1", "latent_need": "Export", "jtbd_statement": "Export."}],
        allowed_need_ids=["N1"],
        temperature=0.0,
    )
    assert client.models.config["max_output_tokens"] == 2048
