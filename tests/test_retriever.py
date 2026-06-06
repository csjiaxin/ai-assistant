"""
Tests for the retriever — re-ranking, deduplication, and caching logic.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agent.models import QAPair, RetrievedContext
from agent.retriever import Retriever, _is_near_duplicate


def make_qa(id: str, question: str, reaction_score: int = 0) -> QAPair:
    return QAPair(
        id=id,
        channel_id="C001",
        thread_ts=id,
        question=question,
        answer=f"Answer to: {question}",
        questioner_id="U001",
        reaction_score=reaction_score,
        slack_url=f"https://slack.com/archives/C001/p{id}",
    )


def make_context(id: str, question: str, similarity: float, reaction_score: int = 0) -> RetrievedContext:
    return RetrievedContext(qa=make_qa(id, question, reaction_score), similarity=similarity)


# ── _is_near_duplicate ─────────────────────────────────────────────────

class TestNearDuplicate:
    def test_identical_is_duplicate(self):
        seen = ["How do I restart the Kafka consumer?"]
        assert _is_near_duplicate("How do I restart the Kafka consumer?", seen)

    def test_completely_different_is_not_duplicate(self):
        seen = ["How do I configure Redis?"]
        assert not _is_near_duplicate("What is the deployment process for staging?", seen)

    def test_empty_seen_is_never_duplicate(self):
        assert not _is_near_duplicate("any question here", [])

    def test_high_overlap_is_duplicate(self):
        # Jaccard ~0.615 — above the 0.60 threshold → duplicate
        seen = ["How do I configure the retry policy for Kafka consumers in production?"]
        candidate = "How do I configure retry policy for Kafka consumers?"
        assert _is_near_duplicate(candidate, seen)

    def test_low_overlap_is_not_duplicate(self):
        seen = ["How do I configure Redis timeout settings?"]
        candidate = "What is the correct Vault secret rotation process?"
        assert not _is_near_duplicate(candidate, seen)


# ── Retriever re-ranking ───────────────────────────────────────────────

class TestRetrieverRerank:
    def test_higher_reaction_score_boosts_ranking(self):
        results = [
            make_context("1", "How to configure Kafka retry?", similarity=0.80, reaction_score=0),
            make_context("2", "Configure Kafka retry policy settings", similarity=0.75, reaction_score=20),
        ]
        reranked = Retriever._rerank(results, top_k=2)
        # Item 2 has more reactions so should come first despite lower similarity
        assert reranked[0].qa.id == "2"

    def test_empty_results_returns_empty(self):
        assert Retriever._rerank([], top_k=5) == []

    def test_deduplicates_near_identical_questions(self):
        results = [
            make_context("1", "How do I restart the Kafka consumer service?", similarity=0.90),
            make_context("2", "How do I restart the Kafka consumer process?", similarity=0.88),
            make_context("3", "What is the Redis eviction policy setting?", similarity=0.75),
        ]
        reranked = Retriever._rerank(results, top_k=3)
        # Items 1 and 2 are near-duplicates — only one should survive
        ids = [r.qa.id for r in reranked]
        assert "1" in ids
        assert "2" not in ids   # deduplicated
        assert "3" in ids

    def test_top_k_is_respected(self):
        results = [
            make_context(str(i), f"Unique question number {i} about different topics", similarity=0.9 - i * 0.05)
            for i in range(10)
        ]
        reranked = Retriever._rerank(results, top_k=3)
        assert len(reranked) <= 3

    def test_single_result_returned_as_is(self):
        results = [make_context("1", "How to deploy to production?", similarity=0.85, reaction_score=5)]
        reranked = Retriever._rerank(results, top_k=5)
        assert len(reranked) == 1
        assert reranked[0].qa.id == "1"


# ── Retriever integration (mocked) ────────────────────────────────────

class TestRetriever:
    def make_retriever(self, vs_results=None, redis=None):
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 1536
        vector_store = MagicMock()
        vector_store.search.return_value = vs_results or []
        return Retriever(embedder=embedder, vector_store=vector_store, redis_client=redis)

    def test_retrieve_returns_results(self):
        vs_results = [
            make_context("1", "How do I configure Kafka retry?", similarity=0.85, reaction_score=3),
            make_context("2", "What is the Redis eviction policy?", similarity=0.72, reaction_score=1),
        ]
        retriever = self.make_retriever(vs_results=vs_results)
        results = retriever.retrieve("How to set Kafka retry policy?", top_k=5)
        assert len(results) >= 1

    def test_retrieve_with_no_results(self):
        retriever = self.make_retriever(vs_results=[])
        results = retriever.retrieve("Some obscure question no one has asked before", top_k=5)
        assert results == []

    def test_cache_hit_skips_embedding(self):
        redis_mock = MagicMock()
        import json
        cached_data = [
            {
                "similarity": 0.9,
                "qa": {
                    "id": "C001_ts1", "channel_id": "C001", "thread_ts": "ts1",
                    "question": "cached question", "answer": "cached answer",
                    "questioner_id": "U001", "reaction_score": 5,
                    "slack_url": "https://slack.com/archives/C001/pts1",
                }
            }
        ]
        redis_mock.get.return_value = json.dumps(cached_data)
        retriever = self.make_retriever(redis=redis_mock)
        results = retriever.retrieve("any question", top_k=5)
        # Embedder should NOT be called on cache hit
        retriever._embedder.embed.assert_not_called()
        assert len(results) == 1
        assert results[0].qa.question == "cached question"

    def test_cache_miss_calls_embedder(self):
        redis_mock = MagicMock()
        redis_mock.get.return_value = None   # cache miss
        retriever = self.make_retriever(redis=redis_mock)
        retriever.retrieve("How do I restart Kafka?", top_k=5)
        retriever._embedder.embed.assert_called_once()

    def test_serialise_deserialise_roundtrip(self):
        results = [
            make_context("1", "How to configure retry?", similarity=0.88, reaction_score=7),
            make_context("2", "What is the eviction policy?", similarity=0.65, reaction_score=2),
        ]
        serialised = Retriever._serialize_results(results)
        restored = Retriever._deserialize_results(serialised)
        assert len(restored) == 2
        assert restored[0].qa.id == "1"
        assert restored[0].similarity == pytest.approx(0.88)
        assert restored[1].qa.reaction_score == 2
