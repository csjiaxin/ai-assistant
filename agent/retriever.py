"""
Retrieval + caching layer.

Wraps the vector store with:
  - Redis caching of query embeddings and results
  - Re-ranking by reaction score
  - Deduplication of near-identical retrieved items
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from agent.config import config
from agent.embedder import Embedder
from agent.models import QAPair, RetrievedContext
from agent.vector_store import VectorStoreBase

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStoreBase,
        redis_client=None,   # optional redis.Redis instance
    ):
        self._embedder = embedder
        self._vs = vector_store
        self._redis = redis_client

    # ── Public API ─────────────────────────────────────────────────────

    def retrieve(
        self,
        question: str,
        top_k: int = config.top_k,
        channel_filter: Optional[str] = None,
    ) -> list[RetrievedContext]:
        """
        Embed the question, search the vector store, re-rank, and return top results.
        Uses Redis cache if available.
        """
        question = question.strip()
        cache_key = self._cache_key(question, top_k, channel_filter)

        # 1. Try cache
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for query: %.60s", question)
            return cached

        # 2. Embed
        embedding = self._embedder.embed(question)

        # 3. Search
        results = self._vs.search(embedding, top_k=top_k * 2, channel_filter=channel_filter)

        # 4. Re-rank: blend similarity + normalised reaction score
        results = self._rerank(results, top_k)

        # 5. Cache result
        self._cache_set(cache_key, results)

        return results

    # ── Re-ranking ─────────────────────────────────────────────────────

    @staticmethod
    def _rerank(results: list[RetrievedContext], top_k: int) -> list[RetrievedContext]:
        """
        Blend cosine similarity (70%) with reaction score (30%) to boost
        answers that have been positively acknowledged by the team.
        """
        if not results:
            return results

        max_reactions = max((r.qa.reaction_score for r in results), default=1) or 1

        scored = []
        for r in results:
            reaction_norm = r.qa.reaction_score / max_reactions
            combined = 0.70 * r.similarity + 0.30 * reaction_norm
            scored.append((combined, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate — skip items whose question is very similar to an already-selected one
        seen_questions: list[str] = []
        deduped: list[RetrievedContext] = []
        for _, ctx in scored:
            if not _is_near_duplicate(ctx.qa.question, seen_questions):
                deduped.append(ctx)
                seen_questions.append(ctx.qa.question)
            if len(deduped) >= top_k:
                break

        return deduped

    # ── Cache helpers ──────────────────────────────────────────────────

    @staticmethod
    def _cache_key(question: str, top_k: int, channel_filter: Optional[str]) -> str:
        raw = f"{question}|{top_k}|{channel_filter or ''}"
        return "slack_agent:query:" + hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _cache_get(self, key: str) -> Optional[list[RetrievedContext]]:
        if self._redis is None:
            return None
        try:
            data = self._redis.get(key)
            if data:
                return self._deserialize_results(json.loads(data))
        except Exception as exc:
            logger.warning("Redis get failed: %s", exc)
        return None

    def _cache_set(self, key: str, results: list[RetrievedContext]) -> None:
        if self._redis is None:
            return
        try:
            self._redis.setex(key, config.cache_ttl_seconds, json.dumps(self._serialize_results(results)))
        except Exception as exc:
            logger.warning("Redis set failed: %s", exc)

    # ── Serialisation ──────────────────────────────────────────────────

    @staticmethod
    def _serialize_results(results: list[RetrievedContext]) -> list[dict]:
        return [
            {
                "similarity": r.similarity,
                "qa": {
                    "id": r.qa.id,
                    "channel_id": r.qa.channel_id,
                    "thread_ts": r.qa.thread_ts,
                    "question": r.qa.question,
                    "answer": r.qa.answer,
                    "questioner_id": r.qa.questioner_id,
                    "reaction_score": r.qa.reaction_score,
                    "slack_url": r.qa.slack_url,
                },
            }
            for r in results
        ]

    @staticmethod
    def _deserialize_results(data: list[dict]) -> list[RetrievedContext]:
        results = []
        for item in data:
            qa_data = item["qa"]
            qa = QAPair(
                id=qa_data["id"],
                channel_id=qa_data["channel_id"],
                thread_ts=qa_data["thread_ts"],
                question=qa_data["question"],
                answer=qa_data["answer"],
                questioner_id=qa_data["questioner_id"],
                reaction_score=qa_data["reaction_score"],
                slack_url=qa_data["slack_url"],
            )
            results.append(RetrievedContext(qa=qa, similarity=item["similarity"]))
        return results


# ── Helpers ────────────────────────────────────────────────────────────

def _is_near_duplicate(candidate: str, seen: list[str], threshold: float = 0.60) -> bool:
    """
    Simple token-overlap (Jaccard similarity) check for deduplication.
    Two questions are considered near-duplicates when their token Jaccard
    similarity exceeds `threshold` (default 0.70).
    """
    if not seen:
        return False
    cand_tokens = set(candidate.lower().split())
    for existing in seen:
        ex_tokens = set(existing.lower().split())
        if not cand_tokens or not ex_tokens:
            continue
        intersection = len(cand_tokens & ex_tokens)
        union = len(cand_tokens | ex_tokens)
        jaccard = intersection / union if union > 0 else 0.0
        if jaccard >= threshold:
            return True
    return False
