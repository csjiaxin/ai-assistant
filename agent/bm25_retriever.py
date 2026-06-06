"""
BM25 keyword retrieval baseline.

Uses rank-bm25 (Okapi BM25) over the same Q&A corpus as the dense retriever.
Serves as a baseline to compare against sentence-transformer dense retrieval.

BM25 excels at:
  - Exact-match queries (service names, error codes, config keys)
  - Rare/specific technical terms that embeddings may not capture well

BM25 struggles with:
  - Paraphrase / synonym queries ("503" vs "service unavailable")
  - Semantic queries where vocabulary diverges from indexed text
"""
from __future__ import annotations

import logging
import re
import string
from typing import Optional

from agent.config import config
from agent.database import Database
from agent.models import QAPair, RetrievedContext

logger = logging.getLogger(__name__)

# Simple English stopwords for tokenisation (no NLTK dependency)
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "are", "was",
    "be", "have", "has", "do", "does", "not", "we", "i", "my", "our",
    "you", "your", "he", "she", "they", "their", "its", "hi", "hey",
    "hello", "please", "thanks", "thank", "team", "anyone",
}


def _tokenise(text: str) -> list[str]:
    """
    Lowercase, strip punctuation, remove stopwords, split on whitespace.
    Keeps technical tokens like 'obsdeck-metrics', '503', 'atl-paas-icg-dependency-ic'.
    """
    text = text.lower()
    # Remove Slack markup remnants
    text = re.sub(r"<[^>]+>", " ", text)
    # Replace punctuation except hyphens (important in service names) and dots
    text = re.sub(r"[^\w\s\-\.]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25Retriever:
    """
    BM25 retriever built over the same Q&A pairs stored in the database.
    Index is built in-memory at construction time (~instant for 48 docs).
    """

    def __init__(self, database: Optional[Database] = None):
        from rank_bm25 import BM25Okapi  # type: ignore
        self._db = database or Database()
        self._qa_pairs: list[QAPair] = []
        self._index = None
        self._BM25Okapi = BM25Okapi
        self._build_index()

    def _build_index(self) -> None:
        """Load all Q&A pairs from the database and build the BM25 index."""
        self._qa_pairs = self._db.get_all_qa_pairs()
        if not self._qa_pairs:
            logger.warning("BM25: no Q&A pairs in database — index is empty")
            return
        # Tokenise combined text (same field used for dense embedding)
        corpus = [_tokenise(qa.combined_text) for qa in self._qa_pairs]
        self._index = self._BM25Okapi(corpus)
        logger.info("BM25 index built over %d documents", len(self._qa_pairs))

    def retrieve(
        self,
        question: str,
        top_k: int = config.top_k,
        min_score: float = 0.0,
    ) -> list[RetrievedContext]:
        """
        Retrieve top-K Q&A pairs by BM25 score.
        Returns RetrievedContext with similarity = normalised BM25 score [0, 1].
        """
        if self._index is None or not self._qa_pairs:
            return []

        query_tokens = _tokenise(question)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)
        max_score = max(scores) if max(scores) > 0 else 1.0

        # Pair scores with QA pairs and sort descending
        ranked = sorted(
            zip(scores, self._qa_pairs),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for raw_score, qa in ranked[:top_k]:
            if raw_score <= min_score:
                continue
            # Normalise to [0, 1] for fair comparison with cosine similarity
            norm_score = float(raw_score) / max_score
            results.append(RetrievedContext(qa=qa, similarity=norm_score))

        return results

    def rebuild(self) -> None:
        """Rebuild the index (call after new documents are ingested)."""
        self._build_index()

    @property
    def doc_count(self) -> int:
        return len(self._qa_pairs)
