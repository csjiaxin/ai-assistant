"""
Embedding service — uses local sentence-transformers by default (free, no API key).
Optional OpenAI backend available if OPENAI_API_KEY is set and EMBEDDING_BACKEND=openai.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Union

import numpy as np

from agent.config import config

logger = logging.getLogger(__name__)

# Free local models ranked by quality vs speed
LOCAL_MODELS = {
    "minilm":   "sentence-transformers/all-MiniLM-L6-v2",       # 384-dim, very fast, ~80MB
    "mpnet":    "sentence-transformers/all-mpnet-base-v2",       # 768-dim, best quality, ~420MB
    "bge-small":"BAAI/bge-small-en-v1.5",                       # 384-dim, strong retrieval, ~130MB
    "bge-base": "BAAI/bge-base-en-v1.5",                        # 768-dim, excellent, ~440MB
    "e5-small": "intfloat/e5-small-v2",                         # 384-dim, multilingual-ready
}


class Embedder:
    """
    Produces dense vector embeddings for text.

    Default:  sentence-transformers (local, free, no API key needed)
              Model controlled by EMBEDDING_MODEL env var (default: all-MiniLM-L6-v2)
    Optional: OpenAI text-embedding-3-small (set EMBEDDING_BACKEND=openai)
    """

    def __init__(self):
        self._openai_client = None
        self._st_model = None
        self._backend: str = "sentence_transformers"
        self._init_backend()

    # ── Init ───────────────────────────────────────────────────────────

    def _init_backend(self) -> None:
        backend_pref = config.embedding_backend.lower()
        if backend_pref == "openai" and config.openai_api_key:
            try:
                from openai import OpenAI  # type: ignore
                self._openai_client = OpenAI(api_key=config.openai_api_key)
                self._openai_client.models.list()
                self._backend = "openai"
                logger.info("Embedder: using OpenAI backend (%s)", config.embedding_model)
                return
            except Exception as exc:
                logger.warning("OpenAI not available (%s), falling back to sentence-transformers", exc)
        # Default: local sentence-transformers
        self._init_sentence_transformers()

    def _init_sentence_transformers(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            # Allow shorthand names like "minilm", "bge-small", or full HF model IDs
            model_name = LOCAL_MODELS.get(config.embedding_model, config.embedding_model)
            logger.info("Embedder: loading sentence-transformers model '%s' (this may download ~80-440MB once)…", model_name)
            self._st_model = SentenceTransformer(model_name)
            self._backend = "sentence_transformers"
            logger.info("Embedder: sentence-transformers ready (%s, dim=%d)", model_name, self._st_model.get_sentence_embedding_dimension())
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            )

    # ── Public API ─────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Embed a single text string → list of floats."""
        text = text.strip()
        if not text:
            raise ValueError("Cannot embed empty text")
        if self._backend == "openai":
            return self._embed_openai(text)
        return self._embed_st(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one API call (more efficient)."""
        texts = [t.strip() for t in texts if t.strip()]
        if not texts:
            return []
        if self._backend == "openai":
            return self._embed_openai_batch(texts)
        return self._embed_st_batch(texts)

    @staticmethod
    def text_hash(text: str) -> str:
        """Stable hash of text — used as cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        if self._backend == "sentence_transformers":
            return self._st_model.get_sentence_embedding_dimension()
        # OpenAI text-embedding-3-small → 1536
        return 1536

    # ── OpenAI backend ─────────────────────────────────────────────────

    def _embed_openai(self, text: str) -> list[float]:
        response = self._openai_client.embeddings.create(
            model=config.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _embed_openai_batch(self, texts: list[str]) -> list[list[float]]:
        # OpenAI allows up to 2048 inputs per request; chunk if needed
        all_embeddings: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._openai_client.embeddings.create(
                model=config.embedding_model,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    # ── Sentence-Transformers backend ──────────────────────────────────

    def _embed_st(self, text: str) -> list[float]:
        vec: np.ndarray = self._st_model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def _embed_st_batch(self, texts: list[str]) -> list[list[float]]:
        vecs: np.ndarray = self._st_model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vecs.tolist()

    @property
    def backend(self) -> str:
        return self._backend
