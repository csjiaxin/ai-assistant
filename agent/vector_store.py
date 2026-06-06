"""
Vector store abstraction — supports ChromaDB (default), Pinecone, and Qdrant.
All backends expose the same interface: upsert() and search().
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from agent.config import config
from agent.models import QAPair, RetrievedContext

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Abstract base
# ══════════════════════════════════════════════════════════════════════

class VectorStoreBase(ABC):
    @abstractmethod
    def upsert(self, qa: QAPair, embedding: list[float]) -> None: ...

    @abstractmethod
    def upsert_batch(self, pairs: list[tuple[QAPair, list[float]]]) -> None: ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int, channel_filter: Optional[str] = None) -> list[RetrievedContext]: ...

    @abstractmethod
    def delete(self, qa_id: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...


# ══════════════════════════════════════════════════════════════════════
# ChromaDB backend (default — zero infra, persistent)
# ══════════════════════════════════════════════════════════════════════

class ChromaVectorStore(VectorStoreBase):
    def __init__(self):
        import chromadb  # type: ignore
        from chromadb.config import Settings  # type: ignore

        self._client = chromadb.PersistentClient(
            path=config.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=config.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready (%d docs)", config.chroma_collection, self._col.count())

    def upsert(self, qa: QAPair, embedding: list[float]) -> None:
        self._col.upsert(
            ids=[qa.id],
            embeddings=[embedding],
            documents=[qa.combined_text],
            metadatas=[qa.to_metadata()],
        )

    def upsert_batch(self, pairs: list[tuple[QAPair, list[float]]]) -> None:
        if not pairs:
            return
        ids, embeddings, documents, metadatas = [], [], [], []
        for qa, emb in pairs:
            ids.append(qa.id)
            embeddings.append(emb)
            documents.append(qa.combined_text)
            metadatas.append(qa.to_metadata())
        self._col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        logger.info("Upserted %d documents into ChromaDB", len(pairs))

    def search(self, query_embedding: list[float], top_k: int, channel_filter: Optional[str] = None) -> list[RetrievedContext]:
        where = {"channel_id": channel_filter} if channel_filter else None
        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._col.count() or 1),
            where=where,
            include=["metadatas", "distances", "documents"],
        )

        retrieved: list[RetrievedContext] = []
        for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
            # Chroma cosine distance → similarity: similarity = 1 - distance
            similarity = max(0.0, 1.0 - distance)
            if similarity < config.min_similarity:
                continue
            qa = self._meta_to_qa(meta)
            retrieved.append(RetrievedContext(qa=qa, similarity=similarity))
        return retrieved

    def delete(self, qa_id: str) -> None:
        self._col.delete(ids=[qa_id])

    def count(self) -> int:
        return self._col.count()

    @staticmethod
    def _meta_to_qa(meta: dict) -> QAPair:
        from datetime import datetime
        created_at = None
        if meta.get("created_at"):
            try:
                created_at = datetime.fromisoformat(meta["created_at"])
            except ValueError:
                pass
        return QAPair(
            id=meta["id"],
            channel_id=meta.get("channel_id", ""),
            thread_ts=meta.get("thread_ts", ""),
            question=meta.get("question", ""),
            answer=meta.get("answer", ""),
            questioner_id=meta.get("questioner_id", ""),
            reaction_score=int(meta.get("reaction_score", 0)),
            slack_url=meta.get("slack_url", ""),
            created_at=created_at,
        )


# ══════════════════════════════════════════════════════════════════════
# Pinecone backend (cloud-scale)
# ══════════════════════════════════════════════════════════════════════

class PineconeVectorStore(VectorStoreBase):
    def __init__(self):
        from pinecone import Pinecone  # type: ignore
        pc = Pinecone(api_key=config.pinecone_api_key)
        self._index = pc.Index(config.pinecone_index)
        logger.info("Pinecone index '%s' connected", config.pinecone_index)

    def upsert(self, qa: QAPair, embedding: list[float]) -> None:
        self._index.upsert(vectors=[(qa.id, embedding, qa.to_metadata())])

    def upsert_batch(self, pairs: list[tuple[QAPair, list[float]]]) -> None:
        vectors = [(qa.id, emb, qa.to_metadata()) for qa, emb in pairs]
        # Pinecone recommends batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self._index.upsert(vectors=vectors[i : i + batch_size])
        logger.info("Upserted %d vectors to Pinecone", len(pairs))

    def search(self, query_embedding: list[float], top_k: int, channel_filter: Optional[str] = None) -> list[RetrievedContext]:
        filter_expr = {"channel_id": {"$eq": channel_filter}} if channel_filter else None
        result = self._index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True, filter=filter_expr
        )
        retrieved = []
        for match in result["matches"]:
            if match["score"] < config.min_similarity:
                continue
            qa = ChromaVectorStore._meta_to_qa(match["metadata"])
            retrieved.append(RetrievedContext(qa=qa, similarity=match["score"]))
        return retrieved

    def delete(self, qa_id: str) -> None:
        self._index.delete(ids=[qa_id])

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        return stats.get("total_vector_count", 0)


# ══════════════════════════════════════════════════════════════════════
# Qdrant backend
# ══════════════════════════════════════════════════════════════════════

class QdrantVectorStore(VectorStoreBase):
    def __init__(self, embedding_dim: int = 384):
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.models import Distance, VectorParams  # type: ignore
        self._client = QdrantClient(url=config.qdrant_url)
        self._col = config.qdrant_collection
        self._dim = embedding_dim
        # Create collection if it doesn't exist
        existing = [c.name for c in self._client.get_collections().collections]
        if self._col not in existing:
            self._client.create_collection(
                collection_name=self._col,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )
        logger.info("Qdrant collection '%s' ready", self._col)

    def upsert(self, qa: QAPair, embedding: list[float]) -> None:
        from qdrant_client.models import PointStruct  # type: ignore
        self._client.upsert(
            collection_name=self._col,
            points=[PointStruct(id=abs(hash(qa.id)) % (10**9), vector=embedding, payload=qa.to_metadata())],
        )

    def upsert_batch(self, pairs: list[tuple[QAPair, list[float]]]) -> None:
        from qdrant_client.models import PointStruct  # type: ignore
        points = [
            PointStruct(id=abs(hash(qa.id)) % (10**9), vector=emb, payload=qa.to_metadata())
            for qa, emb in pairs
        ]
        self._client.upsert(collection_name=self._col, points=points)

    def search(self, query_embedding: list[float], top_k: int, channel_filter: Optional[str] = None) -> list[RetrievedContext]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
        query_filter = None
        if channel_filter:
            query_filter = Filter(must=[FieldCondition(key="channel_id", match=MatchValue(value=channel_filter))])
        results = self._client.search(
            collection_name=self._col, query_vector=query_embedding, limit=top_k, query_filter=query_filter
        )
        retrieved = []
        for hit in results:
            if hit.score < config.min_similarity:
                continue
            qa = ChromaVectorStore._meta_to_qa(hit.payload)
            retrieved.append(RetrievedContext(qa=qa, similarity=hit.score))
        return retrieved

    def delete(self, qa_id: str) -> None:
        from qdrant_client.models import PointIdsList  # type: ignore
        self._client.delete(collection_name=self._col, points_selector=PointIdsList(points=[abs(hash(qa_id)) % (10**9)]))

    def count(self) -> int:
        info = self._client.get_collection(self._col)
        return info.points_count or 0


# ══════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════

def create_vector_store() -> VectorStoreBase:
    backend = config.vector_store_type.lower()
    if backend == "chroma":
        return ChromaVectorStore()
    elif backend == "pinecone":
        return PineconeVectorStore()
    elif backend == "qdrant":
        return QdrantVectorStore()
    else:
        raise ValueError(f"Unknown vector store type: {backend!r}. Choose: chroma | pinecone | qdrant")
