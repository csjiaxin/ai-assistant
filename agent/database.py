"""
SQLite / PostgreSQL persistence layer for raw Slack thread data.
Uses SQLAlchemy Core (no ORM) for simplicity.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, DateTime, Integer, MetaData, String, Table, Text,
    create_engine, select, insert, update, func,
)
from sqlalchemy.engine import Engine

from agent.config import config
from agent.models import QAPair

logger = logging.getLogger(__name__)

metadata = MetaData()

qa_pairs_table = Table(
    "qa_pairs",
    metadata,
    Column("id", String, primary_key=True),
    Column("channel_id", String, nullable=False, index=True),
    Column("thread_ts", String, nullable=False),
    Column("question", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("questioner_id", String),
    Column("respondents_json", Text, default="[]"),
    Column("reaction_score", Integer, default=0),
    Column("slack_url", String),
    Column("created_at", DateTime),
    Column("indexed_at", DateTime),
    Column("embedding_id", String),   # ID in vector store
)


class Database:
    def __init__(self, database_url: str = config.database_url):
        self._engine: Engine = create_engine(database_url, echo=False)
        metadata.create_all(self._engine)
        logger.info("Database initialised at %s", database_url)

    # ── Write ──────────────────────────────────────────────────────────

    def upsert_qa_pair(self, qa: QAPair) -> None:
        """Insert or update a Q&A pair record."""
        with self._engine.begin() as conn:
            existing = conn.execute(
                select(qa_pairs_table.c.id).where(qa_pairs_table.c.id == qa.id)
            ).fetchone()

            row = {
                "id": qa.id,
                "channel_id": qa.channel_id,
                "thread_ts": qa.thread_ts,
                "question": qa.question,
                "answer": qa.answer,
                "questioner_id": qa.questioner_id,
                "respondents_json": json.dumps(qa.respondents),
                "reaction_score": qa.reaction_score,
                "slack_url": qa.slack_url,
                "created_at": qa.created_at,
                "indexed_at": datetime.utcnow(),
            }

            if existing:
                conn.execute(
                    update(qa_pairs_table)
                    .where(qa_pairs_table.c.id == qa.id)
                    .values(**{k: v for k, v in row.items() if k != "id"})
                )
            else:
                conn.execute(insert(qa_pairs_table).values(**row))

    def mark_embedded(self, qa_id: str, embedding_id: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                update(qa_pairs_table)
                .where(qa_pairs_table.c.id == qa_id)
                .values(embedding_id=embedding_id)
            )

    # ── Read ───────────────────────────────────────────────────────────

    def get_qa_by_id(self, qa_id: str) -> Optional[QAPair]:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(qa_pairs_table).where(qa_pairs_table.c.id == qa_id)
            ).fetchone()
        return self._row_to_qa(row) if row else None

    def get_all_qa_pairs(self, channel_id: Optional[str] = None) -> list[QAPair]:
        with self._engine.connect() as conn:
            stmt = select(qa_pairs_table)
            if channel_id:
                stmt = stmt.where(qa_pairs_table.c.channel_id == channel_id)
            rows = conn.execute(stmt).fetchall()
        return [self._row_to_qa(r) for r in rows]

    def count(self) -> int:
        with self._engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(qa_pairs_table)).scalar() or 0

    def already_indexed(self, qa_id: str) -> bool:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(qa_pairs_table.c.embedding_id).where(qa_pairs_table.c.id == qa_id)
            ).fetchone()
        return bool(row and row[0])

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_qa(row) -> QAPair:
        return QAPair(
            id=row.id,
            channel_id=row.channel_id,
            thread_ts=row.thread_ts,
            question=row.question,
            answer=row.answer,
            questioner_id=row.questioner_id,
            respondents=json.loads(row.respondents_json or "[]"),
            reaction_score=row.reaction_score or 0,
            slack_url=row.slack_url or "",
            created_at=row.created_at,
        )
