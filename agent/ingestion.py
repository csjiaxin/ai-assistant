"""
Ingestion pipeline — fetches Slack thread history, extracts Q&A pairs,
embeds them, and upserts into the vector + message stores.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from agent.config import config
from agent.database import Database
from agent.embedder import Embedder
from agent.models import QAPair, SlackMessage
from agent.vector_store import VectorStoreBase

logger = logging.getLogger(__name__)

# Reaction emojis considered "positive" for scoring
POSITIVE_REACTIONS = {"thumbsup", "+1", "white_check_mark", "heavy_check_mark", "100", "fire"}
# Bot user IDs to exclude from Q detection (populated at runtime)
_BOT_USER_IDS: set[str] = set()


class IngestionPipeline:
    """
    Orchestrates the full ingestion flow:
      Slack API → parse threads → Q&A pairs → embed → upsert
    """

    def __init__(
        self,
        slack_client,        # slack_sdk WebClient
        embedder: Embedder,
        vector_store: VectorStoreBase,
        database: Database,
    ):
        self._slack = slack_client
        self._embedder = embedder
        self._vs = vector_store
        self._db = database

    # ── Public entry point ─────────────────────────────────────────────

    def run(
        self,
        channel_ids: Optional[list[str]] = None,
        lookback_days: int = config.ingestion_lookback_days,
        force_reindex: bool = False,
    ) -> dict:
        """
        Full ingestion run.
        Returns stats dict: {channels, threads_found, qa_pairs, embedded, skipped}
        """
        channels = channel_ids or config.monitored_channels
        channels = [c.strip() for c in channels if c.strip()]

        stats = {"channels": len(channels), "threads_found": 0, "qa_pairs": 0, "embedded": 0, "skipped": 0}
        oldest_ts = str((datetime.utcnow() - timedelta(days=lookback_days)).timestamp())

        for channel_id in channels:
            logger.info("Ingesting channel %s ...", channel_id)
            try:
                ch_stats = self._ingest_channel(channel_id, oldest_ts, force_reindex)
                for k in ("threads_found", "qa_pairs", "embedded", "skipped"):
                    stats[k] += ch_stats[k]
            except Exception:
                logger.exception("Error ingesting channel %s", channel_id)

        logger.info("Ingestion complete: %s", stats)
        return stats

    # ── Channel ingestion ──────────────────────────────────────────────

    def _ingest_channel(self, channel_id: str, oldest_ts: str, force_reindex: bool) -> dict:
        stats = {"threads_found": 0, "qa_pairs": 0, "embedded": 0, "skipped": 0}
        qa_batch: list[tuple[QAPair, list[float]]] = []

        for root_msg in self._iter_channel_messages(channel_id, oldest_ts):
            # Only process messages that have thread replies
            reply_count = root_msg.get("reply_count", 0)
            if reply_count < config.min_thread_replies:
                continue

            stats["threads_found"] += 1
            thread_ts = root_msg["ts"]

            qa = self._build_qa_pair(channel_id, thread_ts, root_msg)
            if qa is None:
                continue

            stats["qa_pairs"] += 1
            self._db.upsert_qa_pair(qa)

            # Skip re-embedding if already indexed and not forced
            if not force_reindex and self._db.already_indexed(qa.id):
                stats["skipped"] += 1
                continue

            embedding = self._embedder.embed(qa.combined_text)
            qa_batch.append((qa, embedding))

            # Flush batch every 50
            if len(qa_batch) >= 50:
                self._flush_batch(qa_batch, stats)
                qa_batch = []

        # Flush remainder
        if qa_batch:
            self._flush_batch(qa_batch, stats)

        return stats

    def _flush_batch(self, batch: list[tuple[QAPair, list[float]]], stats: dict) -> None:
        self._vs.upsert_batch(batch)
        for qa, _ in batch:
            self._db.mark_embedded(qa.id, qa.id)
        stats["embedded"] += len(batch)

    # ── Thread fetching ────────────────────────────────────────────────

    def _iter_channel_messages(self, channel_id: str, oldest_ts: str):
        """Paginate through conversations.history, yielding root messages."""
        cursor = None
        while True:
            kwargs = dict(channel=channel_id, oldest=oldest_ts, limit=200)
            if cursor:
                kwargs["cursor"] = cursor
            try:
                resp = self._slack.conversations_history(**kwargs)
            except Exception:
                logger.exception("Failed to fetch history for %s", channel_id)
                break

            for msg in resp.get("messages", []):
                # Skip bot messages as question roots
                if msg.get("bot_id") or msg.get("subtype") in ("bot_message", "channel_join", "channel_leave"):
                    continue
                yield msg

            meta = resp.get("response_metadata", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break
            time.sleep(0.5)   # Slack rate limit: ~50 req/min

    def _fetch_thread_replies(self, channel_id: str, thread_ts: str) -> list[dict]:
        """Fetch all replies for a thread, excluding the root message."""
        replies = []
        cursor = None
        while True:
            kwargs = dict(channel=channel_id, ts=thread_ts, limit=200)
            if cursor:
                kwargs["cursor"] = cursor
            try:
                resp = self._slack.conversations_replies(**kwargs)
            except Exception:
                logger.exception("Failed to fetch thread %s", thread_ts)
                break

            msgs = resp.get("messages", [])
            # First message is root — skip it on subsequent pages
            start = 1 if not cursor else 0
            for msg in msgs[start:]:
                if not msg.get("bot_id"):
                    replies.append(msg)

            meta = resp.get("response_metadata", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break
            time.sleep(0.3)
        return replies

    # ── Q&A extraction ─────────────────────────────────────────────────

    def _build_qa_pair(self, channel_id: str, thread_ts: str, root_msg: dict) -> Optional[QAPair]:
        question_text = self._clean_text(root_msg.get("text", ""))
        if len(question_text.strip()) < 10:
            return None

        replies = self._fetch_thread_replies(channel_id, thread_ts)
        if not replies:
            return None

        # Build answer from all replies
        answer_parts = []
        respondents = []
        reaction_score = 0
        for reply in replies:
            text = self._clean_text(reply.get("text", ""))
            if text:
                answer_parts.append(text)
            uid = reply.get("user", "")
            if uid and uid not in respondents:
                respondents.append(uid)
            # Count positive reactions on each reply
            for reaction in reply.get("reactions", []):
                if reaction["name"] in POSITIVE_REACTIONS:
                    reaction_score += reaction["count"]

        # Also count reactions on root message
        for reaction in root_msg.get("reactions", []):
            if reaction["name"] in POSITIVE_REACTIONS:
                reaction_score += reaction["count"]

        answer_text = "\n".join(answer_parts)
        if not answer_text.strip():
            return None

        qa_id = f"{channel_id}_{thread_ts}"
        ts_clean = thread_ts.replace(".", "")
        slack_url = f"https://slack.com/archives/{channel_id}/p{ts_clean}"

        created_at = None
        try:
            created_at = datetime.fromtimestamp(float(thread_ts))
        except (ValueError, OverflowError):
            pass

        return QAPair(
            id=qa_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            question=question_text[:2000],
            answer=answer_text[:4000],
            questioner_id=root_msg.get("user", "unknown"),
            respondents=respondents,
            reaction_score=reaction_score,
            created_at=created_at,
            slack_url=slack_url,
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip Slack markup, user mentions, channel refs."""
        import re
        # Replace <@USER_ID> with @user
        text = re.sub(r"<@[A-Z0-9]+>", "@user", text)
        # Replace <#CHANNEL_ID|name> with #name
        text = re.sub(r"<#[A-Z0-9]+\|([^>]+)>", r"#\1", text)
        # Remove other angle-bracket tokens like <!here>, <!channel>
        text = re.sub(r"<!([^>]+)>", r"@\1", text)
        # Remove bare URLs inside angle brackets
        text = re.sub(r"<(https?://[^|>]+)(?:\|[^>]*)?>", r"\1", text)
        return text.strip()
