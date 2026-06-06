"""
CSV Ingestion — reads Slack export CSVs with the format:
    Time, User, Message

Thread structure is inferred from the ↳ prefix on reply rows.
Root messages (no ↳) start a new thread; ↳ rows are replies to the latest root.

This allows the agent to run 100% locally without Slack API access,
using only the exported CSV file.
"""
from __future__ import annotations

import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from agent.database import Database
from agent.embedder import Embedder
from agent.models import QAPair
from agent.vector_store import VectorStoreBase

logger = logging.getLogger(__name__)

# Positive reaction keywords in message text that boost score
POSITIVE_SIGNAL_WORDS = {"thanks", "thank", "solved", "fixed", "worked", "correct", "perfect", "great", "helpful"}
# Bots / system users to skip
BOT_USER_PATTERNS = re.compile(r"bot_message|jiraservicemanagement|Unknown User", re.IGNORECASE)
# Reply prefix
REPLY_PREFIX = "↳"


class CSVIngestionPipeline:
    """
    Ingests Slack history from a CSV export file.
    Groups messages into Q&A threads and embeds them into the vector store.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStoreBase,
        database: Database,
        channel_id: str = "csv_import",
    ):
        self._embedder = embedder
        self._vs = vector_store
        self._db = database
        self._channel_id = channel_id

    # ── Public API ─────────────────────────────────────────────────────

    def ingest_file(self, csv_path: str | Path, force_reindex: bool = False) -> dict:
        """
        Parse a CSV file and ingest all threads into the vector + message stores.
        Returns stats dict.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info("Ingesting CSV: %s", csv_path)
        threads = self._parse_csv(csv_path)
        logger.info("Parsed %d threads from CSV", len(threads))

        stats = {"threads_found": len(threads), "qa_pairs": 0, "embedded": 0, "skipped": 0}
        batch: list[tuple[QAPair, list[float]]] = []

        for thread in threads:
            qa = self._thread_to_qa(thread)
            if qa is None:
                continue

            stats["qa_pairs"] += 1
            self._db.upsert_qa_pair(qa)

            if not force_reindex and self._db.already_indexed(qa.id):
                stats["skipped"] += 1
                continue

            embedding = self._embedder.embed(qa.combined_text)
            batch.append((qa, embedding))

            if len(batch) >= 50:
                self._flush(batch, stats)
                batch = []

        if batch:
            self._flush(batch, stats)

        logger.info("CSV ingestion complete: %s", stats)
        return stats

    # ── CSV Parsing ────────────────────────────────────────────────────

    def _parse_csv(self, csv_path: Path) -> list[dict]:
        """
        Parse CSV into a list of thread dicts:
          {root_time, root_user, root_text, replies: [{time, user, text}]}
        """
        threads: list[dict] = []
        current_thread: Optional[dict] = None

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                time_str = row.get("Time", "").strip()
                user     = row.get("User", "").strip()
                message  = row.get("Message", "").strip()

                if not message:
                    continue

                # ── Key fix: the ↳ reply prefix is in the USER column ──
                is_reply = user.startswith(REPLY_PREFIX)
                clean_user = user.lstrip(REPLY_PREFIX).strip() if is_reply else user

                if is_reply:
                    # Skip bot/system replies by user name
                    if BOT_USER_PATTERNS.search(clean_user):
                        continue
                    # Skip bare @mention-only messages (jiraservicemanagement pings)
                    if re.fullmatch(r'\s*<@[A-Z0-9]+>\s*', message):
                        continue
                    # Skip karma bot messages
                    if "karma has increased" in message:
                        continue
                    # Skip all-quoted replies (just echoing previous message)
                    lines = [l.strip() for l in message.splitlines() if l.strip()]
                    if lines and all(l.startswith("&gt;") or l.startswith(">") for l in lines):
                        continue
                    # Skip very short replies after stripping Slack markup
                    plain = re.sub(r"<[^>]+>", "", message).strip()
                    if len(plain) < 8:
                        continue
                    if current_thread is not None:
                        current_thread["replies"].append({
                            "time": time_str,
                            "user": clean_user,
                            "text": self._clean_text(message),
                        })
                else:
                    # New root message → new thread
                    if current_thread and current_thread["replies"]:
                        threads.append(current_thread)

                    # Skip bot root messages (on-call alerts, system messages)
                    if BOT_USER_PATTERNS.search(clean_user):
                        current_thread = None
                        continue
                    # Skip very short root messages
                    if len(re.sub(r"<[^>]+>", "", message).strip()) < 15:
                        current_thread = None
                        continue

                    current_thread = {
                        "root_time": time_str,
                        "root_user": clean_user,
                        "root_text": self._clean_text(message),
                        "replies": [],
                    }

        # Don't forget the last thread
        if current_thread and current_thread["replies"]:
            threads.append(current_thread)

        return threads

    # ── Thread → QAPair ───────────────────────────────────────────────

    def _thread_to_qa(self, thread: dict) -> Optional[QAPair]:
        question = thread["root_text"].strip()
        if len(question) < 15:
            return None

        # Filter out pure bot/noise replies
        answer_parts = []
        respondents = []
        for reply in thread["replies"]:
            text = reply["text"].strip()
            if len(text) < 5:
                continue
            answer_parts.append(text)
            uid = reply["user"]
            if uid and uid not in respondents and uid != thread["root_user"]:
                respondents.append(uid)

        if not answer_parts:
            return None

        answer = "\n".join(answer_parts)

        # Heuristic reaction score: count positive-signal words in all replies
        combined = (question + " " + answer).lower()
        reaction_score = sum(1 for w in POSITIVE_SIGNAL_WORDS if w in combined)
        # Also boost threads with more respondents (more engagement)
        reaction_score += len(respondents)

        # Build a stable ID from time + user
        raw_id = f"{self._channel_id}_{thread['root_time']}_{thread['root_user']}"
        qa_id = re.sub(r"[^a-zA-Z0-9_]", "_", raw_id)[:80]

        # Parse timestamp
        created_at = None
        try:
            created_at = datetime.strptime(thread["root_time"], "%m/%d/%Y, %I:%M:%S %p")
        except ValueError:
            pass

        return QAPair(
            id=qa_id,
            channel_id=self._channel_id,
            thread_ts=qa_id,
            question=question[:2000],
            answer=answer[:4000],
            questioner_id=thread["root_user"],
            respondents=respondents,
            reaction_score=reaction_score,
            created_at=created_at,
            slack_url="",   # no Slack URL for CSV imports
        )

    def _flush(self, batch: list[tuple[QAPair, list[float]]], stats: dict) -> None:
        self._vs.upsert_batch(batch)
        for qa, _ in batch:
            self._db.mark_embedded(qa.id, qa.id)
        stats["embedded"] += len(batch)
        logger.info("Flushed batch of %d embeddings", len(batch))

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip Slack markup from CSV-exported messages."""
        # Remove HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        # Replace <@USER_ID> with @user
        text = re.sub(r"<@[A-Z0-9]+>", "@user", text)
        # Replace <#CHANNEL_ID|name> with #name
        text = re.sub(r"<#[A-Z0-9]+\|([^>]+)>", r"#\1", text)
        # Remove <!here>, <!channel>
        text = re.sub(r"<!([^>]+)>", r"@\1", text)
        # Unwrap URLs: <https://...|label> → label or URL
        text = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"\2", text)
        text = re.sub(r"<(https?://[^>]+)>", r"\1", text)
        # Remove quoted reply lines ("> ...") — these echo previous messages
        # Note: &gt; is already decoded to > above, so we check for >
        lines = [l for l in text.splitlines() if not l.strip().startswith(">")]
        return "\n".join(lines).strip()
