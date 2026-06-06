"""
Slack Bolt application — event handlers for the AI support agent.

Handles:
  - app_mention: @bot-mention triggers answer generation
  - message (in monitored channels): question detection
  - reaction_added: feedback collection (👍 / 👎)
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from slack_bolt import App  # type: ignore
from slack_bolt.adapter.socket_mode import SocketModeHandler  # type: ignore

from agent.config import config
from agent.database import Database
from agent.embedder import Embedder
from agent.ingestion import IngestionPipeline
from agent.llm import LLMBase, create_llm
from agent.models import AgentResponse
from agent.retriever import Retriever
from agent.vector_store import VectorStoreBase, create_vector_store

logger = logging.getLogger(__name__)

# ── Bot message prefix to avoid self-triggering loops ─────────────────
BOT_ANSWER_PREFIX = "🤖 *"

# ── Minimum question length to process ────────────────────────────────
MIN_QUESTION_LEN = 15

# ── Channels that require @mention (vs. answering all messages) ────────
MENTION_ONLY_CHANNELS: set[str] = set()  # populated from config


class SupportAgent:
    """
    Central agent class — wires together all components.
    """

    def __init__(self):
        self._embedder = Embedder()
        self._vector_store: VectorStoreBase = create_vector_store()
        self._db = Database()
        self._retriever: Optional[Retriever] = None
        self._llm: Optional[LLMBase] = None
        self._redis = None
        self._app: Optional[App] = None

    def init(self) -> "SupportAgent":
        """Lazy-init heavy components."""
        # Redis (optional)
        try:
            import redis  # type: ignore
            self._redis = redis.from_url(config.redis_url, decode_responses=True, socket_timeout=2)
            self._redis.ping()
            logger.info("Redis connected at %s", config.redis_url)
        except Exception as exc:
            logger.warning("Redis not available (%s) — caching disabled", exc)
            self._redis = None

        self._retriever = Retriever(self._embedder, self._vector_store, self._redis)
        self._llm = create_llm()

        # Build Slack app
        self._app = App(
            token=config.slack_bot_token,
            signing_secret=config.slack_signing_secret,
        )
        self._register_handlers()
        logger.info("SupportAgent initialised. Vector store has %d documents.", self._vector_store.count())
        return self

    # ── Handler registration ───────────────────────────────────────────

    def _register_handlers(self) -> None:
        app = self._app

        @app.event("app_mention")
        def handle_mention(event, say, client):
            """Respond when the bot is @mentioned anywhere."""
            self._handle_question(event, say, client, triggered_by_mention=True)

        @app.event("message")
        def handle_message(event, say, client):
            """
            Respond to messages in monitored channels.
            Skips: bot messages, thread replies (to avoid noise),
                   short messages, and channels not configured.
            """
            # Ignore bot messages (including our own)
            if event.get("bot_id") or event.get("subtype"):
                return
            # Ignore thread replies (only process root messages)
            if event.get("thread_ts") and event["thread_ts"] != event["ts"]:
                return
            channel = event.get("channel", "")
            if channel not in (config.monitored_channels or []):
                return
            text = event.get("text", "").strip()
            if len(text) < MIN_QUESTION_LEN:
                return
            # In mention-only mode, skip non-mentions
            if channel in MENTION_ONLY_CHANNELS:
                return
            self._handle_question(event, say, client, triggered_by_mention=False)

        @app.event("reaction_added")
        def handle_reaction(event, client):
            """Collect 👍/👎 feedback on bot answers."""
            self._handle_reaction(event, client)

        @app.error
        def global_error_handler(error, body, logger):
            logger.exception("Unhandled Slack app error: %s", error)

    # ── Core question handler ──────────────────────────────────────────

    def _handle_question(self, event: dict, say, client, triggered_by_mention: bool) -> None:
        start = time.time()
        channel = event["channel"]
        ts = event["ts"]
        text = self._strip_mention(event.get("text", ""))

        if len(text.strip()) < MIN_QUESTION_LEN:
            return

        logger.info("Processing question in %s: %.80s", channel, text)

        # Post a "thinking" placeholder in the thread
        placeholder = say(
            text="🤔 _Searching through past discussions…_",
            thread_ts=ts,
        )
        placeholder_ts = placeholder.get("ts")

        try:
            # Retrieve relevant historical Q&As
            retrieved = self._retriever.retrieve(
                question=text,
                top_k=config.top_k,
                channel_filter=channel if len(config.monitored_channels) > 1 else None,
            )

            # Generate answer
            answer_text = self._llm.generate(question=text, retrieved=retrieved)

            latency = (time.time() - start) * 1000
            response = AgentResponse(
                answer=answer_text,
                sources=retrieved,
                question=text,
                from_cache=False,
                latency_ms=latency,
            )

            slack_msg = response.format_slack_message(show_sources=config.show_sources)

            # Update the placeholder with the real answer
            if placeholder_ts:
                try:
                    client.chat_update(
                        channel=channel,
                        ts=placeholder_ts,
                        text=slack_msg,
                    )
                except Exception:
                    # Fallback: post a new message
                    say(text=slack_msg, thread_ts=ts)
            else:
                say(text=slack_msg, thread_ts=ts)

            logger.info("Answer posted in %.0fms for: %.60s", latency, text)

        except Exception as exc:
            logger.exception("Error generating answer")
            error_msg = (
                f"⚠️ Sorry, I ran into an error while processing your question: `{type(exc).__name__}`\n"
                "Please try again or ask a human teammate."
            )
            if placeholder_ts:
                try:
                    client.chat_update(channel=channel, ts=placeholder_ts, text=error_msg)
                except Exception:
                    say(text=error_msg, thread_ts=ts)
            else:
                say(text=error_msg, thread_ts=ts)

    # ── Reaction feedback handler ──────────────────────────────────────

    def _handle_reaction(self, event: dict, client) -> None:
        """
        Record 👍/👎 reactions on bot messages.
        Could be used to update reaction scores and improve ranking.
        """
        emoji = event.get("reaction", "")
        if emoji not in ("thumbsup", "+1", "thumbsdown", "-1"):
            return

        item = event.get("item", {})
        channel = item.get("channel", "")
        msg_ts = item.get("ts", "")

        if not channel or not msg_ts:
            return

        try:
            # Fetch the original message to check if it's a bot answer
            resp = client.conversations_replies(channel=channel, ts=msg_ts, limit=1)
            messages = resp.get("messages", [])
            if not messages:
                return
            msg = messages[0]
            if not msg.get("bot_id"):
                return   # Not a bot message

            polarity = "positive" if emoji in ("thumbsup", "+1") else "negative"
            logger.info("Feedback '%s' received for bot answer %s in %s", polarity, msg_ts, channel)
            # TODO: persist feedback and use it to bias retrieval scores
        except Exception as exc:
            logger.debug("Could not process reaction feedback: %s", exc)

    # ── Ingestion trigger ──────────────────────────────────────────────

    def run_ingestion(self, force_reindex: bool = False) -> dict:
        """Trigger a manual or scheduled ingestion run."""
        from slack_sdk import WebClient  # type: ignore
        client = WebClient(token=config.slack_bot_token)
        pipeline = IngestionPipeline(
            slack_client=client,
            embedder=self._embedder,
            vector_store=self._vector_store,
            database=self._db,
        )
        return pipeline.run(force_reindex=force_reindex)

    # ── Start ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the bot using Socket Mode (no public URL needed)."""
        logger.info("Starting %s via Socket Mode…", config.bot_name)
        handler = SocketModeHandler(self._app, config.slack_app_token)
        handler.start()

    @property
    def app(self) -> App:
        return self._app

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _strip_mention(text: str) -> str:
        """Remove leading @bot mention from message text."""
        import re
        return re.sub(r"^<@[A-Z0-9]+>\s*", "", text).strip()
