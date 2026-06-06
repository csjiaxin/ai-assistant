"""
Background scheduler — periodic ingestion and re-indexing jobs.
Uses APScheduler (lightweight, in-process).
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore
from apscheduler.triggers.interval import IntervalTrigger  # type: ignore
from apscheduler.triggers.cron import CronTrigger  # type: ignore

from agent.config import config

logger = logging.getLogger(__name__)


class AgentScheduler:
    def __init__(self, agent):
        self._agent = agent
        self._scheduler = BackgroundScheduler(timezone="UTC")

    def start(self) -> None:
        # ── Incremental ingestion every N minutes ──────────────────────
        self._scheduler.add_job(
            func=self._incremental_ingest,
            trigger=IntervalTrigger(minutes=config.ingestion_interval_minutes),
            id="incremental_ingest",
            name="Incremental Slack ingestion",
            replace_existing=True,
            max_instances=1,
        )

        # ── Full nightly re-index (3 AM UTC) ───────────────────────────
        self._scheduler.add_job(
            func=self._full_reindex,
            trigger=CronTrigger(hour=3, minute=0),
            id="full_reindex",
            name="Nightly full re-index",
            replace_existing=True,
            max_instances=1,
        )

        self._scheduler.start()
        logger.info(
            "Scheduler started: incremental every %d min, full reindex at 03:00 UTC",
            config.ingestion_interval_minutes,
        )

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

    # ── Jobs ───────────────────────────────────────────────────────────

    def _incremental_ingest(self) -> None:
        logger.info("Running incremental ingestion…")
        try:
            stats = self._agent.run_ingestion(force_reindex=False)
            logger.info("Incremental ingestion done: %s", stats)
        except Exception:
            logger.exception("Incremental ingestion failed")

    def _full_reindex(self) -> None:
        logger.info("Running full nightly reindex…")
        try:
            stats = self._agent.run_ingestion(force_reindex=True)
            logger.info("Full reindex done: %s", stats)
        except Exception:
            logger.exception("Full reindex failed")
