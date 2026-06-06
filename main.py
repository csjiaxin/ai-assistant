#!/usr/bin/env python3
"""
Slack AI Support Agent — main entrypoint.

Usage:
    python main.py serve        # Start bot + scheduler (production)
    python main.py ingest       # Run ingestion once and exit
    python main.py reindex      # Force full reindex and exit
    python main.py stats        # Print vector store stats
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def cmd_serve(args) -> None:
    """Start the bot and background scheduler."""
    from agent.bot import SupportAgent
    from agent.scheduler import AgentScheduler
    from agent.metrics import AgentMetrics

    agent = SupportAgent().init()
    scheduler = AgentScheduler(agent)
    metrics = AgentMetrics(port=int(getattr(args, "metrics_port", 8000)))

    metrics.start_server()
    scheduler.start()

    # Run initial ingestion on startup
    if not getattr(args, "no_initial_ingest", False):
        logger.info("Running initial ingestion on startup…")
        try:
            stats = agent.run_ingestion()
            logger.info("Initial ingestion: %s", stats)
        except Exception:
            logger.exception("Initial ingestion failed — continuing anyway")

    def _shutdown(sig, frame):
        logger.info("Shutting down…")
        scheduler.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    agent.start()   # Blocks (Socket Mode handler)


def cmd_ingest(args) -> None:
    """Run incremental ingestion once and exit."""
    from agent.bot import SupportAgent
    agent = SupportAgent().init()
    stats = agent.run_ingestion(force_reindex=False)
    print("Ingestion complete:", stats)


def cmd_reindex(args) -> None:
    """Force full reindex of all threads."""
    from agent.bot import SupportAgent
    agent = SupportAgent().init()
    stats = agent.run_ingestion(force_reindex=True)
    print("Reindex complete:", stats)


def cmd_ingest_csv(args) -> None:
    """Ingest Slack history from a local CSV export file (no Slack API needed)."""
    from agent.csv_ingestion import CSVIngestionPipeline
    from agent.embedder import Embedder
    from agent.vector_store import create_vector_store
    from agent.database import Database

    logger.info("Loading embedding model (first run downloads ~80MB)…")
    embedder = Embedder()
    vs = create_vector_store()
    db = Database()

    pipeline = CSVIngestionPipeline(
        embedder=embedder,
        vector_store=vs,
        database=db,
        channel_id=getattr(args, "channel_id", "csv_import"),
    )
    stats = pipeline.ingest_file(
        csv_path=args.csv_file,
        force_reindex=getattr(args, "force_reindex", False),
    )
    print("\n✅ CSV ingestion complete:")
    print(f"   Threads found : {stats['threads_found']}")
    print(f"   Q&A pairs     : {stats['qa_pairs']}")
    print(f"   Embedded      : {stats['embedded']}")
    print(f"   Skipped       : {stats['skipped']} (already indexed)")
    print(f"\nVector store now has {vs.count()} documents.")


def cmd_stats(args) -> None:
    """Print current vector store and database stats."""
    from agent.vector_store import create_vector_store
    from agent.database import Database
    vs = create_vector_store()
    db = Database()
    print(f"Vector store documents : {vs.count()}")
    print(f"Database Q&A pairs     : {db.count()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Slack AI Support Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = sub.add_parser("serve", help="Start the Slack bot and scheduler")
    p_serve.add_argument("--metrics-port", type=int, default=8000, help="Prometheus metrics port")
    p_serve.add_argument("--no-initial-ingest", action="store_true", help="Skip ingestion on startup")
    p_serve.set_defaults(func=cmd_serve)

    # ingest (from Slack API)
    p_ingest = sub.add_parser("ingest", help="Run incremental ingestion via Slack API and exit")
    p_ingest.set_defaults(func=cmd_ingest)

    # ingest-csv (from local CSV export — no Slack API needed)
    p_csv = sub.add_parser("ingest-csv", help="Ingest from a local Slack CSV export (no API key needed)")
    p_csv.add_argument("csv_file", help="Path to the CSV file (columns: Time, User, Message)")
    p_csv.add_argument("--channel-id", default="csv_import", help="Logical channel ID to tag records with")
    p_csv.add_argument("--force-reindex", action="store_true", help="Re-embed even already-indexed records")
    p_csv.set_defaults(func=cmd_ingest_csv)

    # reindex
    p_reindex = sub.add_parser("reindex", help="Force full reindex and exit")
    p_reindex.set_defaults(func=cmd_reindex)

    # stats
    p_stats = sub.add_parser("stats", help="Print vector store stats")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
