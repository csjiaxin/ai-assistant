"""
Tests for the CSV ingestion pipeline.
"""
from __future__ import annotations

import csv
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from agent.csv_ingestion import CSVIngestionPipeline


# ── Helpers ────────────────────────────────────────────────────────────

def make_pipeline():
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 384
    vs = MagicMock()
    db = MagicMock()
    db.already_indexed.return_value = False
    return CSVIngestionPipeline(
        embedder=embedder,
        vector_store=vs,
        database=db,
        channel_id="test_channel",
    )


def write_csv(rows: list[dict], path: str) -> None:
    """Write test CSV. Reply rows must have User starting with '↳ '."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Time", "User", "Message"])
        writer.writeheader()
        writer.writerows(rows)


# ── Thread parsing ─────────────────────────────────────────────────────

class TestCSVParsing:

    def test_parses_single_thread(self, tmp_path):
        csv_file = str(tmp_path / "test.csv")
        write_csv([
            {"Time": "12/1/2025, 2:00:00 AM", "User": "alice",
             "Message": "How do I configure the Kafka retry policy?"},
            {"Time": "12/1/2025, 2:05:00 AM", "User": "↳ bob",
             "Message": "Set max.poll.interval.ms to 300000 in your consumer config."},
            {"Time": "12/1/2025, 2:07:00 AM", "User": "↳ alice",
             "Message": "Thanks! That worked perfectly."},
        ], csv_file)

        pipeline = make_pipeline()
        threads = pipeline._parse_csv(Path(csv_file))

        assert len(threads) == 1
        assert "Kafka" in threads[0]["root_text"]
        assert len(threads[0]["replies"]) == 2
        assert threads[0]["root_user"] == "alice"

    def test_parses_multiple_threads(self, tmp_path):
        csv_file = str(tmp_path / "test.csv")
        write_csv([
            {"Time": "12/1/2025, 2:00:00 AM", "User": "alice",
             "Message": "How do I restart the service proxy?"},
            {"Time": "12/1/2025, 2:05:00 AM", "User": "↳ bob",
             "Message": "Run: systemctl restart service-proxy"},
            {"Time": "12/1/2025, 3:00:00 AM", "User": "charlie",
             "Message": "Why is my Docker build failing with disk space error?"},
            {"Time": "12/1/2025, 3:10:00 AM", "User": "↳ dave",
             "Message": "Clean up with: docker system prune -af"},
        ], csv_file)

        pipeline = make_pipeline()
        threads = pipeline._parse_csv(Path(csv_file))

        assert len(threads) == 2
        assert threads[0]["root_user"] == "alice"
        assert threads[1]["root_user"] == "charlie"

    def test_skips_bot_replies(self, tmp_path):
        csv_file = str(tmp_path / "test.csv")
        write_csv([
            {"Time": "12/1/2025, 2:00:00 AM", "User": "alice",
             "Message": "How do I configure the retry policy?"},
            {"Time": "12/1/2025, 2:01:00 AM", "User": "↳ Unknown User (bot_message)",
             "Message": "Please follow these guidelines to ensure we can assist you."},
            {"Time": "12/1/2025, 2:05:00 AM", "User": "↳ bob",
             "Message": "Set max.poll.interval.ms to 300000."},
        ], csv_file)

        pipeline = make_pipeline()
        threads = pipeline._parse_csv(Path(csv_file))

        assert len(threads) == 1
        # Bot reply should be excluded
        assert len(threads[0]["replies"]) == 1
        assert threads[0]["replies"][0]["user"] == "bob"

    def test_skips_thread_without_replies(self, tmp_path):
        csv_file = str(tmp_path / "test.csv")
        write_csv([
            {"Time": "12/1/2025, 2:00:00 AM", "User": "alice",
             "Message": "How do I configure the retry policy?"},
            # No replies — should not produce a thread
            {"Time": "12/1/2025, 3:00:00 AM", "User": "bob",
             "Message": "Anyone know about the deployment process?"},
            {"Time": "12/1/2025, 3:10:00 AM", "User": "↳ charlie",
             "Message": "Check the runbook in Confluence."},
        ], csv_file)

        pipeline = make_pipeline()
        threads = pipeline._parse_csv(Path(csv_file))

        assert len(threads) == 1  # Only bob's thread (has a reply)
        assert threads[0]["root_user"] == "bob"

    def test_skips_bot_root_messages(self, tmp_path):
        csv_file = str(tmp_path / "test.csv")
        write_csv([
            {"Time": "12/1/2025, 2:00:00 AM", "User": "jiraservicemanagement",
             "Message": "Ticket created for this issue."},
            {"Time": "12/1/2025, 2:05:00 AM", "User": "↳ bob",
             "Message": "Thanks, I'll follow up there."},
            {"Time": "12/1/2025, 3:00:00 AM", "User": "alice",
             "Message": "How do I rotate Vault secrets in staging?"},
            {"Time": "12/1/2025, 3:10:00 AM", "User": "↳ charlie",
             "Message": "Use vault kv put secret/app key=value"},
        ], csv_file)

        pipeline = make_pipeline()
        threads = pipeline._parse_csv(Path(csv_file))

        assert len(threads) == 1
        assert threads[0]["root_user"] == "alice"


# ── Text cleaning ──────────────────────────────────────────────────────

class TestCSVCleanText:

    def test_html_entities(self):
        result = CSVIngestionPipeline._clean_text("this &amp; that &lt;value&gt;")
        assert "&amp;" not in result
        assert "this & that" in result

    def test_strips_user_mentions(self):
        result = CSVIngestionPipeline._clean_text("<@U12345> please check this")
        assert "<@" not in result
        assert "@user" in result

    def test_unwraps_url_with_label(self):
        result = CSVIngestionPipeline._clean_text("see <https://example.com|this link>")
        assert "this link" in result
        assert "<https" not in result

    def test_unwraps_bare_url(self):
        result = CSVIngestionPipeline._clean_text("see <https://example.com>")
        assert "https://example.com" in result
        assert "<https" not in result

    def test_removes_quoted_reply_lines(self):
        text = "&gt; original message\nActual reply here"
        result = CSVIngestionPipeline._clean_text(text)
        assert "original message" not in result
        assert "Actual reply here" in result

    def test_plain_text_unchanged(self):
        text = "Set max.poll.interval.ms = 300000 in consumer.properties"
        assert CSVIngestionPipeline._clean_text(text) == text


# ── QAPair building from thread ────────────────────────────────────────

class TestThreadToQA:

    def make_thread(self, question="How do I restart the service proxy?",
                    replies=None, root_user="alice"):
        return {
            "root_time": "12/1/2025, 2:00:00 AM",
            "root_user": root_user,
            "root_text": question,
            "replies": replies or [
                {"time": "12/1/2025, 2:05:00 AM", "user": "bob",
                 "text": "Run: systemctl restart service-proxy"},
            ],
        }

    def test_builds_valid_qa_pair(self):
        pipeline = make_pipeline()
        thread = self.make_thread()
        qa = pipeline._thread_to_qa(thread)
        assert qa is not None
        assert "restart" in qa.question
        assert "systemctl" in qa.answer
        assert qa.questioner_id == "alice"
        assert qa.channel_id == "test_channel"

    def test_returns_none_for_short_question(self):
        pipeline = make_pipeline()
        thread = self.make_thread(question="hi")
        qa = pipeline._thread_to_qa(thread)
        assert qa is None

    def test_returns_none_for_no_substantive_replies(self):
        pipeline = make_pipeline()
        thread = self.make_thread(replies=[
            {"time": "12/1/2025, 2:05:00 AM", "user": "bot", "text": "ok"},
        ])
        qa = pipeline._thread_to_qa(thread)
        assert qa is None

    def test_reaction_score_includes_positive_words(self):
        pipeline = make_pipeline()
        thread = self.make_thread(replies=[
            {"time": "12/1/2025, 2:05:00 AM", "user": "bob",
             "text": "Try this config. It worked for me."},
            {"time": "12/1/2025, 2:10:00 AM", "user": "alice",
             "text": "Thanks! That solved my problem."},
        ])
        qa = pipeline._thread_to_qa(thread)
        assert qa is not None
        assert qa.reaction_score > 0

    def test_respondents_excludes_questioner(self):
        pipeline = make_pipeline()
        thread = self.make_thread(
            root_user="alice",
            replies=[
                {"time": "12/1/2025, 2:05:00 AM", "user": "bob", "text": "Try this approach."},
                {"time": "12/1/2025, 2:08:00 AM", "user": "alice", "text": "Let me try that."},
                {"time": "12/1/2025, 2:10:00 AM", "user": "charlie", "text": "Also check the docs."},
            ]
        )
        qa = pipeline._thread_to_qa(thread)
        assert qa is not None
        assert "alice" not in qa.respondents
        assert "bob" in qa.respondents
        assert "charlie" in qa.respondents

    def test_qa_id_is_stable_and_safe(self):
        pipeline = make_pipeline()
        thread = self.make_thread()
        qa = pipeline._thread_to_qa(thread)
        assert qa is not None
        # ID should only have safe characters
        import re
        assert re.match(r'^[a-zA-Z0-9_]+$', qa.id)

    def test_timestamp_parsed_correctly(self):
        from datetime import datetime
        pipeline = make_pipeline()
        thread = self.make_thread()
        qa = pipeline._thread_to_qa(thread)
        assert qa is not None
        assert qa.created_at is not None
        assert qa.created_at.year == 2025
        assert qa.created_at.month == 12


# ── Full ingestion run ────────────────────────────────────────────────

class TestCSVIngestionRun:

    def test_full_ingest_produces_embeddings(self, tmp_path):
        csv_file = str(tmp_path / "slack.csv")
        write_csv([
            {"Time": "12/1/2025, 2:00:00 AM", "User": "alice",
             "Message": "How do I configure the Kafka consumer retry policy?"},
            {"Time": "12/1/2025, 2:05:00 AM", "User": "↳ bob",
             "Message": "Set max.poll.interval.ms = 300000 in consumer.properties."},
            {"Time": "12/1/2025, 3:00:00 AM", "User": "charlie",
             "Message": "Why does my Docker build keep failing on the CI runner?"},
            {"Time": "12/1/2025, 3:10:00 AM", "User": "↳ dave",
             "Message": "The runner is out of disk space. Run: docker system prune -af"},
        ], csv_file)

        pipeline = make_pipeline()
        stats = pipeline.ingest_file(csv_file)

        assert stats["threads_found"] == 2
        assert stats["qa_pairs"] == 2
        assert stats["embedded"] == 2
        assert stats["skipped"] == 0
        # Embedder called once per QA pair
        assert pipeline._embedder.embed.call_count == 2

    def test_skips_already_indexed(self, tmp_path):
        csv_file = str(tmp_path / "slack.csv")
        write_csv([
            {"Time": "12/1/2025, 2:00:00 AM", "User": "alice",
             "Message": "How do I configure the Kafka retry policy?"},
            {"Time": "12/1/2025, 2:05:00 AM", "User": "↳ bob",
             "Message": "Set max.poll.interval.ms = 300000."},
        ], csv_file)

        pipeline = make_pipeline()
        pipeline._db.already_indexed.return_value = True  # pretend already indexed
        stats = pipeline.ingest_file(csv_file, force_reindex=False)

        assert stats["skipped"] == 1
        assert stats["embedded"] == 0
        pipeline._embedder.embed.assert_not_called()

    def test_force_reindex_ignores_already_indexed(self, tmp_path):
        csv_file = str(tmp_path / "slack.csv")
        write_csv([
            {"Time": "12/1/2025, 2:00:00 AM", "User": "alice",
             "Message": "How do I configure the Kafka retry policy?"},
            {"Time": "12/1/2025, 2:05:00 AM", "User": "↳ bob",
             "Message": "Set max.poll.interval.ms = 300000."},
        ], csv_file)

        pipeline = make_pipeline()
        pipeline._db.already_indexed.return_value = True
        stats = pipeline.ingest_file(csv_file, force_reindex=True)

        assert stats["embedded"] == 1
        assert stats["skipped"] == 0

    def test_file_not_found_raises(self):
        pipeline = make_pipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.ingest_file("/nonexistent/path/file.csv")
