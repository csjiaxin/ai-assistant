"""
Tests for the ingestion pipeline — Q&A extraction and text cleaning.
"""
from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from agent.ingestion import IngestionPipeline
from agent.models import QAPair


# ── Fixtures ───────────────────────────────────────────────────────────

def make_pipeline():
    slack_client = MagicMock()
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 1536
    embedder.embed_batch.return_value = [[0.1] * 1536]
    vector_store = MagicMock()
    database = MagicMock()
    database.already_indexed.return_value = False
    return IngestionPipeline(
        slack_client=slack_client,
        embedder=embedder,
        vector_store=vector_store,
        database=database,
    )


# ── Text cleaning ──────────────────────────────────────────────────────

class TestCleanText:
    def test_strips_user_mentions(self):
        result = IngestionPipeline._clean_text("<@U12345> hey there")
        assert "<@" not in result
        assert "@user" in result

    def test_strips_channel_refs(self):
        result = IngestionPipeline._clean_text("post in <#C12345|general> please")
        assert "<#" not in result
        assert "#general" in result

    def test_strips_here_mention(self):
        result = IngestionPipeline._clean_text("<!here> anyone around?")
        assert "<!here>" not in result

    def test_unwraps_urls(self):
        result = IngestionPipeline._clean_text("see <https://example.com|this link>")
        assert "https://example.com" in result

    def test_plain_text_unchanged(self):
        text = "How do I restart the Kafka consumer?"
        assert IngestionPipeline._clean_text(text) == text

    def test_empty_string(self):
        assert IngestionPipeline._clean_text("") == ""


# ── Q&A pair building ──────────────────────────────────────────────────

class TestBuildQAPair:
    def test_returns_none_for_short_question(self):
        pipeline = make_pipeline()
        root_msg = {"ts": "1234567890.000100", "text": "hi", "user": "U001", "reactions": []}
        pipeline._fetch_thread_replies = MagicMock(return_value=[])
        result = pipeline._build_qa_pair("C001", "1234567890.000100", root_msg)
        assert result is None

    def test_returns_none_when_no_replies(self):
        pipeline = make_pipeline()
        root_msg = {
            "ts": "1234567890.000100",
            "text": "How do I configure the retry policy for our Kafka consumer?",
            "user": "U001",
            "reactions": [],
        }
        pipeline._fetch_thread_replies = MagicMock(return_value=[])
        result = pipeline._build_qa_pair("C001", "1234567890.000100", root_msg)
        assert result is None

    def test_builds_valid_qa_pair(self):
        pipeline = make_pipeline()
        root_msg = {
            "ts": "1234567890.000100",
            "text": "How do I configure the retry policy for our Kafka consumer?",
            "user": "U001",
            "reactions": [{"name": "thumbsup", "count": 3}],
        }
        replies = [
            {"text": "Set max.poll.interval.ms to a higher value.", "user": "U002", "reactions": []},
            {"text": "Also check retry.backoff.ms in your consumer config.", "user": "U003", "reactions": [{"name": "+1", "count": 2}]},
        ]
        pipeline._fetch_thread_replies = MagicMock(return_value=replies)
        qa = pipeline._build_qa_pair("C001", "1234567890.000100", root_msg)
        assert qa is not None
        assert isinstance(qa, QAPair)
        assert "Kafka" in qa.question
        assert "max.poll.interval.ms" in qa.answer
        assert "retry.backoff.ms" in qa.answer
        assert qa.reaction_score == 5  # 3 on root + 2 on reply
        assert qa.questioner_id == "U001"
        assert "U002" in qa.respondents
        assert "U003" in qa.respondents
        assert qa.channel_id == "C001"

    def test_slack_url_format(self):
        pipeline = make_pipeline()
        root_msg = {
            "ts": "1710012345.678900",
            "text": "What is the correct way to rotate Vault secrets in staging?",
            "user": "U001",
            "reactions": [],
        }
        pipeline._fetch_thread_replies = MagicMock(return_value=[
            {"text": "Use vault kv put secret/myapp key=value", "user": "U002", "reactions": []}
        ])
        qa = pipeline._build_qa_pair("C04ABCDEF", "1710012345.678900", root_msg)
        assert qa is not None
        assert "C04ABCDEF" in qa.slack_url
        assert qa.id == "C04ABCDEF_1710012345.678900"

    def test_reaction_counting(self):
        pipeline = make_pipeline()
        root_msg = {
            "ts": "1234567890.000100",
            "text": "Why does my Docker build keep failing on the CI pipeline?",
            "user": "U001",
            "reactions": [
                {"name": "thumbsup", "count": 5},
                {"name": "fire", "count": 2},
                {"name": "eyes", "count": 10},   # not a positive reaction
            ],
        }
        pipeline._fetch_thread_replies = MagicMock(return_value=[
            {"text": "Check disk space on the CI runner.", "user": "U002",
             "reactions": [{"name": "white_check_mark", "count": 3}]}
        ])
        qa = pipeline._build_qa_pair("C001", "1234567890.000100", root_msg)
        assert qa.reaction_score == 10  # 5 + 2 (root) + 3 (reply); "eyes" excluded


# ── QAPair model ───────────────────────────────────────────────────────

class TestQAPairModel:
    def make_qa(self, **kwargs):
        defaults = dict(
            id="C001_123456.000",
            channel_id="C001",
            thread_ts="123456.000",
            question="How do I restart the service?",
            answer="Run: systemctl restart myservice",
            questioner_id="U001",
        )
        defaults.update(kwargs)
        return QAPair(**defaults)

    def test_combined_text_includes_both(self):
        qa = self.make_qa()
        assert "Question:" in qa.combined_text
        assert "Answer:" in qa.combined_text
        assert qa.question in qa.combined_text
        assert qa.answer in qa.combined_text

    def test_metadata_dict_has_required_keys(self):
        qa = self.make_qa(created_at=datetime(2024, 3, 10))
        meta = qa.to_metadata()
        for key in ("id", "channel_id", "question", "answer", "reaction_score", "slack_url"):
            assert key in meta

    def test_metadata_question_truncated_at_500(self):
        long_q = "a" * 600
        qa = self.make_qa(question=long_q)
        meta = qa.to_metadata()
        assert len(meta["question"]) <= 500

    def test_metadata_answer_truncated_at_2000(self):
        long_a = "b" * 3000
        qa = self.make_qa(answer=long_a)
        meta = qa.to_metadata()
        assert len(meta["answer"]) <= 2000


# ── RetrievedContext formatting ────────────────────────────────────────

class TestRetrievedContext:
    def test_format_for_prompt_contains_source_index(self):
        from agent.models import RetrievedContext
        qa = QAPair(
            id="C001_ts",
            channel_id="C001",
            thread_ts="ts",
            question="How do I set up Redis?",
            answer="Install with: apt install redis-server",
            questioner_id="U001",
            reaction_score=4,
            slack_url="https://slack.com/archives/C001/p123",
        )
        ctx = RetrievedContext(qa=qa, similarity=0.87)
        formatted = ctx.format_for_prompt(2)
        assert "[Source 2]" in formatted
        assert "87%" in formatted
        assert "👍 4" in formatted
        assert qa.question in formatted
        assert qa.answer in formatted
