"""
Tests for the LLM prompt building and response formatting.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agent.llm import _build_context_block, _build_user_message, SYSTEM_PROMPT
from agent.models import AgentResponse, QAPair, RetrievedContext


def make_context(index: int, similarity: float = 0.85, reaction_score: int = 3) -> RetrievedContext:
    qa = QAPair(
        id=f"C001_ts{index}",
        channel_id="C001",
        thread_ts=f"ts{index}",
        question=f"Question number {index} about Kafka configuration?",
        answer=f"Answer number {index}: use max.poll.interval.ms = 300000",
        questioner_id="U001",
        reaction_score=reaction_score,
        slack_url=f"https://slack.com/archives/C001/pts{index}",
    )
    return RetrievedContext(qa=qa, similarity=similarity)


# ── Context block building ─────────────────────────────────────────────

class TestBuildContextBlock:
    def test_no_results_returns_no_history_message(self):
        result = _build_context_block([], max_chars=6000)
        assert "No relevant" in result

    def test_single_result_is_included(self):
        ctx = make_context(1, similarity=0.90)
        result = _build_context_block([ctx], max_chars=6000)
        assert "[Source 1]" in result
        assert "90%" in result
        assert ctx.qa.question in result
        assert ctx.qa.answer in result

    def test_multiple_results_all_included(self):
        contexts = [make_context(i) for i in range(1, 4)]
        result = _build_context_block(contexts, max_chars=6000)
        assert "[Source 1]" in result
        assert "[Source 2]" in result
        assert "[Source 3]" in result

    def test_max_chars_truncates_context(self):
        # Create many large contexts
        contexts = [make_context(i) for i in range(20)]
        result = _build_context_block(contexts, max_chars=200)
        # Should only include a few sources
        assert "[Source 1]" in result
        # Not all 20 should fit
        assert "[Source 15]" not in result

    def test_reaction_score_shown(self):
        ctx = make_context(1, reaction_score=42)
        result = _build_context_block([ctx], max_chars=6000)
        assert "👍 42" in result


# ── User message building ─────────────────────────────────────────────

class TestBuildUserMessage:
    def test_contains_question(self):
        msg = _build_user_message("How do I configure Kafka?", "some context")
        assert "How do I configure Kafka?" in msg

    def test_contains_context(self):
        msg = _build_user_message("question?", "MY CONTEXT BLOCK")
        assert "MY CONTEXT BLOCK" in msg

    def test_contains_citation_instruction(self):
        msg = _build_user_message("question?", "context")
        assert "Source" in msg or "cite" in msg.lower()


# ── System prompt ─────────────────────────────────────────────────────

class TestSystemPrompt:
    def test_system_prompt_is_nonempty(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_mentions_hallucination(self):
        assert "hallucinate" in SYSTEM_PROMPT.lower() or "not" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_sources(self):
        assert "Source" in SYSTEM_PROMPT or "cite" in SYSTEM_PROMPT.lower()


# ── AgentResponse formatting ─────────────────────────────────────────

class TestAgentResponseFormat:
    def make_response(self, sources=None, from_cache=False):
        # Use sentinel to distinguish "not provided" from "explicitly empty list"
        resolved_sources = [make_context(1, similarity=0.91, reaction_score=5)] if sources is None else sources
        return AgentResponse(
            answer="You should set `max.poll.interval.ms = 300000` in your Kafka consumer config.",
            sources=resolved_sources,
            question="How do I configure Kafka retry?",
            from_cache=from_cache,
            latency_ms=342.5,
        )

    def test_slack_message_contains_answer(self):
        resp = self.make_response()
        msg = resp.format_slack_message(show_sources=True)
        assert "max.poll.interval.ms" in msg

    def test_slack_message_with_sources_has_links(self):
        resp = self.make_response()
        msg = resp.format_slack_message(show_sources=True)
        assert "Source 1" in msg
        assert "slack.com" in msg

    def test_slack_message_without_sources_has_no_links(self):
        resp = self.make_response()
        msg = resp.format_slack_message(show_sources=False)
        assert "Related discussions" not in msg

    def test_cached_response_shows_indicator(self):
        resp = self.make_response(from_cache=True)
        msg = resp.format_slack_message()
        assert "cached" in msg.lower() or "⚡" in msg

    def test_no_sources_hides_section(self):
        resp = self.make_response(sources=[])
        msg = resp.format_slack_message(show_sources=True)
        assert "Related discussions" not in msg
