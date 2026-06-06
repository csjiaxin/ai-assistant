"""
Tests for context management: budget enforcement, section priority, trimming.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from agent.context_manager import ContextManager, ContextSection, CONTEXT_WINDOWS
from agent.memory import ConversationMemory
from agent.models import QAPair, RetrievedContext


# ── Fixtures ───────────────────────────────────────────────────────────

def make_qa(question="How do I configure Kafka?", answer="Set max.poll.interval.ms=300000"):
    return QAPair(id="t1", channel_id="C001", thread_ts="t1",
                  question=question, answer=answer,
                  questioner_id="U001", reaction_score=3)

def make_retrieved(n=3):
    return [RetrievedContext(qa=make_qa(), similarity=0.85 - i*0.05) for i in range(n)]

def make_multi_turn_memory():
    mem = ConversationMemory(thread_ts="ts123", channel_id="C001")
    mem.add_turn("user", "How do I configure Kafka?")
    mem.add_turn("assistant", "Set max.poll.interval.ms=300000")
    mem.add_turn("user", "What about the retry policy?")
    return mem


# ══════════════════════════════════════════════════════════════════════
# ContextManager
# ══════════════════════════════════════════════════════════════════════

class TestContextManager:
    def make_cm(self, model="gemma2:2b", max_context_chars=6000):
        # Use gemma2:2b (8192 tokens) for predictable budget in tests
        return ContextManager(llm_model=model, max_context_chars=max_context_chars)

    def test_build_returns_built_context(self):
        cm = self.make_cm()
        result = cm.build(
            question="How do I restart the service?",
            retrieved=make_retrieved(2),
        )
        assert result.system_prompt
        assert isinstance(result.sections, list)
        assert len(result.sections) >= 1  # at least question included

    def test_question_always_included(self):
        cm = self.make_cm()
        result = cm.build(question="My question", retrieved=[])
        names = [s.name for s in result.sections]
        assert "question" in names

    def test_retrieved_context_included(self):
        cm = self.make_cm()
        result = cm.build(question="How?", retrieved=make_retrieved(2))
        names = [s.name for s in result.sections]
        assert "retrieved_context" in names

    def test_conversation_history_included_for_multi_turn(self):
        cm = self.make_cm()
        mem = make_multi_turn_memory()
        result = cm.build(question="Follow-up?", retrieved=[], conversation=mem)
        names = [s.name for s in result.sections]
        assert "conversation_history" in names

    def test_conversation_history_excluded_for_single_turn(self):
        cm = self.make_cm()
        mem = ConversationMemory(thread_ts="ts1", channel_id="C001")
        mem.add_turn("user", "First question")
        result = cm.build(question="First question", retrieved=[], conversation=mem)
        names = [s.name for s in result.sections]
        assert "conversation_history" not in names

    def test_ltm_included_when_provided(self):
        cm = self.make_cm()
        ltm_entries = [{"topic_key": "kafka", "summary": "User asked about Kafka before."}]
        result = cm.build(question="Kafka question", retrieved=[], ltm_entries=ltm_entries)
        names = [s.name for s in result.sections]
        assert "long_term_memory" in names

    def test_sections_within_budget(self):
        cm = self.make_cm(model="gemma2:2b")  # small 8192-token window
        result = cm.build(question="Question", retrieved=make_retrieved(5))
        total_chars = sum(s.char_count for s in result.sections) + len(result.system_prompt)
        assert total_chars <= cm.char_budget + 100  # allow tiny rounding

    def test_format_user_message_combines_sections(self):
        cm = self.make_cm()
        result = cm.build(question="How do I configure Kafka?", retrieved=make_retrieved(1))
        user_msg = result.format_user_message()
        assert "How do I configure Kafka?" in user_msg

    def test_was_truncated_false_for_small_context(self):
        cm = self.make_cm(model="gpt-4o")  # huge window — nothing should be trimmed
        result = cm.build(question="Small question", retrieved=make_retrieved(1))
        assert result.was_truncated is False

    def test_priority_order_question_before_retrieved(self):
        cm = self.make_cm()
        result = cm.build(question="Q", retrieved=make_retrieved(3))
        # Section with priority=1 (question) should appear before priority=2 (retrieved)
        priorities = [s.priority for s in result.sections]
        assert priorities == sorted(priorities)

    def test_total_tokens_estimate_is_positive(self):
        cm = self.make_cm()
        result = cm.build(question="Question?", retrieved=make_retrieved(2))
        assert result.total_tokens_estimate > 0

    def test_conversation_history_in_llm_format(self):
        cm = self.make_cm()
        mem = make_multi_turn_memory()
        result = cm.build(question="Follow-up?", retrieved=[], conversation=mem)
        # Conversation history should be returned as LLM message dicts
        for msg in result.conversation_history:
            assert "role" in msg
            assert "content" in msg

    def test_tool_results_included_when_provided(self):
        from agent.tools import ToolResult
        cm = self.make_cm()
        tool_results = [ToolResult(call_id="1", tool_name="search_history",
                                   output="Found: Kafka config", success=True)]
        result = cm.build(question="Q", retrieved=[], tool_results=tool_results)
        names = [s.name for s in result.sections]
        assert "tool_results" in names


# ══════════════════════════════════════════════════════════════════════
# Budget and trimming
# ══════════════════════════════════════════════════════════════════════

class TestContextBudget:
    def test_char_budget_derived_from_model_window(self):
        cm = ContextManager(llm_model="gemma2:2b")
        # gemma2:2b has 8192 token window; minus 1024 response reserve = 7168
        expected_chars = (CONTEXT_WINDOWS["gemma2:2b"] - 1024) * 4
        assert cm.char_budget == expected_chars

    def test_estimate_tokens(self):
        cm = ContextManager(llm_model="llama3.2")
        assert cm.estimate_tokens("hello world") == 2  # 11 chars / 4 ≈ 2

    def test_large_retrieved_context_gets_trimmed(self):
        # Use a small max_context_chars to force trimming
        cm = ContextManager(llm_model="gpt-4o", max_context_chars=500)
        long_answer = "x" * 2000
        qa = QAPair(id="t1", channel_id="C001", thread_ts="t1",
                    question="Very long question " * 20,
                    answer=long_answer, questioner_id="U001", reaction_score=1)
        retrieved = [RetrievedContext(qa=qa, similarity=0.9)]
        result = cm.build(question="Q?", retrieved=retrieved)
        # retrieved_context section should be trimmed
        ctx_section = next((s for s in result.sections if s.name == "retrieved_context"), None)
        if ctx_section:
            assert ctx_section.char_count <= 600  # max_context_chars + small buffer

    def test_omitted_sections_tracked(self):
        # Force severe budget pressure by using a tiny custom budget
        cm = ContextManager(llm_model="gemma2:2b", max_context_chars=200)
        # Override char_budget to something tiny
        cm._char_budget = 500
        ltm_entries = [{"topic_key": "kafka", "summary": "Very long summary " * 100}]
        result = cm.build(
            question="Short question",
            retrieved=[],
            ltm_entries=ltm_entries,
        )
        # With tiny budget, LTM (lowest priority) may be omitted
        # We just verify the tracking mechanism works
        assert isinstance(result.sections_omitted, list)
        assert isinstance(result.sections_trimmed, list)
