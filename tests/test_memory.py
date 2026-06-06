"""
Tests for short-term conversation memory and long-term episodic memory.
"""
from __future__ import annotations

import time
import pytest
from unittest.mock import patch

from agent.memory import (
    ConversationTurn, ConversationMemory, ShortTermMemory, LongTermMemory
)


# ══════════════════════════════════════════════════════════════════════
# ConversationMemory (short-term, single thread)
# ══════════════════════════════════════════════════════════════════════

class TestConversationMemory:
    def make_mem(self):
        return ConversationMemory(thread_ts="ts123", channel_id="C001")

    def test_add_turn_stores_turn(self):
        mem = self.make_mem()
        mem.add_turn("user", "How do I configure Kafka?")
        assert len(mem.turns) == 1
        assert mem.turns[0].role == "user"
        assert mem.turns[0].content == "How do I configure Kafka?"

    def test_is_multi_turn_false_for_single_turn(self):
        mem = self.make_mem()
        mem.add_turn("user", "How do I configure Kafka?")
        assert mem.is_multi_turn() is False

    def test_is_multi_turn_true_for_two_user_turns(self):
        mem = self.make_mem()
        mem.add_turn("user", "How do I configure Kafka?")
        mem.add_turn("assistant", "Set max.poll.interval.ms=300000")
        mem.add_turn("user", "What about the retry policy?")
        assert mem.is_multi_turn() is True

    def test_to_llm_messages_format(self):
        mem = self.make_mem()
        mem.add_turn("user", "Hello")
        mem.add_turn("assistant", "Hi there!")
        messages = mem.to_llm_messages()
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}

    def test_format_for_prompt_excludes_last_turn(self):
        mem = self.make_mem()
        mem.add_turn("user", "First question about Kafka")
        mem.add_turn("assistant", "Here is the Kafka answer")
        mem.add_turn("user", "Follow-up about retry")
        prompt = mem.format_for_prompt()
        assert "First question" in prompt
        assert "Kafka answer" in prompt
        # Last turn (current question) should not appear
        assert "Follow-up about retry" not in prompt

    def test_format_for_prompt_empty_for_single_turn(self):
        mem = self.make_mem()
        mem.add_turn("user", "Just one question")
        prompt = mem.format_for_prompt()
        assert prompt == ""

    def test_max_turns_limit_drops_oldest(self):
        mem = self.make_mem()
        for i in range(25):   # exceed MAX_TURNS=20
            mem.add_turn("user" if i % 2 == 0 else "assistant", f"Message {i}")
        assert len(mem.turns) <= ConversationMemory.MAX_TURNS

    def test_max_chars_limit_trims(self):
        mem = self.make_mem()
        # Add large content
        for i in range(10):
            mem.add_turn("user", "x" * 1000)
            mem.add_turn("assistant", "y" * 1000)
        total = sum(len(t.content) for t in mem.turns)
        assert total <= ConversationMemory.MAX_CHARS + 2000  # allow 2 turns grace

    def test_tool_turn_in_messages(self):
        mem = self.make_mem()
        mem.add_turn("tool", "search result content", tool_name="search_history")
        messages = mem.to_llm_messages()
        assert messages[0]["role"] == "tool"
        assert messages[0]["name"] == "search_history"

    def test_turn_count(self):
        mem = self.make_mem()
        mem.add_turn("user", "Q1")
        mem.add_turn("assistant", "A1")
        assert mem.turn_count == 2


# ══════════════════════════════════════════════════════════════════════
# ShortTermMemory (in-process store)
# ══════════════════════════════════════════════════════════════════════

class TestShortTermMemory:
    def test_get_or_create_creates_new(self):
        store = ShortTermMemory()
        mem = store.get_or_create("ts001", "C001")
        assert mem.thread_ts == "ts001"
        assert mem.channel_id == "C001"

    def test_get_or_create_returns_existing(self):
        store = ShortTermMemory()
        mem1 = store.get_or_create("ts001", "C001")
        mem1.add_turn("user", "hello")
        mem2 = store.get_or_create("ts001", "C001")
        assert mem2.turn_count == 1  # same object

    def test_get_returns_none_for_missing(self):
        store = ShortTermMemory()
        assert store.get("nonexistent") is None

    def test_delete_removes_entry(self):
        store = ShortTermMemory()
        store.get_or_create("ts001", "C001")
        store.delete("ts001")
        assert store.get("ts001") is None

    def test_active_thread_count(self):
        store = ShortTermMemory()
        store.get_or_create("ts001", "C001")
        store.get_or_create("ts002", "C001")
        assert store.active_thread_count == 2

    def test_expired_entries_evicted(self):
        store = ShortTermMemory()
        mem = store.get_or_create("ts001", "C001")
        # Manually age the entry beyond TTL
        mem.last_updated = time.time() - ShortTermMemory.TTL_SECONDS - 1
        store._evict_expired()
        assert store.get("ts001") is None

    def test_max_threads_enforced(self):
        store = ShortTermMemory()
        store.MAX_THREADS = 5  # lower limit for test
        for i in range(10):
            store.get_or_create(f"ts{i:03d}", "C001")
            time.sleep(0.001)  # ensure distinct timestamps
        assert store.active_thread_count <= 5


# ══════════════════════════════════════════════════════════════════════
# LongTermMemory (SQLite)
# ══════════════════════════════════════════════════════════════════════

class TestLongTermMemory:
    @pytest.fixture
    def ltm(self, tmp_path):
        db_url = f"sqlite:///{tmp_path}/test_ltm.db"
        return LongTermMemory(database_url=db_url)

    def test_store_and_recall(self, ltm):
        ltm.store(
            user_id="U001",
            topic_key="kafka retry policy",
            summary="User asked about Kafka retry. Advised to set max.poll.interval.ms=300000.",
            channel_id="C001",
            thread_ts="ts123",
        )
        entries = ltm.recall_for_user("U001")
        assert len(entries) == 1
        assert "Kafka" in entries[0]["summary"]

    def test_store_updates_existing(self, ltm):
        ltm.store("U001", "kafka retry", "First summary")
        ltm.store("U001", "kafka retry", "Updated summary")
        entries = ltm.recall_for_user("U001")
        assert len(entries) == 1
        assert "Updated" in entries[0]["summary"]

    def test_recall_returns_empty_for_unknown_user(self, ltm):
        entries = ltm.recall_for_user("UNKNOWN")
        assert entries == []

    def test_search_by_topic_finds_relevant(self, ltm):
        ltm.store("U001", "kafka retry config", "Set max.poll.interval.ms=300000")
        ltm.store("U002", "vault secrets rotation", "Use vault kv put secret/app key=value")
        results = ltm.search_by_topic("kafka retry")
        assert len(results) >= 1
        assert "kafka" in results[0]["topic_key"].lower()

    def test_search_by_topic_returns_empty_when_no_match(self, ltm):
        ltm.store("U001", "kafka retry", "Some kafka info")
        results = ltm.search_by_topic("completely unrelated topic xyz123")
        assert results == []

    def test_forget_removes_entry(self, ltm):
        entry_id = ltm.store("U001", "kafka retry", "Kafka info")
        ltm.forget(entry_id)
        entries = ltm.recall_for_user("U001")
        assert len(entries) == 0

    def test_forget_user_removes_all(self, ltm):
        ltm.store("U001", "kafka retry", "Kafka info")
        ltm.store("U001", "vault secrets", "Vault info")
        ltm.store("U002", "gcp egress", "GCP info")
        count = ltm.forget_user("U001")
        assert count == 2
        assert ltm.recall_for_user("U001") == []
        assert ltm.recall_for_user("U002") != []  # U002 unaffected

    def test_recall_limit(self, ltm):
        for i in range(10):
            ltm.store("U001", f"topic_{i}", f"Summary {i}")
        entries = ltm.recall_for_user("U001", limit=3)
        assert len(entries) == 3
