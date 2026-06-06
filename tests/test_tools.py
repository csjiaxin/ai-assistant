"""
Tests for the tool-use framework: registry, parsing, execution, and individual tools.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock

from agent.tools import (
    ToolExecutor, ToolCall, ToolResult,
    search_history, search_bm25, get_thread,
    extract_config, clarify, tool_schema_for_prompt, get_tool_definitions,
)
from agent.models import QAPair, RetrievedContext


# ── Fixtures ───────────────────────────────────────────────────────────

def make_qa(id="t1", question="How do I configure Kafka?", answer="Set max.poll.interval.ms=300000"):
    return QAPair(id=id, channel_id="C001", thread_ts=id,
                  question=question, answer=answer,
                  questioner_id="U001", reaction_score=3)

def make_retriever(results=None):
    r = MagicMock()
    r.retrieve.return_value = results or [
        RetrievedContext(qa=make_qa(), similarity=0.85)
    ]
    return r

def make_bm25_retriever(results=None):
    r = MagicMock()
    r.retrieve.return_value = results or [
        RetrievedContext(qa=make_qa(), similarity=0.72)
    ]
    return r

def make_db(qa=None):
    db = MagicMock()
    db.get_qa_by_id.return_value = qa or make_qa()
    return db


# ── Registry ───────────────────────────────────────────────────────────

class TestRegistry:
    def test_all_tools_registered(self):
        defs = get_tool_definitions()
        names = {d.name for d in defs}
        assert "search_history" in names
        assert "search_bm25" in names
        assert "get_thread" in names
        assert "extract_config" in names
        assert "clarify" in names

    def test_tool_schema_for_prompt_contains_all_tools(self):
        schema = tool_schema_for_prompt()
        assert "search_history" in schema
        assert "search_bm25" in schema
        assert "get_thread" in schema
        assert "clarify" in schema

    def test_tool_definitions_have_required_fields(self):
        for td in get_tool_definitions():
            assert td.name
            assert td.description
            assert isinstance(td.parameters, dict)
            assert callable(td.fn)


# ── ToolExecutor: parsing ──────────────────────────────────────────────

class TestToolCallParsing:
    def make_executor(self):
        return ToolExecutor()

    def test_parses_single_tool_call(self):
        output = '<tool_call>{"name": "search_history", "arguments": {"query": "Kafka retry"}, "id": "1"}</tool_call>'
        executor = self.make_executor()
        calls = executor.parse_tool_calls(output)
        assert len(calls) == 1
        assert calls[0].name == "search_history"
        assert calls[0].arguments["query"] == "Kafka retry"
        assert calls[0].call_id == "1"

    def test_parses_multiple_tool_calls(self):
        output = (
            '<tool_call>{"name": "search_history", "arguments": {"query": "503 error"}, "id": "1"}</tool_call>\n'
            '<tool_call>{"name": "search_bm25", "arguments": {"query": "obsdeck-metrics"}, "id": "2"}</tool_call>'
        )
        executor = self.make_executor()
        calls = executor.parse_tool_calls(output)
        assert len(calls) == 2
        assert calls[0].name == "search_history"
        assert calls[1].name == "search_bm25"

    def test_returns_empty_when_no_tool_calls(self):
        output = "This is a regular LLM response with no tool calls."
        executor = self.make_executor()
        calls = executor.parse_tool_calls(output)
        assert calls == []

    def test_has_tool_calls_true(self):
        output = '<tool_call>{"name": "clarify", "arguments": {"question": "Which env?"}, "id": "1"}</tool_call>'
        executor = self.make_executor()
        assert executor.has_tool_calls(output) is True

    def test_has_tool_calls_false(self):
        executor = self.make_executor()
        assert executor.has_tool_calls("plain response") is False

    def test_handles_malformed_json_gracefully(self):
        output = '<tool_call>not valid json at all</tool_call>'
        executor = self.make_executor()
        calls = executor.parse_tool_calls(output)
        assert calls == []  # graceful failure, no exception

    def test_handles_multiline_tool_call(self):
        output = '''<tool_call>{
  "name": "search_history",
  "arguments": {
    "query": "cross-IC service proxy headers",
    "top_k": 3
  },
  "id": "42"
}</tool_call>'''
        executor = self.make_executor()
        calls = executor.parse_tool_calls(output)
        assert len(calls) == 1
        assert calls[0].arguments["top_k"] == 3


# ── ToolExecutor: execution ────────────────────────────────────────────

class TestToolExecution:
    def make_executor(self, retriever=None, bm25=None, db=None):
        return ToolExecutor(
            retriever=retriever or make_retriever(),
            bm25_retriever=bm25 or make_bm25_retriever(),
            database=db or make_db(),
        )

    def test_executes_search_history(self):
        executor = self.make_executor()
        call = ToolCall(name="search_history", arguments={"query": "Kafka retry"}, call_id="1")
        result = executor.execute(call)
        assert result.success is True
        assert result.tool_name == "search_history"
        assert "Result 1" in result.output or "Kafka" in result.output

    def test_executes_search_bm25(self):
        executor = self.make_executor()
        call = ToolCall(name="search_bm25", arguments={"query": "obsdeck-metrics"}, call_id="2")
        result = executor.execute(call)
        assert result.success is True

    def test_executes_get_thread(self):
        executor = self.make_executor()
        call = ToolCall(name="get_thread", arguments={"thread_id": "t1"}, call_id="3")
        result = executor.execute(call)
        assert result.success is True
        assert "Kafka" in result.output

    def test_returns_error_for_unknown_tool(self):
        executor = self.make_executor()
        call = ToolCall(name="nonexistent_tool", arguments={}, call_id="99")
        result = executor.execute(call)
        assert result.success is False
        assert result.error is not None
        assert "Unknown tool" in result.error

    def test_execute_all_runs_all_calls(self):
        executor = self.make_executor()
        output = (
            '<tool_call>{"name": "search_history", "arguments": {"query": "503"}, "id": "1"}</tool_call>'
            '<tool_call>{"name": "search_bm25", "arguments": {"query": "ERS"}, "id": "2"}</tool_call>'
        )
        results = executor.execute_all(output)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_format_results_for_prompt(self):
        results = [
            ToolResult(call_id="1", tool_name="search_history", output="Found: Kafka retry config", success=True),
            ToolResult(call_id="2", tool_name="search_bm25", output="", error="No results", success=False),
        ]
        formatted = ToolExecutor.format_results_for_prompt(results)
        assert "search_history" in formatted
        assert "Found: Kafka retry config" in formatted
        assert "ERROR" in formatted
        assert "No results" in formatted


# ── Individual tool functions ──────────────────────────────────────────

class TestSearchHistory:
    def test_returns_results(self):
        retriever = make_retriever()
        result = search_history("Kafka retry", top_k=3, _retriever=retriever)
        assert "Result 1" in result
        retriever.retrieve.assert_called_once_with("Kafka retry", top_k=3)

    def test_caps_top_k_at_10(self):
        retriever = make_retriever()
        search_history("question", top_k=50, _retriever=retriever)
        retriever.retrieve.assert_called_once_with("question", top_k=10)

    def test_returns_no_results_message_when_empty(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        result = search_history("obscure", _retriever=retriever)
        assert "No relevant" in result

    def test_returns_error_when_no_retriever(self):
        result = search_history("question", _retriever=None)
        assert "Error" in result


class TestExtractConfig:
    def test_extracts_code_blocks(self):
        text = "Set this:\n```\nmax.poll.interval.ms=300000\n```\nThen restart."
        result = extract_config(text)
        assert "max.poll.interval.ms" in result

    def test_extracts_inline_code(self):
        text = "Use `max.poll.interval.ms` and `retry.backoff.ms` in your config."
        result = extract_config(text)
        assert "max.poll.interval.ms" in result

    def test_keyword_filter(self):
        text = "Use `max.poll.interval.ms` and `retry.backoff.ms` and `auto.offset.reset`."
        result = extract_config(text, keyword="retry")
        assert "retry.backoff.ms" in result
        assert "auto.offset.reset" not in result

    def test_returns_message_when_no_config(self):
        result = extract_config("This is plain text with no configuration.")
        assert "No configuration" in result


class TestClarify:
    def test_returns_clarify_marker(self):
        result = clarify("Which environment are you deploying to?")
        assert result.startswith("__CLARIFY__:")
        assert "environment" in result

    def test_marker_format(self):
        q = "Is this in staging or production?"
        result = clarify(q)
        assert result == f"__CLARIFY__:{q}"


class TestGetThread:
    def test_returns_thread_content(self):
        qa = make_qa(question="How do I restart?", answer="Run systemctl restart")
        db = make_db(qa=qa)
        result = get_thread("t1", _database=db)
        assert "How do I restart?" in result
        assert "systemctl restart" in result

    def test_returns_not_found_message(self):
        db = MagicMock()
        db.get_qa_by_id.return_value = None
        result = get_thread("nonexistent", _database=db)
        assert "not found" in result.lower()

    def test_returns_error_without_database(self):
        result = get_thread("t1", _database=None)
        assert "Error" in result
