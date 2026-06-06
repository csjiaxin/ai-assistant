"""
Tool-use layer for the Slack AI Support Agent.

Implements a lightweight tool-calling framework that lets the LLM invoke
structured actions before generating its final answer. Tools are registered
via a decorator and dispatched by the ToolExecutor.

Design principles:
  - Tools are pure Python functions with JSON-schema-described parameters
  - Tool calls are extracted from LLM output using a structured XML tag format
    (works with any LLM, no vendor-specific function-calling API required)
  - Each tool returns a ToolResult that is injected back into the conversation
  - Tools are sandboxed: they cannot mutate agent state or call the network
    unless explicitly allowed

Available tools:
  - search_history    : semantic search over indexed Q&A threads (RAG)
  - search_bm25       : keyword search over indexed Q&A threads (BM25 fallback)
  - get_thread        : fetch full thread text by ID from the database
  - summarise_thread  : summarise a long thread into key points
  - extract_config    : extract config keys / code snippets from a thread
  - clarify           : ask a clarifying question back to the user (no-op tool)
"""
from __future__ import annotations

import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict          # JSON Schema object
    fn: Callable


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    output: str               # always a string for prompt injection
    error: Optional[str] = None
    success: bool = True


# ══════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════

_REGISTRY: dict[str, ToolDefinition] = {}


def tool(name: str, description: str, parameters: dict):
    """Decorator to register a function as a tool."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = ToolDefinition(
            name=name, description=description, parameters=parameters, fn=fn
        )
        return fn
    return decorator


def get_tool_definitions() -> list[ToolDefinition]:
    return list(_REGISTRY.values())


def tool_schema_for_prompt() -> str:
    """Return a text description of all tools for injection into the system prompt."""
    lines = ["You have access to the following tools. Call them using XML tags:\n",
             "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}, \"id\": \"1\"}</tool_call>\n",
             "After receiving tool results, use them to formulate your final answer.\n",
             "Available tools:\n"]
    for td in _REGISTRY.values():
        props = td.parameters.get("properties", {})
        params_desc = ", ".join(
            f"{k} ({v.get('type','any')}): {v.get('description','')}"
            for k, v in props.items()
        )
        lines.append(f"• {td.name}({params_desc})\n  {td.description}\n")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Tool implementations
# ══════════════════════════════════════════════════════════════════════

@tool(
    name="search_history",
    description="Semantic search over the indexed Slack Q&A knowledge base. "
                "Use this to find past discussions relevant to the current question. "
                "Always call this first before answering.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query — paraphrase the user's question for best results"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (1-10, default 5)"
            },
        },
        "required": ["query"],
    }
)
def search_history(query: str, top_k: int = 5, _retriever=None) -> str:
    if _retriever is None:
        return "Error: retriever not available"
    try:
        results = _retriever.retrieve(query, top_k=min(int(top_k), 10))
        if not results:
            return "No relevant past discussions found for this query."
        parts = []
        for i, ctx in enumerate(results, 1):
            parts.append(
                f"[Result {i}] sim={ctx.similarity:.2f} rxn={ctx.qa.reaction_score}\n"
                f"Q: {ctx.qa.question[:300]}\n"
                f"A: {ctx.qa.answer[:600]}\n"
                f"Source: {ctx.qa.slack_url or 'csv_import'}"
            )
        return "\n\n".join(parts)
    except Exception as exc:
        return f"Search error: {exc}"


@tool(
    name="search_bm25",
    description="Keyword (BM25) search over the indexed Slack Q&A knowledge base. "
                "Use when the query contains specific technical terms, service names, "
                "or error codes that may not be captured by semantic search.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword query — include exact service names, error codes, config keys"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (1-10, default 5)"
            },
        },
        "required": ["query"],
    }
)
def search_bm25(query: str, top_k: int = 5, _bm25_retriever=None) -> str:
    if _bm25_retriever is None:
        return "Error: BM25 retriever not available"
    try:
        results = _bm25_retriever.retrieve(query, top_k=min(int(top_k), 10))
        if not results:
            return "No keyword matches found."
        parts = []
        for i, ctx in enumerate(results, 1):
            parts.append(
                f"[Result {i}] bm25_score={ctx.similarity:.2f} rxn={ctx.qa.reaction_score}\n"
                f"Q: {ctx.qa.question[:300]}\n"
                f"A: {ctx.qa.answer[:600]}"
            )
        return "\n\n".join(parts)
    except Exception as exc:
        return f"BM25 search error: {exc}"


@tool(
    name="get_thread",
    description="Fetch the full text of a specific Q&A thread by its ID. "
                "Use this after search_history when you need the complete answer "
                "that was truncated in search results.",
    parameters={
        "type": "object",
        "properties": {
            "thread_id": {
                "type": "string",
                "description": "The thread ID (from search result metadata)"
            },
        },
        "required": ["thread_id"],
    }
)
def get_thread(thread_id: str, _database=None) -> str:
    if _database is None:
        return "Error: database not available"
    try:
        qa = _database.get_qa_by_id(thread_id)
        if qa is None:
            return f"Thread '{thread_id}' not found in database."
        return (
            f"Thread ID: {qa.id}\n"
            f"Question: {qa.question}\n\n"
            f"Answer: {qa.answer}\n\n"
            f"Respondents: {', '.join(qa.respondents)}\n"
            f"Reaction score: {qa.reaction_score}\n"
            f"Source: {qa.slack_url or 'csv_import'}"
        )
    except Exception as exc:
        return f"Error fetching thread: {exc}"


@tool(
    name="summarise_thread",
    description="Summarise a long answer into 3-5 concise bullet points. "
                "Use when a retrieved answer is very long and you need to extract "
                "the key actionable steps for the user.",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The thread answer text to summarise"
            },
        },
        "required": ["text"],
    }
)
def summarise_thread(text: str, _llm=None) -> str:
    """Uses the LLM to summarise — falls back to truncation if unavailable."""
    if _llm is None or len(text) < 500:
        # Simple fallback: return first 400 chars with note
        return textwrap.shorten(text, width=400, placeholder="… [truncated]")
    try:
        result = _llm.generate(
            question="Summarise the following answer into 3-5 bullet points. "
                     "Focus on actionable steps and key facts only:\n\n" + text[:3000],
            retrieved=[],
        )
        return result
    except Exception as exc:
        return f"Summarise error: {exc}. Raw: {text[:300]}"


@tool(
    name="extract_config",
    description="Extract configuration keys, code snippets, or commands from a thread answer. "
                "Use when the user is asking about specific configuration and you want to "
                "surface only the relevant config values.",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The answer text to extract config from"
            },
            "keyword": {
                "type": "string",
                "description": "Optional keyword to filter results (e.g. 'max.poll', 'MIGRATION')"
            },
        },
        "required": ["text"],
    }
)
def extract_config(text: str, keyword: str = "", _llm=None) -> str:
    """Extract code blocks and config-like patterns from answer text."""
    # Extract code blocks (``` ... ```)
    code_blocks = re.findall(r"```[\w]*\n?(.*?)```", text, re.DOTALL)
    # Extract inline code (`...`)
    inline_code = re.findall(r"`([^`\n]{3,80})`", text)
    # Extract KEY=VALUE or key: value patterns
    kv_patterns = re.findall(r"[\w\.\-]+=[\w\.\-/\"']+", text)

    results = []
    if code_blocks:
        results.append("Code blocks:\n" + "\n---\n".join(b.strip() for b in code_blocks[:3]))
    if inline_code:
        filtered = [c for c in inline_code if not keyword or keyword.lower() in c.lower()]
        if filtered:
            results.append("Inline code: " + " | ".join(filtered[:10]))
    if kv_patterns:
        filtered = [k for k in kv_patterns if not keyword or keyword.lower() in k.lower()]
        if filtered:
            results.append("Config values: " + ", ".join(filtered[:10]))

    return "\n\n".join(results) if results else "No configuration patterns found in text."


@tool(
    name="clarify",
    description="Ask the user a clarifying question when the question is ambiguous "
                "and cannot be answered without more information. "
                "The question will be posted to Slack as a reply.",
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The clarifying question to ask the user"
            },
        },
        "required": ["question"],
    }
)
def clarify(question: str, **kwargs) -> str:
    """Returns a special marker that the bot intercepts to post a clarifying question."""
    return f"__CLARIFY__:{question}"


# ══════════════════════════════════════════════════════════════════════
# Tool executor
# ══════════════════════════════════════════════════════════════════════

class ToolExecutor:
    """
    Parses tool calls from LLM output and dispatches them to registered tools.

    LLM tool call format (XML-like, model-agnostic):
      <tool_call>{"name": "search_history", "arguments": {"query": "..."}, "id": "1"}</tool_call>

    After execution, results are formatted for re-injection into the conversation.
    """

    # Regex to find tool_call tags in LLM output
    TOOL_CALL_RE = re.compile(
        r"<tool_call>(.*?)</tool_call>", re.DOTALL
    )

    def __init__(self, retriever=None, bm25_retriever=None, database=None, llm=None):
        self._deps = {
            "_retriever": retriever,
            "_bm25_retriever": bm25_retriever,
            "_database": database,
            "_llm": llm,
        }

    def parse_tool_calls(self, llm_output: str) -> list[ToolCall]:
        """Extract all tool calls from LLM output text."""
        calls = []
        for match in self.TOOL_CALL_RE.finditer(llm_output):
            raw = match.group(1).strip()
            try:
                data = json.loads(raw)
                calls.append(ToolCall(
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                    call_id=str(data.get("id", len(calls) + 1)),
                ))
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse tool call JSON: %s — %s", raw[:100], exc)
        return calls

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call and return its result."""
        td = _REGISTRY.get(call.name)
        if td is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                output="",
                error=f"Unknown tool: '{call.name}'. Available: {list(_REGISTRY.keys())}",
                success=False,
            )
        try:
            import inspect
            sig = inspect.signature(td.fn)
            # Only pass deps that the function actually accepts
            accepted = {
                k: v for k, v in self._deps.items()
                if k in sig.parameters
            }
            kwargs = {**call.arguments, **accepted}
            output = td.fn(**kwargs)
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                output=str(output),
                success=True,
            )
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", call.name)
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                output="",
                error=str(exc),
                success=False,
            )

    def execute_all(self, llm_output: str) -> list[ToolResult]:
        """Parse and execute all tool calls in an LLM response."""
        calls = self.parse_tool_calls(llm_output)
        return [self.execute(call) for call in calls]

    @staticmethod
    def format_results_for_prompt(results: list[ToolResult]) -> str:
        """Format tool results for re-injection into the conversation."""
        parts = ["--- Tool Results ---"]
        for r in results:
            if r.success:
                parts.append(f"[Tool: {r.tool_name} | id={r.call_id}]\n{r.output}")
            else:
                parts.append(f"[Tool: {r.tool_name} | id={r.call_id} | ERROR]\n{r.error}")
        parts.append("--- End Tool Results ---")
        return "\n\n".join(parts)

    def has_tool_calls(self, text: str) -> bool:
        return bool(self.TOOL_CALL_RE.search(text))
