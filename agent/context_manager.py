"""
Context management for the Slack AI Support Agent.

Responsible for assembling the full prompt context that is sent to the LLM
on every turn. Context is built from multiple sources and must fit within
the LLM's context window.

Context sources (in priority order):
  1. System prompt (agent persona, tool definitions)       — always included
  2. Long-term memory (relevant past conversation summaries) — if available
  3. Short-term memory (current thread history)            — if multi-turn
  4. Retrieved Q&A context (top-K from vector store)       — always included
  5. Tool results (from prior tool calls this turn)        — if tools were called
  6. Current user message                                  — always included

Why a dedicated context manager?
  Without explicit context management, prompts silently exceed the model's
  token limit, causing truncation from the END — meaning the user's actual
  question gets cut off. The context manager enforces a budget, prioritises
  content, and gracefully degrades by trimming lower-priority sections first.

Design principles:
  - Token budget is approximated as chars/4 (conservative estimate)
  - Sections are filled in priority order; lower-priority sections are
    trimmed or omitted when the budget is exceeded
  - All truncation is logged so it's observable
  - Context assembly is pure (no side effects) — easy to test and debug
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from agent.config import config
from agent.models import RetrievedContext
from agent.memory import ConversationMemory

logger = logging.getLogger(__name__)

# Approximate chars per token for English technical text
CHARS_PER_TOKEN = 4

# Default model context windows (tokens)
CONTEXT_WINDOWS = {
    "llama3.2":              128_000,
    "llama3.1:8b":           128_000,
    "mistral":                32_768,
    "mistral-nemo":           128_000,
    "gemma2:2b":               8_192,
    "gemma2:9b":               8_192,
    "phi3.5":                128_000,
    "qwen2.5:3b":             32_768,
    "deepseek-r1:7b":         64_000,
    "gpt-4o":                128_000,
    "gpt-4o-mini":           128_000,
    "default":                 8_192,   # conservative fallback
}

# Reserve tokens for the model's response
RESPONSE_RESERVE_TOKENS = 1024


@dataclass
class ContextSection:
    name: str
    content: str
    priority: int       # lower = higher priority (1 = must include)
    max_chars: int = 0  # 0 = no limit within budget

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def token_estimate(self) -> int:
        return self.char_count // CHARS_PER_TOKEN


@dataclass
class BuiltContext:
    """The assembled context ready for prompt construction."""
    system_prompt: str
    conversation_history: list[dict]    # LLM message format
    sections: list[ContextSection]
    total_tokens_estimate: int
    sections_trimmed: list[str]         # names of sections that were trimmed
    sections_omitted: list[str]         # names of sections that were dropped

    def format_user_message(self) -> str:
        """Assemble all context sections into the user message."""
        parts = []
        for section in sorted(self.sections, key=lambda s: s.priority):
            if section.content.strip():
                parts.append(section.content)
        return "\n\n".join(parts)

    @property
    def was_truncated(self) -> bool:
        return bool(self.sections_trimmed or self.sections_omitted)


class ContextManager:
    """
    Assembles and manages prompt context for each LLM call.

    Usage:
        ctx = context_manager.build(
            question="How do I configure Kafka?",
            retrieved=[...],
            conversation=memory.get(thread_ts),
            tool_results=[...],
            ltm_entries=[...],
        )
        messages = [
            {"role": "system", "content": ctx.system_prompt},
            *ctx.conversation_history,
            {"role": "user", "content": ctx.format_user_message()},
        ]
    """

    def __init__(
        self,
        llm_model: str = config.llm_model,
        max_context_chars: int = config.max_context_chars,
        enable_tools: bool = True,
    ):
        self._llm_model = llm_model.lower().split("ollama:")[-1]
        self._max_context_chars = max_context_chars
        self._enable_tools = enable_tools

        # Compute character budget from model context window
        model_tokens = CONTEXT_WINDOWS.get(self._llm_model, CONTEXT_WINDOWS["default"])
        usable_tokens = model_tokens - RESPONSE_RESERVE_TOKENS
        self._token_budget = usable_tokens
        self._char_budget = usable_tokens * CHARS_PER_TOKEN
        logger.info(
            "ContextManager: model=%s window=%d tokens char_budget=%d",
            llm_model, model_tokens, self._char_budget
        )

    def build(
        self,
        question: str,
        retrieved: list[RetrievedContext],
        conversation: Optional[ConversationMemory] = None,
        tool_results: Optional[list] = None,    # list[ToolResult]
        ltm_entries: Optional[list[dict]] = None,
        system_prompt_override: Optional[str] = None,
    ) -> BuiltContext:
        """
        Build the full context for an LLM call.
        Sections are included in priority order until the budget is exhausted.
        """
        system_prompt = system_prompt_override or self._build_system_prompt()
        system_chars = len(system_prompt)
        remaining = self._char_budget - system_chars

        # ── Define all candidate sections ─────────────────────────────
        sections_candidates: list[ContextSection] = []

        # Priority 1: current question (must always fit)
        sections_candidates.append(ContextSection(
            name="question",
            content=f"Current question: {question}",
            priority=1,
        ))

        # Priority 2: retrieved Q&A context (core RAG content)
        if retrieved:
            ctx_text = self._format_retrieved(retrieved)
            sections_candidates.append(ContextSection(
                name="retrieved_context",
                content=ctx_text,
                priority=2,
                max_chars=self._max_context_chars,
            ))

        # Priority 3: tool results (if tools were called this turn)
        if tool_results:
            from agent.tools import ToolExecutor
            tool_text = ToolExecutor.format_results_for_prompt(tool_results)
            sections_candidates.append(ContextSection(
                name="tool_results",
                content=tool_text,
                priority=3,
            ))

        # Priority 4: short-term conversation history
        if conversation and conversation.is_multi_turn():
            hist_text = conversation.format_for_prompt(max_chars=3000)
            if hist_text:
                sections_candidates.append(ContextSection(
                    name="conversation_history",
                    content=hist_text,
                    priority=4,
                    max_chars=3000,
                ))

        # Priority 5: long-term memory (lowest priority — trim first)
        if ltm_entries:
            ltm_text = self._format_ltm(ltm_entries)
            if ltm_text:
                sections_candidates.append(ContextSection(
                    name="long_term_memory",
                    content=ltm_text,
                    priority=5,
                    max_chars=1500,
                ))

        # ── Fit sections into budget ───────────────────────────────────
        included: list[ContextSection] = []
        trimmed: list[str] = []
        omitted: list[str] = []
        used_chars = 0

        for section in sorted(sections_candidates, key=lambda s: s.priority):
            available = remaining - used_chars
            if available <= 0:
                omitted.append(section.name)
                logger.debug("ContextManager: omitted section '%s' (budget exhausted)", section.name)
                continue

            content = section.content
            limit = min(section.max_chars, available) if section.max_chars else available

            if len(content) > limit:
                content = content[:limit] + "\n… [truncated]"
                trimmed.append(section.name)
                logger.debug(
                    "ContextManager: trimmed section '%s' from %d to %d chars",
                    section.name, len(section.content), limit
                )

            included.append(ContextSection(
                name=section.name,
                content=content,
                priority=section.priority,
            ))
            used_chars += len(content)

        total_tokens = (system_chars + used_chars) // CHARS_PER_TOKEN

        # ── Build conversation_history for multi-turn LLM calls ───────
        conv_history = []
        if conversation and conversation.is_multi_turn():
            # Include prior turns as proper message history (not just text)
            # Skip the last turn (that's the current question)
            for turn in conversation.turns[:-1]:
                conv_history.append(turn.to_llm_message())

        if trimmed or omitted:
            logger.info(
                "ContextManager: context budget used=%d/%d tokens. trimmed=%s omitted=%s",
                total_tokens, self._token_budget, trimmed, omitted
            )

        return BuiltContext(
            system_prompt=system_prompt,
            conversation_history=conv_history,
            sections=included,
            total_tokens_estimate=total_tokens,
            sections_trimmed=trimmed,
            sections_omitted=omitted,
        )

    # ── Helpers ────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        from agent.llm import SYSTEM_PROMPT
        from agent.tools import tool_schema_for_prompt
        if self._enable_tools:
            return SYSTEM_PROMPT + "\n\n" + tool_schema_for_prompt()
        return SYSTEM_PROMPT

    @staticmethod
    def _format_retrieved(retrieved: list[RetrievedContext]) -> str:
        parts = ["--- Historical Q&A Context ---"]
        for i, ctx in enumerate(retrieved, 1):
            pct = int(ctx.similarity * 100)
            parts.append(
                f"[Source {i}] relevance={pct}% engagement={ctx.qa.reaction_score}\n"
                f"Q: {ctx.qa.question[:400]}\n"
                f"A: {ctx.qa.answer[:800]}\n"
                f"ref: {ctx.qa.slack_url or ctx.qa.id}"
            )
        parts.append("--- End Context ---")
        return "\n\n".join(parts)

    @staticmethod
    def _format_ltm(entries: list[dict]) -> str:
        if not entries:
            return ""
        parts = ["--- Your conversation history with this user ---"]
        for e in entries[:3]:
            parts.append(f"Topic: {e.get('topic_key', '')}\n{e.get('summary', '')[:300]}")
        parts.append("--- End history ---")
        return "\n\n".join(parts)

    @property
    def char_budget(self) -> int:
        return self._char_budget

    @property
    def token_budget(self) -> int:
        return self._token_budget

    def estimate_tokens(self, text: str) -> int:
        return len(text) // CHARS_PER_TOKEN
