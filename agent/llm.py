"""
LLM answer generation — defaults to Ollama (free, local). OpenAI optional.

Supported local models via Ollama (all free):
  llama3, llama3.1, llama3.2, mistral, mistral-nemo,
  gemma2, phi3, phi3.5, qwen2.5, deepseek-r1, codellama

Install Ollama: https://ollama.ai
Then pull a model: ollama pull llama3.2
"""
from __future__ import annotations

import logging
from typing import Iterator

from agent.config import config
from agent.models import AgentResponse, RetrievedContext

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable AI support assistant embedded in a Slack workspace.
Your role is to help engineers and team members by answering questions based on the team's \
historical Slack discussions.

Guidelines:
- Ground your answer in the provided context (historical Q&A). Cite sources using [Source N].
- If the context answers the question well, synthesise a clear, complete response.
- If the context is only partially relevant, combine it with your own knowledge and say so.
- If the context is not relevant at all, answer from your general knowledge and clearly state \
  that no historical discussion was found.
- Never hallucinate specific facts (PR numbers, configs, names) not present in context.
- Keep answers concise but complete. Use bullet points or code blocks where helpful.
- Do not repeat the question back to the user.
- Respond in plain text suitable for Slack (mrkdwn). Use *bold* for emphasis, \
  `backticks` for code/commands, and ``` for multi-line code blocks."""


def _build_context_block(retrieved: list[RetrievedContext], max_chars: int) -> str:
    if not retrieved:
        return "No relevant historical discussions found."
    parts = []
    total = 0
    for i, ctx in enumerate(retrieved, 1):
        block = ctx.format_for_prompt(i)
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def _build_user_message(question: str, context_block: str) -> str:
    return (
        f"--- Historical Context ---\n{context_block}\n--- End Context ---\n\n"
        f"New Question: {question}\n\n"
        "Please answer using the context above where relevant. Cite [Source N] when applicable."
    )


# ══════════════════════════════════════════════════════════════════════
# Abstract base
# ══════════════════════════════════════════════════════════════════════

class LLMBase:
    def generate(self, question: str, retrieved: list[RetrievedContext]) -> str:
        raise NotImplementedError

    def generate_stream(self, question: str, retrieved: list[RetrievedContext]) -> Iterator[str]:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════
# OpenAI backend
# ══════════════════════════════════════════════════════════════════════

class OpenAILLM(LLMBase):
    def __init__(self):
        from openai import OpenAI  # type: ignore
        self._client = OpenAI(api_key=config.openai_api_key)
        logger.info("LLM: OpenAI backend (%s)", config.llm_model)

    def generate(self, question: str, retrieved: list[RetrievedContext]) -> str:
        context_block = _build_context_block(retrieved, config.max_context_chars)
        user_msg = _build_user_message(question, context_block)
        response = self._client.chat.completions.create(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, question: str, retrieved: list[RetrievedContext]) -> Iterator[str]:
        context_block = _build_context_block(retrieved, config.max_context_chars)
        user_msg = _build_user_message(question, context_block)
        stream = self._client.chat.completions.create(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ══════════════════════════════════════════════════════════════════════
# Ollama backend (local Llama 3 / Mistral / etc.)
# ══════════════════════════════════════════════════════════════════════

class OllamaLLM(LLMBase):
    """
    Local LLM via Ollama — 100% free, runs on your machine.

    Setup:
      1. Install Ollama: https://ollama.ai  (macOS/Linux/Windows)
      2. Pull a model:   ollama pull llama3.2
      3. Set env:        LLM_MODEL=llama3
    """

    def __init__(self, model: str = "llama3.2"):
        self._model = model
        self._base_url = config.ollama_base_url
        # Use requests directly so we don't need the ollama package
        try:
            import requests  # type: ignore
            self._requests = requests
        except ImportError:
            raise RuntimeError("Install requests: pip install requests")
        logger.info("LLM: Ollama backend — model=%s url=%s", model, self._base_url)
        self._verify_model_available()

    def _verify_model_available(self) -> None:
        """Check Ollama is running and the model is pulled."""
        try:
            resp = self._requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            tags = resp.json()
            model_names = [m["name"].split(":")[0] for m in tags.get("models", [])]
            model_base = self._model.split(":")[0]
            if model_base not in model_names:
                logger.warning(
                    "Model '%s' not found in Ollama. Run: ollama pull %s\n"
                    "Available: %s",
                    self._model, self._model, model_names or "none"
                )
        except Exception as exc:
            logger.warning(
                "Cannot reach Ollama at %s (%s). "
                "Make sure Ollama is running: https://ollama.ai",
                self._base_url, exc
            )

    def _chat(self, messages: list[dict], stream: bool = False):
        """Call Ollama /api/chat endpoint."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": config.llm_temperature,
                "num_predict": config.llm_max_tokens,
            },
        }
        resp = self._requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            stream=stream,
            timeout=120,
        )
        resp.raise_for_status()
        return resp

    def generate(self, question: str, retrieved: list[RetrievedContext]) -> str:
        import json
        context_block = _build_context_block(retrieved, config.max_context_chars)
        user_msg = _build_user_message(question, context_block)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        resp = self._chat(messages, stream=False)
        data = resp.json()
        return data["message"]["content"].strip()

    def generate_stream(self, question: str, retrieved: list[RetrievedContext]) -> Iterator[str]:
        import json
        context_block = _build_context_block(retrieved, config.max_context_chars)
        user_msg = _build_user_message(question, context_block)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        resp = self._chat(messages, stream=True)
        for line in resp.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue


# ══════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════

# Free local Ollama model shorthand → actual model names
OLLAMA_MODEL_ALIASES = {
    "llama3":        "llama3.2",        # latest Llama 3 (3B, fast)
    "llama3-big":    "llama3.1:8b",     # 8B, better quality
    "mistral":       "mistral",         # 7B, excellent for Q&A
    "mistral-nemo":  "mistral-nemo",    # 12B, strong reasoning
    "gemma2":        "gemma2:2b",       # 2B, very fast
    "gemma2-big":    "gemma2:9b",       # 9B, high quality
    "phi3":          "phi3.5",          # 3.8B, great for code
    "qwen":          "qwen2.5:3b",      # 3B, multilingual
    "deepseek":      "deepseek-r1:7b",  # 7B, strong reasoning
}


def create_llm() -> LLMBase:
    """
    Auto-selects LLM backend based on LLM_MODEL env var.

    Local (free, default):
      - "llama3"        → Ollama llama3.2  (recommended)
      - "mistral"       → Ollama mistral
      - "gemma2"        → Ollama gemma2:2b
      - "phi3"          → Ollama phi3.5
      - "ollama:<name>" → Ollama with exact model name

    Cloud (requires API key):
      - "gpt-4o"        → OpenAI GPT-4o
      - "gpt-4o-mini"   → OpenAI GPT-4o-mini (cheaper)
    """
    model = config.llm_model.strip()

    # Explicit ollama: prefix
    if model.lower().startswith("ollama:"):
        ollama_model = model.split(":", 1)[1].strip() or "llama3.2"
        return OllamaLLM(model=ollama_model)

    # Known Ollama aliases (free local)
    if model.lower() in OLLAMA_MODEL_ALIASES:
        return OllamaLLM(model=OLLAMA_MODEL_ALIASES[model.lower()])

    # OpenAI models (require API key)
    if model.lower().startswith("gpt-") or model.lower().startswith("o1"):
        if not config.openai_api_key:
            raise RuntimeError(
                f"LLM_MODEL={model!r} requires OPENAI_API_KEY to be set. "
                "To use a free local model instead, set LLM_MODEL=llama3 and install Ollama."
            )
        return OpenAILLM()

    # Default: try Ollama with the model name as-is
    return OllamaLLM(model=model)
