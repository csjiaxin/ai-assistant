"""
Configuration management — reads from environment variables / .env file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── Slack ──────────────────────────────────────────────────────────
    # Required at runtime; default to empty string so tests can import without env vars.
    slack_bot_token: str = field(default_factory=lambda: os.getenv("SLACK_BOT_TOKEN", ""))
    slack_app_token: str = field(default_factory=lambda: os.getenv("SLACK_APP_TOKEN", ""))
    slack_signing_secret: str = field(default_factory=lambda: os.getenv("SLACK_SIGNING_SECRET", ""))

    # Comma-separated list of channel IDs to monitor & ingest
    monitored_channels: list[str] = field(
        default_factory=lambda: [c for c in os.getenv("MONITORED_CHANNELS", "").split(",") if c]
    )

    # ── OpenAI (optional — only needed if LLM_MODEL=gpt-* or EMBEDDING_BACKEND=openai) ──
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # ── Embedding ─────────────────────────────────────────────────────
    # Backend: "local" (default, free) or "openai" (requires API key)
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "local"))
    # Model name or shorthand: "minilm" | "bge-small" | "bge-base" | "mpnet" | full HF model ID
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "minilm"))

    # ── LLM ───────────────────────────────────────────────────────────
    # Free local models: llama3 | mistral | gemma2 | phi3 | qwen | deepseek | ollama:<name>
    # Cloud (requires OPENAI_API_KEY): gpt-4o | gpt-4o-mini
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3"))
    # Ollama server URL (default: local)
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.3")))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1024")))

    # ── Vector Store ──────────────────────────────────────────────────
    vector_store_type: str = field(default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "chroma"))  # chroma | pinecone | qdrant
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
    chroma_collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "slack_qa"))
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    pinecone_index: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX", "slack-qa"))
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "slack_qa"))

    # ── SQLite / Postgres ─────────────────────────────────────────────
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./data/messages.db"))

    # ── Redis ─────────────────────────────────────────────────────────
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600")))

    # ── Retrieval ─────────────────────────────────────────────────────
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    min_similarity: float = field(default_factory=lambda: float(os.getenv("MIN_SIMILARITY", "0.4")))

    # ── Ingestion ─────────────────────────────────────────────────────
    ingestion_lookback_days: int = field(default_factory=lambda: int(os.getenv("INGESTION_LOOKBACK_DAYS", "180")))
    ingestion_interval_minutes: int = field(default_factory=lambda: int(os.getenv("INGESTION_INTERVAL_MINUTES", "15")))
    min_thread_replies: int = field(default_factory=lambda: int(os.getenv("MIN_THREAD_REPLIES", "1")))

    # ── App behaviour ─────────────────────────────────────────────────
    bot_name: str = field(default_factory=lambda: os.getenv("BOT_NAME", "SupportBot"))
    show_sources: bool = field(default_factory=lambda: os.getenv("SHOW_SOURCES", "true").lower() == "true")
    max_context_chars: int = field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "6000")))


# Singleton
config = Config()
