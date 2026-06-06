# 🤖 Slack AI Support Agent

An intelligent RAG (Retrieval-Augmented Generation) agent that **learns from historical Slack thread discussions** and generates accurate, context-aware answers to new channel questions.

---

## How It Works

```
Historical Slack Threads → Embed → Vector Store
New Question → Embed → Retrieve Top-K Matches → LLM → Answer in Thread
```

1. **Ingestion** — Fetches all historical threads from monitored Slack channels, extracts Q&A pairs, embeds them, and stores them in a vector database.
2. **Retrieval** — When a new question arrives, it's embedded and the top-K most semantically similar past discussions are retrieved.
3. **Generation** — The retrieved context + question are sent to an LLM (GPT-4o or local Llama 3) to generate a grounded answer.
4. **Response** — The answer is posted as a threaded reply in Slack, with links to original source discussions.

---

## Project Structure

```
├── design.html              # Visual architecture & design document (open in browser)
├── main.py                  # CLI entrypoint (serve / ingest / reindex / stats)
├── agent/
│   ├── config.py            # Configuration (env vars)
│   ├── models.py            # Data models (QAPair, RetrievedContext, AgentResponse)
│   ├── database.py          # SQLite/PostgreSQL persistence (SQLAlchemy)
│   ├── embedder.py          # Text embedding (OpenAI / sentence-transformers)
│   ├── vector_store.py      # Vector DB abstraction (Chroma / Pinecone / Qdrant)
│   ├── ingestion.py         # Slack history ingestion pipeline
│   ├── retriever.py         # Semantic retrieval + re-ranking + Redis cache
│   ├── llm.py               # LLM generation (OpenAI GPT-4o / Ollama Llama 3)
│   ├── bot.py               # Slack Bolt app (event handlers)
│   ├── scheduler.py         # Background ingestion scheduler (APScheduler)
│   └── metrics.py           # Prometheus metrics
├── tests/
│   ├── test_ingestion.py    # Ingestion & model tests
│   ├── test_retriever.py    # Retriever & re-ranking tests
│   └── test_llm.py          # LLM prompt & response tests
├── monitoring/
│   └── prometheus.yml       # Prometheus scrape config
├── docker-compose.yml       # Full stack (agent + Redis + Prometheus + Grafana)
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Clone & Install

Uses [`uv`](https://docs.astral.sh/uv/) — a fast Python package manager (10-100× faster than pip).

```bash
git clone <repo-url>
cd slack-ai-agent

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# One-command setup: installs deps, Ollama, pulls LLM, downloads embedding model
bash setup_local.sh
```

Or manually with uv:
```bash
uv venv                          # create .venv with auto-detected Python
uv pip install -r requirements.txt   # install all deps (fast!)
```

### 2. Create Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → From Manifest
2. Enable **Socket Mode** and generate an App-Level Token (`xapp-...`)
3. Add Bot Token Scopes: `channels:history`, `channels:read`, `chat:write`, `reactions:read`
4. Install app to workspace, copy **Bot Token** (`xoxb-...`)

### 3. Configure

```bash
cp .env.example .env
# Only Slack credentials are required — all model settings default to free/local
# Edit: SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_SIGNING_SECRET, MONITORED_CHANNELS
```

### 4. Ingest (choose one)

```bash
# From a local Slack CSV export (no Slack API needed — recommended for first run)
uv run python main.py ingest-csv data/your_export.csv

# Or from Slack API directly (requires tokens in .env)
uv run python main.py ingest
```

### 5. Start the Bot

```bash
uv run python main.py serve
```

---

## Docker Deployment

```bash
cp .env.example .env   # fill in your values
docker compose up -d
```

With monitoring:
```bash
docker compose --profile monitoring up -d
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

---

## CLI Commands

All commands work with `uv run python main.py <command>` or `python main.py <command>` if venv is activated.

| Command | Description |
|---|---|
| `serve` | Start bot + scheduler (production) |
| `ingest` | Run incremental ingestion via Slack API |
| `ingest-csv <file>` | Ingest from local CSV export (no API key needed) |
| `reindex` | Force full reindex of all threads |
| `stats` | Print vector store & DB stats |

```bash
# Examples
uv run python main.py ingest-csv data/202512_slack_help_service_proxy.csv
uv run python main.py stats
uv run python main.py serve --no-initial-ingest
```

---

## Configuration Reference

See [`.env.example`](.env.example) for all options. Key settings:

| Variable | Default | Description |
|---|---|---|
| `MONITORED_CHANNELS` | — | Comma-separated Slack channel IDs |
| `LLM_MODEL` | `gpt-4o` | Use `ollama:llama3` for local inference |
| `VECTOR_STORE_TYPE` | `chroma` | `chroma` \| `pinecone` \| `qdrant` |
| `TOP_K` | `5` | Number of historical Q&As to retrieve |
| `MIN_SIMILARITY` | `0.4` | Minimum cosine similarity threshold |
| `INGESTION_INTERVAL_MINUTES` | `15` | How often to sync new Slack threads |

---

## Vector Store Options

| Store | Best For | Setup |
|---|---|---|
| **ChromaDB** (default) | Dev & small teams | Zero infra — embedded, local |
| **Pinecone** | Large scale, cloud | Set `PINECONE_API_KEY` + `PINECONE_INDEX` |
| **Qdrant** | Self-hosted, production | Run Qdrant via Docker |

---

## Running Tests

```bash
uv run pytest tests/ -v

# Or with venv activated:
pytest tests/ -v
```

---

## Architecture

Open [`design.html`](design.html) in a browser for the full interactive architecture diagram.

Key design decisions:
- **RAG over fine-tuning** — No retraining needed; new threads are indexed continuously
- **Reaction-score re-ranking** — Answers with more 👍 reactions rank higher
- **Redis caching** — Identical/similar recent queries skip embedding + retrieval
- **Pluggable backends** — Swap LLM, vector store, or embedding model via env vars
- **Feedback loop** — 👍/👎 reactions on bot answers feed back into the system

---

## License

MIT
