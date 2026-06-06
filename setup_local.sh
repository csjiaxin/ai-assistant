#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# Slack AI Support Agent — Local Setup Script
# Uses `uv` for fast, reliable Python dependency management.
# Sets up everything needed to run with FREE, LOCAL models only.
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Slack AI Support Agent — Local Setup                 ║"
echo "║        Free models: sentence-transformers + Ollama          ║"
echo "║        Package manager: uv (fast, reproducible)            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Install uv ────────────────────────────────────────────────────
echo "▶ [1/5] Checking uv (Python package manager)…"
if command -v uv &>/dev/null; then
    echo "  ✓ uv $(uv --version) already installed"
else
    echo "  ℹ uv not found. Installing via official installer…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo "  ✗ uv install failed. Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    echo "  ✓ uv $(uv --version) installed"
fi

# ── 2. Create venv + install dependencies ────────────────────────────
echo ""
echo "▶ [2/5] Installing Python dependencies with uv…"

# uv will auto-detect Python and create .venv
uv venv --python 3.11 2>/dev/null || uv venv  # fallback to any available Python >= 3.9
echo "  ✓ Virtual environment ready at .venv/"

# Sync all dependencies from requirements.txt
# uv is 10-100x faster than pip for this
uv pip install -r requirements.txt
echo "  ✓ All dependencies installed"

# ── 3. Copy .env if not present ──────────────────────────────────────
echo ""
echo "▶ [3/5] Checking .env configuration…"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  ✓ Created .env from .env.example"
    echo "  ⚠  Please edit .env and add your Slack credentials:"
    echo "     SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_SIGNING_SECRET, MONITORED_CHANNELS"
else
    echo "  ✓ .env already exists"
fi

# ── 4. Check / Install Ollama ─────────────────────────────────────────
echo ""
echo "▶ [4/5] Checking Ollama (local LLM server)…"
if command -v ollama &>/dev/null; then
    echo "  ✓ Ollama is installed ($(ollama --version 2>/dev/null || echo 'version unknown'))"
else
    echo "  ℹ Ollama not found. Installing…"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &>/dev/null; then
            brew install ollama
        else
            echo "  → Install Homebrew first: https://brew.sh"
            echo "    Or download Ollama directly: https://ollama.ai/download"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "  → Please install Ollama manually: https://ollama.ai/download"
        echo "    Then re-run this script."
        exit 1
    fi
fi

# Start Ollama server in background if not already running
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "  ℹ Starting Ollama server in background…"
    ollama serve &>/dev/null &
    sleep 3
fi

# Pull the LLM model (downloads once, cached locally)
echo "  ▶ Pulling model: $OLLAMA_MODEL (downloads ~2-5GB the first time)…"
ollama pull "$OLLAMA_MODEL"
echo "  ✓ Model $OLLAMA_MODEL ready"

# ── 5. Pre-download embedding model ──────────────────────────────────
echo ""
echo "▶ [5/5] Pre-downloading sentence-transformers embedding model…"
uv run python -c "
from sentence_transformers import SentenceTransformer
import os
model_name = os.getenv('EMBEDDING_MODEL', 'minilm')
aliases = {
    'minilm':    'sentence-transformers/all-MiniLM-L6-v2',
    'bge-small': 'BAAI/bge-small-en-v1.5',
    'bge-base':  'BAAI/bge-base-en-v1.5',
    'mpnet':     'sentence-transformers/all-mpnet-base-v2',
    'e5-small':  'intfloat/e5-small-v2',
}
model_name = aliases.get(model_name, model_name)
print(f'  Downloading: {model_name}')
m = SentenceTransformer(model_name)
dim = m.get_sentence_embedding_dimension()
print(f'  ✓ Embedding model ready (dim={dim})')
"

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Setup complete! Everything runs locally — no API costs. ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Common commands (use 'uv run' to run inside the venv):"
echo ""
echo "  Index your Slack CSV export:"
echo "    uv run python main.py ingest-csv data/202512_slack_help_service_proxy.csv"
echo ""
echo "  Start the bot (needs Slack credentials in .env):"
echo "    uv run python main.py serve"
echo ""
echo "  Check vector store stats:"
echo "    uv run python main.py stats"
echo ""
echo "  Run tests:"
echo "    uv run pytest tests/ -v"
echo ""
echo "  Or activate the venv manually first:"
echo "    source .venv/bin/activate"
echo "    python main.py ingest-csv data/202512_slack_help_service_proxy.csv"
echo ""
