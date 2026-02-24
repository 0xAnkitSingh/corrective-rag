# Corrective RAG Agent

A Corrective Retrieval-Augmented Generation system built with **Strand Agents** and **Amazon Bedrock**.

The agent queries a Bedrock Knowledge Base, evaluates chunk relevance with RAGAS, and falls back to Tavily web search when the retrieved context is insufficient.

![architecture](./images/architecture.png)

| Component       | Description                                          |
| --------------- | ---------------------------------------------------- |
| Agent Framework | Strands SDK with modular tool registration           |
| Retrieval       | `retrieve` (strands-agents-tools native)             |
| Evaluation      | `check_chunks_relevance` — RAGAS context precision   |
| Web Fallback    | `web_search` — Tavily API                            |

## Project Layout

```
├── pyproject.toml                  # UV / PEP 621 project config
├── .env.example                    # Environment variable template
├── src/rag/
│   ├── config.py                   # Pydantic-settings configuration
│   ├── clients.py                  # AWS Bedrock client factories
│   ├── main.py                     # CLI entry point
│   ├── tools/
│   │   ├── relevance.py            # RAGAS chunk relevance evaluation
│   │   └── search.py               # Tavily web search
│   ├── agents/
│   │   ├── prompts.py              # System prompts
│   │   ├── simple.py               # Baseline RAG agent
│   │   └── corrective.py           # Corrective RAG agent
│   └── knowledge_base/
│       └── provisioner.py          # KB creation + S3 upload + ingestion
└── onboarding_files/               # Sample documents for KB ingestion
```

## Prerequisites

- Python 3.10+
- [UV](https://docs.astral.sh/uv/) package manager
- AWS account with Anthropic Claude 3.7 enabled in Amazon Bedrock
- IAM role with permissions for Bedrock Knowledge Base, S3, and OpenSearch Serverless
- [Tavily](https://tavily.com/) API key

## Getting Started

### 1. Install dependencies

```bash
uv sync
```

### 2. Provision the Knowledge Base

Place your onboarding documents in `./onboarding_files/`, then run:

```bash
uv run rag-provision-kb --documents-dir ./onboarding_files
```

This creates the S3 bucket, uploads documents, builds the Bedrock Knowledge Base, and writes `KNOWLEDGE_BASE_ID` into `.env`.

### 3. Configure environment

Copy the example and fill in your values:

```bash
cp .env.example .env
```

The provisioner already writes `KNOWLEDGE_BASE_ID` and `AWS_REGION`. Add your `TAVILY_API_KEY`.

### 4. Run the agent

**Interactive mode (default — Bedrock):**

```bash
uv run rag
```

**Single query:**

```bash
uv run rag -q "Who is the medical insurance provider?"
```

**Simple (non-corrective) agent:**

```bash
uv run rag --simple -q "What are the onboarding steps?"
```

**Switch LLM provider from the CLI:**

```bash
# Anthropic direct API
uv run rag --provider anthropic --model claude-sonnet-4-20250514 -q "What benefits do I get?"

# OpenAI / compatible endpoint
uv run rag --provider openai --model gpt-4o -q "Summarise the onboarding process"

# Ollama (local)
uv run rag --provider ollama --model llama3 -q "What is the PTO policy?"
```

**Debug logging:**

```bash
uv run rag -v -q "What is Allstate's phone number?"
```

## How It Works

1. **Retrieve** — the agent queries the Bedrock Knowledge Base using the strands `retrieve` tool.
2. **Evaluate** — `check_chunks_relevance` scores the retrieved chunks using RAGAS `LLMContextPrecisionWithoutReference`. If the score exceeds the threshold (default 0.5), the context is used directly.
3. **Web Search** — when relevance is below the threshold, `web_search` queries Tavily for supplementary information.
4. **Answer** — the agent synthesises a final answer from all available context.

## Supported LLM Providers

The agent LLM is pluggable. Set `AGENT_PROVIDER` (or `--provider`) to switch:

| Provider    | `AGENT_PROVIDER` | Default `AGENT_MODEL_ID`                              | Credentials needed                |
| ----------- | ---------------- | ----------------------------------------------------- | --------------------------------- |
| **Bedrock** | `bedrock`        | `us.anthropic.claude-3-7-sonnet-20250219-v1:0`        | AWS credentials                   |
| **Anthropic** | `anthropic`    | `claude-sonnet-4-20250514`                             | `ANTHROPIC_API_KEY`               |
| **OpenAI**  | `openai`         | `gpt-4o`                                              | `OPENAI_API_KEY`                  |
| **Ollama**  | `ollama`         | `llama3`                                              | Ollama running locally            |

For OpenAI-compatible endpoints (Azure OpenAI, vLLM, LM Studio, etc.), set `OPENAI_BASE_URL` alongside `OPENAI_API_KEY`.

## Configuration

All settings are managed via environment variables or `.env`:

| Variable              | Default                                           | Description                       |
| --------------------- | ------------------------------------------------- | --------------------------------- |
| `KNOWLEDGE_BASE_ID`   | *(required)*                                      | Bedrock Knowledge Base ID         |
| `AWS_REGION`          | `us-east-1`                                       | AWS region                        |
| `TAVILY_API_KEY`      | *(required for corrective agent)*                 | Tavily web search API key         |
| `MIN_SCORE`           | `0.2`                                             | Minimum retrieval score           |
| `AGENT_PROVIDER`      | `bedrock`                                         | LLM provider for the agent        |
| `AGENT_MODEL_ID`      | *(provider default)*                              | Model ID (provider-specific)      |
| `ANTHROPIC_API_KEY`   | —                                                 | Anthropic API key                 |
| `OPENAI_API_KEY`      | —                                                 | OpenAI API key                    |
| `OPENAI_BASE_URL`     | —                                                 | Custom OpenAI-compatible endpoint |
| `OLLAMA_HOST`         | `http://localhost:11434`                          | Ollama server address             |
| `EVAL_MODEL_ID`       | `us.anthropic.claude-3-7-sonnet-20250219-v1:0`    | Model for RAGAS evaluation        |
| `EMBEDDING_MODEL_ID`  | `amazon.titan-embed-text-v2:0`                    | Embedding model                   |
| `RELEVANCE_THRESHOLD` | `0.5`                                             | Chunk relevance cutoff            |
