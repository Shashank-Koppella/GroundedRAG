# GroundedRAG

> **Status (Oct 5, 2026): in progress — Phase B.** Built and tested: ingestion, hybrid retrieval with a
> cross-encoder reranker, the eval harness (Recall@k / MRR@10 with question-level bootstrap CIs, NLI
> faithfulness with validity controls), generation, the XBRL facts table, the calculator and the routing
> rules. Not built yet: the agent loop (design done: `docs/day8_routing_design.md`), guardrails, the API,
> the LoRA reranker, any UI. The full README with results lands Oct 14; the running log is `CHANGES.md`.

Agentic RAG system over SEC 10-K/10-Q filings for 8 companies (AAPL, MSFT, GOOGL, AMZN, META, NVDA, ORCL,
AVGO): hybrid retrieval (BM25 + bge-base dense + hand-written RRF fusion + a cross-encoder reranker, LoRA
fine-tune planned), a hand-rolled bounded agent (plan -> guard -> execute -> compose) that routes each
question to filing-text retrieval, the XBRL facts table plus calculator, both, or a refusal, a custom eval
harness, and planned prompt-injection guardrails and FastAPI + Docker deployment.

**LLM:** Groq, `openai/gpt-oss-120b` (Llama 3.3 70B was planned but is not available to the project's API
key). On the structured path the LLM never writes a number: values come from the facts table, arithmetic
from the calculator, and the sentence from a template.

## Layout

```
src/
  ingestion/    # SEC EDGAR pull, section splitting, chunking, XBRL fact extraction
  retrieval/    # BM25, dense (Qdrant), RRF fusion, reranker
  generation/   # Groq client with an on-disk response cache, RAG generator (prompt v1/v2)
  agent/        # facts table (SQLite), calculator, routing rules (agent loop: in progress)
  eval/         # Recall@k/MRR, equivalence-aware relevance, NLI faithfulness + controls, bootstrap CIs
  guardrails/   # (empty — planned)
  api/          # (empty — planned)
scripts/        # every run, diagnostic and build step (python -m scripts.<name>)
data/
  raw/          # gitignored
  processed/    # gitignored — regenerate with python -m src.ingestion.pipeline
  eval_set/     # tracked — 50-question eval set, 48-question blind routing probe, all result files
docs/           # project plan, routing design
tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # fill in GROQ_API_KEY
```

## Results

_(filled in Oct 14 — ablation table, LoRA before/after, faithfulness CI, guardrails probe results)_
