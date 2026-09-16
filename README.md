# GroundedRAG

> **Status:** scaffolding only — build begins Oct 2. This README gets filled in for real on Oct 14 (see project plan Section 7).

Agentic RAG system over SEC 10-K/10-Q filings for 8 companies: hybrid retrieval
(BM25 + dense + hand-written RRF fusion + a LoRA-fine-tuned reranker), a
hand-rolled ReAct-style agent that routes between retrieval / SQL / calculator,
prompt-injection guardrails on retrieved context, a custom eval harness
(Recall@k, MRR, NLI-based faithfulness with bootstrapped CIs), and deployment
via FastAPI + Docker.

## Planned layout

```
src/
  ingestion/    # SEC EDGAR pull, text/XBRL split, chunking
  retrieval/    # BM25, dense embeddings, RRF fusion, reranker (+ LoRA)
  agent/        # ReAct-style routing loop, tools (retrieve/SQL/calculator)
  eval/         # Recall@k/MRR, NLI faithfulness, bootstrap CIs
  guardrails/   # injection filtering on retrieved chunks
  api/          # FastAPI app, /metrics, logging
data/
  raw/          # gitignored — regenerate via ingestion scripts
  processed/    # gitignored — regenerate via ingestion scripts
  eval_set/     # tracked — hand-labeled 40-60 question eval set
notebooks/      # scratch/exploration only, nothing load-bearing lives here
docs/           # architecture notes, ablation results, threat model
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
