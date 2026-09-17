"""
Run this LOCALLY, against your real data/processed/ chunks, to fill in gold_chunk_ids
for the "chunk"-type questions in data/eval_set/eval_questions.json.

This does not run in a sandbox without your corpus — it expects your actual
Day 1 pipeline output to exist at data/processed/<TICKER>/chunks.jsonl (or wherever
your chunker writes to; adjust CHUNKS_PATH_TEMPLATE below to match your real layout).

Workflow per question:
  1. Shows the question, target company, and target section.
  2. Runs a simple keyword search over that company's chunks (reusing keyword_baseline.py's
     scorer) and shows the top 8 candidates with chunk_id + a text preview.
  3. You read the candidates, type the chunk_id(s) that actually answer the question
     (comma-separated; empty = skip / come back later), and it's written into the JSON.
  4. Ctrl+C at any point is safe — progress is saved after every question.

This keeps hand-labeling fast (SEC filing prose is verbose; skimming 8 candidates beats
reading a full 10-K section) while keeping a human in the loop for the actual judgment
call, which is the point of hand-labeling an eval set rather than generating it synthetically.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = REPO_ROOT / "data" / "eval_set" / "eval_questions.json"

# Matches src/ingestion/pipeline.py's real output path: data/processed/chunks/{ticker}.jsonl
CHUNKS_PATH_TEMPLATE = str(REPO_ROOT / "data" / "processed" / "chunks" / "{ticker}.jsonl")


def load_chunks(ticker: str):
    """Expects one JSON object per line, matching pipeline.py's real schema:
    {"chunk_id": ..., "ticker": ..., "item": ..., "text": ..., ...}"""
    path = Path(CHUNKS_PATH_TEMPLATE.format(ticker=ticker))
    if not path.exists():
        print(f"  [!] No chunks file found at {path} — skipping search, label manually if you have another source.")
        return []
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def keyword_score(query: str, text: str) -> int:
    """Trivial overlap count — good enough for surfacing labeling candidates, not for eval scoring."""
    q_terms = set(query.lower().split())
    t_terms = text.lower().split()
    return sum(1 for t in t_terms if t in q_terms)


def top_candidates(query: str, chunks: list, item_key: str = None, k: int = 8):
    """item_key: exact match against pipeline.py's "item" field, e.g. "item_1a" —
    NOT the human-readable target_section string. Falls back to the full pool if
    the filter matches nothing (better to surface irrelevant candidates than none)."""
    pool = chunks
    if item_key:
        filtered = [c for c in chunks if c.get("item") == item_key]
        pool = filtered or chunks
    scored = sorted(pool, key=lambda c: keyword_score(query, c.get("text", "")), reverse=True)
    return scored[:k]


def main():
    questions = json.loads(EVAL_SET_PATH.read_text())
    chunk_questions = [q for q in questions if q["gold_answer_type"] == "chunk"]
    remaining = [q for q in chunk_questions if not q["gold_chunk_ids"]]

    print(f"{len(chunk_questions)} chunk-type questions total, {len(remaining)} unlabeled.\n")

    chunks_cache = {}

    for q in remaining:
        ticker = q["ticker"]
        if ticker not in chunks_cache:
            chunks_cache[ticker] = load_chunks(ticker)
        chunks = chunks_cache[ticker]

        print("=" * 80)
        print(f"[{q['id']}] ({q['ticker']} / {q['target_section']})")
        print(q["question"])
        print("-" * 80)

        candidates = top_candidates(q["question"], chunks, q["target_item_key"])
        if not candidates:
            print("  (no candidates found — label manually or check CHUNKS_PATH_TEMPLATE)")
        for c in candidates:
            preview = c.get("text", "")[:220].replace("\n", " ")
            print(f"  [{c.get('chunk_id')}] {preview}...")

        try:
            raw = input("\nGold chunk_id(s), comma-separated (Enter to skip): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nStopped. Progress saved.")
            break

        if raw:
            q["gold_chunk_ids"] = [c.strip() for c in raw.split(",") if c.strip()]

        EVAL_SET_PATH.write_text(json.dumps(questions, indent=2))

    labeled = sum(1 for q in chunk_questions if q["gold_chunk_ids"])
    print(f"\nDone for now. {labeled}/{len(chunk_questions)} chunk-type questions labeled.")


if __name__ == "__main__":
    main()
