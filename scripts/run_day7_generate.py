"""
Phase B, Day 7 -- stage 1 of 2: generate answers with a Groq-hosted LLM over retrieved chunks
(default openai/gpt-oss-120b: the plan's Llama 3.3 70B was not available to this project's Groq key).

Runs the 20 single-hop questions (answerable from text) and the 10 out-of-scope questions
(correct behaviour = refuse) through retrieve -> generate. The 20 multi-hop-numeric questions
are deliberately NOT run: they have no gold chunk by design and need the SQL/calculator tools
(Day 8) -- generating prose for them would just measure the "confident wrong number" failure
the agent exists to prevent.

Needs: local Qdrant running with the 11,245 points (docker), huggingface.co reachable (query
embedding model + reranker), a GROQ_API_KEY (env var or .env at the repo root). Every Groq
response is cached under data/cache/groq/, so a re-run or a resume after a rate-limit stall
costs nothing for questions already answered.

Defaults come from Day 5's ablation: reranked retrieval at k=5 (R@5 0.20 / MRR 0.133) beats
RRF hybrid at k=5 (R@5 0.10 / MRR 0.080). Hybrid wins at k=10 (R@10 0.35 vs 0.25) -- use
`--retriever hybrid --k 10` to feed the generator a wider net instead.

--context-mode retrieved   what the real system sees (default)
--context-mode oracle      gold chunk(s) + retrieved filler: an optimistic UPPER BOUND on the
                           generator's faithfulness, isolating it from retrieval failures
                           (single-hop only; out-of-scope questions have no gold).

Usage (from the repo root, venv active) -- see the Day 7 summary in docs/ for the full sequence:
    python -m scripts.run_day7_generate --retriever reranked --k 5 --context-mode retrieved --limit 2
    python -m scripts.run_day7_generate --retriever reranked --k 5 --context-mode retrieved
    python -m scripts.run_day7_generate --retriever reranked --k 5 --context-mode oracle
Writes data/eval_set/day7_answers_<retriever>_k<k>_<mode>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.keyword_baseline import load_chunks
from src.generation.generator import RAGGenerator, build_oracle_context
from src.generation.groq_client import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, GroqClient, GroqError, ResponseCache

try:  # python-dotenv is already in requirements.txt (Day 1)
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


def preflight_qdrant(expected_points, host="localhost", port=6333):
    """Fail fast, in plain words, if Qdrant is down or holds the wrong amount of data. Day 5's
    point-id collision left the collection at ~19% of the real data while every upload reported
    success -- so the point count is checked here too, not just reachability."""
    from qdrant_client import QdrantClient
    from src.eval.qdrant_setup import COLLECTION_NAME
    try:
        count = QdrantClient(host=host, port=port).count(collection_name=COLLECTION_NAME, exact=True).count
    except Exception as exc:  # connection refused, collection missing, etc.
        short = str(exc).splitlines()[0][:160]
        sys.exit(
            f"\nCannot use Qdrant at {host}:{port} ({short}).\n"
            f"  1. Is Docker Desktop running?   2. Is the Qdrant container running?  (docker ps -a)\n"
            f"  3. If the container is gone, the collection '{COLLECTION_NAME}' must be re-uploaded "
            f"(commands: CHANGES.md, 'Day 7 follow-up').\n")
    if count != expected_points:
        sys.exit(f"\nQdrant collection '{COLLECTION_NAME}' holds {count} points but the local chunk files "
                 f"have {expected_points}. Retrieval would silently run on partial data (the Day 5 bug). "
                 f"Re-upload all 8 companies before continuing.\n")
    print(f"Qdrant OK: {count} points in '{COLLECTION_NAME}'")


def build_retriever(kind, all_chunks, chunks_by_id):
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    bm25 = BM25Retriever(all_chunks)
    hybrid = HybridRetriever(bm25, DenseRetriever(), k_rrf=60, candidate_pool_size=100)
    if kind == "hybrid":
        return hybrid
    from src.retrieval.reranker import RerankerRetriever
    return RerankerRetriever(hybrid, chunks_by_id, candidate_pool_size=50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", choices=["reranked", "hybrid"], default="reranked")
    ap.add_argument("--k", type=int, default=5, help="chunks shown to the generator")
    ap.add_argument("--context-mode", choices=["retrieved", "oracle"], default="retrieved")
    ap.add_argument("--prompt-version", choices=["v1", "v2"], default="v1",
                    help="v1 = frozen Day 7 baseline prompt; v2 = exact-date citation + each fact stated once")
    ap.add_argument("--types", default="single_hop,out_of_scope",
                    help="comma-separated question types to run. Add multi_hop_numeric for the Day 8 RAG-only "
                         "baseline on the 20 computed-answer questions (those have no gold chunks, so "
                         "gold_in_context is None and the faithfulness script does not apply to them)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="completion cap; reasoning models (gpt-oss) need headroom beyond the visible answer")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions of each type (smoke test)")
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "data" / "cache" / "groq"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "data" / "eval_set" / f"day7_answers_{args.retriever}_k{args.k}_{args.context_mode}"
        f"{'' if args.prompt_version == 'v1' else '_p' + args.prompt_version}"
        f"{'' if args.types == 'single_hop,out_of_scope' else '_' + args.types.replace(',', '+')}.json")

    t0 = time.time()
    all_chunks = []
    for f in sorted((REPO_ROOT / "data" / "processed" / "chunks").glob("*.jsonl")):
        all_chunks.extend(load_chunks(str(f)))
    chunks_by_id = {c["chunk_id"]: c for c in all_chunks}
    print(f"Loaded {len(all_chunks)} chunks in {time.time() - t0:.1f}s")

    questions = json.loads((REPO_ROOT / "data" / "eval_set" / "eval_questions.json").read_text(encoding="utf-8"))
    selected = []
    wanted_types = [t.strip() for t in args.types.split(",") if t.strip()]
    known_types = {q["type"] for q in questions}
    bad = [t for t in wanted_types if t not in known_types]
    if bad:
        sys.exit(f"unknown question type(s) {bad}; the eval set has {sorted(known_types)}")
    for qtype in wanted_types:
        of_type = [q for q in questions if q["type"] == qtype]
        selected.extend(of_type[: args.limit] if args.limit else of_type)

    preflight_qdrant(len(all_chunks))

    t0 = time.time()
    retriever = build_retriever(args.retriever, all_chunks, chunks_by_id)
    print(f"Built {args.retriever} retriever in {time.time() - t0:.1f}s")

    llm = GroqClient(cache=ResponseCache(args.cache_dir))
    generator = RAGGenerator(retriever, chunks_by_id, llm, k=args.k, model=args.model, max_tokens=args.max_tokens,
                              prompt_version=args.prompt_version)

    results, errors = [], []
    for i, q in enumerate(selected, start=1):
        forced = None
        tk = q.get("ticker")
        tk = None if (tk and "," in tk) else tk   # cross-company question: no single-ticker filter (both modes)
        if args.context_mode == "oracle" and q["type"] == "single_hop":
            retrieved = retriever.rank(q["question"], ticker=tk, k=args.k)
            forced = build_oracle_context(q["gold_chunk_ids"], retrieved, args.k)
        try:
            r = generator.answer(q["question"], ticker=tk, qid=q["id"], context_chunk_ids=forced)
        except GroqError as exc:
            if exc.status_code in (400, 401, 403, 404):  # permanent: retrying every question is pointless
                sys.exit(f"\nGroq rejected the request and retrying cannot help:\n  {exc}\n"
                         f"401 = bad/missing key in .env.  404 model_not_found = this account cannot use "
                         f"'{args.model}'.\nRun `python -m scripts.list_groq_models` to see what your key can "
                         f"use, then re-run with --model <one of those ids>.\n")
            print(f"[{i}/{len(selected)}] {q['id']}: FAILED -- {exc}")
            errors.append({"id": q["id"], "error": str(exc)})
            continue
        r["type"] = q["type"]
        r["gold_chunk_ids"] = q.get("gold_chunk_ids") or []
        r["gold_in_context"] = (bool(set(r["gold_chunk_ids"]) & set(r["context_chunk_ids"]))
                                if q["type"] == "single_hop" else None)
        results.append(r)
        tag = "REFUSED" if r["is_refusal"] else "answered"
        src = "cache" if r["cached"] else ("no-context" if r["no_context"] else "groq")
        print(f"[{i}/{len(selected)}] {q['id']}: {tag} ({src}) gold_in_context={r['gold_in_context']}")

    payload = {
        "config": {"retriever": args.retriever, "k": args.k, "context_mode": args.context_mode,
                   "prompt_version": args.prompt_version, "types": args.types, "model": args.model, "n_questions_attempted": len(selected), "limit": args.limit},
        "errors": errors,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    sh = [r for r in results if r["type"] == "single_hop"]
    oos = [r for r in results if r["type"] == "out_of_scope"]
    print("\n" + "=" * 60)
    print(f"Wrote {len(results)} answers ({len(errors)} failed) to {out_path}")
    print(f"single-hop: {len(sh)} answered-or-refused, {sum(r['is_refusal'] for r in sh)} refused, "
          f"gold in context for {sum(bool(r['gold_in_context']) for r in sh)}")
    print(f"out-of-scope: {sum(r['is_refusal'] for r in oos)}/{len(oos)} correctly refused")
    mhn = [r for r in results if r["type"] == "multi_hop_numeric"]
    if mhn:
        print(f"multi-hop numeric (RAG-only baseline): {len(mhn)} asked, {sum(r['is_refusal'] for r in mhn)} refused, "
              f"{sum(not r['is_refusal'] for r in mhn)} answered -- compare each answer with "
              f"data/eval_set/day8_mhn_reference.json")
    if errors:
        print(f"{len(errors)} question(s) failed (rate limit or network after retries) -- just re-run the same command; "
              f"finished questions come from the cache.")
    try:
        shown = out_path.resolve().relative_to(REPO_ROOT)
    except ValueError:  # --out pointed outside the repo
        shown = out_path
    if sh or oos:
        print("Next: python -m scripts.run_day7_faithfulness --answers " + str(shown).replace("\\", "/"))


if __name__ == "__main__":
    main()
