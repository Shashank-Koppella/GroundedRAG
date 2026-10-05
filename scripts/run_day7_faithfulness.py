"""
Phase B, Day 7 -- stage 2 of 2: NLI faithfulness scoring with bootstrapped 95% CIs.

Reads a day7_answers_*.json written by scripts/run_day7_generate.py, scores every non-refusal
answer sentence-by-sentence against the chunks it was generated from using
cross-encoder/nli-deberta-v3-base, and reports question-level (cluster) bootstrap CIs.

Needs only the local chunk files plus the NLI model from huggingface.co -- NO Qdrant, NO Groq.
So thresholds / window sizes can be changed and re-scored freely without touching either.

Run the real-model sanity check ONCE first -- it verifies the model's label order and that
the three basic probes classify correctly before any number is trusted:
    python -m src.eval.faithfulness --sanity-check

Usage (from the repo root, venv active):
    python -m scripts.run_day7_faithfulness --answers data/eval_set/day7_answers_reranked_k5_retrieved.json
    python -m scripts.run_day7_faithfulness --answers data/eval_set/day7_answers_reranked_k5_oracle.json
Writes data/eval_set/day7_faithfulness_<same tag>.json (per-sentence detail for Phase D's failure analysis).
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.bootstrap import cluster_mean_bootstrap
from src.eval.controls import MODES as CONTROL_MODES, control_suffix, header_only, live_answers, pair_contexts
from src.eval.faithfulness import (
    NLI_MODEL_NAME, NLIScorer, aggregate_by, aggregate_faithfulness, recount_at_threshold, score_answer,
)
from src.eval.keyword_baseline import load_chunks

HEADLINES = [
    ("mean_answer_faithfulness", "mean answer faithfulness (HEADLINE, each question weighs 1)"),
    ("sentence_support_rate", "pooled sentence support rate"),
    ("sentence_contradiction_rate", "pooled sentence contradiction rate"),
    ("answer_any_unsupported_rate", "answers with >=1 non-entailed sentence"),
]


def fmt(m):
    flag = "  (n<30: interval is optimistic)" if m["small_n"] else ""
    return f"{m['estimate']:.3f}  95% CI [{m['ci_95'][0]:.3f}, {m['ci_95'][1]:.3f}]{flag}"


def print_block(title, agg):
    print(f"\n{title}  --  {agg['n_answers_scored']} answers, {agg['n_sentences_total']} sentences")
    if agg["n_answers_scored"] == 0:
        print("  (nothing to score)")
        return
    for key, label in HEADLINES:
        print(f"  {label:<58} {fmt(agg[key])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--entail-threshold", type=float, default=0.5)
    ap.add_argument("--contradiction-threshold", type=float, default=0.5)
    ap.add_argument("--window-words", type=int, default=140)
    ap.add_argument("--no-premise-header", action="store_true",
                    help="ablation: score against bare chunk text without the provenance header")
    ap.add_argument("--no-scope-to-named-filing", action="store_true",
                    help="ablation: score every claim against all context windows even if it names a filing date")
    ap.add_argument("--strip-framing", action="store_true",
                    help="experiment (off by default, scored worse on Day 7): strip the 'X filing states that' wrapper before NLI")
    ap.add_argument("--negative-control", nargs="?", const="other_company", default=None, choices=CONTROL_MODES,
                    help="validity check (src/eval/controls.py): 'other_company' (default when the flag is bare; the "
                         "Day 7 control) or 'same_company' (stronger: the header matches, so only content can reject)")
    ap.add_argument("--header-only", action="store_true",
                    help="validity check: real provenance headers, content replaced by a filler sentence; "
                         "measures support produced by the header alone (should be ~0)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    answers_path = Path(args.answers)
    if not answers_path.is_absolute():
        answers_path = REPO_ROOT / answers_path
    data = json.loads(answers_path.read_text(encoding="utf-8"))
    out_name = answers_path.name.replace("day7_answers_", "day7_faithfulness_")
    if args.no_premise_header:
        out_name = out_name.replace(".json", "_nohdr.json")
    out_path = Path(args.out) if args.out else answers_path.with_name(out_name)
    if args.no_scope_to_named_filing and not args.out:
        out_path = out_path.with_name(out_path.name.replace(".json", "_noscope.json"))
    if args.strip_framing and not args.out:
        out_path = out_path.with_name(out_path.name.replace(".json", "_stripped.json"))
    suffix = control_suffix(args.negative_control, args.header_only)
    if suffix and not args.out:
        out_path = out_path.with_name(out_path.name.replace(".json", f"{suffix}.json"))
    include_header = not args.no_premise_header
    strip_framing = args.strip_framing
    scope_named = not args.no_scope_to_named_filing

    chunks_by_id = {}
    for f in sorted((REPO_ROOT / "data" / "processed" / "chunks").glob("*.jsonl")):
        for c in load_chunks(str(f)):
            chunks_by_id[c["chunk_id"]] = c

    scorer = NLIScorer(batch_size=args.batch_size)
    scored, oos_answered = [], []

    # Negative controls (src/eval/controls.py): pair each single-hop answer with a context that cannot
    # (other_company) or should mostly not (same_company) support it. Pairing compares the companies
    # actually present in the contexts, not question tickers (an Oct 5 audit fix).
    swapped_ctx = {}
    if args.negative_control:
        from src.eval.equivalence import relevant_ids
        eval_qs = json.loads((REPO_ROOT / "data" / "eval_set" / "eval_questions.json").read_text(encoding="utf-8"))
        swapped_ctx = pair_contexts(data["results"], chunks_by_id, args.negative_control,
                                    {q["id"]: relevant_ids(q) for q in eval_qs})
        unpaired = [r["id"] for r in live_answers(data["results"]) if r["id"] not in swapped_ctx]
        print(f"NEGATIVE CONTROL ({args.negative_control}): {len(swapped_ctx)} answers paired; "
              f"no valid partner for {unpaired or 'none'}")
    if args.header_only:
        print("HEADER-ONLY CONTROL: chunk text replaced by a filler sentence; provenance headers kept")

    for r in data["results"]:
        if r["is_refusal"]:
            continue
        if args.negative_control:
            if r["id"] not in swapped_ctx:
                continue
            r = {**r, "context_chunk_ids": swapped_ctx[r["id"]]}
        if (args.negative_control or args.header_only) and r["type"] != "single_hop":
            continue
        missing = [c for c in r["context_chunk_ids"] if c not in chunks_by_id]
        if missing:
            sys.exit(f"{r['id']}: context chunk ids not found in data/processed/chunks ({missing[:2]}...). "
                     f"Answers file and chunk files are out of sync.")
        ctx_chunks = [chunks_by_id[c] for c in r["context_chunk_ids"]]
        if args.header_only:
            ctx_chunks = [header_only(c) for c in ctx_chunks]
        s = score_answer(r["answer"], ctx_chunks, scorer,
                         args.entail_threshold, args.contradiction_threshold, args.window_words, include_header, scope_named, strip_framing)
        row = {"id": r["id"], "type": r["type"], "question": r["question"], "answer": r["answer"],
               "gold_in_context": r["gold_in_context"], "context_chunk_ids": r["context_chunk_ids"], **s}
        (scored if r["type"] == "single_hop" else oos_answered).append(row)
        print(f"scored {r['id']}: {s['n_supported']}/{s['n_sentences']} sentences supported")
    print(f"\nNLI label order used: {scorer.label_order if scored or oos_answered else 'n/a'}")

    for row in scored:
        row["gold_group"] = "gold_in_context" if row["gold_in_context"] else "gold_missing_from_context"

    overall = aggregate_faithfulness(scored, args.n_resamples, args.seed)
    by_gold = aggregate_by(scored, lambda a: a["gold_group"], args.n_resamples, args.seed)

    sensitivity = {}
    for t in (0.3, 0.5, 0.7, 0.9):
        recounted = [recount_at_threshold(a, t, args.contradiction_threshold) for a in scored]
        sensitivity[str(t)] = aggregate_faithfulness(recounted, args.n_resamples, args.seed)["mean_answer_faithfulness"]

    all_results = data["results"]
    sh_all = [r for r in all_results if r["type"] == "single_hop"]
    oos_all = [r for r in all_results if r["type"] == "out_of_scope"]
    refusal = {
        "out_of_scope_refusal_rate": cluster_mean_bootstrap([1.0 if r["is_refusal"] else 0.0 for r in oos_all],
                                                            args.n_resamples, seed=args.seed),
        "single_hop_refused": {
            "gold_in_context": sum(r["is_refusal"] for r in sh_all if r["gold_in_context"]),
            "of_gold_in_context": sum(1 for r in sh_all if r["gold_in_context"]),
            "gold_missing": sum(r["is_refusal"] for r in sh_all if not r["gold_in_context"]),
            "of_gold_missing": sum(1 for r in sh_all if not r["gold_in_context"]),
        },
    }

    cfg = data["config"]
    print("\n" + "=" * 72)
    print(f"DAY 7 FAITHFULNESS  |  retriever={cfg['retriever']} k={cfg['k']} context={cfg['context_mode']} "
          f"| generator={cfg['model']} | NLI={NLI_MODEL_NAME}")
    print(f"premise provenance header: {'ON' if include_header else 'OFF (ablation)'} | "
          f"claims scored against the filing they name: {'ON' if scope_named else 'OFF (ablation)'} | "
          f"attribution wrapper stripped before NLI: {'ON (experiment)' if strip_framing else 'OFF'}")
    print(f"entail threshold={args.entail_threshold}, window={args.window_words} words, "
          f"resamples={args.n_resamples} (resampling unit = question)")
    print("=" * 72)
    print_block("ALL SINGLE-HOP ANSWERS", overall)
    for group, agg in by_gold.items():
        print_block(f"  stratum: {group}", agg)
    all_claims = [x for a in scored for x in a["sentences"]]
    n_scoped = sum(1 for x in all_claims if x.get("scope") == "named_filing")
    print(f"\nClaims scored against the filing they name: {n_scoped}/{len(all_claims)} "
          f"(the rest name no matchable filing date and are scored against all context)")
    print("\nHeadline sensitivity to the entailment threshold (point estimates):")
    for t, m in sensitivity.items():
        print(f"  threshold {t}: {m['estimate']:.3f}  [{m['ci_95'][0]:.3f}, {m['ci_95'][1]:.3f}]")
    print(f"\nOut-of-scope correctly refused: {fmt(refusal['out_of_scope_refusal_rate'])}")
    sr = refusal["single_hop_refused"]
    print(f"Single-hop refusals: {sr['gold_in_context']}/{sr['of_gold_in_context']} when gold was in context "
          f"(false refusals), {sr['gold_missing']}/{sr['of_gold_missing']} when gold was missing (correct abstentions)")
    if oos_answered:
        print(f"Out-of-scope questions the model ANSWERED instead of refusing: {[r['id'] for r in oos_answered]}")

    out = {"config": cfg, "scoring": {"nli_model": NLI_MODEL_NAME, "label_order": scorer.label_order if (scored or oos_answered) else None,
                                      "entail_threshold": args.entail_threshold,
                                      "contradiction_threshold": args.contradiction_threshold,
                                      "window_words": args.window_words, "premise_header": include_header, "scope_to_named_filing": scope_named, "strip_framing": strip_framing, "negative_control": args.negative_control, "header_only": args.header_only, "n_resamples": args.n_resamples,
                                      "seed": args.seed},
           "faithfulness_single_hop": overall, "by_gold_in_context": by_gold,
           "threshold_sensitivity_mean_answer_faithfulness": sensitivity, "refusal": refusal,
           "per_answer_single_hop": scored, "out_of_scope_answered": oos_answered,
           "generation_errors": data.get("errors", [])}
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
