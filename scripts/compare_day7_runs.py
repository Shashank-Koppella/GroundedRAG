"""
Paired comparison of two Day 7 faithfulness files, e.g. prompt v1 vs v2:

    python -m scripts.compare_day7_runs --a data/eval_set/day7_faithfulness_reranked_k5_retrieved.json ^
        --b data/eval_set/day7_faithfulness_reranked_k5_retrieved_pv2.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval.compare import paired_difference, scoped_fraction  # noqa: E402


def load(path):
    p = Path(path)
    p = p if p.is_absolute() else REPO_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def describe(tag, d):
    cfg, sc = d["config"], d["scoring"]
    f = d["faithfulness_single_hop"]
    sf = scoped_fraction(d["per_answer_single_hop"])
    print(f"{tag}: context={cfg['context_mode']} prompt={cfg.get('prompt_version', 'v1')} | answers scored "
          f"{f['n_answers_scored']}, claims {f['n_sentences_total']} ({sf['scoped']} scoped to a named filing) | "
          f"mean faithfulness {f['mean_answer_faithfulness']['estimate']:.3f} "
          f"[{f['mean_answer_faithfulness']['ci_95'][0]:.3f}, {f['mean_answer_faithfulness']['ci_95'][1]:.3f}] | "
          f"OOS refused {d['refusal']['out_of_scope_refusal_rate']['estimate']:.2f}")
    sr = d["refusal"]["single_hop_refused"]
    print(f"   single-hop refusals with gold in context: {sr['gold_in_context']}/{sr['of_gold_in_context']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline faithfulness JSON")
    ap.add_argument("--b", required=True, help="comparison faithfulness JSON")
    ap.add_argument("--n-resamples", type=int, default=10000)
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    print()
    describe("A (baseline)  ", a)
    describe("B (comparison)", b)
    c = paired_difference(a["per_answer_single_hop"], b["per_answer_single_hop"], args.n_resamples)
    m = c["mean_difference"]
    print(f"\nPAIRED on the {c['n_shared']} questions scored in both runs (difference = B - A):")
    if m:
        flag = "  (n<30: interval is optimistic)" if m["small_n"] else ""
        print(f"  mean per-question difference {m['estimate']:+.3f}  95% CI [{m['ci_95'][0]:+.3f}, {m['ci_95'][1]:+.3f}]{flag}")
    w, t, l = c["wins_ties_losses_for_b"]
    print(f"  B better / tied / worse on {w} / {t} / {l} questions")
    print(f"  scored only in A: {c['only_in_a']}   only in B: {c['only_in_b']}")
    print("\n  id       A      B")
    for q, v in c["per_question"].items():
        print(f"  {q:7} {v['a']:5.2f}  {v['b']:5.2f}")
    print("\nA CI that contains 0 means the difference is not distinguishable from noise at this sample size.")


if __name__ == "__main__":
    main()
