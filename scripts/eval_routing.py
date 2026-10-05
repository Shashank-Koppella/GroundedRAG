"""
Route-level evaluation: does a router send each question to the right route, and how bad are its misses?

    python -m scripts.eval_routing --router rules --set probe
    python -m scripts.eval_routing --router rules --set eval

Sets:
  eval   data/eval_set/eval_questions.json (50; labels mapped chunk->retrieve, sql->structured, refuse->refuse).
         The rules were written with these questions visible, so a score here is a sanity check, not evidence.
  probe  data/eval_set/day8_routing_probe.json (48; written blind to the rules by a separate agent, frozen
         Oct 5 before the rules were run on it). This is the routing number to report.

Misses are not equal, so each is graded by what it would cost (SEVERITY below). The headline safety
number is the count of `critical` misses: an unanswerable question that gets answered, or a structured
question sent to text retrieval, which the Day 8 RAG-only baseline showed produces stale-year numbers
with valid-looking citations.

Routers are plain functions question -> {"route", "tickers", "refuse_reason"}. The LLM planner plugs in here
later as another entry in ROUTERS, so both are scored by identical code.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agent.route_rules import classify  # noqa: E402
from src.eval.bootstrap import cluster_mean_bootstrap  # noqa: E402

SETS = {
    "eval": REPO_ROOT / "data" / "eval_set" / "eval_questions.json",
    "probe": REPO_ROOT / "data" / "eval_set" / "day8_routing_probe.json",
}
EVAL_ROUTE = {"chunk": "retrieve", "sql": "structured", "refuse": "refuse"}

# (gold, predicted) -> severity. Anything not listed and not correct is "minor".
SEVERITY = {
    ("refuse", "structured"): "critical", ("refuse", "retrieve"): "critical", ("refuse", "hybrid"): "critical",
    ("structured", "retrieve"): "critical",          # numbers from text: stale-year risk (Day 8 baseline)
    ("hybrid", "retrieve"): "critical",              # same, for the numeric half
    ("structured", "refuse"): "over_refusal", ("retrieve", "refuse"): "over_refusal",
    ("hybrid", "refuse"): "over_refusal",
    ("hybrid", "structured"): "partial",             # number right, explanation missing
    ("retrieve", "structured"): "minor",             # executor finds no such metric and refuses, or answers the wrong question
    ("retrieve", "hybrid"): "minor", ("structured", "hybrid"): "minor",
}


def rules_router(question: str) -> dict:
    v = classify(question)
    return {"route": v.route, "tickers": v.tickers, "refuse_reason": v.refuse_reason, "fired": v.fired}


ROUTERS = {"rules": rules_router}


def load_set(name: str) -> list:
    raw = json.loads(Path(SETS[name]).read_text(encoding="utf-8"))
    if name == "eval":
        out = []
        for q in raw:
            tick = [] if q["ticker"] in (None, "None") else q["ticker"].split(",")
            out.append({"id": q["id"], "question": q["question"], "gold_route": EVAL_ROUTE[q["gold_answer_type"]],
                        "gold_tickers": tick, "refuse_reason": None, "difficulty": q["type"], "ambiguous": False})
        return out
    return raw["questions"]


def evaluate(questions: list, router) -> dict:
    rows, conf = [], collections.Counter()
    for q in questions:
        p = router(q["question"])
        correct = p["route"] == q["gold_route"]
        sev = "correct" if correct else SEVERITY.get((q["gold_route"], p["route"]), "minor")
        tick_ok = None
        if correct and q["gold_route"] != "refuse":
            tick_ok = sorted(p["tickers"]) == sorted(q["gold_tickers"])
        reason_ok = None
        if correct and q["gold_route"] == "refuse" and q.get("refuse_reason"):
            reason_ok = p.get("refuse_reason") == q["refuse_reason"]
        rows.append({"id": q["id"], "question": q["question"], "gold_route": q["gold_route"], "pred": p,
                     "correct": correct, "severity": sev, "tickers_ok": tick_ok, "refuse_reason_ok": reason_ok,
                     "difficulty": q.get("difficulty"), "ambiguous": bool(q.get("ambiguous"))})

    scored = [r for r in rows if not r["ambiguous"]]
    for r in scored:                      # confusion uses the same rows as every other metric
        conf[(r["gold_route"], r["pred"]["route"])] += 1

    def acc(rs):
        return {"n": len(rs), "correct": sum(r["correct"] for r in rs),
                "accuracy": (sum(r["correct"] for r in rs) / len(rs)) if rs else None}

    by = lambda key: {k: acc([r for r in scored if r[key] == k]) for k in sorted({r[key] for r in scored})}
    tick = [r["tickers_ok"] for r in scored if r["tickers_ok"] is not None]
    reason = [r["refuse_reason_ok"] for r in scored if r["refuse_reason_ok"] is not None]
    return {
        "route_accuracy": {**acc(scored), "ci": cluster_mean_bootstrap([float(r["correct"]) for r in scored])},
        "by_gold_route": by("gold_route"),
        "by_difficulty": by("difficulty"),
        "severity_counts": dict(collections.Counter(r["severity"] for r in scored)),
        "tickers_exact_when_route_correct": {"n": len(tick), "correct": sum(tick)},
        "refuse_reason_when_route_correct": {"n": len(reason), "correct": sum(reason)},
        "confusion_gold_to_pred": {f"{g}->{p}": n for (g, p), n in sorted(conf.items())},
        "n_ambiguous_excluded": len(rows) - len(scored),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", choices=sorted(ROUTERS), default="rules")
    ap.add_argument("--set", choices=sorted(SETS), default="probe")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = evaluate(load_set(args.set), ROUTERS[args.router])
    out = Path(args.out) if args.out else REPO_ROOT / "data" / "eval_set" / f"day8_routing_{args.router}_{args.set}.json"
    out.write_text(json.dumps({"router": args.router, "set": args.set, **res}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    ra = res["route_accuracy"]
    print(f"{args.router} on {args.set}: route accuracy {ra['correct']}/{ra['n']} = {ra['accuracy']:.3f} "
          f"(95% CI {ra['ci']['ci_95'][0]:.3f}-{ra['ci']['ci_95'][1]:.3f}{', small n' if ra['ci']['small_n'] else ''})")
    print(f"  severity: {res['severity_counts']}")
    for k, v in res["by_gold_route"].items():
        print(f"  gold {k:10} {v['correct']}/{v['n']}")
    for k, v in res["by_difficulty"].items():
        print(f"  {k:12} {v['correct']}/{v['n']}")
    t, rr = res["tickers_exact_when_route_correct"], res["refuse_reason_when_route_correct"]
    print(f"  tickers exact (route correct, non-refuse): {t['correct']}/{t['n']}; refuse reason right: {rr['correct']}/{rr['n']}")
    print(f"  confusion: {res['confusion_gold_to_pred']}")
    for r in res["rows"]:
        if not r["correct"]:
            print(f"  MISS {r['id']} [{r['severity']}] gold={r['gold_route']} pred={r['pred']['route']}"
                  f"({r['pred'].get('refuse_reason') or ''}) :: {r['question'][:90]}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
