"""
Reference answers for the 20 multi_hop_numeric eval questions, computed ONLY through the Day 8 tools
(typed fact lookups + calculator), so the agent in Day 9 has something concrete to be scored against.

    python -m scripts.build_mhn_reference

Writes data/eval_set/day8_mhn_reference.json. Honest labelling of what this is: the eval set stores
only a `gold_sql_description` for these questions (no numeric gold), so these are reference values
derived from the same XBRL table the SQL tool reads. They check that the agent *uses the tools
correctly*; they do not independently validate the XBRL data. The tool code itself is checked
against a separate plain-Python read of the raw CSVs in tests/test_facts_db.py.

Every entry carries `status`:
  ok            computed from complete data
  caveat        computed, but something a reader should know (listed in `caveats`)
  unanswerable  the data cannot answer it as asked; `reason` says why (the agent should say the same)
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agent.calculator import CalcError, calculate  # noqa: E402
from src.agent.facts_db import FactsDB  # noqa: E402

# question id -> spec. Derived by hand from each question's text / gold_sql_description.
SPECS = {
    "mhn_001": ("growth", "AAPL", "revenue"), "mhn_002": ("growth", "AAPL", "net_income"),
    "mhn_003": ("growth", "MSFT", "revenue"), "mhn_004": ("growth", "MSFT", "net_income"),
    "mhn_005": ("growth", "GOOGL", "revenue"), "mhn_006": ("growth", "GOOGL", "net_income"),
    "mhn_007": ("growth", "AMZN", "revenue"), "mhn_008": ("margin", "AMZN"),
    "mhn_009": ("growth", "META", "revenue"), "mhn_010": ("growth", "META", "net_income"),
    "mhn_011": ("growth", "NVDA", "revenue"), "mhn_012": ("growth", "NVDA", "net_income"),
    "mhn_013": ("growth", "AVGO", "revenue"), "mhn_014": ("growth", "AVGO", "net_income"),
    "mhn_015": ("growth", "ORCL", "revenue"), "mhn_016": ("growth", "ORCL", "net_income"),
    "mhn_017": ("compare_growth", ("AAPL", "MSFT"), "revenue", "higher"),
    "mhn_018": ("compare_growth", ("NVDA", "AVGO"), "net_income", "larger"),
    "mhn_019": ("avg_quarters", "GOOGL", "revenue", 4),
    "mhn_020": ("ratio", "ORCL"),
}


def growth(db: FactsDB, ticker: str, metric: str) -> dict:
    lk = db.fiscal_year_pair(ticker, metric)
    inputs = [f.as_dict() for f in lk.facts]
    if lk.status == "unanswerable":
        return {"status": "unanswerable", "reason": lk.reason, "inputs": inputs}
    new, old = lk.facts
    try:
        r = calculate(f"pct_change({old.value}, {new.value})")
    except CalcError as e:
        return {"status": "unanswerable", "reason": str(e), "inputs": inputs}
    return {"status": lk.status, "caveats": list(lk.caveats), "answer_percent": round(r.value, 4),
            "expression": r.expression, "inputs": inputs}


def build_reference(db: FactsDB) -> dict:
    out = {}
    for qid, spec in SPECS.items():
        kind = spec[0]
        if kind == "growth":
            out[qid] = {"kind": kind, **growth(db, spec[1], spec[2])}
        elif kind == "compare_growth":
            _, tickers, metric, word = spec
            parts = {t: growth(db, t, metric) for t in tickers}
            if any(p["status"] == "unanswerable" for p in parts.values()):
                out[qid] = {"kind": kind, "status": "unanswerable", "parts": parts,
                            "reason": "one of the growth rates could not be computed"}
            else:
                best = max(tickers, key=lambda t: parts[t]["answer_percent"])
                cav = [c for p in parts.values() for c in p["caveats"]]
                top = [t for t in tickers if parts[t]["answer_percent"] == parts[best]["answer_percent"]]
                if len(top) > 1:          # never pick a winner silently on a tie
                    best = None
                    cav.append(f"tie: {', '.join(top)} have the same growth rate to 4 decimal places")
                out[qid] = {"kind": kind, "status": "caveat" if cav else "ok", "caveats": cav,
                            "winner": best, "criterion": word, "parts": parts}
        elif kind == "margin" or kind == "ratio":
            t = spec[1]
            lk = db.same_year_pair(t, "net_income", "revenue")
            if lk.status == "unanswerable":
                out[qid] = {"kind": kind, "status": "unanswerable", "reason": lk.reason}
                continue
            ni, rev = lk.facts
            if kind == "margin":
                r = calculate(f"pct({ni.value}, {rev.value})")
                out[qid] = {"kind": kind, "status": lk.status, "answer_percent": round(r.value, 4),
                            "expression": r.expression, "inputs": [ni.as_dict(), rev.as_dict()]}
            else:
                r = calculate(f"ratio({rev.value}, {ni.value})")
                out[qid] = {"kind": kind, "status": lk.status, "answer": round(r.value, 4),
                            "expression": r.expression, "inputs": [rev.as_dict(), ni.as_dict()]}
        elif kind == "avg_quarters":
            _, t, metric, n = spec
            lk = db.quarter_window(t, metric, n)
            inputs = [f.as_dict() for f in lk.facts]
            if lk.status == "unanswerable":
                out[qid] = {"kind": kind, "status": "unanswerable", "reason": lk.reason, "inputs": inputs}
            else:
                vals = ", ".join(str(f.value) for f in lk.facts)
                r = calculate(f"mean({vals})")
                out[qid] = {"kind": kind, "status": lk.status, "caveats": list(lk.caveats),
                            "answer": round(r.value, 2), "expression": r.expression, "inputs": inputs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO_ROOT / "data" / "processed" / "facts.sqlite"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "eval_set" / "day8_mhn_reference.json"))
    args = ap.parse_args()
    db = FactsDB(args.db)
    ref = build_reference(db)
    Path(args.out).write_text(json.dumps({"note": __doc__.strip().split("\n\n")[1], "reference": ref},
                                         indent=2, ensure_ascii=False), encoding="utf-8")
    counts = {}
    for v in ref.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    print(f"Wrote {args.out}: {len(ref)} questions, status counts {counts}")
    for qid, v in ref.items():
        ans = v.get("answer_percent", v.get("answer", v.get("winner", "-")))
        extra = f" | {'; '.join(v.get('caveats', []))}" if v.get("caveats") else ""
        extra += f" | {v['reason']}" if v.get("reason") else ""
        print(f"  {qid} [{v['status']}] {v['kind']}: {ans}{extra}")
    db.close()


if __name__ == "__main__":
    main()
