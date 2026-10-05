"""
Compare a rebuilt corpus (e.g. data/processed_v2) with the current one (data/processed) BEFORE replacing it.

    python -m scripts.compare_corpus --new data/processed_v2
    python -m scripts.compare_corpus --new data/processed_v2 --xbrl-only

Chunk ids are positional (ticker_accession_item_index), so a change anywhere in the text pipeline can
silently move gold labels onto different text. This script exits non-zero if ANY eval gold or stored
equivalent chunk is missing from the new corpus or has different text; otherwise it reports what changed
(new filings, changed sections, the new 10-Q Item 1 openings for a human read) and the XBRL differences.
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL = REPO_ROOT / "data" / "eval_set" / "eval_questions.json"


def load_chunks(d: Path) -> dict:
    out = {}
    for f in sorted((d / "chunks").glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[r["chunk_id"]] = r
    return out


def load_facts(d: Path) -> dict:
    out = {}
    for f in sorted((d / "xbrl").glob("*_facts.csv")):
        with open(f, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                out[(r["ticker"], r["metric"], r["form"], r["start"], r["end"])] = r
    return out


def compare_text(old: dict, new: dict) -> int:
    problems = 0
    questions = json.loads(EVAL.read_text(encoding="utf-8"))
    labelled = {cid: q["id"] for q in questions
                for cid in (q.get("gold_chunk_ids") or []) + (q.get("equivalent_chunk_ids") or [])}
    print(f"\n== eval labels ({len(labelled)} gold + equivalent ids) ==")
    for cid, qid in sorted(labelled.items(), key=lambda kv: kv[1]):
        if cid not in new:
            print(f"  MISSING  {qid} {cid}")
            problems += 1
        elif cid in old and new[cid]["text"] != old[cid]["text"]:
            print(f"  CHANGED  {qid} {cid}")
            problems += 1
    if not problems:
        print("  all present with identical text")

    common = set(old) & set(new)
    changed = [c for c in common if old[c]["text"] != new[c]["text"]]
    added, removed = set(new) - set(old), set(old) - set(new)
    by = lambda ids: collections.Counter((i.split("_")[0], new.get(i, old.get(i))["form"], new.get(i, old.get(i))["item"]) for i in ids)
    print(f"\n== chunks: old {len(old)}, new {len(new)}; identical {len(common) - len(changed)}, "
          f"changed text {len(changed)}, added {len(added)}, removed {len(removed)} ==")
    for name, ids in (("changed", changed), ("added", added), ("removed", removed)):
        if ids:
            print(f"  {name} by (ticker, form, item):")
            for k, n in sorted(by(ids).items()):
                print(f"    {k}: {n}")

    old_acc = {(c["ticker"], c["accession_number"], c["form"], c["filing_date"]) for c in old.values()}
    new_acc = {(c["ticker"], c["accession_number"], c["form"], c["filing_date"]) for c in new.values()}
    print(f"\n== filings: old {len(old_acc)}, new {len(new_acc)} ==")
    for t, a, f, d in sorted(new_acc - old_acc):
        print(f"  NEW      {t} {f} filed {d} ({a})")
    for t, a, f, d in sorted(old_acc - new_acc):
        print(f"  DROPPED  {t} {f} filed {d} ({a})")

    print("\n== 10-Q Item 1 openings in the new corpus (read these: they must be financial statements, "
          "not a table of contents and not Part II 'Legal Proceedings') ==")
    firsts = sorted((c for c in new.values() if c["form"] == "10-Q" and c["item"] == "item_1" and c["chunk_index"] == 0),
                    key=lambda c: (c["ticker"], c["filing_date"]))
    seen = collections.Counter()
    for c in firsts:
        seen[c["ticker"]] += 1
        if seen[c["ticker"]] <= 2:
            n = sum(1 for x in new.values() if x["accession_number"] == c["accession_number"] and x["item"] == "item_1")
            print(f"  {c['ticker']} {c['filing_date']} ({n} chunks): {' '.join(c['text'].split()[:28])} ...")
    return problems


def compare_facts(old: dict, new: dict) -> None:
    print(f"\n== XBRL rows: old {len(old)}, new {len(new)} ==")
    changed = [k for k in set(old) & set(new) if old[k]["val"] != new[k]["val"]]
    retagged = [k for k in set(old) & set(new) if old[k]["tag_used"] != new[k]["tag_used"]]
    print(f"  same period, different value: {len(changed)}; different tag: {len(retagged)}")
    for k in sorted(changed)[:20]:
        print(f"    VALUE  {k}: {old[k]['val']} -> {new[k]['val']}")
    for k in sorted(retagged)[:20]:
        print(f"    TAG    {k}: {old[k]['tag_used']} -> {new[k]['tag_used']} (val {old[k]['val']} -> {new[k]['val']})")
    count = lambda rows: collections.Counter((k[0], k[1]) for k in rows)
    co, cn = count(old), count(new)
    for key in sorted(set(co) | set(cn)):
        if co[key] != cn[key]:
            print(f"  rows {key}: {co[key]} -> {cn[key]}")
    added = sorted(set(new) - set(old))
    for k in added[:40]:
        print(f"    ADDED  {k} val={new[k]['val']} tag={new[k]['tag_used']}")
    if len(added) > 40:
        print(f"    ... {len(added) - 40} more added")
    for k in sorted(set(old) - set(new))[:20]:
        print(f"    REMOVED {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=str(REPO_ROOT / "data" / "processed"))
    ap.add_argument("--new", required=True)
    ap.add_argument("--xbrl-only", action="store_true")
    args = ap.parse_args()
    old_d, new_d = Path(args.old), Path(args.new)
    problems = 0
    if not args.xbrl_only:
        problems = compare_text(load_chunks(old_d), load_chunks(new_d))
    compare_facts(load_facts(old_d), load_facts(new_d))
    if problems:
        print(f"\nFAIL: {problems} eval label(s) missing or changed. Do NOT replace data/processed.")
        sys.exit(1)
    print("\nOK: every eval label survives unchanged. Review the changes above before replacing data/processed.")


if __name__ == "__main__":
    main()
