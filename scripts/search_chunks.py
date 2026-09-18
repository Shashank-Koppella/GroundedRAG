"""
For when label_eval_set.py's top-8 candidates don't contain a real answer (this
happened for 8 of the 20 single-hop questions on the first labeling pass — see
scripts/apply_reviewed_labels.py's NOT_LABELED dict for which ones and why).

Two things this gives you that label_eval_set.py doesn't: a bigger candidate pool
(top 20 instead of top 8), and FULL untruncated chunk text instead of a 220-char
preview, since the truncation is often exactly why the real answer got missed on
a skim.

Usage (run from repo root):

    python scripts/search_chunks.py AMZN "AWS operating income segment"
    python scripts/search_chunks.py GOOGL "headcount employees operating expenses" --item item_7
    python scripts/search_chunks.py ORCL "risk factors competition cloud infrastructure" --item item_1a

--item filters to one section (e.g. item_1a, item_7, item_2) if you know it;
omit to search across all of that company's chunks.
--k controls how many results to show (default 20).
"""
import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = REPO_ROOT / "data" / "processed" / "chunks"

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("query")
    parser.add_argument("--item", default=None, help="filter to one item key, e.g. item_1a")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    path = CHUNKS_DIR / f"{args.ticker.upper()}.jsonl"
    if not path.exists():
        raise SystemExit(f"No chunks file at {path} — check the ticker and that pipeline.py has been run.")

    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if args.item:
        chunks = [c for c in chunks if c.get("item") == args.item]

    q_terms = set(tokenize(args.query))
    scored = sorted(
        chunks,
        key=lambda c: len(q_terms & set(tokenize(c.get("text", "")))),
        reverse=True,
    )

    print(f"{len(chunks)} chunks searched, top {args.k} by term overlap with: {args.query!r}\n")
    for c in scored[: args.k]:
        overlap = len(q_terms & set(tokenize(c.get("text", ""))))
        print("=" * 80)
        print(f"[{c['chunk_id']}]  form={c.get('form')}  filed={c.get('filing_date')}  overlap={overlap}")
        print(c["text"])
        print()


if __name__ == "__main__":
    main()
