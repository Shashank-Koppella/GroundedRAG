"""
Applies gold_chunk_ids that Claude determined by reading the candidate text from
your `python scripts/label_eval_set.py` run (Sep 17), instead of you re-typing 20
answers by hand. Run once, from the repo root:

    python scripts/apply_reviewed_labels.py

Safe to re-run — only fills in questions that are still unlabeled, never overwrites
an existing gold_chunk_ids list you (or a prior run of this script) already set.

IMPORTANT — read the confidence notes before trusting these blindly:
- HIGH: the candidate text directly and clearly answers the question. Low risk.
- MODERATE: plausible, on-topic, but the ~220-char preview didn't fully confirm it.
  Worth a 10-second manual check of the full chunk text before treating as ground
  truth for anything that matters (e.g. before using it to judge a real retriever).
- 1 question (sh_019) got NO label — a genuine section-extraction bug (Oracle's 10-K Item 1A isn't being
  found at all, see NOT_LABELED below), not a search problem. Everything else that was initially unresolved
  got labeled in round 2 via scripts/search_chunks.py's wider (top-20, full-text) search.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = REPO_ROOT / "data" / "eval_set" / "eval_questions.json"

# (question_id, [chunk_ids], confidence, note)
REVIEWED_LABELS = [
    ("sh_001", ["AAPL_000032019324000123_item_1a_0010"], "HIGH",
     "Explicitly names China/India/Japan/S.Korea/Taiwan/Vietnam manufacturing concentration."),
    ("sh_003", ["AAPL_000032019325000079_item_1a_0042", "AAPL_000032019324000123_item_1a_0042"], "HIGH",
     "Both are the FX-fluctuation risk paragraph from two different fiscal years; either is a valid answer."),
    ("sh_004", ["MSFT_000119312526323660_item_1a_0008"], "MODERATE",
     "About customers' purchasing/deployment decisions not materializing as expected — plausible capacity-investment "
     "risk framing, but preview didn't show an explicit 'AI infrastructure capacity' phrase. Verify full text."),
    ("sh_006", ["MSFT_000119312526323660_item_1a_0041"], "MODERATE",
     "Covers cybersecurity REGULATORY cost risk specifically, not necessarily a 'we could suffer a data breach' "
     "risk factor, which may be a separate chunk not in the top 8. Verify full text."),
    ("sh_007", ["GOOGL_000165204426000018_item_1a_0040"], "HIGH",
     "Explicitly mentions 'competition matters' alongside claims/lawsuits/regulatory investigations."),
    ("sh_012", ["AMZN_000101872424000161_item_1a_0017"], "HIGH",
     "Directly discusses fulfillment/data center capacity risk and cost of adding capacity."),
    ("sh_014", ["META_000162828026003942_item_7_0010"], "MODERATE",
     "Mentions 'significant investments in our RL efforts' (RL = Reality Labs) but preview didn't show the actual "
     "operating-loss figures. Verify full text — may need a neighboring chunk with the $ numbers."),
    ("sh_015", ["NVDA_000104581026000021_item_1a_0050", "NVDA_000104581026000021_item_1a_0052"], "HIGH",
     "Directly discusses export control restrictions on AI/GPU products, from the most recent 10-K."),
    ("sh_016", ["NVDA_000104581026000021_item_7_0015"], "MODERATE",
     "Says revenue 'primarily attributable to the Compute & Networking segment' — Data Center is a platform WITHIN "
     "that segment, not a perfect 1:1 match to the question's exact segment name. Verify full text."),
    ("sh_018", ["AVGO_000173016825000121_item_7_0001", "AVGO_000173016825000121_item_7_0002"], "HIGH",
     "Directly describes both segments (semiconductor solutions; infrastructure software) and their relationship."),
    ("sh_020", ["ORCL_000095017025087926_item_7_0002", "ORCL_000095017025087926_item_7_0005"], "HIGH",
     "Directly lists cloud/license revenue components and drivers (contract renewals, customer satisfaction, pricing)."),
    # --- Labeled from scripts/search_chunks.py output (Sep 17, round 2) ---
    ("sh_002", ["AAPL_000032019325000079_item_7_0004"], "HIGH",
     "Most recent 10-K (FY2025): 'Services net sales increased during 2025 compared to 2024 primarily due to higher "
     "net sales from advertising, the App Store and cloud services.' Exact driver statement."),
    ("sh_005", ["MSFT_000119312526323660_item_7_0008", "MSFT_000119312526323660_item_7_0012"], "HIGH",
     "Most recent 10-K (FY2026): 'Intelligent Cloud revenue increased driven by Azure' plus the detailed breakout "
     "('Azure and other cloud services revenue grew 41% driven by demand for services across the platform')."),
    ("sh_008", ["GOOGL_000165204426000018_item_7_0009", "GOOGL_000165204426000018_item_7_0021"], "HIGH",
     "Most recent 10-K (FY2025): Cloud-specific fluctuation factors (customer usage/demand/supply availability) "
     "plus the explicit operating-income driver (revenue increase, partially offset by infrastructure usage costs "
     "and compensation expenses)."),
    ("sh_009", ["GOOGL_000165204425000014_item_7_0024"], "MODERATE",
     "Mentions 'a decrease in employee compensation expenses of $285 million, primarily due to a decrease in "
     "average headcount' — a real headcount-driven opex statement, but from the prior (FY2024) 10-K, and the "
     "second wider search still didn't surface a cleaner multi-year headcount TREND narrative. Best available."),
    ("sh_010", ["AMZN_000101872426000004_item_1a_0001"], "HIGH",
     "Most recent 10-K (filed 2026-02-06): the full 'We Face Intense Competition' risk factor, explicitly naming "
     "'physical, e-commerce, and omnichannel retail' competitors, pricing pressure, and resource disparities."),
    ("sh_011", ["AMZN_000101872426000004_item_7_0025"], "HIGH",
     "Most recent 10-K (FY2025): 'The increase in AWS operating income in 2025... is primarily due to increased "
     "sales, partially offset by spending on technology infrastructure... to support AWS business growth.'"),
    ("sh_013", ["META_000162828026003942_item_1a_0011"], "HIGH",
     "Most recent 10-K (FY2025): explicitly names 'changes to the content or application of third-party policies "
     "that limit our ability to deliver, target, or measure the effectiveness of advertising, including changes by "
     "mobile operating system and browser providers such as Apple and Google' — the App Store/platform-policy risk."),
    ("sh_017", ["AVGO_000173016825000121_item_1a_0012"], "HIGH",
     "Most recent 10-K (FY2025): explicit customer concentration risk factor with real numbers (top 5 customers "
     "~40% of net revenue; distributors 48%) — exactly the risk factor the question asks about."),
    # --- Labeled after the section_splitter mid-word-title-split fix + pipeline re-run (Sep 17, round 3) ---
    ("sh_019", ["ORCL_000119312526277521_item_1a_0022", "ORCL_000119312526277521_item_1a_0023"], "HIGH",
     "Most recent 10-K (filed 2026-06-22, the same filing the mid-word title-split bug was diagnosed against): "
     "explicitly names OCI's multicloud competition with 'Microsoft Azure, Amazon Web Services and Google Cloud' "
     "and the risk that this strategy could lead customers to migrate away from Oracle's cloud offerings. This "
     "question was unlabelable before the section_splitter.py fix — Oracle's Item 1A content didn't exist in the "
     "corpus at all until it was re-extracted."),
]

# All 20 single-hop questions are now labeled (as of Sep 17, round 3, after the
# section_splitter.py mid-word-title-split fix unblocked sh_019). Kept as an empty dict
# rather than deleted, so future questions added to the eval set have an obvious place to
# record a considered "no" with a reason, the way sh_019 was tracked here before it was
# resolved.
NOT_LABELED = {}


def main():
    questions = json.loads(EVAL_SET_PATH.read_text())
    by_id = {q["id"]: q for q in questions}

    applied, skipped_already_labeled, skipped_missing = [], [], []
    for qid, chunk_ids, confidence, note in REVIEWED_LABELS:
        q = by_id.get(qid)
        if q is None:
            skipped_missing.append(qid)
            continue
        if q["gold_chunk_ids"]:
            skipped_already_labeled.append(qid)
            continue
        q["gold_chunk_ids"] = chunk_ids
        q["notes"] = (q.get("notes", "") + f" | Labeled by Claude, confidence={confidence}: {note}").strip(" |")
        applied.append(qid)

    for qid, reason in NOT_LABELED.items():
        q = by_id.get(qid)
        if q and not q["gold_chunk_ids"]:
            q["notes"] = (q.get("notes", "") + f" | NOT LABELED — reviewed and rejected: {reason}").strip(" |")

    EVAL_SET_PATH.write_text(json.dumps(questions, indent=2))

    print(f"Applied {len(applied)} labels: {applied}")
    if skipped_already_labeled:
        print(f"Skipped (already labeled, not overwritten): {skipped_already_labeled}")
    if skipped_missing:
        print(f"WARNING — question ids not found in eval_questions.json: {skipped_missing}")
    print(f"{len(NOT_LABELED)} questions marked as reviewed-but-unresolved (see notes in the JSON): "
          f"{list(NOT_LABELED.keys())}")


if __name__ == "__main__":
    main()
