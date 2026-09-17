"""
Generates data/eval_set/eval_questions.json — the Day 3 eval harness question set.

Design notes (why the schema looks like this):

- `gold_answer_type` is the key field. Not every question type has a "gold chunk":
    - "chunk"  -> single-hop retrieval questions. Correct behavior = the vector/keyword
                  retriever surfaces the right chunk(s) in the top-k. Recall@k/MRR apply.
    - "sql"    -> multi-hop-numeric questions. Correct behavior = the agent routes to the
                  SQL/calculator tool, NOT the text retriever. These questions are included
                  in the Day 3 set on purpose, even though no retrieval-side agent exists
                  yet: they let the harness check a real failure mode early — does the
                  keyword/dense baseline confidently return a *prose* chunk for a question
                  that actually needs a computed number? That's a measurable false-precision
                  failure, and it's cheap to check before Phase B's agent exists.
                  Recall@k/MRR are reported as N/A for these until the SQL tool exists
                  (Phase B, Oct 8) — scoring.py skips them from the retrieval metrics and
                  reports them separately as "sql_gold_pending".
    - "refuse" -> out-of-scope questions. Correct behavior = the system declines rather
                  than fabricating from whatever it retrieves. These test both today's
                  keyword baseline (does it force a confident-looking top match even when
                  nothing relevant exists?) and, later, the agent's refusal behavior.

- `gold_chunk_ids` is deliberately left empty ([]) for every "chunk" question. This chat
  does not have the actual processed corpus/chunk IDs — only the plan doc was uploaded.
  Filling these in requires running scripts/label_eval_set.py against the real
  data/processed/ chunks locally. Shipping fabricated chunk IDs here would silently poison
  the eval set's ground truth, which is worse than leaving it explicitly TODO.

- `gold_sql_description` is a plain-English spec of the computation (which two XBRL rows,
  what formula) rather than a hardcoded number — precise fiscal-year figures live in your
  XBRL SQL table, not in this chat's knowledge, and hardcoding a guessed number here would
  be the same silent-poisoning problem as above.

- Section targeting respects the Day 1 log's known limitation: no single-hop question
  targets 10-Q Item 1 for MSFT/GOOGL/NVDA/ORCL, since that section is documented as
  truncated (~90-115 words) for those four companies. All single-hop questions target
  Item 1A (Risk Factors) or Item 7/Item 2 (MD&A), which the log confirms are clean across
  all 8 companies.

- `ticker` (not `company`) and `target_item_key` (not just prose `target_section`) match
  the real field names your ingestion pipeline actually writes (src/ingestion/pipeline.py:
  base_metadata has "ticker" and "item", with item values like "item_1a"). Keeping this
  set's vocabulary identical to the corpus's is what lets scripts/label_eval_set.py filter
  candidate chunks by exact field match instead of parsing prose section names.

Counts: 20 single-hop / 20 multi-hop-numeric / 10 out-of-scope = 50 total (within the
40-60 target range, upper-middle since this set also has to serve Phase C hard-negative
mining later).
"""
import json
from pathlib import Path

COMPANIES = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL"]

single_hop = [
    dict(company="AAPL", section="10-K Item 1A",
         question="What does Apple identify as a risk related to its dependence on manufacturing partners concentrated in China?",
         note="Concentration/supply-chain risk theme; directly tests the kind of question the corpus was chosen for."),
    dict(company="AAPL", section="10-K Item 7",
         question="What does Apple's MD&A cite as a driver of change in Services segment revenue?"),
    dict(company="AAPL", section="10-K Item 1A",
         question="What does Apple's risk factors section say about exposure to foreign currency exchange rate fluctuations?"),
    dict(company="MSFT", section="10-K Item 1A",
         question="What does Microsoft's risk factors describe as risks associated with its investment in cloud and AI infrastructure capacity?"),
    dict(company="MSFT", section="10-K Item 7",
         question="What does Microsoft's MD&A attribute as a driver of Intelligent Cloud segment revenue growth?"),
    dict(company="MSFT", section="10-K Item 1A",
         question="What does Microsoft's risk factors section say about cybersecurity and data breach risk?"),
    dict(company="GOOGL", section="10-K Item 1A",
         question="What does Alphabet's risk factors section say about regulatory and antitrust scrutiny of its advertising business?"),
    dict(company="GOOGL", section="10-K Item 7",
         question="What does Alphabet's MD&A identify as a factor affecting Google Cloud segment performance?"),
    dict(company="GOOGL", section="10-K Item 7",
         question="What does Alphabet's MD&A say about trends in headcount and operating expenses?"),
    dict(company="AMZN", section="10-K Item 1A",
         question="What does Amazon's risk factors section describe about competitive pressure in its retail business?"),
    dict(company="AMZN", section="10-K Item 7",
         question="What does Amazon's MD&A cite as a driver of change in AWS segment operating income?"),
    dict(company="AMZN", section="10-K Item 1A",
         question="What does Amazon's risk factors section say about risks related to fulfillment network capacity?"),
    dict(company="META", section="10-K Item 1A",
         question="What does Meta's risk factors section say about risks tied to third-party platform or app-store policies?"),
    dict(company="META", section="10-K Item 7",
         question="What does Meta's MD&A say about Reality Labs segment operating losses?"),
    dict(company="NVDA", section="10-K Item 1A",
         question="What does Nvidia's risk factors section describe about export control and trade restriction risk?"),
    dict(company="NVDA", section="10-K Item 7",
         question="What does Nvidia's MD&A attribute as the primary driver of Data Center segment revenue growth?"),
    dict(company="AVGO", section="10-K Item 1A",
         question="What does Broadcom's risk factors section describe about customer concentration risk?"),
    dict(company="AVGO", section="10-K Item 7",
         question="What does Broadcom's MD&A say about the relationship between its semiconductor solutions and infrastructure software segments?"),
    dict(company="ORCL", section="10-K Item 1A",
         question="What does Oracle's risk factors section describe about competition in cloud infrastructure (OCI)?"),
    dict(company="ORCL", section="10-K Item 7",
         question="What does Oracle's MD&A cite as a driver of cloud services and license support revenue?"),
]

multi_hop_numeric = [
    dict(company="AAPL", sql_spec="Total revenue for Apple's two most recent fiscal years in the XBRL table; compute % change."),
    dict(company="AAPL", sql_spec="Net income for Apple's two most recent fiscal years; compute % change."),
    dict(company="MSFT", sql_spec="Total revenue for Microsoft's two most recent fiscal years; compute % change."),
    dict(company="MSFT", sql_spec="Net income for Microsoft's two most recent fiscal years; compute % change."),
    dict(company="GOOGL", sql_spec="Total revenue for Alphabet's two most recent fiscal years; compute % change."),
    dict(company="GOOGL", sql_spec="Net income for Alphabet's two most recent fiscal years; compute % change."),
    dict(company="AMZN", sql_spec="Total revenue for Amazon's two most recent fiscal years; compute % change."),
    dict(company="AMZN", sql_spec="Net income margin (net income / revenue) for Amazon's most recent fiscal year."),
    dict(company="META", sql_spec="Total revenue for Meta's two most recent fiscal years; compute % change."),
    dict(company="META", sql_spec="Net income for Meta's two most recent fiscal years; compute % change."),
    dict(company="NVDA", sql_spec="Total revenue for Nvidia's two most recent fiscal years; compute % change."),
    dict(company="NVDA", sql_spec="Net income for Nvidia's two most recent fiscal years; compute % change."),
    dict(company="AVGO", sql_spec="Total revenue for Broadcom's two most recent fiscal years; compute % change."),
    dict(company="AVGO", sql_spec="Net income for Broadcom's two most recent fiscal years; compute % change."),
    dict(company="ORCL", sql_spec="Total revenue for Oracle's two most recent fiscal years; compute % change."),
    dict(company="ORCL", sql_spec="Net income for Oracle's two most recent fiscal years; compute % change."),
    dict(company="AAPL,MSFT", sql_spec="Revenue growth rate for both Apple and Microsoft across their two most recent fiscal years; compare which is higher. True cross-company multi-hop: two SQL rows per company, one comparison."),
    dict(company="NVDA,AVGO", sql_spec="Net income % change for both Nvidia and Broadcom across their two most recent fiscal years; compare which increase is larger."),
    dict(company="GOOGL", sql_spec="Average quarterly revenue across the four most recent quarters in Alphabet's XBRL data (requires quarterly rows, not just annual)."),
    dict(company="ORCL", sql_spec="Revenue-to-net-income ratio for Oracle's most recent fiscal year (tests a less common derived metric than straight growth rate)."),
]

out_of_scope = [
    dict(refusal_reason="real_time_data", question="What is Apple's current stock price?"),
    dict(refusal_reason="off_corpus_company", question="What did Tesla report as a risk factor in its most recent 10-K?"),
    dict(refusal_reason="investment_advice", question="Should I buy Nvidia stock right now?"),
    dict(refusal_reason="non_disclosed_info", question="What is Microsoft's internal source code architecture for Azure?"),
    dict(refusal_reason="non_filing_source", question="What did Amazon's CEO say on last week's earnings call?"),
    dict(refusal_reason="irrelevant_trivia", question="What color is Oracle's corporate logo?"),
    dict(refusal_reason="forward_looking_prediction", question="What will Meta's revenue be next fiscal year?"),
    dict(refusal_reason="unrelated_historical", question="What did Google's founder say about AI in a 2015 interview?"),
    dict(refusal_reason="off_corpus_comparison", question="Compare Apple's product design philosophy to Samsung's."),
    dict(refusal_reason="unrelated_topic", question="What is the weather forecast for Cupertino this week?"),
]

_SECTION_TO_ITEM_KEY = {
    "10-K Item 1A": "item_1a",
    "10-K Item 7": "item_7",
}


def build():
    questions = []
    qid = 1
    for q in single_hop:
        questions.append(dict(
            id=f"sh_{qid:03d}",
            type="single_hop",
            ticker=q["company"],
            target_section=q["section"],  # human-readable, for display
            target_item_key=_SECTION_TO_ITEM_KEY[q["section"]],  # machine key matching pipeline.py's "item" field
            question=q["question"],
            gold_answer_type="chunk",
            gold_chunk_ids=[],  # TODO: fill via scripts/label_eval_set.py against real chunks
            gold_sql_description=None,
            refusal_reason=None,
            notes=q.get("note", ""),
        ))
        qid += 1
    qid = 1
    for q in multi_hop_numeric:
        questions.append(dict(
            id=f"mhn_{qid:03d}",
            type="multi_hop_numeric",
            ticker=q["company"],  # may be "AAPL" or "AAPL,MSFT" for cross-company questions
            target_section=None,
            target_item_key=None,
            question=None,  # filled below from sql_spec-derived phrasing
            gold_answer_type="sql",
            gold_chunk_ids=[],
            gold_sql_description=q["sql_spec"],
            refusal_reason=None,
            notes="Recall@k/MRR reported as N/A for this question until the SQL tool exists (Phase B). "
                  "Used pre-Phase-B to check the retriever doesn't falsely surface a prose chunk as if it answers a computed question.",
        ))
        qid += 1
    qid = 1
    for q in out_of_scope:
        questions.append(dict(
            id=f"oos_{qid:03d}",
            type="out_of_scope",
            ticker=None,
            target_section=None,
            target_item_key=None,
            question=q["question"],
            gold_answer_type="refuse",
            gold_chunk_ids=[],
            gold_sql_description=None,
            refusal_reason=q["refusal_reason"],
            notes="Correct behavior is refusal/no-confident-match, not a retrieved chunk.",
        ))
        qid += 1

    # Fill in question text for multi_hop_numeric now that spec is set (kept as a
    # natural-language question a labeler / agent would actually see)
    numeric_questions = {
        "mhn_001": "What was Apple's year-over-year total revenue growth rate between its two most recent fiscal years in the dataset?",
        "mhn_002": "By what percentage did Apple's net income change between its two most recent fiscal years in the dataset?",
        "mhn_003": "What was Microsoft's year-over-year revenue growth rate between its two most recent fiscal years in the dataset?",
        "mhn_004": "What was the percentage change in Microsoft's net income between its two most recent fiscal years in the dataset?",
        "mhn_005": "What was Alphabet's year-over-year revenue growth rate between its two most recent fiscal years in the dataset?",
        "mhn_006": "What was the percentage change in Alphabet's net income between its two most recent fiscal years in the dataset?",
        "mhn_007": "What was Amazon's year-over-year revenue growth rate between its two most recent fiscal years in the dataset?",
        "mhn_008": "What was Amazon's net income margin (net income divided by revenue) for its most recent fiscal year in the dataset?",
        "mhn_009": "What was Meta's year-over-year revenue growth rate between its two most recent fiscal years in the dataset?",
        "mhn_010": "What was the percentage change in Meta's net income between its two most recent fiscal years in the dataset?",
        "mhn_011": "What was Nvidia's year-over-year revenue growth rate between its two most recent fiscal years in the dataset?",
        "mhn_012": "What was the percentage change in Nvidia's net income between its two most recent fiscal years in the dataset?",
        "mhn_013": "What was Broadcom's year-over-year revenue growth rate between its two most recent fiscal years in the dataset?",
        "mhn_014": "What was the percentage change in Broadcom's net income between its two most recent fiscal years in the dataset?",
        "mhn_015": "What was Oracle's year-over-year revenue growth rate between its two most recent fiscal years in the dataset?",
        "mhn_016": "What was the percentage change in Oracle's net income between its two most recent fiscal years in the dataset?",
        "mhn_017": "Between Apple and Microsoft, which had the higher year-over-year revenue growth rate in their most recent fiscal years in the dataset?",
        "mhn_018": "Between Nvidia and Broadcom, which had the larger percentage increase in net income in their most recent fiscal years in the dataset?",
        "mhn_019": "What was Alphabet's average quarterly revenue across the four most recent quarters in the dataset?",
        "mhn_020": "What was Oracle's revenue-to-net-income ratio for its most recent fiscal year in the dataset?",
    }
    for q in questions:
        if q["id"] in numeric_questions:
            q["question"] = numeric_questions[q["id"]]

    return questions

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "data" / "eval_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = build()
    out_path = out_dir / "eval_questions.json"
    with open(out_path, "w") as f:
        json.dump(questions, f, indent=2)
    counts = {}
    for q in questions:
        counts[q["type"]] = counts.get(q["type"], 0) + 1
    print(f"Wrote {len(questions)} questions to {out_path}")
    print("Breakdown:", counts)
