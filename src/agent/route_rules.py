"""
Deterministic routing rules: the guard the agent's LLM planner runs behind, and the rules-only
routing baseline it has to beat.

Routes (docs/day8_routing_design.md has the reasoning):
  structured  the answer is a company-total revenue or net-income value from the XBRL facts table,
              or arithmetic over such values (SQL lookups + calculator; numbers never come from the LLM)
  retrieve    the answer is stated in filing text (qualitative, or a figure the text states, including
              segment figures and metrics the facts table does not hold)
  hybrid      a structured value AND a text explanation are both asked for
  refuse      the corpus cannot answer it: a company outside the eight, real-time data, a forecast,
              investment advice, or a source that is not a filing (calls, interviews, internals)

Two uses, deliberately different:
  classify(q)       the rules-only router (a measured baseline, and the degraded mode when the planner
                    call fails)
  hard_refusal(q)   the guard: refusal triggers precise enough to overrule the planner. The guard only
                    ever moves a decision TOWARD refusal; it never upgrades a refusal into an action.

Frozen on Oct 5 before the routing probe set was written, so that set measures these rules rather than
being fitted by them. Changes after that point must be logged in CHANGES.md with the probe score
before and after.
Revision 2 (Oct 5, audit): precision fixes found by a code audit on ordinary questions, NOT on probe
questions: "invest in" vetoed filing questions ("how much did Microsoft invest in OpenAI"); the
uncovered-company list held common words ("arm", "visa", "hp", "sap"); "sales and marketing" read as
revenue. NOT changed: owned brands (VMware, Instagram, WhatsApp) stay company aliases only, because a
brand can name the parent ("VMware's parent company") or a segment ("revenue from VMware") and rules
cannot tell which; adding them as segments broke a probe item, so that call is left to the planner.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

ROUTES = ("structured", "retrieve", "hybrid", "refuse")
TICKERS = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "AVGO")

# Name or owned brand -> ticker. Brands are included because people ask "how is AWS doing", not
# "how is Amazon's AWS segment doing".
ALIASES = {
    "apple": "AAPL", "aapl": "AAPL", "iphone": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT", "azure": "MSFT",
    "alphabet": "GOOGL", "google": "GOOGL", "googl": "GOOGL", "goog": "GOOGL", "youtube": "GOOGL",
    "amazon": "AMZN", "amzn": "AMZN", "aws": "AMZN",
    "meta": "META", "meta platforms": "META", "facebook": "META", "instagram": "META", "whatsapp": "META",
    "nvidia": "NVDA", "nvda": "NVDA",
    "oracle": "ORCL", "orcl": "ORCL",
    "broadcom": "AVGO", "avgo": "AVGO", "vmware": "AVGO",
}

# Companies people plausibly ask about that are NOT in the corpus. A finite list is the rules' known
# weakness (an unlisted company is invisible to them); catching the rest is the planner's job.
UNCOVERED = (
    "tesla", "samsung", "intel", "amd", "ibm", "netflix", "salesforce", "adobe", "qualcomm", "cisco",
    "openai", "anthropic", "uber", "walmart", "berkshire", "jpmorgan", "tsmc", "sony", "dell", "hp inc",
    "hewlett packard", "hewlett-packard", "micron", "arm holdings", "palantir", "snowflake", "spotify", "twitter", "xiaomi", "huawei", "baidu",
    "alibaba", "tencent", "sap se", "servicenow", "shopify", "paypal", "visa inc", "mastercard",
)

_W = r"\b{}\b"


def _has(text: str, patterns) -> Optional[str]:
    for p in patterns:
        if re.search(p, text):
            return p
    return None


# --- refusal triggers ---------------------------------------------------------------------------
REAL_TIME = (r"\b(stock|share) price\b", r"\btrading at\b", r"\bright now\b", r"\btoday\b",
             r"\bthis (week|month)\b", r"\blast week\b", r"\byesterday\b", r"\bweather\b",
             r"\bmarket (cap|capitalization)\b")
ADVICE = (r"\bshould (i|we|you)\b.*\b(buy|sell|invest|hold|short)\b", r"\b(good|bad|smart) (buy|investment)\b",
          r"\bworth (buying|investing)\b", r"\b(should|could|would) (i|we) invest in\b",
          r"\b(i|we) (want|plan|am planning|are planning) to invest in\b", r"\b(buy|sell) (rating|recommendation)\b")
FORECAST = (r"\bwhat will\b", r"\bwill be\b", r"\bnext (fiscal )?(year|quarter)\b", r"\b(predict|forecast)\b")
NOT_IN_FILINGS = (r"\binterview\b", r"\b(earnings|conference) call\b", r"\bpress release\b", r"\bpodcast\b",
                  r"\btweet(ed|s)?\b", r"\bkeynote\b", r"\bsource code\b", r"\blogo\b",
                  r"\bwhat colou?r\b", r"\binternal (architecture|roadmap|memo|documents?)\b")
COMPARE = (r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b", r"\bthan\b", r"\bbetween\b")

# --- content cues -------------------------------------------------------------------------------
TEXT_CUES = (r"\bsay(s)?\b", r"\bsaid\b", r"\bdescribe[sd]?\b", r"\bdiscuss(es|ed)?\b", r"\bcite[sd]?\b",
             r"\battribute[sd]?\b", r"\bidentif(y|ies|ied)\b", r"\bexplain(s|ed)?\b", r"\bwhy\b",
             r"\bdriver(s)?\b", r"\bdr(ive|ove|iven)\b", r"\bfactor(s)?\b", r"\breason(s)?\b", r"\brisk(s)?\b",
             r"\bmention(s|ed)?\b", r"\baccording to\b", r"\bmd&a\b", r"\bstrateg(y|ies)\b",
             r"\bwhat caused\b", r"\bcontribut(e|ed|ing)\b", r"\bguidance\b", r"\boutlook\b")
TOTAL_METRIC = (r"\brevenues?\b", r"\bnet revenues?\b", r"\b(net )?sales\b", r"\bnet income\b", r"\bnet (profit|loss)\b",
                r"\bprofits?\b", r"\bearnings\b", r"\btop line\b", r"\bbottom line\b", r"\bhow much (did|does) \w+ (earn|make)\b")
# Metrics the facts table does not hold: the filing text may state them, so they go to retrieval.
UNSUPPORTED_METRIC = (r"\bsales and marketing\b", r"\bselling, general\b", r"\bgeneral and administrative\b",
                      r"\bgross (margin|profit)\b", r"\boperating (income|margin|loss|expenses?)\b",
                      r"\bearnings per share\b", r"\beps\b", r"\bfree cash flow\b", r"\bcash flow\b",
                      r"\bcapital expenditures?\b", r"\bcapex\b", r"\br&d\b", r"\bresearch and development\b",
                      r"\bheadcount\b", r"\bemployees\b", r"\bbacklog\b", r"\bremaining performance obligations?\b",
                      r"\bdividends?\b", r"\bbuybacks?\b", r"\brepurchases?\b", r"\bdebt\b", r"\bcash\b")
SEGMENT = (r"\bsegments?\b", r"\bservices\b", r"\biphone\b", r"\bmac\b", r"\bipad\b", r"\bwearables\b",
           r"\bcloud\b", r"\bazure\b", r"\bdata center\b", r"\bgaming\b", r"\baws\b", r"\badvertising\b",
           r"\bads\b", r"\breality labs\b", r"\bfamily of apps\b", r"\bsubscriptions?\b", r"\bsemiconductor\b",
           r"\binfrastructure software\b", r"\blicen[cs]e\b", r"\bhardware\b", r"\bnorth america\b",
           r"\binternational\b", r"\bproductivity\b", r"\bpersonal computing\b", r"\bsearch\b", r"\byoutube\b",
           r"\bautomotive\b", r"\bproduct line\b", r"\bgeograph(y|ic|ical)\b", r"\bregion(s|al)?\b")
HYBRID_JOIN = (r"\band (why|what|how)\b", r"\band explain\b", r"\balong with\b", r"\band the reasons?\b",
               r"\band what (drove|caused)\b")
CORPUS_WIDE = (r"\bwhich (company|companies|of the (companies|eight|8))\b", r"\bamong (the|these|all)\b",
               r"\ball (eight|8) companies\b", r"\beach company\b", r"\bevery company\b", r"\bthe eight companies\b")


def find_tickers(question: str) -> List[str]:
    """Covered tickers mentioned in the question, in order of first mention, de-duplicated."""
    q = question.lower()
    hits: List[Tuple[int, str]] = []
    for alias, ticker in ALIASES.items():
        for m in re.finditer(_W.format(re.escape(alias)), q):
            hits.append((m.start(), ticker))
    out: List[str] = []
    for _, t in sorted(hits):
        if t not in out:
            out.append(t)
    return out


def find_uncovered(question: str) -> List[str]:
    q = question.lower()
    return [c for c in UNCOVERED if re.search(_W.format(re.escape(c)), q)]


@dataclass
class RuleVerdict:
    route: str
    tickers: List[str]
    refuse_reason: Optional[str] = None
    hard: bool = False            # True only for refusals the guard enforces over the planner
    fired: List[str] = field(default_factory=list)

    def as_dict(self):
        return {"route": self.route, "tickers": self.tickers, "refuse_reason": self.refuse_reason,
                "hard": self.hard, "fired": self.fired}


def hard_refusal(question: str) -> Optional[Tuple[str, str]]:
    """(reason, pattern) when the question is unanswerable from the corpus by a precise trigger, else None.
    Precision over recall: a false positive here refuses a good question and the planner cannot undo it."""
    q = question.lower()
    text_cue = _has(q, TEXT_CUES)
    for reason, pats in (("real_time", REAL_TIME), ("advice", ADVICE), ("not_in_filings", NOT_IN_FILINGS)):
        p = _has(q, pats)
        if p:
            return reason, p
    p = _has(q, FORECAST)
    if p and not text_cue:          # "what does the MD&A say about its outlook" is a text question
        return "forecast", p
    unc = find_uncovered(question)
    if unc:
        covered = find_tickers(question)
        if not covered:
            return "uncovered_company", unc[0]
        cmp = _has(q, COMPARE)
        if cmp:                     # comparing a covered company with one outside the corpus
            return "uncovered_company", f"{unc[0]} + {cmp}"
    return None


def classify(question: str) -> RuleVerdict:
    """Rules-only routing decision."""
    q = question.lower()
    tickers = find_tickers(question)
    hr = hard_refusal(question)
    if hr:
        return RuleVerdict("refuse", tickers, hr[0], True, [hr[1]])
    corpus_wide = _has(q, CORPUS_WIDE)
    if not tickers and not corpus_wide:
        return RuleVerdict("refuse", [], "no_covered_company", False, ["no covered company named"])
    if corpus_wide and not tickers:
        tickers = list(TICKERS)

    text_cue = _has(q, TEXT_CUES)
    total_metric = _has(q, TOTAL_METRIC)
    unsupported = _has(q, UNSUPPORTED_METRIC)
    segment = _has(q, SEGMENT)
    structured = bool(total_metric) and not unsupported and not segment
    fired = [p for p in (text_cue, total_metric, unsupported, segment) if p]

    if structured and text_cue:
        join = _has(q, HYBRID_JOIN)
        if join:
            return RuleVerdict("hybrid", tickers, fired=fired + [join])
        return RuleVerdict("retrieve", tickers, fired=fired)   # "what drove revenue growth" is a text question
    if structured:
        return RuleVerdict("structured", tickers, fired=fired)
    return RuleVerdict("retrieve", tickers, fired=fired or ["default: filing text"])
