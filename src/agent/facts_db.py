"""
SQLite facts table built from the XBRL CSVs -- Phase B, Day 8 (SQL tool).

WHY THE TABLE IS NOT A STRAIGHT CSV LOAD
Profiling data/processed/xbrl (375 rows, revenue and net_income only) found four traps that a
naive `WHERE fy = 2024 AND fp = 'FY'` would walk straight into:

  1. The raw `fy` label is the fiscal year of the FILING, not of the period. AVGO's three annual
     net-income rows (periods ending 2022, 2023, 2024) are all labelled fy=2024. So fiscal years
     are identified here by PERIOD END DATE, never by the raw label. Raw labels are kept
     (fy_label, fp_label) for provenance only.
  2. `fp` and `form` do not say what kind of period a row covers. 12 rows are 10-Q rows whose
     period is ~365 days (AMZN trailing-twelve-month figures) and Q2/Q3 10-Qs carry year-to-date
     rows next to the quarter rows (AAPL fiscal 2023 Q2: 211,990M is six months, 94,836M is the
     quarter). `basis` is therefore derived from the period LENGTH:
        quarter      < 110 days
        ytd          110-299 days (6- or 9-month cumulative, from a 10-Q)
        fiscal_year  >= 300 days AND form == 10-K
        ttm          >= 300 days AND form == 10-Q   (a trailing twelve months, NOT a fiscal year)
  3. Q4 is never reported in a 10-Q, so "the last four quarters" is not all in the table. Q4 can
     be derived as (10-K annual figure) - (9-month YTD figure of the same fiscal year); that is
     exposed as `derive_q4` and always flagged `derived=True`, never mixed in silently.
  4. Coverage is uneven (AVGO has 3 net-income rows against 24 revenue rows). Lookups return
     what exists and say how many rows came back; they never fill a gap.

The query surface is deliberately small and typed (`fiscal_years`, `quarters`, `derive_q4`) because
the routing LLM is better at choosing among a few safe tools than at writing correct SQL against a
schema full of the traps above. `run_select` is the escape hatch for questions the typed calls do
not cover, and it is read-only by construction (read-only connection, SELECT-only authorizer,
single statement, length cap, step limit, row cap, recursive CTEs denied, and a denylist of SQL
functions that can allocate unbounded memory in one call, e.g. randomblob/zeroblob/printf).
"""
import csv
import datetime as dt
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

METRICS = ("revenue", "net_income")
QUARTER_MAX_DAYS = 110     # < this -> a single quarter (13 weeks ~ 91 days)
ANNUAL_MIN_DAYS = 300      # >= this -> 12-month period (52/53 weeks, ~364-371 days)
CONSECUTIVE_QUARTER_DAYS = (80, 105)  # gap between consecutive quarter end dates
MAX_SQL_CHARS = 2000
# SQL functions that can allocate unbounded memory in a single call (or amplify when nested)
DENIED_SQL_FUNCTIONS = {"randomblob", "zeroblob", "printf", "format", "replace", "hex", "unhex",
                        "char", "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}

SCHEMA = """
CREATE TABLE facts (
    ticker       TEXT NOT NULL,
    metric       TEXT NOT NULL,
    tag_used     TEXT NOT NULL,
    form         TEXT NOT NULL,
    start_date   TEXT NOT NULL,
    end_date     TEXT NOT NULL,
    period_days  INTEGER NOT NULL,
    basis        TEXT NOT NULL CHECK (basis IN ('quarter','ytd','fiscal_year','ttm')),
    val          INTEGER NOT NULL,          -- whole US dollars, as reported
    fy_label     TEXT,                      -- raw XBRL label; NOT reliable, provenance only
    fp_label     TEXT,
    UNIQUE (ticker, metric, start_date, end_date)
);
CREATE INDEX idx_facts_lookup ON facts (ticker, metric, basis, end_date);
"""


def classify_basis(start: str, end: str, form: str) -> Tuple[int, str]:
    days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days
    if days < QUARTER_MAX_DAYS:
        return days, "quarter"
    if days < ANNUAL_MIN_DAYS:
        return days, "ytd"
    return days, ("fiscal_year" if form == "10-K" else "ttm")


def build_db(csv_dir: str, db_path: str) -> Dict:
    """(Re)build the SQLite file from `*_facts.csv`. Idempotent. Raises on a conflicting duplicate
    (same ticker/metric/period, different value) rather than letting one silently win."""
    csv_dir_p, db_p = Path(csv_dir), Path(db_path)
    files = sorted(csv_dir_p.glob("*_facts.csv"))
    if not files:
        raise FileNotFoundError(f"no *_facts.csv files in {csv_dir_p}")
    rows, seen = [], {}
    for f in files:
        with open(f, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["metric"] not in METRICS:
                    raise ValueError(f"{f.name}: unexpected metric {r['metric']!r}")
                days, basis = classify_basis(r["start"], r["end"], r["form"])
                if days <= 0:
                    raise ValueError(f"{f.name}: non-positive period {r['start']}..{r['end']}")
                val = int(r["val"])  # raises on non-integer: all values are whole dollars
                key = (r["ticker"], r["metric"], r["start"], r["end"])
                if key in seen:
                    if seen[key] != val:
                        raise ValueError(f"conflicting duplicate for {key}: {seen[key]} vs {val}")
                    continue
                seen[key] = val
                rows.append((r["ticker"], r["metric"], r["tag_used"], r["form"], r["start"], r["end"],
                             days, basis, val, r["fy"], r["fp"]))
    db_p.parent.mkdir(parents=True, exist_ok=True)
    # Remove the old file AND any stale SQLite sidecars: a leftover -journal next to a freshly created
    # database can be mistaken for a hot journal and "rolled back" onto it.
    for stale in (db_p, Path(str(db_p) + "-journal"), Path(str(db_p) + "-wal"), Path(str(db_p) + "-shm")):
        if stale.exists():
            stale.unlink()
    con = sqlite3.connect(str(db_p))
    try:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
        counts = dict(con.execute("SELECT basis, COUNT(*) FROM facts GROUP BY basis").fetchall())
    finally:
        con.close()
    return {"rows": len(rows), "by_basis": counts, "files": len(files), "db_path": str(db_p)}


@dataclass(frozen=True)
class Fact:
    ticker: str
    metric: str
    value: int                 # whole US dollars
    basis: str                 # fiscal_year | quarter | ytd | ttm | quarter_derived
    start_date: str
    end_date: str
    form: str
    tag_used: str
    fy_label: Optional[str] = None
    fp_label: Optional[str] = None
    derived: bool = False
    note: str = ""

    def citation(self) -> str:
        label = {"fiscal_year": "fiscal year", "quarter": "quarter", "ytd": "year-to-date",
                 "ttm": "trailing twelve months", "quarter_derived": "quarter (derived)"}[self.basis]
        extra = f" [{self.note}]" if self.note else ""
        return (f"XBRL {self.ticker} {self.metric.replace('_', ' ')}, {label} ended {self.end_date} "
                f"({self.start_date} to {self.end_date}, {self.form}, tag {self.tag_used}){extra}")

    def as_dict(self) -> Dict:
        return {**asdict(self), "citation": self.citation()}


STALE_YEARS = 0.8                 # the other metric reaches a fiscal year this much later -> this series stopped early
CONSECUTIVE_YEARS = (0.9, 1.1)    # two "consecutive" fiscal years are this many years apart


def _years_apart(a: str, b: str) -> float:
    return (dt.date.fromisoformat(a) - dt.date.fromisoformat(b)).days / 365.25


@dataclass(frozen=True)
class Lookup:
    """What a status-bearing lookup returns. `status` is the contract the agent relies on:
      ok            the facts answer the question as asked
      caveat        they answer it, but `caveats` must be shown to the reader (verbatim)
      unanswerable  they cannot; `reason` says why and the agent relays it (never retries around it)
    `facts` may be non-empty even when unanswerable, so the refusal can show what was found."""
    status: str
    facts: Tuple[Fact, ...] = ()
    caveats: Tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> Dict:
        return {"status": self.status, "facts": [f.as_dict() for f in self.facts],
                "caveats": list(self.caveats), "reason": self.reason}


def _lookup(facts, caveats=(), reason="") -> "Lookup":
    if reason:
        return Lookup("unanswerable", tuple(facts), tuple(caveats), reason)
    return Lookup("caveat" if caveats else "ok", tuple(facts), tuple(caveats))


class FactsDB:
    def __init__(self, db_path: str):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"{db_path} not found; build it with `python -m scripts.build_facts_db`")
        self.db_path = str(db_path)
        self._con = sqlite3.connect(self.db_path)
        self._con.row_factory = sqlite3.Row

    def close(self):
        self._con.close()

    # ------------------------------------------------------------------ typed lookups
    @staticmethod
    def _check(ticker: str, metric: str):
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; available: {', '.join(METRICS)}")
        if not ticker or not ticker.isalpha():
            raise ValueError(f"bad ticker {ticker!r}")

    def _row_to_fact(self, r, basis=None, derived=False, note="") -> Fact:
        return Fact(ticker=r["ticker"], metric=r["metric"], value=r["val"], basis=basis or r["basis"],
                    start_date=r["start_date"], end_date=r["end_date"], form=r["form"],
                    tag_used=r["tag_used"], fy_label=r["fy_label"], fp_label=r["fp_label"],
                    derived=derived, note=note)

    def fiscal_years(self, ticker: str, metric: str, n: int = 2) -> List[Fact]:
        """The n most recent FISCAL YEARS (10-K annual rows), newest first, identified by period
        end date. Returns fewer than n if the table has fewer; the caller must check len()."""
        self._check(ticker, metric)
        cur = self._con.execute(
            "SELECT * FROM facts WHERE ticker=? AND metric=? AND basis='fiscal_year' "
            "ORDER BY end_date DESC LIMIT ?", (ticker.upper(), metric, int(n)))
        return [self._row_to_fact(r) for r in cur]

    def quarters(self, ticker: str, metric: str, n: int = 4) -> Dict:
        """The n most recent single-quarter rows from 10-Qs, newest first, plus whether they are
        consecutive. Q4 is never in a 10-Q, so a window that crosses a fiscal-year end will NOT be
        consecutive; use `recent_quarters` to fill it with derived Q4s explicitly."""
        self._check(ticker, metric)
        cur = self._con.execute(
            "SELECT * FROM facts WHERE ticker=? AND metric=? AND basis='quarter' AND form='10-Q' "
            "ORDER BY end_date DESC LIMIT ?", (ticker.upper(), metric, int(n)))
        facts = [self._row_to_fact(r) for r in cur]
        return {"facts": facts, **_consecutiveness(facts)}

    def derive_q4(self, ticker: str, metric: str, fiscal_year_end: str) -> Optional[Fact]:
        """Q4 = annual (10-K) minus the 9-month YTD row that starts on the same date as the annual
        row. Returns None (never a guess) if either side is missing."""
        self._check(ticker, metric)
        fy = self._con.execute(
            "SELECT * FROM facts WHERE ticker=? AND metric=? AND basis='fiscal_year' AND end_date=?",
            (ticker.upper(), metric, fiscal_year_end)).fetchone()
        if fy is None:
            return None
        ytd = self._con.execute(
            "SELECT * FROM facts WHERE ticker=? AND metric=? AND basis='ytd' AND start_date=? "
            "AND period_days BETWEEN 240 AND 290 ORDER BY end_date DESC LIMIT 1",
            (ticker.upper(), metric, fy["start_date"])).fetchone()
        if ytd is None:
            return None
        q4_start = (dt.date.fromisoformat(ytd["end_date"]) + dt.timedelta(days=1)).isoformat()
        return Fact(ticker=fy["ticker"], metric=metric, value=fy["val"] - ytd["val"],
                    basis="quarter_derived", start_date=q4_start, end_date=fy["end_date"],
                    form="10-K minus 10-Q", tag_used=fy["tag_used"], derived=True,
                    note=f"Q4 derived = fiscal year {fy['val']:,} - nine-month YTD {ytd['val']:,}")

    def recent_quarters(self, ticker: str, metric: str, n: int = 4) -> Dict:
        """The n most recent consecutive quarters, filling fiscal-year-end gaps with derived Q4s.
        Every derived quarter is flagged. `complete` is False if fewer than n could be assembled
        (the caller must then say so instead of averaging fewer quarters)."""
        self._check(ticker, metric)
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        reported = self.quarters(ticker, metric, n=40)["facts"]
        derived = []
        for fy in self.fiscal_years(ticker, metric, n=10):
            q4 = self.derive_q4(ticker, metric, fy.end_date)
            if q4 is not None:
                derived.append(q4)
        pool = sorted(reported + derived, key=lambda f: f.end_date, reverse=True)
        window = []
        for f in pool:
            if window:
                gap = (dt.date.fromisoformat(window[-1].end_date) - dt.date.fromisoformat(f.end_date)).days
                if not CONSECUTIVE_QUARTER_DAYS[0] <= gap <= CONSECUTIVE_QUARTER_DAYS[1]:
                    break  # a hole: stop rather than skip over it
            window.append(f)
            if len(window) == n:   # checked after EVERY append, including the first (n=1)
                break
        return {"facts": window, "complete": len(window) == n,
                "n_derived": sum(f.derived for f in window), "requested": n}

    # ------------------------------------------------------------------ status-bearing lookups (the agent's surface)
    def _staleness_caveat(self, ticker: str, metric: str, latest_end: str, what: str) -> Optional[str]:
        """If the company's OTHER metric reaches a later fiscal year than this one, this series stopped
        early (a coverage gap upstream) and "the latest fiscal year in the dataset" is not the company's
        latest year. AVGO net income ends FY2024 while AVGO revenue runs through FY2025."""
        other = "revenue" if metric == "net_income" else "net_income"
        other_fy = self.fiscal_years(ticker, other, n=1)
        if other_fy and _years_apart(other_fy[0].end_date, latest_end) > STALE_YEARS:
            return (f"{metric} data ends at the fiscal year ended {latest_end}, but {other} for {ticker} "
                    f"runs through {other_fy[0].end_date}: this series is stale (a gap in the XBRL extraction), "
                    f"so the result is not the company's latest {what}")
        return None

    def latest_fiscal_year(self, ticker: str, metric: str) -> Lookup:
        fys = self.fiscal_years(ticker, metric, n=1)
        if not fys:
            return _lookup([], reason=f"no fiscal-year row for {ticker.upper()} {metric}")
        c = self._staleness_caveat(ticker.upper(), metric, fys[0].end_date, "fiscal year")
        return _lookup(fys, [c] if c else [])

    def fiscal_year_pair(self, ticker: str, metric: str) -> Lookup:
        """facts = (newest, previous). Consecutive and not stale, or a caveat says otherwise."""
        fys = self.fiscal_years(ticker, metric, n=2)
        if len(fys) < 2:
            return _lookup(fys, reason=f"only {len(fys)} fiscal-year row(s) for {ticker.upper()} {metric}")
        caveats = []
        c = self._staleness_caveat(ticker.upper(), metric, fys[0].end_date, "year-over-year change")
        if c:
            caveats.append(c)
        gap = _years_apart(fys[0].end_date, fys[1].end_date)
        if not CONSECUTIVE_YEARS[0] <= gap <= CONSECUTIVE_YEARS[1]:
            caveats.append(f"the two fiscal years are {gap:.2f} years apart, not consecutive")
        return _lookup(fys, caveats)

    def same_year_pair(self, ticker: str, first_metric: str, second_metric: str) -> Lookup:
        """facts = (first_metric, second_metric) for the latest fiscal year of each. They must share a
        fiscal-year end date: a margin or ratio built from two different years is a wrong number."""
        a = self.fiscal_years(ticker, first_metric, n=1)
        b = self.fiscal_years(ticker, second_metric, n=1)
        found = a + b
        if not a or not b:
            return _lookup(found, reason=f"missing a fiscal-year row for {ticker.upper()} "
                                         f"{first_metric if not a else second_metric}")
        if a[0].end_date != b[0].end_date:
            return _lookup(found, reason=f"{first_metric} and {second_metric} do not share the same most recent "
                                         f"fiscal year ({a[0].end_date} vs {b[0].end_date})")
        return _lookup(found)

    def quarter_window(self, ticker: str, metric: str, n: int = 4) -> Lookup:
        """facts = the n most recent consecutive quarters, newest first. Derived Q4s are allowed but make
        the result a caveat; a hole makes it unanswerable (never average fewer quarters than asked)."""
        rq = self.recent_quarters(ticker, metric, n)
        if not rq["complete"]:
            return _lookup(rq["facts"], reason=f"only {len(rq['facts'])} of {n} consecutive quarters "
                                               f"could be assembled")
        cav = ([f"{rq['n_derived']} of {n} quarters are derived Q4 (annual minus nine-month YTD)"]
               if rq["n_derived"] else [])
        return _lookup(rq["facts"], cav)

    # ------------------------------------------------------------------ read-only SQL escape hatch
    def run_select(self, sql: str, max_rows: int = 200, max_steps: int = 2_000_000) -> Dict:
        """Execute ONE read-only SELECT. Enforced by the database, not by string matching: a
        read-only connection, an authorizer that only permits SELECT/READ/FUNCTION, a statement
        count check, a VM step limit, a query-length cap, and a denylist of memory-unbounded functions
        (recursive CTEs are denied outright, not merely step-limited)."""
        text = sql.strip().rstrip(";").strip()
        if len(text) > MAX_SQL_CHARS:
            raise ValueError(f"query is too long (max {MAX_SQL_CHARS} characters)")
        if not text or ";" in text:
            raise ValueError("exactly one statement is allowed (a ';' inside a string literal is also rejected)")
        if not text.lower().startswith(("select", "with")):
            raise ValueError("only SELECT statements are allowed")
        uri = Path(self.db_path).resolve().as_uri() + "?mode=ro"
        con = sqlite3.connect(uri, uri=True)
        try:
            con.execute("PRAGMA query_only = ON")
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}

            def authorize(action, a, b, c, d):
                if action not in allowed:
                    return sqlite3.SQLITE_DENY      # also denies recursive CTEs (SQLITE_RECURSIVE)
                # one function call is one VM step, so the step limit cannot bound the memory these
                # can allocate (randomblob(1e9) was OOM-killed in the audit); deny them by name
                if action == sqlite3.SQLITE_FUNCTION and (b or "").lower() in DENIED_SQL_FUNCTIONS:
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            con.set_authorizer(authorize)
            steps = {"n": 0}

            def progress():
                steps["n"] += 1000
                return 1 if steps["n"] > max_steps else 0
            con.set_progress_handler(progress, 1000)
            cur = con.execute(text)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(max_rows + 1)
        except sqlite3.Error as e:
            raise ValueError(f"SQL rejected or failed: {e}") from e
        finally:
            con.close()
        return {"columns": cols, "rows": [list(r) for r in rows[:max_rows]], "truncated": len(rows) > max_rows}


def _consecutiveness(facts: List[Fact]) -> Dict:
    gaps = []
    for newer, older in zip(facts, facts[1:]):
        g = (dt.date.fromisoformat(newer.end_date) - dt.date.fromisoformat(older.end_date)).days
        if not (CONSECUTIVE_QUARTER_DAYS[0] <= g <= CONSECUTIVE_QUARTER_DAYS[1]):
            gaps.append((older.end_date, newer.end_date, g))
    return {"consecutive": not gaps, "gaps": gaps}
