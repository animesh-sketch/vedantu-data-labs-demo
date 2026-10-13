"""AI Analytics Copilot.

Deterministic natural-language -> intent -> validated SQL -> computed answer.
Every answer carries the SQL, the result table, assumptions and limitations.
An optional LLM may only *propose SQL*; it is validated and executed like any other
query, so numbers always come from the data, never from the model.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace

import pandas as pd

from . import sql_guard
from .profiling import Profile

RATE_PAT = r"\b(conversion|convert\w*|win rate|won rate|success rate|resolution rate|completion rate|rate)\b"
AGG_WORDS = [
    ("avg", r"\b(average|avg|mean|typical)\b"),
    ("count", r"\b(how many|count|number of|volume)\b"),
    ("median", r"\bmedian\b"),
    ("sum", r"\b(total|sum|overall)\b"),
    ("max", r"\b(maximum|max|largest|biggest)\b"),
    ("min", r"\b(minimum|min|smallest)\b"),
]
SYNONYMS = {
    "revenue": ["revenue", "amount", "deal_value", "value", "fee", "price"],
    "sales": ["revenue", "amount", "deal_value", "value"],
    "score": ["score", "marks", "rating", "csat"],
    "time": ["resolution_hours", "hours", "minutes", "duration"],
}
GRAINS = {
    "day": r"\b(daily|per day|by day)\b",
    "week": r"\b(weekly|per week|by week)\b",
    "month": r"\b(monthly|per month|by month|over time|trend|trending)\b",
    "quarter": r"\b(quarterly|by quarter)\b",
    "year": r"\b(yearly|annual|by year)\b",
}
FOLLOW_UP = r"^\s*(and|what about|how about|only|now|same|instead|also|break it|split it|filter)\b"
AGG_LABEL = {"rate": "rate", "sum": "total", "avg": "average", "count": "count", "median": "median", "max": "maximum", "min": "minimum"}


@dataclass
class Intent:
    agg: str = "sum"
    metric: str | None = None
    group: str | None = None
    grain: str | None = None
    date_col: str | None = None
    filters: dict[str, str] = field(default_factory=dict)
    top_n: int | None = None
    order: str | None = None  # desc | asc
    kind: str = "aggregate"  # aggregate | summary | correlation
    other: str | None = None  # second column for correlation
    rate_col: str | None = None  # for agg == "rate": share of rows where rate_col == rate_val
    rate_val: str | None = None


@dataclass
class Answer:
    text: str
    sql: str | None = None
    table: pd.DataFrame | None = None
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    intent: Intent | None = None
    source: str = "deterministic"
    ok: bool = True


def _q(c: str) -> str:
    return '"' + c.replace('"', '""') + '"'


def _lit(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _variants(col: str) -> list[str]:
    base = col.lower().replace("_", " ")
    out = {base, base.rstrip("s"), base + "s"}
    return sorted(out, key=len, reverse=True)


def find_columns(q: str, candidates: list[str]) -> list[str]:
    """Columns mentioned in the question, in order of appearance; synonyms as fallback."""
    hits: list[tuple[int, str]] = []
    for c in candidates:
        for v in _variants(c):
            m = re.search(rf"\b{re.escape(v)}\b", q)
            if m and len(v) > 2:
                hits.append((m.start(), c))
                break
    if not hits:  # a distinctive word of one column name, e.g. "satisfaction" -> satisfaction_score
        for c in candidates:
            for tok in re.split(r"[_\s]+", c.lower()):
                owners = [o for o in candidates if tok in re.split(r"[_\s]+", o.lower())]
                if len(tok) > 3 and tok not in ("date", "rate") and len(owners) == 1 and (m := re.search(rf"\b{re.escape(tok)}\b", q)):
                    hits.append((m.start(), c))
                    break
    if not hits:
        for key, words in SYNONYMS.items():
            if m := re.search(rf"\b{key}\b", q):
                for w in words:
                    match = [c for c in candidates if w in c.lower()]
                    if match:
                        hits.append((m.start(), match[0]))
                        break
    seen, out = set(), []
    for _, c in sorted(hits):
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def find_filters(q: str, df: pd.DataFrame, prof: Profile) -> dict[str, str]:
    found: dict[str, str] = {}
    for c in prof.categorical:
        vals = df[c].dropna().astype(str).unique()
        for v in sorted(vals, key=len, reverse=True):
            if len(v) >= 2 and re.search(rf"(?<!\w){re.escape(v.lower())}(?!\w)", q):
                found[c] = v
                break
    return found


def parse(question: str, df: pd.DataFrame, prof: Profile, prev: Intent | None = None) -> Intent | None:
    q = " " + question.lower().strip().rstrip("?").replace("_", " ") + " "
    is_follow = prev is not None and prev.kind == "aggregate" and bool(re.match(FOLLOW_UP, q))
    it = replace(prev, filters=dict(prev.filters)) if is_follow else Intent()

    if not is_follow and re.search(r"\b(summary|summarise|summarize|describe|overview)\b", q):
        return Intent(kind="summary")
    if re.search(r"\bcorrelat\w*\b", q):
        cols = find_columns(q, prof.numeric)
        if len(cols) >= 2:
            return Intent(kind="correlation", metric=cols[0], other=cols[1])

    explicit_agg = False
    for agg, pat in AGG_WORDS:
        if re.search(pat, q):
            it.agg, explicit_agg = agg, True
            break
    if m := re.search(r"\b(top|bottom|best|worst|highest|lowest)\s*(\d+)?\b", q):
        it.top_n = int(m.group(2)) if m.group(2) else (1 if re.search(r"\b(which|what|who)\b", q) else 5)
        it.order = "asc" if m.group(1) in ("bottom", "worst", "lowest") else "desc"
    elif re.search(r"\b(which|what)\b.*\b(most|highest|best|largest)\b", q):
        it.top_n, it.order = 1, "desc"
    elif re.search(r"\b(which|what)\b.*\b(least|lowest|worst|smallest|fewest)\b", q):
        it.top_n, it.order = 1, "asc"

    for grain, pat in GRAINS.items():
        if prof.datetime and re.search(pat, q):
            it.grain, it.group, it.date_col = grain, None, prof.datetime[0]
            break

    if re.search(RATE_PAT, q):
        from .dashboard import outcome_column
        oc = outcome_column(df, prof)
        if oc:
            it.agg, it.rate_col, it.rate_val, explicit_agg = "rate", oc[0], oc[1], True
    metrics = [] if it.agg == "rate" else find_columns(q, prof.numeric)
    if metrics:
        it.metric = metrics[0]
        if it.agg == "count" and not is_follow:
            it.agg = "sum"  # "number of X" where X is numeric -> sum is ambiguous; prefer explicit sum
            if re.search(r"\b(how many|count|number of)\b", q):
                it.agg = "count"
    elif not is_follow and it.agg != "rate" and (it.agg == "count" or not explicit_agg):
        it.agg = "count"

    if m := re.search(r"\b(?:by|per|each|across|for each|split by|down by)\s+([a-z_ ]+)", q):
        g = find_columns(" " + m.group(1) + " ", prof.categorical)
        if g:
            it.group, it.grain = g[0], None
    if it.group is None and it.grain is None and it.top_n:
        g = [c for c in find_columns(q, prof.categorical) if c != it.metric]
        if g:
            it.group = g[0]

    new_filters = find_filters(q, df, prof)
    if it.group:
        new_filters.pop(it.group, None)
    if is_follow:
        if re.search(r"\b(all|everything|overall|remove filter|no filter)\b", q):
            it.filters = {}
        it.filters.update(new_filters)
    else:
        it.filters = new_filters

    if it.group == it.rate_col:
        it.group = None
    if it.agg not in ("count", "rate") and it.metric is None:
        it.agg = "count"
    if not is_follow and not any([metrics, it.group, it.grain, it.filters, explicit_agg]):
        return None
    return it


def build_sql(it: Intent) -> str:
    if it.kind == "correlation":
        return (f"SELECT corr({_q(it.metric)}, {_q(it.other)}) AS correlation, count(*) AS n FROM data "
                f"WHERE {_q(it.metric)} IS NOT NULL AND {_q(it.other)} IS NOT NULL")
    if it.agg == "rate":
        measure = f"100.0 * avg(CASE WHEN {_q(it.rate_col)} = {_lit(it.rate_val)} THEN 1 ELSE 0 END)"
        alias = f"pct_{it.rate_val.lower().replace(' ', '_')}"
    elif it.agg == "count" or not it.metric:
        measure, alias = "count(*)", "count"
    else:
        measure, alias = f"{it.agg}({_q(it.metric)})", f"{it.agg}_{it.metric}"
    where = [f"{_q(c)} = {_lit(v)}" for c, v in it.filters.items()]
    if it.agg == "rate":
        where.append(f"{_q(it.rate_col)} IS NOT NULL")
    if it.agg not in ("count", "rate") and it.metric:
        where.append(f"{_q(it.metric)} IS NOT NULL")
    if it.grain:
        where.append(f"{_q(it.date_col)} IS NOT NULL")
    where_sql = f" WHERE {' AND '.join(where)}" if where else ""
    if it.grain:
        return (f"SELECT date_trunc('{it.grain}', {_q(it.date_col)}) AS period, round({measure}, 2) AS {_q(alias)} "
                f"FROM data{where_sql} GROUP BY 1 ORDER BY 1")
    if it.group:
        lim = f" LIMIT {it.top_n}" if it.top_n else ""
        return (f"SELECT {_q(it.group)}, round({measure}, 2) AS {_q(alias)}, count(*) AS rows_used FROM data{where_sql} "
                f"GROUP BY 1 ORDER BY 2 {'ASC' if it.order == 'asc' else 'DESC'}{lim}")
    return f"SELECT round({measure}, 2) AS {_q(alias)}, count(*) AS rows_used FROM data{where_sql}"


def _fmt(v) -> str:
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    try:
        f = float(v)
        return f"{int(f):,}" if f.is_integer() else f"{f:,.2f}"
    except (TypeError, ValueError):
        return str(v)


def _narrate(it: Intent, res: pd.DataFrame, total_rows: int) -> str:
    scope = (" where " + ", ".join(f"{c} = {v}" for c, v in it.filters.items())) if it.filters else ""
    if res.empty:
        return f"No rows matched{scope}. Nothing to report."
    if it.kind == "correlation":
        r = res.iloc[0, 0]
        if pd.isna(r):
            return "Correlation could not be computed (not enough data or variation)."
        strength = "strong" if abs(r) >= 0.6 else "moderate" if abs(r) >= 0.3 else "weak"
        return (f"Pearson correlation between **{it.metric}** and **{it.other}** is **{r:.2f}** "
                f"({strength}, {'positive' if r > 0 else 'negative'}) over {int(res.iloc[0, 1]):,} rows. "
                "Correlation does not prove causation.")
    if it.agg == "rate":
        label = f"% of rows with {it.rate_col} = {it.rate_val}"
    else:
        label = AGG_LABEL[it.agg] + (f" {it.metric}" if it.metric and it.agg != "count" else " of records")
    if it.grain:
        val = res.columns[1]
        first, last = res.iloc[0], res.iloc[-1]
        peak = res.loc[res[val].idxmax()]
        chg = (last[val] - first[val]) / first[val] * 100 if first[val] else float("nan")
        return (f"{label[0].upper() + label[1:]} by {it.grain}{scope} across {len(res)} periods: from **{_fmt(first[val])}** "
                f"({_fmt(first['period'])}) to **{_fmt(last[val])}** ({_fmt(last['period'])})"
                + (f", a {chg:+.1f}% change" if pd.notna(chg) else "")
                + f". Peak **{_fmt(peak[val])}** in {_fmt(peak['period'])}. The latest period may be incomplete.")
    if it.group:
        val = res.columns[1]
        top = res.iloc[0]
        word = "Lowest" if it.order == "asc" else "Highest"
        share = ""
        if it.agg in ("sum", "count") and not it.top_n and res[val].sum():
            share = f" ({top[val] / res[val].sum():.1%} of the total)"
        return (f"{label[0].upper() + label[1:]} by **{it.group}**{scope}: {word} is **{top[it.group]}** at "
                f"**{_fmt(top[val])}**{share}. {len(res)} group(s) shown.")
    return (f"The {label}{scope} is **{_fmt(res.iloc[0, 0])}**, computed over "
            f"{int(res.iloc[0, 1]):,} of {total_rows:,} rows.")


def summary_answer(df: pd.DataFrame, prof: Profile) -> Answer:
    desc = df[prof.numeric].describe().T.round(2).reset_index(names="column") if prof.numeric else None
    text = (f"The dataset has **{len(df):,} rows** and **{df.shape[1]} columns**: {len(prof.numeric)} numeric, "
            f"{len(prof.categorical)} categorical, {len(prof.datetime)} date.")
    if prof.datetime:
        d = prof.datetime[0]
        text += f" `{d}` spans {_fmt(df[d].min())} to {_fmt(df[d].max())}."
    return Answer(text, table=desc, assumptions=["Summary statistics exclude missing values."],
                  intent=Intent(kind="summary"))


def suggestions(prof: Profile) -> list[str]:
    m = prof.numeric[0].replace("_", " ") if prof.numeric else None
    g = prof.categorical[0].replace("_", " ") if prof.categorical else None
    s = []
    if m:
        s.append(f"What is the total {m}?")
    if m and g:
        s.append(f"Top 5 {g} by average {m}")
    if prof.datetime:
        s.append(f"Monthly trend of {m or 'records'}")
    if g:
        s.append(f"How many records by {g}?")
    s.append("Give me a summary of the data")
    return s


def ask(question: str, df: pd.DataFrame, prof: Profile, prev: Intent | None = None) -> Answer:
    it = parse(question, df, prof, prev)
    if it is None:
        llm = _llm_answer(question, df, prof)
        if llm:
            return llm
        return Answer("I couldn't map that question to columns in this dataset, so I won't guess. Try: "
                      + " · ".join(f"*{s}*" for s in suggestions(prof)), ok=False,
                      limitations=["The offline parser supports totals, averages, counts, median, min/max, "
                                   "group-by, top-N, value filters, time trends, correlation and follow-ups."])
    if it.kind == "summary":
        return summary_answer(df, prof)
    assumptions, limitations = [], []
    if it.grain:
        assumptions.append(f"Time axis uses `{it.date_col}`, truncated to {it.grain}.")
    if it.agg == "rate":
        assumptions.append(f"'Rate' means share of rows where `{it.rate_col}` = '{it.rate_val}' (in %).")
    if it.kind == "aggregate" and it.agg == "count":
        assumptions.append("Counted records (rows).")
    if it.filters:
        assumptions.append("Filters inferred from values you mentioned: "
                           + ", ".join(f"{c} = '{v}'" for c, v in it.filters.items()))
    sql = build_sql(it)
    try:
        res = sql_guard.run(sql, df)
    except Exception as e:  # noqa: BLE001 - surfaced to the user, never hidden
        return Answer(f"The query failed safely: {e}", sql=sql, ok=False, intent=it)
    if it.metric and it.kind == "aggregate" and it.agg != "count":
        nulls = int(df[it.metric].isna().sum())
        if nulls:
            limitations.append(f"{nulls:,} row(s) with missing `{it.metric}` were excluded.")
    limitations.append("Computed only from the loaded data; no external data or model estimates.")
    return Answer(_narrate(it, res, len(df)), sql=sql, table=res, assumptions=assumptions,
                  limitations=limitations, intent=it)


# ---------- optional LLM: proposes SQL only ----------
def llm_available() -> bool:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _llm_answer(question: str, df: pd.DataFrame, prof: Profile) -> Answer | None:
    if not llm_available():
        return None
    import anthropic

    schema = ", ".join(f"{c} ({t})" for c, t in prof.types.items())
    prompt = (f"Write ONE DuckDB SELECT query over table `data` with columns: {schema}.\n"
              f"Question: {question}\nReturn only the SQL, no prose or comments. If unanswerable, return NONE.")
    try:
        msg = anthropic.Anthropic().messages.create(
            model=os.getenv("VDL_LLM_MODEL", "claude-sonnet-5-5"), max_tokens=400,
            messages=[{"role": "user", "content": prompt}])
        sql = msg.content[0].text.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
        if sql.upper().startswith("NONE"):
            return None
        res = sql_guard.run(sql, df)
    except Exception as e:  # noqa: BLE001
        return Answer(f"The LLM-proposed query was rejected or failed: {e}", ok=False, source="llm")
    return Answer(f"Computed from the data with an LLM-proposed, validated read-only query ({len(res)} row(s)). "
                  "See the table for actual values.", sql=sql, table=res, source="llm",
                  assumptions=["SQL was written by an LLM and passed read-only validation; review it below."],
                  limitations=["The LLM did not see the data values and wrote no numbers in this answer."])
