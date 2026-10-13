"""Chat with your data: ask anything about the 12 tables (+ a per-learner table).

With ANTHROPIC_API_KEY set (and `anthropic` installed), Claude answers by calling one tool, `run_sql`,
which goes through the same read-only SQL guard as the Data Explorer. Claude only sees the schema and
the rows its own queries return, and is told to use numbers from those results only. Every query that
ran is shown with its result. Without a key, or if the LLM call fails, the deterministic analyst answers."""
from __future__ import annotations

import os

import pandas as pd

from .. import sql_guard
from . import analyst as X
from .model import Model

MAX_TURNS = 6
ROWS_TO_LLM = 60


def llm_available() -> bool:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def tables(m: Model) -> dict[str, pd.DataFrame]:
    return {**X.explorer_tables(m), "learners": X.student_fact(m)}


def schema(tbls: dict[str, pd.DataFrame]) -> str:
    """Column names, types and (for low-cardinality text columns) the allowed values."""
    lines = []
    for name, df in tbls.items():
        cols = []
        for c in df.columns:
            s = df[c]
            desc = f"{c} {s.dtype}"
            if s.dtype == object and 0 < s.nunique() <= 15:
                desc += " [" + ", ".join(map(str, sorted(s.dropna().unique()))) + "]"
            cols.append(desc)
        lines.append(f"- {name} ({len(df):,} rows): " + "; ".join(cols))
    return "\n".join(lines)


SYSTEM = """You are the data analyst inside an EdTech analytics app. All data is synthetic demo data.
Answer the user's question about the data by calling run_sql (DuckDB, read-only, one SELECT/WITH per call; max 10,000 rows).
Tables (already filtered to the user's current scope: {scope}):
{schema}
Keys: student_id, course_id, enrollment_id, class_id, instructor_id link the tables. `learners` has one row per enrolled learner
(segment, risk_level, active_days_28d, attendance_28d_pct, avg_score_pct, net_revenue, tickets, days_since_activity).
Rules:
- Every number in your answer must come from a run_sql result. Never estimate, invent or extrapolate numbers.
- If the data cannot answer the question, say so plainly and suggest what can be answered.
- Describe relationships as associations, not causes. Do not profile individuals by personal attributes.
- Reply in concise markdown: a 1-2 sentence direct answer first, then a few bullets of supporting detail. Use ₹ for money.
- Don't paste large tables; the app shows your query results below your answer."""

TOOL = {"name": "run_sql", "description": "Run one read-only DuckDB SELECT over the tables and get the rows back as CSV.",
        "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}


def llm_answer(m: Model, q: str, history: list[tuple[str, str]]) -> X.EdAnswer:
    import anthropic

    tbls = tables(m)
    msgs: list[dict] = []
    for user, bot in history[-4:]:
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": bot or "(answered)"}]
    msgs.append({"role": "user", "content": q})
    client = anthropic.Anthropic()
    system = SYSTEM.format(scope=m.f.describe(), schema=schema(tbls))
    ran: list[tuple[str, pd.DataFrame]] = []
    text = ""
    for _ in range(MAX_TURNS):
        r = client.messages.create(model=os.getenv("VDL_LLM_MODEL", "claude-sonnet-5-5"), max_tokens=1500,
                                   system=system, tools=[TOOL], messages=msgs)
        text = "".join(b.text for b in r.content if b.type == "text")
        if r.stop_reason != "tool_use":
            break
        msgs.append({"role": "assistant", "content": r.content})
        results = []
        for b in r.content:
            if b.type != "tool_use":
                continue
            sql = str(b.input.get("sql", ""))
            try:
                df = sql_guard.run_multi(sql, tbls)
                ran.append((sql, df))
                out = df.head(ROWS_TO_LLM).to_csv(index=False)[:8000] + (f"\n({len(df)} rows total)" if len(df) > ROWS_TO_LLM else "")
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out or "(no rows)"})
            except Exception as e:  # noqa: BLE001 - the model sees the error and can fix its query
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": f"ERROR: {e}", "is_error": True})
        msgs.append({"role": "user", "content": results})
    else:
        text = text or "I couldn't finish within the query limit. Try a narrower question."
    last = ran[-1][1] if ran else None
    return X.EdAnswer(q, text or "(no answer)", table=last, sql="\n\n".join(s for s, _ in ran) or None, intent="llm",
                      confidence="Numbers come from the SQL shown" if ran else "No query ran: treat as unverified",
                      limitations=["Written by an LLM from query results; check the SQL and tables below."],
                      scope=X._scope(m, len(tbls["learners"])), computation=f"Claude + {len(ran)} read-only SQL quer{'y' if len(ran) == 1 else 'ies'}")


def answer(m: Model, q: str, history: list[tuple[str, str]], prev: X.EdAnswer | None = None, use_llm: bool = True) -> X.EdAnswer:
    if use_llm and llm_available():
        try:
            return llm_answer(m, q, history)
        except Exception as e:  # noqa: BLE001 - fall back to the offline analyst, and say so
            a = X.ask(m, q, prev)
            a.limitations = [f"LLM unavailable ({type(e).__name__}); answered offline."] + a.limitations
            return a
    return X.ask(m, q, prev)
