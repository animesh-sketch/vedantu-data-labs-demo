"""Custom-dataset pages (the original single-table workflow) and the Data Sources page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import connectors, copilot, dashboard, insights, quality, sql_guard
from ..datasets import SAMPLES, load_sample
from ..edtech.generate import DATE_COLS, TABLES, generate
from ..profiling import clean_columns, load_file, prepare


def run_analysis(raw: pd.DataFrame, name: str) -> bool:
    ss = st.session_state
    if raw is None or raw.empty or raw.shape[1] == 0:
        st.error("The file has no rows or columns.")
        return False
    raw_clean = clean_columns(raw)
    df, prof = prepare(raw)
    checks = quality.run_all(raw_clean, df, prof)
    ss.update(raw=raw_clean, df=df, prof=prof, name=name, checks=checks, insights=insights.generate(df, prof, checks),
              analysed=True, chat=[], last_intent=None, workspace="custom")
    return True


def overview() -> None:
    ss = st.session_state
    df, prof, checks = ss.df, ss.prof, ss.checks
    score = quality.trust_score(checks)
    st.subheader(f"Overview · {ss.name}")
    ks = dashboard.kpis(df, prof) + [dashboard.KPI("Trust score", f"{score}/100", "100 − 10 per WARN − 25 per FAIL")]
    for col, k in zip(st.columns(len(ks)), ks):
        col.markdown(f"<div class='kpi'><div class='l'>{k.label}</div><div class='v'>{k.value}</div>"
                     f"<div class='b'>{k.help}</div></div>", unsafe_allow_html=True)
    st.write("")
    figs = dashboard.charts(df, prof)
    for i in range(0, len(figs), 2):
        for col, (title, fig) in zip(st.columns(2), figs[i:i + 2]):
            with col.container(border=True):
                st.markdown(f"**{title}**")
                st.plotly_chart(fig, width="stretch")
    st.subheader("Business recommendations")
    st.caption("Verified findings are computed from the data. Explanations are hypotheses. Actions are suggestions.")
    if not ss.insights:
        st.write("No notable trends, risks or opportunities passed the thresholds.")
    icons = {"trend": "📈", "risk": "⚠️", "opportunity": "💡", "quality": "🧪"}
    for ins in ss.insights:
        with st.expander(f"{icons[ins.kind]} {ins.title}", expanded=ins.kind != "quality"):
            st.markdown(f"<span class='tag t-ver'>VERIFIED FINDING</span> {ins.finding}", unsafe_allow_html=True)
            st.caption(f"Calculation: {ins.calculation}")
            st.markdown(f"<span class='tag t-assoc'>POSSIBLE EXPLANATION</span> {ins.explanation}", unsafe_allow_html=True)
            st.markdown(f"<span class='tag t-rule'>SUGGESTED ACTION</span> {ins.action}", unsafe_allow_html=True)
            if not ins.evidence.empty:
                st.dataframe(ins.evidence, width="stretch", hide_index=True)
    with st.expander("Detected schema"):
        st.dataframe(pd.DataFrame({"column": list(prof.types), "detected_type": list(prof.types.values()),
                                   "non_null": [int(df[c].notna().sum()) for c in prof.types],
                                   "unique": [int(df[c].nunique()) for c in prof.types]}), width="stretch", hide_index=True)


def ai_analyst() -> None:
    ss = st.session_state
    df, prof = ss.df, ss.prof
    st.caption("Ask in plain English. Every number is computed with a validated read-only SQL query you can inspect.")
    sug = copilot.suggestions(prof)
    clicked = None
    for c, s in zip(st.columns(len(sug)), sug):
        if c.button(s, width="stretch"):
            clicked = s
    if st.button("Clear conversation"):
        ss.chat, ss.last_intent = [], None

    def show(a: copilot.Answer) -> None:
        st.markdown(a.text)
        if a.table is not None and not a.table.empty:
            t = a.table
            if a.intent and a.intent.grain and len(t) > 1:
                st.line_chart(t.set_index("period").iloc[:, 0])
            elif a.intent and a.intent.group and 1 < len(t) <= 30:
                st.bar_chart(t.set_index(t.columns[0]).iloc[:, 0])
            st.dataframe(t, width="stretch", hide_index=True)
        if a.sql or a.assumptions or a.limitations:
            with st.expander("Evidence · SQL · assumptions · limitations"):
                if a.sql:
                    st.code(a.sql, language="sql")
                for x in a.assumptions:
                    st.markdown(f"- **Assumption:** {x}")
                for x in a.limitations:
                    st.markdown(f"- **Limitation:** {x}")
                st.caption(f"Engine: {a.source} · Status: {'validated' if a.ok else 'not answered'}")

    for role, content in ss.chat:
        with st.chat_message(role):
            show(content) if role == "assistant" else st.markdown(content)
    q = st.chat_input("e.g. Which city has the highest total deal value?") or clicked
    if q:
        ss.chat.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)
        ans = copilot.ask(q, df, prof, ss.last_intent)
        if ans.intent and ans.intent.kind == "aggregate":
            ss.last_intent = ans.intent
        ss.chat.append(("assistant", ans))
        with st.chat_message("assistant"):
            show(ans)


def data_explorer() -> None:
    ss = st.session_state
    df, prof = ss.df, ss.prof
    c1, c2 = st.columns([2, 1])
    cols = c1.multiselect("Columns", list(df.columns), default=list(df.columns))
    fcol = c2.selectbox("Filter column", ["(none)"] + prof.categorical)
    view = df
    if fcol != "(none)":
        vals = c2.multiselect("Values", sorted(df[fcol].dropna().astype(str).unique()))
        if vals:
            view = df[df[fcol].astype(str).isin(vals)]
    st.caption(f"{len(view):,} of {len(df):,} rows")
    st.dataframe(view[cols], width="stretch", height=380)
    st.markdown("#### SQL workbench (read-only)")
    st.caption("Table name: `data`. Only single SELECT/WITH statements; writes, file and network access are blocked.")
    sql = st.text_area("SQL", "SELECT * FROM data LIMIT 20", height=110)
    if st.button("Run query"):
        try:
            st.dataframe(sql_guard.run(sql, df), width="stretch", hide_index=True)
            st.success("Validated read-only query executed.")
        except sql_guard.UnsafeSQLError as e:
            st.error(f"Blocked: {e}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Query error: {e}")


def data_quality() -> None:
    checks = st.session_state.checks
    score = quality.trust_score(checks)
    c = st.columns(len(checks) + 1)
    c[0].metric("Trust score", f"{score}/100")
    for col, ch in zip(c[1:], checks):
        col.markdown(f"**{ch.name}**  \n{ch.status}")
    for ch in checks:
        with st.expander(f"{ch.name} — {ch.status}", expanded=ch.status != "PASS"):
            st.write(ch.detail)
            st.caption(f"Calculation: {ch.calculation}")
            if ch.evidence is not None and not ch.evidence.empty:
                st.dataframe(ch.evidence, width="stretch")


def validate_edtech_upload(files: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame] | None, list[str]]:
    """Check an uploaded table set against the demo schema; returns (tables, problems)."""
    ref = {k: list(v.columns) for k, v in generate(scale=0.02).items()}
    problems = [f"Missing table `{t}.csv`." for t in TABLES if t not in files]
    for t, df in files.items():
        if t not in ref:
            problems.append(f"Unknown table `{t}` (ignored).")
            continue
        miss = [c for c in ref[t] if c not in df.columns]
        if miss:
            problems.append(f"`{t}` is missing columns: {', '.join(miss)}")
    if any(p.startswith("Missing") or "missing columns" in p for p in problems):
        return None, problems
    out = {}
    for t in TABLES:
        df = files[t].copy()
        for c in DATE_COLS.get(t, []):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        out[t] = df
    return out, problems


def data_sources() -> None:
    ss = st.session_state
    st.markdown("#### 1 · Demo data (default)")
    st.write("Synthetic, interconnected EdTech data loads automatically. Manage it under **Demo Data Management**.")
    if ss.get("workspace") != "edtech" and st.button("⚡ Load Demo Data", type="primary"):
        ss.workspace = "edtech"
        st.rerun()
    st.markdown("#### 2 · Upload a single CSV/Excel file (generic analysis)")
    src = st.radio("Source", ["Sample dataset", "Upload file"], horizontal=True)
    name, up = None, None
    if src == "Sample dataset":
        name = st.selectbox("Dataset", list(SAMPLES))
    else:
        up = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
        name = up.name if up is not None else None
    if st.button("⚡ Analyse", type="primary", disabled=name is None):
        try:
            raw = load_sample(name) if src == "Sample dataset" else load_file(up, up.name)
            with st.spinner("Profiling, validating and generating insights…"):
                if run_analysis(raw, name):
                    st.success(f"Analysed {name}. Workspace switched to **Custom data**: use Overview, Chat with Data, Data Explorer, Data Quality.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read this file: {e}")
    st.markdown("#### 3 · Upload your own EdTech tables (all 12 modules)")
    files = st.file_uploader("CSV files named like the demo tables (students.csv, enrollments.csv, …)", type=["csv"], accept_multiple_files=True)
    if files and st.button("Validate & load tables"):
        got = {}
        for f in files:
            try:
                got[f.name.rsplit(".", 1)[0].lower()] = pd.read_csv(f)
            except Exception as e:  # noqa: BLE001
                st.error(f"{f.name}: {e}")
        tables, problems = validate_edtech_upload(got)
        for p in problems:
            st.warning(p)
        if tables:
            ss.uploaded_tables, ss.workspace = tables, "edtech"
            ss.data_version = ss.get("data_version", 0) + 1
            st.success("Schema valid. Loaded as the active EdTech data model (Demo Mode off).")
    st.markdown("#### 4 · CRM & database connectors")
    st.caption("Status reflects real configuration only. Nothing is shown as connected unless it is.")
    for info in connectors.all_info():
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{info.name}**  \n{info.auth}")
            c2.markdown(f"**{info.status}**")
            if info.env_vars:
                st.caption(f"Env vars: `{'`, `'.join(info.env_vars)}` · Scopes: {info.scopes}")
            st.caption(info.notes)
    with st.expander("Integration architecture"):
        st.markdown("1. **Connector interface** (`vdl/connectors.py`): `status()`, `fetch(object, limit) → DataFrame`.\n"
                    "2. **Auth**: OAuth 2.0 authorization-code flow; tokens stored server-side, never in the browser or repo.\n"
                    "3. **Least privilege**: read-only scopes; per-object allow-list.\n"
                    "4. **Same pipeline**: fetched data → schema validation → quality checks → read-only SQL guard.\n"
                    "5. **Audit**: log who pulled what, when, and which queries ran.")

