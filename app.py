"""Vedantu Data Labs - Demo  ·  run with:  streamlit run app.py"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from vdl import APP_NAME, connectors, copilot, dashboard, insights, quality, sql_guard
from vdl.datasets import SAMPLES, load_sample
from vdl.profiling import clean_columns, load_file, prepare

try:  # optional .env support
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title=APP_NAME, page_icon="🔬", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg,#2B1D8F 0%,#5B4BFF 55%,#00B3A4 100%); color:#fff;
       padding: 1.4rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;}
.hero h1 {margin:0; font-size:1.7rem; font-weight:700; color:#fff;}
.hero p {margin:.3rem 0 0; opacity:.88;}
.kpi {border:1px solid rgba(128,128,128,.22); border-radius:14px; padding:1rem 1.1rem; height:100%;}
.kpi .l {font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; opacity:.65;}
.kpi .v {font-size:1.55rem; font-weight:700; margin-top:.2rem;}
.kpi .h {font-size:.72rem; opacity:.55; margin-top:.3rem;}
.badge {display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.72rem; font-weight:600;}
.PASS {background:#DDF6EE; color:#0B7A55;} .WARN {background:#FFF1DB; color:#9A5B00;}
.FAIL {background:#FDE2E4; color:#B4232C;}
.tag-verified {background:#DDF6EE; color:#0B7A55;} .tag-hyp {background:#EEE9FF; color:#4B3BCB;}
.tag-act {background:#E2F1FF; color:#0B5CA8;}
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("analysed", False)
ss.setdefault("chat", [])
ss.setdefault("last_intent", None)


def run_analysis(raw: pd.DataFrame, name: str) -> None:
    if raw.empty or raw.shape[1] == 0:
        st.error("The file has no rows or columns.")
        return
    raw_clean = clean_columns(raw)
    df, prof = prepare(raw)
    checks = quality.run_all(raw_clean, df, prof)
    ss.update(raw=raw_clean, df=df, prof=prof, name=name, checks=checks,
              insights=insights.generate(df, prof, checks), analysed=True, chat=[], last_intent=None)


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown(f"### 🔬 {APP_NAME}")
    st.caption("Upload or pick data → **Analyse** → explore.")
    src = st.radio("Data source", ["Sample dataset", "Upload file"], horizontal=True)
    raw, name = None, None
    if src == "Sample dataset":
        name = st.selectbox("Dataset", list(SAMPLES))
    else:
        up = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
        if up is not None:
            name = up.name
    if st.button("⚡ Analyse", type="primary", use_container_width=True, disabled=name is None):
        try:
            raw = load_sample(name) if src == "Sample dataset" else load_file(up, up.name)
            with st.spinner("Profiling, validating and generating insights…"):
                run_analysis(raw, name)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read this file: {e}")
    st.divider()
    page = st.radio("Navigate", ["Overview", "AI Analyst", "Data Explorer", "Data Quality", "Connections"],
                    label_visibility="collapsed")
    st.divider()
    st.caption("🧠 LLM: " + ("enabled (SQL proposals only)" if copilot.llm_available() else
                            "off – deterministic engine"))
    st.caption("🔒 Read-only SQL · in-memory · no data leaves this machine" +
               ("" if not copilot.llm_available() else " (schema sent to LLM on fallback)"))

st.markdown(f"<div class='hero'><h1>{APP_NAME}</h1><p>AI-native analytics: dashboards, evidence-backed "
            "insights, data trust checks and plain-English questions, computed from your data.</p></div>",
            unsafe_allow_html=True)

if page == "Connections":
    st.subheader("Connections")
    st.caption("Status reflects real configuration only. Nothing is shown as connected unless it is.")
    for info in connectors.all_info():
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{info.name}**  \n{info.auth}")
            colour = "PASS" if info.status.startswith("Available") else "WARN" if "present" in info.status else "FAIL"
            c2.markdown(f"<span class='badge {colour}'>{info.status}</span>", unsafe_allow_html=True)
            if info.env_vars:
                st.caption(f"Env vars: `{'`, `'.join(info.env_vars)}` · Scopes: {info.scopes}")
            st.caption(info.notes)
    with st.expander("Integration architecture"):
        st.markdown("""
1. **Connector interface** (`vdl/connectors.py`): `status()`, `fetch(object, limit) → DataFrame`.
2. **Auth**: OAuth 2.0 authorization-code flow; tokens stored server-side (secret manager), never in the browser or repo.
3. **Least privilege**: read-only scopes / integration users; per-object allow-list.
4. **Same pipeline**: fetched data goes through profiling → quality checks → read-only SQL guard.
5. **Audit**: log who pulled what, when, and which queries ran.""")
    st.stop()

if not ss.analysed:
    st.info("👈 Choose a sample dataset or upload a CSV/Excel file, then click **Analyse**.")
    st.stop()

df: pd.DataFrame = ss.df
prof = ss.prof
checks = ss.checks
score = quality.trust_score(checks)

# ---------------- Overview ----------------
if page == "Overview":
    st.subheader(f"Overview · {ss.name}")
    ks = dashboard.kpis(df, prof) + [dashboard.KPI("Trust score", f"{score}/100", "100 − 10 per WARN − 25 per FAIL")]
    for col, k in zip(st.columns(len(ks)), ks):
        col.markdown(f"<div class='kpi'><div class='l'>{k.label}</div><div class='v'>{k.value}</div>"
                     f"<div class='h'>{k.help}</div></div>", unsafe_allow_html=True)
    st.write("")
    figs = dashboard.charts(df, prof)
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, (title, fig) in zip(cols, figs[i:i + 2]):
            with col.container(border=True):
                st.markdown(f"**{title}**")
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Business recommendations")
    st.caption("Verified findings are computed from the data. Explanations are hypotheses. Actions are suggestions.")
    if not ss.insights:
        st.write("No notable trends, risks or opportunities passed the thresholds.")
    icons = {"trend": "📈", "risk": "⚠️", "opportunity": "💡", "quality": "🧪"}
    for ins in ss.insights:
        with st.expander(f"{icons[ins.kind]} {ins.title}", expanded=ins.kind != "quality"):
            st.markdown(f"<span class='badge tag-verified'>VERIFIED FINDING</span> {ins.finding}", unsafe_allow_html=True)
            st.caption(f"Calculation: {ins.calculation}")
            st.markdown(f"<span class='badge tag-hyp'>POSSIBLE EXPLANATION</span> {ins.explanation}", unsafe_allow_html=True)
            st.markdown(f"<span class='badge tag-act'>SUGGESTED ACTION</span> {ins.action}", unsafe_allow_html=True)
            if not ins.evidence.empty:
                st.dataframe(ins.evidence, use_container_width=True, hide_index=True)

    with st.expander("Detected schema"):
        st.dataframe(pd.DataFrame({"column": list(prof.types), "detected_type": list(prof.types.values()),
                                   "non_null": [int(df[c].notna().sum()) for c in prof.types],
                                   "unique": [int(df[c].nunique()) for c in prof.types]}),
                     use_container_width=True, hide_index=True)

# ---------------- AI Analyst ----------------
elif page == "AI Analyst":
    st.subheader("AI Analyst")
    st.caption("Ask in plain English. Every number is computed with a validated read-only SQL query you can inspect. "
               "Follow-ups like *“what about Delhi?”* or *“break it down by product”* keep context.")
    cols = st.columns(len(copilot.suggestions(prof)))
    clicked = None
    for c, s in zip(cols, copilot.suggestions(prof)):
        if c.button(s, use_container_width=True):
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
            st.dataframe(t, use_container_width=True, hide_index=True)
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

# ---------------- Data Explorer ----------------
elif page == "Data Explorer":
    st.subheader("Data Explorer")
    c1, c2 = st.columns([2, 1])
    cols = c1.multiselect("Columns", list(df.columns), default=list(df.columns))
    fcol = c2.selectbox("Filter column", ["(none)"] + prof.categorical)
    view = df
    if fcol != "(none)":
        vals = c2.multiselect("Values", sorted(df[fcol].dropna().astype(str).unique()))
        if vals:
            view = df[df[fcol].astype(str).isin(vals)]
    st.caption(f"{len(view):,} of {len(df):,} rows")
    st.dataframe(view[cols], use_container_width=True, height=380)

    st.markdown("#### SQL workbench (read-only)")
    st.caption("Table name: `data`. Only single SELECT/WITH statements; writes, file and network access are blocked.")
    default_sql = f"SELECT * FROM data LIMIT 20"
    sql = st.text_area("SQL", default_sql, height=110)
    if st.button("Run query"):
        try:
            st.dataframe(sql_guard.run(sql, df), use_container_width=True, hide_index=True)
            st.success("Validated read-only query executed.")
        except sql_guard.UnsafeSQLError as e:
            st.error(f"Blocked: {e}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Query error: {e}")

# ---------------- Data Quality ----------------
elif page == "Data Quality":
    st.subheader("Data Quality & Trust")
    c = st.columns(len(checks) + 1)
    c[0].metric("Trust score", f"{score}/100")
    for col, ch in zip(c[1:], checks):
        col.markdown(f"**{ch.name}**  \n<span class='badge {ch.status}'>{ch.status}</span>", unsafe_allow_html=True)
    for ch in checks:
        with st.expander(f"{ch.name} — {ch.status}", expanded=ch.status != "PASS"):
            st.write(ch.detail)
            st.caption(f"Calculation: {ch.calculation}")
            if ch.evidence is not None and not ch.evidence.empty:
                st.dataframe(ch.evidence, use_container_width=True)
    st.caption("Assumptions: column types are auto-detected from values and names; anomalies are flagged for review, "
               "never removed automatically. Trust score = 100 − 10 per WARN − 25 per FAIL.")
