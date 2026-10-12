"""Vedantu Data Labs – EdTech Data Intelligence Platform (demo)  ·  run with:  streamlit run app.py"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from vdl import APP_NAME, copilot
from vdl.edtech import generate as gen
from vdl.edtech import insights as edins
from vdl.edtech.model import Filters, Model
from vdl.ui import components as C
from vdl.ui import edtech_pages as P
from vdl.ui import generic_pages as GP

try:  # optional .env support
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title=APP_NAME, page_icon="🎓", layout="wide")
C.inject_css()

NAV = {
    "Executive": ["Overview", "Business Intelligence"],
    "Students & Academics": ["Student Lifecycle", "Engagement", "Academic Performance", "Retention & Cohorts", "Student Success"],
    "Business Operations": ["Revenue & Finance", "Marketing & Growth", "Courses & Instructors", "Support Analytics"],
    "Intelligence": ["AI Analyst", "Forecasting", "Data Explorer", "Data Quality"],
    "Administration": ["Data Sources", "Metric Dictionary", "Demo Data Management"],
}
EDTECH = {"Overview": P.overview, "Business Intelligence": P.business_intelligence, "Student Lifecycle": P.lifecycle,
          "Engagement": P.engagement, "Academic Performance": P.academics, "Retention & Cohorts": P.retention,
          "Student Success": P.student_success, "Revenue & Finance": P.revenue, "Marketing & Growth": P.marketing,
          "Courses & Instructors": P.courses, "Support Analytics": P.support, "AI Analyst": P.ai_analyst,
          "Forecasting": P.forecasting, "Data Explorer": P.data_explorer, "Data Quality": P.data_quality,
          "Metric Dictionary": P.metric_dictionary, "Demo Data Management": P.demo_management}
GENERIC = {"Overview": GP.overview, "AI Analyst": GP.ai_analyst, "Data Explorer": GP.data_explorer, "Data Quality": GP.data_quality}
SUBTITLE = {
    "Overview": "Executive summary: health, trends, funnel, cohorts, courses and ranked actions.",
    "Business Intelligence": "Every KPI with definition, baseline, RAG status, plus top trends and risks.",
    "Student Lifecycle": "Lead → qualified → counselling → enrollment → first class → active → assessment → completion → renewal.",
    "Engagement": "How learners interact: DAU/WAU/MAU, timing, segments and declining engagement.",
    "Academic Performance": "Scores, pass rates, topic mastery, improvement and learners needing help.",
    "Retention & Cohorts": "Cohort retention, confirmed churn vs inferred inactivity, renewals and reactivation.",
    "Student Success": "Explainable early-warning flags, interventions and follow-up tracking.",
    "Revenue & Finance": "Bookings, collections, refunds, waterfall and unit economics. Actuals vs estimates labelled.",
    "Marketing & Growth": "Leads, conversion, CAC, ROAS and the quality of learners each channel brings.",
    "Courses & Instructors": "Course scorecards, growth × quality quadrant and fair instructor comparisons.",
    "Support Analytics": "Ticket volume, SLAs, contact reasons, CSAT and support-to-engagement analysis.",
    "AI Analyst": "Plain-English business questions answered from computed data.",
    "Forecasting": "Backtested forecasts with intervals and scenarios, or a clear data-sufficiency warning.",
    "Data Explorer": "Browse tables and run read-only SQL.",
    "Data Quality": "Quality checks, governance score, lineage, freshness and safe cleaning previews.",
    "Data Sources": "Demo data, file uploads, EdTech table uploads and CRM connectors.",
    "Metric Dictionary": "Single source of truth for every metric and rule.",
    "Demo Data Management": "Load, reset or regenerate the synthetic demo data.",
}

ss = st.session_state
ss.setdefault("page", "Overview")
ss.setdefault("workspace", "edtech")
ss.setdefault("analysed", False)
ss.setdefault("chat", [])
ss.setdefault("last_intent", None)
ss.setdefault("data_version", 0)
ss.setdefault("seed", gen.SEED)


@st.cache_data(show_spinner="Loading demo data…")
def load_demo(version: int) -> dict[str, pd.DataFrame]:
    return gen.load_all()


@st.cache_data(show_spinner="Computing insights…", max_entries=32)
def cached_insights(version: int, f: tuple) -> list:
    tables = ss.get("uploaded_tables") or load_demo(version)
    return edins.generate(Model(tables, Filters(*f)))


# demo-data admin actions (set by Demo Data Management page)
if ss.pop("reload_demo", False):
    ss.pop("uploaded_tables", None)
    ss.workspace, ss.data_version = "edtech", ss.data_version + 1
    st.toast("Demo data loaded.")
if (seed := ss.pop("regen_seed", None)) is not None:
    with st.spinner(f"Regenerating demo data with seed {seed}…"):
        gen.write_all(seed)
    ss.pop("uploaded_tables", None)
    ss.seed, ss.workspace, ss.data_version = seed, "edtech", ss.data_version + 1
    st.toast(f"Regenerated with seed {seed}.")

demo_mode = ss.workspace == "edtech" and "uploaded_tables" not in ss

# ---------------- sidebar: workspace, navigation, global filters ----------------
with st.sidebar:
    st.markdown(f"### 🎓 {APP_NAME}")
    if ss.workspace == "edtech":
        st.markdown("<span class='pill demo'>DEMO MODE</span>" if demo_mode else "<span class='pill live'>UPLOADED EDTECH DATA</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='pill live'>CUSTOM: {ss.get('name', '')}</span>", unsafe_allow_html=True)
        if st.button("⚡ Load Demo Data", width="stretch"):
            ss.workspace = "edtech"
            st.rerun()
    for section, pages in NAV.items():
        st.caption(section.upper())
        for p in pages:
            if st.button(p, key=f"nav_{p}", width="stretch", type="primary" if ss.page == p else "tertiary"):
                ss.page = p
                st.rerun()

filters = Filters()
model = None
if ss.workspace == "edtech":
    tables = ss.get("uploaded_tables") or load_demo(ss.data_version)
    s = tables["students"].merge(tables["courses"][["course_id", "course_name", "subject"]], on="course_id", how="left")
    with st.sidebar:
        st.divider()
        st.caption("GLOBAL FILTERS")
        lo, hi = gen.START.date(), gen.END.date()
        default = (max(lo, (gen.END - pd.Timedelta(days=89)).date()), hi)
        rng = st.date_input("Date range", value=default, min_value=lo, max_value=hi, key="f_dates")
        start, end = (rng if isinstance(rng, (list, tuple)) and len(rng) == 2 else default)
        courses = st.multiselect("Course", sorted(s.course_name.dropna().unique()), key="f_course")
        subjects = st.multiselect("Subject", sorted(s.subject.dropna().unique()), key="f_subject")
        cohorts = st.multiselect("Enrollment cohort", sorted(s.enrollment_date.dropna().dt.strftime("%Y-%m").unique()), key="f_cohort")
        channels = st.multiselect("Acquisition channel", sorted(s.acquisition_channel.dropna().unique()), key="f_channel")
        statuses = st.multiselect("Student status", sorted(s.student_status.dropna().unique()), key="f_status")
        filters = Filters(pd.Timestamp(start), pd.Timestamp(end), tuple(courses), tuple(subjects), tuple(cohorts), tuple(channels), tuple(statuses))
        st.caption("Default period: last 90 days of data, compared with the previous 90 days.")
    model = Model(tables, filters)

with st.sidebar:
    st.divider()
    st.caption("🧠 LLM: " + ("enabled (SQL proposals only)" if copilot.llm_available() else "off – deterministic analytics"))
    st.caption("🔒 Read-only SQL · in-memory · no data leaves this machine")

page = ss.page
scope = filters.describe() if model is not None else ss.get("name", "")
C.hero(page, SUBTITLE.get(page, ""), demo_mode, scope)

try:
    if page == "Data Sources":
        GP.data_sources()
    elif ss.workspace == "custom":
        if page in GENERIC and ss.analysed:
            GENERIC[page]()
        else:
            C.empty("This module needs the EdTech data model. Click **⚡ Load Demo Data** in the sidebar, or upload EdTech tables "
                    "in Data Sources. Custom single-file data supports Overview, AI Analyst, Data Explorer and Data Quality.")
    else:
        if model.empty() and page not in ("Metric Dictionary", "Demo Data Management", "Data Quality", "Data Explorer"):
            C.empty("No enrolled learners match the current filters. Clear some filters or widen the date range.")
        else:
            needs_ins = page in ("Overview", "Business Intelligence", "Student Lifecycle", "Engagement", "Academic Performance",
                                 "Retention & Cohorts", "Revenue & Finance", "Marketing & Growth", "Courses & Instructors", "Support Analytics")
            ins = cached_insights(ss.data_version, tuple(filters.__dict__.values())) if needs_ins else []
            EDTECH[page](model, ins)
except Exception as e:  # noqa: BLE001 - never show a raw traceback to business users
    st.error(f"Something went wrong rendering **{page}**: {e}. Try widening filters; the error has been contained.")
    if st.toggle("Show technical details"):
        st.exception(e)
