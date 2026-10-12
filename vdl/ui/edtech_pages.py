"""EdTech Data Intelligence pages. Each page: decision brief (5 questions) → visuals → detail tables.
All numbers come from vdl.edtech.metrics / analytics / insights / forecast; nothing is hard-coded."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..edtech import analytics as A
from ..edtech import analyst as X
from ..edtech import forecast as F
from ..edtech import governance as G
from ..edtech import metrics as M
from ..edtech.generate import SEED, make_dirty
from ..edtech.model import Model
from .. import sql_guard
from . import components as C


def _kpis(m: Model, keys: list[str]) -> list[M.KPI]:
    return [M.kpi(m, k) for k in keys]


def _actions_for(ins, names: list[str], n: int = 4) -> list[str]:
    out = [i.action for i in ins if any(x.lower() in i.metric.lower() or x.lower() in i.title.lower() for x in names)]
    return list(dict.fromkeys(out))[:n]


def _next_text(m: Model, what: str, label: str) -> str:
    fc = F.forecast(m, what)
    if fc.forecast is None:
        return f"⚠️ {fc.warning}"
    n4 = fc.forecast.head(4)
    return (f"**{label}, next 4 weeks:** baseline **{n4.baseline.sum():,.0f}** (80% range {n4.lo80.sum():,.0f} – {n4.hi80.sum():,.0f}). "
            f"Model: {fc.backtest['chosen_model']}, backtest MAPE {fc.backtest.get('MAPE_pct')}%. "
            + (f"⚠️ {fc.warning} " if fc.warning else "") + "See **Forecasting** for assumptions.")


def _monthly_line(df: pd.DataFrame, cols: list[str], title: str, caption: str | None = None, money: bool = False) -> None:
    long = df.melt(id_vars=df.columns[0], value_vars=cols, var_name="series", value_name="value")
    fig = px.line(long, x=df.columns[0], y="value", color="series", markers=True)
    if money:
        fig.update_yaxes(tickprefix="₹", tickformat=",.0s")
    C.chart(fig, title, caption)


# =========================================================== Executive
def overview(m: Model, ins: list) -> None:
    C.kpi_strip(_kpis(m, ["new_enrollments", "active_learners", "net_revenue", "completion_rate", "attendance_rate", "avg_score"]))
    C.kpi_strip(_kpis(m, ["lead_to_enrol", "retention_rate", "inactive_rate", "refund_rate", "nps", "cac"]))
    c1, c2 = st.columns(2)
    with c1:
        en = M.monthly(m, "new_enrollments").rename(columns={"value": "New enrollments"})
        rv = A.revenue_monthly(m)[["month", "net_revenue"]]
        fig = go.Figure()
        fig.add_bar(x=en.month, y=en["New enrollments"], name="New enrollments", marker_color=C.PALETTE[0])
        fig.add_scatter(x=rv.month, y=rv.net_revenue, name="Net revenue (₹)", yaxis="y2", mode="lines+markers", line_color=C.PALETTE[1])
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickprefix="₹", tickformat=",.2s", showgrid=False))
        C.chart(fig, "Enrollment & revenue trend", "Monthly. Latest month may be partial. Revenue = collections − refunds.")
    with c2:
        fn = A.funnel(m)
        if fn.empty:
            C.empty("No leads in the selected period.")
        else:
            fig = go.Figure(go.Funnel(y=fn.stage, x=fn.students, textinfo="value+percent initial", marker_color=C.PALETTE[0]))
            C.chart(fig, "Learner funnel (leads created in period)", "Strict funnel: each stage requires the previous one. "
                    "Completion/renewal need time, so recent lead cohorts look low.", legend=False)
    c1, c2 = st.columns(2)
    with c1:
        at = A.active_trend(m, "W")
        C.chart(px.area(at, x="period", y="active_learners"), "Weekly active learners", "Distinct learners with activity or attended class per week.", legend=False)
    with c2:
        mat = A.cohort_matrix(m)
        if not mat.empty:
            C.heatmap(mat.drop(columns="cohort_size").iloc[:, :9], "Cohort retention (% active by month since enrollment)",
                      "Rows: enrollment month. Blank = not yet observable.", fmt=".0f", height=340)
    c1, c2 = st.columns(2)
    sc = A.course_scorecard(m)
    with c1:
        if not sc.empty:
            cols = ["attendance_pct", "completion_pct", "avg_score_pct", "refund_rate_pct", "inactive_30d_pct"]
            h = sc.set_index("course")[cols]
            z = (h - h.mean()) / h.std(ddof=0).replace(0, 1)
            z[["refund_rate_pct", "inactive_30d_pct"]] *= -1
            fig = px.imshow(z, aspect="auto", color_continuous_scale="RdYlGn", zmin=-2, zmax=2)
            fig.update_traces(text=h.round(1).values, texttemplate="%{text}", hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>")
            C.chart(fig, "Course performance heatmap", "Values shown; colour = standardised vs other courses (green = better; "
                    "refund and inactivity inverted).", height=400, legend=False)
    with c2:
        rb = A.revenue_by(m, "course_name")
        if not rb.empty:
            C.chart(px.treemap(rb, path=["course_name"], values="net_revenue", color="net_revenue", color_continuous_scale=C.SEQ),
                    "Revenue contribution by course", "Net cash revenue in period.", height=400, legend=False)
    c1, c2 = st.columns(2)
    with c1:
        g = A.by_group(m, "course_name")
        if not g.empty:
            g = g.merge(m.courses[["course_name", "course_level"]], on="course_name").sort_values("avg_score_pct")
            C.chart(px.bar(g, x="avg_score_pct", y="course_name", color="course_level", orientation="h",
                           hover_data=["pass_rate_pct", "learners"]), "Academic outcomes by course",
                    "Average assessment score (% of max). Levels differ in difficulty — compare within level.")
    with c2:
        r = A.student_risk(m)
        if not r.empty:
            rc = r.groupby(["course", "risk_level"]).size().reset_index(name="learners")
            rc = rc[rc.risk_level.isin(["High", "Medium"])]
            fig = px.bar(rc, x="learners", y="course", color="risk_level", orientation="h",
                         color_discrete_map={"High": "#E11D48", "Medium": "#F59E0B"})
            C.chart(fig, f"At-risk learners: {int((r.risk_level == 'High').sum())} High · {int((r.risk_level == 'Medium').sum())} Medium",
                    "Rules-based flags (see Student Success for rules and backtest).")
    st.subheader("Ranked business insights")
    st.caption("Ranked by impact score = metric weight × change size × reach. Verified metrics, associations and rules-based flags are labelled.")
    for n, i in enumerate(ins[:10], 1):
        C.insight_card(i, n)


def business_intelligence(m: Model, ins: list) -> None:
    from ..edtech.insights import top
    areas = {"Growth": ["leads", "new_enrollments", "lead_to_enrol", "activation_rate", "cac", "marketing_spend"],
             "Learning & engagement": ["active_learners", "dau", "wau", "mau", "attendance_rate", "avg_score", "pass_rate", "completion_rate"],
             "Retention": ["retention_rate", "churn_confirmed", "inactive_rate"],
             "Finance": ["gross_bookings", "collections", "refunds", "net_revenue", "arps", "aov", "refund_rate", "payment_failure_rate"],
             "Experience": ["csat", "satisfaction", "nps", "tickets", "sla_compliance"]}
    health = []
    for area, keys in areas.items():
        ks = _kpis(m, keys)
        reds, ambers = sum(k.rag == "red" for k in ks), sum(k.rag == "amber" for k in ks)
        health.append((area, "red" if reds >= 2 else "amber" if reds or ambers >= 2 else "green", ks))
    cols = st.columns(len(health))
    for c, (area, rag, ks) in zip(cols, health):
        bg, fg, label = C.RAG[rag]
        c.markdown(f"<div class='kpi' style='background:{bg}'><div class='l' style='color:{fg}'>{area}</div>"
                   f"<div class='v' style='color:{fg}'>{label}</div><div class='b' style='color:{fg}'>"
                   f"{sum(k.rag == 'red' for k in ks)} red · {sum(k.rag == 'amber' for k in ks)} amber · {sum(k.rag == 'green' for k in ks)} green</div></div>",
                   unsafe_allow_html=True)
    st.caption("Area health: red if ≥2 KPIs red; amber if 1 red or ≥2 amber; else green. KPI RAG compares to the previous equal-length "
               "period (rates: ±1 pp green tolerance, red beyond −4 pp; levels: ±2% / −8%).")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 5 positive trends")
        for n, i in enumerate(top(ins, "positive"), 1):
            C.insight_card(i, n, show_evidence=False)
        if not top(ins, "positive"):
            C.empty("No positive movements passed the thresholds.")
    with c2:
        st.subheader("Top 5 emerging risks")
        for n, i in enumerate(top(ins, "risk"), 1):
            C.insight_card(i, n, show_evidence=False)
    for area, rag, ks in health:
        st.markdown(f"#### {area}")
        C.kpi_strip(ks, per_row=4)
    with st.expander("KPI definitions, periods, baselines and calculations"):
        rows = [{"KPI": k.name, "value": M.fmt(k.value, k.unit), "previous": M.fmt(k.prev, k.unit), "change": M.fmt_change(k),
                 "status": C.RAG[k.rag][2], "period": k.period, "baseline": k.baseline, "definition": k.definition, "calculation": k.formula}
                for _, _, ks in health for k in ks]
        C.table(pd.DataFrame(rows))


# =========================================================== Students & Academics
def lifecycle(m: Model, ins: list) -> None:
    fn = A.funnel(m)
    by_ch = A.funnel_by(m, "acquisition_channel")
    C.five_questions(_kpis(m, ["leads", "new_enrollments", "lead_to_enrol", "activation_rate"]),
                     ("Largest stage-to-stage losses in this period's lead cohort:", fn.sort_values("drop_off", ascending=False).head(3)
                      if not fn.empty else None),
                     ("Conversion by acquisition channel:", by_ch),
                     _next_text(m, "enrollments", "New enrollments"),
                     _actions_for(ins, ["enroll", "lead", "first class", "conversion"]))
    if fn.empty:
        C.empty("No leads in the selected period. Widen the date range.")
        return
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure(go.Funnel(y=fn.stage, x=fn.students, textinfo="value+percent previous", marker_color=C.PALETTE[0]))
        C.chart(fig, "Learner journey funnel", "Leads created in the period, followed to today. % = conversion from previous stage.", height=430, legend=False)
    with c2:
        st.markdown("**Largest drop-off points**")
        C.table(fn.sort_values("drop_off", ascending=False)[["stage", "students", "step_conversion", "drop_off"]].head(5),
                "drop_off = learners lost entering this stage; step_conversion = % of previous stage.")
        st.markdown("**Time between stages**")
        C.table(A.stage_times(m))
    c1, c2 = st.columns(2)
    with c1:
        C.chart(px.bar(by_ch.sort_values("lead_to_enrol_pct"), x="lead_to_enrol_pct", y="acquisition_channel", orientation="h",
                       hover_data=["leads", "enrolled"], text="lead_to_enrol_pct"), "Lead → enrollment by channel", legend=False)
    with c2:
        bc = A.funnel_by(m, "counselor")
        C.chart(px.bar(bc.sort_values("lead_to_enrol_pct"), x="lead_to_enrol_pct", y="counselor", orientation="h", hover_data=["leads", "enrolled"]),
                "Lead → enrollment by counsellor", "Lead mix differs by counsellor; small gaps may be noise (see lead counts).", legend=False)
    c1, c2 = st.columns(2)
    with c1:
        bcs = A.funnel_by(m, "course_name")
        C.chart(px.bar(bcs.sort_values("lead_to_enrol_pct"), x="lead_to_enrol_pct", y="course_name", orientation="h", hover_data=["leads"]),
                "Lead → enrollment by course", legend=False)
    with c2:
        fce = A.first_class_effect(m)
        if not fce.empty:
            C.chart(px.bar(fce, x="group", y="avg_active_days_day15_60", text="avg_active_days_day15_60", hover_data=["students"]),
                    "Time to first class vs later engagement", "Average active days between day 15 and 60 after enrollment. "
                    f"Sample sizes: {', '.join(f'{g}: {n}' for g, n in zip(fce.group, fce.students))}. Association, not causation.", legend=False)
    st.markdown("#### Cohort funnel (by lead month)")
    cf = A.cohort_funnel(m)
    C.table(cf, "All lead history within segment filters. Later cohorts have had less time to progress.")
    st.markdown("#### Enrollment cancellations & refunds (enrollments in period)")
    C.table(A.cancellations(m))


def engagement(m: Model, ins: list) -> None:
    seg = A.segments(m)
    C.five_questions(_kpis(m, ["active_learners", "dau", "wau", "attendance_rate"]),
                     ("Engagement segments explain where activity is concentrated:", seg.segment.value_counts().rename_axis("segment").reset_index(name="learners") if not seg.empty else None),
                     ("Learners with declining engagement (last 14 days ≤ half the previous 14):", A.declining_learners(m).head(20)),
                     _next_text(m, "active_learners", "Weekly active learners"),
                     _actions_for(ins, ["active", "attendance", "engag", "inactiv"]))
    d = A.dau_wau_mau(m)
    if not d.empty:
        _monthly_line(d[["date", "DAU", "WAU (rolling 7d)", "MAU (rolling 30d)"]], ["DAU", "WAU (rolling 7d)", "MAU (rolling 30d)"],
                      "DAU / WAU / MAU", f"Stickiness (DAU/MAU) latest: {d.stickiness_DAU_MAU_pct.iloc[-1]:.1f}%.")
    c1, c2 = st.columns(2)
    with c1:
        hm = A.activity_heatmap(m)
        C.heatmap(hm, "Engagement by day of week × hour", "Learning-activity events in period (live classes excluded).", fmt="", height=360)
    with c2:
        cal = A.calendar(m)
        if not cal.empty:
            cal["week"] = cal.date.dt.to_period("W").dt.start_time.dt.strftime("%d %b")
            cal["weekday"] = cal.date.dt.day_name().str[:3]
            pv = cal.pivot_table(index="weekday", columns="week", values="active_learners").reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            C.heatmap(pv, "Engagement calendar (active learners per day)", fmt="", height=360)
    c1, c2 = st.columns(2)
    with c1:
        if not seg.empty:
            sc = seg.segment.value_counts().reset_index()
            sc.columns = ["segment", "learners"]
            C.chart(px.bar(sc, x="learners", y="segment", orientation="h", color="segment"), "Engagement segments (as of period end)", legend=False)
    with c2:
        wv = A.watch_vs_score(m)
        if not wv.empty:
            corr = wv[["watch_minutes", "avg_score_pct"]].corr().iloc[0, 1]
            C.chart(px.scatter(wv, x="watch_minutes", y="avg_score_pct", color="course_name", opacity=0.55, hover_data=["student_id"]),
                    "Video watch time vs assessment score", f"Pearson r = {corr:.2f} over {len(wv):,} learners. Correlation ≠ causation.", legend=False)
    c1, c2 = st.columns(2)
    with c1:
        C.chart(px.bar(A.engagement_by(m, "course_name"), x="avg_active_days", y="course_name", orientation="h", hover_data=["enrolled", "active_pct"]),
                "Avg active days per enrolled learner, by course", legend=False)
    with c2:
        rc = A.retention_curve(m, "W", 16)
        if not rc.empty:
            C.chart(px.line(rc, x="period_since_enrollment", y="retained_pct", markers=True, hover_data=["eligible_learners"]),
                    "Weekly engagement retention", "% of learners active in week k after enrollment (only learners with k weeks of history).", legend=False)
    with st.expander("Segment rules (applied in order)"):
        C.table(pd.DataFrame(A.SEGMENT_RULES, columns=["segment", "rule"]))
    with st.expander("Learner activity distribution"):
        if not seg.empty:
            st.plotly_chart(C.style(px.histogram(seg[~seg.status.isin(["Completed", "Cancelled"])], x="active_days_28d", nbins=28)), width="stretch")


def academics(m: Model, ins: list) -> None:
    ta = A.topic_accuracy(m)
    iv = A.needs_intervention(m)
    C.five_questions(_kpis(m, ["avg_score", "pass_rate", "attendance_rate", "completion_rate"]),
                     ("Topic difficulty drives score gaps — lowest-accuracy topics:", ta.head(5) if not ta.empty else None),
                     (f"{len(iv)} open learners need academic intervention (below pass mark or declining ≥10 pp).", iv.head(20)),
                     "Assessment scores are not forecast (no stable trend); monitor monthly trend below.",
                     _actions_for(ins, ["score", "topic", "assessment", "completion"]))
    a = A.assessments_w(m)
    if a.empty:
        C.empty("No assessments in the selected filters.")
        return
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(a, x="pct", nbins=40, color_discrete_sequence=[C.PALETTE[0]])
        fig.add_vline(x=M.PASS_MARK, line_dash="dash", annotation_text="pass mark")
        C.chart(fig, "Score distribution", f"Mean {a.pct.mean():.1f}%, median {a.pct.median():.1f}% over {len(a):,} assessments.", legend=False)
    with c2:
        tr = pd.DataFrame({"Average score %": M.monthly(m, "avg_score").set_index("month").value,
                           "Pass rate %": M.monthly(m, "pass_rate").set_index("month").value}).reset_index()
        _monthly_line(tr, ["Average score %", "Pass rate %"], "Assessment performance trend")
    C.heatmap(A.course_topic_matrix(m), "Topic mastery matrix (accuracy % by course × topic)", "Accuracy = correct / attempted.", fmt=".0f",
              height=420, colorscale="RdYlGn")
    c1, c2 = st.columns(2)
    with c1:
        wv = A.watch_vs_score(m).dropna(subset=["attendance_pct"])
        if not wv.empty:
            r = wv[["attendance_pct", "avg_score_pct"]].corr().iloc[0, 1]
            C.chart(px.scatter(wv, x="attendance_pct", y="avg_score_pct", opacity=0.5, trendline=None, hover_data=["student_id", "course_name"]),
                    "Attendance vs score (per learner)", f"Pearson r = {r:.2f}, n = {len(wv):,}. Association only.", legend=False)
    with c2:
        imp = A.improvement(m)
        if not imp.empty:
            b = imp.groupby("course_name")[["baseline_pct", "latest_pct"]].mean().round(1).reset_index().melt(id_vars="course_name")
            C.chart(px.bar(b, x="value", y="course_name", color="variable", barmode="group", orientation="h"),
                    "Before vs after: first-2 vs latest-2 assessments",
                    f"{len(imp):,} learners with ≥3 assessments; {(imp.change_pp > 0).mean():.0%} improved, "
                    f"{(imp.change_pp <= -10).mean():.0%} declined ≥10 pp.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**By cohort (enrollment month)**")
        coh = a.merge(m.students[["student_id", "cohort"]], on="student_id").groupby("cohort").pct.agg(["mean", "count"]).round(1).reset_index()
        C.chart(px.line(coh, x="cohort", y="mean", markers=True, hover_data=["count"]), "Cohort performance comparison", legend=False)
    with c2:
        st.markdown("**By batch**")
        C.table(A.by_group(m, "batch_id").sort_values("avg_score_pct"), height=300)
    st.markdown("#### Topic accuracy & difficulty")
    C.table(ta)
    st.markdown("#### Learners needing academic intervention")
    C.table(iv, "Synthetic IDs only. Flags are rules, not predictions.")


def retention(m: Model, ins: list) -> None:
    cb = A.churn_by(m, "acquisition_channel")
    C.five_questions(_kpis(m, ["retention_rate", "churn_confirmed", "inactive_rate", "completion_rate"]),
                     ("Churn is split into **confirmed cancellations** and **inferred 30-day inactivity**, by channel:", cb),
                     ("Inactivity by engagement segment & course:", A.churn_by(m, "course_name").head(6)),
                     _next_text(m, "active_learners", "Weekly active learners"),
                     _actions_for(ins, ["retention", "inactiv", "churn", "cancel"]))
    st.info("**Definitions.** *Confirmed churn* = enrollment cancelled (with refund rules). *Inferred inactivity* = open enrollment with no "
            "activity for 30+ days — a warning sign, **not** churn; 12% of synthetic learners reactivate after a gap. "
            "Learners are never labelled churned for missing a few classes.")
    mat = A.cohort_matrix(m)
    if mat.empty:
        C.empty("No enrolled learners in scope.")
        return
    C.heatmap(mat.drop(columns="cohort_size"), "Enrollment-cohort retention (% active in month k)", "k = months since enrollment month. "
              "Cohort sizes: " + ", ".join(f"{c}: {int(n)}" for c, n in mat.cohort_size.items()), fmt=".0f", height=420)
    c1, c2 = st.columns(2)
    with c1:
        rc = A.retention_curve(m, "M", 12)
        C.chart(px.line(rc, x="period_since_enrollment", y="retained_pct", markers=True, hover_data=["eligible_learners"]),
                "Retention curve (months since enrollment)", legend=False)
    with c2:
        ch = pd.DataFrame({"Confirmed churn %": M.monthly(m, "churn_confirmed").set_index("month").value,
                           "Inferred inactive (30d) %": M.monthly(m, "inactive_rate").set_index("month").value}).reset_index()
        _monthly_line(ch, ["Confirmed churn %", "Inferred inactive (30d) %"], "Churn trend (monthly)")
    c1, c2 = st.columns(2)
    with c1:
        C.chart(px.bar(cb, x="acquisition_channel", y=["confirmed_cancel_pct", "inferred_inactive_pct"], barmode="group"), "Churn by channel")
    with c2:
        seg = A.segments(m)
        if not seg.empty:
            sc = seg.segment.value_counts().reset_index()
            sc.columns = ["segment", "learners"]
            C.chart(px.pie(sc, names="segment", values="learners", hole=0.5), "At-risk distribution (engagement segments)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Renewals by course** (completed ≥14 days before period end)")
        C.table(A.renewals(m))
    with c2:
        st.markdown("**Enrollment status breakdown**")
        C.chart(px.pie(A.status_breakdown(m), names="enrollment_status", values="enrollments", hole=0.5), "Renewal & cancellation breakdown")
        st.metric("Reactivated learners in period (≥30-day gap, then active)", A.reactivations(m))
    tti = A.time_to_inactivity(m)
    if len(tti) >= 30:
        C.chart(px.histogram(tti, x="days_enrolled_before_last_activity", nbins=30), "Time to inactivity",
                f"Days from enrollment to last activity for {len(tti):,} currently inactive (30d+) learners. Median {tti.iloc[:, 0].median():.0f} days.", legend=False)
    st.markdown("#### Churn by cohort")
    C.table(A.churn_by(m, "cohort"))


def student_success(m: Model, ins: list) -> None:
    r = A.student_risk(m)
    if r.empty:
        C.empty("No open learners in the selected filters.")
        return
    ss = st.session_state.setdefault("followups", {})
    r["follow_up_status"] = r.student_id.map(ss).fillna("Not started")
    hi, med = (r.risk_level == "High").sum(), (r.risk_level == "Medium").sum()
    C.five_questions(_kpis(m, ["inactive_rate", "attendance_rate", "avg_score", "retention_rate"]),
                     ("Most common contributing indicators among flagged learners:",
                      r.indicators.str.split(", ").explode().replace("", np.nan).dropna().value_counts().rename_axis("indicator").reset_index(name="learners")),
                     (f"**{hi} High** and **{med} Medium** risk learners out of {len(r):,} open learners.",
                      r.groupby(["course", "risk_level"]).size().unstack(fill_value=0).reset_index()),
                     "Risk flags look ahead ~30–60 days; see the backtest below for how often flags were right historically.",
                     ["Contact High-risk learners within 48 hours using the recommended intervention.",
                      "Review Medium-risk learners weekly; escalate if a second indicator appears.",
                      "Record follow-up status below so the team can track coverage."])
    c1, c2, c3 = st.columns(3)
    c1.metric("High risk", int(hi))
    c2.metric("Medium risk", int(med))
    c3.metric("Follow-ups recorded", sum(v != "Not started" for v in ss.values()))
    level = st.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium"])
    view = r[r.risk_level.isin(level)].head(500)
    edited = st.data_editor(view, hide_index=True, width="stretch", disabled=[c for c in view.columns if c != "follow_up_status"],
                            column_config={"follow_up_status": st.column_config.SelectboxColumn(
                                "Follow-up status", options=["Not started", "Contacted", "In progress", "Resolved", "No response"])},
                            key="risk_editor")
    for sid, stt in zip(edited.student_id, edited.follow_up_status):
        if stt != "Not started" or sid in ss:
            ss[sid] = stt
    st.caption("Synthetic student IDs only. Follow-up status is stored for this session. No personal or sensitive attributes are used.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Risk indicators (rules-based, transparent)**")
        C.table(pd.DataFrame([(k, p, d, A.INTERVENTION[k]) for k, (p, d) in A.INDICATORS.items()],
                             columns=["indicator", "points", "rule", "recommended intervention"]),
                "Risk level: High ≥ 5 points, Medium 3–4, Low 1–2.")
    with c2:
        st.markdown("**Time-based backtest of the rules**")
        bt = A.risk_backtest(m)
        if bt:
            st.markdown(f"Flags computed as of **{bt['as_of']:%d %b %Y}** for {bt['learners']:,} learners; outcome = cancelled or no activity in "
                        f"days 31–60 after that date (base rate {bt['base_rate_pct']}%).")
            C.table(pd.DataFrame([{"flag": k, **v} for k, v in bt.items() if isinstance(v, dict)]))
            st.caption("This validates the **rules** on held-out future data; it is not a trained ML model. Precision = share of flagged learners "
                       "who disengaged; recall = share of disengaged learners who were flagged.")


# =========================================================== Business operations
def revenue(m: Model, ins: list) -> None:
    rb = A.revenue_by(m, "course_name")
    C.five_questions(_kpis(m, ["net_revenue", "collections", "refunds", "gross_bookings"]),
                     ("Revenue change by course (current vs previous period):", _rev_driver(m)),
                     ("Revenue concentration:", rb.head(5)),
                     _next_text(m, "net_revenue", "Net revenue (₹)"),
                     _actions_for(ins, ["revenue", "refund", "payment", "collection"]))
    C.kpi_strip(_kpis(m, ["arps", "aov", "refund_rate", "payment_failure_rate", "cac", "marketing_spend"]))
    c1, c2 = st.columns(2)
    with c1:
        _monthly_line(A.revenue_monthly(m), ["gross_bookings", "collections", "refunds", "net_revenue"], "Monthly revenue", money=True)
    with c2:
        wf = A.waterfall(m)
        if not wf.empty:
            fig = go.Figure(go.Waterfall(x=wf.step, y=wf.amount, measure=wf.measure, connector={"line": {"color": "#94A3B8"}},
                                         increasing={"marker": {"color": C.PALETTE[1]}}, decreasing={"marker": {"color": "#E11D48"}},
                                         totals={"marker": {"color": C.PALETTE[0]}}))
            fig.update_yaxes(tickprefix="₹", tickformat=",.2s")
            C.chart(fig, "Revenue waterfall (enrollments made in period)", "Actuals. 'Not yet collected' = scheduled installments, pending and failed payments.", legend=False)
    c1, c2 = st.columns(2)
    with c1:
        C.chart(px.bar(rb.sort_values("net_revenue"), x="net_revenue", y="course_name", orientation="h", text="share_pct"),
                "Net revenue by course", "Label = share of total.", legend=False)
    with c2:
        ps = A.payment_status(m)
        C.chart(px.bar(ps, x="payment_status", y="transactions", color="payment_status", hover_data=["amount", "refund_amount"]),
                "Payment collection status (transactions in period)", legend=False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Discount effectiveness**")
        C.table(A.discount_effectiveness(m), "Discounts are only observed on enrolled learners, so their effect on conversion cannot be measured here.")
        st.markdown("**Refunds by course**")
        C.table(A.cancellations(m)[["course_name", "enrollments", "cancelled", "refunded_inr"]] if not A.cancellations(m).empty else None)
    with c2:
        st.markdown("**Unit economics by channel**")
        ue = A.unit_economics(m)
        C.table(ue, "CAC = recorded spend / new enrollments in period (Referral & Organic have no recorded spend). "
                    "Observed revenue per learner = cash to date for learners enrolled ≥180 days ago — a **lower bound** on LTV, not a projection.")
        ren = M.compute(m, "gross_bookings")
        renewal = Model.window(m.enrollments[m.enrollments.is_renewal.astype(bool)], "enrollment_date", m.start, m.end).price.sum()
        st.markdown(f"- **Renewal bookings:** {M.fmt(renewal, 'inr')} ({renewal / ren * 100 if ren else 0:.1f}% of gross bookings)\n"
                    f"- **Recognised revenue (estimate):** {M.fmt(A.recognized_revenue(m), 'inr')} — straight-line over course duration; an assumption, not accounting output.\n"
                    "- **MRR:** not applicable — all plans in this data are one-time course fees (installments are not subscriptions).\n"
                    "- **Contribution margin:** not available — no delivery/teaching cost data. Not estimated.")


def _rev_driver(m: Model) -> pd.DataFrame | None:
    pw = m.prev_window()
    if pw is None:
        return None
    from ..edtech.insights import _shift
    cur = A.revenue_by(m, "course_name").set_index("course_name").net_revenue
    prev = A.revenue_by(_shift(m, *pw), "course_name").set_index("course_name").net_revenue
    return A.driver_table(cur, prev, "course").round(0)


def marketing(m: Model, ins: list) -> None:
    cp = A.channel_performance(m)
    C.five_questions(_kpis(m, ["leads", "lead_to_enrol", "cac", "marketing_spend"]),
                     ("Channel-level conversion and cost:", cp[["acquisition_channel", "leads", "lead_to_enrol_pct", "cpl", "cac", "roas_to_date"]] if not cp.empty else None),
                     ("Campaigns in period:", A.channel_performance(m, "campaign")[["campaign", "leads", "lead_to_enrol_pct", "cac"]] if not cp.empty else None),
                     _next_text(m, "enrollments", "New enrollments"),
                     _actions_for(ins, ["lead", "acquisition", "cac", "channel", "conversion"]))
    if cp.empty:
        C.empty("No leads in the selected period.")
        return
    st.warning("Channel comparisons are **correlational**. Differences in conversion or retention reflect who each channel attracts; "
               "they are not incremental lift. Causal claims need a holdout experiment or an attribution design.")
    c1, c2 = st.columns(2)
    with c1:
        fn = cp[["acquisition_channel", "leads", "enrolled"]].melt(id_vars="acquisition_channel")
        C.chart(px.bar(fn, x="acquisition_channel", y="value", color="variable", barmode="group"), "Marketing funnel by channel")
    with c2:
        sv = A.spend_vs_revenue_monthly(m)
        _monthly_line(sv, ["spend", "net_revenue"], "Spend vs net revenue (monthly)", "Revenue includes all channels incl. unpaid ones.", money=True)
    full = A.channel_performance(type(m)(m.tables, type(m.f)(m.data_start, m.f.end, m.f.courses, m.f.subjects, m.f.cohorts, m.f.channels, m.f.statuses)))
    c1, c2 = st.columns(2)
    with c1:
        q = full.dropna(subset=["retained_day60_90_pct"])
        C.chart(px.scatter(q, x="lead_to_enrol_pct", y="retained_day60_90_pct", size="leads", color="acquisition_channel", text="acquisition_channel",
                           hover_data=["mature_learners_n", "refund_rate_pct"]),
                "Channel quality vs conversion", "All lead history up to period end (so learners have 90 days). Bubble = leads.", legend=False)
    with c2:
        cm = A.channel_performance(m, "campaign")
        cm = cm.dropna(subset=["spend"])
        C.chart(px.scatter(cm, x="cac", y="lead_to_enrol_pct", size="spend", color="campaign", hover_data=["leads", "roas_to_date"]),
                "Campaign performance matrix (paid campaigns)", "Lower-right = expensive and poorly converting.", legend=False)
    st.markdown("#### Acquisition source comparison")
    C.table(cp, "cpl = spend/leads; cac = spend/enrolled; roas_to_date = net revenue to date from these leads / spend. "
                "retained_day60_90 needs leads ≥90 days old (often n/a for recent periods).")
    st.markdown("#### Lead quality & retention by source (all history)")
    C.table(full[["acquisition_channel", "leads", "qualified_pct", "lead_to_enrol_pct", "refund_rate_pct", "retained_day60_90_pct", "mature_learners_n"]])


def courses(m: Model, ins: list) -> None:
    q = A.course_quadrant(m)
    im = A.instructor_matrix(m)
    C.five_questions(_kpis(m, ["new_enrollments", "completion_rate", "avg_score", "satisfaction"]),
                     ("Quality index = mean z-score of completion, score, attendance and satisfaction:", q[["course", "level", "growth_pct", "quality_index"]] if not q.empty else None),
                     ("Courses below average quality:", q[q.quality_index < 0][["course", "completion_pct", "avg_score_pct", "satisfaction"]] if not q.empty else None),
                     _next_text(m, "completions", "Course completions"),
                     _actions_for(ins, ["completion", "score", "course", "satisfaction"]))
    if q.empty:
        C.empty("No course data in scope.")
        return
    st.markdown("#### Course performance scorecard")
    C.table(q, "completion/satisfaction/refund use all history to period end (enough matured enrollments); the others use the selected period.")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(q, x="growth_pct", y="quality_index", size="active_learners", color="level", text="course", hover_data=["new_enrollments"])
        fig.add_hline(y=0, line_dash="dot")
        fig.add_vline(x=0, line_dash="dot")
        C.chart(fig, "Course growth × quality quadrant", "Growth = enrollments last 90 days vs prior 90. Top-right: scale; bottom-right: fix quality before scaling.", height=420)
    with c2:
        C.chart(px.scatter(q, x="new_enrollments", y="completion_pct", size="net_revenue", color="level", text="course"),
                "Enrollment vs completion", height=420)
    cols = ["attendance_pct", "completion_pct", "avg_score_pct", "satisfaction", "refund_rate_pct", "inactive_30d_pct"]
    h = q.set_index("course")[cols]
    C.heatmap(h, "Course-level KPI heatmap (raw values)", fmt=".1f", height=420)
    st.markdown("#### Instructor comparison matrix")
    st.caption("Fair comparison: **score vs course average** is each learner's score minus the mean for the same course and topic "
               "(adjusts for course level and topic difficulty), with a 95% interval. Instructors are **not** ranked by raw scores; "
               "'Not distinguishable' means the interval includes zero.")
    C.table(im)
    if not im.empty:
        imv = im.dropna(subset=["score_vs_course_avg_pp"])
        fig = px.scatter(imv, x="classes_taught", y="score_vs_course_avg_pp", error_y="ci95_pp", color="subject", text="instructor_name",
                         hover_data=["learners_assessed", "attendance_pct"])
        fig.add_hline(y=0, line_dash="dot")
        C.chart(fig, "Instructor workload vs learner outcome uplift", "Only instructors with an in-course peer are shown.")


def support(m: Model, ins: list) -> None:
    s = A.support_summary(m)
    C.five_questions(_kpis(m, ["tickets", "sla_compliance", "csat", "nps"]),
                     ("Top contact reasons (Pareto):", A.pareto(m)),
                     ("Support vs later engagement (association):", A.support_engagement(m)),
                     _next_text(m, "tickets", "Support tickets"),
                     _actions_for(ins, ["ticket", "sla", "csat", "support", "promoter"]))
    if not s:
        C.empty("No tickets in the selected period.")
        return
    cols = st.columns(6)
    for c, (lbl, v) in zip(cols, [("Tickets", f"{s['tickets']:,}"), ("Median first response", f"{s['median_first_response_h']:.1f} h"),
                                  ("Median resolution", f"{s['median_resolution_h']:.1f} h"), ("SLA met", f"{s['sla_pct']:.1f}%"),
                                  ("Reopen rate", f"{s['reopen_pct']:.1f}%"), ("Backlog at period end", f"{s['backlog']:,}")]):
        c.metric(lbl, v)
    st.caption(f"Repeat-contact rate (another ticket within 30 days): {s['repeat_contact_pct']:.1f}% of ticket-raising learners. "
               f"CSAT {s['csat']:.2f}/5 from {s['csat_responses']:,} ratings." if s.get("csat") else "")
    c1, c2 = st.columns(2)
    t = A.tickets_w(m)
    with c1:
        wk = t.groupby([t.created_at.dt.to_period("W").dt.start_time, "issue_category"]).size().reset_index(name="tickets")
        wk.columns = ["week", "issue_category", "tickets"]
        C.chart(px.bar(wk, x="week", y="tickets", color="issue_category"), "Ticket volume trend")
    with c2:
        p = A.pareto(m)
        fig = go.Figure()
        fig.add_bar(x=p.issue_category, y=p.tickets, name="Tickets", marker_color=C.PALETTE[0])
        fig.add_scatter(x=p.issue_category, y=p.cumulative_pct, name="Cumulative %", yaxis="y2", mode="lines+markers", line_color="#E11D48")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%", showgrid=False))
        C.chart(fig, "Contact reasons (Pareto)")
    c1, c2 = st.columns(2)
    with c1:
        C.chart(px.bar(A.sla_by_priority(m), x="priority", y="sla_met_pct", text="sla_met_pct", hover_data=["tickets", "sla_hours", "median_first_response_h"]),
                "SLA performance by priority", "First-response SLA: Urgent 2h, High 4h, Medium 12h, Low 24h.", legend=False)
    with c2:
        C.chart(px.histogram(t.dropna(subset=["resolution_h"]), x="resolution_h", nbins=50), "Resolution-time distribution (hours)", legend=False)
    c1, c2 = st.columns(2)
    with c1:
        C.chart(px.bar(A.csat_vs_resolution(m), x="resolution_band", y="avg_csat", text="avg_csat", hover_data=["rated_tickets"]),
                "CSAT vs resolution time", legend=False)
    with c2:
        se = A.support_engagement(m)
        if not se.empty:
            C.chart(px.bar(se.melt(id_vars=["group", "learners"], value_vars=["avg_active_days_before", "avg_active_days_after"]),
                           x="group", y="value", color="variable", barmode="group", hover_data=["learners"]),
                    "Support-to-engagement analysis", "Active days in the 30 days before vs after a learner's first ticket in period. "
                    "Association only — support issues are not shown to cause churn.")
    st.markdown("#### Support by course")
    C.table(t.groupby("course_name").agg(tickets=("ticket_id", "count"), median_resolution_h=("resolution_h", "median"),
                                         sla_met_pct=("sla_met", lambda x: round(x.mean() * 100, 1)), csat=("csat_score", "mean")).round(2).reset_index())


# =========================================================== Intelligence
def ai_analyst(m: Model, ins: list) -> None:
    ss = st.session_state
    ss.setdefault("ed_chat", [])
    st.caption("Ask business questions in plain English. Answers are computed from the loaded data under the current global filters, "
               "with summary, findings, metrics, evidence, explanations (hypotheses), actions, confidence and scope.")
    cols = st.columns(4)
    clicked = None
    for n, ex in enumerate(X.EXAMPLES):
        if cols[n % 4].button(ex, key=f"ex{n}", width="stretch"):
            clicked = ex
    if st.button("Clear conversation"):
        ss.ed_chat = []
    for role, content in ss.ed_chat:
        with st.chat_message(role):
            _render_answer(content) if role == "assistant" else st.markdown(content)
    q = st.chat_input("e.g. Which acquisition channels produce students who stay longer?") or clicked
    if q:
        ss.ed_chat.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)
        prev = next((c for r, c in reversed(ss.ed_chat[:-1]) if r == "assistant" and c.intent), None)
        with st.spinner("Computing from data…"):
            a = X.ask(m, q, prev)
        ss.ed_chat.append(("assistant", a))
        with st.chat_message("assistant"):
            _render_answer(a)


def _render_answer(a: X.EdAnswer) -> None:
    st.markdown(f"**Executive summary.** {a.summary}")
    if a.findings:
        st.markdown("**Key findings**\n" + "\n".join(f"- {f}" for f in a.findings))
    if a.metrics:
        cols = st.columns(min(len(a.metrics), 4))
        for c, (k, v) in zip(cols * 2, a.metrics):
            c.metric(k, v)
    if a.table is not None and not a.table.empty:
        t = a.table
        num = t.select_dtypes("number").columns
        if a.chart == "bar" and len(num) and 1 < len(t) <= 30:
            st.bar_chart(t.set_index(t.columns[0])[num[0]])
        elif a.chart == "line" and "baseline" in t.columns:
            st.line_chart(t.set_index("week")[["baseline", "lo80", "hi80"]])
        st.dataframe(t, hide_index=True, width="stretch")
    if a.explanations:
        st.markdown("**Possible explanations (hypotheses, not verified)**\n" + "\n".join(f"- {e}" for e in a.explanations))
    if a.actions:
        st.markdown("**Recommended actions**\n" + "\n".join(f"{n + 1}. {x}" for n, x in enumerate(a.actions)))
    with st.expander("Confidence, limitations, scope & computation"):
        st.markdown(f"- **Confidence:** {a.confidence}\n" + "".join(f"- **Limitation:** {x}\n" for x in a.limitations) +
                    f"- **Scope:** {a.scope}\n- **Computation:** {a.computation}")
        if a.sql:
            st.markdown("**SQL executed (read-only, validated):**")
            st.code(a.sql, language="sql")


def forecasting(m: Model, ins: list) -> None:
    labels = {"enrollments": "New enrollments", "net_revenue": "Net revenue (₹)", "active_learners": "Weekly active learners",
              "completions": "Course completions", "tickets": "Support tickets"}
    c1, c2, c3 = st.columns([2, 1, 1])
    what = c1.selectbox("Series", list(labels), format_func=labels.get)
    horizon = c2.slider("Horizon (weeks)", 4, 16, 12)
    pct = c3.slider("Scenario spread (±%)", 5, 30, 10)
    fc = F.forecast(m, what, horizon, pct)
    if fc.forecast is None:
        st.warning(fc.warning)
        if not fc.history.empty:
            st.line_chart(fc.history.set_index("week").actual)
        return
    f = fc.forecast
    fig = go.Figure()
    fig.add_scatter(x=fc.history.week, y=fc.history.actual, name="Actual", mode="lines+markers", line_color=C.PALETTE[0])
    fig.add_scatter(x=pd.concat([f.week, f.week[::-1]]), y=pd.concat([f.hi95, f.lo95[::-1]]), fill="toself", fillcolor="rgba(79,70,229,.08)",
                    line=dict(width=0), name="95% interval", hoverinfo="skip")
    fig.add_scatter(x=pd.concat([f.week, f.week[::-1]]), y=pd.concat([f.hi80, f.lo80[::-1]]), fill="toself", fillcolor="rgba(79,70,229,.18)",
                    line=dict(width=0), name="80% interval", hoverinfo="skip")
    fig.add_scatter(x=f.week, y=f.baseline, name="Baseline", line=dict(color=C.PALETTE[0], dash="dash"))
    fig.add_scatter(x=f.week, y=f.conservative, name=f"Conservative (−{pct}%)", line=dict(color="#E11D48", dash="dot"))
    fig.add_scatter(x=f.week, y=f.optimistic, name=f"Optimistic (+{pct}%)", line=dict(color=C.PALETTE[6], dash="dot"))
    C.chart(fig, f"{labels[what]}: actuals and {horizon}-week forecast", height=420)
    if fc.warning:
        st.warning(fc.warning)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Scenario comparison (sum over horizon)**")
        C.table(pd.DataFrame({"scenario": ["Conservative", "Baseline", "Optimistic", "80% interval"],
                              "total": [f"{f.conservative.sum():,.0f}", f"{f.baseline.sum():,.0f}", f"{f.optimistic.sum():,.0f}",
                                        f"{f.lo80.sum():,.0f} – {f.hi80.sum():,.0f}"]}))
        st.markdown("**Backtest (8-week holdout)**")
        C.table(pd.DataFrame([{"model": k, **v, "chosen": k == fc.backtest["chosen_model"]} for k, v in fc.backtest["candidates"].items()]))
    with c2:
        st.markdown("**Assumptions**\n" + "\n".join(f"- {a}" for a in fc.assumptions))
    with st.expander("Forecast table"):
        C.table(f)


def data_explorer(m: Model, ins: list) -> None:
    tabs = X.explorer_tables(m)
    c1, c2 = st.columns([1, 3])
    name = c1.selectbox("Table", list(tabs) + ["learner_fact (derived)"])
    df = X.student_fact(m) if name.startswith("learner_fact") else tabs[name]
    c1.caption(f"{len(df):,} rows × {df.shape[1]} columns (after global filters)")
    with c2:
        st.dataframe(df.head(1000), width="stretch", height=320)
    st.markdown("#### SQL workbench (read-only, all 12 tables)")
    st.caption("Tables: " + ", ".join(f"`{t}`" for t in tabs) + ". Single SELECT/WITH only; writes, file, network and settings are blocked. Max 10,000 rows.")
    sql = st.text_area("SQL", "SELECT c.course_name, count(*) AS enrollments, round(avg(e.price)) AS avg_price\n"
                              "FROM enrollments e JOIN courses c USING (course_id)\nGROUP BY 1 ORDER BY 2 DESC", height=120)
    if st.button("Run query", type="primary"):
        try:
            st.dataframe(sql_guard.run_multi(sql, tabs), width="stretch", hide_index=True)
            st.success("Validated read-only query executed.")
        except sql_guard.UnsafeSQLError as e:
            st.error(f"Blocked: {e}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Query error: {e}")


def data_quality(m: Model, ins: list) -> None:
    fixture = st.toggle("Run checks on the test fixture (copy of demo data with injected issues)", value=False,
                        help="The executive demo data is kept clean. The fixture shows what the checks catch; source data is never modified.")
    t = make_dirty(m.tables) if fixture else m.tables
    checks = G.run_checks(t)
    fresh = G.freshness(t)
    lag = None  # demo snapshot: freshness reported, not penalised
    total, dims = G.score(checks, lag)
    c1, c2 = st.columns([1, 3])
    c1.metric("Data-quality score", f"{total}/100")
    c1.caption("Weighted: " + ", ".join(f"{k} {v}%" for k, v in G.WEIGHTS.items()))
    with c2:
        C.table(dims)
    bad = [c for c in checks if c.failing]
    st.markdown(f"#### Checks: {len(checks)} run · {sum(c.status == 'PASS' for c in checks)} pass · {len(bad)} with issues")
    C.table(pd.DataFrame([{"dimension": c.dimension, "table": c.table, "check": c.check, "rule": c.rule, "checked_rows": c.checked,
                           "failing_rows": c.failing, "status": c.status} for c in sorted(checks, key=lambda c: c.status)]), height=320)
    for c in bad:
        with st.expander(f"{c.status} · {c.table} · {c.check} ({c.failing} rows)"):
            C.table(c.sample)
    st.markdown("#### Safe, previewable cleaning actions")
    st.caption("Previews never modify source data. Download a cleaned copy if you want to use it.")
    act = st.selectbox("Action", G.ACTIONS)
    tab, affected, cleaned = G.preview_action(t, act)
    st.write(f"**{len(affected)}** row(s) in `{tab}` would change; cleaned copy has {len(cleaned):,} rows (source has {len(t[tab]):,}).")
    if len(affected):
        C.table(affected.head(20))
        st.download_button(f"Download cleaned {tab}.csv", cleaned.to_csv(index=False).encode(), f"{tab}_cleaned.csv", "text/csv")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Data freshness")
        C.table(fresh, "Demo data is a synthetic snapshot ending 30 Sep 2026; freshness is shown but not penalised.")
    with c2:
        st.markdown("#### Lineage (metric → source)")
        C.table(pd.DataFrame(G.LINEAGE, columns=["metrics", "source tables"]))
    with st.expander("Schema & keys"):
        C.table(G.schema_report(t))


# =========================================================== Administration
def metric_dictionary(m: Model, ins: list) -> None:
    st.markdown("One definition per metric, used by every page, insight and AI answer (`vdl/edtech/metrics.py`).")
    C.table(M.dictionary())
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Engagement segments")
        C.table(pd.DataFrame(A.SEGMENT_RULES, columns=["segment", "rule"]))
        st.markdown("#### Churn definitions")
        st.markdown("- **Confirmed churn:** enrollment cancelled (`enrollment_status = Cancelled`).\n"
                    "- **Inferred inactivity:** open enrollment, ≥30 days old, no activity in the last 30 days. Not churn.\n"
                    "- **Retention:** active in the previous equal-length period and again in this one.\n"
                    "- **Active:** ≥1 learning activity or attended live class in the period.")
    with c2:
        st.markdown("#### Risk indicators")
        C.table(pd.DataFrame([(k, p, d) for k, (p, d) in A.INDICATORS.items()], columns=["indicator", "points", "rule"]))
        st.markdown("#### Reporting conventions")
        st.markdown("- Period = global date range; baseline = previous period of equal length (omitted if before data start).\n"
                    "- Rates show their denominator in the definition; changes in rates are in percentage points (pp).\n"
                    "- Currency is INR; L = lakh (1e5), Cr = crore (1e7).\n- Associations are labelled and never presented as causes.")


def demo_management(m: Model, ins: list) -> None:
    ss = st.session_state
    st.markdown(f"**Demo mode** uses 12 interconnected synthetic tables (fictional learners), seed `{ss.get('seed', SEED)}`, "
                "12 months from 1 Oct 2025 to 30 Sep 2026. Stored locally in `data/edtech/`.")
    C.table(pd.DataFrame([{"table": k, "rows": len(v), "columns": v.shape[1]} for k, v in m.tables.items()]))
    c1, c2, c3 = st.columns(3)
    if c1.button("⟳ Load demo data", type="primary", width="stretch"):
        ss.reload_demo = True
        st.rerun()
    seed = c2.number_input("Seed", value=int(ss.get("seed", SEED)), step=1)
    if c3.button("Regenerate demo data", width="stretch"):
        ss.regen_seed = int(seed)
        st.rerun()
    st.caption("Regenerating writes new CSVs (about 10–20 s). The default seed always reproduces the same data.")
