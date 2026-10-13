"""EdTech AI Analyst: business questions → computed analyses → structured, grounded answers.

Deterministic routing (no API key needed). Each answer has: executive summary, key findings, metrics,
a supporting table, possible explanations (labelled hypotheses), actions, confidence & limitations, and
the scope used (filters, dates, sample size). SQL is shown only when SQL actually ran (the per-learner
fallback); otherwise the computation names the pandas function that produced the numbers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .. import copilot
from ..profiling import prepare
from . import analytics as A
from . import insights as I
from . import metrics as M
from .forecast import forecast
from .model import Filters, Model


@dataclass
class EdAnswer:
    question: str
    summary: str
    findings: list[str] = field(default_factory=list)
    metrics: list[tuple[str, str]] = field(default_factory=list)
    table: pd.DataFrame | None = None
    explanations: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    confidence: str = "Medium"
    limitations: list[str] = field(default_factory=list)
    scope: str = ""
    computation: str = ""
    sql: str | None = None
    intent: str | None = None
    extra_filter: dict = field(default_factory=dict)
    chart: str | None = None  # hint: bar | line


EXAMPLES = [
    "Why did enrollment decline this month?",
    "Which courses have the highest completion rates?",
    "Which cohorts have the worst retention?",
    "Which topics have the lowest student accuracy?",
    "Which acquisition channels produce students who stay longer?",
    "Which students need immediate academic support?",
    "What are the three biggest business risks this month?",
    "What should leadership prioritize next week?",
]


def _scope(m: Model, n: int, unit: str = "learners") -> str:
    return f"Filters: {m.f.describe()} · sample: {n:,} {unit}"


def _with(m: Model, **kw) -> Model:
    f = m.f
    d = dict(start=f.start, end=f.end, courses=f.courses, subjects=f.subjects, cohorts=f.cohorts, channels=f.channels, statuses=f.statuses)
    d.update(kw)
    return Model(m.tables, Filters(**d))


def _month_model(m: Model) -> Model:
    """'This month' = latest calendar month within the window (partial months flagged)."""
    s = max(m.end.replace(day=1), m.start)
    return _with(m, start=s, end=m.end)


# ---------------- intents ----------------
def enrollment_why(m: Model, q: str) -> EdAnswer:
    mm = _month_model(m) if re.search(r"this month|last month|month", q) else m
    k = M.kpi(mm, "new_enrollments")
    lk, ck = M.kpi(mm, "leads"), M.kpi(mm, "lead_to_enrol")
    drv = I._enrol_driver(mm)
    ans = EdAnswer(q, "", intent="enrollment_why", chart="bar")
    if k.change is None:
        ans.summary = "No comparable previous period exists in the data, so a change cannot be explained."
        ans.confidence = "Low"
        return ans
    dirn = "declined" if k.change < 0 else "rose"
    ans.summary = (f"New enrollments {dirn} {abs(k.change):.1f}% ({M.fmt(k.prev, 'count')} → {M.fmt(k.value, 'count')}) "
                   f"for {k.period} vs the {k.baseline}.")
    if lk.change is not None and ck.change is not None:
        ans.findings.append(f"Lead volume changed {M.fmt_change(lk)} ({M.fmt(lk.prev, 'count')} → {M.fmt(lk.value, 'count')}).")
        ans.findings.append(f"Lead-to-enrollment conversion changed {M.fmt_change(ck)} ({M.fmt(ck.prev, 'pct')} → {M.fmt(ck.value, 'pct')}).")
    if not drv.empty:
        neg = drv[drv.change < 0].head(3)
        for r in neg.itertuples():
            ans.findings.append(f"{r.channel}: {int(r.previous)} → {int(r.current)} enrollments ({r.change:+.0f}, {r.share_of_change:.0f}% of the net change).")
        ans.table = drv
    ans.metrics = [("New enrollments", M.fmt(k.value, "count")), ("Previous period", M.fmt(k.prev, "count")),
                   ("Leads", M.fmt(lk.value, "count")), ("Lead → enrollment", M.fmt(ck.value, "pct"))]
    ans.explanations = ["Seasonality: lead volume in this synthetic data peaks Mar–Jun and dips in Sep/Dec.",
                        "Channel conversion: a channel whose conversion fell (see table) may have weaker targeting or lead quality.",
                        "Maturity: leads from the last ~2 weeks may still enroll; recent conversion is understated."]
    ans.actions = ["Review the top negative channel's campaigns and lead qualification.",
                   "Compare counsellor follow-up times for recent leads.", "Re-check this view in 2 weeks once recent leads mature."]
    ans.limitations = ["Decomposition is descriptive; it identifies where the change happened, not why.",
                       "'This month' may be a partial month; the previous period has the same length."]
    ans.confidence = "High for the numbers; Low for explanations (hypotheses)"
    ans.scope = _scope(mm, int(k.value or 0), "new enrollments")
    ans.computation = "metrics.kpi('new_enrollments' | 'leads' | 'lead_to_enrol'); insights._enrol_driver (pandas, no SQL)"
    return ans


def completion_courses(m: Model, q: str) -> EdAnswer:
    sc = A.course_scorecard(m)
    ans = EdAnswer(q, "", intent="completion_courses", chart="bar")
    if not sc.empty:
        sc = sc.dropna(subset=["completion_pct"]).sort_values("completion_pct", ascending=False)
    if sc.empty:
        ans.summary = "No enrollments have reached their scheduled end date in this scope."
        return ans
    best, worst = sc.iloc[0], sc.iloc[-1]
    ans.summary = (f"{best.course} has the highest completion rate ({best.completion_pct:.1f}%); "
                   f"{worst.course} the lowest ({worst.completion_pct:.1f}%).")
    ans.findings = [f"{r.course}: {r.completion_pct:.1f}% completion, avg score {r.avg_score_pct:.1f}%" for r in sc.head(3).itertuples()]
    ans.table = sc[["course", "level", "weeks", "completion_pct", "avg_score_pct", "attendance_pct", "satisfaction"]]
    ans.metrics = [("Courses compared", str(len(sc))), ("Median completion", f"{sc.completion_pct.median():.1f}%")]
    ans.explanations = ["Shorter, foundation-level courses tend to show higher completion; compare like with like (level, weeks)."]
    ans.actions = [f"Study {best.course}'s structure (pacing, checkpoints) for transfer to similar-level courses.",
                   f"Add mid-course check-ins for {worst.course}."]
    ans.limitations = ["Completion uses first enrollments whose scheduled end date has passed (all history up to window end).",
                       "Course length and level differ; raw rankings are not like-for-like."]
    ans.confidence = "High (computed)"
    ans.scope = _scope(m, int(len(m.enrolled)))
    ans.computation = "analytics.course_scorecard → metrics.completion_rate per course (pandas, no SQL)"
    return ans


def cohort_retention(m: Model, q: str) -> EdAnswer:
    mat = A.cohort_matrix(m)
    ans = EdAnswer(q, "", intent="cohort_retention", chart="bar")
    if mat.empty or 3 not in mat.columns:
        ans.summary = "Not enough cohort history to compare retention."
        return ans
    r3 = mat[3].dropna()
    r3 = r3[mat.loc[r3.index, "cohort_size"] >= 30].sort_values()
    if r3.empty:
        ans.summary = "No cohort with ≥30 learners has 3 months of history yet."
        return ans
    ans.summary = (f"The {r3.index[0]} enrollment cohort has the worst month-3 retention ({r3.iloc[0]:.1f}% active), "
                   f"vs {r3.iloc[-1]:.1f}% for the best ({r3.index[-1]}).")
    ans.findings = [f"{c}: {v:.1f}% active in month 3 (cohort size {int(mat.loc[c, 'cohort_size'])})" for c, v in r3.head(3).items()]
    ans.table = mat.reset_index()
    ans.metrics = [("Cohorts with month-3 data", str(len(r3))), ("Median month-3 retention", f"{r3.median():.1f}%")]
    ans.explanations = ["Cohort mix (course, channel) differs by month; filter to a course or channel to compare like with like."]
    ans.actions = ["Investigate onboarding and batch timing for the weakest cohorts.", "Track month-1 activity as an early signal."]
    ans.limitations = ["Retention = any learning activity or attended class in month k after the enrollment month.",
                       "Inactive learners are not necessarily churned; cancellations are tracked separately."]
    ans.confidence = "High (computed)"
    ans.scope = _scope(m, int(mat.cohort_size.sum()), "enrolled learners")
    ans.computation = "analytics.cohort_matrix (pandas, no SQL)"
    return ans


def topic_accuracy(m: Model, q: str) -> EdAnswer:
    ta = A.topic_accuracy(m)
    ans = EdAnswer(q, "", intent="topic_accuracy", chart="bar")
    if ta.empty:
        ans.summary = "No assessments in this scope."
        return ans
    t = ta.iloc[0]
    ans.summary = f"Lowest accuracy: {t.topic} ({t.subject}) at {t.accuracy_pct:.1f}% correct, vs a median of {ta.accuracy_pct.median():.1f}% across topics."
    ans.findings = [f"{r.topic} ({r.subject}): {r.accuracy_pct:.1f}% accuracy over {int(r.attempts)} assessments" for r in ta.head(5).itertuples()]
    ans.table = ta.head(15)
    ans.metrics = [("Topics", str(len(ta))), ("Assessments", f"{int(ta.attempts.sum()):,}")]
    ans.explanations = ["Topic difficulty, question calibration, or teaching sequence may explain low accuracy (not tested)."]
    ans.actions = [f"Add worked examples and adaptive practice for {t.topic}.", "Review item difficulty for the bottom 3 topics."]
    ans.limitations = ["Accuracy = correct answers / questions attempted."]
    ans.confidence = "High (computed)"
    ans.scope = _scope(m, int(ta.attempts.sum()), "assessments")
    ans.computation = "analytics.topic_accuracy (pandas, no SQL)"
    return ans


def channel_retention(m: Model, q: str) -> EdAnswer:
    mm = _with(m, start=m.data_start)  # need mature learners: use all lead history up to window end
    cp = A.channel_performance(mm)
    ans = EdAnswer(q, "", intent="channel_retention", chart="bar")
    if cp.empty:
        ans.summary = "No leads in scope."
        return ans
    cp = cp.dropna(subset=["retained_day60_90_pct"])
    cp = cp[cp.mature_learners_n >= 30].sort_values("retained_day60_90_pct", ascending=False)
    if cp.empty:
        ans.summary = "Not enough learners with 90 days of history per channel."
        return ans
    b, w = cp.iloc[0], cp.iloc[-1]
    ans.summary = (f"{b.acquisition_channel} learners stay longest: {b.retained_day60_90_pct:.1f}% still active 60–90 days after enrolling, "
                   f"vs {w.retained_day60_90_pct:.1f}% for {w.acquisition_channel}.")
    ans.findings = [f"{r.acquisition_channel}: {r.retained_day60_90_pct:.1f}% retained (n={int(r.mature_learners_n)}), refund rate {r.refund_rate_pct:.1f}%" for r in cp.itertuples()]
    ans.table = cp[["acquisition_channel", "mature_learners_n", "retained_day60_90_pct", "refund_rate_pct", "lead_to_enrol_pct", "cac", "roas_to_date"]]
    ans.explanations = ["Channels attract different intent levels (e.g. referrals come pre-sold) — this is selection, not channel causal lift."]
    ans.actions = [f"Weight budget decisions on retention-adjusted value, not just CAC: favour {b.acquisition_channel}.",
                   f"Tighten qualification for {w.acquisition_channel} leads."]
    ans.limitations = ["Uses all lead history up to the window end so learners have 90 days of data.",
                       "Association only: no experiment or attribution model."]
    ans.confidence = "High for the numbers; channel causality not established"
    ans.scope = _scope(mm, int(cp.mature_learners_n.sum()), "learners with ≥90 days")
    ans.computation = "analytics.channel_performance (pandas, no SQL)"
    return ans


def students_support(m: Model, q: str) -> EdAnswer:
    r = A.student_risk(m)
    ans = EdAnswer(q, "", intent="students_support")
    if r.empty:
        ans.summary = "No open learners in scope."
        return ans
    hi = r[r.risk_level == "High"]
    acad = r[r.indicators.str.contains("assessment|pass", case=False)]
    ans.summary = (f"{len(hi)} learners are High risk and {len(acad)} have academic indicators (low or declining scores). "
                   "List below, highest risk first (synthetic IDs).")
    ans.findings = [f"{x.student_id} ({x.course}): {x.indicators}" for x in hi.head(5).itertuples()]
    ans.table = r[r.risk_level.isin(["High", "Medium"])].head(100)
    ans.metrics = [("High risk", str(len(hi))), ("Medium risk", str(int((r.risk_level == 'Medium').sum()))), ("Open learners", str(len(r)))]
    ans.actions = ["Assign High-risk learners to academic mentors this week.", "Use the recommended intervention column as the first step."]
    ans.limitations = ["Rules-based flags, not a trained prediction; see Student Success for the time-based backtest (precision/recall).",
                       "No personal or sensitive attributes are used."]
    ans.confidence = "Medium (transparent rules, backtested)"
    ans.scope = _scope(m, len(r), "open learners")
    ans.computation = "analytics.student_risk (pandas, no SQL)"
    return ans


def top_risks(m: Model, q: str, n: int = 3, generated: list | None = None) -> EdAnswer:
    mm = _month_model(m) if re.search(r"this month", q) else m
    ins = I.top(generated if generated is not None else I.generate(mm), "risk", n)
    ans = EdAnswer(q, "", intent="top_risks")
    if not ins:
        ans.summary = "No risks passed the thresholds for this scope."
        return ans
    ans.summary = f"Top {len(ins)} risks by impact score: " + "; ".join(i.title for i in ins) + "."
    ans.findings = [f"{i.title} — {i.observed} vs {i.baseline}. Segment: {i.segment}. [{i.basis}]" for i in ins]
    ans.actions = [i.action for i in ins]
    ans.explanations = [i.implication for i in ins]
    ans.table = pd.DataFrame([{"insight": i.title, "basis": i.basis, "impact_score": i.impact, "observed": i.observed,
                               "baseline": i.baseline} for i in ins])
    ans.limitations = ["Impact score = metric weight × change size × reach; it ranks attention, not certainty."]
    ans.confidence = "High for metrics; associations are labelled"
    ans.scope = _scope(mm, len(mm.enrolled), "enrolled learners")
    ans.computation = "insights.generate (metrics.kpi + analytics; pandas, no SQL)"
    return ans


def priorities(m: Model, q: str) -> EdAnswer:
    gen = I.generate(m)
    ans = top_risks(m, q, 3, gen)
    pos = I.top(gen, "positive", 1)
    ans.intent = "priorities"
    if ans.findings:
        ans.summary = "Recommended priorities for next week (highest-impact risks first): " + " | ".join(f"{k + 1}. {a}" for k, a in enumerate(ans.actions))
    if pos:
        ans.findings.append(f"Protect what works: {pos[0].title}.")
    return ans


def revenue(m: Model, q: str) -> EdAnswer:
    keys = ["net_revenue", "collections", "refunds", "gross_bookings", "refund_rate", "arps"]
    ks = [M.kpi(m, k) for k in keys]
    rb = A.revenue_by(m, "course_name")
    ans = EdAnswer(q, f"Net revenue (cash) was {M.fmt(ks[0].value, 'inr')} ({M.fmt_change(ks[0])} vs previous period).",
                   intent="revenue", chart="bar")
    ans.metrics = [(k.name, f"{M.fmt(k.value, k.unit)} ({M.fmt_change(k)})") for k in ks]
    if not rb.empty:
        ans.findings = [f"{r.course_name}: {M.fmt(r.net_revenue, 'inr')} ({r.share_pct:.1f}% of net revenue)" for r in rb.head(3).itertuples()]
        ans.table = rb
    ans.limitations = ["Cash basis (collections − refunds). Recognised revenue is an estimate on the Revenue page.",
                       "No cost data: margin and profitability are not available."]
    ans.confidence = "High (computed)"
    ans.scope = _scope(m, len(m.enrolled), "enrolled learners")
    ans.computation = "metrics.kpi; analytics.revenue_by (pandas, no SQL)"
    return ans


def forecast_answer(m: Model, q: str) -> EdAnswer:
    what = "net_revenue" if "revenue" in q else "tickets" if "ticket" in q else "active_learners" if "active" in q else "enrollments"
    fc = forecast(m, what)
    ans = EdAnswer(q, "", intent="forecast", chart="line")
    if fc.forecast is None:
        ans.summary = fc.warning or "Forecast unavailable."
        ans.confidence = "Low"
        return ans
    nxt = fc.forecast.head(4)
    ans.summary = (f"Next 4 weeks {what.replace('_', ' ')}: baseline {nxt.baseline.sum():,.0f} "
                   f"(80% range {nxt.lo80.sum():,.0f}–{nxt.hi80.sum():,.0f}).")
    ans.table = fc.forecast
    ans.metrics = [("Model", fc.backtest["chosen_model"]), ("Backtest MAE", str(fc.backtest["MAE"])),
                   ("Backtest MAPE", f"{fc.backtest.get('MAPE_pct')}%")]
    ans.limitations = fc.assumptions + ([fc.warning] if fc.warning else [])
    ans.confidence = "Medium" if not fc.warning else "Low"
    ans.scope = _scope(m, len(fc.history), "weeks of history")
    ans.computation = "forecast.forecast (backtested model selection, pandas/numpy, no SQL)"
    return ans


INTENTS = [
    (r"\bwhy\b.*\b(enrol|admission)\w*|\b(enrol|admission)\w*\b.*\b(declin|drop|down|fall|fell|decreas)", enrollment_why),
    (r"\bcomplet\w*", completion_courses),
    (r"\bcohort\w*", cohort_retention),
    (r"\btopic\w*|\baccuracy\b", topic_accuracy),
    (r"\b(channel|source)\w*\b.*\b(stay|retain|retention|longer|quality)|\b(stay|retain)\w*\b.*\b(channel|source)", channel_retention),
    (r"\b(student|learner)s?\b.*\b(need|support|intervention|help|at.risk)|\bat.risk\b", students_support),
    (r"\bprioriti[sz]\w*|\bpriorities\b|\bfocus\b|\bnext week\b", priorities),
    (r"\b(biggest|top|main|key|business|major|emerging)\b.*\brisks?\b|\brisks?\b.*\b(this month|business|facing)\b", top_risks),
    (r"\bforecast\w*|\bpredict\w*|\bnext (month|weeks?)\b", forecast_answer),
    (r"\brevenue\b|\bcollection\w*|\brefund\w*|\bbookings?\b", revenue),
]
FOLLOW = r"^\s*(and|what about|how about|only|now|same|instead|also|for)\b"


def student_fact(m: Model) -> pd.DataFrame:
    """One row per enrolled learner — the table behind the free-form (SQL) fallback."""
    seg = A.segments(m)
    if seg.empty:
        return seg
    risk = A.student_risk(m).set_index("student_id")
    s = m.enrolled.set_index("student_id")
    pay = m.payments
    net = (pay.amount.where(pay.payment_status == "Success", 0) - pay.refund_amount.where(pay.payment_status == "Refunded", 0)).groupby(pay.student_id).sum()
    sc = m.assessments.groupby("student_id").pct.mean()
    tk = m.tickets.groupby("student_id").size()
    f = seg.set_index("student_id")
    out = pd.DataFrame({"course": f.course, "subject": s.subject, "city": s.city, "channel": f.channel, "cohort": f.cohort,
                        "status": f.status, "segment": f.segment, "risk_level": f.index.map(risk.risk_level).fillna("None"),
                        "active_days_28d": f.active_days_28d, "attendance_28d_pct": f.attendance_28d_pct,
                        "avg_score_pct": f.index.map(sc).round(1), "net_revenue": f.index.map(net).fillna(0).round(0),
                        "tickets": f.index.map(tk).fillna(0).astype(int), "days_since_activity": f.days_since_activity})
    return out.reset_index()


def _entities(m: Model, q: str) -> dict:
    ql = q.lower()
    found = {}
    for c in m.courses.course_name:
        if c.lower() in ql:
            found["courses"] = (c,)
    for ch in sorted(set(m.tables["students"].acquisition_channel)):
        if ch.lower() in ql:
            found["channels"] = (ch,)
    for sub in sorted(set(m.courses.subject)):
        if re.search(rf"\b{re.escape(sub.lower())}\b", ql) and "courses" not in found:
            found["subjects"] = (sub,)
    return found


def ask(m: Model, q: str, prev: EdAnswer | None = None) -> EdAnswer:
    ql = q.lower()
    ents = _entities(m, q)
    if prev is not None and prev.intent and re.match(FOLLOW, ql) and ents:
        fn = dict((f.__name__, f) for _, f in INTENTS)[prev.intent] if prev.intent in {f.__name__ for _, f in INTENTS} else None
        if fn:
            merged = {**prev.extra_filter, **ents}
            a = fn(_with(m, **merged), prev.question)
            a.question, a.extra_filter = q, merged
            a.summary = f"(Follow-up applied: {', '.join(f'{k}={v[0]}' for k, v in merged.items())}) " + a.summary
            return a
    for pat, fn in INTENTS:
        if re.search(pat, ql):
            mm = _with(m, **ents) if ents else m
            a = fn(mm, q)
            a.extra_filter = ents
            return a
    # free-form fallback: deterministic NL→SQL over the best-matching table, then the per-learner table
    fact = student_fact(m)
    tables = {**explorer_tables(m), "learners": fact}
    for name in pick_tables(q, tables):
        t = tables[name]
        if t.empty:
            continue
        df, prof = prepare(t)
        a = copilot.ask(q, df, prof)
        if a.ok:
            return EdAnswer(q, a.text, table=a.table, sql=a.sql, intent=None, chart="bar" if a.table is not None and len(a.table) > 1 else None,
                            limitations=a.limitations + a.assumptions, confidence="High (computed by SQL)",
                            scope=_scope(m, len(t), f"rows of `{name}`"),
                            computation=f"copilot NL→SQL over table `{name}` (shown as `data`; DuckDB, read-only guard)")
    if fact.empty:
        return EdAnswer(q, "No enrolled learners in scope.", confidence="n/a")
    return EdAnswer(q, "I couldn't map that to the data, so I won't guess. Try a business question (" + " · ".join(EXAMPLES[:3]) +
                    ") or name a table and measure, e.g. *average csat_score by priority*, *total spend by channel*, "
                    "*count of tickets by issue_category*.", confidence="n/a", scope=_scope(m, len(fact)))


TABLE_WORDS = {
    "support_tickets": r"ticket|support|csat|complain|issue|priority|sla",
    "payments": r"payment|paid|refund|collect|transaction|failed",
    "marketing_spend": r"spend|campaign|impression|click|cpl|ad\b|ads\b|marketing",
    "learning_activity": r"activit|watch|video|minutes|content|practice|login",
    "attendance": r"attend|present|absent|live class",
    "assessments": r"assessment|quiz|test|exam|marks|correct|attempt",
    "classes": r"\bclass(es)?\b|session|capacity|batch",
    "student_surveys": r"survey|satisfaction|nps|recommend",
    "instructors": r"instructor|teacher|faculty|experience",
    "courses": r"\bcourses?\b.*\b(price|level|duration)|list.price|catalog",
    "enrollments": r"enrol|discount|renewal|cancel|complet",
    "students": r"lead|signup|sign.up|counsel|qualified|city|cities",
}


def pick_tables(q: str, tables: dict[str, pd.DataFrame]) -> list[str]:
    """Rank tables by keyword + column-name matches; the per-learner table is always a fallback."""
    ql = q.lower()
    scores = {}
    for name, df in tables.items():
        s = 2 * len(re.findall(TABLE_WORDS.get(name, r"$^"), ql))
        s += len(copilot.find_columns(" " + ql.replace("_", " ") + " ", list(df.columns)))
        if s:
            scores[name] = s
    ranked = sorted(scores, key=lambda n: -scores[n])
    return [n for n in ranked if n != "learners"][:3] + ["learners"]


def explorer_tables(m: Model) -> dict[str, pd.DataFrame]:
    """Filtered tables exposed (read-only) to the SQL workbench."""
    return {"students": m.students, "enrollments": m.enrollments, "courses": m.courses, "learning_activity": m.activity,
            "attendance": m.attendance, "assessments": m.assessments, "payments": m.payments, "support_tickets": m.tickets,
            "marketing_spend": m.marketing, "instructors": m.instructors, "classes": m.classes, "student_surveys": m.surveys}

