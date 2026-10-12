"""Single source of truth for EdTech KPIs. Every page, insight and AI answer calls these functions.

Each metric declares its definition, formula, denominator, unit and direction, and is computed for an
explicit window [start, end]. Comparisons use the previous window of equal length and are omitted (never
fabricated) when that window falls before the start of available data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .model import Model

PASS_MARK = 40.0  # % score
INACTIVE_DAYS = 30


@dataclass
class MetricDef:
    key: str
    name: str
    unit: str  # count | pct | inr | score | ratio | hours | nps
    higher_is_better: bool
    definition: str
    formula: str
    fn: Callable[[Model, pd.Timestamp, pd.Timestamp], float | None]


@dataclass
class KPI:
    key: str
    name: str
    value: float | None
    prev: float | None
    change: float | None  # relative change (or pp for pct/nps)
    rag: str  # green | amber | red | grey
    unit: str
    definition: str
    formula: str
    period: str
    baseline: str


# ---------------- metric functions ----------------
def _first_enrol(m: Model, s, e) -> pd.DataFrame:
    en = m.enrollments[~m.enrollments.is_renewal.astype(bool)]
    return Model.window(en, "enrollment_date", s, e)


def leads(m, s, e):
    return float(len(Model.window(m.students, "signup_date", s, e)))


def new_enrollments(m, s, e):
    return float(_first_enrol(m, s, e).student_id.nunique())


def total_enrolled(m, s, e):
    en = m.enrollments[m.enrollments.enrollment_date <= pd.Timestamp(e)]
    return float(en.student_id.nunique())


def lead_to_enrol(m, s, e):
    ld = Model.window(m.students, "signup_date", s, e)
    return None if ld.empty else float(ld.enrollment_date.notna().mean() * 100)


def active_learners(m, s, e):
    return float(Model.window(m.events(), "ts", s, e).student_id.nunique())


def _avg_active(m, s, e, freq):
    ev = Model.window(m.events(), "ts", s, e)
    if ev.empty:
        return 0.0
    return float(ev.groupby(ev.ts.dt.to_period(freq)).student_id.nunique().mean())


def dau(m, s, e):
    return _avg_active(m, s, e, "D")


def wau(m, s, e):
    return _avg_active(m, s, e, "W")


def mau(m, s, e):
    return _avg_active(m, s, e, "M")


def activation_rate(m, s, e):
    cutoff = min(pd.Timestamp(e), m.data_end - pd.Timedelta(days=14))
    en = _first_enrol(m, s, cutoff)
    if en.empty:
        return None
    ev = m.events().merge(en[["student_id", "enrollment_date"]], on="student_id")
    ok = ev[(ev.ts >= ev.enrollment_date) & (ev.ts < ev.enrollment_date + pd.Timedelta(days=14))].student_id.unique()
    return float(en.student_id.isin(ok).mean() * 100)


def completion_rate(m, s, e):
    en = m.enrollments[~m.enrollments.is_renewal.astype(bool)].merge(m.courses[["course_id", "duration_weeks"]], on="course_id")
    en = en.assign(sched_end=en.enrollment_date + pd.to_timedelta(en.duration_weeks * 7, unit="D"))
    en = Model.window(en, "sched_end", s, e)
    return None if en.empty else float((en.enrollment_status == "Completed").mean() * 100)


def attendance_rate(m, s, e):
    a = Model.window(m.attendance, "class_date", s, e)
    return None if a.empty else float(a.attended.mean() * 100)


def avg_score(m, s, e):
    a = Model.window(m.assessments, "assessment_date", s, e)
    return None if a.empty else float(a.pct.mean())


def pass_rate(m, s, e):
    a = Model.window(m.assessments, "assessment_date", s, e)
    return None if a.empty else float((a.pct >= PASS_MARK).mean() * 100)


def retention_rate(m, s, e):
    pw = m.prev_window(s, e)
    if pw is None:
        return None
    ev = m.events()
    prev_ids = set(Model.window(ev, "ts", *pw).student_id)
    if not prev_ids:
        return None
    cur_ids = set(Model.window(ev, "ts", s, e).student_id)
    return len(prev_ids & cur_ids) / len(prev_ids) * 100


def _open_enrollments_at(m, day) -> pd.DataFrame:
    en = m.enrollments
    day = pd.Timestamp(day)
    return en[(en.enrollment_date <= day) & ~(en.cancellation_date <= day) & ~(en.completion_date <= day)]


def churn_confirmed(m, s, e):
    """Confirmed churn = cancellations in window / enrollments open at window start + new in window."""
    base = len(_open_enrollments_at(m, pd.Timestamp(s) - pd.Timedelta(days=1))) + len(Model.window(m.enrollments, "enrollment_date", s, e))
    canc = len(Model.window(m.enrollments.dropna(subset=["cancellation_date"]), "cancellation_date", s, e))
    return None if base == 0 else canc / base * 100


def inactive_rate(m, s, e):
    """Inferred inactivity: open enrollments (≥30 days old) with no learner activity in the last 30 days of window."""
    e = pd.Timestamp(e)
    op = _open_enrollments_at(m, e)
    op = op[op.enrollment_date <= e - pd.Timedelta(days=INACTIVE_DAYS)]
    if op.empty:
        return None
    recent = set(Model.window(m.events(), "ts", e - pd.Timedelta(days=INACTIVE_DAYS), e).student_id)
    return float((~op.student_id.isin(recent)).mean() * 100)


def gross_bookings(m, s, e):
    return float(Model.window(m.enrollments, "enrollment_date", s, e).price.sum())


def collections(m, s, e):
    p = Model.window(m.payments, "payment_date", s, e)
    return float(p.loc[p.payment_status == "Success", "amount"].sum())


def refunds(m, s, e):
    p = Model.window(m.payments, "payment_date", s, e)
    return float(p.loc[p.payment_status == "Refunded", "refund_amount"].sum())


def net_revenue(m, s, e):
    return collections(m, s, e) - refunds(m, s, e)


def refund_rate(m, s, e):
    en = Model.window(m.enrollments, "enrollment_date", s, e)
    return None if en.empty else float((en.refund_amount > 0).mean() * 100)


def arps(m, s, e):
    p = Model.window(m.payments, "payment_date", s, e)
    payers = p.loc[p.payment_status == "Success", "student_id"].nunique()
    return None if payers == 0 else net_revenue(m, s, e) / payers


def aov(m, s, e):
    en = Model.window(m.enrollments, "enrollment_date", s, e)
    return None if en.empty else float(en.price.mean())


def payment_failure_rate(m, s, e):
    p = Model.window(m.payments, "payment_date", s, e)
    p = p[p.payment_status.isin(["Success", "Failed", "Pending"])]
    return None if p.empty else float((p.payment_status == "Failed").mean() * 100)


def csat(m, s, e):
    t = Model.window(m.tickets, "created_at", s, e).csat_score.dropna()
    return None if t.empty else float(t.mean())


def satisfaction(m, s, e):
    v = Model.window(m.surveys, "survey_date", s, e).satisfaction_score
    return None if v.empty else float(v.mean())


def nps(m, s, e):
    v = Model.window(m.surveys, "survey_date", s, e).recommendation_score
    if len(v) < 20:
        return None
    return float(((v >= 9).mean() - (v <= 6).mean()) * 100)


def tickets(m, s, e):
    return float(len(Model.window(m.tickets, "created_at", s, e)))


def sla_compliance(m, s, e):
    from .generate import SLA_HOURS
    t = Model.window(m.tickets, "created_at", s, e)
    if t.empty:
        return None
    frt = (t.first_response_at - t.created_at).dt.total_seconds() / 3600
    return float((frt <= t.priority.map(SLA_HOURS)).mean() * 100)


def marketing_spend(m, s, e):
    return float(Model.window(m.marketing, "date", s, e).spend.sum())


def cac(m, s, e):
    spend = marketing_spend(m, s, e)
    paid = set(m.marketing.channel)
    en = _first_enrol(m, s, e).merge(m.students[["student_id", "acquisition_channel"]], on="student_id")
    n = en[en.acquisition_channel.isin(paid)].student_id.nunique()
    return None if n == 0 or spend == 0 else spend / n


METRICS: dict[str, MetricDef] = {d.key: d for d in [
    MetricDef("leads", "Leads", "count", True, "New leads (sign-ups) created in the period.", "count(students.signup_date in period)", leads),
    MetricDef("new_enrollments", "New enrollments", "count", True, "Students whose first (non-renewal) enrollment date is in the period.", "count distinct student_id, enrollments.is_renewal = false", new_enrollments),
    MetricDef("total_enrolled", "Total enrolled students", "count", True, "Distinct students with any enrollment on or before period end.", "count distinct student_id where enrollment_date ≤ end", total_enrolled),
    MetricDef("lead_to_enrol", "Lead → enrollment", "pct", True, "Share of leads created in the period that have enrolled (as of data end). Recent leads may still convert.", "enrolled leads / leads", lead_to_enrol),
    MetricDef("active_learners", "Active learners", "count", True, "Students with ≥1 learning activity or attended live class in the period.", "count distinct student_id in activity ∪ attended classes", active_learners),
    MetricDef("dau", "Daily active learners (avg)", "count", True, "Average distinct active learners per day.", "mean over days of distinct active students", dau),
    MetricDef("wau", "Weekly active learners (avg)", "count", True, "Average distinct active learners per ISO week.", "mean over weeks of distinct active students", wau),
    MetricDef("mau", "Monthly active learners (avg)", "count", True, "Average distinct active learners per calendar month.", "mean over months of distinct active students", mau),
    MetricDef("activation_rate", "Activation rate", "pct", True, "Share of new enrollments with any learning activity or attended class within 14 days. Enrollments younger than 14 days at data end are excluded.", "activated within 14 days / mature new enrollments", activation_rate),
    MetricDef("completion_rate", "Course completion rate", "pct", True, "Share of first enrollments whose scheduled end date falls in the period that were completed (cancelled and inactive count as not completed).", "completed / enrollments with scheduled end in period", completion_rate),
    MetricDef("attendance_rate", "Live-class attendance", "pct", True, "Attended class seats / scheduled class seats for enrolled learners.", "sum(attended) / count(attendance rows)", attendance_rate),
    MetricDef("avg_score", "Avg assessment score", "pct", True, "Mean assessment score as % of max score.", "mean(score / max_score × 100)", avg_score),
    MetricDef("pass_rate", "Assessment pass rate", "pct", True, f"Share of assessments scoring ≥ {PASS_MARK:.0f}% of max.", f"count(pct ≥ {PASS_MARK:.0f}) / count(assessments)", pass_rate),
    MetricDef("retention_rate", "Learner retention", "pct", True, "Of learners active in the previous equal-length period, the share also active in this period.", "|active(prev) ∩ active(curr)| / |active(prev)|", retention_rate),
    MetricDef("churn_confirmed", "Confirmed churn (cancellations)", "pct", False, "Confirmed cancellations in the period / enrollments open at period start plus new enrollments in period.", "cancellations / (open at start + new)", churn_confirmed),
    MetricDef("inactive_rate", "Inferred inactivity (30d)", "pct", False, "Open enrollments (≥30 days old) with no activity in the last 30 days of the period. Inferred, not confirmed churn.", "no activity in last 30d / open enrollments", inactive_rate),
    MetricDef("gross_bookings", "Gross bookings", "inr", True, "Sum of enrollment prices (after discount) for enrollments in the period.", "sum(enrollments.price)", gross_bookings),
    MetricDef("collections", "Collections", "inr", True, "Successful payments received in the period.", "sum(payments.amount where status = Success)", collections),
    MetricDef("refunds", "Refunds", "inr", False, "Refunds paid out in the period.", "sum(payments.refund_amount where status = Refunded)", refunds),
    MetricDef("net_revenue", "Net revenue (cash)", "inr", True, "Collections minus refunds in the period (cash basis; not GAAP recognised revenue).", "collections − refunds", net_revenue),
    MetricDef("arps", "Avg revenue per paying student", "inr", True, "Net revenue / distinct students with a successful payment in the period.", "net_revenue / paying students", arps),
    MetricDef("aov", "Average order value", "inr", True, "Mean enrollment price after discount.", "mean(enrollments.price)", aov),
    MetricDef("refund_rate", "Refund rate", "pct", False, "Share of enrollments made in the period that have been refunded.", "enrollments with refund > 0 / enrollments", refund_rate),
    MetricDef("payment_failure_rate", "Payment failure rate", "pct", False, "Failed payment attempts / all payment attempts in the period.", "Failed / (Success + Failed + Pending)", payment_failure_rate),
    MetricDef("csat", "Support CSAT", "score", True, "Mean CSAT (1–5) of tickets created in the period that were rated.", "mean(csat_score)", csat),
    MetricDef("satisfaction", "Student satisfaction", "score", True, "Mean survey satisfaction (1–5).", "mean(satisfaction_score)", satisfaction),
    MetricDef("nps", "Net Promoter Score", "nps", True, "% promoters (9–10) − % detractors (0–6) among surveys in the period (min 20 responses).", "%promoters − %detractors", nps),
    MetricDef("tickets", "Support tickets", "count", False, "Tickets created in the period.", "count(support_tickets)", tickets),
    MetricDef("sla_compliance", "First-response SLA", "pct", True, "Tickets with first response within SLA (Urgent 2h, High 4h, Medium 12h, Low 24h).", "within SLA / tickets", sla_compliance),
    MetricDef("marketing_spend", "Marketing spend", "inr", False, "Recorded spend for paid channels (Referral and Organic have none).", "sum(marketing_spend.spend)", marketing_spend),
    MetricDef("cac", "Customer acquisition cost", "inr", False, "Spend / new enrollments from channels with recorded spend.", "spend / paid-channel new enrollments", cac),
]}


def compute(m: Model, key: str, start=None, end=None) -> float | None:
    start = pd.Timestamp(start if start is not None else m.start)
    end = pd.Timestamp(end if end is not None else m.end)
    try:
        v = METRICS[key].fn(m, start, end)
    except (KeyError, ValueError, ZeroDivisionError):
        return None
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def _rag(d: MetricDef, value, prev, change) -> str:
    if value is None or prev is None or change is None:
        return "grey"
    good = change if d.higher_is_better else -change
    tol = 1.0 if d.unit in ("pct", "nps") else 2.0  # pp for rates, % for levels
    return "green" if good >= -tol else "amber" if good >= -4 * tol else "red"


def kpi(m: Model, key: str) -> KPI:
    d = METRICS[key]
    value = compute(m, key)
    pw = m.prev_window()
    prev = compute(m, key, *pw) if pw else None
    change = None
    if value is not None and prev is not None:
        change = (value - prev) if d.unit in ("pct", "nps") else ((value - prev) / abs(prev) * 100 if prev else None)
    baseline = (f"previous {(m.end - m.start).days + 1} days ({pw[0]:%d %b %y} – {pw[1]:%d %b %y})" if pw
                else "no comparison: previous period is before available data")
    return KPI(key, d.name, value, prev, change, _rag(d, value, prev, change), d.unit, d.definition, d.formula,
               f"{m.start:%d %b %y} – {m.end:%d %b %y}", baseline)


def fmt(v: float | None, unit: str) -> str:
    if v is None:
        return "n/a"
    if unit == "inr":
        a = abs(v)
        return f"₹{v / 1e7:,.2f} Cr" if a >= 1e7 else f"₹{v / 1e5:,.2f} L" if a >= 1e5 else f"₹{v:,.0f}"
    if unit == "pct":
        return f"{v:.1f}%"
    if unit == "score":
        return f"{v:.2f}"
    if unit == "nps":
        return f"{v:+.0f}"
    if unit == "hours":
        return f"{v:.1f} h"
    return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:,.1f}"


def fmt_change(k: KPI) -> str:
    if k.change is None:
        return "no baseline"
    return f"{k.change:+.1f} pp" if k.unit in ("pct", "nps") else f"{k.change:+.1f}%"


def monthly(m: Model, key: str, months: int = 12) -> pd.DataFrame:
    """Metric computed per calendar month (same definition) for trend charts."""
    ends = pd.date_range(m.data_start, m.end, freq="ME")
    rows = []
    for me in ends[-months:]:
        ms = me.replace(day=1)
        rows.append({"month": ms, "value": compute(m, key, max(ms, m.data_start), min(me, m.end))})
    return pd.DataFrame(rows)


def dictionary() -> pd.DataFrame:
    return pd.DataFrame([{"metric": d.name, "key": d.key, "unit": d.unit,
                          "direction": "higher is better" if d.higher_is_better else "lower is better",
                          "definition": d.definition, "calculation": d.formula} for d in METRICS.values()])
