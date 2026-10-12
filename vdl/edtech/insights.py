"""Ranked, evidence-backed business insights for the executive layer.

Every insight is built from computed metrics (metrics.py / analytics.py). Impact score =
business weight of the metric × size of the change (capped) × reach (log of learners/leads affected).
It ranks by potential business impact, not statistical significance; both are shown."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import analytics as A
from . import metrics as M
from .model import Model

WEIGHTS = {"new_enrollments": 3, "net_revenue": 3, "lead_to_enrol": 2.5, "active_learners": 2, "completion_rate": 2,
           "retention_rate": 2, "inactive_rate": 2, "churn_confirmed": 2, "cac": 2, "attendance_rate": 1.5,
           "avg_score": 1.5, "refund_rate": 1.5, "nps": 1.5, "sla_compliance": 1, "payment_failure_rate": 1}


@dataclass
class Insight:
    title: str
    metric: str
    observed: str
    baseline: str
    change: str
    segment: str
    implication: str
    action: str
    polarity: str  # positive | risk
    impact: float
    basis: str  # Verified metric | Association (not causal) | Rules-based flag
    evidence: pd.DataFrame = field(default_factory=pd.DataFrame)


def _magnitude(k: M.KPI) -> float:
    return min(abs(k.change) / (2 if k.unit in ("pct", "nps") else 5), 3)


def _kpi_insights(m: Model) -> list[Insight]:
    out = []
    reach = max(np.log10(max(len(m.enrolled), 10)), 1)
    for key, w in WEIGHTS.items():
        k = M.kpi(m, key)
        if k.change is None or k.value is None or abs(k.change) < (1 if k.unit in ("pct", "nps") else 3):
            continue
        d = M.METRICS[key]
        good = (k.change > 0) == d.higher_is_better
        seg, ev = "All learners in current filters", pd.DataFrame()
        if key in ("new_enrollments", "lead_to_enrol"):
            ev = _enrol_driver(m)
            if not ev.empty:
                top = ev.iloc[0] if not good else ev.iloc[-1]
                seg = f"Largest contributor: {top.iloc[0]} ({top.change:+.0f} enrollments)"
        elif key == "net_revenue":
            cur = A.revenue_by(m, "course_name").set_index("course_name").net_revenue
            pw = m.prev_window()
            prev = A.revenue_by(_shift(m, *pw), "course_name").set_index("course_name").net_revenue
            ev = A.driver_table(cur, prev, "course")
            top = ev.iloc[0] if not good else ev.iloc[-1]
            seg = f"Largest contributor: {top.course} ({M.fmt(top.change, 'inr')})"
        direction = "improved" if good else "worsened"
        out.append(Insight(
            title=f"{d.name} {direction}: {M.fmt(k.value, k.unit)} ({M.fmt_change(k)})",
            metric=d.name, observed=M.fmt(k.value, k.unit), baseline=f"{M.fmt(k.prev, k.unit)} in {k.baseline}",
            change=M.fmt_change(k), segment=seg,
            implication=_implication(key, good), action=_action(key, good),
            polarity="positive" if good else "risk", impact=round(w * _magnitude(k) * reach, 2),
            basis="Verified metric", evidence=ev))
    return out


def _shift(m: Model, start, end) -> Model:
    from .model import Filters
    f = m.f
    return Model(m.tables, Filters(start, end, f.courses, f.subjects, f.cohorts, f.channels, f.statuses))


def _enrol_driver(m: Model) -> pd.DataFrame:
    pw = m.prev_window()
    if pw is None:
        return pd.DataFrame()
    def by_channel(mm):
        en = Model.window(mm.enrollments[~mm.enrollments.is_renewal.astype(bool)], "enrollment_date", mm.start, mm.end)
        return A.student_dim(mm, en, "acquisition_channel").groupby("acquisition_channel").student_id.nunique()
    return A.driver_table(by_channel(m), by_channel(_shift(m, *pw)), "channel")


_IMPL = {
    "new_enrollments": ("More learners entering the funnel supports revenue growth.", "Fewer new learners will reduce bookings and future renewals."),
    "net_revenue": ("Cash generation is strengthening.", "Cash generation is weakening; check collections, refunds and bookings."),
    "lead_to_enrol": ("Sales efficiency is improving.", "Each lead converts less often, raising effective acquisition cost."),
    "active_learners": ("More learners are engaging.", "Fewer active learners precede lower completion and renewal."),
    "completion_rate": ("More learners finish their course, supporting outcomes and renewals.", "Lower completion threatens outcomes, reviews and renewals."),
    "retention_rate": ("Learners are staying engaged period to period.", "More learners are dropping out of active learning."),
    "inactive_rate": ("Fewer enrolled learners have gone quiet.", "A growing share of paying learners are inactive for 30+ days (inferred, not cancelled)."),
    "churn_confirmed": ("Fewer cancellations.", "More confirmed cancellations and refunds."),
    "cac": ("Acquisition is getting cheaper per enrollment.", "Each paid enrollment costs more."),
    "attendance_rate": ("Live-class participation is rising.", "Live-class participation is falling."),
    "avg_score": ("Assessment performance is improving.", "Assessment performance is slipping."),
    "refund_rate": ("Fewer refunds protect revenue.", "More refunds erode revenue and signal mismatch at sale."),
    "nps": ("Learners are more likely to recommend.", "Learners are less likely to recommend; word of mouth weakens."),
    "sla_compliance": ("Support responds faster.", "Support is missing response SLAs more often."),
    "payment_failure_rate": ("Fewer failed payments.", "More failed payments put collections at risk."),
}
_ACT = {
    "new_enrollments": ("Scale the channels driving growth while monitoring lead quality.", "Review channel mix and counselling capacity; see the enrollment driver table."),
    "net_revenue": ("Protect the drivers: keep collection follow-ups and low refund practices.", "Prioritise pending-payment follow-up and review refund reasons by course."),
    "lead_to_enrol": ("Document what top channels and counsellors do and replicate.", "Audit lead quality by channel and counsellor follow-up speed."),
    "active_learners": ("Keep engagement programmes running.", "Launch re-engagement nudges for dormant and at-risk segments."),
    "completion_rate": ("Share completion practices across courses.", "Target courses with lowest completion; add mid-course check-ins."),
    "retention_rate": ("Maintain cadence of live classes and practice.", "Contact at-risk learners (Student Success page) within 7 days."),
    "inactive_rate": ("Continue proactive outreach.", "Call inactive learners; distinguish pauses from silent churn."),
    "churn_confirmed": ("Keep onboarding quality high.", "Interview cancelling learners; review sales promises vs delivery."),
    "cac": ("Reinvest savings into best-retaining channels.", "Shift budget from high-CAC, low-retention channels; test before scaling."),
    "attendance_rate": ("Keep class timings and reminders.", "Test reminder timing and alternate batch times."),
    "avg_score": ("Spread effective practice sets.", "Add targeted practice on hardest topics (Academic page)."),
    "refund_rate": ("Keep counselling expectations accurate.", "Review refund reasons and early-week onboarding."),
    "nps": ("Ask promoters for referrals.", "Follow up with detractors; fix top complaint themes."),
    "sla_compliance": ("Keep staffing model.", "Add support capacity at peak hours; triage Urgent/High first."),
    "payment_failure_rate": ("Keep retry logic.", "Add smart retries and alternate payment methods."),
}


def _implication(key, good):
    return _IMPL.get(key, ("Positive movement.", "Negative movement."))[0 if good else 1]


def _action(key, good):
    return _ACT.get(key, ("Sustain.", "Investigate."))[0 if good else 1]


def _special_insights(m: Model) -> list[Insight]:
    out = []
    fce = A.first_class_effect(m)
    if not fce.empty and {"First class ≤ 7 days", "First class > 7 days"} <= set(fce.group):
        a = fce.set_index("group")
        early, late = a.loc["First class ≤ 7 days"], a.loc["First class > 7 days"]
        diff = early.avg_active_days_day15_60 - late.avg_active_days_day15_60
        if diff >= 2 and late.students >= 30:
            out.append(Insight(
                "Learners whose first class is >7 days after enrollment engage less afterwards", "Active days, days 15–60 after enrollment",
                f"{late.avg_active_days_day15_60:.1f} days (n={int(late.students)})",
                f"{early.avg_active_days_day15_60:.1f} days for first class ≤7 days (n={int(early.students)})",
                f"{-diff:.1f} days ({-diff / early.avg_active_days_day15_60 * 100:.0f}%)", "Learners with a late first class",
                "Early activation is associated with sustained engagement (association in this synthetic data, not proven causal).",
                "Book the first live class during counselling; nudge anyone not attended by day 5.",
                "risk", round(2.5 * min(diff / 2, 3) * np.log10(max(late.students, 10)), 2), "Association (not causal)", fce))
    fb_cur = A.funnel_by(m, "acquisition_channel")
    pw = m.prev_window()
    if pw and not fb_cur.empty:
        fb_prev = A.funnel_by(_shift(m, *pw), "acquisition_channel")
        j = fb_cur.set_index("acquisition_channel")[["leads", "lead_to_enrol_pct"]].join(
            fb_prev.set_index("acquisition_channel")[["lead_to_enrol_pct"]], rsuffix="_prev")
        j["change_pp"] = j.lead_to_enrol_pct - j.lead_to_enrol_pct_prev
        j = j[j.leads >= 50]
        if not j.empty:
            w = j.change_pp.idxmin()
            if j.loc[w, "change_pp"] <= -5:
                out.append(Insight(
                    f"{w} lead-to-enrollment conversion fell {abs(j.loc[w, 'change_pp']):.1f} pp", "Lead → enrollment (by channel)",
                    f"{j.loc[w, 'lead_to_enrol_pct']:.1f}% of {int(j.loc[w, 'leads'])} leads", f"{j.loc[w, 'lead_to_enrol_pct_prev']:.1f}% in previous period",
                    f"{j.loc[w, 'change_pp']:+.1f} pp", f"Channel: {w}",
                    "Spend on this channel now buys fewer enrollments; recent leads may still convert (check maturity).",
                    f"Pause scaling {w}; audit targeting/creative and lead qualification; compare counsellor follow-up times.",
                    "risk", round(3 * min(abs(j.loc[w, "change_pp"]) / 2, 3) * np.log10(max(j.loc[w, "leads"], 10)), 2),
                    "Verified metric", j.reset_index().round(1)))
    risk = A.student_risk(m)
    if not risk.empty:
        hi = risk[risk.risk_level == "High"]
        if len(hi):
            out.append(Insight(
                f"{len(hi)} learners flagged High risk of disengagement", "Rules-based risk flags",
                f"{len(hi)} High, {int((risk.risk_level == 'Medium').sum())} Medium of {len(risk)} open learners",
                "Rules backtested on Student Success page", "n/a", "See Student Success list",
                "Without intervention these learners are likely to stop learning and not renew.",
                "Assign High-risk learners to mentors this week; track follow-up status.",
                "risk", round(2 * min(len(hi) / len(risk) * 20, 3) * np.log10(max(len(hi), 10)), 2), "Rules-based flag",
                hi.head(15)))
    ta = A.topic_accuracy(m)
    if not ta.empty and ta.attempts.iloc[0] >= 50:
        t = ta.iloc[0]
        out.append(Insight(
            f"Lowest-accuracy topic: {t.topic} ({t.subject}) at {t.accuracy_pct:.1f}%", "Topic accuracy",
            f"{t.accuracy_pct:.1f}% over {int(t.attempts)} assessments", f"median topic accuracy {ta.accuracy_pct.median():.1f}%",
            f"{t.accuracy_pct - ta.accuracy_pct.median():+.1f} pp", f"{t.subject} learners",
            "Weak mastery here can depress overall scores and completion.",
            f"Add worked examples and adaptive practice for {t.topic}; review question difficulty calibration.",
            "risk", round(1.5 * min(abs(t.accuracy_pct - ta.accuracy_pct.median()) / 2, 3) * np.log10(max(t.attempts, 10)), 2),
            "Verified metric", ta.head(8)))
    se = A.support_engagement(m)
    if not se.empty and len(se) == 2:
        s = se.set_index("group")
        if {"Resolved > 72h", "Resolved ≤ 72h"} <= set(s.index) and s.loc["Resolved > 72h", "learners"] >= 20:
            gap = s.loc["Resolved ≤ 72h", "change"] - s.loc["Resolved > 72h", "change"]
            if gap >= 0.5:
                out.append(Insight(
                    "Slow ticket resolution (>72h) is followed by weaker engagement", "Active days 30d after vs before ticket",
                    f"{s.loc['Resolved > 72h', 'change']:+.1f} days (n={int(s.loc['Resolved > 72h', 'learners'])})",
                    f"{s.loc['Resolved ≤ 72h', 'change']:+.1f} days when resolved ≤72h", f"{-gap:+.1f} days", "Learners with slow tickets",
                    "Support speed may matter for engagement (association only).", "Set a 72h resolution target for learning-blocking tickets.",
                    "risk", round(1 * min(gap, 3) * np.log10(max(s.loc["Resolved > 72h", "learners"], 10)), 2),
                    "Association (not causal)", se))
    return out


def generate(m: Model) -> list[Insight]:
    if m.empty():
        return []
    return sorted(_kpi_insights(m) + _special_insights(m), key=lambda i: -i.impact)


def top(insights: list[Insight], polarity: str, n: int = 5) -> list[Insight]:
    return [i for i in insights if i.polarity == polarity][:n]
