"""Analytics modules (lifecycle, engagement, academics, retention, revenue, courses, marketing, support, risk).

All functions are pure: Model in, DataFrames out. The date window is m.start..m.end unless stated.
Associations are reported with sample sizes and comparison groups; nothing here claims causation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .generate import SLA_HOURS
from .metrics import PASS_MARK
from .model import Model

W = Model.window
DAY = pd.Timedelta(days=1)


# ============================== helpers ==============================
def driver_table(cur: pd.Series, prev: pd.Series, label: str) -> pd.DataFrame:
    """Decompose a change in a total into segment contributions (current vs previous window)."""
    d = pd.DataFrame({"current": cur, "previous": prev}).fillna(0)
    d["change"] = d.current - d.previous
    total = d.change.sum()
    d["share_of_change"] = (d.change / total * 100).round(1) if total else 0.0
    d["change_pct"] = np.where(d.previous > 0, (d.change / d.previous.replace(0, np.nan) * 100).round(1), np.nan)
    return d.sort_values("change").reset_index(names=label)


def student_dim(m: Model, df: pd.DataFrame, dim: str) -> pd.DataFrame:
    cols = ["student_id", dim] if dim in m.students.columns else ["student_id"]
    return df.merge(m.students[cols], on="student_id", how="left")


# ============================== B. lifecycle & funnel ==============================
FUNNEL = ["Lead", "Qualified lead", "Counselling", "Enrollment", "First class", "Active learning (≥4 wks)",
          "Assessment taken", "Course completion", "Renewal"]


def _stage_flags(m: Model, leads: pd.DataFrame) -> pd.DataFrame:
    ids = leads.student_id
    att = m.attendance[m.attendance.attended & m.attendance.student_id.isin(ids)]
    first_cls = att.groupby("student_id").class_date.min()
    ev = m.events()
    ev = ev[ev.student_id.isin(ids)]
    active_wks = ev.groupby("student_id").ts.apply(lambda x: x.dt.to_period("W").nunique())
    en = m.enrollments[m.enrollments.student_id.isin(ids)]
    f = pd.DataFrame({"student_id": ids.values}).set_index("student_id")
    f["Lead"] = True
    f["Qualified lead"] = leads.set_index("student_id").lead_qualified.astype(bool)
    f["Counselling"] = leads.set_index("student_id").counseling_date.notna()
    f["Enrollment"] = leads.set_index("student_id").enrollment_date.notna()
    f["First class"] = f.index.isin(first_cls.index) & f.Enrollment
    f["Active learning (≥4 wks)"] = f.index.isin(active_wks[active_wks >= 4].index) & f.Enrollment
    f["Assessment taken"] = f.index.isin(m.assessments.student_id) & f.Enrollment
    f["Course completion"] = f.index.isin(en.loc[en.enrollment_status == "Completed", "student_id"])
    f["Renewal"] = f.index.isin(en.loc[en.is_renewal.astype(bool), "student_id"])
    for prev, cur in zip(FUNNEL, FUNNEL[1:]):  # strict funnel: each stage requires the previous one
        f[cur] = f[cur] & f[prev]
    return f


def funnel(m: Model) -> pd.DataFrame:
    leads = W(m.students, "signup_date", m.start, m.end)
    if leads.empty:
        return pd.DataFrame(columns=["stage", "students", "pct_of_leads", "step_conversion", "drop_off"])
    f = _stage_flags(m, leads)
    counts = [int(f[s].sum()) for s in FUNNEL]
    out = pd.DataFrame({"stage": FUNNEL, "students": counts})
    out["pct_of_leads"] = (out.students / counts[0] * 100).round(1)
    prev = out.students.shift(1)
    out["step_conversion"] = (out.students / prev * 100).round(1)
    out.loc[0, "step_conversion"] = 100.0
    out["drop_off"] = (prev - out.students).fillna(0).astype(int)
    return out


def funnel_by(m: Model, dim: str) -> pd.DataFrame:
    leads = W(m.students, "signup_date", m.start, m.end)
    if leads.empty:
        return pd.DataFrame()
    f = _stage_flags(m, leads).join(leads.set_index("student_id")[dim])
    g = f.groupby(dim)
    out = pd.DataFrame({"leads": g.Lead.sum(), "qualified_pct": g["Qualified lead"].mean() * 100,
                        "enrolled": g.Enrollment.sum(), "lead_to_enrol_pct": g.Enrollment.mean() * 100})
    out["first_class_pct_of_enrolled"] = g["First class"].sum() / out.enrolled.replace(0, np.nan) * 100
    return out.round(1).sort_values("lead_to_enrol_pct", ascending=False).reset_index()


def stage_times(m: Model) -> pd.DataFrame:
    s = W(m.students, "signup_date", m.start, m.end)
    if s.empty:
        return pd.DataFrame(columns=["transition", "students", "median_days", "p75_days"])
    att = m.attendance[m.attendance.attended].groupby("student_id").class_date.min()
    s = s.assign(first_class=s.student_id.map(att))
    rows = [("Lead → counselling", (s.counseling_date - s.signup_date).dt.total_seconds() / 86400),
            ("Counselling → enrollment", (s.enrollment_date - s.counseling_date.dt.normalize()).dt.days),
            ("Enrollment → first class", (s.first_class - s.enrollment_date).dt.total_seconds() / 86400)]
    return pd.DataFrame([{"transition": n, "students": int(v.notna().sum()), "median_days": round(v.median(), 1),
                          "p75_days": round(v.quantile(0.75), 1)} for n, v in rows])


def cohort_funnel(m: Model) -> pd.DataFrame:
    s = m.students.copy()
    if s.empty:
        return pd.DataFrame()
    f = _stage_flags(m, s)
    f["month"] = s.set_index("student_id").signup_date.dt.strftime("%Y-%m")
    g = f.groupby("month")
    return pd.DataFrame({"leads": g.Lead.sum(), "enrolled_pct": g.Enrollment.mean() * 100,
                         "first_class_pct": g["First class"].mean() * 100,
                         "active_4wk_pct": g["Active learning (≥4 wks)"].mean() * 100,
                         "completed_pct": g["Course completion"].mean() * 100}).round(1).reset_index()


def cancellations(m: Model) -> pd.DataFrame:
    en = W(m.enrollments, "enrollment_date", m.start, m.end).merge(m.courses[["course_id", "course_name"]], on="course_id")
    if en.empty:
        return pd.DataFrame()
    en["cancelled"] = en.enrollment_status == "Cancelled"
    en["days_to_cancel"] = (en.cancellation_date - en.enrollment_date).dt.days
    g = en.groupby("course_name")
    return pd.DataFrame({"enrollments": g.size(), "cancelled": g.cancelled.sum(),
                         "cancel_rate_pct": (g.cancelled.mean() * 100).round(1),
                         "refunded_inr": g.refund_amount.sum().round(0),
                         "median_days_to_cancel": g.days_to_cancel.median()}).sort_values(
        "cancel_rate_pct", ascending=False).reset_index()


def first_class_effect(m: Model) -> pd.DataFrame:
    """Compare later engagement for learners whose first class was ≤7 days vs >7 days after enrollment."""
    en = m.enrolled[m.enrolled.enrollment_date <= m.data_end - pd.Timedelta(days=60)]
    if en.empty:
        return pd.DataFrame()
    att = m.attendance[m.attendance.attended].groupby("student_id").class_date.min()
    lag = (en.student_id.map(att) - en.enrollment_date).dt.days
    grp = np.where(lag.isna(), "No class attended", np.where(lag <= 7, "First class ≤ 7 days", "First class > 7 days"))
    ev = m.events().merge(en[["student_id", "enrollment_date"]], on="student_id")
    later = ev[(ev.ts >= ev.enrollment_date + pd.Timedelta(days=15)) & (ev.ts < ev.enrollment_date + pd.Timedelta(days=60))]
    days = later.groupby("student_id").ts.apply(lambda x: x.dt.normalize().nunique())
    d = pd.DataFrame({"group": grp, "active_days_15_60": en.student_id.map(days).fillna(0).values,
                      "completed": en.student_id.isin(m.enrollments.loc[m.enrollments.enrollment_status == "Completed", "student_id"]).values})
    g = d.groupby("group")
    return pd.DataFrame({"students": g.size(), "avg_active_days_day15_60": g.active_days_15_60.mean().round(1),
                         "pct_active_day15_60": (g.active_days_15_60.apply(lambda x: (x > 0).mean()) * 100).round(1)}).reset_index()


# ============================== C. engagement ==============================
def active_trend(m: Model, freq: str = "W") -> pd.DataFrame:
    ev = W(m.events(), "ts", m.data_start, m.end)
    if ev.empty:
        return pd.DataFrame(columns=["period", "active_learners"])
    g = ev.groupby(ev.ts.dt.to_period(freq)).student_id.nunique()
    out = g.reset_index(name="active_learners")
    out["period"] = out.ts.dt.start_time
    return out[["period", "active_learners"]]


def dau_wau_mau(m: Model) -> pd.DataFrame:
    ev = W(m.events(), "ts", m.data_start, m.end)
    if ev.empty:
        return pd.DataFrame()
    d = ev.groupby(ev.ts.dt.normalize()).student_id.nunique().rename("DAU")
    days = pd.date_range(d.index.min(), d.index.max())
    d = d.reindex(days, fill_value=0)
    ev_sets = ev.assign(day=ev.ts.dt.normalize()).groupby("day").student_id.apply(set).reindex(days, fill_value=set())
    wau, mau = [], []
    for i in range(len(days)):
        wau.append(len(set().union(*ev_sets.iloc[max(0, i - 6):i + 1])))
        mau.append(len(set().union(*ev_sets.iloc[max(0, i - 29):i + 1])))
    out = pd.DataFrame({"date": days, "DAU": d.values, "WAU (rolling 7d)": wau, "MAU (rolling 30d)": mau})
    out["stickiness_DAU_MAU_pct"] = (out.DAU / out["MAU (rolling 30d)"].replace(0, np.nan) * 100).round(1)
    return out[out.date >= days.min() + pd.Timedelta(days=29)]


def activity_heatmap(m: Model) -> pd.DataFrame:
    a = W(m.activity, "activity_date", m.start, m.end)
    if a.empty:
        return pd.DataFrame()
    t = a.groupby([a.activity_date.dt.day_name(), a.activity_date.dt.hour]).size().unstack(fill_value=0)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return t.reindex([d for d in order if d in t.index])


def calendar(m: Model) -> pd.DataFrame:
    ev = W(m.events(), "ts", m.start, m.end)
    out = ev.groupby(ev.ts.dt.normalize()).student_id.nunique().rename("active_learners").reset_index()
    return out.rename(columns={out.columns[0]: "date"})


def learner_engagement(m: Model, as_of=None) -> pd.DataFrame:
    """One row per enrolled learner with engagement indicators as of `as_of` (default: window end)."""
    as_of = pd.Timestamp(as_of or m.end) + pd.Timedelta(hours=23, minutes=59)
    en = m.enrolled[m.enrolled.enrollment_date <= as_of]
    if en.empty:
        return pd.DataFrame()
    ev = m.events()
    ev = ev[ev.ts <= as_of]
    day = ev.assign(day=ev.ts.dt.normalize())
    last = day.groupby("student_id").ts.max()
    d28 = day[day.ts > as_of - pd.Timedelta(days=28)].groupby("student_id").day.nunique()
    d14 = day[day.ts > as_of - pd.Timedelta(days=14)].groupby("student_id").day.nunique()
    p14 = day[(day.ts <= as_of - pd.Timedelta(days=14)) & (day.ts > as_of - pd.Timedelta(days=28))].groupby("student_id").day.nunique()
    act = m.activity[(m.activity.activity_date <= as_of) & (m.activity.activity_date > as_of - pd.Timedelta(days=28))]
    watch = act[act.activity_type == "video_watch"].groupby("student_id").duration_minutes.sum()
    practice = act[act.activity_type == "practice_quiz"].groupby("student_id").size()
    assign = act[act.activity_type == "assignment_submit"].groupby("student_id").size()
    att = m.attendance[(m.attendance.class_date <= as_of) & (m.attendance.class_date > as_of - pd.Timedelta(days=28))]
    att_rate = att.groupby("student_id").attended.mean() * 100
    # current streak of consecutive active days ending at as_of (or the day before)
    streak = {}
    recent = day[day.ts > as_of - pd.Timedelta(days=60)].groupby("student_id").day.apply(lambda x: set(x))
    for sid, days in recent.items():
        k, cur = 0, as_of.normalize()
        if cur not in days:
            cur -= DAY
        while cur in days:
            k += 1
            cur -= DAY
        streak[sid] = k
    first_en = m.enrollments[~m.enrollments.is_renewal.astype(bool)].groupby("student_id").enrollment_date.min()
    out = pd.DataFrame({"student_id": en.student_id.values}).set_index("student_id")
    out["course"] = en.set_index("student_id").course_name
    out["cohort"] = en.set_index("student_id").cohort
    out["channel"] = en.set_index("student_id").acquisition_channel
    out["status"] = en.set_index("student_id").student_status
    out["enrolled_on"] = out.index.map(first_en)
    out["last_activity"] = out.index.map(last)
    out["days_since_activity"] = (as_of.normalize() - out.last_activity.dt.normalize()).dt.days
    out["active_days_28d"] = out.index.map(d28).fillna(0).astype(int)
    out["active_days_14d"] = out.index.map(d14).fillna(0).astype(int)
    out["active_days_prev14d"] = out.index.map(p14).fillna(0).astype(int)
    out["watch_minutes_28d"] = out.index.map(watch).fillna(0).round(0)
    out["practice_attempts_28d"] = out.index.map(practice).fillna(0).astype(int)
    out["assignments_28d"] = out.index.map(assign).fillna(0).astype(int)
    out["attendance_28d_pct"] = out.index.map(att_rate).round(1)
    out["streak_days"] = out.index.map(streak).fillna(0).astype(int)
    return out.reset_index()


SEGMENT_RULES = [
    ("Closed (completed / cancelled)", "Student status is Completed or Cancelled – excluded from engagement segments."),
    ("Newly activated", "Enrolled ≤ 21 days ago and active in the last 14 days."),
    ("Highly engaged", "≥ 12 active days in the last 28 days and live-class attendance ≥ 70% (or no classes scheduled)."),
    ("At risk of disengagement", "Active in the last 28 days but active days in the last 14 < half of the previous 14 (and previous 14 ≥ 2), or last activity 15–28 days ago."),
    ("Consistently active", "≥ 4 active days in the last 28 days (and none of the rules above)."),
    ("Low activity", "1–3 active days in the last 28 days."),
    ("Dormant", "No activity in the last 28 days (still enrolled)."),
]


def segment(row) -> str:
    if row.status in ("Completed", "Cancelled"):
        return "Closed (completed / cancelled)"
    age = (row.ref - row.enrolled_on).days if pd.notna(row.enrolled_on) else 999
    if age <= 21 and row.active_days_14d > 0:
        return "Newly activated"
    if row.active_days_28d >= 12 and (pd.isna(row.attendance_28d_pct) or row.attendance_28d_pct >= 70):
        return "Highly engaged"
    if row.active_days_28d > 0 and ((row.active_days_prev14d >= 2 and row.active_days_14d < row.active_days_prev14d / 2)
                                    or 15 <= (row.days_since_activity if pd.notna(row.days_since_activity) else 99) <= 28):
        return "At risk of disengagement"
    if row.active_days_28d >= 4:
        return "Consistently active"
    if row.active_days_28d >= 1:
        return "Low activity"
    return "Dormant"


def segments(m: Model, as_of=None) -> pd.DataFrame:
    le = learner_engagement(m, as_of)
    if le.empty:
        return le
    le["ref"] = pd.Timestamp(as_of or m.end)
    le["segment"] = le.apply(segment, axis=1)
    return le.drop(columns="ref")


def watch_vs_score(m: Model) -> pd.DataFrame:
    a = W(m.activity, "activity_date", m.start, m.end)
    w = a[a.activity_type == "video_watch"].groupby("student_id").duration_minutes.sum().rename("watch_minutes")
    sc = W(m.assessments, "assessment_date", m.start, m.end).groupby("student_id").pct.mean().rename("avg_score_pct")
    att = W(m.attendance, "class_date", m.start, m.end).groupby("student_id").attended.mean().mul(100).rename("attendance_pct")
    d = pd.concat([w, sc, att], axis=1).dropna(subset=["avg_score_pct"]).fillna({"watch_minutes": 0})
    return student_dim(m, d.reset_index(names="student_id"), "course_name")


def engagement_by(m: Model, dim: str) -> pd.DataFrame:
    ev = W(m.events(), "ts", m.start, m.end)
    base = m.enrolled[m.enrolled.enrollment_date <= m.end]
    if base.empty:
        return pd.DataFrame(columns=[dim, "enrolled", "active_pct", "avg_active_days"])
    days = ev.groupby("student_id").ts.apply(lambda x: x.dt.normalize().nunique())
    b = base.assign(active_days=base.student_id.map(days).fillna(0))
    g = b.groupby(dim)
    return pd.DataFrame({"enrolled": g.size(), "active_pct": (g.active_days.apply(lambda x: (x > 0).mean()) * 100).round(1),
                         "avg_active_days": g.active_days.mean().round(1)}).sort_values("avg_active_days", ascending=False).reset_index()


def declining_learners(m: Model) -> pd.DataFrame:
    le = learner_engagement(m)
    if le.empty:
        return le
    d = le[(le.active_days_prev14d >= 4) & (le.active_days_14d <= le.active_days_prev14d / 2) & ~le.status.isin(["Completed", "Cancelled"])]
    return d.sort_values("active_days_prev14d", ascending=False)[["student_id", "course", "active_days_prev14d", "active_days_14d",
                                                                    "days_since_activity", "attendance_28d_pct"]]


# ============================== D. academics ==============================
def assessments_w(m: Model) -> pd.DataFrame:
    a = W(m.assessments, "assessment_date", m.start, m.end).merge(m.courses[["course_id", "course_name", "subject"]], on="course_id")
    en = m.enrollments.drop_duplicates("student_id")[["student_id", "batch_id"]]
    a = a.merge(en, on="student_id", how="left")
    inst = m.classes.drop_duplicates("batch_id").set_index("batch_id").instructor_id
    a["instructor_id"] = a.batch_id.map(inst)
    a["accuracy_pct"] = (a.correct_answers / a.questions_attempted * 100).round(1)
    return a


def topic_accuracy(m: Model) -> pd.DataFrame:
    a = assessments_w(m)
    if a.empty:
        return pd.DataFrame()
    g = a.groupby(["subject", "topic"])
    out = pd.DataFrame({"attempts": g.size(), "accuracy_pct": (g.correct_answers.sum() / g.questions_attempted.sum() * 100).round(1),
                        "avg_score_pct": g.pct.mean().round(1), "pass_rate_pct": (g.pct.apply(lambda x: (x >= PASS_MARK).mean()) * 100).round(1)})
    out["difficulty"] = pd.cut(out.accuracy_pct, [0, 45, 55, 101], labels=["Hard", "Medium", "Easy"], right=False)
    return out.sort_values("accuracy_pct").reset_index()


def course_topic_matrix(m: Model) -> pd.DataFrame:
    a = assessments_w(m)
    if a.empty:
        return pd.DataFrame()
    a["topic_no"] = a.groupby("course_id").topic.transform(lambda x: x)
    return a.pivot_table(index="course_name", columns="topic", values="accuracy_pct", aggfunc="mean").round(1)


def improvement(m: Model) -> pd.DataFrame:
    """Paired first-vs-latest assessment per learner (learners with ≥3 assessments, all-time up to window end)."""
    a = m.assessments[m.assessments.assessment_date <= m.end + pd.Timedelta(hours=23)].sort_values("assessment_date")
    g = a.groupby("student_id")
    n = g.size()
    keep = n[n >= 3].index
    if not len(keep):
        return pd.DataFrame()
    first = g.pct.apply(lambda x: x.iloc[:2].mean())[keep]
    last = g.pct.apply(lambda x: x.iloc[-2:].mean())[keep]
    d = pd.DataFrame({"baseline_pct": first.round(1), "latest_pct": last.round(1)})
    d["change_pp"] = (d.latest_pct - d.baseline_pct).round(1)
    return student_dim(m, d.reset_index(names="student_id"), "course_name")


def by_group(m: Model, dim: str) -> pd.DataFrame:
    a = assessments_w(m)
    if a.empty or dim not in a.columns:
        return pd.DataFrame()
    g = a.groupby(dim)
    return pd.DataFrame({"assessments": g.size(), "learners": g.student_id.nunique(), "avg_score_pct": g.pct.mean().round(1),
                         "median_score_pct": g.pct.median().round(1),
                         "pass_rate_pct": (g.pct.apply(lambda x: (x >= PASS_MARK).mean()) * 100).round(1)}).reset_index()


def needs_intervention(m: Model) -> pd.DataFrame:
    imp = improvement(m)
    a = m.assessments[m.assessments.assessment_date <= m.end + pd.Timedelta(hours=23)].sort_values("assessment_date")
    last3 = a.groupby("student_id").pct.apply(lambda x: x.iloc[-3:].mean()).rename("last3_avg_pct").round(1)
    d = imp.set_index("student_id").join(last3, how="outer") if not imp.empty else last3.to_frame()
    open_ids = set(m.enrolled.loc[~m.enrolled.student_status.isin(["Completed", "Cancelled"]), "student_id"])
    d = d[d.index.isin(open_ids)]
    reason = np.where(d.last3_avg_pct < PASS_MARK, "Below pass mark (last 3 avg)",
                      np.where(d.get("change_pp", pd.Series(0, index=d.index)).fillna(0) <= -10, "Score declined ≥10 pp", ""))
    d = d.assign(reason=reason)
    d = d[d.reason != ""]
    return d.reset_index(names="student_id").merge(m.students[["student_id", "course_name"]], on="student_id", how="left")


# ============================== E. retention ==============================
def cohort_matrix(m: Model, max_months: int = 12) -> pd.DataFrame:
    """% of each enrollment-month cohort active in month k after enrollment (k=0 is enrollment month).
    Cells whose month is after the data end are left blank (not zero)."""
    en = m.enrolled[["student_id", "enrollment_date"]]
    if en.empty:
        return pd.DataFrame()
    ev = m.events().merge(en, on="student_id")
    ev = ev[ev.ts >= ev.enrollment_date]
    ev["k"] = (ev.ts.dt.year - ev.enrollment_date.dt.year) * 12 + ev.ts.dt.month - ev.enrollment_date.dt.month
    ev["cohort"] = ev.enrollment_date.dt.strftime("%Y-%m")
    size = en.groupby(en.enrollment_date.dt.strftime("%Y-%m")).size()
    act = ev.groupby(["cohort", "k"]).student_id.nunique().unstack(fill_value=0)
    mat = act.div(size, axis=0).mul(100).round(1)
    end_period = m.end.to_period("M")
    for c in mat.index:
        for k in mat.columns:
            if pd.Period(c, "M") + int(k) > end_period:
                mat.loc[c, k] = np.nan
    mat = mat.reindex(columns=range(0, max_months))
    mat.insert(0, "cohort_size", size)
    return mat


def retention_curve(m: Model, freq: str = "W", horizon: int = 16) -> pd.DataFrame:
    en = m.enrolled[["student_id", "enrollment_date"]]
    ev = m.events().merge(en, on="student_id")
    ev = ev[ev.ts >= ev.enrollment_date]
    unit = 7 if freq == "W" else 30
    ev["k"] = ((ev.ts - ev.enrollment_date).dt.days // unit).astype(int)
    rows = []
    for k in range(horizon):
        eligible = en[en.enrollment_date + pd.Timedelta(days=unit * (k + 1)) <= m.data_end]
        if eligible.empty:
            break
        act = ev[(ev.k == k) & ev.student_id.isin(eligible.student_id)].student_id.nunique()
        rows.append({"period_since_enrollment": k, "eligible_learners": len(eligible), "retained_pct": round(act / len(eligible) * 100, 1)})
    return pd.DataFrame(rows)


def churn_by(m: Model, dim: str) -> pd.DataFrame:
    """Confirmed cancellation vs inferred 30-day inactivity, by a student dimension (as of window end)."""
    seg = segments(m)
    if seg.empty:
        return pd.DataFrame()
    base = seg.merge(m.students[["student_id", "course_name", "acquisition_channel", "cohort", "subject"]], on="student_id", how="left")
    if dim not in base.columns:
        return pd.DataFrame()
    base["cancelled"] = base.status == "Cancelled"
    base["inactive_30d"] = (~base.status.isin(["Completed", "Cancelled"])) & ((base.days_since_activity.fillna(999) > 30)) & \
                           ((pd.Timestamp(m.end) - base.enrolled_on).dt.days > 30)
    g = base.groupby(dim)
    return pd.DataFrame({"learners": g.size(), "confirmed_cancel_pct": (g.cancelled.mean() * 100).round(1),
                         "inferred_inactive_pct": (g.inactive_30d.mean() * 100).round(1)}).sort_values(
        "inferred_inactive_pct", ascending=False).reset_index()


def renewals(m: Model) -> pd.DataFrame:
    en = m.enrollments
    comp = en[(en.enrollment_status == "Completed") & (en.completion_date <= m.end - pd.Timedelta(days=14))]
    ren = en[en.is_renewal.astype(bool)]
    comp = comp.assign(renewed=comp.student_id.isin(ren.student_id)).merge(m.courses[["course_id", "course_name"]], on="course_id")
    if comp.empty:
        return pd.DataFrame()
    g = comp.groupby("course_name")
    return pd.DataFrame({"completed": g.size(), "renewed": g.renewed.sum(), "renewal_rate_pct": (g.renewed.mean() * 100).round(1)}).reset_index()


def status_breakdown(m: Model) -> pd.DataFrame:
    return m.enrollments.enrollment_status.value_counts().rename_axis("enrollment_status").reset_index(name="enrollments")


def reactivations(m: Model, gap_days: int = 30) -> int:
    ev = m.events().sort_values("ts")
    ev = ev.assign(day=ev.ts.dt.normalize()).drop_duplicates(["student_id", "day"])
    gaps = ev.groupby("student_id").day.diff().dt.days
    w = ev[(gaps >= gap_days) & (ev.ts >= m.start) & (ev.ts <= m.end + pd.Timedelta(hours=23))]
    return int(w.student_id.nunique())


def time_to_inactivity(m: Model) -> pd.DataFrame:
    seg = segments(m)
    if seg.empty:
        return pd.DataFrame(columns=["days_enrolled_before_last_activity"])
    d = seg[(seg.days_since_activity > 30) & ~seg.status.isin(["Completed", "Cancelled"])]
    days = (d.last_activity - d.enrolled_on).dt.days.dropna()
    return days.rename("days_enrolled_before_last_activity").to_frame()


# ============================== F. revenue ==============================
def revenue_monthly(m: Model) -> pd.DataFrame:
    p = m.payments[m.payments.payment_date <= m.end + pd.Timedelta(hours=23)]
    mo = p.payment_date.dt.to_period("M")
    coll = p[p.payment_status == "Success"].groupby(mo).amount.sum()
    ref = p[p.payment_status == "Refunded"].groupby(mo).refund_amount.sum()
    en = m.enrollments[m.enrollments.enrollment_date <= m.end]
    book = en.groupby(en.enrollment_date.dt.to_period("M")).price.sum()
    d = pd.DataFrame({"gross_bookings": book, "collections": coll, "refunds": ref}).fillna(0)
    d["net_revenue"] = d.collections - d.refunds
    d.index = d.index.to_timestamp()
    return d.reset_index(names="month")


def waterfall(m: Model) -> pd.DataFrame:
    """Bridge for enrollments made in the window: list value → discounts → bookings → cancelled (unbilled)
    → not yet collected (scheduled, pending or failed) → collected → refunds → net cash."""
    en = W(m.enrollments, "enrollment_date", m.start, m.end)
    if en.empty:
        return pd.DataFrame()
    p = m.payments[m.payments.enrollment_id.isin(en.enrollment_id)]
    collected = p.loc[p.payment_status == "Success"].groupby("enrollment_id").amount.sum()
    refunded = p.loc[p.payment_status == "Refunded", "refund_amount"].sum()
    paid = en.enrollment_id.map(collected).fillna(0)
    canc = en.enrollment_status == "Cancelled"
    list_value = (en.price + en.discount).sum()
    unbilled = (en.price - paid)[canc].clip(lower=0).sum()
    total_collected = paid.sum()
    outstanding = en.price.sum() - total_collected - unbilled
    steps = [("List-price value", list_value, "absolute"), ("Discounts", -en.discount.sum(), "relative"),
             ("Gross bookings", en.price.sum(), "total"), ("Cancelled before billing", -unbilled, "relative"),
             ("Not yet collected", -outstanding, "relative"), ("Collected", total_collected, "total"),
             ("Refunds", -refunded, "relative"), ("Net cash", total_collected - refunded, "total")]
    return pd.DataFrame(steps, columns=["step", "amount", "measure"])


def revenue_by(m: Model, dim: str) -> pd.DataFrame:
    p = W(m.payments, "payment_date", m.start, m.end)
    p = p.assign(net=np.where(p.payment_status == "Success", p.amount, 0) - np.where(p.payment_status == "Refunded", p.refund_amount, 0))
    d = student_dim(m, p, dim).groupby(dim).agg(net_revenue=("net", "sum"), paying_students=("student_id", "nunique"))
    d["share_pct"] = (d.net_revenue / d.net_revenue.sum() * 100).round(1) if d.net_revenue.sum() else 0
    return d.sort_values("net_revenue", ascending=False).reset_index()


def payment_status(m: Model) -> pd.DataFrame:
    p = W(m.payments, "payment_date", m.start, m.end)
    return p.groupby("payment_status").agg(transactions=("payment_id", "count"), amount=("amount", "sum"),
                                           refund_amount=("refund_amount", "sum")).reset_index()


def discount_effectiveness(m: Model) -> pd.DataFrame:
    en = m.enrollments[~m.enrollments.is_renewal.astype(bool)].merge(m.courses[["course_id", "list_price", "duration_weeks"]], on="course_id")
    en = en[en.enrollment_date <= m.end]
    if en.empty:
        return pd.DataFrame()
    en["discount_band"] = pd.cut(en.discount / en.list_price * 100, [-1, 1, 12, 22, 40], labels=["None", "~10%", "~20%", "~30%"])
    en["matured"] = en.enrollment_date + pd.to_timedelta(en.duration_weeks * 7, unit="D") <= m.end
    g = en.groupby("discount_band", observed=True)
    return pd.DataFrame({"enrollments": g.size(), "avg_price": g.price.mean().round(0),
                         "refund_rate_pct": (g.refund_amount.apply(lambda x: (x > 0).mean()) * 100).round(1),
                         "completion_pct_matured": g.apply(lambda d: (d.loc[d.matured, "enrollment_status"] == "Completed").mean() * 100
                                                           if d.matured.any() else np.nan).round(1)}).reset_index()


def recognized_revenue(m: Model) -> float:
    """ESTIMATE: straight-line recognition of each non-cancelled enrollment price over its course duration."""
    en = m.enrollments[m.enrollments.enrollment_status != "Cancelled"].merge(m.courses[["course_id", "duration_weeks"]], on="course_id")
    if en.empty:
        return 0.0
    s, e = m.start, m.end + DAY
    end_c = en.enrollment_date + pd.to_timedelta(en.duration_weeks * 7, unit="D")
    overlap = (np.minimum(end_c, e) - np.maximum(en.enrollment_date, s)).dt.days.clip(lower=0)
    return float((en.price / (en.duration_weeks * 7) * overlap).sum())


def unit_economics(m: Model) -> pd.DataFrame:
    """Per channel. CAC uses recorded spend only. 'Observed revenue per learner' is cash to date for learners
    enrolled ≥180 days before data end — a lower bound on lifetime value, not a projection."""
    s = m.students
    spend = W(m.marketing, "date", m.start, m.end).groupby("channel").spend.sum()
    new = W(m.enrollments[~m.enrollments.is_renewal.astype(bool)], "enrollment_date", m.start, m.end)
    new = student_dim(m, new, "acquisition_channel").groupby("acquisition_channel").student_id.nunique()
    pay = m.payments
    net = (pay.amount.where(pay.payment_status == "Success", 0) - pay.refund_amount.where(pay.payment_status == "Refunded", 0)).groupby(pay.student_id).sum()
    mature = m.enrolled[m.enrolled.enrollment_date <= m.data_end - pd.Timedelta(days=180)]
    obs = mature.assign(rev=mature.student_id.map(net).fillna(0)).groupby("acquisition_channel").rev.mean()
    d = pd.DataFrame({"spend": spend, "new_enrollments": new, "observed_rev_per_learner_180d": obs.round(0)})
    d.index.name = "channel"
    d = d.reindex(sorted(set(s.acquisition_channel))).fillna({"new_enrollments": 0})
    d["cac"] = (d.spend / d.new_enrollments.replace(0, np.nan)).round(0)
    d["rev_to_cac"] = (d.observed_rev_per_learner_180d / d.cac).round(2)
    return d.reset_index()


# ============================== G. courses & instructors ==============================
def course_scorecard(m: Model) -> pd.DataFrame:
    from . import metrics as M
    rows = []
    for c in m.courses.itertuples():
        sub = Model(m.tables, type(m.f)(m.f.start, m.f.end, (c.course_id,), m.f.subjects, m.f.cohorts, m.f.channels, m.f.statuses))
        if sub.enrolled.empty:
            continue
        rows.append({"course": c.course_name, "level": c.course_level, "weeks": c.duration_weeks,
                     "new_enrollments": M.compute(sub, "new_enrollments"), "active_learners": M.compute(sub, "active_learners"),
                     "attendance_pct": M.compute(sub, "attendance_rate"),
                     "completion_pct": M.compute(sub, "completion_rate", sub.data_start, sub.end),
                     "avg_score_pct": M.compute(sub, "avg_score"), "satisfaction": M.compute(sub, "satisfaction", sub.data_start, sub.end),
                     "refund_rate_pct": M.compute(sub, "refund_rate", sub.data_start, sub.end),
                     "net_revenue": M.compute(sub, "net_revenue"), "inactive_30d_pct": M.compute(sub, "inactive_rate")})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    num = d.select_dtypes("number").columns
    d[num] = d[num].astype(float).round(1)
    return d


def course_quadrant(m: Model) -> pd.DataFrame:
    sc = course_scorecard(m)
    if sc.empty:
        return sc
    en = m.enrollments[~m.enrollments.is_renewal.astype(bool)].merge(m.courses[["course_id", "course_name"]], on="course_id")
    e, L = m.end, pd.Timedelta(days=90)
    cur = W(en, "enrollment_date", e - L + DAY, e).groupby("course_name").size()
    prev = W(en, "enrollment_date", e - 2 * L + DAY, e - L).groupby("course_name").size()
    sc["growth_pct"] = sc.course.map(((cur - prev) / prev.replace(0, np.nan) * 100).round(1))
    z = lambda s: (s - s.mean()) / (s.std(ddof=0) or 1)  # noqa: E731
    sc["quality_index"] = (z(sc.completion_pct.fillna(sc.completion_pct.mean())) + z(sc.avg_score_pct.fillna(sc.avg_score_pct.mean()))
                           + z(sc.attendance_pct.fillna(sc.attendance_pct.mean())) + z(sc.satisfaction.fillna(sc.satisfaction.mean()))).div(4).round(2)
    return sc


def instructor_matrix(m: Model) -> pd.DataFrame:
    """Fair comparison: score uplift = learner score minus the course×topic average (adjusts for course
    level and topic difficulty), with sample size and a 95% interval. Raw scores are not ranked."""
    a = assessments_w(m)
    if a.empty:
        return pd.DataFrame()
    a["resid"] = a.pct - a.groupby(["course_id", "topic"]).pct.transform("mean")
    cls = W(m.classes, "class_date", m.start, m.end)
    att = W(m.attendance, "class_date", m.start, m.end).merge(m.classes[["class_id", "instructor_id"]], on="class_id")
    per_class = att.groupby(["instructor_id", "class_id"]).attended.agg(["sum", "count"])
    g = a.groupby("instructor_id")
    d = pd.DataFrame({"learners_assessed": g.student_id.nunique(), "assessments": g.size(),
                      "score_vs_course_avg_pp": g.resid.mean().round(2),
                      "ci95_pp": (1.96 * g.resid.std() / np.sqrt(g.size())).round(2)})
    d["classes_taught"] = cls.groupby("instructor_id").size()
    d["avg_attended_per_class"] = per_class.groupby("instructor_id")["sum"].mean().round(1)
    d["attendance_pct"] = (per_class.groupby("instructor_id")["sum"].sum() / per_class.groupby("instructor_id")["count"].sum() * 100).round(1)
    d["utilization_pct"] = (d.avg_attended_per_class / 150 * 100).round(1)
    d = d.join(m.instructors.set_index("instructor_id")[["instructor_name", "subject", "experience_band"]])
    d["signal"] = np.where(d.score_vs_course_avg_pp - d.ci95_pp > 0, "Above course avg",
                           np.where(d.score_vs_course_avg_pp + d.ci95_pp < 0, "Below course avg", "Not distinguishable"))
    peers = m.classes.groupby("course_id").instructor_id.nunique()
    solo = m.classes.assign(n=m.classes.course_id.map(peers)).groupby("instructor_id").n.max() <= 1
    solo_ids = solo[solo].index
    d.loc[d.index.isin(solo_ids), ["score_vs_course_avg_pp", "ci95_pp"]] = np.nan
    d.loc[d.index.isin(solo_ids), "signal"] = "Sole instructor – no in-course peer"
    return d.reset_index()


# ============================== H. marketing ==============================
def channel_performance(m: Model, dim: str = "acquisition_channel") -> pd.DataFrame:
    s = W(m.students, "signup_date", m.start, m.end)
    if s.empty:
        return pd.DataFrame()
    key = "channel" if dim == "acquisition_channel" else "campaign"
    spend = W(m.marketing, "date", m.start, m.end).groupby(key).spend.sum()
    s = s.assign(enrolled=s.enrollment_date.notna(), days_to_enrol=(s.enrollment_date - s.signup_date.dt.normalize()).dt.days)
    en = m.enrollments.drop_duplicates("student_id").set_index("student_id")
    s["refunded"] = s.student_id.map(en.refund_amount).fillna(0) > 0
    ev = m.events()
    s_en = s[s.enrolled]
    ret = ev.merge(s_en[["student_id", "enrollment_date"]], on="student_id")
    ret = ret[(ret.ts >= ret.enrollment_date + pd.Timedelta(days=60)) & (ret.ts < ret.enrollment_date + pd.Timedelta(days=90))]
    s["active_day60_90"] = s.student_id.isin(ret.student_id)
    s["mature90"] = s.enrolled & (s.enrollment_date <= m.data_end - pd.Timedelta(days=90))
    pay = m.payments
    net = (pay.amount.where(pay.payment_status == "Success", 0) - pay.refund_amount.where(pay.payment_status == "Refunded", 0)).groupby(pay.student_id).sum()
    s["net_revenue"] = s.student_id.map(net).fillna(0)
    g = s.groupby(dim)
    d = pd.DataFrame({"leads": g.size(), "qualified_pct": (g.lead_qualified.mean() * 100).round(1), "enrolled": g.enrolled.sum(),
                      "lead_to_enrol_pct": (g.enrolled.mean() * 100).round(1), "median_days_to_enrol": g.days_to_enrol.median(),
                      "net_revenue_to_date": g.net_revenue.sum().round(0),
                      "refund_rate_pct": g.apply(lambda x: x.loc[x.enrolled, "refunded"].mean() * 100 if x.enrolled.any() else np.nan).round(1),
                      "retained_day60_90_pct": g.apply(lambda x: x.loc[x.mature90, "active_day60_90"].mean() * 100 if x.mature90.any() else np.nan).round(1),
                      "mature_learners_n": g.mature90.sum()})
    d["spend"] = d.index.map(spend)
    d["cpl"] = (d.spend / d.leads).round(0)
    d["cac"] = (d.spend / d.enrolled.replace(0, np.nan)).round(0)
    d["roas_to_date"] = (d.net_revenue_to_date / d.spend).round(2)
    return d.reset_index()


def spend_vs_revenue_monthly(m: Model) -> pd.DataFrame:
    mk = m.marketing[m.marketing.date <= m.end]
    sp = mk.groupby(mk.date.dt.to_period("M")).spend.sum()
    rv = revenue_monthly(m).set_index("month").net_revenue
    rv.index = rv.index.to_period("M")
    d = pd.DataFrame({"spend": sp, "net_revenue": rv}).fillna(0)
    d.index = d.index.to_timestamp()
    return d.reset_index(names="month")


# ============================== I. support ==============================
def tickets_w(m: Model) -> pd.DataFrame:
    t = W(m.tickets, "created_at", m.start, m.end).copy()
    t["first_response_h"] = ((t.first_response_at - t.created_at).dt.total_seconds() / 3600).round(2)
    t["resolution_h"] = ((t.resolved_at - t.created_at).dt.total_seconds() / 3600).round(2)
    t["sla_met"] = t.first_response_h <= t.priority.map(SLA_HOURS)
    return t.merge(m.courses[["course_id", "course_name"]], on="course_id", how="left")


def support_summary(m: Model) -> dict:
    t = tickets_w(m)
    if t.empty:
        return {}
    end = m.end + pd.Timedelta(hours=23, minutes=59)
    allt = m.tickets
    backlog = int(((allt.created_at <= end) & (allt.resolved_at.isna() | (allt.resolved_at > end))).sum())
    rep = t.sort_values("created_at").groupby("student_id").created_at.diff().dt.days
    return {"tickets": len(t), "median_first_response_h": float(t.first_response_h.median()),
            "median_resolution_h": float(t.resolution_h.median()), "sla_pct": float(t.sla_met.mean() * 100),
            "reopen_pct": float((t.status == "Reopened").mean() * 100), "backlog": backlog,
            "repeat_contact_pct": float((rep <= 30).sum() / t.student_id.nunique() * 100),
            "csat": float(t.csat_score.mean()) if t.csat_score.notna().any() else None,
            "csat_responses": int(t.csat_score.notna().sum())}


def pareto(m: Model) -> pd.DataFrame:
    t = tickets_w(m)
    d = t.issue_category.value_counts().rename_axis("issue_category").reset_index(name="tickets")
    d["cumulative_pct"] = (d.tickets.cumsum() / d.tickets.sum() * 100).round(1)
    return d


def sla_by_priority(m: Model) -> pd.DataFrame:
    t = tickets_w(m)
    g = t.groupby("priority")
    return pd.DataFrame({"tickets": g.size(), "sla_hours": g.priority.first().map(SLA_HOURS),
                         "median_first_response_h": g.first_response_h.median().round(1),
                         "sla_met_pct": (g.sla_met.mean() * 100).round(1)}).reindex(list(SLA_HOURS)).dropna().reset_index()


def csat_vs_resolution(m: Model) -> pd.DataFrame:
    t = tickets_w(m).dropna(subset=["csat_score", "resolution_h"])
    t["resolution_band"] = pd.cut(t.resolution_h, [0, 12, 24, 48, 72, 1e6], labels=["<12h", "12-24h", "24-48h", "48-72h", ">72h"])
    g = t.groupby("resolution_band", observed=True)
    return pd.DataFrame({"rated_tickets": g.size(), "avg_csat": g.csat_score.mean().round(2)}).reset_index()


def support_engagement(m: Model) -> pd.DataFrame:
    """Association: learner active days in the 30 days after a ticket, by resolution speed. Not causal."""
    t = tickets_w(m).dropna(subset=["resolution_h"])
    t = t[t.created_at <= m.data_end - pd.Timedelta(days=30)].drop_duplicates("student_id")
    if t.empty:
        return pd.DataFrame()
    ev = m.events().assign(day=lambda d: d.ts.dt.normalize())
    x = ev.merge(t[["student_id", "created_at"]], on="student_id")
    after = x[(x.ts > x.created_at) & (x.ts <= x.created_at + pd.Timedelta(days=30))].groupby("student_id").day.nunique()
    before = x[(x.ts < x.created_at) & (x.ts >= x.created_at - pd.Timedelta(days=30))].groupby("student_id").day.nunique()
    t["active_days_30d_before"] = t.student_id.map(before).fillna(0)
    t["active_days_30d_after"] = t.student_id.map(after).fillna(0)
    t["group"] = np.where(t.resolution_h > 72, "Resolved > 72h", "Resolved ≤ 72h")
    g = t.groupby("group")
    return pd.DataFrame({"learners": g.size(), "avg_active_days_before": g.active_days_30d_before.mean().round(1),
                         "avg_active_days_after": g.active_days_30d_after.mean().round(1)}).assign(
        change=lambda d: (d.avg_active_days_after - d.avg_active_days_before).round(1)).reset_index()


# ============================== J. student success / early warning ==============================
INDICATORS = {  # name: (points, description)
    "inactive_30d": (3, "No learning activity for > 30 days"),
    "inactive_14d": (2, "No learning activity for 15–30 days"),
    "attendance_drop": (2, "Live-class attendance in last 28 days below 40%"),
    "score_decline": (2, "Average of last 2 assessments ≥ 10 pp below the first 2"),
    "low_scores": (1, f"Last 3 assessments average below {PASS_MARK:.0f}%"),
    "missed_assignments": (1, "No assignment submitted in last 21 days (but submitted before)"),
    "repeat_support": (1, "≥ 2 support tickets in last 30 days"),
}
INTERVENTION = {
    "inactive_30d": "Counsellor call to re-engage; offer catch-up plan",
    "inactive_14d": "Personalised nudge (WhatsApp/app) with next class & recorded lesson",
    "attendance_drop": "Mentor check-in on schedule fit; suggest alternate batch",
    "score_decline": "Academic mentor: targeted practice on weak topics",
    "low_scores": "Doubt-clearing session + remedial worksheet",
    "missed_assignments": "Reminder + deadline extension offer",
    "repeat_support": "Escalate to support lead; resolve root issue first",
}


def student_risk(m: Model, as_of=None) -> pd.DataFrame:
    """Explainable rules-based risk flags (not a statistical prediction). No personal attributes are used."""
    as_of = pd.Timestamp(as_of or m.end)
    le = learner_engagement(m, as_of)
    if le.empty:
        return le
    le = le[~le.status.isin(["Completed", "Cancelled"]) & (le.enrolled_on <= as_of - pd.Timedelta(days=14))]
    a = m.assessments[m.assessments.assessment_date <= as_of].sort_values("assessment_date")
    g = a.groupby("student_id").pct
    first2, last2, last3, cnt = g.apply(lambda x: x.iloc[:2].mean()), g.apply(lambda x: x.iloc[-2:].mean()), g.apply(lambda x: x.iloc[-3:].mean()), g.size()
    act = m.activity[m.activity.activity_date <= as_of]
    sub = act[act.activity_type == "assignment_submit"]
    last_sub = sub.groupby("student_id").activity_date.max()
    tk = m.tickets[(m.tickets.created_at <= as_of) & (m.tickets.created_at > as_of - pd.Timedelta(days=30))].groupby("student_id").size()
    d = le.set_index("student_id")
    dsa = d.days_since_activity.fillna(999)
    flags = pd.DataFrame(index=d.index)
    flags["inactive_30d"] = dsa > 30
    flags["inactive_14d"] = (dsa > 14) & (dsa <= 30)
    flags["attendance_drop"] = d.attendance_28d_pct < 40
    has4 = d.index.map(cnt).fillna(0) >= 4
    flags["score_decline"] = has4 & ((d.index.map(last2) - d.index.map(first2)) <= -10)
    flags["low_scores"] = d.index.map(last3) < PASS_MARK
    ls = d.index.map(last_sub)
    flags["missed_assignments"] = pd.notna(ls) & ((as_of - pd.to_datetime(ls)).days > 21) if len(d) else []
    flags["repeat_support"] = d.index.map(tk).fillna(0) >= 2
    flags = flags.fillna(False).astype(bool)
    pts = pd.Series({k: v[0] for k, v in INDICATORS.items()})
    d["risk_score"] = flags.mul(pts).sum(axis=1)
    d["risk_level"] = np.select([d.risk_score >= 5, d.risk_score >= 3, d.risk_score >= 1], ["High", "Medium", "Low"], "None")
    d["indicators"] = flags.apply(lambda r: ", ".join(INDICATORS[k][1] for k in r.index[r]), axis=1)
    top = flags.mul(pts).idxmax(axis=1).where(d.risk_score > 0)
    d["recommended_intervention"] = top.map(INTERVENTION).fillna("")
    d["score_trend_pp"] = (d.index.map(last2) - d.index.map(first2)).round(1)
    d["follow_up_status"] = "Not started"
    cols = ["course", "cohort", "risk_level", "risk_score", "indicators", "last_activity", "days_since_activity",
            "attendance_28d_pct", "score_trend_pp", "recommended_intervention", "follow_up_status"]
    return d[cols].sort_values(["risk_score", "days_since_activity"], ascending=False).reset_index()


def risk_backtest(m: Model, lookback_days: int = 60) -> dict:
    """Time-based validation of the rules: flag at T0 = end − 60 days; outcome = cancelled or no activity
    in (T0+30, T0+60]. Reports precision and recall. This validates the rules, it is not an ML model."""
    t0 = m.end - pd.Timedelta(days=lookback_days)
    r = student_risk(m, t0)
    if r.empty:
        return {}
    ev = m.events()
    later = set(ev[(ev.ts > t0 + pd.Timedelta(days=30)) & (ev.ts <= t0 + pd.Timedelta(days=lookback_days))].student_id)
    canc = set(m.enrollments[(m.enrollments.cancellation_date > t0)].student_id)
    r["outcome_disengaged"] = ~r.student_id.isin(later) | r.student_id.isin(canc)
    out = {"as_of": t0, "learners": len(r), "base_rate_pct": round(r.outcome_disengaged.mean() * 100, 1)}
    for name, mask in (("High", r.risk_level == "High"), ("High+Medium", r.risk_level.isin(["High", "Medium"]))):
        tp = (mask & r.outcome_disengaged).sum()
        out[name] = {"flagged": int(mask.sum()), "precision_pct": round(tp / mask.sum() * 100, 1) if mask.sum() else None,
                     "recall_pct": round(tp / r.outcome_disengaged.sum() * 100, 1) if r.outcome_disengaged.sum() else None}
    return out
