"""Online vs offline (learning-centre) analytics, Vedantu-style.

Online = LIVE interactive classes on the app; Offline = classroom batches at learning centres.
Each mode is analysed with the same metric layer (metrics.kpi on a mode-filtered Model), so numbers
match every other page. Mode is self-selected (offline learners live near a centre and chose it), so
gaps between modes are associations, never the effect of the mode itself."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import metrics as M
from .insights import Insight
from .model import Model

MODES = ["Online", "Offline"]
W = Model.window
COMPARE = ["leads", "new_enrollments", "lead_to_enrol", "active_learners", "activation_rate", "attendance_rate", "avg_score",
           "pass_rate", "retention_rate", "completion_rate", "churn_confirmed", "inactive_rate", "net_revenue", "arps",
           "refund_rate", "payment_failure_rate", "tickets", "csat", "satisfaction", "nps"]
GAP_KEYS = ["lead_to_enrol", "attendance_rate", "retention_rate", "inactive_rate", "churn_confirmed", "avg_score", "arps",
            "refund_rate", "nps"]
TECH = ["Login / Access", "Technical Issue"]


def modes_in(m: Model) -> list[str]:
    present = set(m.students.learning_mode.dropna())
    return [x for x in MODES if x in present and (not m.f.modes or x in m.f.modes)]


def mode_model(m: Model, mode: str) -> Model:
    return Model(m.tables, replace(m.f, modes=(mode,)))


def comparison(m: Model, keys: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, dict[str, M.KPI]]]:
    """Every comparable KPI for each mode, from the shared metric layer."""
    keys = keys or COMPARE
    kpis = {md: {k: M.kpi(mode_model(m, md), k) for k in keys} for md in modes_in(m)}
    rows = []
    for k in keys:
        d = M.METRICS[k]
        row = {"metric": d.name, "key": k}
        for md in kpis:
            kp = kpis[md][k]
            row[md] = M.fmt(kp.value, kp.unit)
            row[f"{md} vs prev"] = M.fmt_change(kp)
        if len(kpis) == 2:
            on, off = kpis["Online"][k].value, kpis["Offline"][k].value
            if on is not None and off is not None and not (pd.isna(on) or pd.isna(off)):
                diff = off - on
                row["gap (offline − online)"] = (f"{diff:+.1f} pp" if d.unit in ("pct", "nps") else
                                                 f"{diff / on * 100:+.0f}%" if on else "n/a")
                if d.unit in ("count", "inr") and k not in ("arps",):
                    row["leads on"] = "volume (scale differs)"
                else:
                    better = (diff > 0) == d.higher_is_better
                    row["leads on"] = "≈ equal" if abs(diff) < 1e-9 else ("Offline" if better else "Online")
        rows.append(row)
    return pd.DataFrame(rows), kpis


def mode_trend(m: Model) -> pd.DataFrame:
    """Monthly first enrollments and lead → enrollment by mode, from data start to the window end."""
    s = m.students[m.students.signup_date <= m.end + pd.Timedelta(days=1)]
    leads = s.groupby([s.signup_date.dt.to_period("M").dt.to_timestamp(), "learning_mode"]).size().rename("leads")
    en = s[s.enrollment_date.notna() & (s.enrollment_date <= m.end)]
    enr = en.groupby([en.enrollment_date.dt.to_period("M").dt.to_timestamp(), "learning_mode"]).size().rename("enrollments")
    out = pd.concat([leads, enr], axis=1).fillna(0).reset_index()
    out.columns = ["month", "learning_mode", "leads", "enrollments"]
    out["offline_share_pct"] = out.groupby("month").enrollments.transform(lambda x: x / x.sum() * 100 if x.sum() else np.nan)
    return out


def channel_mix(m: Model) -> pd.DataFrame:
    en = m.enrolled
    if en.empty:
        return pd.DataFrame()
    t = pd.crosstab(en.acquisition_channel, en.learning_mode, normalize="columns").mul(100).round(1)
    return t.reset_index()


def engagement_profile(m: Model) -> pd.DataFrame:
    """How learners in each mode learn: app use, classes, timing, support (window, active learners)."""
    rows = []
    for md in modes_in(m):
        mm = mode_model(m, md)
        weeks = max((m.end - m.start).days + 1, 7) / 7
        ev = mm.events()
        ev = W(ev, "ts", m.start, m.end)
        active = ev.student_id.nunique()
        act = W(mm.activity, "activity_date", m.start, m.end)
        att = W(mm.attendance, "class_date", m.start, m.end)
        tk = W(mm.tickets, "created_at", m.start, m.end)
        per = lambda x: round(x / active / weeks, 2) if active else np.nan  # noqa: E731
        rows.append({
            "mode": md, "active learners": active,
            "app sessions / learner / week": per(len(act)),
            "video minutes / learner / week": per(act.loc[act.activity_type == "video_watch", "duration_minutes"].sum()),
            "practice quizzes / learner / week": per((act.activity_type == "practice_quiz").sum()),
            "classes attended / learner / week": per(att.attended.sum()),
            "peak activity hour": f"{int(act.activity_date.dt.hour.mode().iat[0]):02d}:00" if len(act) else "n/a",
            "tickets / 100 learners": round(len(tk) / active * 100, 1) if active else np.nan,
            "tech-friction tickets %": round(tk.issue_category.isin(TECH).mean() * 100, 1) if len(tk) else np.nan,
            "top ticket reason": tk.issue_category.mode().iat[0] if len(tk) else "n/a",
        })
    return pd.DataFrame(rows)


def centre_scorecard(m: Model) -> pd.DataFrame:
    c = m.centres
    off = m.students[m.students.learning_mode == "Offline"]
    if c.empty or off.empty:
        return pd.DataFrame()
    who = off[["student_id", "centre_id"]]
    att = W(m.attendance, "class_date", m.start, m.end).merge(who, on="student_id")
    pw = m.prev_window()
    att_p = W(m.attendance, "class_date", *pw).merge(who, on="student_id") if pw else att.iloc[0:0]
    last28 = W(m.attendance[m.attendance.attended], "class_date", m.end - pd.Timedelta(days=27), m.end).merge(who, on="student_id")
    en = off[off.enrollment_date.between(m.start, m.end + pd.Timedelta(hours=23))]
    pay = W(m.payments, "payment_date", m.start, m.end).merge(who, on="student_id")
    pay = pay.assign(net=pay.amount.where(pay.payment_status == "Success", 0) - pay.refund_amount.where(pay.payment_status == "Refunded", 0))
    asm = W(m.assessments, "assessment_date", m.start, m.end).merge(who, on="student_id")
    tk = W(m.tickets, "created_at", m.start, m.end).merge(who, on="student_id")
    out = c.set_index("centre_id")[["centre_name", "city", "seat_capacity"]].copy()
    out["active_learners_28d"] = last28.groupby("centre_id").student_id.nunique()
    out["utilisation_pct"] = out.active_learners_28d / out.seat_capacity * 100
    out["new_enrollments"] = en.groupby("centre_id").size()
    out["attendance_pct"] = att.groupby("centre_id").attended.mean() * 100
    out["attendance_prev_pct"] = att_p.groupby("centre_id").attended.mean() * 100
    out["attendance_change_pp"] = out.attendance_pct - out.attendance_prev_pct
    out["class_seats"] = att.groupby("centre_id").size()
    out["avg_score_pct"] = asm.groupby("centre_id").pct.mean()
    out["net_revenue"] = pay.groupby("centre_id").net.sum()
    out["revenue_per_seat"] = out.net_revenue / out.seat_capacity
    out["facility_tickets"] = tk[tk.issue_category == "Centre Facilities"].groupby("centre_id").size()
    out["csat"] = tk.groupby("centre_id").csat_score.mean()
    out = out[out.index.isin(m.students.centre_id.dropna())]
    fill0 = ["active_learners_28d", "new_enrollments", "class_seats", "net_revenue", "facility_tickets", "utilisation_pct", "revenue_per_seat"]
    out[fill0] = out[fill0].fillna(0)
    return out.reset_index().round(1).sort_values("utilisation_pct", ascending=False)


def centre_trend(m: Model) -> pd.DataFrame:
    """Monthly classroom attendance % by centre (data start → window end)."""
    off = m.students.loc[m.students.learning_mode == "Offline", ["student_id", "centre_name"]]
    a = m.attendance[m.attendance.class_date <= m.end + pd.Timedelta(days=1)].merge(off, on="student_id")
    if a.empty:
        return pd.DataFrame()
    return (a.groupby([a.class_date.dt.to_period("M").dt.to_timestamp().rename("month"), "centre_name"]).attended.mean()
            .mul(100).round(1).rename("attendance_pct").reset_index())


# ---------------- insights ----------------
def _imp(mag: float, n: float, w: float = 2.0) -> float:
    return round(w * min(mag, 3) * np.log10(max(n, 10)), 2)


def mode_insights(m: Model, kpis: dict[str, dict[str, M.KPI]], centres: pd.DataFrame, profile: pd.DataFrame) -> list[Insight]:
    out: list[Insight] = []
    n = {md: max((kpis[md]["active_learners"].value if "active_learners" in kpis[md] else 0) or 0, 1) for md in kpis}
    # 1) biggest gaps between modes (association: mode is self-selected)
    if len(kpis) == 2:
        gaps = []
        for k in [k for k in GAP_KEYS if k in kpis["Online"]]:
            on, off = kpis["Online"][k], kpis["Offline"][k]
            if on.value is None or off.value is None or pd.isna(on.value) or pd.isna(off.value) or not on.value:
                continue
            pp = on.unit in ("pct", "nps")
            diff = off.value - on.value
            mag = abs(diff) / 2 if pp else abs(diff / on.value * 100) / 5
            gaps.append((mag, k, diff, pp, on, off))
        for mag, k, diff, pp, on, off in sorted(gaps, reverse=True)[:3]:
            if mag < 1:
                continue
            d = M.METRICS[k]
            lead = "Offline" if (diff > 0) == d.higher_is_better else "Online"
            ch = f"{diff:+.1f} pp" if pp else f"{diff / on.value * 100:+.0f}%"
            out.append(Insight(
                f"{lead} leads on {d.name.lower()}: offline {M.fmt(off.value, off.unit)} vs online {M.fmt(on.value, on.unit)}",
                d.name, f"Offline {M.fmt(off.value, off.unit)} (≈{int(n['Offline'])} active)",
                f"Online {M.fmt(on.value, on.unit)} (≈{int(n['Online'])} active)", ch, "Online vs Offline",
                "Learners choose their mode and offline learners live near a centre, so this gap is an association, not the effect of the mode.",
                ("Bring what works offline into online: fixed batch slots, a mentor check-in and a weekly in-person-style test."
                 if lead == "Offline" else "Check why offline trails here: centre timetable, faculty load and fee/discount policy."),
                "positive" if lead == "Offline" else "risk", _imp(mag, min(n.values())), "Association (not causal)"))
    # 2) what is deteriorating inside each mode (vs its own previous period)
    for md, ks in kpis.items():
        reds = [k for k in ks.values() if k.rag == "red" and k.change is not None and k.key in COMPARE]
        reds.sort(key=lambda k: -abs(k.change) / (2 if k.unit in ("pct", "nps") else 5))
        for k in reds[:2]:
            out.append(Insight(
                f"[{md}] {k.name} {M.fmt_change(k)} vs previous period", k.name, M.fmt(k.value, k.unit), M.fmt(k.prev, k.unit),
                M.fmt_change(k), f"{md} learners",
                f"An {md.lower()}-specific decline: it is masked when both modes are blended.",
                ("Review LIVE-class experience: app/tech issues, class timing and teacher-student ratio."
                 if md == "Online" else "Review centre operations: batch timetable, faculty attendance and commute/facility complaints."),
                "risk", _imp(abs(k.change) / (2 if k.unit in ("pct", "nps") else 5), n[md], 2.5), "Verified metric"))
    # 3) centre-level flags
    if not centres.empty:
        c = centres
        drop = c[(c.attendance_change_pp <= -10) & (c.class_seats >= 100)]
        for _, r in drop.iterrows():
            out.append(Insight(
                f"{r.centre_name}: classroom attendance fell {abs(r.attendance_change_pp):.0f} pp", "Classroom attendance (centre)",
                f"{r.attendance_pct:.0f}% of {int(r.class_seats)} class seats", f"{r.attendance_prev_pct:.0f}% in previous period",
                f"{r.attendance_change_pp:+.1f} pp", r.centre_name,
                "Absenteeism at one centre precedes drop-off and refunds there; other centres are stable, so look locally.",
                "Centre head to check faculty substitutions, timetable changes and commute issues; call absent learners this week.",
                "risk", _imp(abs(r.attendance_change_pp) / 2, r.class_seats, 3), "Verified metric",
                c[["centre_name", "attendance_pct", "attendance_prev_pct", "attendance_change_pp", "class_seats"]]))
        for _, r in c[c.utilisation_pct >= 90].iterrows():
            out.append(Insight(
                f"{r.centre_name} is at {r.utilisation_pct:.0f}% of seat capacity", "Seat utilisation (28 days)",
                f"{int(r.active_learners_28d)} active learners / {int(r.seat_capacity)} seats", "90% utilisation threshold",
                f"{r.utilisation_pct - 90:+.0f} pp above threshold", r.centre_name,
                "New admissions here may be turned away or crowd existing batches.",
                "Open an extra batch slot or a hybrid (online) batch for overflow; route new leads to the nearest centre with seats.",
                "risk", _imp(r.utilisation_pct / 40, r.active_learners_28d, 2), "Verified metric"))
        low = c[(c.utilisation_pct < 45) & (c.seat_capacity >= 50)]
        if len(low):
            out.append(Insight(
                f"{len(low)} centre(s) under 45% seat utilisation: " + ", ".join(low.centre_name), "Seat utilisation (28 days)",
                ", ".join(f"{r.centre_name} {r.utilisation_pct:.0f}%" for _, r in low.iterrows()), "45% threshold", "below threshold",
                "Under-used centres", "Fixed centre costs are spread over fewer learners (cost data not available, so margin is not computed).",
                "Run local school tie-ups and walk-in events; offer online learners in these cities a trial classroom week.",
                "risk", _imp(1.5, low.seat_capacity.sum(), 1.5), "Verified metric", low[["centre_name", "seat_capacity", "active_learners_28d", "utilisation_pct"]]))
    # 4) online-specific: tech friction
    if not profile.empty and "Online" in set(profile["mode"]):
        p = profile.set_index("mode")
        tf = p.loc["Online", "tech-friction tickets %"]
        if pd.notna(tf) and tf >= 25:
            out.append(Insight(
                f"{tf:.0f}% of online support tickets are login/technical issues", "Ticket mix (online)",
                f"{tf:.0f}% tech-friction", "offline: n/a (no app-access issues)" if "Offline" not in p.index else
                f"offline {p.loc['Offline', 'tech-friction tickets %']:.0f}%", "", "Online learners",
                "Tech friction interrupts LIVE classes and is associated with lower engagement.",
                "Prioritise app login/AV fixes; add a 'join class' health check and a pre-class tech helpline.",
                "risk", _imp(tf / 20, n.get("Online", 10), 1.5), "Verified metric"))
    return sorted(out, key=lambda i: -i.impact)
