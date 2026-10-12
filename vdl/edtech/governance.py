"""Data quality & governance for the 12-table EdTech model. Checks never modify source data; cleaning
actions return a preview (affected rows + cleaned copy) that the user may download."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .generate import DATE_COLS, END

PKS = {"students": "student_id", "enrollments": "enrollment_id", "courses": "course_id", "learning_activity": "activity_id",
       "attendance": "attendance_id", "assessments": "assessment_id", "payments": "payment_id", "support_tickets": "ticket_id",
       "instructors": "instructor_id", "classes": "class_id", "student_surveys": "survey_id"}
FKS = [("enrollments", "student_id", "students", "student_id"), ("enrollments", "course_id", "courses", "course_id"),
       ("learning_activity", "student_id", "students", "student_id"), ("attendance", "student_id", "students", "student_id"),
       ("attendance", "class_id", "classes", "class_id"), ("assessments", "student_id", "students", "student_id"),
       ("payments", "student_id", "students", "student_id"), ("payments", "enrollment_id", "enrollments", "enrollment_id"),
       ("support_tickets", "student_id", "students", "student_id"), ("classes", "instructor_id", "instructors", "instructor_id"),
       ("classes", "course_id", "courses", "course_id"), ("student_surveys", "student_id", "students", "student_id"),
       ("students", "course_id", "courses", "course_id")]
REQUIRED = {"students": ["student_id", "signup_date", "acquisition_channel", "course_id"],
            "enrollments": ["enrollment_id", "student_id", "course_id", "enrollment_date", "price"],
            "payments": ["payment_id", "enrollment_id", "payment_date", "payment_status"],
            "assessments": ["assessment_id", "student_id", "score", "max_score"],
            "attendance": ["attendance_id", "student_id", "class_id", "attended"],
            "learning_activity": ["activity_id", "student_id", "activity_date"],
            "support_tickets": ["ticket_id", "student_id", "created_at"]}
WEIGHTS = {"Completeness": 20, "Uniqueness": 15, "Validity": 25, "Referential integrity": 25, "Consistency": 10, "Freshness": 5}
LINEAGE = [
    ("New enrollments, Activation, Completion, Refund rate, Gross bookings", "enrollments (+ courses.duration_weeks)"),
    ("Active learners, DAU/WAU/MAU, Retention, Inactivity, Segments", "learning_activity ∪ attendance(attended)"),
    ("Attendance rate, Instructor workload", "attendance → classes → instructors"),
    ("Avg score, Pass rate, Topic accuracy, Improvement", "assessments"),
    ("Collections, Refunds, Net revenue, Payment failure", "payments"),
    ("Leads, Lead → enrollment, Funnel, Channel performance", "students (+ enrollments, attendance, assessments)"),
    ("CAC, CPL, ROAS", "marketing_spend + students + payments"),
    ("Tickets, SLA, CSAT, Backlog", "support_tickets"),
    ("Satisfaction, NPS", "student_surveys"),
    ("Risk flags", "learning_activity, attendance, assessments, support_tickets"),
]


@dataclass
class GCheck:
    dimension: str
    table: str
    check: str
    checked: int
    failing: int
    rule: str
    sample: pd.DataFrame

    @property
    def status(self) -> str:
        if self.failing == 0:
            return "PASS"
        return "FAIL" if self.failing / max(self.checked, 1) >= 0.01 else "WARN"


def run_checks(t: dict[str, pd.DataFrame], today: pd.Timestamp | None = None) -> list[GCheck]:
    today = today or pd.Timestamp.today().normalize()
    out: list[GCheck] = []
    add = lambda dim, tab, chk, n, mask, rule, df: out.append(GCheck(dim, tab, chk, int(n), int(mask.sum()), rule, df[mask].head(10)))  # noqa: E731
    for tab, cols in REQUIRED.items():
        df = t[tab]
        for c in cols:
            if c not in df.columns:
                out.append(GCheck("Completeness", tab, f"Column `{c}` present", 1, 1, "required column exists", pd.DataFrame()))
                continue
            add("Completeness", tab, f"`{c}` not null", len(df), df[c].isna(), "required field is populated", df)
    for tab, pk in PKS.items():
        df = t[tab]
        add("Uniqueness", tab, f"`{pk}` unique", len(df), df[pk].duplicated(keep=False), "primary key appears once", df)
    for tab, cols in DATE_COLS.items():
        if tab == "classes":  # future classes are a legitimate schedule
            continue
        for c in cols:
            df = t[tab]
            add("Validity", tab, f"`{c}` not in future", len(df), df[c] > today + pd.Timedelta(days=1), "date ≤ today", df)
    s, e, a, p, tk = t["students"], t["enrollments"], t["assessments"], t["payments"], t["support_tickets"]
    add("Validity", "students", "enrollment after signup", s.enrollment_date.notna().sum(), s.enrollment_date < s.signup_date.dt.normalize(), "enrollment_date ≥ signup_date", s)
    add("Validity", "support_tickets", "resolved after created", tk.resolved_at.notna().sum(), tk.resolved_at < tk.created_at, "resolved_at ≥ created_at", tk)
    add("Validity", "support_tickets", "first response after created", len(tk), tk.first_response_at < tk.created_at, "first_response_at ≥ created_at", tk)
    add("Validity", "assessments", "score within 0..max_score", len(a), (a.score < 0) | (a.score > a.max_score), "0 ≤ score ≤ max_score", a)
    add("Validity", "assessments", "correct ≤ attempted", len(a), a.correct_answers > a.questions_attempted, "correct_answers ≤ questions_attempted", a)
    add("Validity", "payments", "amount non-negative", len(p), p.amount < 0, "amount ≥ 0", p)
    add("Validity", "enrollments", "refund ≤ price", len(e), e.refund_amount > e.price + 0.01, "refund_amount ≤ price", e)
    add("Validity", "support_tickets", "CSAT within 1..5", tk.csat_score.notna().sum(), ~tk.csat_score.between(1, 5) & tk.csat_score.notna(), "1 ≤ csat ≤ 5", tk)
    sv = t["student_surveys"]
    add("Validity", "student_surveys", "recommendation within 0..10", len(sv), ~sv.recommendation_score.between(0, 10), "0 ≤ score ≤ 10", sv)
    att = t["attendance"].merge(t["classes"][["class_id", "scheduled_minutes"]], on="class_id", how="left")
    add("Validity", "attendance", "duration ≤ scheduled", len(att), att.class_duration_minutes > att.scheduled_minutes, "class_duration_minutes ≤ scheduled_minutes", att)
    for tab, col, ref, rcol in FKS:
        df = t[tab]
        add("Referential integrity", tab, f"`{col}` → {ref}.{rcol}", df[col].notna().sum(), df[col].notna() & ~df[col].isin(t[ref][rcol]), "foreign key exists in parent table", df)
    ec = e.merge(s[["student_id", "course_id"]], on="student_id", how="left", suffixes=("", "_student"))
    add("Consistency", "enrollments", "course matches student's course", len(ec), ec.course_id_student.notna() & (ec.course_id != ec.course_id_student), "enrollments.course_id = students.course_id", ec)
    paid = p[p.payment_status == "Success"].groupby("enrollment_id").amount.sum()
    ep = e.assign(collected=e.enrollment_id.map(paid).fillna(0))
    add("Consistency", "payments", "collected ≤ enrollment price", len(ep), ep.collected > ep.price * 1.01, "sum(successful payments) ≤ price", ep)
    first = e[~e.is_renewal.astype(bool)].drop_duplicates("student_id").set_index("student_id").enrollment_date
    sd = s.dropna(subset=["enrollment_date"])
    mism = sd.student_id.map(first)
    add("Consistency", "students", "enrollment date matches enrollments", len(sd), mism.notna() & (mism != sd.enrollment_date), "students.enrollment_date = first enrollment date", sd)
    return out


def freshness(t: dict[str, pd.DataFrame], today: pd.Timestamp | None = None) -> pd.DataFrame:
    today = today or pd.Timestamp.today().normalize()
    rows = []
    for tab, cols in DATE_COLS.items():
        latest = t[tab][cols[0]].max()
        lag = (today - latest.normalize()).days if pd.notna(latest) else None
        rows.append({"table": tab, "rows": len(t[tab]), "latest_record": latest, "days_since_latest": lag,
                     "status": "Demo snapshot" if latest <= END + pd.Timedelta(days=1) else "Live"})
    for tab in ("courses", "instructors"):
        rows.append({"table": tab, "rows": len(t[tab]), "latest_record": pd.NaT, "days_since_latest": None, "status": "Reference data"})
    return pd.DataFrame(rows)


def score(checks: list[GCheck], fresh_lag_days: int | None = None) -> tuple[float, pd.DataFrame]:
    rows = []
    for dim, w in WEIGHTS.items():
        cs = [c for c in checks if c.dimension == dim]
        if dim == "Freshness":
            s = 100.0 if fresh_lag_days is None else max(0.0, 100 - max(fresh_lag_days - 1, 0) * 2)
            rows.append({"dimension": dim, "weight": w, "checks": 1, "score": round(s, 1),
                         "method": "100 − 2 points per day of lag beyond 1 day (demo snapshot exempt if lag unknown)"})
            continue
        pts = [{"PASS": 100, "WARN": 70, "FAIL": 0}[c.status] for c in cs]
        s = sum(pts) / len(pts) if pts else 100.0
        rows.append({"dimension": dim, "weight": w, "checks": len(cs), "score": round(s, 1),
                     "failing_rows": sum(c.failing for c in cs),
                     "method": "mean over checks of PASS=100, WARN=70 (<1% rows fail), FAIL=0 (≥1% rows fail)"})
    d = pd.DataFrame(rows)
    return round(float((d.score * d.weight).sum() / d.weight.sum()), 1), d


# ---------------- previewable cleaning actions (never modify the source) ----------------
def preview_action(t: dict[str, pd.DataFrame], action: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Returns (table name, affected rows, cleaned copy)."""
    if action == "Drop duplicate enrollment IDs (keep first)":
        e = t["enrollments"]
        mask = e.enrollment_id.duplicated(keep="first")
        return "enrollments", e[mask], e[~mask]
    if action == "Remove activity rows with unknown student_id":
        a = t["learning_activity"]
        mask = ~a.student_id.isin(t["students"].student_id)
        return "learning_activity", a[mask], a[~mask]
    if action == "Cap assessment scores at max_score":
        a = t["assessments"]
        mask = a.score > a.max_score
        fixed = a.copy()
        fixed.loc[mask, "score"] = fixed.loc[mask, "max_score"]
        return "assessments", a[mask], fixed
    if action == "Exclude negative payment amounts":
        p = t["payments"]
        mask = p.amount < 0
        return "payments", p[mask], p[~mask]
    raise ValueError(action)


ACTIONS = ["Drop duplicate enrollment IDs (keep first)", "Remove activity rows with unknown student_id",
           "Cap assessment scores at max_score", "Exclude negative payment amounts"]


def schema_report(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in t.items():
        for c in df.columns:
            rows.append({"table": name, "column": c, "dtype": str(df[c].dtype), "null_pct": round(df[c].isna().mean() * 100, 2),
                         "distinct": int(df[c].nunique()), "pk": PKS.get(name) == c,
                         "fk": next((f"{r}.{rc}" for tb, col, r, rc in FKS if tb == name and col == c), "")})
    return pd.DataFrame(rows)


def has_issues(checks: list[GCheck]) -> bool:
    return any(c.failing for c in checks)

