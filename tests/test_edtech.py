import hashlib

import numpy as np
import pandas as pd
import pytest

from vdl import sql_guard
from vdl.edtech import analyst as X
from vdl.edtech import analytics as A
from vdl.edtech import forecast as F
from vdl.edtech import governance as G
from vdl.edtech import insights as I
from vdl.edtech import metrics as M
from vdl.edtech.generate import TABLES, generate, load_all, make_dirty
from vdl.edtech.model import Filters, Model

T = pd.Timestamp
WIN = Filters(T("2026-07-02"), T("2026-09-30"))


@pytest.fixture(scope="module")
def tables():
    return load_all()


@pytest.fixture(scope="module")
def m(tables):
    return Model(tables, WIN)


# ---------------- demo data ----------------
def test_generation_is_reproducible():
    a, b = generate(seed=7, scale=0.05), generate(seed=7, scale=0.05)
    for t in TABLES:
        assert hashlib.md5(pd.util.hash_pandas_object(a[t], index=False).values).hexdigest() == \
               hashlib.md5(pd.util.hash_pandas_object(b[t], index=False).values).hexdigest(), t
    c = generate(seed=8, scale=0.05)
    assert not a["students"].equals(c["students"])


def test_demo_has_enough_history_and_scale(tables):
    s = tables["students"]
    assert s.signup_date.min() <= T("2025-10-02") and s.signup_date.max() >= T("2026-09-28")
    assert s.enrollment_date.notna().sum() >= 2000
    assert len(tables["learning_activity"]) > 50_000 and len(tables["attendance"]) > 30_000


def test_relationships_are_valid(tables):
    for tab, col, ref, rcol in G.FKS:
        orphans = ~tables[tab][col].dropna().isin(tables[ref][rcol])
        assert not orphans.any(), f"{tab}.{col} → {ref}.{rcol}"
    for tab, pk in G.PKS.items():
        assert tables[tab][pk].is_unique, tab


def test_default_demo_is_clean_and_fixture_is_not(tables):
    clean = G.run_checks(tables)
    assert not G.has_issues(clean)
    assert G.score(clean)[0] == 100.0
    dirty = G.run_checks(make_dirty(tables))
    failing = {(c.table, c.check) for c in dirty if c.failing}
    assert ("assessments", "score within 0..max_score") in failing
    assert ("enrollments", "`enrollment_id` unique") in failing
    assert ("learning_activity", "`student_id` → students.student_id") in failing
    assert G.score(dirty)[0] < 100


def test_fictional_identities(tables):
    assert tables["instructors"].instructor_name.str.contains("(fictional)", regex=False).all()
    assert tables["students"].student_id.str.match(r"^S\d{5}$").all()
    assert not {"name", "email", "phone", "gender", "religion", "caste"} & set(tables["students"].columns)


# ---------------- tiny known fixture ----------------
def _fixture() -> dict[str, pd.DataFrame]:
    cols = {k: v.columns for k, v in generate(scale=0.02).items()}
    t = {k: pd.DataFrame(columns=c) for k, c in cols.items()}
    t["courses"] = pd.DataFrame([("C01", "Course A", "Mathematics", "Intermediate", 4, 10000)], columns=cols["courses"])
    t["students"] = pd.DataFrame({
        "student_id": ["S1", "S2", "S3", "S4"], "signup_date": [T("2026-01-01")] * 3 + [T("2026-01-05")],
        "city": "X", "acquisition_channel": ["Organic", "Organic", "Paid Search", "Paid Search"], "campaign": "c",
        "course_id": "C01", "counselor": "K", "lead_qualified": [True, True, True, False],
        "counseling_date": [T("2026-01-02"), T("2026-01-02"), T("2026-01-02"), pd.NaT],
        "enrollment_date": [T("2026-01-03"), T("2026-01-04"), pd.NaT, pd.NaT],
        "student_status": ["Completed", "Cancelled", "Lead - lost", "Lead - not qualified"]})
    t["enrollments"] = pd.DataFrame({
        "enrollment_id": ["E1", "E2"], "student_id": ["S1", "S2"], "course_id": "C01", "batch_id": "C01-B1",
        "enrollment_date": [T("2026-01-03"), T("2026-01-04")], "enrollment_status": ["Completed", "Cancelled"],
        "discount": [0.0, 1000.0], "price": [10000.0, 9000.0], "refund_amount": [0.0, 9000.0], "is_renewal": False,
        "cancellation_date": [pd.NaT, T("2026-01-08")], "completion_date": [T("2026-01-31"), pd.NaT]})
    t["payments"] = pd.DataFrame({
        "payment_id": ["P1", "P2", "P3", "P4"], "student_id": ["S1", "S2", "S2", "S1"], "enrollment_id": ["E1", "E2", "E2", "E1"],
        "payment_date": [T("2026-01-03"), T("2026-01-04"), T("2026-01-11"), T("2026-01-02")],
        "amount": [10000.0, 9000.0, 0.0, 10000.0], "payment_status": ["Success", "Success", "Refunded", "Failed"],
        "refund_amount": [0.0, 0.0, 9000.0, 0.0]})
    t["classes"] = pd.DataFrame({"class_id": ["K1", "K2"], "course_id": "C01", "batch_id": "C01-B1", "instructor_id": "I1",
                                 "class_date": [T("2026-01-06 18:00"), T("2026-01-13 18:00")], "scheduled_minutes": 60, "enrolled_capacity": 150})
    t["instructors"] = pd.DataFrame({"instructor_id": ["I1"], "instructor_name": ["T (fictional)"], "subject": ["Mathematics"], "experience_band": ["7+ yrs"]})
    t["attendance"] = pd.DataFrame({"attendance_id": ["A1", "A2", "A3"], "student_id": ["S1", "S1", "S2"], "course_id": "C01",
                                    "class_id": ["K1", "K2", "K1"], "class_date": [T("2026-01-06 18:00"), T("2026-01-13 18:00"), T("2026-01-06 18:00")],
                                    "attended": [True, False, True], "class_duration_minutes": [55.0, 0.0, 60.0]})
    t["learning_activity"] = pd.DataFrame({"activity_id": ["L1", "L2"], "student_id": ["S1", "S1"], "course_id": "C01",
                                           "activity_date": [T("2026-01-20 10:00"), T("2026-01-21 10:00")], "activity_type": "video_watch",
                                           "duration_minutes": 20.0, "content_id": "x"})
    t["assessments"] = pd.DataFrame({"assessment_id": ["X1", "X2"], "student_id": ["S1", "S2"], "course_id": "C01", "topic": "Algebra",
                                     "assessment_date": [T("2026-01-10"), T("2026-01-10")], "score": [30.0, 80.0], "max_score": [50, 100],
                                     "questions_attempted": [20, 20], "correct_answers": [12, 16]})
    return t


@pytest.fixture()
def fx():
    return Model(_fixture(), Filters(T("2026-01-01"), T("2026-01-31")))


def test_kpis_on_known_fixture(fx):
    assert M.compute(fx, "leads") == 4
    assert M.compute(fx, "new_enrollments") == 2
    assert M.compute(fx, "lead_to_enrol") == pytest.approx(50.0)
    assert M.compute(fx, "gross_bookings") == 19000
    assert M.compute(fx, "collections") == 19000          # failed payment excluded
    assert M.compute(fx, "refunds") == 9000
    assert M.compute(fx, "net_revenue") == 10000
    assert M.compute(fx, "refund_rate") == pytest.approx(50.0)
    assert M.compute(fx, "payment_failure_rate") == pytest.approx(100 / 3)
    assert M.compute(fx, "attendance_rate") == pytest.approx(200 / 3)
    assert M.compute(fx, "avg_score") == pytest.approx(70.0)  # (60% + 80%) / 2, uses max_score
    assert M.compute(fx, "pass_rate") == pytest.approx(100.0)
    assert M.compute(fx, "completion_rate") == pytest.approx(100.0)  # only E1 ends in Jan (E2 ends 1 Feb)
    assert M.compute(fx, "completion_rate", T("2026-01-01"), T("2026-02-28")) == pytest.approx(50.0)
    assert M.compute(fx, "active_learners") == 2
    assert M.compute(fx, "arps") == pytest.approx(5000.0)  # 10000 net / 2 paying students


def test_no_fabricated_baseline(fx):
    k = M.kpi(fx, "new_enrollments")  # previous window would be Dec 2025, before START? no: data_start is 2025-10-01
    m0 = Model(fx.tables, Filters(T("2025-10-01"), T("2025-12-31")))
    k0 = M.kpi(m0, "new_enrollments")
    assert k0.prev is None and k0.change is None and k0.rag == "grey" and "no comparison" in k0.baseline
    assert k.baseline.startswith("previous")


def test_funnel_denominators_on_fixture(fx):
    f = A.funnel(fx).set_index("stage")
    assert f.loc["Lead", "students"] == 4
    assert f.loc["Qualified lead", "students"] == 3
    assert f.loc["Counselling", "students"] == 3
    assert f.loc["Enrollment", "students"] == 2
    assert f.loc["First class", "students"] == 2
    assert f.loc["Enrollment", "pct_of_leads"] == 50.0
    assert f.loc["Enrollment", "step_conversion"] == pytest.approx(66.7, abs=0.1)
    assert (f.students.diff().dropna() <= 0).all()  # strictly non-increasing


def test_funnel_monotonic_and_consistent_on_demo(m):
    f = A.funnel(m)
    assert (f.students.diff().dropna() <= 0).all()
    assert f.students.iloc[0] == M.compute(m, "leads")
    assert f.set_index("stage").loc["Enrollment", "pct_of_leads"] == pytest.approx(M.compute(m, "lead_to_enrol"), abs=0.05)


def test_cohort_retention_consistent(m):
    mat = A.cohort_matrix(m)
    size = m.enrolled.groupby(m.enrolled.enrollment_date.dt.strftime("%Y-%m")).size()
    assert mat.cohort_size.equals(size.rename("cohort_size")) or (mat.cohort_size == size).all()
    vals = mat.drop(columns="cohort_size")
    assert vals.stack().dropna().between(0, 100).all()
    last = mat.index.max()
    assert vals.loc[last].iloc[1:].isna().all()  # future months are blank, not zero


def test_refunds_and_failed_payments_on_demo(m, tables):
    p = tables["payments"]
    assert (p.loc[p.payment_status == "Failed", "amount"] > 0).all()
    wf = A.waterfall(m).set_index("step").amount
    assert wf["Gross bookings"] == pytest.approx(wf["List-price value"] + wf["Discounts"])
    assert wf["Net cash"] == pytest.approx(wf["Collected"] + wf["Refunds"])
    assert wf["Collected"] == pytest.approx(wf["Gross bookings"] + wf["Cancelled before billing"] + wf["Not yet collected"])
    e = tables["enrollments"]
    assert (e.refund_amount <= e.price + 0.01).all()
    assert (e.loc[e.refund_amount > 0, "enrollment_status"] == "Cancelled").all()


def test_global_filters_affect_modules(tables, m):
    mf = Model(tables, Filters(WIN.start, WIN.end, channels=("Referral",)))
    assert set(mf.students.acquisition_channel) == {"Referral"}
    assert M.compute(mf, "new_enrollments") < M.compute(m, "new_enrollments")
    assert A.funnel(mf).students.iloc[0] < A.funnel(m).students.iloc[0]
    assert set(A.student_risk(mf).channel if "channel" in A.student_risk(mf) else A.segments(mf).channel) <= {"Referral"}
    assert A.revenue_by(mf, "course_name").net_revenue.sum() == pytest.approx(M.compute(mf, "net_revenue"))
    assert set(X.explorer_tables(mf)["enrollments"].student_id) <= set(mf.students.student_id)
    mc = Model(tables, Filters(WIN.start, WIN.end, courses=("NEET Biology",)))
    assert set(A.assessments_w(mc).course_name) == {"NEET Biology"}


def test_empty_filters_do_not_crash(tables):
    me = Model(tables, Filters(WIN.start, WIN.end, channels=("Nonexistent",)))
    assert me.empty()
    assert A.funnel(me).empty and A.student_risk(me).empty and A.cohort_matrix(me).empty
    assert M.compute(me, "net_revenue") == 0 and M.compute(me, "avg_score") is None
    assert I.generate(me) == []
    assert X.ask(me, "Which courses have the highest completion rates?").summary


def test_malformed_upload_is_rejected():
    from vdl.ui.generic_pages import validate_edtech_upload
    good = generate(scale=0.02)
    tables, problems = validate_edtech_upload({"students": good["students"].drop(columns=["acquisition_channel"])})
    assert tables is None and any("Missing table" in p for p in problems) and any("acquisition_channel" in p for p in problems)
    tables, problems = validate_edtech_upload({k: v.astype(str) for k, v in good.items()})
    assert tables is not None and pd.api.types.is_datetime64_any_dtype(tables["enrollments"].enrollment_date)


def test_forecast_handles_short_history(tables):
    short = Model(tables, Filters(T("2025-10-01"), T("2025-12-31")))
    fc = F.forecast(short, "enrollments")
    assert fc.forecast is None and "weeks" in fc.warning


def test_forecast_has_backtest_and_intervals(m):
    fc = F.forecast(m, "enrollments", horizon=8)
    f = fc.forecast
    assert len(f) == 8 and (f.lo95 <= f.lo80).all() and (f.lo80 <= f.baseline).all() and (f.baseline <= f.hi80).all()
    assert fc.backtest["chosen_model"] in fc.backtest["candidates"] and fc.assumptions


def test_segments_and_risk_are_explainable(m):
    seg = A.segments(m)
    assert set(seg.segment) <= {s for s, _ in A.SEGMENT_RULES}
    dormant = seg[seg.segment == "Dormant"]
    assert (dormant.active_days_28d == 0).all()
    r = A.student_risk(m)
    hi = r[r.risk_level == "High"]
    assert (hi.risk_score >= 5).all() and (hi.indicators.str.len() > 0).all()
    assert not {"city", "gender"} & set(r.columns)
    bt = A.risk_backtest(m)
    assert 0 <= bt["High"]["precision_pct"] <= 100 and 0 <= bt["High+Medium"]["recall_pct"] <= 100


def test_instructor_comparison_is_adjusted(m):
    im = A.instructor_matrix(m)
    solo = im[im.signal.str.startswith("Sole")]
    assert solo.score_vs_course_avg_pp.isna().all()
    assert "ci95_pp" in im.columns and "avg_score_pct" not in im.columns  # no raw-score ranking


def test_insights_are_grounded_in_metrics(m):
    ins = I.generate(m)
    assert ins and all(i.basis in ("Verified metric", "Association (not causal)", "Rules-based flag") for i in ins)
    k = M.kpi(m, "new_enrollments")
    ne = [i for i in ins if i.metric == "New enrollments"]
    if ne:
        assert ne[0].observed == M.fmt(k.value, k.unit) and ne[0].change == M.fmt_change(k)
    assert [i.impact for i in ins] == sorted([i.impact for i in ins], reverse=True)


def test_ai_answers_are_grounded(m):
    a = X.ask(m, "Why did enrollment decline this month?")
    mm = X._month_model(m)
    k = M.kpi(mm, "new_enrollments")
    assert M.fmt(k.value, "count") in a.summary and M.fmt(k.prev, "count") in a.summary
    assert a.sql is None and "no SQL" in a.computation  # never show fictional SQL
    b = X.ask(m, "Which topics have the lowest student accuracy?")
    assert b.table.iloc[0].topic in b.summary
    c = X.ask(m, "average avg score pct by segment")
    assert c.sql and c.table is not None  # free-form fallback actually ran SQL
    for ans in (a, b, c):
        assert ans.scope and ans.confidence and ans.limitations


def test_ai_followup_applies_filter(m):
    a = X.ask(m, "Which courses have the highest completion rates?")
    b = X.ask(m, "what about Referral?", a)
    assert b.extra_filter == {"channels": ("Referral",)} and "Follow-up applied" in b.summary


def test_all_eight_example_questions_answer(m):
    for q in X.EXAMPLES:
        a = X.ask(m, q)
        assert a.intent is not None and a.summary, q


def test_explorer_is_read_only(m):
    tabs = X.explorer_tables(m)
    assert len(sql_guard.run_multi("SELECT count(*) n FROM enrollments e JOIN courses USING (course_id)", tabs)) == 1
    for bad in ("DELETE FROM students", "DROP TABLE courses", "UPDATE payments SET amount = 0",
                "SELECT * FROM read_csv('data/edtech/students.csv')", "CREATE TABLE x AS SELECT 1"):
        with pytest.raises(sql_guard.UnsafeSQLError):
            sql_guard.run_multi(bad, tabs)
    assert len(tabs["students"]) == len(m.students)  # source frames untouched


def test_metric_dictionary_complete():
    d = M.dictionary()
    assert d.definition.str.len().min() > 10 and d.calculation.str.len().min() > 3 and len(d) == len(M.METRICS)


def test_previewable_cleaning_never_mutates(tables):
    dirty = make_dirty(tables)
    before = len(dirty["enrollments"])
    tab, affected, cleaned = G.preview_action(dirty, "Drop duplicate enrollment IDs (keep first)")
    assert len(affected) == 4 and len(cleaned) == before - 4 and len(dirty["enrollments"]) == before
    _, aff2, fixed = G.preview_action(dirty, "Cap assessment scores at max_score")
    assert (fixed.score <= fixed.max_score).all() and (dirty["assessments"].score > dirty["assessments"].max_score).sum() == len(aff2)


_ = np
