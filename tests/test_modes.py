import pandas as pd
import pytest

from vdl.edtech import analyst as X
from vdl.edtech import governance as G
from vdl.edtech import metrics as M
from vdl.edtech import modes as MD
from vdl.edtech.generate import CENTRES, OFFLINE_COURSES, ensure_modes, load_all
from vdl.edtech.model import Filters, Model

T = pd.Timestamp
WIN = Filters(T("2026-07-02"), T("2026-09-30"))


@pytest.fixture(scope="module")
def tables():
    return load_all()


@pytest.fixture(scope="module")
def m(tables):
    return Model(tables, WIN)


def test_offline_only_where_a_centre_and_programme_exist(tables):
    s = tables["students"]
    off = s[s.learning_mode == "Offline"]
    assert set(s.learning_mode) == {"Online", "Offline"} and len(off) > 500
    assert set(off.city) <= {c[2] for c in CENTRES} and set(off.course_id) <= OFFLINE_COURSES
    assert s.loc[s.learning_mode == "Online", "centre_id"].isna().all() and off.centre_id.notna().all()
    e = tables["enrollments"].merge(off[["student_id", "centre_id"]], on="student_id")
    assert (e.batch_id == e.course_id + "-" + e.centre_id).all()  # offline learners sit in their centre's batch


def test_mode_filters_partition_learners(tables, m):
    on, off = MD.mode_model(m, "Online"), MD.mode_model(m, "Offline")
    assert len(on.enrolled) + len(off.enrolled) == len(m.enrolled)
    assert M.compute(on, "new_enrollments") + M.compute(off, "new_enrollments") == M.compute(m, "new_enrollments")


def test_comparison_uses_metric_layer(m):
    comp, kpis = MD.comparison(m, ["attendance_rate", "new_enrollments"])
    assert kpis["Offline"]["attendance_rate"].value == M.kpi(MD.mode_model(m, "Offline"), "attendance_rate").value
    row = comp.set_index("key").loc["attendance_rate"]
    assert row["leads on"] == "Offline" and row["gap (offline − online)"].endswith("pp")


def test_centre_flags_find_embedded_stories(m):
    c = MD.centre_scorecard(m).set_index("centre_name")
    assert c.loc["Patna Centre", "attendance_change_pp"] <= -15
    assert c.loc["Hyderabad Centre", "utilisation_pct"] >= 90
    titles = [i.title for i in MD.mode_insights(m, {}, c.reset_index(), pd.DataFrame())]
    assert any(t.startswith("Patna Centre: classroom attendance fell") for t in titles)
    assert any(t.startswith("Hyderabad Centre is at") for t in titles)


def test_online_only_filter_has_no_centres(tables):
    mo = Model(tables, Filters(T("2026-07-02"), T("2026-09-30"), modes=("Online",)))
    assert MD.modes_in(mo) == ["Online"] and MD.centre_scorecard(mo).empty


def test_upload_without_mode_columns_is_online_only(tables):
    t = {k: v.drop(columns=[c for c in ("learning_mode", "centre_id") if c in v.columns]) for k, v in tables.items() if k != "centres"}
    mm = Model(t, WIN)
    assert set(mm.students.learning_mode) == {"Online"} and MD.centre_scorecard(mm).empty
    assert not any(c.failing for c in G.run_checks(ensure_modes(t)) if c.dimension == "Referential integrity")


def test_chat_answers_online_vs_offline(m):
    a = X.ask(m, "How do online and offline learners differ?")
    assert a.intent == "mode_compare" and "attendance" in a.summary and {"Online", "Offline"} <= set(a.table.columns)
