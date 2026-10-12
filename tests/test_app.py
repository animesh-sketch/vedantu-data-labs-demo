from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Overview", "Business Intelligence", "Student Lifecycle", "Engagement", "Academic Performance", "Retention & Cohorts",
         "Student Success", "Revenue & Finance", "Marketing & Growth", "Courses & Instructors", "Support Analytics", "AI Analyst",
         "Forecasting", "Data Explorer", "Data Quality", "Data Sources", "Metric Dictionary", "Demo Data Management"]


@pytest.fixture(scope="module")
def at():
    return AppTest.from_file(APP, default_timeout=240).run()


def _ok(at):
    assert not at.exception, [e.message for e in at.exception]
    assert not [e.value for e in at.error if "Something went wrong" in e.value]


def test_demo_mode_loads_by_default(at):
    _ok(at)
    assert any("DEMO MODE" in m.value for m in at.markdown)
    assert any("Ranked business insights" in s.value for s in at.subheader)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(at, page):
    at.session_state["page"] = page
    at.run()
    _ok(at)


def test_global_filter_changes_kpis(at):
    at.session_state["page"] = "Overview"
    at.run()
    before = [m.value for m in at.markdown if "class='kpi'" in m.value][:1]
    at.sidebar.multiselect(key="f_channel").set_value(["Referral"]).run()
    _ok(at)
    after = [m.value for m in at.markdown if "class='kpi'" in m.value][:1]
    assert before != after
    at.sidebar.multiselect(key="f_channel").set_value([]).run()


def test_empty_filter_shows_empty_state(at):
    at.sidebar.multiselect(key="f_status").set_value(["Lead - not qualified"]).run()
    _ok(at)
    assert any("No enrolled learners" in i.value for i in at.info)
    at.sidebar.multiselect(key="f_status").set_value([]).run()


def test_edtech_ai_analyst_answers(at):
    at.session_state["page"] = "AI Analyst"
    at.run()
    at.chat_input[0].set_value("Which topics have the lowest student accuracy?").run()
    _ok(at)
    assert any("Executive summary" in m.value and "Lowest accuracy" in m.value for m in at.markdown)


def test_custom_dataset_flow_still_works():
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.session_state["page"] = "Data Sources"
    at.run()
    at.selectbox[0].set_value("Sales / CRM leads").run()
    next(b for b in at.button if "Analyse" in b.label).click().run()
    _ok(at)
    assert at.session_state["workspace"] == "custom"
    for page in ("Overview", "AI Analyst", "Data Explorer", "Data Quality"):
        at.session_state["page"] = page
        at.run()
        _ok(at)
    at.session_state["page"] = "AI Analyst"
    at.run()
    at.chat_input[0].set_value("Conversion rate by lead source").run()
    _ok(at)
    assert any("Referral" in m.value for m in at.markdown)
    assert any("avg(CASE" in c.value for c in at.code)
    at.session_state["page"] = "Revenue & Finance"
    at.run()
    assert any("needs the EdTech data model" in i.value for i in at.info)
