from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _analysed(dataset="Sales / CRM leads"):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox[0].set_value(dataset).run()
    next(b for b in at.sidebar.button if "Analyse" in b.label).click().run()
    assert not at.exception
    return at


def test_overview_renders_for_every_sample():
    for ds in ("Sales / CRM leads", "Education – student learning", "Customer support tickets"):
        at = _analysed(ds)
        assert any("Business recommendations" in s.value for s in at.subheader)


def test_pages_render():
    at = _analysed()
    nav = at.sidebar.radio[1]
    for page in ("AI Analyst", "Data Explorer", "Data Quality", "Connections"):
        nav.set_value(page).run()
        assert not at.exception, page


def test_chat_question_answers_with_sql():
    at = _analysed()
    at.sidebar.radio[1].set_value("AI Analyst").run()
    at.chat_input[0].set_value("Conversion rate by lead source").run()
    assert not at.exception
    assert any("Referral" in m.value for m in at.markdown)
    assert any("avg(CASE" in c.value for c in at.code)
