import sys
import types
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from vdl.edtech import analyst as X
from vdl.edtech import chat as CH
from vdl.edtech.generate import load_all
from vdl.edtech.model import Filters, Model

T = pd.Timestamp


@pytest.fixture(scope="module")
def m():
    return Model(load_all(), Filters(T("2026-07-02"), T("2026-09-30")))


@pytest.mark.parametrize("q, table, text", [
    ("average csat_score by priority", "support_tickets", "Average csat_score by **priority**"),
    ("how many tickets by issue_category", "support_tickets", "by **issue_category**"),
    ("total spend by channel", "marketing_spend", "Total spend by **channel**"),
    ("average satisfaction score", "student_surveys", "satisfaction_score"),
])
def test_offline_chat_answers_any_table(m, q, table, text):
    a = X.ask(m, q)
    assert f"`{table}`" in a.computation and text in a.summary and a.sql


def test_offline_chat_matches_sql_ground_truth(m):
    a = X.ask(m, "total spend by channel")
    truth = m.marketing.groupby("channel").spend.sum()
    top = a.table.iloc[0]
    assert top.iloc[0] == truth.idxmax() and round(top.iloc[1]) == round(truth.max())


def test_offline_chat_refuses_unanswerable(m):
    a = X.ask(m, "what is the weather in Paris")
    assert "won't guess" in a.summary and a.sql is None


def test_no_key_uses_offline(m, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not CH.llm_available()
    assert CH.answer(m, "total spend by channel", []).intent != "llm"


def _fake_anthropic(monkeypatch, sqls):
    """A stand-in client that asks for each SQL in turn, then writes a final answer."""
    calls = {"n": 0, "seen": []}

    class Messages:
        def create(self, **kw):
            calls["seen"].append(kw["messages"][-1]["content"])
            i = calls["n"]
            calls["n"] += 1
            if i < len(sqls):
                return NS(stop_reason="tool_use", content=[NS(type="tool_use", id=f"t{i}", input={"sql": sqls[i]})])
            return NS(stop_reason="end_turn", content=[NS(type="text", text="Paid Search has the highest spend.")])

    mod = types.ModuleType("anthropic")
    mod.Anthropic = lambda: NS(messages=Messages())
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    return calls


def test_llm_runs_guarded_sql_across_tables(m, monkeypatch):
    sql = ("SELECT c.subject, count(*) AS n FROM support_tickets t JOIN courses c USING (course_id) "
           "GROUP BY 1 ORDER BY 2 DESC")
    calls = _fake_anthropic(monkeypatch, ["DELETE FROM students", sql])
    a = CH.answer(m, "tickets by subject?", [])
    assert a.intent == "llm" and a.summary.startswith("Paid Search")
    assert "ERROR" in calls["seen"][1][0]["content"] and calls["seen"][1][0]["is_error"]  # write blocked, model told
    assert a.sql == sql and a.table.n.sum() == len(m.tickets)  # only the successful query is reported


def test_llm_failure_falls_back_offline(m, monkeypatch):
    _fake_anthropic(monkeypatch, [])
    monkeypatch.setattr(CH, "llm_answer", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    a = CH.answer(m, "total spend by channel", [])
    assert a.intent != "llm" and "LLM unavailable" in a.limitations[0]
