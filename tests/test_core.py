import pandas as pd
import pytest

from vdl import connectors, copilot, dashboard, insights, quality, sql_guard
from vdl.datasets import education, sales_crm, support
from vdl.profiling import clean_columns, prepare


@pytest.fixture(scope="module")
def sales():
    raw = sales_crm()
    df, prof = prepare(raw)
    return clean_columns(raw), df, prof


def test_type_detection(sales):
    _, _, prof = sales
    assert prof.types["created_date"] == "datetime"
    assert prof.types["deal_value"] == "numeric"
    assert prof.types["city"] == "categorical"
    assert prof.types["lead_id"] == "id"


def test_quality_detects_injected_issues(sales):
    raw, df, prof = sales
    checks = {c.name: c for c in quality.run_all(raw, df, prof)}
    assert checks["Duplicate rows"].status != "PASS"
    assert checks["Missing values"].evidence.loc["city", "missing"] >= 25
    assert "deal_value" in set(checks["Potential anomalies"].evidence["column"])


def test_invalid_types_found():
    raw = education()
    df, prof = prepare(raw)
    check = quality.invalid_types(clean_columns(raw), df, prof)
    assert prof.types["test_score"] == "numeric"
    assert check.evidence.set_index("column").loc["test_score", "invalid_count"] == 6


@pytest.mark.parametrize("sql", [
    "DROP TABLE data", "SELECT 1; DELETE FROM data", "INSERT INTO data VALUES (1)",
    "SELECT * FROM read_csv_auto('secrets.csv')", "COPY data TO 'x.csv'", "SELECT * FROM 'C:/x.csv'",
    "SELECT * FROM data -- comment", "ATTACH 'x.db'", "PRAGMA version", "SELECT getenv('HOME')",
])
def test_sql_guard_blocks(sql):
    with pytest.raises(sql_guard.UnsafeSQLError):
        sql_guard.validate(sql)


def test_sql_guard_allows_reads(sales):
    _, df, _ = sales
    out = sql_guard.run("WITH t AS (SELECT city, sum(deal_value) v FROM data GROUP BY 1) SELECT * FROM t", df)
    assert len(out) > 0
    assert sql_guard.run("SELECT 'drop; table' AS s", df).iloc[0, 0] == "drop; table"


def test_copilot_total_matches_pandas(sales):
    _, df, prof = sales
    a = copilot.ask("What is the total deal value?", df, prof)
    assert a.ok and a.sql
    assert a.table.iloc[0, 0] == pytest.approx(round(df.deal_value.sum(), 2))


def test_copilot_group_top_and_filter(sales):
    _, df, prof = sales
    a = copilot.ask("Which city has the highest total deal value?", df, prof)
    expected = df.groupby("city").deal_value.sum().idxmax()
    assert a.table.iloc[0, 0] == expected and len(a.table) == 1
    b = copilot.ask("Average deal value for Referral by product", df, prof)
    exp = df[df.lead_source == "Referral"].groupby("product").deal_value.mean().round(2)
    assert b.intent.filters == {"lead_source": "Referral"}
    assert dict(zip(b.table["product"], b.table.iloc[:, 1])) == pytest.approx(exp.to_dict())


def test_copilot_followup_keeps_context(sales):
    _, df, prof = sales
    a = copilot.ask("Total deal value by product", df, prof)
    b = copilot.ask("what about Delhi?", df, prof, a.intent)
    assert b.intent.group == "product" and b.intent.filters == {"city": "Delhi"}
    exp = df[df.city == "Delhi"].groupby("product").deal_value.sum().max()
    assert b.table.iloc[0, 1] == pytest.approx(exp)


def test_copilot_count_trend_and_refusal(sales):
    _, df, prof = sales
    c = copilot.ask("How many leads by stage?", df, prof)
    assert c.table.iloc[:, 1].sum() == len(df)
    t = copilot.ask("Monthly trend of deal value", df, prof)
    assert "period" in t.table.columns and len(t.table) >= 12
    r = copilot.ask("What is the weather on Mars?", df, prof)
    assert not r.ok and r.sql is None


def test_dashboard_and_insights_all_samples():
    for raw in (sales_crm(), education(), support()):
        df, prof = prepare(raw)
        assert dashboard.kpis(df, prof)
        assert dashboard.charts(df, prof)
        ins = insights.generate(df, prof, quality.run_all(clean_columns(raw), df, prof))
        for i in ins:
            assert i.finding and i.calculation and i.explanation and i.action


def test_connectors_never_fake_connection(monkeypatch):
    for v in ("HUBSPOT_CLIENT_ID", "HUBSPOT_CLIENT_SECRET", "HUBSPOT_REDIRECT_URI"):
        monkeypatch.delenv(v, raising=False)
    hs = connectors.get("hubspot")
    assert hs.status() == connectors.NOT_CONFIGURED
    with pytest.raises(ConnectionError):
        hs.fetch("contacts")
    for v in ("HUBSPOT_CLIENT_ID", "HUBSPOT_CLIENT_SECRET", "HUBSPOT_REDIRECT_URI"):
        monkeypatch.setenv(v, "x")
    assert "not verified" in hs.status()
    with pytest.raises(NotImplementedError):
        hs.fetch("contacts")


def test_messy_upload_handled():
    raw = pd.DataFrame({"Amount (₹)": ["1,000", "2,500", "bad", None], "Order Date": ["2026-01-01", "2026-02-01", "x", None],
                        "Region": ["N", "S", "N", "S"]})
    df, prof = prepare(raw)
    assert prof.types["amount"] == "numeric" or prof.types["amount"] == "categorical"
    assert list(df.columns) == ["amount", "order_date", "region"]


def test_copilot_conversion_rate(sales):
    _, df, prof = sales
    a = copilot.ask("Conversion rate by lead source", df, prof)
    assert a.intent.agg == "rate" and a.intent.rate_col == "stage"
    exp = (df.assign(w=df.stage.eq("Won")).groupby("lead_source").w.mean() * 100).round(2)
    assert dict(zip(a.table["lead_source"], a.table.iloc[:, 1])) == pytest.approx(exp.to_dict())
    assert a.table.iloc[0, 0] == "Referral"
