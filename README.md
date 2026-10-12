# Vedantu Data Labs - Demo

An AI-native analytics app. You upload or pick a dataset and click **Analyse**. You then get dashboards, evidence-backed recommendations, data-quality checks and plain-English answers. Every number is computed from the data.

Runs fully offline. An LLM is optional and never produces numbers.

## Quick start (Windows PowerShell)

```powershell
cd $HOME\vedantu-data-labs-demo
py -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python -m vdl.datasets                # (re)generate the 3 sample datasets
streamlit run app.py                  # opens http://localhost:8501
```

Run the tests:

```powershell
python -m pytest -q
```

## What's inside

| Section | What it does |
|---|---|
| **Overview** | Auto-selected KPI cards, monthly trend, segment breakdowns, distribution and outcome share. Below them, **business recommendations**, each split into *Verified finding* (with its calculation and an evidence table), *Possible explanation* (a hypothesis) and *Suggested action*. |
| **AI Analyst** | Plain-English Q&A. It handles totals, averages, counts, median, min/max, conversion rates, group-by, top-N, value filters (e.g. "Delhi"), monthly/weekly trends, correlation and follow-ups ("what about Mumbai?"). Each answer shows the SQL, assumptions and limitations. If it can't map a question, it says so and suggests alternatives instead of guessing. |
| **Data Explorer** | Filterable table plus a **read-only SQL workbench** (table `data`). |
| **Data Quality** | Checks for missing values, duplicates, invalid types and anomalies (IQR fences, unexpected negatives, future dates). Each shows PASS/WARN/FAIL with evidence and the formula used, and feeds a trust score. |
| **Connections** | Connector registry (file, HubSpot, Salesforce, Postgres). Status comes only from real configuration. Nothing is ever shown as "connected" when it isn't. |

## Architecture

```
app.py                 Streamlit UI (5 sections)
vdl/profiling.py       column-name cleaning, type detection (datetime/numeric/categorical/id/text), coercion
vdl/quality.py         data-quality checks -> Check(status, detail, calculation, evidence)
vdl/dashboard.py       schema-driven KPIs and Plotly charts (sums money/counts; averages scores/hours)
vdl/insights.py        trends, concentration risk, segment gaps, outcome-rate gaps, quality risks
vdl/copilot.py         NL -> intent -> SQL -> DuckDB -> narrated answer (+ optional LLM SQL proposal)
vdl/sql_guard.py       read-only SQL validation + sandboxed in-memory DuckDB execution
vdl/connectors.py      Connector interface + HubSpot / Salesforce / Postgres extension points
vdl/datasets.py        seeded synthetic datasets with deliberate quality issues
tests/                 unit tests (core) + Streamlit AppTest UI tests
```

### Safety and trust

- **SQL guard:** only a single `SELECT`/`WITH` is allowed. Write/DDL keywords, `PRAGMA`/`ATTACH`/`COPY`, file readers (`read_csv`…), `getenv` and comments are all blocked. Queries run in a fresh in-memory DuckDB with `enable_external_access=false` and locked configuration, capped at 10k rows.
- **No invented metrics:** answers are narrated from query results. If a question can't be mapped, the copilot refuses rather than guessing.
- **No code execution:** user input is never `eval`'d or executed as Python.
- **Optional LLM** (`ANTHROPIC_API_KEY` + `pip install anthropic python-dotenv`): used only as a fallback to *propose SQL*, which then passes the same guard. Only the column schema is sent, not the data values.
- **Credentials:** read from `.env` (git-ignored) and never displayed.

### CRM integration path

1. Implement `fetch()` in `HubSpotConnector` / `SalesforceConnector`. Use the OAuth 2.0 authorization-code flow and request read-only scopes.
2. Store tokens server-side (a secret manager), never in the browser or the repo.
3. Run fetched records through the same pipeline: profiling → quality → SQL guard.
4. Add audit logging (who pulled what, and which queries ran).

## Limitations (honest)

- The NL parser is rule-based. Unusual phrasing may need a column name, or the optional LLM.
- Type detection is heuristic. Check **Overview → Detected schema**.
- Insights are descriptive. Explanations are labelled as hypotheses, and no causal testing is done.
- CRM connectors are documented extension points, not live integrations.

See **DEMO.md** for three demo scenarios and a 5-minute interview script.
