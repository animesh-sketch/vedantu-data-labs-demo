# Vedantu Data Labs - Demo

An **EdTech Data Intelligence & Insights Platform** for academic leaders, business heads, product, finance, operations and student-success teams. Every module answers five questions: **what happened, why, who is affected, what happens next, and what to do**. Every number comes from one shared metric layer.

It opens in **Demo Mode** with 13 interconnected synthetic EdTech tables (online and offline learners) (fictional learners, 12 months, seed 2026), so no API key, database or upload is needed. Runs fully offline. An LLM is optional and never produces numbers.

## Quick start (Windows PowerShell)

```powershell
cd $HOME\vedantu-data-labs-demo
py -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python -m vdl.edtech.generate         # optional: pre-generate demo data (~15 s; otherwise happens on first launch)
streamlit run app.py                  # opens http://localhost:8501
```

Run the tests (about 4 minutes, 88 tests):

```powershell
python -m pytest -q
```

## Modules

| Section | Page | What it answers |
|---|---|---|
| Executive | **Online vs Offline** | Vedantu-style LIVE online classes vs learning-centre classrooms. Side-by-side KPIs from the shared metric layer, each mode vs its own previous period, how learners in each mode learn (app use, classes, timing, tickets), centre scorecard (seat utilisation, attendance change, revenue per seat), centre attendance trend, channel mix and ranked mode-specific insights. Mode gaps are labelled associations (mode is self-selected) |
| | **Overview** | KPI strip with RAG status and previous-period comparison, enrollment and revenue trend, funnel, weekly actives, cohort retention heatmap, course heatmap, revenue contribution, academic outcomes, at-risk summary, ranked insights |
| | **Business Intelligence** | All 31 KPIs grouped by area, with definition, period, baseline and calculation. Area health, top 5 positive trends and top 5 emerging risks |
| Students & Academics | **Student Lifecycle** | 9-stage strict funnel, drop-off table, time between stages, conversion by channel/counsellor/course, cohort funnel, cancellations, first-class-timing effect |
| | **Engagement** | DAU/WAU/MAU and stickiness, day × hour heatmap, engagement calendar, rule-based segments, watch time vs score, weekly retention, declining learners |
| | **Academic Performance** | Score distribution, pass rate trend, topic mastery matrix, topic difficulty, attendance vs score, before/after improvement, cohort and batch comparisons, intervention list |
| | **Retention & Cohorts** | Cohort heatmap, retention curves, *confirmed churn vs inferred inactivity*, churn by channel/course/cohort, renewals, reactivations, time to inactivity |
| | **Student Success** | Explainable risk flags (7 indicators, points, levels), recommended interventions, editable follow-up status, **time-based backtest** with precision/recall |
| Business Operations | **Revenue & Finance** | Bookings, collections, refunds, net revenue, waterfall, payment status, discount effectiveness, unit economics. Recognised revenue is shown as a labelled estimate; MRR and margin are marked *not applicable / not available* rather than invented |
| | **Marketing & Growth** | CPL, CAC, ROAS to date, conversion, refund and day-60–90 retention by channel and campaign, channel quality vs conversion, with a clear correlation-vs-causation warning |
| | **Courses & Instructors** | Scorecard, growth × quality quadrant, KPI heatmap. Instructor comparison uses score uplift vs the same course and topic, with a 95% interval (no raw-score ranking) |
| | **Support Analytics** | Volume, first-response and resolution times, SLA by priority, Pareto of reasons, reopen/repeat-contact rates, backlog, CSAT vs resolution time, support-to-engagement association |
| Intelligence | **Chat with Data** | Chat about any of the 12 tables. Business questions get structured answers (summary, findings, hypotheses, actions, confidence, scope). *Measure by column* questions work on any table offline (e.g. "average csat_score by priority"). With an API key, Claude answers free-form questions by running read-only SQL across all tables, and every query and result is shown. Also reachable from the sidebar via **💬 Ask your data anything** |
| | **Forecasting** | Enrollments, revenue, active learners, completions, tickets. Each uses an 8-week holdout backtest to pick a model (damped trend / naive / moving average), with 80/95% intervals, scenarios and assumptions. Short history gives a sufficiency warning instead of a forecast |
| | **Data Explorer** | Browse all 12 tables and a derived per-learner table. Read-only SQL across all tables, joins included |
| | **Data Quality** | 80+ checks (completeness, uniqueness, validity, referential integrity, cross-table consistency), weighted score, freshness, lineage, previewable cleaning actions with download, and a toggle to run on the test fixture with injected issues |
| Administration | **Data Sources** | Demo data, single CSV/Excel upload (the original generic analyser), upload of your own 12 EdTech tables with schema validation, CRM connectors |
| | **Metric Dictionary** | Single source of truth: every metric, segment rule, risk indicator and churn definition |
| | **Demo Data Management** | Load, reset or regenerate demo data with any seed |

**Global filters** (date range, course, subject, enrollment cohort, channel, student status, learning mode, learning centre) apply to every page, insight and AI answer. The default period is the last 90 days, compared with the previous 90.

## Definitions that matter

- **Active learner:** ≥1 learning activity or attended live class in the period.
- **Confirmed churn:** cancelled enrollment. **Inferred inactivity:** open enrollment with no activity for 30+ days. This is a warning, not churn, and some learners reactivate.
- **Retention:** active in the previous equal-length period and also in this one.
- **Completion rate:** completed / first enrollments whose scheduled end date falls in the period.
- **Net revenue:** collections − refunds (cash basis).
- **Comparisons:** always against the previous equal-length period. If that period is before the data starts, no comparison is shown.

The full list is in **Metric Dictionary**, or in `vdl/edtech/metrics.py`.

## Synthetic demo data

13 tables: `students` (leads incl. funnel fields, `learning_mode`, `centre_id`), `centres` (8 synthetic learning centres with seat capacity), `enrollments`, `courses`, `learning_activity`, `attendance`, `assessments`, `payments`, `support_tickets`, `marketing_spend`, `instructors`, `classes` and `student_surveys`. Together that is about 12k leads, 2.7k enrolled learners (~27% at centres), 143k activity events, 90k attendance rows and 19k assessments, with consistent keys. All identities are fictional.

Built-in stories that the analytics surface:
- Seasonality.
- Social Ads conversion collapsing from Aug 2026.
- Referral/Organic learners retaining better.
- Late first class → lower later engagement.
- Slow tickets → weaker engagement.
- Topic difficulty.
- Improving and declining learners.
- Failed payments, retries and refunds.
- Online vs offline: centre learners pay a centre fee (×1.3), convert better after in-person counselling, attend more, use the app less and go inactive less often. Online reaches every city and has more tech-friction tickets. Patna centre attendance falls from Jul 2026; Hyderabad centre runs at ~97% of seats. Walk-ins and school tie-ups feed centres.

The default data passes every quality check. Data-quality issues are injected only by `make_dirty()`, a test fixture.

## Architecture

```
app.py                    Streamlit shell: grouped navigation, global filters, demo/custom workspaces
vdl/edtech/generate.py    13-table deterministic synthetic data (online + offline centres) (+ make_dirty test fixture)
vdl/edtech/model.py       Filters + filtered Model shared by every page
vdl/edtech/metrics.py     single source of truth for 31 KPIs (definition, formula, baseline, RAG)
vdl/edtech/analytics.py   lifecycle, engagement, academics, retention, revenue, courses, marketing, support, risk
vdl/edtech/forecast.py    backtested forecasting with intervals and scenarios
vdl/edtech/modes.py       online vs offline comparison, centre scorecard, mode-specific insights
vdl/edtech/insights.py    ranked insights (impact = weight × change × reach)
vdl/edtech/analyst.py     EdTech AI analyst (intents + NL→SQL fallback over a per-learner table)
vdl/edtech/governance.py  quality checks, DQ score, lineage, freshness, cleaning previews
vdl/ui/                   components (KPI cards, decision brief, charts), EdTech pages, generic pages
vdl/profiling.py, quality.py, dashboard.py, insights.py, copilot.py   generic single-file analyser
vdl/sql_guard.py          read-only SQL validation + sandboxed in-memory DuckDB (single or multi-table)
vdl/connectors.py         Connector interface + HubSpot / Salesforce / Postgres extension points
tests/                    core, EdTech (known-answer fixtures) and Streamlit AppTest UI tests
```

### Safety and trust

- **SQL guard:** only a single `SELECT`/`WITH` is allowed. Write/DDL keywords, `PRAGMA`/`ATTACH`/`COPY`, file readers, `getenv` and comments are all blocked. Queries run in a fresh in-memory DuckDB with external access disabled, capped at 10k rows.
- **No invented metrics:** KPIs, insights and AI answers all call `metrics.py`/`analytics.py`. Unanswerable questions are refused. Missing data (spend, costs, history) is reported as *not available*.
- **No code execution:** user input is never executed as Python.
- **Privacy:** synthetic IDs only. No sensitive attributes are used for risk, and personal data is not exposed.
- **Optional Claude mode** (`ANTHROPIC_API_KEY` in `.env`, or in Streamlit Cloud → *Settings → Secrets* as `ANTHROPIC_API_KEY = "..."`): Claude sees only the schema and the rows its own queries return. Each query passes the same read-only guard. The prompt forbids numbers that are not in query results, and the SQL is shown under each answer. If the API fails, the offline analyst answers instead.

### CRM integration path

1. Implement `fetch()` in `HubSpotConnector` / `SalesforceConnector` with OAuth 2.0 (read-only scopes).
2. Store tokens server-side, never in the browser or repo.
3. Map records to the 12-table schema. They then pass schema validation → quality checks → metrics → SQL guard.
4. Add audit logging.

## Limitations (honest)

- All demo data is synthetic. Its patterns are designed, so findings demonstrate the method, not real Vedantu behaviour.
- Without an API key, the chat routes questions with rules: business intents, then NL→SQL over the best-matching table, otherwise it says it cannot answer. Free-form questions (joins, multi-step) need Claude mode.
- Forecasts have no yearly seasonality, because 12 months cannot estimate it. Backtest errors are shown.
- Risk flags are transparent rules validated by a time-based backtest. They are not a trained model.
- Channel, support and engagement relationships are associations, not causal effects.
- CRM connectors are documented extension points, not live integrations.

See **DEMO.md** for demo scenarios and an interview script.
