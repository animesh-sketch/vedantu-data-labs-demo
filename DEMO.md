# Demo scenarios

## 1. Sales / CRM: "Where should we put counselling effort?"
1. Choose **Sales / CRM leads** and click **Analyse**.
2. **Overview:** read the KPIs (pipeline value, Won rate). Recommendation: *'Won' rate ranges from ~11% to ~41% across lead_source*. Open the evidence table.
3. **AI Analyst:**
   - `Conversion rate by lead source` (Referral leads)
   - `what about Delhi?` (a follow-up that keeps the grouping and adds a filter)
   - `conversion by counsellor`
   - `Which city has the highest total deal value?`
4. Open **Evidence · SQL** to show that the numbers come from a visible query.
5. **Data Quality:** 8 duplicate leads and 3 deal values of ₹9.5L (anomalies), each flagged with its evidence. Nothing is silently removed.

## 2. Education: "What drives learning outcomes?"
1. Choose **Education – student learning** and click **Analyse**.
2. **Data Quality → Invalid types:** 6 test scores entered as `"absent"`. They are shown as examples, and the affected numbers are excluded from averages.
3. **AI Analyst:**
   - `average test score by learning mode` (1:1 Tutoring is highest)
   - `what about Physics?`
   - `correlation between attendance pct and test score` (~0.55, labelled "not causation")
   - `Monthly trend of enrolments`
4. Point out that the recommendation's *explanation* is labelled a hypothesis, and the *action* proposes a pilot rather than a conclusion.

## 3. Support: "Why are customers unhappy?"
1. Choose **Customer support tickets** and click **Analyse**.
2. **Overview:** Resolved rate by agent, and CSAT gap between Payment and Refund.
3. **AI Analyst:** `average resolution hours by category`, then `only Email`, then `average csat by channel`.
4. **Data Quality:** negative resolution hours are flagged as anomalies. Missing hours are explained (open tickets have none).
5. **Data Explorer → SQL:** run `DELETE FROM data` to show it is blocked, then a valid `SELECT` to show it works.
6. **Connections:** HubSpot/Salesforce show *Not configured*. Explain the OAuth design.

---

# 5-minute interview script

**0:00 – The problem (30s)**
"Business teams wait days for analysis, and when an AI answers, they can't tell if the number is real. I built *Vedantu Data Labs* to show what an AI-native analytics function should feel like: two clicks to insight, and every number traceable."

**0:30 – Two clicks (60s)** *(Sales dataset → Analyse)*
"The app profiles the schema itself: dates, metrics, segments, outcomes. It then picks the KPIs and charts. It knows to *sum* money but *average* scores and hours. That's a small detail, but it's where most auto-dashboards produce wrong numbers."

**1:30 – Trustworthy recommendations (60s)**
"Every recommendation has three labelled parts. The *verified finding* is computed, with its formula and an evidence table. The *possible explanation* is a hypothesis. The *suggested action* is next to test. For example, Referral converts at about 41% vs 11% for Paid Social. That's verified. *Why* it happens is a hypothesis. So the action is to reallocate budget and test, not to declare victory."

**2:30 – AI Analyst (75s)**
"Ask in plain English. It maps the question to an intent, builds SQL, runs it in a sandboxed read-only DuckDB, and narrates only what came back. Follow-ups keep context. Expand the evidence panel to see the SQL, assumptions and limitations. If it can't answer, it says so. It doesn't hallucinate. An LLM is optional, and even then it only *proposes SQL*, which passes the same guard."

**3:45 – Data quality and safety (45s)**
"Analytics is only as good as the data. Here I flag duplicates, invalid types and anomalies with evidence, and roll them into a trust score. On safety: the SQL guard blocks writes, file and network access. Credentials live in `.env`, and connectors never pretend to be live."

**4:30 – How I'd scale this at Vedantu (30s)**
"Next steps: OAuth connectors to the CRM and warehouse, a semantic layer of governed metric definitions (what counts as 'conversion'), an evaluation suite of question → expected-SQL pairs, and usage telemetry, so the analytics team spends time on decisions, not on pulling numbers."
