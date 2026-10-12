# Demo scenarios (EdTech demo mode)

Numbers below are from the default demo data (seed 2026, default period 3 Jul – 30 Sep 2026). If you regenerate with another seed, they change.

## 1. Leadership: "Why are enrollments down, and what do we do on Monday?"
1. **Overview:** New enrollments **605, −38%** vs the previous 90 days (red); net revenue **₹1.45 Cr, −13.5%**. Point out that each KPI card shows its baseline, and hovering shows its definition.
2. Scroll to **Ranked business insights**. The #1 risk is the enrollment decline, and the largest contributor is **Social Ads**. Open *Evidence* to see the driver table.
3. **AI Analyst:** *"Why did enrollment decline this month?"* The answer separates lead volume from conversion, lists the channels that drove the change, labels explanations as hypotheses, and shows scope and sample size.
4. Follow up with *"What should leadership prioritize next week?"*, then *"what about Social Ads?"* (the filter is carried over).
5. **Marketing & Growth:** Social Ads has the highest CAC and lowest conversion. Note the banner saying this is correlation, not incremental lift.

## 2. Academic & student-success head: "Who needs help, and where is learning weak?"
1. **Academic Performance:** the topic mastery matrix shows the lowest-accuracy topic is **Cell Biology (~38%)** vs a median of ~53%. The before/after chart shows the share of learners who improved.
2. **Student Lifecycle:** *time to first class vs later engagement*. Learners whose first class is >7 days after enrollment have fewer active days later, with sample sizes shown. It's labelled as association.
3. **Student Success:** **318 High / 774 Medium** risk learners, each with contributing indicators and a recommended intervention. Change a follow-up status in the table. Show the **backtest**: flags as of 1 Aug, precision ~73% for High, recall ~87% for High+Medium.
4. **Retention & Cohorts:** explain *confirmed churn* (cancellations, ~2%) vs *inferred inactivity* (30+ days quiet, ~37%). We never call a learner churned because they missed a few classes.

## 3. Finance & governance: "Can we trust these numbers?"
1. **Revenue & Finance:** the waterfall goes list price → discounts → bookings → cancelled → not yet collected → collected → refunds → net. Recognised revenue is labelled an *estimate*; MRR is *not applicable*; margin is *not available (no cost data)*.
2. **Courses & Instructors:** instructors are compared on score uplift vs the same course and topic, with 95% intervals. Sole instructors are marked "no in-course peer" rather than ranked.
3. **Data Quality:** the demo data scores 100/100 across 70+ checks. Toggle **test fixture** to show injected duplicates, orphan keys, impossible scores and negative payments being caught. Preview a cleaning action and download the cleaned copy; source data is never modified.
4. **Data Explorer:** run a JOIN across tables, then `DELETE FROM students` to show it is blocked.
5. **Metric Dictionary:** one definition per metric, used everywhere.

---

# 5-minute interview script

**0:00 – Problem (30s)**
"EdTech leaders drown in dashboards that show *what* but not *why* or *what to do*. And when numbers differ between pages, trust collapses. I built a platform where every page answers five questions, and every number comes from one metric layer."

**0:30 – Executive view (60s)** *(Overview)*
"Two clicks: the demo loads, and global filters apply everywhere. Each KPI has a definition, period, baseline and RAG status. When no baseline exists, it says so rather than inventing one. Insights are ranked by business impact (metric weight × change × reach), and each one says whether it is a verified metric, an association, or a rules-based flag."

**1:30 – Diagnose (75s)** *(AI Analyst → Lifecycle)*
"'Why did enrollment decline?' It splits lead volume from conversion and pinpoints Social Ads. Explanations are hypotheses. Actions are concrete. Confidence and sample size are explicit. Where SQL ran, I show it; where pandas computed it, I don't pretend there was SQL."

**2:45 – Act on learners (60s)** *(Student Success)*
"Risk flags are transparent rules, not a black box, and I validated them with a time-based backtest: ~73% precision for High risk. The team can log follow-ups right here. No sensitive attributes are used."

**3:45 – Trust (45s)** *(Data Quality, Revenue)*
"70+ governance checks, lineage, freshness, and safe cleaning previews. Finance metrics separate actuals from estimates, and I refuse to fabricate margin without cost data."

**4:30 – Scale at Vedantu (30s)**
"Next: connect the warehouse and CRM through the same schema contract, add experiment-based attribution for marketing, train and monitor a risk model against this rules baseline, and add an LLM layer for narrative on top of the governed metrics, never in place of them."
