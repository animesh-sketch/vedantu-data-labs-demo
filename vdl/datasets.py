"""Synthetic demo datasets (seeded, reproducible) with a few deliberate data-quality issues."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES = {
    "Sales / CRM leads": "sales_crm.csv",
    "Education – student learning": "education.csv",
    "Customer support tickets": "support_tickets.csv",
}
CITIES = ["Bengaluru", "Delhi", "Mumbai", "Hyderabad", "Kolkata", "Pune", "Chennai", "Lucknow"]


def _dates(rng, n, start="2025-10-01", days=365):
    return pd.Timestamp(start) + pd.to_timedelta(np.sort(rng.integers(0, days, n)), unit="D")


def sales_crm(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    source = rng.choice(["Organic", "Paid Social", "Referral", "Webinar", "School Partnership"], n,
                        p=[.3, .3, .15, .15, .1])
    product = rng.choice(["JEE Pro", "NEET Pro", "Foundation", "Tutoring 1:1", "Boards Crash"], n)
    base = {"JEE Pro": 42000, "NEET Pro": 40000, "Foundation": 18000, "Tutoring 1:1": 30000, "Boards Crash": 6000}
    p_won = np.select([source == "Referral", source == "Webinar", source == "Paid Social"], [.32, .26, .12], .18)
    stage = np.where(rng.random(n) < p_won, "Won",
                     rng.choice(["Lost", "Negotiation", "Demo Booked", "Contacted"], n, p=[.5, .15, .15, .2]))
    df = pd.DataFrame({
        "lead_id": [f"L{100000 + i}" for i in range(n)],
        "created_date": _dates(rng, n),
        "city": rng.choice(CITIES, n, p=[.24, .2, .16, .12, .08, .08, .07, .05]),
        "lead_source": source,
        "product": product,
        "counsellor": rng.choice([f"Counsellor {c}" for c in "ABCDEFGH"], n),
        "stage": stage,
        "deal_value": np.round([base[p] * rng.uniform(.8, 1.2) for p in product], 0),
        "calls_made": rng.poisson(4, n),
    })
    df["deal_value"] = df["deal_value"].astype(float)
    # deliberate issues
    df.loc[rng.choice(n, 25, replace=False), "city"] = np.nan
    df.loc[rng.choice(n, 3, replace=False), "deal_value"] = 950000.0  # anomaly
    df = pd.concat([df, df.sample(8, random_state=seed)], ignore_index=True)  # duplicates
    return df


def education(n: int = 1800, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grade = rng.choice(["Class 9", "Class 10", "Class 11", "Class 12"], n)
    subject = rng.choice(["Physics", "Chemistry", "Maths", "Biology"], n)
    mode = rng.choice(["Live Class", "Recorded", "1:1 Tutoring"], n, p=[.5, .35, .15])
    att = np.clip(rng.normal(78, 14, n), 10, 100).round(1)
    hours = np.clip(rng.gamma(3, 3, n), 0, 60).round(1)
    score = np.clip(25 + 0.45 * att + 0.6 * hours + np.where(mode == "1:1 Tutoring", 6, 0) + rng.normal(0, 9, n), 0, 100)
    df = pd.DataFrame({
        "student_id": [f"S{50000 + i}" for i in range(n)],
        "enrolment_date": _dates(rng, n, "2025-06-01", 400),
        "grade": grade, "subject": subject, "learning_mode": mode,
        "city": rng.choice(CITIES, n),
        "attendance_pct": att, "study_hours_week": hours,
        "test_score": score.round(1),
        "completed_course": np.where(rng.random(n) < (att / 120), "Yes", "No"),
    })
    df["test_score"] = df["test_score"].astype(object)
    df.loc[rng.choice(n, 6, replace=False), "test_score"] = "absent"  # invalid type
    df.loc[rng.choice(n, 40, replace=False), "study_hours_week"] = np.nan
    return df


def support(n: int = 1500, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = rng.choice(["Payment", "Login", "Class Access", "Refund", "Content Error"], n, p=[.25, .2, .3, .1, .15])
    channel = rng.choice(["Chat", "Email", "Phone", "App"], n)
    hrs = rng.gamma(2, 6, n) * np.where(cat == "Refund", 3, 1) * np.where(channel == "Email", 1.6, 1)
    df = pd.DataFrame({
        "ticket_id": [f"T{900000 + i}" for i in range(n)],
        "opened_date": _dates(rng, n),
        "category": cat, "channel": channel,
        "priority": rng.choice(["Low", "Medium", "High", "Urgent"], n, p=[.3, .4, .2, .1]),
        "status": rng.choice(["Resolved", "Open", "Escalated"], n, p=[.82, .12, .06]),
        "resolution_hours": hrs.round(1),
        "csat": np.clip(np.round(5 - hrs / 25 + rng.normal(0, .7, n)), 1, 5),
        "agent": rng.choice([f"Agent {i}" for i in range(1, 11)], n),
    })
    df.loc[df.status != "Resolved", "resolution_hours"] = np.nan
    df.loc[rng.choice(n, 4, replace=False), "resolution_hours"] = -3.0  # anomaly
    return df


def write_all() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    sales_crm().to_csv(DATA_DIR / SAMPLES["Sales / CRM leads"], index=False)
    education().to_csv(DATA_DIR / SAMPLES["Education – student learning"], index=False)
    support().to_csv(DATA_DIR / SAMPLES["Customer support tickets"], index=False)


def load_sample(label: str) -> pd.DataFrame:
    path = DATA_DIR / SAMPLES[label]
    if not path.exists():
        write_all()
    return pd.read_csv(path)


if __name__ == "__main__":
    write_all()
    print(f"Wrote {len(SAMPLES)} datasets to {DATA_DIR}")
