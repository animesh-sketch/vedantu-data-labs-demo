"""Data quality checks. Every check returns evidence, the calculation used and a status."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .profiling import Profile


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str
    calculation: str
    evidence: pd.DataFrame | None = None


def _status(share: float, warn: float, fail: float) -> str:
    return "FAIL" if share >= fail else "WARN" if share >= warn else "PASS"


def missing_values(df: pd.DataFrame) -> Check:
    miss = df.isna().sum()
    total = int(miss.sum())
    share = total / max(df.size, 1)
    ev = pd.DataFrame({"missing": miss, "pct": (miss / max(len(df), 1) * 100).round(2)})
    ev = ev[ev.missing > 0].sort_values("missing", ascending=False)
    return Check("Missing values", _status(share, 0.01, 0.10),
                 f"{total:,} missing cells ({share:.2%} of all cells) across {len(ev)} column(s).",
                 "count(isnull(cell)) per column / row count", ev)


def duplicates(df: pd.DataFrame) -> Check:
    mask = df.duplicated(keep="first")
    n = int(mask.sum())
    return Check("Duplicate rows", _status(n / max(len(df), 1), 0.001, 0.05),
                 f"{n:,} fully duplicated row(s) out of {len(df):,}.",
                 "rows identical to an earlier row across all columns", df[mask].head(20))


def invalid_types(raw: pd.DataFrame, typed: pd.DataFrame, prof: Profile) -> Check:
    rows = []
    for c, t in prof.types.items():
        if t in ("numeric", "datetime"):
            bad = typed[c].isna() & raw[c].notna()
            if bad.any():
                rows.append({"column": c, "expected": t, "invalid_count": int(bad.sum()),
                             "examples": ", ".join(map(str, raw.loc[bad, c].astype(str).unique()[:5]))})
    ev = pd.DataFrame(rows, columns=["column", "expected", "invalid_count", "examples"])
    n = int(ev.invalid_count.sum()) if len(ev) else 0
    return Check("Invalid types", _status(n / max(len(raw), 1), 0.001, 0.05),
                 f"{n:,} value(s) could not be parsed as their detected type.",
                 "non-null raw value that becomes null after type coercion", ev)


def anomalies(df: pd.DataFrame, prof: Profile) -> Check:
    rows = []
    for c in prof.numeric:
        s = df[c].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        out = s[(s < lo) | (s > hi)]
        neg = int((s < 0).sum()) if (s >= 0).mean() > 0.95 else 0
        if len(out) or neg:
            rows.append({"column": c, "outliers": len(out), "lower_fence": round(lo, 2),
                         "upper_fence": round(hi, 2), "unexpected_negatives": neg,
                         "max_value": round(float(s.max()), 2)})
    for c in prof.datetime:
        future = int((df[c] > pd.Timestamp.now() + pd.Timedelta(days=1)).sum())
        if future:
            rows.append({"column": c, "outliers": future, "lower_fence": np.nan, "upper_fence": np.nan,
                         "unexpected_negatives": 0, "max_value": str(df[c].max())})
    ev = pd.DataFrame(rows)
    n = int(ev.outliers.sum() + ev.unexpected_negatives.sum()) if len(ev) else 0
    return Check("Potential anomalies", "PASS" if n == 0 else _status(n / max(len(df), 1), 0, 0.05),
                 f"{n:,} value(s) flagged as potential anomalies (to review, not auto-removed).",
                 "numeric: outside Q1-3*IQR..Q3+3*IQR, or negative in a mostly non-negative column; dates: in the future",
                 ev)


def run_all(raw: pd.DataFrame, typed: pd.DataFrame, prof: Profile) -> list[Check]:
    return [missing_values(typed), duplicates(typed), invalid_types(raw, typed, prof), anomalies(typed, prof)]


def trust_score(checks: list[Check]) -> int:
    penalty = {"PASS": 0, "WARN": 10, "FAIL": 25}
    return max(0, 100 - sum(penalty[c.status] for c in checks))
