"""Business recommendations.

Each insight separates three things:
  finding     - VERIFIED: computed from the data, with the calculation and evidence table
  explanation - HYPOTHESIS: a possible reason, not proven by this data
  action      - SUGGESTION: what a team could do or test next
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .dashboard import agg_of, dimensions, is_additive, outcome_column, primary_metric
from .profiling import Profile
from .quality import Check


@dataclass
class Insight:
    kind: str  # trend | risk | opportunity | quality
    title: str
    finding: str
    calculation: str
    evidence: pd.DataFrame
    explanation: str
    action: str


def _period_change(df: pd.DataFrame, prof: Profile, m: str | None) -> Insight | None:
    if not prof.datetime:
        return None
    d = prof.datetime[0]
    t = df.dropna(subset=[d]).set_index(d)
    series = t[m].resample("MS").agg(agg_of(m)) if m else t.resample("MS").size()
    last_date = t.index.max()
    if len(series) and last_date < (series.index[-1] + pd.offsets.MonthEnd(0)).normalize():
        series = series.iloc[:-1]  # drop incomplete latest month
    if len(series) < 2 or series.iloc[-2] == 0:
        return None
    prev, cur = series.iloc[-2], series.iloc[-1]
    chg = (cur - prev) / prev
    label = f"{'total' if is_additive(m) else 'average'} {m}" if m else "record count"
    ev = series.tail(6).reset_index()
    ev.columns = ["month", label]
    up = chg > 0
    return Insight(
        "trend" if abs(chg) < 0.1 else ("opportunity" if up else "risk"),
        f"{label.capitalize()} {'rose' if up else 'fell'} {abs(chg):.1%} month on month",
        f"{label.capitalize()} was {cur:,.2f} in {series.index[-1]:%b %Y} vs {prev:,.2f} in {series.index[-2]:%b %Y}.",
        f"(month_n - month_n-1) / month_n-1 on monthly {agg_of(m) if m else 'count'} of {m or 'rows'}; "
        "latest incomplete month excluded",
        ev,
        "Could reflect seasonality (exam cycles, admissions windows), campaign timing or pipeline changes. "
        "Not proven by this data.",
        "Compare with the same month last year and with campaign/calendar data before acting." if up else
        "Drill into the segments driving the drop (use the AI Analyst: 'monthly trend by <segment>').",
    )


def _concentration(df: pd.DataFrame, dim: str, m: str | None) -> Insight | None:
    g = (df.groupby(dim)[m].sum() if m else df[dim].value_counts()).sort_values(ascending=False)
    if len(g) < 3 or g.sum() <= 0:
        return None
    share = g.iloc[0] / g.sum()
    if share < 0.35:
        return None
    ev = pd.DataFrame({dim: g.index.astype(str), "value": g.values, "share": (g / g.sum()).round(3).values})
    return Insight(
        "risk", f"{share:.0%} of {'total ' + m if m else 'records'} comes from one {dim}",
        f"'{g.index[0]}' contributes {share:.1%} across {len(g)} {dim} values.",
        f"sum({m or 'rows'}) for top {dim} / sum over all {dim}", ev.head(10),
        "Could be a genuinely stronger segment or under-investment elsewhere.",
        f"Treat as concentration risk: set a target to grow the next two {dim} values and monitor dependency.",
    )


def _segment_gap(df: pd.DataFrame, dim: str, m: str) -> Insight | None:
    g = df.groupby(dim)[m].agg(["mean", "count"])
    g = g[g["count"] >= 20]
    if len(g) < 2:
        return None
    best, worst = g["mean"].idxmax(), g["mean"].idxmin()
    if g.loc[worst, "mean"] <= 0:
        return None
    ratio = g.loc[best, "mean"] / g.loc[worst, "mean"]
    if ratio < 1.25:
        return None
    ev = g.sort_values("mean", ascending=False).round(2).reset_index()
    return Insight(
        "opportunity", f"Average {m} is {ratio:.1f}x higher for {dim} '{best}' than '{worst}'",
        f"mean({m}) = {g.loc[best, 'mean']:,.2f} for '{best}' vs {g.loc[worst, 'mean']:,.2f} for '{worst}' "
        f"(segments with ≥20 rows only).",
        f"mean({m}) grouped by {dim}; segments with fewer than 20 rows excluded", ev,
        "Mix effects (product, region, cohort) may explain part of the gap; not tested here.",
        f"Study what '{best}' does differently and run a controlled pilot for '{worst}'.",
    )


def _outcome_gap(df: pd.DataFrame, oc: tuple[str, str], dim: str) -> Insight | None:
    col, pos = oc
    d = df.dropna(subset=[col, dim])
    g = d.assign(_hit=(d[col].astype(str) == pos)).groupby(dim)["_hit"].agg(["mean", "count"])
    g = g[g["count"] >= 30]
    if len(g) < 2:
        return None
    best, worst = g["mean"].idxmax(), g["mean"].idxmin()
    if g.loc[best, "mean"] - g.loc[worst, "mean"] < 0.05:
        return None
    ev = g.sort_values("mean", ascending=False).reset_index().rename(columns={"mean": f"{pos}_rate", "count": "rows"})
    ev[f"{pos}_rate"] = (ev[f"{pos}_rate"] * 100).round(1)
    return Insight(
        "opportunity", f"'{pos}' rate ranges from {g.loc[worst, 'mean']:.0%} to {g.loc[best, 'mean']:.0%} across {dim}",
        f"{col} = '{pos}' for {g.loc[best, 'mean']:.1%} of '{best}' rows vs {g.loc[worst, 'mean']:.1%} of '{worst}' "
        f"(segments with ≥30 rows).",
        f"count({col} = '{pos}') / count(rows) per {dim}", ev,
        "Differences in lead quality, intent or follow-up process may drive this; not tested here.",
        f"Shift effort/budget towards '{best}' and test process fixes for '{worst}'; track {pos} rate weekly.",
    )


def _quality(checks: list[Check]) -> list[Insight]:
    out = []
    for c in checks:
        if c.status != "PASS":
            out.append(Insight("quality", f"Data quality: {c.name} ({c.status})", c.detail, c.calculation,
                               c.evidence if c.evidence is not None else pd.DataFrame(),
                               "Usually source-system entry or export issues.",
                               "Fix at source; until then, treat affected metrics as approximate."))
    return out


def generate(df: pd.DataFrame, prof: Profile, checks: list[Check]) -> list[Insight]:
    m = primary_metric(prof)
    found: list[Insight | None] = [_period_change(df, prof, m)]
    oc = outcome_column(df, prof)
    for dim in dimensions(df, prof):
        if oc:
            found.append(_outcome_gap(df, oc, dim))
        if is_additive(m) or not m:
            found.append(_concentration(df, dim, m if is_additive(m) else None))
        if m:
            found.append(_segment_gap(df, dim, m))
    return [i for i in found if i] + _quality(checks)
