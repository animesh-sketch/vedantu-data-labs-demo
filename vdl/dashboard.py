"""Schema-driven dashboard: picks KPIs and charts from the detected column types."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px

from .profiling import Profile

METRIC_HINTS = ["revenue", "amount", "deal_value", "value", "fee", "sales", "score", "marks", "csat", "hours"]
ADDITIVE_HINTS = ["revenue", "amount", "value", "fee", "sales", "price", "cost", "calls", "count", "qty", "quantity"]
OUTCOME_HINTS = ["stage", "status", "outcome", "result", "completed", "converted", "churn", "priority"]
POSITIVE = ("won", "converted", "resolved", "yes", "true", "completed", "passed", "closed won")
PALETTE = ["#5B4BFF", "#00B3A4", "#FF8A3D", "#E84A7F", "#3DA5FF", "#9AA4B2"]


@dataclass
class KPI:
    label: str
    value: str
    help: str


def primary_metric(prof: Profile) -> str | None:
    for h in METRIC_HINTS:
        for c in prof.numeric:
            if h in c:
                return c
    return prof.numeric[0] if prof.numeric else None


def is_additive(col: str | None) -> bool:
    """Sums make sense for money/counts; scores, ratings, hours and percentages should be averaged."""
    return bool(col) and any(h in col.lower() for h in ADDITIVE_HINTS)


def agg_of(col: str | None) -> str:
    return "sum" if is_additive(col) else "mean"


def outcome_column(df: pd.DataFrame, prof: Profile) -> tuple[str, str] | None:
    """A categorical column holding a 'success' value (e.g. stage = Won). Returns (column, value)."""
    for c in prof.categorical:
        if df[c].nunique() > 8:
            continue
        values = {str(v).lower(): str(v) for v in df[c].dropna().unique()}
        for pos in POSITIVE:
            if pos in values:
                return c, values[pos]
    return None


def dimensions(df: pd.DataFrame, prof: Profile, limit: int = 3) -> list[str]:
    dims = [c for c in prof.categorical if 2 <= df[c].nunique() <= 25
            and not any(h in c.lower() for h in OUTCOME_HINTS)]
    return sorted(dims, key=lambda c: df[c].nunique())[:limit]


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1e7:
        return f"{v / 1e7:,.2f} Cr"
    if a >= 1e5:
        return f"{v / 1e5:,.2f} L"
    return f"{v:,.0f}" if a >= 100 else f"{v:,.2f}"


def kpis(df: pd.DataFrame, prof: Profile) -> list[KPI]:
    out = [KPI("Records", f"{len(df):,}", "count(*) of all rows")]
    m = primary_metric(prof)
    if m:
        s = df[m].dropna()
        if is_additive(m):
            out.append(KPI(f"Total {m}", _fmt(s.sum()), f"sum({m}) over {len(s):,} non-null rows"))
        out.append(KPI(f"Average {m}", _fmt(s.mean()), f"mean({m}) over {len(s):,} non-null rows"))
    oc = outcome_column(df, prof)
    if oc:
        c, pos = oc
        vals = df[c].dropna().astype(str)
        out.append(KPI(f"{c.replace('_', ' ').title()} = {pos}", f"{(vals == pos).mean():.1%}",
                       f"rows where {c} = '{pos}' / non-null {c} rows ({len(vals):,})"))
    if prof.datetime and len(out) < 5:
        d = prof.datetime[0]
        out.append(KPI("Date range", f"{df[d].min():%d %b %y} – {df[d].max():%d %b %y}", f"min/max of {d}"))
    return out[:5]


def _style(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=330, colorway=PALETTE,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Segoe UI, sans-serif"), legend_title_text="")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


def charts(df: pd.DataFrame, prof: Profile) -> list[tuple[str, object]]:
    figs = []
    m = primary_metric(prof)
    dims = dimensions(df, prof)
    if prof.datetime:
        d = prof.datetime[0]
        t = df.dropna(subset=[d]).set_index(d)
        trend = (t[m].resample("MS").agg(agg_of(m)) if m else t.resample("MS").size()).reset_index(name="value")
        figs.append((f"Monthly {agg_of(m).replace('sum', 'total').replace('mean', 'average')} {m}" if m else "Monthly records",
                     _style(px.area(trend, x=d, y="value", markers=True))))
    for dim in dims[:2]:
        if m:
            g = df.groupby(dim, dropna=False)[m].agg(agg_of(m)).round(2).sort_values().reset_index()
            title = f"{'Total' if is_additive(m) else 'Average'} {m} by {dim}"
        else:
            g = df[dim].value_counts(dropna=False).sort_values().reset_index()
            g.columns = [dim, "count"]
            title = f"Records by {dim}"
        g[dim] = g[dim].astype(str)
        figs.append((title, _style(px.bar(g, x=g.columns[1], y=dim, orientation="h"))))
    if m:
        figs.append((f"Distribution of {m}", _style(px.histogram(df, x=m, nbins=40))))
    oc = outcome_column(df, prof)
    pie_dim = oc[0] if oc else (dims[2] if len(dims) >= 3 else None)
    if pie_dim:
        c = df[pie_dim].astype(str).value_counts().reset_index()
        c.columns = [pie_dim, "count"]
        figs.append((f"Share by {pie_dim}", _style(px.pie(c, names=pie_dim, values="count", hole=0.55))))
    return figs
