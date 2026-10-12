"""Forecasting: damped-trend exponential smoothing (Holt) on weekly series, with backtest metrics,
empirical prediction intervals and scenario bands. No extra dependencies.

Seasonality is NOT modelled: 12 months of history is not enough to estimate a yearly cycle, and the
forecast says so. Series shorter than MIN_WEEKS return a data-sufficiency warning instead of a forecast."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .model import Model

MIN_WEEKS = 26
BACKTEST_WEEKS = 8


@dataclass
class Forecast:
    name: str
    history: pd.DataFrame  # week, actual
    forecast: pd.DataFrame | None = None  # week, baseline, lo80, hi80, lo95, hi95, conservative, optimistic
    backtest: dict = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    warning: str | None = None


def weekly_series(m: Model, what: str) -> pd.Series:
    end = m.end
    last_full = (end - pd.Timedelta(days=(end.weekday() + 1) % 7)).normalize()  # end of last complete week (Sunday)
    if what == "enrollments":
        en = m.enrollments[~m.enrollments.is_renewal.astype(bool)]
        s = en.set_index("enrollment_date").resample("W-SUN").size()
    elif what == "net_revenue":
        p = m.payments
        net = p.amount.where(p.payment_status == "Success", 0) - p.refund_amount.where(p.payment_status == "Refunded", 0)
        s = net.groupby(p.payment_date).sum().resample("W-SUN").sum()
    elif what == "active_learners":
        ev = m.events()
        s = ev.groupby(ev.ts.dt.to_period("W-SUN")).student_id.nunique()
        s.index = s.index.end_time.normalize()
    elif what == "tickets":
        s = m.tickets.set_index("created_at").resample("W-SUN").size()
    elif what == "completions":
        en = m.enrollments.dropna(subset=["completion_date"])
        s = en.set_index("completion_date").resample("W-SUN").size()
    else:
        raise ValueError(what)
    s.index = pd.to_datetime(s.index).normalize()
    s = s[(s.index >= m.data_start + pd.Timedelta(days=6)) & (s.index <= last_full)]
    return s.astype(float)


def _holt(y: np.ndarray, alpha: float, beta: float, phi: float) -> tuple[np.ndarray, float, float]:
    level, trend = y[0], (y[min(3, len(y) - 1)] - y[0]) / max(min(3, len(y) - 1), 1)
    fitted = np.empty(len(y))
    for t, v in enumerate(y):
        fitted[t] = level + phi * trend
        new_level = alpha * v + (1 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    return fitted, level, trend


def _fit(y: np.ndarray) -> tuple[float, float, float]:
    best, params = np.inf, (0.3, 0.1, 0.9)
    for a in (0.1, 0.2, 0.3, 0.5, 0.7):
        for b in (0.02, 0.05, 0.1, 0.2):
            for phi in (0.8, 0.9, 0.98):
                f, _, _ = _holt(y, a, b, phi)
                sse = float(np.sum((y[1:] - f[1:]) ** 2))
                if sse < best:
                    best, params = sse, (a, b, phi)
    return params


def _project(level, trend, phi, h) -> np.ndarray:
    return np.array([level + sum(phi ** i for i in range(1, k + 1)) * trend for k in range(1, h + 1)])


def forecast(m: Model, what: str, horizon: int = 12, scenario_pct: float = 10.0) -> Forecast:
    s = weekly_series(m, what)
    hist = s.reset_index()
    hist.columns = ["week", "actual"]
    fc = Forecast(what, hist)
    if len(s) < MIN_WEEKS:
        fc.warning = (f"Only {len(s)} complete weeks of history; at least {MIN_WEEKS} are needed for a reliable "
                      "forecast. No forecast shown.")
        return fc
    y = s.values
    # backtest every candidate on the last BACKTEST_WEEKS (fit on the rest), then pick the lowest MAE
    tr, te = y[:-BACKTEST_WEEKS], y[-BACKTEST_WEEKS:]
    a, b, phi = _fit(tr)
    _, lv, td = _holt(tr, a, b, phi)
    preds = {"Damped trend (Holt)": np.maximum(_project(lv, td, phi, BACKTEST_WEEKS), 0),
             "Naive (last week)": np.repeat(tr[-1], BACKTEST_WEEKS),
             "Moving average (4 wks)": np.repeat(tr[-4:].mean(), BACKTEST_WEEKS)}
    nz = te != 0
    scores = {k: {"MAE": round(float(np.mean(np.abs(te - v))), 1),
                  "MAPE_pct": round(float(np.mean(np.abs((te[nz] - v[nz]) / te[nz])) * 100), 1) if nz.any() else None}
              for k, v in preds.items()}
    chosen = min(scores, key=lambda k: scores[k]["MAE"])
    fc.backtest = {"holdout_weeks": BACKTEST_WEEKS, "chosen_model": chosen, **scores[chosen],
                   "candidates": scores}
    # final fit on all data with the chosen method
    a, b, phi = _fit(y)
    fitted, lv, td = _holt(y, a, b, phi)
    if chosen == "Damped trend (Holt)":
        resid = (y - fitted)[1:]
        base = np.maximum(_project(lv, td, phi, horizon), 0)
    elif chosen == "Naive (last week)":
        resid = np.diff(y)
        base = np.repeat(y[-1], horizon)
    else:
        ma = pd.Series(y).rolling(4).mean().shift(1).to_numpy()
        resid = (y - ma)[4:]
        base = np.repeat(y[-4:].mean(), horizon)
    sd = float(np.std(resid, ddof=1))
    k = np.sqrt(np.arange(1, horizon + 1))
    weeks = pd.date_range(s.index[-1] + pd.Timedelta(days=7), periods=horizon, freq="7D")
    fc.forecast = pd.DataFrame({"week": weeks, "baseline": base.round(1),
                                "lo80": np.maximum(base - 1.28 * sd * k, 0).round(1), "hi80": (base + 1.28 * sd * k).round(1),
                                "lo95": np.maximum(base - 1.96 * sd * k, 0).round(1), "hi95": (base + 1.96 * sd * k).round(1),
                                "conservative": (base * (1 - scenario_pct / 100)).round(1),
                                "optimistic": (base * (1 + scenario_pct / 100)).round(1)})
    fc.assumptions = [
        f"Model chosen by {BACKTEST_WEEKS}-week holdout backtest (lowest MAE): {chosen}. "
        f"Fitted on {len(y)} complete weeks" + (f" (alpha={a}, beta={b}, phi={phi})." if chosen.startswith("Damped") else "."),
        "No yearly seasonality modelled: one year of history cannot separate seasonality from trend.",
        "Intervals are empirical (±1.28σ / ±1.96σ of in-sample residuals, widening with √h); not guaranteed coverage.",
        f"Scenarios: conservative = baseline −{scenario_pct:.0f}%, optimistic = baseline +{scenario_pct:.0f}% (planning assumptions, not statistical bounds).",
        "Assumes current filters, pricing and marketing mix stay unchanged.",
    ]
    if fc.backtest.get("MAPE_pct") and fc.backtest["MAPE_pct"] > 25:
        fc.warning = f"Backtest error is high (MAPE {fc.backtest['MAPE_pct']}%). Use the range, not the point forecast."
    return fc
