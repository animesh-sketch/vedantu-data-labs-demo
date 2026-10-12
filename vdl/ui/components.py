"""Reusable UI components: theme CSS, KPI cards, insight cards, the five-questions panel, charts, empty states."""
from __future__ import annotations

import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..edtech import metrics as M

PALETTE = ["#4F46E5", "#0EA5A4", "#F59E0B", "#E11D48", "#3B82F6", "#8B5CF6", "#10B981", "#64748B"]
SEQ = ["#EEF2FF", "#C7D2FE", "#818CF8", "#4F46E5", "#312E81"]
RAG = {"green": ("#DCFCE7", "#166534", "On track"), "amber": ("#FEF3C7", "#92400E", "Watch"),
       "red": ("#FFE4E6", "#9F1239", "Act now"), "grey": ("#F1F5F9", "#475569", "No baseline")}

CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1500px;}
.hero {background: linear-gradient(120deg,#1E1B4B 0%,#4F46E5 55%,#0EA5A4 100%); color:#fff; padding:1.1rem 1.5rem;
       border-radius:16px; margin-bottom:1rem; display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;}
.hero h1 {margin:0; font-size:1.45rem; font-weight:700; color:#fff;}
.hero p {margin:.2rem 0 0; opacity:.85; font-size:.9rem;}
.pill {display:inline-block; white-space:nowrap; padding:.18rem .6rem; border-radius:999px; font-size:.72rem; font-weight:600; letter-spacing:.02em;}
.demo {background:#FDE68A; color:#78350F;} .live {background:#BBF7D0; color:#14532D;}
.kpi {border:1px solid rgba(100,116,139,.22); border-radius:14px; padding:.8rem .9rem; height:100%; background:rgba(255,255,255,.02);}
.kpi .l {font-size:.7rem; text-transform:uppercase; letter-spacing:.04em; display:flex; justify-content:space-between; align-items:flex-start; gap:.3rem;}
.kpi .l > span:first-child {opacity:.7;}
.kpi .pill {font-size:.62rem; padding:.1rem .45rem;}
.kpi .v {font-size:1.4rem; font-weight:700; margin-top:.15rem;}
.kpi .d {font-size:.78rem; margin-top:.1rem;}
.kpi .b {font-size:.68rem; opacity:.6; margin-top:.25rem;}
.ins {border-left:4px solid #4F46E5; padding:.55rem .8rem; margin:.4rem 0; border-radius:8px; background:rgba(79,70,229,.05);}
.ins.risk {border-left-color:#E11D48; background:rgba(225,29,72,.05);} .ins.positive {border-left-color:#10B981; background:rgba(16,185,129,.06);}
.ins .t {font-weight:650;} .ins .m {font-size:.82rem; opacity:.85; margin-top:.2rem;}
.tag {display:inline-block; padding:.05rem .45rem; border-radius:6px; font-size:.68rem; font-weight:600; margin-right:.3rem;}
.t-ver {background:#DCFCE7; color:#166534;} .t-assoc {background:#EDE9FE; color:#5B21B6;} .t-rule {background:#E0F2FE; color:#075985;}
.small {font-size:.8rem; opacity:.75;}
section[data-testid="stSidebar"] .stButton button {justify-content:flex-start; padding:.25rem .75rem; min-height:2rem;}
section[data-testid="stSidebar"] .stButton {margin-bottom:-.6rem;}
section[data-testid="stSidebar"] .stButton button p {text-align:left;}
</style>"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, demo: bool, scope: str = "") -> None:
    badge = "<span class='pill demo'>DEMO MODE · synthetic data</span>" if demo else "<span class='pill live'>CUSTOM DATA</span>"
    st.markdown(f"<div class='hero'><div><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>"
                f"<div style='text-align:right'>{badge}<div class='small' style='margin-top:.3rem'>{html.escape(scope)}</div></div></div>",
                unsafe_allow_html=True)


def kpi_card(k: M.KPI) -> str:
    bg, fg, label = RAG[k.rag]
    arrow = "" if k.change is None else ("▲" if k.change > 0 else "▼" if k.change < 0 else "■")
    tip = html.escape(f"{k.definition}\nCalculation: {k.formula}\nPeriod: {k.period}\nBaseline: {k.baseline}")
    return (f"<div class='kpi' title='{tip}'><div class='l'><span>{html.escape(k.name)}</span>"
            f"<span class='pill' style='background:{bg};color:{fg}'>{label}</span></div>"
            f"<div class='v'>{M.fmt(k.value, k.unit)}</div>"
            f"<div class='d' style='color:{fg}'>{arrow} {M.fmt_change(k)} <span class='small'>vs {M.fmt(k.prev, k.unit)}</span></div>"
            f"<div class='b'>{html.escape(k.period)} · hover for definition</div></div>")


def kpi_strip(kpis: list[M.KPI], per_row: int = 6) -> None:
    for i in range(0, len(kpis), per_row):
        cols = st.columns(per_row)
        for c, k in zip(cols, kpis[i:i + per_row]):
            c.markdown(kpi_card(k), unsafe_allow_html=True)
        st.write("")


def basis_tag(basis: str) -> str:
    cls = "t-ver" if basis.startswith("Verified") else "t-assoc" if basis.startswith("Assoc") else "t-rule"
    return f"<span class='tag {cls}'>{html.escape(basis)}</span>"


def insight_card(i, rank: int | None = None, show_evidence: bool = True) -> None:
    head = f"{rank}. " if rank else ""
    st.markdown(
        f"<div class='ins {i.polarity}'><div class='t'>{head}{html.escape(i.title)}</div>"
        f"<div class='m'>{basis_tag(i.basis)} <b>Observed:</b> {html.escape(i.observed)} · <b>Baseline:</b> {html.escape(i.baseline)} · "
        f"<b>Change:</b> {html.escape(i.change)} · <b>Segment:</b> {html.escape(i.segment)}</div>"
        f"<div class='m'><b>Implication:</b> {html.escape(i.implication)}</div>"
        f"<div class='m'><b>Suggested action:</b> {html.escape(i.action)} <span class='small'>· impact score {i.impact}</span></div></div>",
        unsafe_allow_html=True)
    if show_evidence and not i.evidence.empty:
        with st.expander("Evidence"):
            st.dataframe(i.evidence, hide_index=True, width="stretch")


def five_questions(what: list[M.KPI], why: tuple[str, pd.DataFrame | None], who: tuple[str, pd.DataFrame | None],
                   nxt: str, actions: list[str]) -> None:
    """The five questions every module answers."""
    st.markdown("##### Decision brief")
    t = st.tabs(["1 · What happened", "2 · Why", "3 · Who is affected", "4 · What next", "5 · What to do"])
    with t[0]:
        if what:
            st.markdown("  \n".join(f"- **{k.name}:** {M.fmt(k.value, k.unit)} ({M.fmt_change(k)} vs {M.fmt(k.prev, k.unit)}; "
                                    f"{RAG[k.rag][2].lower()})" for k in what))
            st.caption(f"Period {what[0].period}; baseline: {what[0].baseline}.")
    with t[1]:
        st.markdown(why[0])
        if why[1] is not None and not why[1].empty:
            st.dataframe(why[1], hide_index=True, width="stretch")
    with t[2]:
        st.markdown(who[0])
        if who[1] is not None and not who[1].empty:
            st.dataframe(who[1], hide_index=True, width="stretch")
    with t[3]:
        st.markdown(nxt)
    with t[4]:
        st.markdown("\n".join(f"{n + 1}. {a}" for n, a in enumerate(actions)) if actions else "No action needed: metrics within tolerance.")


def style(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), colorway=PALETTE, template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12), legend_title_text="",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hoverlabel=dict(font_size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(100,116,139,.15)")
    return fig


def chart(fig: go.Figure, title: str, caption: str | None = None, height: int = 340, legend: bool = True) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.plotly_chart(style(fig, height, legend), width="stretch", config={"displaylogo": False})
        if caption:
            st.caption(caption)


def heatmap(df: pd.DataFrame, title: str, caption: str | None = None, fmt: str = ".0f", height: int = 380,
            colorscale=None, zmid=None) -> None:
    if df is None or df.empty:
        empty(f"No data for {title.lower()} in the current filters.")
        return
    fig = px.imshow(df, text_auto=fmt, aspect="auto", color_continuous_scale=colorscale or SEQ, zmin=None, zmax=None)
    if zmid is not None:
        fig.update_coloraxes(cmid=zmid)
    fig.update_xaxes(side="top", title=None)
    fig.update_yaxes(title=None)
    chart(fig, title, caption, height, legend=False)


def empty(msg: str) -> None:
    st.info(f"ℹ️ {msg}")


def table(df: pd.DataFrame, caption: str | None = None, height: int | None = None) -> None:
    if df is None or df.empty:
        empty("No rows for the current filters.")
        return
    kw = {"height": height} if height else {}
    st.dataframe(df, hide_index=True, width="stretch", **kw)
    if caption:
        st.caption(caption)
