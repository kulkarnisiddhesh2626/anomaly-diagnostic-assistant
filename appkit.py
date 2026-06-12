"""Shared UI toolkit for the multi-page Streamlit app.

Centralises the design system (CSS), cached data/detection loaders, the sidebar,
and reusable layout/chart helpers so each page in ``views/`` stays small and
declarative. Keeping this in one place is also what lets the four pages share a
single data load and a single visual language.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src import config, detection, diagnosis, llm_client  # noqa: F401 (re-export)
from src.context_builder import build_chat_context, incident_context  # noqa: F401
from src.data_generator import generate_and_save, load_transactions

INK = "#0E1726"
ACCENT = "#5B5BD6"
SEV_COLORS = {
    "high": ("#B91C1C", "#FBE7E7"),
    "medium": ("#B45309", "#FBF0DC"),
    "low": ("#1D6F42", "#E7F6ED"),
}
SUGGESTED_QUESTIONS = [
    "Why did approval rates drop on the worst day?",
    "Which decline reason codes drove the biggest incident?",
    "Was there any fraud activity, and where was it concentrated?",
    "Summarise all detected incidents.",
]


# --------------------------------------------------------------------------- #
# Cached data + detection (shared across every page)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load(n_days: int, seed: int) -> pd.DataFrame:
    if not config.TRANSACTIONS_CSV.exists():
        generate_and_save(n_days=n_days, seed=seed)
    return load_transactions()


@st.cache_data(show_spinner=False)
def _detect(_token: str, z: float, min_vol: int):
    df = load_transactions()
    events = detection.detect(df, z_threshold=z, min_volume=min_vol)
    incidents = detection.consolidate(events)
    return events, incidents


@st.cache_data(show_spinner=False)
def _chat_context(_token: str, z: float, min_vol: int):
    df = load_transactions()
    _, incidents = _detect(_token, z, min_vol)
    return build_chat_context(df, incidents)


def load_context() -> dict:
    """Read sidebar settings from session_state and return the shared context.

    Cheap to call on every page because the underlying work is cached.
    """
    ss = st.session_state
    ss.setdefault("z", config.ROBUST_Z_THRESHOLD)
    ss.setdefault("min_vol", config.MIN_HOURLY_VOLUME)
    ss.setdefault("n_days", 45)
    ss.setdefault("seed", 7)
    df = _load(ss.n_days, int(ss.seed))
    token = f"{len(df)}-{ss.z}-{ss.min_vol}"
    events, incidents = _detect(token, ss.z, ss.min_vol)
    return {"df": df, "events": events, "incidents": incidents,
            "token": token, "online": llm_client.is_online()}


def _regenerate():
    generate_and_save(n_days=st.session_state.n_days, seed=int(st.session_state.seed))
    st.cache_data.clear()


# --------------------------------------------------------------------------- #
# Global chrome: CSS, top bar, sidebar
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
html, body, .stApp, [class*="css"] {{ font-family:'Inter',sans-serif; }}
.stApp {{ background: radial-gradient(1200px 420px at 82% -140px, #EAE9FB 0%, rgba(234,233,251,0) 60%), #F4F6FB; }}
h1,h2,h3,h4 {{ font-family:'Space Grotesk',sans-serif; letter-spacing:-0.01em; color:{INK}; }}
.block-container {{ padding-top:1.0rem; padding-bottom:2.5rem; max-width:1280px; }}
#MainMenu, footer {{ visibility:hidden; }}
[data-testid="stHeader"] {{ background:transparent; height:0rem; }}

/* top bar */
.topbar {{ background:linear-gradient(135deg,{INK} 0%,#1A2740 100%); border-radius:16px;
    padding:16px 22px; margin-bottom:18px; color:#EAF0FB; display:flex;
    justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;
    box-shadow:0 10px 30px -14px rgba(14,23,38,.45); }}
.topbar .brand {{ font-family:'Space Grotesk'; font-weight:700; font-size:1.25rem; color:#fff; }}
.topbar .tag {{ color:#9DB0CC; font-size:.8rem; margin-top:2px; }}
.badge-row {{ display:flex; gap:8px; flex-wrap:wrap; }}
.chip {{ display:inline-flex; align-items:center; gap:6px; font-size:.74rem; font-weight:600;
    padding:5px 11px; border-radius:999px; background:rgba(255,255,255,.08);
    color:#C9D6EC; border:1px solid rgba(255,255,255,.12); white-space:nowrap; }}
.chip .dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; }}
.chip.live {{ background:rgba(34,197,94,.14); border-color:rgba(34,197,94,.35); color:#BBF7D0; }}
.chip.off {{ background:rgba(245,180,80,.14); border-color:rgba(245,180,80,.35); color:#FDE9C7; }}

/* page heading */
.page-title {{ font-family:'Space Grotesk'; font-weight:700; font-size:1.5rem; color:{INK}; margin:2px 0 2px; }}
.page-sub {{ color:#6B7A95; font-size:.9rem; margin-bottom:14px; }}
.section-label {{ font-size:.72rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:#7C8AA5; margin:10px 0 8px; }}

/* KPI tiles */
.kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:4px; }}
.kpi {{ flex:1 1 0; min-width:150px; background:#fff; border:1px solid #E6EBF4; border-radius:16px;
    padding:16px 18px; box-shadow:0 1px 2px rgba(14,23,38,.04); }}
.kpi .label {{ font-size:.72rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:#7C8AA5; }}
.kpi .value {{ font-family:'JetBrains Mono'; font-weight:600; font-size:1.8rem; color:{INK}; margin-top:6px; line-height:1.15; }}
.kpi .sub {{ font-size:.78rem; color:#7C8AA5; margin-top:4px; }}
.kpi .sub b.h {{ color:#B91C1C; }} .kpi .sub b.m {{ color:#B45309; }} .kpi .sub b.l {{ color:#1D6F42; }}
.kpi.accent {{ background:linear-gradient(160deg,#EEEEFB,#FFF); border-color:#D9D9F6; }}

/* pipeline stepper */
.steps {{ display:flex; gap:10px; flex-wrap:wrap; margin:6px 0 18px; }}
.step {{ flex:1 1 0; min-width:170px; background:#fff; border:1px solid #E6EBF4; border-left:4px solid {ACCENT};
    border-radius:12px; padding:13px 15px; position:relative; }}
.step .n {{ font-family:'JetBrains Mono'; font-size:.7rem; color:{ACCENT}; font-weight:600; }}
.step .t {{ font-family:'Space Grotesk'; font-weight:600; color:{INK}; font-size:.96rem; margin:2px 0; }}
.step .d {{ font-size:.78rem; color:#7C8AA5; }}
.step .big {{ font-family:'JetBrains Mono'; font-weight:600; color:{INK}; font-size:1.1rem; }}

/* severity badge + incident card */
.sev {{ font-family:'JetBrains Mono'; font-size:.7rem; font-weight:600; letter-spacing:.05em; padding:3px 9px; border-radius:7px; text-transform:uppercase; }}
.inc-head {{ background:#fff; border:1px solid #E6EBF4; border-radius:16px; padding:18px 20px; margin-bottom:14px; }}
.inc-head-top {{ display:flex; align-items:center; gap:10px; margin-bottom:8px; }}
.inc-id {{ font-family:'JetBrains Mono'; font-size:.78rem; color:#7C8AA5; }}
.inc-title {{ font-family:'Space Grotesk'; font-weight:600; font-size:1.18rem; color:{INK}; margin-bottom:14px; }}
.inc-meta {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px 28px; }}
.inc-meta .k {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.05em; color:#7C8AA5; display:block; }}
.inc-meta .v {{ font-family:'JetBrains Mono'; font-size:.9rem; color:{INK}; }}

/* callout */
.callout {{ background:#EEF0FB; border:1px solid #DADCF6; border-radius:12px; padding:13px 16px;
    color:#33406A; font-size:.9rem; margin-bottom:14px; }}
.callout b {{ color:{INK}; }}

/* sidebar = ink rail */
[data-testid="stSidebar"] {{ background:linear-gradient(180deg,#0E1726 0%,#13203A 100%); }}
[data-testid="stSidebar"] * {{ color:#D7E0F0; }}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {{ color:#fff; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color:#AEBCD6; font-size:.82rem; }}
[data-testid="stSidebar"] .stButton button {{ background:{ACCENT}; color:#fff; border:0; border-radius:10px; font-weight:600; width:100%; }}
[data-testid="stSidebar"] .stButton button:hover {{ background:#4A47C9; }}
[data-testid="stSidebar"] input {{ color:{INK}; }}
[data-testid="stSidebarNav"] a span {{ color:#D7E0F0; font-weight:600; }}

/* tabs (used inside pages occasionally) + suggested chips */
.stTabs [data-baseweb="tab"] {{ font-family:'Space Grotesk'; font-weight:600; color:#7C8AA5; }}
.stTabs [aria-selected="true"] {{ color:{INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{ACCENT}; height:3px; border-radius:3px; }}
[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; border:1px solid #E6EBF4; }}
</style>
""", unsafe_allow_html=True)


def topbar(ctx: dict) -> None:
    if ctx["online"]:
        llm_chip = (f'<span class="chip live"><span class="dot" style="background:#22C55E"></span>'
                    f'LLM · {llm_client.provider_label()}</span>')
    else:
        llm_chip = ('<span class="chip off"><span class="dot" style="background:#F5B450"></span>'
                    'LLM offline · template mode</span>')
    df = ctx["df"]
    window = (f'<span class="chip"><span class="dot" style="background:#5B5BD6"></span>'
              f'{df.timestamp.min():%d %b}–{df.timestamp.max():%d %b} · hourly</span>')
    st.markdown(f"""
<div class="topbar">
  <div><div class="brand">🛰️ Issuer Anomaly Console</div>
  <div class="tag">Detection → grounded diagnosis → analyst assistant · synthetic card-issuer data</div></div>
  <div class="badge-row">{llm_chip}{window}</div>
</div>""", unsafe_allow_html=True)


def render_sidebar() -> None:
    """Status, global controls, and a Help popover — drawn once per run."""
    ss = st.session_state
    ss.setdefault("z", config.ROBUST_Z_THRESHOLD)
    ss.setdefault("min_vol", config.MIN_HOURLY_VOLUME)
    ss.setdefault("n_days", 45)
    ss.setdefault("seed", 7)

    with st.sidebar:
        online = llm_client.is_online()
        st.markdown("#### Status")
        if online:
            st.markdown(f'<div class="chip live"><span class="dot" style="background:#22C55E"></span>'
                        f'LLM online</div><div style="font-family:JetBrains Mono;font-size:.76rem;'
                        f'color:#AEBCD6;margin-top:6px">{llm_client.provider_label()}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="chip off"><span class="dot" style="background:#F5B450"></span>'
                        'offline · templates</div>', unsafe_allow_html=True)
            st.caption("Add an Anthropic or Groq key in .env for live narratives.")

        if hasattr(st, "popover"):
            with st.popover("❔  How to use this"):
                st.markdown(
                    "**1. Overview** — portfolio health and the detection pipeline.\n\n"
                    "**2. Incidents** — pick a detected incident to see its grounded "
                    "diagnosis and the exact facts the LLM was given.\n\n"
                    "**3. Assistant** — ask follow-up questions in plain language.\n\n"
                    "**4. How it works** — the method, design choices, and what's graded.\n\n"
                    "Use the **controls** below to regenerate data or change detection "
                    "sensitivity; every page updates together.")

        st.markdown("#### Data")
        ss.n_days = st.slider("Days of history", 30, 90, ss.n_days, step=5,
                              help="Length of the synthetic history to generate.")
        ss.seed = st.number_input("Random seed", value=int(ss.seed), step=1,
                                  help="Change the seed for a different synthetic scenario.")
        if st.button("Regenerate dataset"):
            _regenerate()
            st.rerun()

        st.markdown("#### Detection sensitivity")
        ss.z = st.slider("z-score threshold", 3.0, 8.0, float(ss.z), step=0.5,
                         help="Higher = stricter: fewer, higher-confidence alerts.")
        ss.min_vol = st.slider("Min hourly volume", 10, 200, int(ss.min_vol), step=10,
                               help="Ignore slices with too little volume to be reliable.")


# --------------------------------------------------------------------------- #
# Reusable view helpers
# --------------------------------------------------------------------------- #
def page_title(title: str, sub: str) -> None:
    st.markdown(f'<div class="page-title">{title}</div>'
                f'<div class="page-sub">{sub}</div>', unsafe_allow_html=True)


def sev_badge(sev: str) -> str:
    fg, bg = SEV_COLORS.get(sev, ("#475569", "#EEF1F6"))
    return f'<span class="sev" style="color:{fg};background:{bg};">{sev}</span>'


def scope_label(scope: dict) -> str:
    return " · ".join(f"{k}={v}" for k, v in scope.items()) if scope else "global"


def kpi_row(ctx: dict) -> None:
    df, events, incidents = ctx["df"], ctx["events"], ctx["incidents"]
    tot = int(df.txn_count.sum())
    appr = 100 * df.approved_count.sum() / tot
    fraud = 100 * df.fraud_count.sum() / tot
    sev = {"high": 0, "medium": 0, "low": 0}
    for i in incidents:
        sev[i.severity] = sev.get(i.severity, 0) + 1
    st.markdown(f"""
<div class="kpi-row">
  <div class="kpi"><div class="label">Transactions</div><div class="value">{tot/1e6:.2f}M</div>
      <div class="sub">{tot:,} simulated auths</div></div>
  <div class="kpi"><div class="label">Approval rate</div><div class="value">{appr:.1f}%</div>
      <div class="sub">portfolio average</div></div>
  <div class="kpi"><div class="label">Fraud rate</div><div class="value">{fraud:.2f}%</div>
      <div class="sub">of transaction count</div></div>
  <div class="kpi accent"><div class="label">Incidents</div><div class="value">{len(incidents)}</div>
      <div class="sub"><b class="h">{sev['high']} high</b> · <b class="m">{sev['medium']} med</b> · <b class="l">{sev['low']} low</b></div></div>
  <div class="kpi"><div class="label">Granular alerts</div><div class="value">{len(events)}</div>
      <div class="sub">hourly slice flags</div></div>
</div>""", unsafe_allow_html=True)


def pipeline_steps(ctx: dict) -> None:
    df, events, incidents = ctx["df"], ctx["events"], ctx["incidents"]
    diag = "ready" if incidents else "—"
    st.markdown(f"""
<div class="steps">
  <div class="step"><div class="n">01 · INGEST</div><div class="t">Synthetic data</div>
      <div class="d"><span class="big">{len(df):,}</span> hourly slice-rows</div></div>
  <div class="step"><div class="n">02 · DETECT</div><div class="t">Statistical + ML</div>
      <div class="d"><span class="big">{len(events)}</span> slice flags → <span class="big">{len(incidents)}</span> incidents</div></div>
  <div class="step"><div class="n">03 · DIAGNOSE</div><div class="t">Grounded GenAI</div>
      <div class="d">narratives <span class="big">{diag}</span></div></div>
  <div class="step"><div class="n">04 · ASK</div><div class="t">Analyst assistant</div>
      <div class="d">grounded Q&amp;A</div></div>
</div>""", unsafe_allow_html=True)


def _alt_theme(chart):
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor="#ECEFF6", gridDash=[2, 3], domain=False,
                            tickColor="#E2E8F2", labelColor="#7C8AA5", titleColor="#7C8AA5",
                            labelFont="Inter", titleFont="Inter", labelFontSize=11, titleFontSize=11)
            .configure_axisX(grid=False)
            .configure_legend(labelColor="#5A6B85", titleColor="#7C8AA5",
                              labelFont="Inter", titleFont="Inter"))


def health_chart(df: pd.DataFrame, incidents: list, metric_choice: str):
    g = df.groupby("timestamp").agg(txn=("txn_count", "sum"), appr=("approved_count", "sum"),
                                    dec=("declined_count", "sum"), fr=("fraud_count", "sum")).reset_index()
    g["Approval rate %"] = 100 * g.appr / g.txn
    g["Decline rate %"] = 100 * g.dec / g.txn
    g["Fraud rate %"] = 100 * g.fr / g.txn
    ycol = "txn" if metric_choice.startswith("Volume") else metric_choice
    plot_df = g[["timestamp", ycol]].rename(columns={ycol: "value"})
    line = alt.Chart(plot_df).mark_line(color=ACCENT, strokeWidth=1.6).encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y("value:Q", title=metric_choice),
        tooltip=[alt.Tooltip("timestamp:T", title="Time"),
                 alt.Tooltip("value:Q", title=metric_choice, format=".2f")])
    layers = [line]
    if incidents:
        bands_df = pd.DataFrame([{"start": pd.to_datetime(i.start), "end": pd.to_datetime(i.end),
                                  "Incident": i.title} for i in incidents])
        bands = alt.Chart(bands_df).mark_rect(color="#DC2626", opacity=0.09).encode(
            x="start:T", x2="end:T", tooltip=[alt.Tooltip("Incident:N")])
        layers = [bands, line]
    return _alt_theme(alt.layer(*layers).properties(height=330))


def decline_chart(facts: dict):
    shift = facts.get("decline_reason_shift")
    if not shift:
        return None
    mdf = pd.DataFrame(shift)
    long = mdf.melt(id_vars="decline_reason_code",
                    value_vars=["baseline_share_pct", "during_share_pct"],
                    var_name="period", value_name="share")
    long["period"] = long["period"].map({"baseline_share_pct": "Baseline",
                                          "during_share_pct": "During"})
    bar = alt.Chart(long).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("decline_reason_code:N", title=None, axis=alt.Axis(labelAngle=-35)),
        xOffset="period:N",
        y=alt.Y("share:Q", title="Share of declines %"),
        color=alt.Color("period:N", title=None,
                        scale=alt.Scale(domain=["Baseline", "During"], range=["#B7C0D6", ACCENT]),
                        legend=alt.Legend(orient="top")),
        tooltip=["decline_reason_code", "period", alt.Tooltip("share:Q", format=".1f")]
    ).properties(height=300)
    return _alt_theme(bar)
