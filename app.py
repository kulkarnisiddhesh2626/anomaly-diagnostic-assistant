"""Streamlit POC: Transaction Anomaly Detection & Diagnostic Assistant.

A risk-operations console for a card-issuing bank:
    synthetic data -> statistical + ML detection -> grounded GenAI diagnosis -> chat

Run with:  streamlit run app.py
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src import config, detection, diagnosis, llm_client
from src.context_builder import build_chat_context, incident_context
from src.data_generator import generate_and_save, load_transactions

st.set_page_config(page_title="Issuer Anomaly Console",
                   layout="wide", page_icon="🛰️",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Design system (palette + type) injected as CSS
# --------------------------------------------------------------------------- #
INK = "#0E1726"
ACCENT = "#5B5BD6"
SEV_COLORS = {  # text, soft-bg
    "high": ("#B91C1C", "#FBE7E7"),
    "medium": ("#B45309", "#FBF0DC"),
    "low": ("#1D6F42", "#E7F6ED"),
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

html, body, .stApp, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background:
    radial-gradient(1200px 400px at 80% -120px, #EAE9FB 0%, rgba(234,233,251,0) 60%),
    #F4F6FB; }}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif; letter-spacing:-0.01em; color:{INK}; }}
.block-container {{ padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1300px; }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; height: 0rem; }}

/* ---------- App header band ---------- */
.app-header {{
    background: linear-gradient(135deg, {INK} 0%, #1A2740 100%);
    border-radius: 18px; padding: 22px 26px; margin-bottom: 18px;
    color: #EAF0FB; box-shadow: 0 10px 30px -12px rgba(14,23,38,.45);
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;
}}
.app-header .title {{ font-family:'Space Grotesk'; font-weight:700; font-size:1.55rem; color:#fff; line-height:1.1; }}
.app-header .sub {{ color:#9DB0CC; font-size:.86rem; margin-top:4px; max-width:640px; }}
.app-header .badge-row {{ display:flex; gap:8px; flex-wrap:wrap; }}
.chip {{ display:inline-flex; align-items:center; gap:6px; font-size:.74rem; font-weight:600;
    padding:5px 11px; border-radius:999px; background:rgba(255,255,255,.08);
    color:#CdD8EC; border:1px solid rgba(255,255,255,.12); white-space:nowrap; }}
.chip .dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; }}
.chip.live {{ background:rgba(34,197,94,.14); border-color:rgba(34,197,94,.35); color:#BBF7D0; }}
.chip.off {{ background:rgba(245,180,80,.14); border-color:rgba(245,180,80,.35); color:#FDE9C7; }}

/* ---------- KPI tiles ---------- */
.kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:6px; }}
.kpi {{ flex:1 1 0; min-width:150px; background:#fff; border:1px solid #E6EBF4;
    border-radius:16px; padding:16px 18px; box-shadow:0 1px 2px rgba(14,23,38,.04); }}
.kpi .label {{ font-size:.72rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:#7C8AA5; }}
.kpi .value {{ font-family:'JetBrains Mono'; font-weight:600; font-size:1.85rem; color:{INK}; line-height:1.15; margin-top:6px; }}
.kpi .sub {{ font-size:.78rem; color:#7C8AA5; margin-top:4px; }}
.kpi .sub b.h {{ color:#B91C1C; }} .kpi .sub b.m {{ color:#B45309; }} .kpi .sub b.l {{ color:#1D6F42; }}
.kpi.accent {{ background:linear-gradient(160deg,#EEEEFB,#FFFFFF); border-color:#D9D9F6; }}

/* ---------- Severity badges ---------- */
.sev {{ font-family:'JetBrains Mono'; font-size:.7rem; font-weight:600; letter-spacing:.05em;
    padding:3px 9px; border-radius:7px; text-transform:uppercase; }}

/* ---------- Incident detail card ---------- */
.inc-head {{ background:#fff; border:1px solid #E6EBF4; border-radius:16px; padding:18px 20px; margin-bottom:14px; }}
.inc-head-top {{ display:flex; align-items:center; gap:10px; margin-bottom:8px; }}
.inc-id {{ font-family:'JetBrains Mono'; font-size:.78rem; color:#7C8AA5; }}
.inc-title {{ font-family:'Space Grotesk'; font-weight:600; font-size:1.2rem; color:{INK}; margin-bottom:14px; }}
.inc-meta {{ display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:10px 28px; }}
.inc-meta .k {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.05em; color:#7C8AA5; display:block; }}
.inc-meta .v {{ font-family:'JetBrains Mono'; font-size:.9rem; color:{INK}; }}
.section-label {{ font-size:.72rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
    color:#7C8AA5; margin:6px 0 8px; }}

/* diagnosis prose */
[data-testid="stVerticalBlockBorderWrapper"] strong {{ color:{INK}; }}

/* ---------- Sidebar = dark ink rail ---------- */
[data-testid="stSidebar"] {{ background: linear-gradient(180deg,#0E1726 0%,#13203A 100%); }}
[data-testid="stSidebar"] * {{ color:#D7E0F0; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color:#fff; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color:#AEBCD6; font-size:.82rem; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color:#8194B4; }}
[data-testid="stSidebar"] .stButton button {{ background:{ACCENT}; color:#fff; border:0;
    border-radius:10px; font-weight:600; width:100%; }}
[data-testid="stSidebar"] .stButton button:hover {{ background:#4A47C9; color:#fff; }}
[data-testid="stSidebar"] input {{ color:{INK}; }}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid #E2E8F2; }}
.stTabs [data-baseweb="tab"] {{ font-family:'Space Grotesk'; font-weight:600; font-size:.95rem;
    color:#7C8AA5; padding:8px 14px; }}
.stTabs [aria-selected="true"] {{ color:{INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{ACCENT}; height:3px; border-radius:3px; }}

/* dataframes a touch softer */
[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; border:1px solid #E6EBF4; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data / detection (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load(n_days: int, seed: int) -> pd.DataFrame:
    if not config.TRANSACTIONS_CSV.exists():
        generate_and_save(n_days=n_days, seed=seed)
    return load_transactions()


@st.cache_data(show_spinner=False)
def _detect(_df_token: str, z: float, min_vol: int):
    df = load_transactions()
    events = detection.detect(df, z_threshold=z, min_volume=min_vol)
    incidents = detection.consolidate(events)
    return events, incidents


@st.cache_data(show_spinner=False)
def _chat_context(_token: str):
    df = load_transactions()
    _, incidents = _detect(_token, st.session_state.z, st.session_state.min_vol)
    return build_chat_context(df, incidents)


def _regenerate(n_days: int, seed: int):
    generate_and_save(n_days=n_days, seed=seed)
    st.cache_data.clear()


# --------------------------------------------------------------------------- #
# Small view helpers
# --------------------------------------------------------------------------- #
def sev_badge(sev: str) -> str:
    fg, bg = SEV_COLORS.get(sev, ("#475569", "#EEF1F6"))
    return f'<span class="sev" style="color:{fg};background:{bg};">{sev}</span>'


def scope_label(scope: dict) -> str:
    return " · ".join(f"{k}={v}" for k, v in scope.items()) if scope else "global"


def alt_theme(chart):
    """Apply consistent axis/typography styling to a (possibly layered) chart."""
    return (chart
            .configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor="#ECEFF6", gridDash=[2, 3],
                            domain=False, tickColor="#E2E8F2",
                            labelColor="#7C8AA5", titleColor="#7C8AA5",
                            labelFont="Inter", titleFont="Inter",
                            labelFontSize=11, titleFontSize=11)
            .configure_axisX(grid=False)
            .configure_legend(labelColor="#5A6B85", titleColor="#7C8AA5",
                              labelFont="Inter", titleFont="Inter"))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.markdown("### 🛰️ Console controls")

online = llm_client.is_online()
if online:
    st.sidebar.markdown(
        f'<div class="chip live"><span class="dot" style="background:#22C55E"></span>'
        f'LLM online</div><div style="font-family:JetBrains Mono;font-size:.78rem;'
        f'color:#AEBCD6;margin-top:6px">{llm_client.provider_label()}</div>',
        unsafe_allow_html=True)
else:
    st.sidebar.markdown(
        '<div class="chip off"><span class="dot" style="background:#F5B450"></span>'
        'LLM offline · template mode</div>', unsafe_allow_html=True)
    st.sidebar.caption("Add an API key in .env (Anthropic or Groq) for live "
                       "narratives. Everything still works offline.")

st.sidebar.markdown("#### Data")
n_days = st.sidebar.slider("Days of history", 30, 90, 45, step=5)
seed = st.sidebar.number_input("Random seed", value=7, step=1)
if st.sidebar.button("Regenerate dataset"):
    _regenerate(n_days, int(seed))
    st.rerun()

st.sidebar.markdown("#### Detection sensitivity")
st.session_state.setdefault("z", config.ROBUST_Z_THRESHOLD)
st.session_state.setdefault("min_vol", config.MIN_HOURLY_VOLUME)
st.session_state.z = st.sidebar.slider("z-score threshold", 3.0, 8.0,
                                       float(st.session_state.z), step=0.5)
st.session_state.min_vol = st.sidebar.slider("Min hourly volume", 10, 200,
                                             int(st.session_state.min_vol), step=10)
st.sidebar.caption("Higher z = stricter (fewer, higher-confidence alerts).")

df = _load(n_days, int(seed))
token = f"{len(df)}-{st.session_state.z}-{st.session_state.min_vol}"
events, incidents = _detect(token, st.session_state.z, st.session_state.min_vol)


# --------------------------------------------------------------------------- #
# Header + KPI row
# --------------------------------------------------------------------------- #
llm_chip = (f'<span class="chip live"><span class="dot" style="background:#22C55E"></span>'
            f'LLM · {llm_client.provider_label()}</span>' if online else
            '<span class="chip off"><span class="dot" style="background:#F5B450"></span>'
            'LLM offline (template mode)</span>')
window_chip = (f'<span class="chip"><span class="dot" style="background:#5B5BD6"></span>'
               f'{df.timestamp.min():%d %b} – {df.timestamp.max():%d %b} · {n_days}d hourly</span>')

st.markdown(f"""
<div class="app-header">
  <div>
    <div class="title">🛰️ Issuer Anomaly Console</div>
    <div class="sub">Statistical + Isolation-Forest detection on issuer transaction
    health, with grounded GenAI diagnosis and a conversational analyst assistant.
    Proof of concept on synthetic card-issuer data.</div>
  </div>
  <div class="badge-row">{llm_chip}{window_chip}</div>
</div>
""", unsafe_allow_html=True)

tot = int(df.txn_count.sum())
appr = 100 * df.approved_count.sum() / tot
fraud = 100 * df.fraud_count.sum() / tot
sev_counts = {"high": 0, "medium": 0, "low": 0}
for i in incidents:
    sev_counts[i.severity] = sev_counts.get(i.severity, 0) + 1

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi"><div class="label">Transactions</div>
      <div class="value">{tot/1e6:.2f}M</div><div class="sub">{tot:,} simulated auths</div></div>
  <div class="kpi"><div class="label">Approval rate</div>
      <div class="value">{appr:.1f}%</div><div class="sub">portfolio average</div></div>
  <div class="kpi"><div class="label">Fraud rate</div>
      <div class="value">{fraud:.2f}%</div><div class="sub">of transaction count</div></div>
  <div class="kpi accent"><div class="label">Incidents</div>
      <div class="value">{len(incidents)}</div>
      <div class="sub"><b class="h">{sev_counts['high']} high</b> ·
      <b class="m">{sev_counts['medium']} med</b> ·
      <b class="l">{sev_counts['low']} low</b></div></div>
  <div class="kpi"><div class="label">Granular alerts</div>
      <div class="value">{len(events)}</div><div class="sub">hourly slice flags</div></div>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_incidents, tab_chat = st.tabs(
    ["  Overview  ", "  Incidents & Diagnosis  ", "  Ask the data  "])


# --------------------------------------------------------------------------- #
# Overview tab
# --------------------------------------------------------------------------- #
def _hourly_overall(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("timestamp").agg(
        txn=("txn_count", "sum"), appr=("approved_count", "sum"),
        dec=("declined_count", "sum"), fr=("fraud_count", "sum")).reset_index()
    g["Approval rate %"] = 100 * g.appr / g.txn
    g["Decline rate %"] = 100 * g.dec / g.txn
    g["Fraud rate %"] = 100 * g.fr / g.txn
    return g


with tab_overview:
    g = _hourly_overall(df)
    cleft, cright = st.columns([3, 1])
    with cright:
        metric_choice = st.selectbox(
            "Metric", ["Approval rate %", "Decline rate %", "Fraud rate %",
                       "Volume (txns/hr)"])
    ycol = "txn" if metric_choice.startswith("Volume") else metric_choice
    with cleft:
        st.markdown('<div class="section-label">Portfolio health over time · '
                    'shaded = detected incidents</div>', unsafe_allow_html=True)

    plot_df = g[["timestamp", ycol]].rename(columns={ycol: "value"})
    line = alt.Chart(plot_df).mark_line(color=ACCENT, strokeWidth=1.6).encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y("value:Q", title=metric_choice),
        tooltip=[alt.Tooltip("timestamp:T", title="Time"),
                 alt.Tooltip("value:Q", title=metric_choice, format=".2f")])
    layers = [line]
    if incidents:
        bands_df = pd.DataFrame([{"start": pd.to_datetime(i.start),
                                  "end": pd.to_datetime(i.end),
                                  "Incident": i.title} for i in incidents])
        bands = alt.Chart(bands_df).mark_rect(color="#DC2626", opacity=0.09).encode(
            x="start:T", x2="end:T",
            tooltip=[alt.Tooltip("Incident:N")])
        layers = [bands, line]
    chart = alt.layer(*layers).properties(height=330)
    st.altair_chart(alt_theme(chart), use_container_width=True)

    st.markdown('<div class="section-label">Detected incidents (newest first)</div>',
                unsafe_allow_html=True)
    rows = sorted([e for inc in incidents for e in inc.members[:1]],
                  key=lambda e: e.start, reverse=True)
    if rows:
        f = detection.events_to_frame(rows)
        disp = f[["start", "end", "metric", "grain", "severity", "peak_z",
                  "observed_pct", "expected_pct", "volume",
                  "iso_forest_corroborated"]].rename(columns={
            "start": "Start", "end": "End", "metric": "Metric", "grain": "Scope",
            "severity": "Severity", "peak_z": "Peak z", "observed_pct": "Observed %",
            "expected_pct": "Expected %", "volume": "Volume",
            "iso_forest_corroborated": "IF✓"})
        st.dataframe(disp, use_container_width=True, hide_index=True,
                     column_config={
                         "Peak z": st.column_config.NumberColumn(format="%.1f"),
                         "Observed %": st.column_config.NumberColumn(format="%.1f"),
                         "Expected %": st.column_config.NumberColumn(format="%.1f"),
                         "Volume": st.column_config.NumberColumn(format="%d")})
    else:
        st.info("No incidents at the current sensitivity. Lower the z-threshold "
                "in the sidebar.")


# --------------------------------------------------------------------------- #
# Incidents tab
# --------------------------------------------------------------------------- #
with tab_incidents:
    if not incidents:
        st.info("No incidents at the current sensitivity. Lower the z-threshold "
                "in the sidebar to surface weaker signals.")
    else:
        labels = [f"{i.incident_id} · {i.severity.upper()} · {i.title}"
                  for i in incidents]
        idx = st.selectbox("Select an incident", range(len(incidents)),
                           format_func=lambda k: labels[k])
        inc = incidents[idx]
        facts = incident_context(df, inc)

        st.markdown(f"""
<div class="inc-head">
  <div class="inc-head-top">{sev_badge(inc.severity)}
    <span class="inc-id">{inc.incident_id}</span></div>
  <div class="inc-title">{inc.title}</div>
  <div class="inc-meta">
    <div><span class="k">Window</span><span class="v">{inc.start} → {inc.end}</span></div>
    <div><span class="k">Primary metric</span><span class="v">{inc.primary_metric} · {inc.primary_grain}</span></div>
    <div><span class="k">Peak robust z</span><span class="v">{inc.peak_z:.1f}</span></div>
    <div><span class="k">Scope</span><span class="v">{scope_label(inc.primary_scope)}</span></div>
    <div><span class="k">Slices affected</span><span class="v">{inc.n_grains_affected}</span></div>
    <div><span class="k">Isolation-Forest check</span><span class="v">{'corroborated' if inc.iso_forest_corroborated else 'not corroborated'}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

        col_diag, col_chart = st.columns([3, 2])
        with col_diag:
            src_label = llm_client.provider_label() if online else "offline template"
            st.markdown(f'<div class="section-label">🧠 Diagnostic narrative · '
                        f'{src_label}</div>', unsafe_allow_html=True)
            with st.container(border=True):
                with st.spinner("Generating diagnosis…"):
                    key = (inc.incident_id, token, online)
                    cache = st.session_state.setdefault("_diag_cache", {})
                    if key not in cache:
                        cache[key] = diagnosis.diagnose_incident(df, inc)[0]
                    st.markdown(cache[key])

        with col_chart:
            st.markdown('<div class="section-label">Decline-reason mix shift</div>',
                        unsafe_allow_html=True)
            shift = facts.get("decline_reason_shift")
            if shift:
                mdf = pd.DataFrame(shift)
                long = mdf.melt(id_vars="decline_reason_code",
                                value_vars=["baseline_share_pct", "during_share_pct"],
                                var_name="period", value_name="share")
                long["period"] = long["period"].map(
                    {"baseline_share_pct": "Baseline", "during_share_pct": "During"})
                bar = alt.Chart(long).mark_bar(cornerRadiusTopLeft=3,
                                               cornerRadiusTopRight=3).encode(
                    x=alt.X("decline_reason_code:N", title=None,
                            axis=alt.Axis(labelAngle=-35)),
                    xOffset="period:N",
                    y=alt.Y("share:Q", title="Share of declines %"),
                    color=alt.Color("period:N", title=None,
                                    scale=alt.Scale(domain=["Baseline", "During"],
                                                    range=["#B7C0D6", ACCENT]),
                                    legend=alt.Legend(orient="top")),
                    tooltip=["decline_reason_code", "period",
                             alt.Tooltip("share:Q", format=".1f")]
                ).properties(height=300)
                st.altair_chart(alt_theme(bar), use_container_width=True)
            else:
                st.caption("This incident is volume-driven; no decline-mix shift.")

        with st.expander("🔎 Grounding facts passed to the LLM (the model sees only this — never raw rows)"):
            st.json(facts)


# --------------------------------------------------------------------------- #
# Chat tab
# --------------------------------------------------------------------------- #
with tab_chat:
    st.markdown('<div class="section-label">Grounded analyst assistant</div>',
                unsafe_allow_html=True)
    st.caption("Answers are grounded only in detector outputs and pre-computed "
               "aggregates — e.g. “Why did approval rates drop on the worst day?” "
               "or “Which decline codes drove the spike?”")
    ctx = _chat_context(token)
    st.session_state.setdefault("chat", [])

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask about the transaction data…"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                ans = diagnosis.answer_question(
                    df, incidents, prompt,
                    history=st.session_state.chat[:-1],
                    _cached_context=ctx)
            st.markdown(ans)
        st.session_state.chat.append({"role": "assistant", "content": ans})
