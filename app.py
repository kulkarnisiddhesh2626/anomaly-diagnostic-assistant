"""Streamlit POC: Transaction Anomaly Detection & Diagnostic Assistant.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config, detection, diagnosis, llm_client
from src.context_builder import build_chat_context, incident_context
from src.data_generator import generate_and_save, load_transactions

st.set_page_config(page_title="Card Anomaly Diagnostic Assistant",
                   layout="wide", page_icon="💳")


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
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Controls")

online = llm_client.is_online()
st.sidebar.markdown(
    f"**LLM status:** {'🟢 online (' + config.DEFAULT_MODEL + ')' if online else '🟡 offline (template mode)'}")
if not online:
    st.sidebar.caption("Set ANTHROPIC_API_KEY (and `pip install anthropic`) for "
                       "LLM-generated narratives. Everything still works offline.")

st.sidebar.subheader("Data")
n_days = st.sidebar.slider("Days of history", 30, 90, 45, step=5)
seed = st.sidebar.number_input("Random seed", value=7, step=1)
if st.sidebar.button("🔄 Regenerate dataset"):
    _regenerate(n_days, int(seed))
    st.rerun()

st.sidebar.subheader("Detection sensitivity")
st.session_state.setdefault("z", config.ROBUST_Z_THRESHOLD)
st.session_state.setdefault("min_vol", config.MIN_HOURLY_VOLUME)
st.session_state.z = st.sidebar.slider("z-score threshold", 3.0, 8.0,
                                       float(st.session_state.z), step=0.5)
st.session_state.min_vol = st.sidebar.slider("Min hourly volume", 10, 200,
                                             int(st.session_state.min_vol), step=10)

df = _load(n_days, int(seed))
token = f"{len(df)}-{st.session_state.z}-{st.session_state.min_vol}"
events, incidents = _detect(token, st.session_state.z, st.session_state.min_vol)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("💳 Transaction Anomaly Detection & Diagnostic Assistant")
st.caption("Detection layer (statistical + Isolation Forest) → grounded GenAI "
           "diagnosis → conversational Q&A. POC on synthetic issuer data.")

c1, c2, c3, c4 = st.columns(4)
tot = int(df.txn_count.sum())
c1.metric("Transactions", f"{tot:,}")
c2.metric("Approval rate", f"{100*df.approved_count.sum()/tot:.1f}%")
c3.metric("Incidents found", len(incidents))
c4.metric("Granular alerts", len(events))

tab_overview, tab_incidents, tab_chat = st.tabs(
    ["📈 Overview", "🚨 Incidents & Diagnosis", "💬 Ask the data"])


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
    metric_choice = st.selectbox(
        "Metric", ["Approval rate %", "Decline rate %", "Fraud rate %", "Volume (txns/hr)"])
    ycol = "txn" if metric_choice.startswith("Volume") else metric_choice

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(g.timestamp, g[ycol], lw=0.9, color="#2563eb")
    for inc in incidents:
        ax.axvspan(pd.to_datetime(inc.start), pd.to_datetime(inc.end),
                   color="#ef4444", alpha=0.12)
    ax.set_title(metric_choice + "  (red bands = detected incidents)")
    ax.set_ylabel(metric_choice)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)

    st.markdown("**Detected incidents (newest first):**")
    st.dataframe(detection.events_to_frame(
        sorted([e for inc in incidents for e in inc.members[:1]],
               key=lambda e: e.start, reverse=True))[
        ["start", "end", "metric", "grain", "severity", "peak_z",
         "observed_pct", "expected_pct", "volume", "iso_forest_corroborated"]],
        use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Incidents tab
# --------------------------------------------------------------------------- #
with tab_incidents:
    if not incidents:
        st.info("No incidents at the current sensitivity. Lower the z-threshold.")
    else:
        labels = [f"{i.incident_id} · {i.severity.upper()} · {i.title} · "
                  f"{i.start[:16]}" for i in incidents]
        idx = st.selectbox("Select an incident", range(len(incidents)),
                           format_func=lambda k: labels[k])
        inc = incidents[idx]

        left, right = st.columns([2, 3])
        with left:
            st.subheader(inc.title)
            st.markdown(
                f"- **ID:** {inc.incident_id}\n"
                f"- **Severity:** {inc.severity}\n"
                f"- **Window:** {inc.start} → {inc.end}\n"
                f"- **Primary metric:** {inc.primary_metric} ({inc.primary_grain})\n"
                f"- **Robust z:** {inc.peak_z}\n"
                f"- **Slices affected:** {inc.n_grains_affected}\n"
                f"- **Isolation Forest corroborated:** "
                f"{'yes' if inc.iso_forest_corroborated else 'no'}")
            facts = incident_context(df, inc)
            with st.expander("🔎 Grounding facts passed to the LLM (JSON)"):
                st.json(facts)

        with right:
            st.subheader("🧠 Diagnostic narrative")
            with st.spinner("Generating diagnosis…"):
                key = (inc.incident_id, token, online)
                cache = st.session_state.setdefault("_diag_cache", {})
                if key not in cache:
                    cache[key] = diagnosis.diagnose_incident(df, inc)[0]
                st.markdown(cache[key])

        # decline-reason breakdown chart for the window
        st.markdown("**Decline-reason shift during the incident**")
        shift = facts["decline_reason_shift"]
        if shift:
            sdf = pd.DataFrame(shift).set_index("decline_reason_code")[
                ["baseline_share_pct", "during_share_pct"]]
            st.bar_chart(sdf)


# --------------------------------------------------------------------------- #
# Chat tab
# --------------------------------------------------------------------------- #
with tab_chat:
    st.markdown("Ask grounded questions, e.g. *“Why did approval rates drop on "
                "the worst day?”* or *“Which decline reason codes drove the spike?”*")
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
