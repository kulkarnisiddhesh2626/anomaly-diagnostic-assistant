"""Incidents page: select an incident, see its grounded diagnosis and evidence."""
import streamlit as st

import appkit
from appkit import diagnosis, incident_context, llm_client

ctx = appkit.load_context()
appkit.topbar(ctx)
appkit.page_title("Incidents & Diagnosis",
                  "Each detected incident, explained in plain language and grounded "
                  "strictly in detector facts.")

incidents = ctx["incidents"]
if not incidents:
    st.markdown('<div class="callout">No incidents at the current sensitivity. '
                'Lower the <b>z-score threshold</b> in the sidebar.</div>',
                unsafe_allow_html=True)
    st.stop()

labels = [f"{i.incident_id} · {i.severity.upper()} · {i.title}" for i in incidents]
idx = st.selectbox("Select an incident", range(len(incidents)),
                   format_func=lambda k: labels[k])
inc = incidents[idx]
df = ctx["df"]
facts = incident_context(df, inc)

st.markdown(f"""
<div class="inc-head">
  <div class="inc-head-top">{appkit.sev_badge(inc.severity)}
    <span class="inc-id">{inc.incident_id}</span></div>
  <div class="inc-title">{inc.title}</div>
  <div class="inc-meta">
    <div><span class="k">Window</span><span class="v">{inc.start} → {inc.end}</span></div>
    <div><span class="k">Primary metric</span><span class="v">{inc.primary_metric} · {inc.primary_grain}</span></div>
    <div><span class="k">Peak robust z</span><span class="v">{inc.peak_z:.1f}</span></div>
    <div><span class="k">Scope</span><span class="v">{appkit.scope_label(inc.primary_scope)}</span></div>
    <div><span class="k">Slices affected</span><span class="v">{inc.n_grains_affected}</span></div>
    <div><span class="k">Isolation-Forest check</span><span class="v">{'corroborated' if inc.iso_forest_corroborated else 'not corroborated'}</span></div>
  </div>
</div>""", unsafe_allow_html=True)

col_diag, col_chart = st.columns([3, 2])
with col_diag:
    src = llm_client.provider_label() if ctx["online"] else "offline template"
    st.markdown(f'<div class="section-label">🧠 Diagnostic narrative · {src}</div>',
                unsafe_allow_html=True)
    with st.container(border=True):
        with st.spinner("Generating diagnosis…"):
            key = (inc.incident_id, ctx["token"], ctx["online"])
            cache = st.session_state.setdefault("_diag_cache", {})
            if key not in cache:
                cache[key] = diagnosis.diagnose_incident(df, inc)[0]
            st.markdown(cache[key])

with col_chart:
    st.markdown('<div class="section-label">Decline-reason mix shift</div>',
                unsafe_allow_html=True)
    chart = appkit.decline_chart(facts)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("This incident is volume-driven; no decline-mix shift to show.")

with st.expander("🔎 Grounding facts passed to the LLM "
                 "(the model sees only this — never raw transactions)"):
    st.json(facts)
