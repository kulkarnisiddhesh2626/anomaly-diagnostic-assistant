"""Overview page: portfolio health, the detection pipeline, and recent incidents."""
import streamlit as st

import appkit

ctx = appkit.load_context()
appkit.topbar(ctx)
appkit.page_title("Overview",
                  "Portfolio transaction health and the detection pipeline at a glance.")

if not ctx["incidents"]:
    st.markdown('<div class="callout">No incidents at the current sensitivity. '
                'Lower the <b>z-score threshold</b> in the sidebar to surface weaker '
                'signals.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="callout">This is a proof of concept on <b>synthetic</b> '
                'issuer data with deliberately injected anomalies. Start here, then open '
                '<b>Incidents</b> to see each one diagnosed, or <b>Assistant</b> to ask '
                'questions.</div>', unsafe_allow_html=True)

appkit.kpi_row(ctx)
appkit.pipeline_steps(ctx)

left, right = st.columns([3, 1])
with right:
    metric_choice = st.selectbox("Metric", ["Approval rate %", "Decline rate %",
                                            "Fraud rate %", "Volume (txns/hr)"])
with left:
    st.markdown('<div class="section-label">Portfolio health over time · '
                'shaded = detected incidents</div>', unsafe_allow_html=True)

st.altair_chart(appkit.health_chart(ctx["df"], ctx["incidents"], metric_choice),
                use_container_width=True)
with st.expander("How to read this chart"):
    st.markdown("Each red band is a window the detector flagged as anomalous. Switch the "
                "metric (top-right) to see approval, decline, or fraud rates, or raw "
                "volume. Hover any point for exact values. The detector tests each metric "
                "against its own seasonal baseline, so a band means *unusual for that hour "
                "and weekday* — not merely high or low.")

st.markdown('<div class="section-label">Recent incidents</div>', unsafe_allow_html=True)
recent = sorted(ctx["incidents"], key=lambda i: i.start, reverse=True)[:5]
for i in recent:
    st.markdown(
        f'<div class="inc-head" style="padding:12px 16px;margin-bottom:8px;">'
        f'<div class="inc-head-top">{appkit.sev_badge(i.severity)}'
        f'<span class="inc-id">{i.incident_id}</span>'
        f'<span style="color:#0E1726;font-weight:600;">{i.title}</span></div>'
        f'<div style="font-size:.82rem;color:#7C8AA5;">{i.start} → {i.end} · '
        f'peak z {i.peak_z:.1f} · {i.primary_metric}</div></div>',
        unsafe_allow_html=True)
st.caption("Open the **Incidents** page for the full diagnosis of any incident.")
