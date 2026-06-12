"""Entry point for the Issuer Anomaly Console (multi-page Streamlit app).

Run with:  streamlit run app.py

Wires the four pages in views/ into a navigation rail and draws the shared
chrome (theme, sidebar status + controls) once per run. All shared logic lives
in appkit.py so each page stays small.
"""
from __future__ import annotations

import streamlit as st

import appkit

st.set_page_config(page_title="Issuer Anomaly Console", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

appkit.inject_css()

nav = st.navigation([
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/incidents.py", title="Incidents", icon=":material/warning:"),
    st.Page("views/assistant.py", title="Assistant", icon=":material/chat:"),
    st.Page("views/methodology.py", title="How it works", icon=":material/help:"),
])

# Shared sidebar (status + global controls + help) appears under the nav.
appkit.render_sidebar()

nav.run()
