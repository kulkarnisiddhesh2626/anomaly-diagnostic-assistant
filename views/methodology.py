"""How it works: method, design choices, and how the POC maps to the brief."""
import streamlit as st

import appkit

ctx = appkit.load_context()
appkit.topbar(ctx)
appkit.page_title("How it works",
                  "The approach, the detection method, the GenAI grounding strategy, "
                  "and how this POC maps to the assessment criteria.")

st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
st.markdown(
    "The system keeps a hard wall between two layers. The **detection layer** answers "
    "*what moved, where, and how much* using explainable statistics. The **GenAI layer** "
    "answers *what it means and what to do* — and it only ever consumes the detector's "
    "structured output, never raw transactions. That separation is the foundation of the "
    "hallucination control: a model handed only verified numbers has nothing to invent.")
st.code(
    "synthetic data  →  detection (stats + Isolation Forest)  →  fact sheet  →\n"
    "                    grounded LLM diagnosis  →  conversational Q&A", language="text")

c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="section-label">Detection method</div>', unsafe_allow_html=True)
    st.markdown(
        "- **Rate metrics** (approval / decline / fraud) use a **two-proportion z-test** "
        "against a pooled **seasonal** baseline (same weekday × hour). The volume term in "
        "the denominator means sparse, low-volume noise scores low — directly targeting "
        "the low-false-positive requirement.\n"
        "- **Volume** uses a **Poisson** deviation against its seasonal mean.\n"
        "- **Effect-size floors** suppress changes that are statistically significant but "
        "operationally trivial.\n"
        "- An **Isolation Forest** runs as a multivariate **cross-check only** — it can "
        "corroborate but never solely trigger an alert, so everything stays explainable.\n"
        "- Flagged hours merge into **events**; overlapping events consolidate into "
        "**incidents** — one operational story instead of hundreds of hourly alerts.")
with c2:
    st.markdown('<div class="section-label">GenAI design</div>', unsafe_allow_html=True)
    st.markdown(
        "- The LLM receives a compact **fact sheet** per incident (before/during/after "
        "values, decline-mix shift, the dimension that localises it, a code glossary) — "
        "visible on the Incidents page under *Grounding facts*.\n"
        "- A fixed four-section narrative (**what / where / probable cause / next steps**) "
        "at low temperature keeps output skimmable and repeatable.\n"
        "- **Hallucination control** is layered: architectural (no raw data), instructional "
        "(facts are the only source of truth; say so when data is insufficient), and "
        "parametric (low temperature).\n"
        "- **Provider-pluggable**: Anthropic or Groq via one env var; an offline template "
        "path keeps the whole flow working with no key at all.")

st.markdown('<div class="section-label">How this maps to the assessment</div>',
            unsafe_allow_html=True)
st.table({
    "Criterion": ["Problem framing", "Detection soundness", "GenAI grounding & prompts",
                  "Engineering quality", "Communication"],
    "Where to see it": [
        "Overview pipeline + this page",
        "Detection method above; seasonal, volume-aware, explainable; Isolation-Forest cross-check",
        "Incidents → Grounding facts JSON; fact-only prompting; offline determinism",
        "Modular src/ layers, shared appkit, cached loaders, provider abstraction",
        "Multi-page UI, in-app methodology, README + WRITEUP in the repo"],
})

st.caption("Injected anomalies in the synthetic data: issuer/processor outage, "
           "card-testing fraud attack, 3DS authentication failure, cross-border approval "
           "drop, and a gambling-MCC volume surge. The current detector recovers all five.")
