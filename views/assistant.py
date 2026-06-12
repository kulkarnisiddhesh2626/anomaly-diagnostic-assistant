"""Assistant page: grounded conversational Q&A with suggested starters."""
import streamlit as st

import appkit
from appkit import diagnosis

ctx = appkit.load_context()
appkit.topbar(ctx)
appkit.page_title("Analyst Assistant",
                  "Ask follow-up questions in plain language. Answers are grounded only "
                  "in detector outputs and pre-computed aggregates — never raw rows.")

chat_ctx = appkit._chat_context(ctx["token"], st.session_state.z, st.session_state.min_vol)
st.session_state.setdefault("chat", [])

# Suggested-question chips (only before the first message, to guide the user).
if not st.session_state.chat:
    st.markdown('<div class="section-label">Try asking</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for n, q in enumerate(appkit.SUGGESTED_QUESTIONS):
        if cols[n % 2].button(q, key=f"sugg_{n}", use_container_width=True):
            st.session_state.pending_q = q
            st.rerun()

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

typed = st.chat_input("Ask about the transaction data…")
prompt = st.session_state.pop("pending_q", None) or typed

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ans = diagnosis.answer_question(
                ctx["df"], ctx["incidents"], prompt,
                history=st.session_state.chat[:-1], _cached_context=chat_ctx)
        st.markdown(ans)
    st.session_state.chat.append({"role": "assistant", "content": ans})

if st.session_state.chat:
    if st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()
