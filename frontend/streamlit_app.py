"""Streamlit chat UI; all retrieval and generation are performed by FastAPI."""
import requests
import streamlit as st

from config import API_URL

st.set_page_config(page_title="Customer Support FAQ", page_icon="💬", layout="centered")
st.title("Customer Support Assistant")
st.caption("Answers use a static FAQ knowledge base. This assistant cannot access live order, shipment, payment, or account records.")

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        st.caption(f"Outcome: {item['outcome']} · Retries: {item['retry_count']} · {item['latency_ms']:.1f} ms")
        with st.expander("Sources and evidence"):
            if not item["sources"]:
                st.write("No FAQ passages were used.")
            for source in item["sources"]:
                st.markdown(f"**{source.get('category') or 'FAQ'} / {source.get('intent') or 'general'}** · "
                            f"record `{source.get('record_id') or 'n/a'}` · score `{source.get('score')}`")
                st.write(source.get("passage", ""))

with st.expander("Try a sample question"):
    samples = ["How do I cancel my order?", "How can I change my shipping address?",
               "What should I do if my payment was charged twice?", "What is the return policy?"]
    cols = st.columns(2)
    for i, sample in enumerate(samples):
        if cols[i % 2].button(sample, key=f"sample_{i}"):
            st.session_state.pending_question = sample

with st.form("ask_form", clear_on_submit=True):
    question = st.text_input("How can we help?", max_chars=1000, value=st.session_state.pop("pending_question", ""))
    submitted = st.form_submit_button("Send")

if submitted:
    if not question.strip():
        st.warning("Enter a question before sending.")
    elif len(question) > 1000:
        st.warning("Questions must be 1000 characters or fewer.")
    else:
        try:
            with st.spinner("Checking the FAQ…"):
                response = requests.post(f"{API_URL.rstrip('/')}/ask", json={"question": question.strip()}, timeout=40)
            if response.status_code == 429:
                st.error("The assistant is busy. Please wait a moment and try again.")
            elif response.status_code >= 400:
                detail = response.json().get("detail", "The service could not answer right now.")
                st.error(detail)
            else:
                result = response.json()
                st.session_state.history.append({"question": question.strip(), **result})
                st.rerun()
        except requests.RequestException:
            st.error(f"Can't reach the API at {API_URL}. Start FastAPI and try again.")
