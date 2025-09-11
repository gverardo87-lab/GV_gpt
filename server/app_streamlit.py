# server/app_streamlit.py — Interfaccia grafica minimale (Streamlit)
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
from nlp_layer.preprocessing import analyze_text
from orchestrator.orchestrator import compose_prompt
from core.gpt_clienti import call_gpt
import os

st.set_page_config(page_title="GV GPT Custom", page_icon="🤖", layout="centered")
st.title("🤖 GV GPT Custom — Demo Middleware")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OPENAI_API_KEY non trovato. Crea .env nella root e riavvia.")

if "history" not in st.session_state:
    st.session_state.history = []  # [(role, content)]

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

if user := st.chat_input("Scrivi qui…"):
    st.session_state.history.append(("user", user))

    nlp_data = analyze_text(user)
    prompt = compose_prompt(user, nlp_data)
    try:
        reply = call_gpt(prompt)
    except Exception as e:
        reply = f"Errore chiamando OpenAI: {e}"

    st.session_state.history.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)
