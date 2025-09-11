# server/app_streamlit.py
# ─────────────────────────────────────────────────────────────────────────────
# GV GPT Custom — Demo Middleware (Streamlit)
# File completo e pronto: include ponte sys.path, lettura .env, logger, history,
# streaming token-by-token e system prompt dinamico (se presente).
# ─────────────────────────────────────────────────────────────────────────────

# 0) Ponte: assicura che la root del progetto sia nel PYTHONPATH
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 1) Env & imports di base
import os
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")  # carica .env dalla root (robusto a working dir diverse)

import streamlit as st

# 2) Import dei moduli del progetto
from nlp_layer.preprocessing import analyze_text
from orchestrator.orchestrator import compose_prompt
# system_prompt_for_intent è opzionale: se non c'è, usiamo un fallback
try:
    from orchestrator.orchestrator import system_prompt_for_intent
except Exception:
    def system_prompt_for_intent(intent: str) -> str:
        base = "Rispondi in italiano, chiaro e operativo. Usa elenchi dove utile. "
        if intent == "coding":
            return base + "Se chiedono codice, fornisci snippet minimi e passi di debug."
        if intent == "business":
            return base + "Dai struttura, KPI e passi eseguibili con focus PMI."
        if intent == "nutrition":
            return base + "Ricorda che non sostituisci il medico; cita linee guida generali."
        return base + "Adatta tono al contesto e resta sintetico."

from core.gpt_clienti import call_gpt_chat, stream_gpt_chat  # entrambe supportate
from core.logger import get_logger

# 3) Logger
log = get_logger()

# 4) Streamlit UI setup
st.set_page_config(page_title="GV GPT Custom", page_icon="🤖", layout="centered")
st.title("🤖 GV GPT Custom — Demo Middleware")

# Sidebar (opzionale ma utile)
with st.sidebar:
    st.header("⚙️ Impostazioni")
    model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    st.caption(f"Modello: {model}")
    if st.button("🧹 Svuota chat"):
        st.session_state.history = []
        st.rerun()

# Avviso se manca la chiave
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OPENAI_API_KEY non trovato. Crea un file `.env` nella root del progetto.")

# 5) Stato conversazione
if "history" not in st.session_state:
    st.session_state.history = []   # ciascun item: {"role": "user/assistant", "content": "..."}

# 6) Mostra conversazione già presente
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7) Input utente
if user := st.chat_input("Scrivi qui…"):
    # salva input
    st.session_state.history.append({"role": "user", "content": user})
    log.info(f"USER: {user}")

    # NLP + orchestrator
    nlp_data = analyze_text(user)
    log.info(f"NLP: {nlp_data}")

    # (facoltativo) pannellino di debug
    with st.expander("🔎 NLP insight", expanded=False):
        st.write(nlp_data)

    enriched_user = compose_prompt(user, nlp_data)
    intent = nlp_data.get("intent", "general")
    system = {"role": "system", "content": system_prompt_for_intent(intent)}

    # Costruiamo i messaggi: system + tutta la history esistente (escluso l'ultimo user crudo)
    # + l'utente "arricchito" (prompt orchestrato)
    messages = [system] + st.session_state.history[:-1] + [
        {"role": "user", "content": enriched_user}
    ]

    # 8) Risposta (streaming se possibile)
    with st.chat_message("assistant"):
        try:
            # Stream token-by-token
            reply = st.write_stream(stream_gpt_chat(messages))
        except Exception as e:
            # Fallback non-streaming
            try:
                reply = call_gpt_chat(messages)
                st.markdown(reply)
            except Exception as e2:
                reply = "⚠️ Errore OpenAI: " + str(e2).split("\n")[0]
                st.markdown(reply)

    # salva risposta
    st.session_state.history.append({"role": "assistant", "content": reply})
    log.info(f"ASSISTANT: {str(reply)[:500]}")
