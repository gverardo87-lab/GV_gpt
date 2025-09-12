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
from core.memory import load_memory, save_memory, clear_memory
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
import json
from datetime import datetime

def export_chat_md(history: list) -> str:
    lines = ["# Conversazione GV GPT\n"]
    for msg in history:
        role = "Tu" if msg["role"] == "user" else "GV"
        lines.append(f"**{role}:** {msg['content']}\n")
    return "\n".join(lines)

def export_chat_json(history: list) -> str:
    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "model": os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        "messages": history,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

# 3) Logger
log = get_logger()

# 4) Streamlit UI setup
st.set_page_config(page_title="GV GPT Custom", page_icon="🤖", layout="centered")
st.title("🤖 GV GPT Custom — la AI di Giacomino")

# Sidebar (opzionale ma utile)
with st.sidebar:
    st.header("⚙️ Impostazioni")
    model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    st.caption(f"Modello: {model}")
    if st.button("🧹 Svuota chat"):
        st.session_state.history = []
        st.rerun()
    st.header("🧠 Memoria")
    persist = st.checkbox(
        "Mantieni chat tra riavvii",
        value=True,
        help="Salva gli ultimi messaggi su disco (data/memory_default.json)."
    )
    if st.button("🗑️ Cancella memoria salvata"):
        clear_memory()
        st.success("Memoria persistente cancellata.")
    st.header("📤 Export")
    has_chat = bool(st.session_state.get("history"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Markdown .md",
            data=export_chat_md(st.session_state.history if has_chat else []),
            file_name=f"chat_{ts}.md",
            mime="text/markdown",
            disabled=not has_chat
        )
    with col2:
        st.download_button(
            "JSON .json",
            data=export_chat_json(st.session_state.history if has_chat else []),
            file_name=f"chat_{ts}.json",
            mime="application/json",
            disabled=not has_chat
        )

# Avviso se manca la chiave
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OPENAI_API_KEY non trovato. Crea un file `.env` nella root del progetto.")

# 5) Stato conversazione (usa memoria persistente se presente)
if "history" not in st.session_state:
    st.session_state.history = load_memory()   # ciascun item: {"role": "user/assistant", "content": "..."}

# 6) Mostra conversazione già presente
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7) Input utente
if user := st.chat_input("Scrivi qui…"):
    # salva input
    st.session_state.history.append({"role": "user", "content": user})
    log.info(f"USER: {user}")
    # salva memoria persistente se attiva
    if persist:
        save_memory(st.session_state.history)

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
