# server/app_streamlit.py
# ─────────────────────────────────────────────────────────────────────────────
# GV_GPT — L’aria sta cambiando (Streamlit, tema nautico chiaro)
# - Engine switch: OpenAI ↔ HuggingFace ↔ Ollama
# - 🔒 Engine Lock + Fallback smart (se lock OFF): OpenAI → HuggingFace → Ollama
# - Didattica opzionale (solo OpenAI/HF)
# - Streaming: OpenAI/Ollama; HuggingFace pseudo-stream (tutto in un colpo)
# - Memoria persistente, export, diagnostica, slider num_predict e temperatura per Ollama
# - Toggle “Alta leggibilità”
# - Sanitizer anti meta-marker + filtri anti drift
# - Pannello “🖊️ Correzione intent”
# ─────────────────────────────────────────────────────────────────────────────

# 0) Ponte: assicura che la root del progetto sia nel PYTHONPATH
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 1) Env & imports base
import os
import json
import time
import re
import difflib
import hashlib  # ← PATCH: serve per _intent_key
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# --- Boot Guard (anti-mix progetti) ------------------------------------------
try:
    import core.boot_guard  # noqa: F401
except Exception:
    EXPECTED = os.getenv("GV_EXPECTED_PROJECT", "GV_EXT")
    PID = os.getenv("PROJECT_ID")
    if PID and PID != EXPECTED:
        raise SystemExit(f"[ABORT] Wrong PROJECT_ID. Expected {EXPECTED}, got {PID!r}.")

# Default Ollama model se non in .env (ADEGUATO al Modelfile creato)
if not os.getenv("OLLAMA_MODEL"):
    os.environ["OLLAMA_MODEL"] = "gv/phi35-mini-gv:latest"

import streamlit as st
import requests  # Tenuto per compat
from PIL import Image

# 2) Import moduli progetto
from core.engine import (
    call_chat, stream_chat, call_chat_smart, build_continue_only_messages
)
from core.memory import load_memory, save_memory, clear_memory

# Ripulisci memoria all'avvio se richiesto da env (utile in demo o dopo bug)
if (os.getenv("GV_BOOT_CLEAR_MEMORY", "0").lower() in ("1","true","yes","on")):
    clear_memory()

from core.logger import get_logger
log = get_logger()
log.info(f"INIT: persist={st.session_state.get('persist', True)} "
         f"loaded_history={len(st.session_state.get('history', []))}")

# NLP/Orchestrator – fallback soft se mancanti
try:
    from nlp_layer.preprocessing import analyze_text
except Exception:
    def analyze_text(txt: str):
        return {"intent": "general", "entities": [], "lang": "it"}

try:
    from orchestrator.orchestrator import compose_prompt, system_prompt_for_intent
except Exception:
    def compose_prompt(user_text: str, nlp: dict) -> str:
        return user_text
    def system_prompt_for_intent(intent: str) -> str:
        base = "Rispondi in italiano, chiaro e operativo. Usa elenchi dove utile. "
        if intent == "coding":
            return base + "Se chiedono codice, fornisci snippet minimi e passi di debug."
        if intent == "business":
            return base + "Dai struttura, KPI e passi eseguibili con focus PMI."
        if intent == "nutrition":
            return base + "Ricorda che non sostituisci il medico; cita linee guida generali."
        return base + "Adatta tono al contesto e resta sintetico."

# Diagnostica Ollama (/api/generate legacy)
try:
    from core.ollama_client import call_ollama_generate
except Exception:
    def call_ollama_generate(*args, **kwargs):
        raise RuntimeError("Diagnostica Ollama non disponibile (core/ollama_client.py mancante).")

log = get_logger()

# --- Sanitizer / Post-processor anti drift -----------------------------------
META_RX = re.compile(
    r"(?is)(<\|[^>]*\|>"
    r"|\[(?:INTENT|EXPLICIT[\s_-]*ACTION[\s_-]*LIST)[^\]]*\]"
    r"|\[[A-Z][A-Z0-9 _-]{2,}:[^\]]*\]"
    r"|\[[A-Z][A-Z0-9 _-]{2,}\])"
)
NLP_INSIGHT_RX = re.compile(r"(?im)^\s*🔎\s*NLP\s*insight\s*$")

def _sanitize_meta(s: str) -> str:
    if not s:
        return s
    return META_RX.sub("", s).strip()

# ▶︎ PATCH 3: estensione sanitizer per rimuovere anche rumore tipo “🔎 NLP insight”
def _sanitize_meta_and_noise(s: str) -> str:
    if not s:
        return s
    s = META_RX.sub("", s)
    s = NLP_INSIGHT_RX.sub("", s)
    return s.strip()

DRIFT_PREFIX_RX = re.compile(r"(?is)^\s*(instruction[s]?:.*?\n+|\s*i['’]m\s+sorry[^.\n]*[.\n]+\s*)")
def _strip_drift_prefix(text: str) -> str:
    if not text:
        return text
    out = DRIFT_PREFIX_RX.sub("", text).lstrip()
    return out if out else text

DRIFT_LINES_RX = re.compile(r"(?im)^\s*(your\s+task|instruction|begin\s+by)\s*:\s.*$")
def _strip_drift_lines(text: str) -> str:
    if not text:
        return text
    return DRIFT_LINES_RX.sub("", text)

# ▶︎ PATCH 1: Sentinel di chiusura (coerente con Modelfile)
OLLAMA_SENTINEL = "[[END_OF_OUTPUT]]"
SENTINELS = ("[[END_OF_OUTPUT]]", "<<FINE>>")  # ← supporto multiplo

# ==== Export helpers ==========================================================
def export_chat_md(history: list) -> str:
    lines = ["# Conversazione GV_GPT\n"]
    for msg in history:
        role = "Tu" if msg["role"] == "user" else "GV"
        lines.append(f"**{role}:** {msg['content']}\n")
    return "\n".join(lines)

def export_chat_json(history: list) -> str:
    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "engine": os.getenv("GV_ENGINE", "openai"),
        "openai_model": os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        "ollama_model": os.getenv("OLLAMA_MODEL") or "",
        "hugging_model": os.getenv("HUGGINGFACE_MODEL") or "",
        "messages": history,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

# === Utils ===================================================================
def _tail(text: str, chars: int = 1800) -> str:
    return (text or "")[-chars:]

def _word_count(s: str) -> int:
    return len(re.findall(r"\w+", s or ""))

def _looks_restart(chunk: str) -> bool:
    head = (chunk or "").strip().lower()[:400]
    patterns = [r"^introduzione\b", r"^capitolo\s+\d+\b", r"^in\s+questa\s+risposta\b",
                r"^la\s+storia\s+di\b", r"^prefazione\b"]
    return any(re.search(p, head) for p in patterns)

def _is_reduant(chunk: str, acc: str) -> bool:
    tail = _tail(acc, 1200).lower()
    c = (chunk or "").lower()
    if not tail or not c:
        return False
    c_words = set(re.findall(r"\w+", c)[:200])
    t_words = set(re.findall(r"\w+", tail))
    if not c_words or not t_words:
        return False
    overlap = len(c_words & t_words) / max(1, len(c_words | t_words))
    return overlap > 0.55 or c[:150] in tail

def _too_similar(a: str, b: str, threshold: float = 0.86) -> bool:
    if not a or not b:
        return False
    ratio = difflib.SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()
    return ratio >= threshold

# ── PATCH: continue command detection & robust last-turn finder ──────────────
CONT_RX = re.compile(
    r"^(?:continua|prosegui|vai avanti|ancora|dammi(?:\s+di)?\s+pi[uù]|continua\s*pure)\s*[.!?]*$",
    re.IGNORECASE
)
def _is_continue_cmd(s: str) -> bool:
    return bool(CONT_RX.match((s or "").strip()))

def _find_last_user_and_assistant(history: list) -> tuple[str, str]:
    """
    Cerca l'ultimo assistant e il relativo ultimo user NON di tipo 'continua'.
    Evita che i comandi di continue si 'incollino' a turni vecchi.
    """
    last_assistant = ""
    last_user = ""
    skipping_trailing_continue = True
    for m in reversed(history):
        role = m.get("role")
        content = m.get("content", "")
        if skipping_trailing_continue and role == "user" and _is_continue_cmd(content):
            continue
        skipping_trailing_continue = False
        if not last_assistant and role == "assistant":
            last_assistant = content
        elif role == "user":
            last_user = content
            break
    return last_user, last_assistant
# ─────────────────────────────────────────────────────────────────────────────

def _looks_cutoff(txt: str) -> bool:
    if not txt:
        return False
    return not re.search(r'[.!?…]"?\s*\Z', txt.strip())

def _short_history_by_chars(history_msgs: list, max_chars: int = 9000) -> list:
    sel = []
    total = 0
    for m in reversed(history_msgs):
        if m.get("role") == "system":
            continue
        c = m.get("content") or ""
        total += len(c)
        sel.append(m)
        if total >= max_chars:
            break
    return list(reversed(sel))

# === Post-formatter di impaginazione (micro) =================================
def post_format_response(text: str) -> str:
    """
    Correzioni leggere:
    - chiude blocchi ``` dispari
    - rimuove link placeholder [titolo]()
    - normalizza heading profondi a '## '
    - bullet coerenti ('- ')
    - sopprime boilerplate inglese in righe isolate
    - normalizza spazi finali e ritorni multipli
    """
    if not text:
        return text
    s = text

    # Chiudi blocchi ``` dispari
    if s.count("```") % 2 == 1:
        s += "\n```"

    # Togli link placeholder [titolo]()
    s = re.sub(r"\[([^\]]+)\]\(\s*\)", r"\1", s)

    # Normalizza heading troppo profondi a "## "
    s = re.sub(r"^\s*#{4,}\s*", "## ", s, flags=re.MULTILINE)

    # Normalizza bullet (• o * → "- ")
    s = re.sub(r"^[\t ]*[•*]\s+", "- ", s, flags=re.MULTILINE)

    # Sopprimi righe boilerplate in inglese comuni
    s = re.sub(r"(?im)^\s*(ready to help|here (?:are|is)|let'?s |i can help)\b.*$", "", s)

    # Spazi/punteggiatura
    s = re.sub(r"[ \t]+$", "", s, flags=re.MULTILINE)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()

# === THEME (nautico chiaro) & GLOBAL CSS ====================================
def _nautical_css(pro_mode: bool = False) -> str:
    if pro_mode:
        return """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700&family=Inter:wght@400;600&display=swap');
          :root{
            --bg-top: #ffffff;
            --bg-bottom: #f7fbff;
            --fg: #0b1220;
            --fg-muted: #273043;
            --border: #d6e3f8;
            --card: #ffffff;
            --bubble: #f7faff;
            --link: #0e7ddf;
            --accent: #1e4ed8;
            --input-bg: #ffffff;
            --placeholder: #50627a;
          }
          [data-testid="stAppViewContainer"]{
            background: linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
          }
          [data-testid="stAppViewContainer"] .main .block-container{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: 0 4px 20px rgba(15, 23, 42, .04);
            padding: .6rem .8rem .9rem .8rem;
          }
          .main .block-container, .main .block-container p, .main .block-container li, 
          .main .block-container label, .main .block-container h1, .main .block-container h2, .main .block-container h3{
            color: var(--fg);
            font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            font-size: 0.95rem;
            line-height: 1.45;
          }
          .main .block-container h1 { font-size: 1.25rem; margin: .35rem 0 .25rem; }
          .main .block-container h2 { font-size: 1.1rem; margin: .35rem 0 .2rem; }
          .main .block-container h3 { font-size: 1.0rem; margin: .3rem 0 .15rem; }
          .brand-title{
            font-family: 'Plus Jakarta Sans', Inter, system-ui;
            font-weight: 700; letter-spacing: .1px;
            font-size: clamp(18px, 2.4vw, 26px);
            color: var(--fg);
          }
          .brand-sub{ color: var(--fg-muted); font-size: 12px; margin-top: .2rem; }
          .hero-card{
            border-radius: 12px; padding: 10px 12px; border: 1px solid var(--border);
            background: #fffffff6;
          }
          .side-card{
            border-radius: 12px; padding: 8px; border: 1px solid var(--border); background: #ffffff;
            display:flex;align-items:center;justify-content:center; aspect-ratio: 1.8/1;
          }
          [data-testid="stChatMessage"] > div:first-child{
            border-radius: 10px !important; border: 1px solid var(--border); background: var(--bubble);
            padding: 8px 10px !important;
          }
          [data-testid="stChatInput"] textarea{
            background: var(--input-bg) !important; border: 1px solid var(--border) !important;
            color: var(--fg) !important; caret-color: var(--accent) !important;
            font-size: 0.95rem !important;
          }
          [data-testid="stChatInput"] textarea::placeholder{ color: var(--placeholder) !important; opacity: 1 !important; }
          .stTextInput input, .stTextArea textarea{
            background: var(--input-bg) !important; border: 1px solid var(--border) !important; color: var(--fg) !important;
            font-size: 0.95rem !important;
          }
          .stTextInput input::placeholder, .stTextArea textarea::placeholder{ color: var(--placeholder) !important; opacity: 1 !important; }
          .main .block-container a{ color: var(--link); text-decoration: none; }
          .wave-wrap{height: 26px; overflow: hidden; margin-top: 2px;}
          [data-testid="stMarkdownContainer"] p { margin: .25rem 0; }
          [data-testid="stMarkdownContainer"] ul { margin: .25rem 0 .3rem 1.1rem; }
          [data-testid="stMarkdownContainer"] li { margin: .05rem 0; }
          [data-testid="stChatMessage"] p, [data-testid="stMarkdownContainer"] p { overflow-wrap: break-word; white-space: pre-wrap; }
        </style>
        """
    else:
        return """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700&family=Inter:wght@400;600&display=swap');
          :root{
            --bg-top: #f7fbff; --bg-bottom: #edf6ff; --fg: #0f172a; --fg-muted: #334155; --border: #dbeafe;
            --card: #ffffffee; --bubble: #f8fbff; --link: #0ea5e9; --accent: #2563eb; --input-bg: #ffffff; --placeholder: #64748b;
          }
          [data-testid="stAppViewContainer"]{
            background: linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
          }
          [data-testid="stAppViewContainer"] .main .block-container{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: 0 6px 18px rgba(2,6,23,.06);
            padding: .7rem .9rem 1rem .9rem;
          }
          .main .block-container, .main .block-container p, .main .block-container li, 
          .main .block-container label, .main .block-container h1, .main .block-container h2, .main .block-container h3{
            color: var(--fg);
            font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            font-size: 0.98rem;
            line-height: 1.5;
          }
          .main .block-container h1 { font-size: 1.28rem; margin: .4rem 0 .25rem; }
          .main .block-container h2 { font-size: 1.14rem; margin: .35rem 0 .2rem; }
          .main .block-container h3 { font-size: 1.02rem; margin: .3rem 0 .15rem; }
          .brand-title{
            font-family: 'Plus Jakarta Sans', Inter, system-ui;
            font-weight: 700; letter-spacing: .1px;
            font-size: clamp(19px, 2.8vw, 28px);
            color: var(--fg);
          }
          .brand-sub{ color: var(--fg-muted); font-size: 13px; margin-top: .2rem; }
          .hero-card{
            border-radius: 12px; padding: 10px 12px; border: 1px solid var(--border);
            background: #fffffff6; box-shadow: 0 6px 18px rgba(2,6,23,.06);
          }
          .side-card{
            border-radius: 12px; padding: 8px; border: 1px solid var(--border); background: #ffffffbf;
            display:flex;align-items:center;justify-content:center; aspect-ratio: 1.8/1;
          }
          [data-testid="stChatMessage"] > div:first-child{
            border-radius: 10px !important; border: 1px solid var(--border); background: var(--bubble);
            padding: 8px 10px !important;
          }
          [data-testid="stChatInput"] textarea{
            background: var(--input-bg) !important; border: 1px solid var(--border) !important; color: var(--fg) !important; caret-color: var(--accent) !important;
            font-size: 0.98rem !important;
          }
          [data-testid="stChatInput"] textarea::placeholder{ color: var(--placeholder) !important; opacity: 1 !important; }
          .stTextInput input, .stTextArea textarea{
            background: var(--input-bg) !important; border: 1px solid var(--border) !important; color: var(--fg) !important;
            font-size: 0.98rem !important;
          }
          .stTextInput input::placeholder, .stTextArea textarea::placeholder{ color: var(--placeholder) !important; opacity: 1 !important; }
          .main .block-container a{ color: var(--link); text-decoration: none; }
          .wave-wrap{height: 28px; overflow: hidden; margin-top: 3px;}
          [data-testid="stMarkdownContainer"] p { margin: .28rem 0; }
          [data-testid="stMarkdownContainer"] ul { margin: .28rem 0 .35rem 1.15rem; }
          [data-testid="stMarkdownContainer"] li { margin: .06rem 0; }
          [data-testid="stChatMessage"] p, [data-testid="stMarkdownContainer"] p { overflow-wrap: break-word; white-space: pre-wrap; }
        </style>
        """

def _sailboat_svg(width=220):
    return f"""
    <svg width="{width}" viewBox="0 0 256 128" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <linearGradient id="sea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#93c5fd"/>
          <stop offset="1" stop-color="#60a5fa"/>
        </linearGradient>
        <linearGradient id="sail" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#e0f2fe"/>
          <stop offset="1" stop-color="#bfdbfe"/>
        </linearGradient>
      </defs>
      <path d="M24 96c18 8 40 12 64 12s46-4 64-12c18 8 40 12 64 12" fill="none" stroke="url(#sea)" stroke-width="6" stroke-linecap="round"/>
      <path d="M128 22 L128 92" stroke="#1e40af" stroke-width="4" />
      <path d="M126 22 L70 80 L126 80 Z" fill="url(#sail)" stroke="#93c5fd" stroke-width="2"/>
      <path d="M130 28 L186 76 L130 76 Z" fill="url(#sail)" stroke="#93c5fd" stroke-width="2"/>
      <rect x="110" y="92" width="36" height="6" rx="3" fill="#1e3a8a"/>
      <circle cx="128" cy="22" r="3" fill="#1e40af"/>
    </svg>
    """

def _wave_svg(width="100%", height=28):
    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 1440 120" preserveAspectRatio="none"
         xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M0,64 C240,120 480,0 720,64 C960,128 1200,32 1440,96 L1440,120 L0,120 Z"
            fill="#e0f2fe" opacity=".65"/>
    </svg>
    """

def render_hero(high_readability: bool = False, logo_bytes: bytes | None = None) -> None:
    st.markdown(_nautical_css(high_readability), unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 2.6])
    with c1:
        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        if logo_bytes:
            try:
                im = Image.open(BytesIO(logo_bytes))
                st.image(im, use_column_width=True)
            except Exception:
                st.markdown(_sailboat_svg(220), unsafe_allow_html=True)
        else:
            st.markdown(_sailboat_svg(220), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            """
            <div class="hero-card">
              <div class="brand-title">GV_GPT — L’aria sta cambiando</div>
              <div class="brand-sub">Tema nautico chiaro • impaginazione compatta • attenzione al dettaglio</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="wave-wrap">{_wave_svg()}</div>', unsafe_allow_html=True)

def _warn_keys():
    eng = st.session_state.get("engine", "openai")
    if eng == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ OPENAI_API_KEY non trovato. Aggiungilo al file `.env`.")
    if eng == "hugging" and not os.getenv("HUGGINGFACE_API_KEY"):
        st.warning("⚠️ HUGGINGFACE_API_KEY non trovato. Inseriscilo in `.env`.")

# 4) UI base
st.set_page_config(page_title="GV_GPT — L’aria sta cambiando", page_icon="⛵", layout="wide")

# 5) Stato app e sidebar
st.session_state.setdefault("persist", True)
st.session_state.setdefault("engine", (os.getenv("GV_ENGINE", "openai") or "openai").lower())
st.session_state.setdefault("openai_model", os.getenv("OPENAI_MODEL") or "gpt-4o-mini")
st.session_state.setdefault("hf_model", os.getenv("HUGGINGFACE_MODEL") or "meta-llama/Meta-Llama-3.1-8B-Instruct")
st.session_state.setdefault("ollama_model", os.getenv("OLLAMA_MODEL") or "gv/phi35-mini-gv:latest")
st.session_state.setdefault("ollama_base", os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434")
st.session_state.setdefault("engine_lock", (os.getenv("GV_ENGINE_LOCK", "0").lower() in ("1","true","yes","on")))
st.session_state.setdefault("outline", "")
st.session_state.setdefault("didactic", False)
st.session_state.setdefault("thinking", False)
st.session_state.setdefault("target_words", 800)
st.session_state.setdefault("max_rounds", 4)
st.session_state.setdefault("streaming_on", True)
st.session_state.setdefault("stop_generation", False)
st.session_state.setdefault("high_readability", True)
st.session_state.setdefault("logo_bytes", None)
st.session_state.setdefault("history", load_memory() if st.session_state.get("persist", True)
else []
)
# <- init sicuro
st.session_state.setdefault("intent_overrides", {})     # ← PATCH: storage override intent

# ← nuovo log “READY” dopo init history (non rimuovo il tuo log precedente)
log.info(f"READY: persist={st.session_state.get('persist', True)} "
         f"history_len={len(st.session_state.get('history', []))} "
         f"engine={st.session_state.get('engine')}")

with st.sidebar:
    st.header("⚙️ Aspetto")
    st.session_state["high_readability"] = st.toggle("Alta leggibilità", value=st.session_state["high_readability"])
    logo = st.file_uploader("Carica logo (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if logo is not None:
        st.session_state["logo_bytes"] = logo.read()

render_hero(high_readability=st.session_state["high_readability"], logo_bytes=st.session_state.get("logo_bytes"))

with st.sidebar:
    st.header("🔀 Engine & Modelli")

    eng_display = (
        "OpenAI" if st.session_state["engine"] == "openai"
        else "HuggingFace" if st.session_state["engine"] == "hugging"
        else "Ollama"
    )
    engine_choice = st.radio(
        "Seleziona engine:",
        ["OpenAI", "HuggingFace", "Ollama"],
        index=0 if eng_display == "OpenAI" else (1 if eng_display == "HuggingFace" else 2),
        horizontal=True
    )

    # OpenAI
    st.subheader("OpenAI")
    openai_suggestions = ["gpt-4o-mini", "o4-mini", "gpt-4o", "Custom…"]
    current_openai = st.session_state["openai_model"]
    if current_openai not in openai_suggestions:
        openai_suggestions.insert(-1, current_openai)
    openai_sel = st.selectbox(
        "Modello OpenAI",
        options=openai_suggestions,
        index=openai_suggestions.index(current_openai) if current_openai in openai_suggestions else len(openai_suggestions)-1
    )
    if openai_sel == "Custom…":
        current_openai = st.text_input("Modello OpenAI (custom)", value=st.session_state["openai_model"])
    else:
        current_openai = openai_sel

    # HuggingFace
    st.subheader("HuggingFace (test)")
    current_hf = st.text_input(
        "Model (HF)",
        value=st.session_state["hf_model"],
        help="Esempi: meta-llama/Meta-Llama-3.1-8B-Instruct, mistralai/Mixtral-8x7B-Instruct-v0.1, google/gemma-2-9b-it"
    )

    # Ollama (whitelist curata)
    st.subheader("Ollama (locale)")
    base_val = st.text_input("Base URL", value=st.session_state["ollama_base"], help="Di solito http://localhost:11434")
    os.environ["OLLAMA_BASE_URL"] = base_val.strip()
    OLLAMA_WHITELIST = ["gv/phi35-mini-gv:latest"]
    current_ollama = st.session_state["ollama_model"]
    if current_ollama not in OLLAMA_WHITELIST:
        current_ollama = OLLAMA_WHITELIST[0]
    ollama_sel = st.selectbox("Modello Ollama", options=OLLAMA_WHITELIST, index=OLLAMA_WHITELIST.index(current_ollama))
    current_ollama = ollama_sel
    st.session_state["ollama_model"] = current_ollama

    st.header("📝 Output (Ollama)")
    default_tok = int(os.getenv("OLLAMA_NUM_PREDICT", "900") or "900")
    tok = st.slider("Token di output max", 200, 5000, default_tok, help="Aumenta se la risposta si ferma a metà.")
    os.environ["OLLAMA_NUM_PREDICT"] = str(tok)
    temp = st.slider(
        "Temperatura (decodifica)", 0.1, 1.0, float(os.getenv("OLLAMA_TEMPERATURE", "0.3")), 0.1,
        help="Valori più bassi = più stabile e meno divagazioni."
    )
    os.environ["OLLAMA_TEMPERATURE"] = str(temp)

    if st.button("✅ Applica"):
        st.session_state["engine"] = "openai" if engine_choice == "OpenAI" else ("hugging" if engine_choice == "HuggingFace" else "ollama")
        st.session_state["openai_model"] = (current_openai or "gpt-4o-mini").strip()
        st.session_state["hf_model"] = (current_hf or "meta-llama/Meta-Llama-3.1-8B-Instruct").strip()
        st.session_state["ollama_model"] = (current_ollama or "gv/phi35-mini-gv:latest").strip()
        st.session_state["ollama_base"]  = (base_val or "http://localhost:11434").strip()

        os.environ["GV_ENGINE"]        = st.session_state["engine"]
        os.environ["OPENAI_MODEL"]     = st.session_state["openai_model"]
        os.environ["HUGGINGFACE_MODEL"]= st.session_state["hf_model"]
        os.environ["OLLAMA_MODEL"]     = st.session_state["ollama_model"]
        os.environ["OLLAMA_BASE_URL"]  = st.session_state["ollama_base"]

        st.success(f"Impostato: {engine_choice} • OpenAI={st.session_state['openai_model']} • HF={st.session_state['hf_model']} • Ollama={st.session_state['ollama_model']}")
        st.rerun()

    st.header("🔒 Lock engine")
    lock = st.checkbox("Blocca engine (niente fallback automatico)", value=st.session_state["engine_lock"])
    st.session_state["engine_lock"] = lock
    os.environ["GV_ENGINE_LOCK"] = "1" if lock else "0"

    st.header("🧠 Outline (auto)")
    st.caption("Aggiornamento periodico dell’outline dalla bozza corrente (esperimentale).")

    st.header("🎓 Modalità didattica")
    st.session_state["didactic"] = st.checkbox("Spiega passo-passo (sezioni, esempi, mini-quiz)", value=st.session_state["didactic"])

    st.header("⚡ Streaming")
    st.session_state["streaming_on"] = st.checkbox("Streaming live (dove supportato)", value=st.session_state["streaming_on"])
    if st.button("🛑 Stop generazione (forza stop)"):
        st.session_state["stop_generation"] = True
        st.toast("Interruzione richiesta", icon="🛑")

    # 💾 Memoria
    st.header("💾 Memoria")
    st.session_state["persist"] = st.checkbox("Mantieni chat tra riavvii", value=st.session_state["persist"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Svuota chat"):
            st.session_state.history = []
            if st.session_state.get("persist", True):
                clear_memory()  # <- svuota anche il file data/memory_default.json
            st.success("Chat svuotata" + (" (anche memoria salvata)" if st.session_state.get("persist", True) else "."))
    with c2:
        if st.button("🗑️ Cancella memoria salvata"):
            clear_memory()
            st.success("Memoria persistente cancellata.")

    st.header("▶ Continua")
    has_chat_state = bool(st.session_state.get("history"))
    if st.button("▶ Continua ultima risposta", disabled=not has_chat_state):
        st.session_state["do_continue"] = True
        st.rerun()

    st.header("📤 Export")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Markdown .md",
                           data=export_chat_md(st.session_state.history if has_chat_state else []),
                           file_name=f"chat_{ts}.md", mime="text/markdown", disabled=not has_chat_state)
    with col2:
        st.download_button("JSON .json",
                           data=export_chat_json(st.session_state.history if has_chat_state else []),
                           file_name=f"chat_{ts}.json", mime="application/json", disabled=not has_chat_state)

    st.header("🛠️ Diagnostica Ollama")
    if st.button("Esegui test diagnostico"):
        if st.session_state["engine"] != "ollama":
            st.warning("Imposta l'engine su Ollama per testare.")
        else:
            try:
                reply, diag = call_ollama_generate("Di' soltanto: OK", debug=True, timeout=60)
                st.success("Diagnostica completata.")
                st.write("**Risposta:**", reply[:200])
                st.json(diag)
            except Exception as e:
                st.error(f"Diagnostica fallita: {e}")

_warn_keys()

# 6) Stato conversazione (già inizializzato con setdefault sopra)
# 7) Mostra conversazione
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div style='padding:.2rem .25rem'>{msg['content']}</div>", unsafe_allow_html=True)

# ---------- OUTLINE MANAGEMENT -----------------------------------------------
def _update_outline_if_needed(full_text: str, every_round: int, round_idx: int, system_text: str):
    ask = (
        "Sintetizza in 10-12 bullet point l'outline dei contenuti già scritti qui sotto."
        " Niente frasi lunghe, niente introduzioni, niente conclusioni, niente ripetizioni."
        " Solo bullet compatti.\n\n---\n" + _tail(full_text, 3200)
    )
    msgs = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": ask}
    ]
    try:
        outline = call_chat_smart(msgs, temperature=0.3)
        st.session_state["outline"] = (outline or "").strip()[:4000]
    except Exception as e:
        log.warning(f"Outline update failed: {e}")

# ---------- CONTINUE BUTTON ---------------------------------------------------
# ---------- CONTINUE BUTTON ---------------------------------------------------
if st.session_state.get("do_continue"):
    st.session_state.pop("do_continue", None)

    hist = st.session_state.history
    last_role = hist[-1]["role"] if hist else None
    engine_now = st.session_state.get("engine", "openai")
    reply = ""

    if last_role == "assistant":
        # Branch A: prosegui l’ULTIMA RISPOSTA (comportamento classico)
        last_user_text, last_assistant_text = _find_last_user_and_assistant(hist)
        san_last_user = _sanitize_meta_and_noise(last_user_text)
        nlp_last = analyze_text(san_last_user) if san_last_user else {"intent": "general"}
        try:
            k = hashlib.sha1((san_last_user or "").strip().lower().encode("utf-8")).hexdigest()
            ov = st.session_state["intent_overrides"].get(k)
            if ov:
                nlp_last["intent"] = ov
                nlp_last["score"] = 0.99
                nlp_last["override"] = True
        except Exception:
            pass
        system_text = system_prompt_for_intent(nlp_last.get("intent", "general"))
        if engine_now == "ollama":
            system_text += " Mantieni rigorosamente la lingua italiana per tutta la risposta; non passare a inglese o spagnolo. Se accade, correggiti e torna subito all'italiano."

        san_last_user = _sanitize_meta_and_noise(last_user_text)
        san_last_assistant = _sanitize_meta_and_noise(last_assistant_text)

        with st.chat_message("assistant"):
            t0 = time.time()
            try:
                msgs = build_continue_only_messages(
                    system_once=system_text,
                    history=[{"role": "user", "content": san_last_user},
                             {"role": "assistant", "content": san_last_assistant}],
                    final_text_tail=_tail(san_last_assistant, 3000),
                    round_words=800,
                    didactic=False,
                )
                if engine_now == "ollama" and st.session_state.get("streaming_on", True):
                    # Streaming per evitare lock della POST non-stream
                    placeholder = st.empty()
                    pieces = []
                    stopped_by_sentinel = False
                    for delta in stream_chat(
                            msgs,
                            model_override=st.session_state["ollama_model"],
                            temperature=0.7,
                            idle_timeout=8.0,
                            heartbeat_sec=2.0,
                    ):
                        if not isinstance(delta, str) or delta == "":
                            continue
                        pieces.append(delta)
                        joined = "".join(pieces)
                        # Stop immediato su sentinel
                        if any(s in joined for s in SENTINELS):
                            for s in SENTINELS:
                                if s in joined:
                                    joined = joined.split(s, 1)[0]
                                    stopped_by_sentinel = True
                                    break
                        # Light formatting durante lo stream
                        placeholder.markdown(re.sub(r"(?m)^\s*#{4,}\s*", "## ", joined))
                        if stopped_by_sentinel:
                            break
                    chunk = "".join(pieces)
                    for s in SENTINELS:
                        if s in chunk:
                            chunk = chunk.split(s, 1)[0]
                            break
                else:
                    # Non-stream con timeout difensivo
                    chunk = call_chat_smart(msgs, temperature=0.7, timeout=120)

                chunk = _strip_drift_lines(_strip_drift_prefix(chunk))
                if _looks_restart(chunk) or _is_reduant(chunk, san_last_assistant) or _too_similar(chunk, san_last_assistant):
                    import re
                    chunk = re.sub(r'(?is)^.{0,300}\n', '', chunk, count=1).strip()
                reply = (chunk or "").replace("<<FINE>>", "").strip()
                reply = post_format_response(reply)
                st.markdown(reply if reply else "_(nessun avanzamento)_")
            except Exception as e:
                reply = "⚠️ Errore (continua): " + str(e).split("\n")[0]
                st.markdown(reply)
            finally:
                st.session_state["stop_generation"] = False  # ← reset sicuro SEMPRE
            log.info(
                f"ENGINE={engine_now} DIDACTIC={st.session_state.get('didactic', False)} ELAPSED={time.time() - t0:.1f}s")

        st.session_state.history.append({"role": "assistant", "content": reply})

    else:
        # Branch B: l’ultimo messaggio è dell’utente → “continua” = approfondisci l’ULTIMA RICHIESTA
        last_user = next((m["content"] for m in reversed(hist) if m.get("role") == "user"), "")
        san_user = _sanitize_meta_and_noise(last_user)

        if engine_now in ("openai", "hugging"):
            nlp = analyze_text(san_user)
            sys = system_prompt_for_intent(nlp.get("intent", "general"))
            msgs = [
                {"role": "system", "content": sys},
                {"role": "user", "content": compose_prompt(san_user, nlp) + "\n\nApprofondisci e vai più a fondo. Evita premesse introduttive."}
            ]
        else:
            # Ollama: prompt minimale per evitare conflitti col SYSTEM del Modelfile
            msgs = [{"role": "user", "content": san_user + "\n\nApprofondisci e vai più a fondo. Evita premesse introduttive."}]

        with st.chat_message("assistant"):
            try:
                if engine_now == "ollama" and st.session_state.get("streaming_on", True):
                    placeholder = st.empty()
                    pieces = []
                    stopped_by_sentinel = False
                    for delta in stream_chat(
                            msgs,
                            model_override=st.session_state["ollama_model"],
                            temperature=0.6,
                            idle_timeout=8.0,
                            heartbeat_sec=2.0,
                    ):
                        if not isinstance(delta, str) or delta == "":
                            continue
                        pieces.append(delta)
                        joined = "".join(pieces)
                        if any(s in joined for s in SENTINELS):
                            for s in SENTINELS:
                                if s in joined:
                                    joined = joined.split(s, 1)[0]
                                    stopped_by_sentinel = True
                                    break
                        placeholder.markdown(re.sub(r"(?m)^\s*#{4,}\s*", "## ", joined))
                        if stopped_by_sentinel:
                            break
                    reply = "".join(pieces)
                    for s in SENTINELS:
                        if s in reply:
                            reply = reply.split(s, 1)[0]
                            break
                    reply = reply.strip()
                else:
                    reply = call_chat_smart(msgs, temperature=0.6, timeout=120)

                reply = _strip_drift_lines(_strip_drift_prefix(reply))
                reply = post_format_response(reply)
                st.markdown(reply)

            except Exception as e:
                reply = f"⚠️ Errore (continua/approfondisci): {str(e).splitlines()[0]}"
                st.markdown(reply)

        st.session_state.history.append({"role": "assistant", "content": reply})

    if st.session_state.get("persist", True):
        save_memory(st.session_state.history)

# 8) Input utente (turno normale)
user = st.chat_input("Scrivi qui…")
if user:
    san_user = _sanitize_meta_and_noise(user)

    # ── PATCH: se è un comando di "continua", NON salvarlo in history, attiva azione
    if _is_continue_cmd(san_user):
        st.session_state["do_continue"] = True
        st.rerun()

    st.session_state.history.append({"role": "user", "content": san_user})
    log.info(f"USER: {san_user}")

    with st.chat_message("user"):
        st.markdown(san_user)

    nlp_data = analyze_text(san_user)

    # ── PATCH: applica override intent se presente per questo testo
    try:
        k = hashlib.sha1((san_user or "").strip().lower().encode("utf-8")).hexdigest()
        ov = st.session_state["intent_overrides"].get(k)
        if ov:
            nlp_data["intent"] = ov
            nlp_data["score"] = 0.99
            nlp_data["override"] = True
    except Exception:
        pass

    with st.expander("🔎 NLP insight", expanded=False):
        st.write(nlp_data)

    engine_now = st.session_state.get("engine", "openai")
    intent = nlp_data.get("intent", "general")

    # Costruzione messaggi:
    if engine_now in ("openai", "hugging"):
        system_text = system_prompt_for_intent(intent)
        system_text += " Rispondi riferendoti soltanto all'ultimo messaggio dell'utente; ignora il contesto precedente salvo riferimenti espliciti."
        didactic_suffix = (
            "\n\nStile didattico: organizza a sezioni con titoli brevi, definizioni chiare, esempi pratici e, in chiusura, 3 domande-quiz con relative risposte."
            if st.session_state.get("didactic", False) else ""
        )
        short_history = _short_history_by_chars(st.session_state.history[:-1], max_chars=9000)
        short_history = [{"role": m["role"], "content": _sanitize_meta_and_noise(m.get("content",""))} for m in short_history]
        base_messages = [{"role": "system", "content": system_text}] + short_history + [
            {"role": "user", "content": compose_prompt(san_user, nlp_data) + didactic_suffix}
        ]
    else:
        # OLLAMA: prompt minimale (evita conflitti con SYSTEM nel Modelfile)
        base_messages = [{"role": "user", "content": san_user}]

    with st.chat_message("assistant"):
        t0 = time.time()
        reply = ""

        try:
            if st.session_state.get("streaming_on", True):
                # ▶︎ PATCH 2: streaming “light formatting” + sentinel stop
                placeholder = st.empty()
                pieces = []
                def _should_stop_cb() -> bool:
                    return bool(st.session_state.get("stop_generation"))

                stopped_by_sentinel = False
                for delta in stream_chat(
                    base_messages,
                    model_override=(
                        st.session_state["openai_model"] if engine_now == "openai"
                        else st.session_state["hf_model"] if engine_now == "hugging"
                        else st.session_state["ollama_model"]
                    ),
                    should_stop=_should_stop_cb,
                    idle_timeout=8.0,
                    heartbeat_sec=2.0,
                ):
                    if not isinstance(delta, str):
                        continue
                    if delta == "":
                        continue

                    pieces.append(delta)
                    joined = "".join(pieces)

                    # Stop immediato su uno dei sentinel supportati
                    if any(s in joined for s in SENTINELS):
                        for s in SENTINELS:
                            if s in joined:
                                joined = joined.split(s, 1)[0]
                                stopped_by_sentinel = True
                                break

                    # Solo normalizzazione minima durante lo stream
                    light = re.sub(r"(?m)^\s*#{4,}\s*", "## ", joined)
                    placeholder.markdown(light)

                    if stopped_by_sentinel:
                        break

                reply = "".join(pieces)
                for s in SENTINELS:
                    if s in reply:
                        reply = reply.split(s, 1)[0]
                        break
                reply = reply.strip()
                if not reply:
                    raise RuntimeError("Nessun testo dallo stream")

                # Post-processing completo **a fine stream**
                reply = _strip_drift_lines(_strip_drift_prefix(reply))
                reply = re.sub(r"(?im)^\s*🔎\s*NLP\s*insight\s*$", "", reply)
                reply = re.sub(r"\s+([.,;:!?])", r"\1", reply)
                reply = re.sub(r"[ \t]{2,}", " ", reply)
                reply = post_format_response(reply)
                placeholder.markdown(reply)

            else:
                reply = call_chat(
                    base_messages,
                    model_override=(
                        st.session_state["openai_model"] if engine_now == "openai"
                        else st.session_state["hf_model"] if engine_now == "hugging"
                        else st.session_state["ollama_model"]
                    )
                )
                reply = _strip_drift_lines(_strip_drift_prefix(reply))
                reply = post_format_response(reply)
                st.markdown(reply)
        except Exception as e:
            if not st.session_state.get("engine_lock", False):
                try:
                    reply = call_chat_smart(
                        base_messages,
                        model_override=(
                            st.session_state["openai_model"] if engine_now == "openai"
                            else st.session_state["hf_model"] if engine_now == "hugging"
                            else st.session_state["ollama_model"]
                        )
                    )
                    reply = _strip_drift_lines(_strip_drift_prefix(reply))
                    reply = post_format_response(reply)
                    st.markdown(reply)
                except Exception as e2:
                    reply = f"⚠️ Errore modello: {str(e2).splitlines()[0]}"
                    st.markdown(reply)
            else:
                reply = f"⚠️ Errore modello: {str(e).splitlines()[0]}"
                st.markdown(reply)
        finally:
            st.session_state["stop_generation"] = False  # ← reset sicuro SEMPRE

        log.info(f"ENGINE={engine_now} DIDACTIC={st.session_state.get('didactic', False)} ELAPSED={time.time()-t0:.1f}s")

    st.session_state.history.append({"role": "assistant", "content": reply})
    if st.session_state.get("persist", True):
        save_memory(st.session_state.history)

# ============== Pannello: Correzione intent ultimo messaggio =================
st.divider()
st.subheader("🖊️ Correzione intent dell’ultimo messaggio")

# ── PATCH: helper per chiave override
def _intent_key(text: str) -> str:
    return hashlib.sha1((text or "").strip().lower().encode("utf-8")).hexdigest()

corr_snapshot = {
    "text": "",
    "predicted_intent": "general",
    "predicted_score": 0.0,
}
try:
    last_user_msg = next((m for m in reversed(st.session_state.history) if m.get("role") == "user"), None)
    if last_user_msg:
        nlp = analyze_text(last_user_msg.get("content", ""))
        corr_snapshot["text"] = last_user_msg.get("content", "")
        corr_snapshot["predicted_intent"] = str(nlp.get("intent", "general") or "general")
        corr_snapshot["predicted_score"] = float(nlp.get("score", 0.0) or 0.0)
except Exception:
    pass

text_preview = corr_snapshot["text"][:160] + ("…" if len(corr_snapshot["text"]) > 160 else "")
st.write(f"**Testo:** “{text_preview}”" if text_preview else "_Invia un messaggio per vedere il pannello qui._")
st.write(f"**Predetto:** `{corr_snapshot['predicted_intent']}` — confidenza: {corr_snapshot['predicted_score']:.2f}")

labels = ["coding","business","nutrition","calendar","study","health","motivation","finance","science","general"]
try:
    default_idx = labels.index(corr_snapshot["predicted_intent"])
except Exception:
    default_idx = labels.index("general")

corr_sel = st.selectbox("Seleziona l'intent corretto", labels, index=default_idx, key="corr_global_select")
if st.button("💾 Salva correzione intent"):
    if corr_snapshot["text"]:
        k = _intent_key(corr_snapshot["text"])
        st.session_state["intent_overrides"][k] = corr_sel
        st.success(f"Intent aggiornato per quel messaggio: {corr_sel}")
        st.rerun()  # refresh immediato dell'NLP insight
    else:
        st.warning("Nessun messaggio utente da correggere.")
