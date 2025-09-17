
# -*- coding: utf-8 -*-
# server/app_streamlit.py (locked Ollama model + UI micro-patches)
# ─────────────────────────────────────────────────────────────────────────────
# GV_GPT — L’aria sta cambiando (Streamlit, tema nautico chiaro)
# - Engine switch: OpenAI ↔ HuggingFace ↔ Ollama
# - 🔒 Engine Lock + Fallback smart (se lock OFF): OpenAI → HuggingFace → Ollama
# - Didattica opzionale (solo OpenAI/HF)
# - Streaming: OpenAI/Ollama; HuggingFace pseudo-stream (tutto in un colpo)
# - Memoria persistente, export, diagnostica
# - UI micro-patches: input visibile, avatar, status bar, copy last reply, segmented con key
# - OLLAMA MODEL LOCK: gv/phi35-mini-gv:latest (non editabile)
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
import hashlib
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st
try:
    import streamlit_antd_components as sac
except Exception:
    sac = None
try:
    from streamlit_extras.badges import badge
except Exception:
    badge = None

from PIL import Image

# --- Boot Guard (anti-mix progetti) ------------------------------------------
try:
    import core.boot_guard  # noqa: F401
except Exception:
    EXPECTED = os.getenv("GV_EXPECTED_PROJECT", "GV_EXT")
    PID = os.getenv("PROJECT_ID")
    if PID and PID != EXPECTED:
        raise SystemExit(f"[ABORT] Wrong PROJECT_ID. Expected {EXPECTED}, got {PID!r}.")

# 2) Import moduli progetto (devono esistere nel repo)
from core.engine import (
    call_chat, stream_chat, call_chat_smart, build_continue_only_messages
)
from core.memory import load_memory, save_memory, clear_memory
from core.logger import get_logger

# ---- OLLAMA: modello unico consentito ---------------------------------------
OLLAMA_MODEL_ALLOWED = "gv/phi35-mini-gv:latest"
os.environ["OLLAMA_MODEL"] = OLLAMA_MODEL_ALLOWED  # forza ENV

# Ripulisci memoria all'avvio se richiesto da env (utile in demo o dopo bug)
if (os.getenv("GV_BOOT_CLEAR_MEMORY", "0").lower() in ("1", "true", "yes", "on")):
    clear_memory()

log = get_logger()

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

# === Logging ==================================================================
import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("GV_GPT")

# === Regex / Sanitizer ========================================================
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

# Sentinel di chiusura (coerente col Modelfile)
SENTINELS = ("[[END_OF_OUTPUT]]", "<<FINE>>")

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
        "ollama_model": OLLAMA_MODEL_ALLOWED,
        "hugging_model": os.getenv("HUGGINGFACE_MODEL") or "",
        "messages": history,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

# === Utils ===================================================================

def post_format_response(text: str) -> str:
    """
    Micro-formatting di sicurezza:
    - chiude codefence dispari
    - normalizza heading troppo profondi
    - converte bullet strani in "- "
    - rimuove frasi boilerplate in inglese
    - ripulisce spazi e righe eccessive
    """
    if not text:
        return text
    s = text
    # chiude code fence se dispari
    if s.count("```") % 2 == 1:
        s += "\n```"
    # rimuove link vuoti [text]()
    s = re.sub(r"\[([^\]]+)\]\(\s*\)", r"\1", s)
    # downgrade H4+ a H2
    s = re.sub(r"^\s*#{4,}\s*", "## ", s, flags=re.MULTILINE)
    # bullet uniformi
    s = re.sub(r"^[\t ]*[•*]\s+", "- ", s, flags=re.MULTILINE)
    # boilerplate EN
    s = re.sub(r"(?im)^\s*(ready to help|here (?:are|is)|let'?s |i can help)\b.*$", "", s)
    # spazi finali per riga
    s = re.sub(r"[ \t]+$", "", s, flags=re.MULTILINE)
    # riduci righe vuote consecutive
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

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

# Continue detection & robust last-turn finder
CONT_RX = re.compile(
    r"^(?:continua|prosegui|vai avanti|ancora|dammi(?:\s+di)?\s+pi[uù]|continua\s*pure)\s*[.!?]*$",
    re.IGNORECASE
)
def _is_continue_cmd(s: str) -> bool:
    return bool(CONT_RX.match((s or "").strip()))

def _find_last_user_and_assistant(history: list) -> tuple[str, str]:
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

# ── Segmented boolean control (SAC segmented / radio fallback) ───────────────
def _segmented_bool(label: str, key: str):
    use_sac = False
    try:
        import streamlit_antd_components as sac  # type: ignore
        use_sac = True
    except Exception:
        pass
    current = st.session_state.get(key, True)
    if use_sac:
        idx = 0 if current else 1
        choice = sac.segmented(items=["ON","OFF"], index=idx, size="sm", key=f"seg_{key}")
        st.session_state[key] = (choice == "ON")
    else:
        st.session_state[key] = (st.radio(label, ["ON","OFF"], index=0 if current else 1, horizontal=True, key=f"seg_{key}") == "ON")

# === THEME (nautico chiaro) & GLOBAL CSS =====================================
def _nautical_css(pro_mode: bool = False) -> str:
    return """
    <style>
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
        font-size: 0.98rem; line-height: 1.5;
      }
      .main .block-container h1 { font-size: 1.28rem; margin: .4rem 0 .25rem; }
      .main .block-container h2 { font-size: 1.14rem; margin: .35rem 0 .2rem; }
      .main .block-container h3 { font-size: 1.02rem; margin: .3rem 0 .15rem; }
      .brand-title{ font-weight:700; letter-spacing:.1px; font-size:clamp(19px, 2.8vw, 28px); color:var(--fg); }
      .brand-sub{ color: var(--fg-muted); font-size: 13px; margin-top: .2rem; }
      .hero-card{ border-radius: 12px; padding: 10px 12px; border: 1px solid var(--border);
                  background: #fffffff6; box-shadow: 0 6px 18px rgba(2,6,23,.06); }
      .side-card{ border-radius: 12px; padding: 8px; border: 1px solid var(--border); background: #ffffffbf;
                  display:flex;align-items:center;justify-content:center; aspect-ratio: 1.8/1; }
      [data-testid="stChatMessage"] > div:first-child{
        border-radius: 10px !important; border: 1px solid var(--border); background: var(--bubble);
        padding: 8px 10px !important;
      }
      [data-testid="stChatInput"] textarea{
        background: var(--input-bg) !important; border: 1px solid var(--border) !important; color: var(--fg) !important; caret-color: var(--accent) !important;
        font-size: 0.98rem !important;
      }
      [data-testid="stChatInput"] textarea::placeholder{ color: var(--placeholder) !important; opacity: 1 !important; }
      .mini-toolbar { position:fixed; left:1rem; right:1rem; bottom:4.5rem;
                      background:#ffffffcc; backdrop-filter: blur(6px);
                      border:1px solid var(--border); border-radius:12px; padding:6px 10px; z-index:999; }
      .wave-wrap{height: 28px; overflow: hidden; margin-top: 3px;}
      [data-testid="stMarkdownContainer"] p { margin: .28rem 0; }
      [data-testid="stMarkdownContainer"] ul { margin: .28rem 0 .35rem 1.15rem; }
      [data-testid="stMarkdownContainer"] li { margin: .06rem 0; }
      [data-testid="stChatMessage"] p, [data-testid="stMarkdownContainer"] p { overflow-wrap: break-word; white-space: pre-wrap; }

      /* Extra micro-polish */
      pre, code { white-space: pre-wrap !important; word-break: break-word !important; }
      ::-webkit-scrollbar{ height:10px; width:10px; }
      ::-webkit-scrollbar-thumb{ background:#c7d2fe; border-radius:10px; }
      ::-webkit-scrollbar-track{ background:transparent; }
      .main .block-container{ max-width: 1100px; margin-inline:auto; }
      .mini-toolbar{ box-shadow: 0 8px 30px rgba(2, 12, 27, .08); }
      @media (max-width: 420px){ .mini-toolbar{ bottom: 5.5rem; } }
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
          <stop offset="0" stop-color="#ffffff"/>
          <stop offset="1" stop-color="#e2e8f0"/>
        </linearGradient>
      </defs>
      <rect x="0" y="88" width="256" height="40" fill="url(#sea)" />
      <path d="M30 98 C 60 94 90 104 120 98 C148 92 180 100 210 96 L210 106 L30 106 Z" fill="#1d4ed8" opacity=".15"/>
      <circle cx="220" cy="100" r="3" fill="#3b82f6"/>
      <path d="M96 92 L96 56 L164 92 Z" fill="url(#sail)"/>
      <rect x="94" y="48" width="4" height="44" fill="#475569"/>
      <rect x="84" y="96" width="84" height="8" rx="4" fill="#94a3b8"/>
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

# === HERO =====================================================================
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
              <div class="brand-sub">Motore ibrido | Streaming robusto | Fallback smart | Memoria</div>
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

# ── STATUS BAR (solo UI) ─────────────────────────────────────────────────────
def _gv_status_badge(label: str) -> str:
    return f"<span style='display:inline-block;padding:.18rem .5rem;margin:.12rem;border:1px solid rgba(0,0,0,.06);border-radius:999px;background:#fff'>{label}</span>"

def render_status_bar():
    try:
        eng = st.session_state.get("engine","openai")
        model = (st.session_state.get("openai_model") if eng=="openai"
                 else st.session_state.get("hf_model") if eng=="hugging"
                 else OLLAMA_MODEL_ALLOWED)
        chips = [
            _gv_status_badge(f"Engine: {eng.title()}"),
            _gv_status_badge(f"Modello: {model}"),
            _gv_status_badge("Lock ON" if st.session_state.get("engine_lock") else "Lock OFF"),
            _gv_status_badge("Streaming ON" if st.session_state.get("streaming_on", True) else "Streaming OFF"),
            _gv_status_badge("Memoria ON" if st.session_state.get("persist", True) else "Memoria OFF"),
        ]
        st.markdown(f"<div style='margin:.2rem 0 .4rem'>{' '.join(chips)}</div>", unsafe_allow_html=True)
    except Exception:
        pass

# ── HOTFIX CSS: chat input visibility & toolbar overlap ───────────────────────
def _css_hotfix_input_overlay():
    st.markdown("""
    <style>
    [data-testid="stChatInput"]{ position: relative; z-index: 2000; }
    [data-testid="stChatInput"] textarea{
      background:#ffffff !important; color:#0f172a !important;
      border:1px solid #dbeafe !important; box-shadow:none !important;
    }
    .mini-toolbar{ bottom:7.2rem !important; z-index: 1500 !important; }
    </style>
    """, unsafe_allow_html=True)

# ── AFTER-REPLY UI: caption & copy button ────────────────────────────────────
def _ui_after_reply(reply: str):
    if not reply:
        return
    try:
        wc = _word_count(reply)
        st.caption(f"~{wc} parole • ~{int(wc*1.3)} token stimati")
        uid = str(int(time.time()*1000))
        js_text = json.dumps(reply)
        st.markdown(f"""
        <button id="gv-copy-{uid}" style="margin:.25rem 0;padding:.25rem .6rem;border:1px solid rgba(0,0,0,.1);border-radius:8px;background:#fff;cursor:pointer">
          Copia risposta
        </button>
        <script>
          (function(){{
            const btn = document.getElementById('gv-copy-{uid}');
            const text = {js_text};
            if(btn){{
              btn.onclick = async () => {{
                try{{ await navigator.clipboard.writeText(text); }}catch(e){{}}
              }}
            }}
          }})();
        </script>
        """, unsafe_allow_html=True)
    except Exception:
        pass

# 4) UI base
st.set_page_config(page_title="GV_GPT — L’aria sta cambiando", page_icon="⛵", layout="wide")

# 5) Stato app e sidebar (state base)
st.session_state.setdefault("persist", True)
st.session_state.setdefault("engine", (os.getenv("GV_ENGINE", "openai") or "openai").lower())
st.session_state.setdefault("openai_model", os.getenv("OPENAI_MODEL") or "gpt-4o-mini")
st.session_state.setdefault("hf_model", os.getenv("HUGGINGFACE_MODEL") or "meta-llama/Meta-Llama-3.1-8B-Instruct")
st.session_state.setdefault("ollama_base", os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434")
st.session_state.setdefault("engine_lock", (os.getenv("GV_ENGINE_LOCK", "false").lower() == "true"))
st.session_state.setdefault("didactic", False)
st.session_state.setdefault("high_readability", False)
st.session_state.setdefault("history", load_memory() if True else [])
st.session_state.setdefault("stop_generation", False)
st.session_state.setdefault("stop_epoch", 0.0)
st.session_state.setdefault("streaming_on", True)
st.session_state.setdefault("outline", "")
st.session_state.setdefault("intent_overrides", {})
# Forza il modello Ollama nello state (sanity guard anche contro vecchi valori)
st.session_state["ollama_model"] = OLLAMA_MODEL_ALLOWED

# Logo + Hero
with st.sidebar:
    st.header("⚙️ Aspetto")
    st.session_state["high_readability"] = st.toggle("Alta leggibilità", value=st.session_state["high_readability"])
    logo = st.file_uploader("Carica logo (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if logo is not None:
        st.session_state["logo_bytes"] = logo.read()

_css_hotfix_input_overlay()
render_hero(high_readability=st.session_state["high_readability"], logo_bytes=st.session_state.get("logo_bytes"))
render_status_bar()

with st.sidebar:
    st.header("🔀 Engine & Modelli")

    labels_eng = ["OpenAI", "HuggingFace", "Ollama"]
    idx_map = {"openai": 0, "hugging": 1, "ollama": 2}
    engine_choice = (sac.segmented(items=labels_eng, index=idx_map.get(st.session_state["engine"], 0), size="sm", key="seg_engine")
                     if sac else st.radio("Seleziona engine", labels_eng, index=idx_map.get(st.session_state["engine"], 0), horizontal=True, key="engine_choice"))
    engine_choice = engine_choice.lower()

    if engine_choice != st.session_state["engine"] and not st.session_state.get("engine_lock", False):
        st.session_state["engine"] = engine_choice

    if st.session_state.get("engine_lock", False):
        st.info("🔒 Engine lock attivo: niente switch automatici.")
    st.session_state["engine_lock"] = st.checkbox("Blocca engine (disabilita fallback)", value=st.session_state.get("engine_lock", False))

    st.text_input("OpenAI model", st.session_state["openai_model"], key="openai_model")
    st.text_input("HF model", st.session_state["hf_model"], key="hf_model")
    st.text_input("Ollama base URL", st.session_state["ollama_base"], key="ollama_base")
    # Modello Ollama bloccato (non editabile)
    st.text_input("Ollama model", OLLAMA_MODEL_ALLOWED, disabled=True)

    if st.button("Applica modelli / endpoint"):
        os.environ["GV_ENGINE"]          = st.session_state["engine"]
        os.environ["OPENAI_MODEL"]       = st.session_state["openai_model"]
        os.environ["HUGGINGFACE_MODEL"]  = st.session_state["hf_model"]
        os.environ["OLLAMA_MODEL"]       = OLLAMA_MODEL_ALLOWED
        os.environ["OLLAMA_BASE_URL"]    = st.session_state["ollama_base"]
        st.success(f"Impostato: {engine_choice} • OpenAI={st.session_state['openai_model']} • HF={st.session_state['hf_model']} • Ollama={OLLAMA_MODEL_ALLOWED}")
        st.rerun()

    st.header("🧠 Outline (auto)")
    st.caption("Aggiornamento periodico dell’outline dalla bozza corrente (esperimentale).")
    st.text_area("Outline corrente", value=st.session_state.get("outline",""), height=140, key="outline_view")
    if st.button("↻ Aggiorna outline ora"):
        hist = st.session_state.get("history", [])
        last_assistants = "\n\n".join([m.get("content","") for m in hist if m.get("role")== "assistant"])[-4000:]
        sys_txt = "Sintetizza in 10-12 bullet point l'outline dei contenuti già scritti sotto. Niente intro/conclusioni."
        try:
            _update_outline_if_needed(last_assistants, every_round=0, round_idx=0, system_text=sys_txt)
            st.success("Outline aggiornato.")
            st.rerun()
        except Exception as e:
            st.warning(str(e).splitlines()[0])

    st.header("🎓 Modalità didattica")
    st.session_state["didactic"] = st.checkbox("Spiega passo-passo (aggiungi brevi sezioni, esempi, mini-quiz)", value=st.session_state["didactic"])

    st.header("⚡ Streaming")
    _segmented_bool("Streaming", "streaming_on")
    if st.button("🛑 Stop generazione (forza stop)"):
        st.session_state["stop_generation"] = True
        st.session_state["stop_epoch"] = time.time()
        st.toast("Interruzione richiesta", icon="🛑")

    st.header("🧠 Memoria")
    _segmented_bool("Persisti memoria", "persist")

    st.header("▶ Continua")
    if st.button("▶ Continua ultima risposta", key="btn_continue"):
        st.session_state["do_continue"] = True
        st.rerun()

    with st.expander("📤 Export", expanded=False):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Markdown .md", key="dl_md",
                data=export_chat_md(st.session_state.get("history", [])),
                file_name=f"chat_{ts}.md", mime="text/markdown",
                disabled=not bool(st.session_state.get("history"))
            )
        with c2:
            st.download_button(
                "JSON .json", key="dl_json",
                data=export_chat_json(st.session_state.get("history", [])),
                file_name=f"chat_{ts}.json", mime="application/json",
                disabled=not bool(st.session_state.get("history"))
            )

    with st.expander("🛠️ Diagnostica modello", expanded=False):
        if st.button("Esegui test su engine corrente", key="btn_diag"):
            try:
                reply, diag = (call_ollama_generate("Di' soltanto: OK", debug=True, timeout=60)
                               if st.session_state.get("engine") == "ollama"
                               else (call_chat([{"role": "user", "content": "Di' soltanto: OK"}]), {}))
                st.success("OK")
                st.code(str(diag)[:1200])
            except Exception as e:
                st.error(str(e))

_warn_keys()

# 6) Mostra conversazione
for msg in st.session_state.history:
    with st.chat_message(msg["role"], avatar="⚓" if msg["role"]=="assistant" else "🙋‍♂️"):
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

# ---------- CONTINUE BUTTON / GENERAZIONE ------------------------------------
if st.session_state.get("do_continue"):
    st.session_state.pop("do_continue", None)

    hist = st.session_state.history
    last_role = hist[-1]["role"] if hist else None
    engine_now = st.session_state.get("engine", "openai")
    reply = ""

    if last_role == "assistant":
        # Branch A: prosegui l’ULTIMA RISPOSTA
        last_user_text, last_assistant_text = _find_last_user_and_assistant(hist)
        san_last_user = _sanitize_meta_and_noise(last_user_text)
        nlp_last = analyze_text(san_last_user) if san_last_user else {"intent":"general"}

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

                stream_started_at = time.time()
                def _should_stop_cb() -> bool:
                    sg = st.session_state.get("stop_generation", False)
                    se = st.session_state.get("stop_epoch", 0.0)
                    return bool(sg and se >= stream_started_at)

                if engine_now == "ollama" and st.session_state.get("streaming_on", True):
                    placeholder = st.empty()
                    pieces = []
                    stopped_by_sentinel = False

                    for delta in stream_chat(
                            msgs,
                            model_override=OLLAMA_MODEL_ALLOWED,
                            temperature=0.7,
                            idle_timeout=8.0,
                            heartbeat_sec=2.0,
                            should_stop=_should_stop_cb,
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
                    chunk = "".join(pieces)
                    for s in SENTINELS:
                        if s in chunk:
                            chunk = chunk.split(s, 1)[0]
                            break
                else:
                    chunk = call_chat_smart(
                        msgs, temperature=0.7, timeout=120,
                        model_override=(
                            st.session_state["openai_model"] if engine_now == "openai"
                            else st.session_state["hf_model"] if engine_now == "hugging"
                            else OLLAMA_MODEL_ALLOWED
                        )
                    )

                chunk = _strip_drift_lines(_strip_drift_prefix(chunk))
                if _looks_restart(chunk) or _is_reduant(chunk, san_last_assistant) or _too_similar(chunk, san_last_assistant):
                    import re as _re
                    chunk = _re.sub(r'(?is)^.{0,300}\n', '', chunk, count=1).strip()
                reply = (chunk or "").replace("<<FINE>>", "").strip()
                reply = _sanitize_meta_and_noise(reply)
                reply = post_format_response(reply)
                st.markdown(reply if reply else "_(nessun avanzamento)_")
            except Exception as e:
                reply = "⚠️ Errore (continua): " + str(e).split("\n")[0]
                st.markdown(reply)
            finally:
                st.session_state["stop_generation"] = False  # reset sicuro SEMPRE
            log.info(f"ENGINE={engine_now} DIDACTIC={st.session_state.get('didactic', False)} ELAPSED={time.time() - t0:.1f}s")

        st.session_state.history.append({"role": "assistant", "content": reply})

    else:
        # Branch B: approfondisci l’ULTIMA RICHIESTA
        last_user = next((m["content"] for m in reversed(hist) if m.get("role") == "user"), "")
        san_user = _sanitize_meta_and_noise(last_user)

        if engine_now in ("openai", "hugging"):
            nlp = analyze_text(san_user) if san_user else {}
            sys_txt = system_prompt_for_intent(nlp.get("intent", "general"))
            msgs = [
                {"role": "system", "content": sys_txt},
                {"role": "user", "content": compose_prompt(san_user, nlp) + "\n\nApprofondisci e vai più a fondo. Evita premesse introduttive."}
            ]
        else:
            msgs = [{"role": "user", "content": san_user + "\n\nApprofondisci e vai più a fondo. Evita premesse introduttive."}]

        with st.chat_message("assistant"):
            try:
                stream_started_at = time.time()
                def _should_stop_cb() -> bool:
                    sg = st.session_state.get("stop_generation", False)
                    se = st.session_state.get("stop_epoch", 0.0)
                    return bool(sg and se >= stream_started_at)

                if engine_now == "ollama" and st.session_state.get("streaming_on", True):
                    placeholder = st.empty()
                    pieces = []
                    stopped_by_sentinel = False

                    for delta in stream_chat(
                            msgs,
                            model_override=OLLAMA_MODEL_ALLOWED,
                            temperature=0.6,
                            idle_timeout=8.0,
                            heartbeat_sec=2.0,
                            should_stop=_should_stop_cb,
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
                    reply = call_chat_smart(
                        msgs, temperature=0.6, timeout=120,
                        model_override=(
                            st.session_state["openai_model"] if engine_now == "openai"
                            else st.session_state["hf_model"] if engine_now == "hugging"
                            else OLLAMA_MODEL_ALLOWED
                        )
                    )

                reply = _strip_drift_lines(_strip_drift_prefix(reply))
                reply = _sanitize_meta_and_noise(reply)
                reply = post_format_response(reply)
                st.markdown(reply)

            except Exception as e:
                reply = f"⚠️ Errore (continua/approfondisci): {str(e).splitlines()[0]}"
                st.markdown(reply)

        st.session_state.history.append({"role": "assistant", "content": reply})

    if st.session_state.get("persist", True):
        save_memory(st.session_state.history)

# 7.9) Mini-toolbar fissa (solo STOP, niente doppi toggle)
st.markdown('<div class="mini-toolbar">', unsafe_allow_html=True)
tb1, tb2 = st.columns([3,1])
with tb1:
    st.caption("Pronto.")
with tb2:
    stop_clicked = st.button("Stop", key="stop_btn_toolbar_safe")

if stop_clicked:
    st.session_state["stop_generation"] = True
    st.session_state["stop_epoch"] = time.time()
    st.toast("Interruzione richiesta", icon="🛑")

st.markdown("</div>", unsafe_allow_html=True)

# 8) Input utente (turno normale)
user = st.chat_input("Scrivi qui…")
if user:
    san_user = _sanitize_meta_and_noise(user)

    if _is_continue_cmd(san_user):
        st.session_state["do_continue"] = True
        st.rerun()

    st.session_state.history.append({"role": "user", "content": san_user})
    log.info(f"USER: {san_user}")

    with st.chat_message("user", avatar="🙋‍♂️"):
        st.markdown(san_user)

    engine_now = st.session_state.get("engine", "openai")
    nlp_data = analyze_text(san_user) if san_user else {"intent":"general"}
    intent = nlp_data.get("intent", "general")

    if engine_now in ("openai", "hugging"):
        system_text = system_prompt_for_intent(intent)
        system_text += " Rispondi riferendoti soltanto all'ultimo messaggio dell'utente; ignora il contesto precedente salvo riferimenti espliciti."
        didactic_suffix = ("\n\nStile didattico: organizza a sezioni con titoli brevi, definizioni chiare, esempi pratici e, in chiusura, 3 domande-quiz con relative risposte."
                           if st.session_state.get("didactic", False) else "")
        short_history = _short_history_by_chars(st.session_state.history[:-1], max_chars=9000)
        short_history = [{"role": m["role"], "content": _sanitize_meta_and_noise(m.get("content",""))} for m in short_history]
        base_messages = [{"role": "system", "content": system_text}] + short_history + [
            {"role": "user", "content": compose_prompt(san_user, nlp_data) + didactic_suffix}
        ]
    else:
        base_messages = [{"role": "user", "content": san_user}]

    with st.chat_message("assistant", avatar="⚓"):
        t0 = time.time()
        reply = ""

        try:
            if st.session_state.get("streaming_on", True):
                placeholder = st.empty()
                pieces = []

                stream_started_at = time.time()
                def _should_stop_cb() -> bool:
                    sg = st.session_state.get("stop_generation", False)
                    se = st.session_state.get("stop_epoch", 0.0)
                    return bool(sg and se >= stream_started_at)

                stopped_by_sentinel = False
                for delta in stream_chat(
                    base_messages,
                    model_override=(
                        st.session_state["openai_model"] if engine_now == "openai"
                        else st.session_state["hf_model"] if engine_now == "hugging"
                        else OLLAMA_MODEL_ALLOWED
                    ),
                    should_stop=_should_stop_cb,
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

                reply = _strip_drift_lines(_strip_drift_prefix(reply))
                reply = _sanitize_meta_and_noise(reply)
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
                        else OLLAMA_MODEL_ALLOWED
                    )
                )
                reply = _strip_drift_lines(_strip_drift_prefix(reply))
                reply = _sanitize_meta_and_noise(reply)
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
                            else OLLAMA_MODEL_ALLOWED
                        )
                    )
                    reply = _strip_drift_lines(_strip_drift_prefix(reply))
                    reply = _sanitize_meta_and_noise(reply)
                    reply = post_format_response(reply)
                    st.markdown(reply)
                except Exception as e2:
                    reply = f"⚠️ Errore modello: {str(e2).splitlines()[0]}"
                    st.markdown(reply)
            else:
                reply = f"⚠️ Errore modello: {str(e).splitlines()[0]}"
                st.markdown(reply)
        finally:
            st.session_state["stop_generation"] = False  # reset

        log.info(f"ENGINE={engine_now} DIDACTIC={st.session_state.get('didactic', False)} ELAPSED={time.time()-t0:.1f}s")
        _ui_after_reply(reply)

    st.session_state.history.append({"role": "assistant", "content": reply})
    if st.session_state.get("persist", True):
        save_memory(st.session_state.history)

# ============== Pannello: Correzione intent ultimo messaggio =================
st.divider()
st.subheader("🖊️ Correzione intent dell’ultimo messaggio")

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
        nlp = analyze_text(last_user_msg.get("content", "")) or {}
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

if st.button("💾 Salva correzione intent", key="btn_save_intent"):
    if corr_snapshot["text"]:
        k = _intent_key(corr_snapshot["text"])
        st.session_state["intent_overrides"][k] = corr_sel

        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "type": "correction",
            "text": corr_snapshot["text"],
            "hash": k,
            "intent": corr_sel,
            "source": "ui"
        }
        (ROOT / "data").mkdir(parents=True, exist_ok=True)
        with open(ROOT / "data" / "nlp_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        st.success(f"Intent aggiornato per quel messaggio: {corr_sel}")
        st.rerun()
    else:
        st.warning("Nessun messaggio utente da correggere.")
