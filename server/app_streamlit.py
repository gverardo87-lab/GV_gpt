# server/app_streamlit.py
# ─────────────────────────────────────────────────────────────────────────────
# GV_GPT — L’aria sta cambiando (Streamlit, tema nautico chiaro)
# - Engine switch: OpenAI ↔ HuggingFace ↔ Ollama
# - 🔒 Engine Lock + Fallback smart (se lock OFF): OpenAI → HuggingFace → Ollama
# - Didattica + Thinking longform "continue-only" con outline+tail (Ollama)
# - Streaming: OpenAI/Ollama; HuggingFace pseudo-stream (tutto in un colpo)
# - Memoria persistente, export, diagnostica, slider num_predict e temperatura per Ollama
# - Toggle “Alta leggibilità” per passare da scenografico a pro
# - Sanitizer anti meta-marker + filtri anti drift (“Instruction/Your task/Begin by”)
# ─────────────────────────────────────────────────────────────────────────────

# 0) Ponte: assicura che la root del progetto sia nel PYTHONPATH
import sys
from pathlib import Path

def find_project_root(start: Path | None = None) -> Path:
    start = start or Path(__file__).resolve()
    cur = start if start.is_dir() else start.parent
    for _ in range(7):
        if (cur / "data").exists() and (cur / "nlp_layer").exists():
            return cur
        cur = cur.parent
    return Path(__file__).resolve().parent  # fallback

ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 1) Env & imports base
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv(ROOT / ".env")

import os
import json
import time
import re
import difflib
from datetime import datetime
from io import BytesIO

# --- Boot Guard (anti-mix progetti) ------------------------------------------
try:
    import core.boot_guard  # noqa: F401
except Exception:
    EXPECTED = os.getenv("GV_EXPECTED_PROJECT", "GV_EXT")
    PID = os.getenv("PROJECT_ID")
    if PID and PID != EXPECTED:
        raise SystemExit(f"[ABORT] Wrong PROJECT_ID. Expected {EXPECTED}, got {PID!r}.")

# Default Ollama model se non in .env
if not os.getenv("OLLAMA_MODEL"):
    os.environ["OLLAMA_MODEL"] = "phi3:3.8b"

import streamlit as st
import requests  # per autodiscovery modelli Ollama
from PIL import Image

# 2) Import moduli progetto
from core.engine import (
    call_chat, stream_chat, call_chat_smart, build_continue_only_messages
)
from core.memory import load_memory, save_memory, clear_memory
from core.logger import get_logger

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

# --- (NUOVO) Percorsi log supervisioni + writer robusto ----------------------
from datetime import datetime as _dt_for_log  # alias per non confliggere
import hashlib as _hashlib_for_log

DATA = ROOT / "data"
LOG_FILE = DATA / "nlp_logs.jsonl"

_last_sig = None
_last_sig_at = 0.0

def write_nlp_log(payload: dict):
    """
    Scrive su data/nlp_logs.jsonl e ritorna (ok: bool, err: str|None).
    - timestamp coerenti: 'ts' epoch + 'time' ISO8601
    - anti-duplicato entro 2s (mitiga i rerun Streamlit)
    """
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        safe = dict(payload or {})
        ts = time.time()
        safe.setdefault("ts", ts)
        safe["time"] = _dt_for_log.fromtimestamp(ts).isoformat()
        line = json.dumps(safe, ensure_ascii=False, separators=(",", ":")) + "\n"

        global _last_sig, _last_sig_at
        sig = _hashlib_for_log.sha1(line.encode("utf-8")).hexdigest()
        if _last_sig == sig and (ts - _last_sig_at) < 2.0:
            return True, None  # duplicato evitato → ok

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        _last_sig, _last_sig_at = sig, ts
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# --- Sanitizer / Post-processor anti drift -----------------------------------
# 1) Rimuove token stile LLM (<|...|>) e direttive tra [] (INTENT, ecc.)
META_RX = re.compile(
    r"(?is)(<\|[^>]*\|>"
    r"|\[(?:INTENT|EXPLICIT[\s_-]*ACTION[\s_-]*LIST)[^\]]*\]"
    r"|\[[A-Z][A-Z0-9 _-]{2,}:[^\]]*\]"
    r"|\[[A-Z][A-Z0-9 _-]{2,}\])"
)
def _sanitize_meta(s: str) -> str:
    if not s:
        return s
    return META_RX.sub("", s).strip()

# 2) Rimuove prefissi tipo "Instruction …" / "I'm sorry …"
DRIFT_PREFIX_RX = re.compile(r"(?is)^\s*(instruction[s]?:.*?\n+|\s*i['’]m\s+sorry[^.\n]*[.\n]+\s*)")
def _strip_drift_prefix(text: str) -> str:
    if not text:
        return text
    out = DRIFT_PREFIX_RX.sub("", text).lstrip()
    return out if out else text

# 3) Sopprime righe “Your task … / Instruction … / Begin by …”
DRIFT_LINES_RX = re.compile(r"(?im)^\s*(your\s+task|instruction|begin\s+by)\s*:\s.*$")
def _strip_drift_lines(text: str) -> str:
    if not text:
        return text
    return DRIFT_LINES_RX.sub("", text)

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

def _find_last_user_and_assistant(history: list) -> tuple[str, str]:
    last_assistant = ""
    last_user = ""
    for m in reversed(history):
        if not last_assistant and m["role"] == "assistant":
            last_assistant = m["content"]
        elif m["role"] == "user":
            last_user = m["content"]
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
          [data-testid="stAppViewContainer"]{ background: linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%); }
          [data-testid="stAppViewContainer"] .main .block-container{
            background: var(--card); border: 1px solid var(--border); border-radius: 18px;
            box-shadow: 0 6px 26px rgba(15, 23, 42, .04); padding: 1rem 1.25rem 1.5rem 1.25rem;
          }
          .main .block-container, .main .block-container p, .main .block-container li, 
          .main .block-container label, .main .block-container h1, .main .block-container h2, .main .block-container h3{
            color: var(--fg); font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
          }
          .brand-title{ font-family: 'Plus Jakarta Sans', Inter, system-ui; font-weight: 700; letter-spacing: .2px;
            font-size: clamp(24px, 3.6vw, 36px); color: var(--fg); }
          .brand-sub{ color: var(--fg-muted); font-size: 14px; margin-top: .25rem; }
          .hero-card{ border-radius: 16px; padding: 14px 18px; border: 1px solid var(--border); background: #fffffff6; }
          .side-card{ border-radius: 16px; padding: 12px; border: 1px solid var(--border); background: #ffffff;
            display:flex;align-items:center;justify-content:center; aspect-ratio: 1.8/1; }
          [data-testid="stChatMessage"] > div:first-child{ border-radius: 12px !important; border: 1px solid var(--border); background: var(--bubble); }
          [data-testid="stChatInput"] textarea{
            background: var(--input-bg) !important; border: 1px solid var(--border) !important;
            color: var(--fg) !important; caret-color: var(--accent) !important;
          }
          [data-testid="stChatInput"] textarea::placeholder{ color: var(--placeholder) !important; opacity: 1 !important; }
          .stTextInput input, .stTextArea textarea{ background: var(--input-bg) !important; border: 1px solid var(--border) !important; color: var(--fg) !important; }
          .stTextInput input::placeholder, .stTextArea textarea::placeholder{ color: var(--placeholder) !important; opacity: 1 !important; }
          .main .block-container a{ color: var(--link); text-decoration: none; }
          .wave-wrap{height: 30px; overflow: hidden; margin-top: 6px;}
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
          [data-testid="stAppViewContainer"]{ background: linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%); }
          [data-testid="stAppViewContainer"] .main .block-container{
            background: var(--card); border: 1px solid var(--border); border-radius: 18px; box-shadow: 0 8px 24px rgba(2,6,23,.06);
            padding: 1rem 1.25rem 1.5rem 1.25rem;
          }
          .main .block-container, .main .block-container p, .main .block-container li, 
          .main .block-container label, .main .block-container h1, .main .block-container h2, .main .block-container h3{
            color: var(--fg); font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
          }
          .brand-title{ font-family: 'Plus Jakarta Sans', Inter, system-ui; font-weight: 700; letter-spacing: .2px;
            font-size: clamp(26px, 4vw, 40px); color: var(--fg); }
          .brand-sub{ color: var(--fg-muted); font-size: 14px; margin-top: .25rem; }
          .hero-card{ border-radius: 16px; padding: 14px 18px; border: 1px solid var(--border); background: #fffffff6; box-shadow: 0 8px 24px rgba(2,6,23,.06); }
          .side-card{ border-radius: 16px; padding: 12px; border: 1px solid var(--border); background: #ffffffbf;
            display:flex;align-items:center;justify-content:center; aspect-ratio: 1.8/1; }
          [data-testid="stChatMessage"] > div:first-child{ border-radius: 12px !important; border: 1px solid var(--border); background: var(--bubble); }
          [data-testid="stChatInput"] textarea{
            background: var(--input-bg) !important; border: 1px solid var(--border) !important; color: var(--fg) !important; caret-color: var(--accent) !important;
          }
          [data-testid="stChatInput"] textarea::placeholder{ color: var(--placeholder) !important; opacity: 1 !important; }
          .stTextInput input, .stTextArea textarea{ background: var(--input-bg) !important; border: 1px solid var(--border) !important; color: var(--fg) !important; }
          .stTextInput input::placeholder, .stTextArea textarea::placeholder{ color: var(--placeholder) !important; opacity: 1 !important; }
          .main .block-container a{ color: var(--link); text-decoration: none; }
          .wave-wrap{height: 36px; overflow: hidden; margin-top: 6px;}
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

def _wave_svg(width="100%", height=36):
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
                st.markdown(_sailboat_svg(240), unsafe_allow_html=True)
        else:
            st.markdown(_sailboat_svg(240), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            """
            <div class="hero-card">
              <div class="brand-title">GV_GPT — L’aria sta cambiando</div>
              <div class="brand-sub">Tema nautico chiaro • leggibilità professionale • attenzione al dettaglio</div>
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
st.session_state.setdefault("ollama_model", os.getenv("OLLAMA_MODEL") or "phi3:3.8b")
st.session_state.setdefault("ollama_base", os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434")
st.session_state.setdefault("engine_lock", (os.getenv("GV_ENGINE_LOCK", "0").lower() in ("1","true","yes","on")))
st.session_state.setdefault("outline", "")
st.session_state.setdefault("didactic", True)
st.session_state.setdefault("thinking", False)
st.session_state.setdefault("target_words", 800)
st.session_state.setdefault("max_rounds", 4)
st.session_state.setdefault("streaming_on", True)
st.session_state.setdefault("high_readability", True)
st.session_state.setdefault("logo_bytes", None)
st.session_state.setdefault("corr_snapshot", None)  # ⬅️ NUOVO: snapshot ultimo input+pred

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
    openai_suggestions = ["gpt-4o-mini", "gpt-4o", "o4-mini", "gpt-4.1-mini", "Custom…"]
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

    # Ollama
    st.subheader("Ollama (locale)")
    base_val = st.text_input("Base URL", value=st.session_state["ollama_base"], help="Di solito http://localhost:11434")
    def _discover_ollama_models(base_url: str) -> list[str]:
        base = (base_url or "http://localhost:11434").rstrip("/")
        try:
            r = requests.get(f"{base}/api/tags", timeout=3)
            r.raise_for_status()
            data = r.json() or {}
            models = [m.get("name") for m in data.get("models", []) if isinstance(m.get("name"), str)]
            pref = ["phi3:3.8b", "llama3:8b", "mistral:7b"]
            ordered = [m for m in pref if m in models] + [m for m in models if m not in pref]
            return ordered or ["phi3:3.8b", "llama3:8b", "mistral:7b"]
        except Exception:
            return ["phi3:3.8b", "llama3:8b", "mistral:7b"]
    models_local = _discover_ollama_models(base_val)
    models_local = models_local + ["Custom…"]
    current_ollama = st.session_state["ollama_model"]
    if current_ollama not in models_local:
        models_local.insert(0, current_ollama)
    ollama_sel = st.selectbox("Modello Ollama", options=models_local, index=models_local.index(current_ollama))
    if ollama_sel == "Custom…":
        current_ollama = st.text_input("Modello Ollama (custom)", value=st.session_state["ollama_model"])
    else:
        current_ollama = ollama_sel

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
        st.session_state["ollama_model"] = (current_ollama or "phi3:3.8b").strip()
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

    st.header("🧠 Memoria")
    st.checkbox("Mantieni chat tra riavvii", key="persist")
    cols_mem = st.columns(2)
    with cols_mem[0]:
        if st.button("🧹 Svuota chat"):
            st.session_state.history = []
            st.success("Chat svuotata.")
    with cols_mem[1]:
        if st.button("🗑️ Cancella memoria salvata"):
            clear_memory()
            st.success("Memoria persistente cancellata.")

    st.header("🎓 Modalità didattica")
    st.session_state["didactic"] = st.checkbox("Spiega passo-passo (sezioni, esempi, mini-quiz)", value=st.session_state["didactic"])

    st.header("🧠 Thinking (longform)")
    st.session_state["thinking"] = st.checkbox("Attiva modalità lunga (continue-only su Ollama)", value=st.session_state["thinking"])
    target_label = st.select_slider(
        "Lunghezza desiderata",
        options=["~400 parole", "~800 parole", "~1200 parole", "~2000 parole"],
        value="~800 parole" if st.session_state["target_words"] == 800 else
              "~400 parole" if st.session_state["target_words"] == 400 else
              "~1200 parole" if st.session_state["target_words"] == 1200 else "~2000 parole"
    )
    target_words_map = {"~400 parole": 400, "~800 parole": 800, "~1200 parole": 1200, "~2000 parole": 2000}
    st.session_state["target_words"] = target_words_map[target_label]
    st.session_state["max_rounds"] = st.slider("Max round extra", 0, 8, st.session_state["max_rounds"])

    st.header("⚡ Streaming")
    st.session_state["streaming_on"] = st.checkbox("Streaming live (dove supportato)", value=st.session_state["streaming_on"])

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

# 6) Stato conversazione
if "history" not in st.session_state:
    st.session_state.history = load_memory()

# 7) Mostra conversazione
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div style='padding:.25rem .25rem'>{msg['content']}</div>", unsafe_allow_html=True)

# ---------- OUTLINE MANAGEMENT -----------------------------------------------
def _update_outline_if_needed(full_text: str, every_round: int, round_idx: int, system_text: str):
    if every_round <= 0 or round_idx % every_round != 0:
        return
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
if st.session_state.get("do_continue"):
    st.session_state.pop("do_continue", None)
    last_user_text, last_assistant_text = _find_last_user_and_assistant(st.session_state.history)
    engine_now = st.session_state.get("engine", "openai")
    system_text = system_prompt_for_intent("general")
    if engine_now == "ollama":
        system_text += " Mantieni rigorosamente la lingua italiana per tutta la risposta; non passare a inglese o spagnolo. Se accade, correggiti e torna subito all'italiano."

    # Sanitize degli ultimi turni
    san_last_user = _sanitize_meta(last_user_text)
    san_last_assistant = _sanitize_meta(last_assistant_text)

    with st.chat_message("assistant"):
        t0 = time.time()
        reply = ""
        try:
            msgs = build_continue_only_messages(
                system_once=system_text,
                history=[{"role": "user", "content": san_last_user},
                         {"role": "assistant", "content": san_last_assistant}],
                final_text_tail=_tail(san_last_assistant, 3000),
                round_words=800,
                didactic=False,
            )
            chunk = call_chat_smart(msgs, temperature=0.7)  # non-stream per applicare guardie
            # Anti-drift/apology/Instruction
            chunk = _strip_drift_lines(_strip_drift_prefix(chunk))
            if _looks_restart(chunk) or _is_reduant(chunk, san_last_assistant) or _too_similar(chunk, san_last_assistant):
                chunk = re.sub(r'(?is)^.{0,300}\n', '', chunk, count=1).strip()
            reply = (chunk or "").replace("<<FINE>>","").strip()
            st.markdown(reply if reply else "_(nessun avanzamento)_")
        except Exception as e:
            reply = "⚠️ Errore (continua): " + str(e).split("\n")[0]
            st.markdown(reply)
        log.info(f"CONTINUE elapsed={time.time()-t0:.1f}s")

    st.session_state.history.append({"role": "assistant", "content": reply})
    if st.session_state.get("persist", True):
        save_memory(st.session_state.history)

# 8) Input utente (turno normale)
if user := st.chat_input("Scrivi qui…"):
    # Sanitizza subito l'input per evitare drift
    san_user = _sanitize_meta(user)

    st.session_state.history.append({"role": "user", "content": san_user})
    log.info(f"USER: {san_user}")

    # NLP + orchestrator
    nlp_data = analyze_text(san_user)
    with st.expander("🔎 NLP insight", expanded=False):
        st.write(nlp_data)

    # >>> NUOVO: salviamo snapshot dell'ULTIMO input + predizione (PERSISTE tra i rerun)
    _pred = str(nlp_data.get("intent", "general") or "general")
    _score = float(nlp_data.get("score", 0.0) or 0.0)
    st.session_state["corr_snapshot"] = {
        "text": san_user,
        "predicted_intent": _pred,
        "predicted_score": _score,
    }
    st.caption("🖊️ Puoi correggere l'intent di questo messaggio nel pannello in fondo alla pagina.")

    engine_now = st.session_state.get("engine", "openai")
    # Prompt arricchito (di default)
    enriched_user = compose_prompt(san_user, nlp_data)
    # Su OLLAMA inviamo SOLO la domanda pulita
    if engine_now == "ollama":
        enriched_user = san_user

    intent = nlp_data.get("intent", "general")
    system_text = system_prompt_for_intent(intent)
    if engine_now == "ollama":
        system_text += " Mantieni rigorosamente la lingua italiana per tutta la risposta; non passare a inglese o spagnolo. Se accade, correggiti e torna subito all'italiano."
    didactic_suffix = (
        "\n\nStile didattico: organizza a sezioni con titoli brevi, definizioni chiare, esempi pratici e, in chiusura, 3 domande-quiz con relative risposte."
        if st.session_state.get("didactic", True) else ""
    )

    # History corta + sanitizzazione
    short_history = _short_history_by_chars(st.session_state.history[:-1], max_chars=9000)
    short_history = [{"role": m["role"], "content": _sanitize_meta(m.get("content",""))} for m in short_history]

    base_messages = [{"role": "system", "content": system_text}] + short_history + [
        {"role": "user", "content": enriched_user + didactic_suffix}
    ]

    with st.chat_message("assistant"):
        t0 = time.time()
        reply = ""

        try:
            if st.session_state.get("streaming_on", True):
                placeholder = st.empty()
                pieces = []
                for delta in stream_chat(base_messages, model_override=(
                    st.session_state["openai_model"] if engine_now == "openai"
                    else st.session_state["hf_model"] if engine_now == "hugging"
                    else st.session_state["ollama_model"]
                )):
                    if not isinstance(delta, str) or not delta.strip():
                        continue
                    pieces.append(delta)
                    text = "".join(pieces)
                    text = _strip_drift_lines(_strip_drift_prefix(text))
                    placeholder.markdown(text)
                reply = "".join(pieces).strip()
                if not reply:
                    raise RuntimeError("Nessun testo dallo stream")
            else:
                reply = call_chat(base_messages, model_override=(
                    st.session_state["openai_model"] if engine_now == "openai"
                    else st.session_state["hf_model"] if engine_now == "hugging"
                    else st.session_state["ollama_model"]
                ))
                reply = _strip_drift_lines(_strip_drift_prefix(reply))
                st.markdown(reply)

            # AUTO-CONTINUE: se si ferma a metà (Ollama) e non c'è <<FINE>>
            if engine_now == "ollama" and _looks_cutoff(reply) and "<<FINE>>" not in reply:
                msgs = build_continue_only_messages(
                    system_once=system_text,
                    history=[{"role":"user","content": enriched_user + didactic_suffix},
                             {"role":"assistant","content": reply}],
                    final_text_tail=_tail(reply, 3000),
                    round_words=800,
                    didactic=st.session_state.get("didactic", False),
                )
                more = _strip_drift_lines(_strip_drift_prefix(call_chat_smart(msgs, temperature=0.7)))
                if more and more.strip():
                    more = more.replace("<<FINE>>", "").strip()
                    reply = (reply + ("\n\n" if not reply.endswith("\n\n") else "") + more).strip()
                    if st.session_state.get("streaming_on", True):
                        placeholder.markdown(reply)
                    else:
                        st.markdown(reply)

        except Exception as e:
            if not st.session_state.get("engine_lock", False):
                try:
                    reply = call_chat_smart(base_messages, model_override=(
                        st.session_state["openai_model"] if engine_now == "openai"
                        else st.session_state["hf_model"] if engine_now == "hugging"
                        else st.session_state["ollama_model"]
                    ))
                    reply = _strip_drift_lines(_strip_drift_prefix(reply))
                    st.markdown(reply)
                except Exception as e2:
                    reply = f"⚠️ Errore modello: {str(e2).splitlines()[0]}"
                    st.markdown(reply)
            else:
                reply = f"⚠️ Errore modello: {str(e).splitlines()[0]}"
                st.markdown(reply)

        log.info(f"ENGINE={engine_now} THINKING={st.session_state.get('thinking', False)} ELAPSED={time.time()-t0:.1f}s")

    st.session_state.history.append({"role": "assistant", "content": reply})
    if st.session_state.get("persist", True):
        save_memory(st.session_state.history)

# === PANNELLO PERSISTENTE: Correzione intent dell’ultimo messaggio ===========
st.divider()
st.subheader("🖊️ Correzione intent dell’ultimo messaggio")

_snapshot = st.session_state.get("corr_snapshot")
if not _snapshot:
    st.caption("Invia un messaggio: qui potrai correggerne l'intent.")
else:
    _text = _snapshot.get("text") or ""
    _pred = _snapshot.get("predicted_intent") or "general"
    _score = float(_snapshot.get("predicted_score") or 0.0)
    st.write(f"**Testo:** “{_text[:160]}{'…' if len(_text)>160 else ''}”")
    st.write(f"**Predetto:** `{_pred}` — confidenza: {_score:.2f}")

    _labels = ["coding","business","nutrition","calendar","study","health","motivation","finance","science","general"]
    try:
        _default_idx = _labels.index(_pred)
    except Exception:
        _default_idx = _labels.index("general")

    _corr = st.selectbox("Seleziona l'intent corretto", _labels, index=_default_idx, key="corr_global_select")
    if st.button("💾 Salva correzione intent", key="corr_global_save"):
        ok, err = write_nlp_log({
            "text": _text,
            "predicted_intent": _pred,
            "predicted_score": _score,
            "correct_intent": _corr,
            "source": "streamlit_correction"
        })
        if ok:
            st.success("Correzione salvata nei log.")
        else:
            st.error(f"Correzione NON salvata: {err}")
        st.caption(f"File log correzioni: `{LOG_FILE}`")

    # Expander: vedere subito le ultime correzioni
    with st.expander("🗂️ Correzioni recenti (streamlit_correction)"):
        try:
            st.caption(f"Percorso log: `{LOG_FILE}`")
            rows = []
            if LOG_FILE.exists():
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    lines = [ln for ln in f if '"source":"streamlit_correction"' in ln]
                    for ln in lines[-10:]:
                        obj = json.loads(ln)
                        rows.append(
                            f"- `{obj.get('correct_intent')}` ← \"{(obj.get('text') or '')[:100]}\"  "
                            f"(pred: {obj.get('predicted_intent')} {obj.get('predicted_score')})"
                        )
            st.markdown("\n".join(rows) if rows else "_Nessuna correzione salvata ancora._")
        except Exception as _e:
            st.info(f"Log non leggibile: {_e}")
