# server/app_streamlit.py
# ─────────────────────────────────────────────────────────────────────────────
# GV GPT Custom — Demo Middleware (Streamlit)
# - Toggle engine (OpenAI ↔ Ollama) + scelta modello (autodiscovery Ollama)
# - 🔒 Engine Lock: evita qualsiasi fallback tra engine se attivo
# - Fallback smart (solo se lock disattivo): OpenAI→Ollama su 429/quota; Ollama→OpenAI su errore/vuoto
# - Didattica + Thinking longform “Auto a budget” (/api/generate + bozza parziale, guardie anti-ripetizione)
# - Streaming: OpenAI (nativo), Ollama (sperimentale su /chat)
# - Memoria persistente, export, diagnostica, logging
# Default Ollama se non definito: phi3:3.8b
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
from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Default Ollama model se non in .env
if not os.getenv("OLLAMA_MODEL"):
    os.environ["OLLAMA_MODEL"] = "phi3:3.8b"

import streamlit as st
import requests  # per autodiscovery modelli Ollama

# 2) Import moduli progetto
from core.engine import call_chat, stream_chat, call_chat_smart
from core.memory import load_memory, save_memory, clear_memory
from core.logger import get_logger
from nlp_layer.preprocessing import analyze_text
from orchestrator.orchestrator import compose_prompt
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

# Import per longform diretto su /api/generate
from core.ollama_client import call_ollama_generate

log = get_logger()

# 3) Helpers export
def export_chat_md(history: list) -> str:
    lines = ["# Conversazione GV GPT\n"]
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
        "messages": history,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

# 3.b Helper: override temporaneo di variabili d’ambiente
@contextmanager
def temp_env(**kwargs):
    old = {}
    for k, v in kwargs.items():
        old[k] = os.environ.get(k)
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

def _count_words(txt: str) -> int:
    return len((txt or "").split())

def _discover_ollama_models(base_url: str) -> list[str]:
    """Rileva i modelli locali di Ollama via /api/tags; fallback a suggerimenti noti."""
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

# ── Helper Longform Auto a budget ────────────────────────────────────────────
def _tail(text: str, chars: int = 3200) -> str:
    return (text or "")[-chars:]

def _longform_continue_prompt(system_text: str, user_original: str, partial_tail: str,
                              didactic: bool, round_words: int, round_idx: int) -> str:
    """Prompt di continuazione RIGIDO: continua e basta, niente restart/riassunti."""
    style = ""
    if didactic:
        style = (
            "\n\n[STILE DIDATTICO] Struttura a sezioni con titoli brevi, definizioni chiare,"
            " esempi pratici e, alla fine (solo nell'ultimo chunk), 3 domande quiz con risposte."
        )
    partial_block = ""
    if partial_tail:
        partial_block = f"\n\n[CONTESTO_FINALE]\n{partial_tail}\n"

    return (
        f"[RUOLO]\n{system_text}\n"
        f"[DOMANDA]\n{user_original}\n"
        f"{partial_block}"
        f"[ISTRUZIONI CHUNK]\n"
        f"- Stai scrivendo il CHUNK #{round_idx}. Non riavviare la risposta.\n"
        f"- Continua esattamente dal CONTESTO_FINALE.\n"
        f"- NON ripetere titoli o introduzioni già scritte.\n"
        f"- Lunghezza massima CHUNK: ~{round_words} parole.\n"
        f"- Se completi l'argomento termina con <<FINE>>.\n"
        f"- Vietato ricominciare dall'inizio o fare riassunti di quanto già scritto.\n"
        f"{style}\n\n[RISPOSTA_CHUNK]\n"
    ).strip()

def _word_count(s: str) -> int:
    return len(re.findall(r"\w+", s or ""))

def _looks_restart(chunk: str) -> bool:
    """Heuristica: il chunk sembra ricominciare da capo (titoli comuni/intro ripetitive)."""
    head = (chunk or "").strip().lower()[:400]
    patterns = [
        r"^introduzione\b", r"^capitolo\s+\d+\b", r"^in\s+questa\s+risposta\b",
        r"^la\s+storia\s+di\b", r"^genova\b", r"^pegli\b"
    ]
    return any(re.search(p, head) for p in patterns)

def _is_redundant(chunk: str, acc: str) -> bool:
    """Troppe sovrapposizioni col finale già scritto → probabilmente ripetizione."""
    tail = _tail(acc, 1200).lower()
    c = (chunk or "").lower()
    if not tail or not c:
        return False
    # Jaccard grezzo su parole comuni delle prime ~200 parole del chunk
    c_words = set(re.findall(r"\w+", c)[:200])
    t_words = set(re.findall(r"\w+", tail))
    if not c_words or not t_words:
        return False
    overlap = len(c_words & t_words) / max(1, len(c_words | t_words))
    return overlap > 0.55 or c[:150] in tail

# ─────────────────────────────────────────────────────────────────────────────
# 4) UI base
st.set_page_config(page_title="GV GPT Custom", page_icon="🤖", layout="centered")
st.title("🤖 GV GPT Custom — Demo Middleware")

# Stato persistente e preferenze di default
st.session_state.setdefault("persist", True)
st.session_state.setdefault("engine", (os.getenv("GV_ENGINE", "openai") or "openai").lower())
st.session_state.setdefault("openai_model", os.getenv("OPENAI_MODEL") or "gpt-4o-mini")
st.session_state.setdefault("ollama_model", os.getenv("OLLAMA_MODEL") or "phi3:3.8b")
st.session_state.setdefault("ollama_base", os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434")
st.session_state.setdefault("engine_lock", (os.getenv("GV_ENGINE_LOCK", "0").lower() in ("1","true","yes","on")))

# Sidebar
with st.sidebar:
    st.header("🔀 Engine & Modelli")

    # Toggle engine
    eng_display = "OpenAI" if st.session_state["engine"] == "openai" else "Ollama"
    engine_choice = st.radio("Seleziona engine:", ["OpenAI", "Ollama"], index=0 if eng_display == "OpenAI" else 1)

    # OpenAI: scelta modello
    st.subheader("OpenAI")
    openai_suggestions = ["gpt-4o-mini", "gpt-4o", "o4-mini", "gpt-4.1-mini", "Custom…"]
    current_openai = st.session_state["openai_model"]
    if current_openai not in openai_suggestions:
        openai_suggestions.insert(-1, current_openai)
    openai_sel = st.selectbox("Modello OpenAI", options=openai_suggestions,
                              index=openai_suggestions.index(current_openai) if current_openai in openai_suggestions else len(openai_suggestions)-1)
    if openai_sel == "Custom…":
        current_openai = st.text_input("Modello OpenAI (custom)", value=st.session_state["openai_model"])
    else:
        current_openai = openai_sel

    # Ollama: base URL + modelli locali
    st.subheader("Ollama (locale)")
    base_val = st.text_input("Base URL", value=st.session_state["ollama_base"], help="Di solito http://localhost:11434")
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

    # Applica preferenze
    if st.button("✅ Applica & usa questi modelli"):
        st.session_state["engine"] = "openai" if engine_choice == "OpenAI" else "ollama"
        st.session_state["openai_model"] = (current_openai or "gpt-4o-mini").strip()
        st.session_state["ollama_model"] = (current_ollama or "phi3:3.8b").strip()
        st.session_state["ollama_base"]  = (base_val or "http://localhost:11434").strip()
        # allinea ambiente per il backend
        os.environ["GV_ENGINE"]        = st.session_state["engine"]
        os.environ["OPENAI_MODEL"]     = st.session_state["openai_model"]
        os.environ["OLLAMA_MODEL"]     = st.session_state["ollama_model"]
        os.environ["OLLAMA_BASE_URL"]  = st.session_state["ollama_base"]
        st.success(f"Impostato: {engine_choice} • OpenAI={st.session_state['openai_model']} • Ollama={st.session_state['ollama_model']}")
        st.rerun()

    st.header("🔒 Lock engine")
    lock = st.checkbox("Blocca engine (niente fallback automatico)", value=st.session_state["engine_lock"],
                       help="Se attivo, non passerà mai automaticamente all'altro engine.")
    st.session_state["engine_lock"] = lock
    os.environ["GV_ENGINE_LOCK"] = "1" if lock else "0"

    st.header("⚙️ Impostazioni rapide")
    if st.session_state["engine"] == "ollama":
        st.caption(f"Engine: **ollama** • Modello: **{st.session_state['ollama_model']}**")
    else:
        st.caption(f"Engine: **openai** • Modello: **{st.session_state['openai_model']}**")

    if st.button("🧹 Svuota chat"):
        st.session_state.history = []
        st.rerun()

    st.header("🧠 Memoria")
    st.checkbox("Mantieni chat tra riavvii", key="persist",
                help="Salva gli ultimi messaggi su disco (data/memory_default.json).")
    if st.button("🗑️ Cancella memoria salvata"):
        clear_memory()
        st.success("Memoria persistente cancellata.")

    st.header("🎓 Modalità didattica")
    didactic = st.checkbox(
        "Spiega passo-passo (sezioni, esempi, mini-quiz)",
        value=True,
        help="Istruzioni di stile didattiche (no catene di pensiero interne)."
    )

    st.header("🧠 Thinking (longform)")
    thinking = st.checkbox(
        "Attiva modalità lunga (auto-continue)",
        value=False,
        help="Genera in più round fino alla lunghezza target o fino a <<FINE>>."
    )
    target_label = st.select_slider(
        "Lunghezza desiderata",
        options=["~400 parole", "~800 parole", "~1200 parole", "~2000 parole"],
        value="~800 parole"
    )
    target_words_map = {"~400 parole": 400, "~800 parole": 800, "~1200 parole": 1200, "~2000 parole": 2000}
    target_words = target_words_map[target_label]
    max_rounds = st.slider("Max round", 1, 10, 6, help="Quanti cicli di continuazione consentire.")

    st.header("📝 Lunghezza per round (solo Ollama)")
    length_label = st.radio(
        "Preset round:",
        ["Breve", "Media", "Lunga", "Very Long", "Very Long+"],
        index=2,
        help="Token massimi per singolo round di Ollama."
    )
    length_presets = {
        "Breve":     {"pred": 120,   "ctx": 1024, "timeout": 40,  "note": "4–6 frasi"},
        "Media":     {"pred": 256,   "ctx": 1536, "timeout": 60,  "note": "2–4 paragrafi"},
        "Lunga":     {"pred": 512,   "ctx": 2048, "timeout": 120, "note": "6–10 paragrafi"},
        "Very Long": {"pred": 1200,  "ctx": 3072, "timeout": 220, "note": "≈1300 parole"},
        "Very Long+":{"pred": 2200,  "ctx": 4096, "timeout": 360, "note": "≈2200+ parole"},
    }
    preset = length_presets[length_label]
    st.caption(f"Preset round: {preset['pred']} tok, ctx {preset['ctx']}, timeout {preset['timeout']}s — {preset['note']}")

    st.header("⚡ Streaming (Ollama)")
    ollama_stream = st.checkbox(
        "Streaming live (sperimentale)",
        value=False,
        help="Se attivo, Ollama invia testo mentre elabora. Disattivalo se vedi instabilità."
    )
    st.session_state["ollama_stream"] = ollama_stream

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
                from core.ollama_client import call_ollama_generate
                reply, diag = call_ollama_generate("Di' soltanto: OK", debug=True)
                st.success("Diagnostica completata.")
                st.write("**Risposta:**", reply[:500])
                st.json(diag)
            except Exception as e:
                st.error(f"Diagnostica fallita: {e}")

# Avviso se manca la chiave (solo per engine openai)
if st.session_state["engine"] == "openai" and not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OPENAI_API_KEY non trovato. Crea un file `.env` nella root del progetto.")

# 5) Stato conversazione (usa memoria persistente se presente)
if "history" not in st.session_state:
    st.session_state.history = load_memory()

# 6) Mostra conversazione
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- LONGFORM PROMPTS (base compat) ----------
def _longform_prompt(system_text: str, user_original: str, partial_tail: str, didactic: bool, target_words: int) -> str:
    style = ""
    if didactic:
        style = (
            "\n\n[STILE DIDATTICO] Struttura a sezioni con titoli, definizioni chiare,"
            " esempi pratici e, alla fine, 3 domande quiz con risposte."
        )
    partial_block = ""
    if partial_tail:
        partial_block = f"\n\n[BOZZA_PARZIALE]\n{partial_tail}\n"
    return (
        f"[ISTRUZIONI]\n{system_text}\n"
        f"[UTENTE]\n{user_original}\n"
        f"{partial_block}"
        f"[COMPITO]\n"
        f"Scrivi la risposta completa e coerente, senza ripetere testo già scritto."
        f" Approfondisci fino a circa {target_words} parole complessive (tolleranza ±20%)."
        f" Mantieni lo stesso stile della bozza. Se completi l'argomento scrivi esattamente <<FINE>> alla fine.\n"
        f"{style}\n\n[RISPOSTA]\n"
    ).strip()

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

# 6.b Continua ultima risposta (senza input utente) — 1 round forte, senza loop
if st.session_state.get("do_continue"):
    st.session_state.pop("do_continue", None)
    last_user_text, last_assistant_text = _find_last_user_and_assistant(st.session_state.history)
    intent = "general"
    system_text = system_prompt_for_intent(intent)

    with st.chat_message("assistant"):
        t0 = time.time()
        engine_now = st.session_state["engine"]
        if engine_now == "ollama":
            # Longform Auto a budget: 1 giro di continuazione “forte”
            ctx_val = max(preset["ctx"], 3072 if preset.get("pred",0) >= 1200 else preset["ctx"])
            with temp_env(
                OLLAMA_NUM_PREDICT=preset["pred"],
                OLLAMA_NUM_CTX=min(ctx_val, 4096),
                GV_OLLAMA_TIMEOUT=max(preset["timeout"], 220),
                OLLAMA_MODEL=st.session_state["ollama_model"],
                OLLAMA_BASE_URL=st.session_state["ollama_base"],
            ):
                try:
                    prompt = _longform_continue_prompt(
                        system_text, last_user_text, _tail(last_assistant_text, 3200),
                        didactic=False, round_words=800, round_idx=1
                    )
                    chunk = call_ollama_generate(prompt)
                    # guardie anti-ripetizione
                    if _looks_restart(chunk) or _is_redundant(chunk, last_assistant_text):
                        chunk = re.sub(r'(?is)^.*?\n', '', chunk, count=2).strip()
                    reply = (chunk or "").replace("<<FINE>>","").strip()
                    st.markdown(reply if reply else "_(nessun avanzamento)_")
                except Exception as e2:
                    reply = "⚠️ Errore (continua): " + str(e2).split("\n")[0]
                    st.markdown(reply)
        else:
            # OpenAI: resta su OpenAI (lock rispettato); niente fallback automatico
            try:
                os.environ["GV_ENGINE"] = "openai"
                os.environ["OPENAI_MODEL"] = st.session_state["openai_model"]
                msg = [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content":
                        "Continua dal punto in cui ti sei interrotto. Non ripetere. "
                        "Lunghezza ~600-800 parole. <<FINE>> se completi."},
                    {"role": "assistant", "content": _tail(last_assistant_text, 3000)},
                ]
                reply = call_chat(msg)  # non-stream OpenAI
                reply = reply.replace("<<FINE>>","").strip()
                st.markdown(reply if reply else "_(nessun avanzamento)_")
            except Exception as e1:
                if st.session_state.get("engine_lock", False):
                    reply = f"⚠️ OpenAI errore: {str(e1).splitlines()[0]}"
                    st.markdown(reply)
                else:
                    # fallback smart solo se lock OFF (potrebbe andare su Ollama solo per quota)
                    reply = call_chat_smart([
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": "Continua senza ripetere. <<FINE>> alla fine."},
                        {"role": "assistant", "content": _tail(last_assistant_text, 3000)},
                    ]).replace("<<FINE>>","").strip()
                    st.markdown(reply if reply else "_(nessun avanzamento)_")

        log.info(f"ENGINE={engine_now} CONTINUE ELAPSED={time.time()-t0:.1f}s")

    st.session_state.history.append({"role": "assistant", "content": reply})
    if st.session_state.get("persist", True):
        save_memory(st.session_state.history)

# 7) Input utente
if user := st.chat_input("Scrivi qui…"):
    st.session_state.history.append({"role": "user", "content": user})
    log.info(f"USER: {user}")

    # NLP + orchestrator
    nlp_data = analyze_text(user)
    log.info(f"NLP: {nlp_data}")
    with st.expander("🔎 NLP insight", expanded=False):
        st.write(nlp_data)

    enriched_user = compose_prompt(user, nlp_data)
    intent = nlp_data.get("intent", "general")
    system = {"role": "system", "content": system_prompt_for_intent(intent)}

    didactic_suffix = ""
    if didactic:
        didactic_suffix = (
            "\n\n[STILE DIDATTICO] Struttura a sezioni con titoli, definizioni chiare, esempi pratici e, alla fine, 3 domande quiz con risposte."
        )

    if st.session_state["engine"] == "ollama":
        messages = [system, {"role": "user", "content": enriched_user + didactic_suffix}]
    else:
        messages = [system] + st.session_state.history[:-1] + [
            {"role": "user", "content": enriched_user + didactic_suffix}
        ]

    with st.chat_message("assistant"):
        t0 = time.time()

        if thinking and st.session_state["engine"] == "ollama":
            # Longform Auto a budget: continua finché budget > 0 o <<FINE>>
            placeholder = st.empty()
            acc = ""
            rounds = 0

            def _render():
                placeholder.markdown(acc if acc else "_(sto generando…)_")

            # calcola budget parole rimanenti
            budget = target_words
            ctx_val = max(preset["ctx"], 3072 if preset.get("pred",0) >= 1200 else preset["ctx"])

            while rounds < max_rounds and budget > 0:
                round_words = min(budget, 600 if preset["pred"] < 1000 else 1000)
                with temp_env(
                    OLLAMA_NUM_PREDICT=preset["pred"],
                    OLLAMA_NUM_CTX=min(ctx_val, 4096),
                    GV_OLLAMA_TIMEOUT=max(preset["timeout"], 220),
                    OLLAMA_MODEL=st.session_state["ollama_model"],
                    OLLAMA_BASE_URL=st.session_state["ollama_base"],
                ):
                    try:
                        prompt = _longform_continue_prompt(
                            system["content"], enriched_user + didactic_suffix,
                            _tail(acc, 3200), didactic, round_words, round_idx=rounds+1
                        )
                        chunk = call_ollama_generate(prompt)
                    except Exception as e:
                        chunk = f"\n[Errore Ollama] {str(e).splitlines()[0]}"

                if not chunk or not str(chunk).strip():
                    break

                # guardie anti-ripetizione/restart
                if _looks_restart(chunk) or _is_redundant(chunk, acc):
                    # prova a “tagliare” un possibile header ripetuto
                    chunk = re.sub(r'(?is)^.{0,300}\n', '', chunk, count=1).strip()

                # append
                acc += (("\n\n" if acc else "") + str(chunk).strip())
                _render()

                # stop conditions
                if "<<FINE>>" in acc:
                    acc = acc.replace("<<FINE>>", "").strip()
                    break

                wrote = _word_count(chunk)
                budget -= max(0, wrote)
                rounds += 1

            reply = acc.strip() or "_Nessuna risposta generata._"
            st.markdown(reply)

        else:
            if st.session_state["engine"] == "ollama":
                with temp_env(
                    OLLAMA_NUM_PREDICT=preset["pred"],
                    OLLAMA_NUM_CTX=preset["ctx"],
                    GV_OLLAMA_TIMEOUT=preset["timeout"],
                    OLLAMA_MODEL=st.session_state["ollama_model"],
                    OLLAMA_BASE_URL=st.session_state["ollama_base"],
                ):
                    try:
                        if st.session_state.get("ollama_stream", False):
                            try:
                                reply = st.write_stream(stream_chat(messages))  # /api/chat streaming
                                if not reply or not str(reply).strip():
                                    reply = call_chat_smart(messages)
                                    st.markdown(reply)
                            except Exception:
                                reply = call_chat_smart(messages)
                                st.markdown(reply)
                        else:
                            reply = call_chat_smart(messages)  # generate-first → chat → OpenAI (se lock OFF)
                            st.markdown(reply.replace("<<FINE>>","").strip())
                    except Exception as e2:
                        reply = "⚠️ Errore modello (Ollama/OpenAI): " + str(e2).split("\n")[0]
                        st.markdown(reply)
            else:
                # ENGINE = OpenAI
                try:
                    os.environ["GV_ENGINE"] = "openai"
                    os.environ["OPENAI_MODEL"] = st.session_state["openai_model"]

                    # 1) prova streaming OpenAI
                    reply = None
                    try:
                        reply = st.write_stream(stream_chat(messages))
                    except Exception as e_stream:
                        log.warning(f"OpenAI streaming failed: {e_stream}")

                    # 2) se streaming non ha prodotto testo, prova OpenAI non-stream
                    if not reply or (isinstance(reply, str) and not reply.strip()):
                        try:
                            reply = call_chat(messages)
                            st.markdown(reply)
                        except Exception as e_ns:
                            if st.session_state.get("engine_lock", False):
                                reply = f"⚠️ OpenAI errore: {str(e_ns).splitlines()[0]}"
                                st.markdown(reply)
                            else:
                                # 3) lock OFF: fallback smart (può andare su Ollama SOLO per quota/429)
                                reply = call_chat_smart(messages)
                                st.markdown(reply)
                except Exception as e:
                    reply = "⚠️ Errore modello (OpenAI): " + str(e).split("\n")[0]
                    st.markdown(reply)

        log.info(f"ENGINE={st.session_state['engine']} THINKING={thinking} TARGET={target_words} ELAPSED={time.time()-t0:.1f}s")

    st.session_state.history.append({"role": "assistant", "content": reply})
    if st.session_state.get("persist", True):
        save_memory(st.session_state.history)
