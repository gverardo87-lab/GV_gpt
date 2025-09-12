# core/engine.py
# -----------------------------------------------------------------------------
# Orchestratore engine OpenAI ↔ Ollama (chat-first)
# Espone:
#   - call_chat(messages)            → risposta intera (no stream)
#   - stream_chat(messages)          → generator di stringhe (stream)
#   - call_chat_smart(messages)      → con fallback bidirezionale (rispetta lock)
#
# Env utili:
#   GV_ENGINE=openai|ollama              (default openai)
#   GV_ENGINE_LOCK=1|0                   (se 1 → NO fallback)
#   OPENAI_MODEL=gpt-4o-mini (default)
#   OPENAI_API_KEY=...
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_MODEL=phi3:3.8b (default)
#   OLLAMA_NUM_PREDICT, OLLAMA_NUM_CTX, OLLAMA_NUM_THREAD (opzionali)
#   GV_OLLAMA_TIMEOUT=180 (secondi)      (opzionale)
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import time
from typing import List, Dict, Any, Generator, Optional

from core.logger import get_logger
from core.ollama_client import (
    call_ollama_chat,
    stream_ollama_chat,
    call_ollama_generate,              # usato solo se serve fallback legacy
    build_simple_prompt_from_messages,  # per eventuale generate-fallback
)

log = get_logger()

# ==== Utility =================================================================

def _engine() -> str:
    return (os.getenv("GV_ENGINE") or "openai").strip().lower()

def _engine_lock() -> bool:
    v = (os.getenv("GV_ENGINE_LOCK") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")

def _openai_model() -> str:
    return (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()

def _ollama_timeout() -> int:
    try:
        return int(os.getenv("GV_OLLAMA_TIMEOUT") or "180")
    except Exception:
        return 180

def _has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

def _looks_rate_limit(e: Exception) -> bool:
    s = str(e).lower()
    return "rate limit" in s or "429" in s

# ==== OpenAI helpers ==========================================================

def _get_openai_client():
    """
    Ritorna una tupla (mode, client):
      mode = "new"  → SDK moderno `openai.OpenAI`
      mode = "old"  → SDK legacy `openai`
    """
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return "new", client
    except Exception:
        try:
            import openai  # type: ignore
            openai.api_key = os.getenv("OPENAI_API_KEY")
            return "old", openai
        except Exception as e:
            raise RuntimeError("OpenAI SDK non disponibile") from e

def _openai_call(messages: List[Dict[str, str]], *, model: Optional[str] = None, temperature: float = 0.7, timeout: int = 90) -> str:
    if not _has_openai_key():
        raise RuntimeError("OPENAI_API_KEY mancante")
    mode, client = _get_openai_client()
    model = model or _openai_model()

    if mode == "new":
        # SDK moderno
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,   # richiede openai>=1.51; se assente viene ignorato
        )
        return (resp.choices[0].message.content or "").strip()
    else:
        # SDK legacy
        resp = client.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            request_timeout=timeout,
        )
        return (resp.choices[0]["message"]["content"] or "").strip()

def _openai_stream(messages: List[Dict[str, str]], *, model: Optional[str] = None, temperature: float = 0.7, timeout: int = 120) -> Generator[str, None, None]:
    if not _has_openai_key():
        # niente eccezione qui: facciamo fallire subito sul consumer
        yield "[OpenAI] OPENAI_API_KEY mancante"
        return

    mode, client = _get_openai_client()
    model = model or _openai_model()

    try:
        if mode == "new":
            stream = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, stream=True, timeout=timeout
            )
            for chunk in stream:
                delta = (chunk.choices[0].delta.content or "") if chunk.choices and chunk.choices[0].delta else ""
                if delta:
                    yield delta
        else:
            # legacy
            stream = client.ChatCompletion.create(
                model=model, messages=messages, temperature=temperature, stream=True, request_timeout=timeout
            )
            for chunk in stream:
                try:
                    delta = chunk["choices"][0]["delta"].get("content") or ""
                except Exception:
                    delta = ""
                if delta:
                    yield delta
    except Exception as e:
        yield f"\n[OpenAI stream error] {e}\n"

# ==== OLLAMA wrappers =========================================================

def _ollama_call(messages: List[Dict[str, str]], *, temperature: float = 0.7) -> str:
    # /api/chat (no stream) — preferito per stabilità
    return call_ollama_chat(
        messages,
        temperature=temperature,
        timeout=_ollama_timeout(),
        # Le options anti-ripetizione/mirostat sono di default nel client
    )

def _ollama_stream(messages: List[Dict[str, str]], *, temperature: float = 0.7) -> Generator[str, None, None]:
    return stream_ollama_chat(
        messages,
        temperature=temperature,
        timeout=max(_ollama_timeout(), 180),
    )

# ==== API pubblica ============================================================

def call_chat(messages: List[Dict[str, str]], *, temperature: float = 0.7) -> str:
    """
    Esegue la chiamata sul motore selezionato (senza fallback).
    """
    eng = _engine()
    t0 = time.time()
    try:
        if eng == "ollama":
            out = _ollama_call(messages, temperature=temperature)
        else:
            out = _openai_call(messages, temperature=temperature)
        log.info(f"call_chat engine={eng} t={time.time()-t0:.2f}s len={len(out)}")
        return out
    except Exception as e:
        log.error(f"call_chat failed engine={eng}: {e}")
        raise

def stream_chat(messages: List[Dict[str, str]], *, temperature: float = 0.7) -> Generator[str, None, None]:
    """
    Streaming sul motore selezionato (senza fallback).
    """
    eng = _engine()
    if eng == "ollama":
        yield from _ollama_stream(messages, temperature=temperature)
    else:
        yield from _openai_stream(messages, temperature=temperature)

def call_chat_smart(messages: List[Dict[str, str]], *, temperature: float = 0.7) -> str:
    """
    Scelta intelligente con fallback bidirezionale:
      - GV_ENGINE=openai → prova OpenAI, se 429/quota o errori → fallback a Ollama (se lock OFF)
      - GV_ENGINE=ollama → prova Ollama, se errore/vuoto → fallback a OpenAI (se lock OFF)
    Ritorna SEMPRE una stringa (anche messaggio d'errore formattato).
    """
    eng = _engine()
    lock = _engine_lock()
    primary_err: Optional[Exception] = None

    # --- branch OpenAI first --------------------------------------------------
    if eng == "openai":
        try:
            return _openai_call(messages, temperature=temperature)
        except Exception as e:
            primary_err = e
            log.warning(f"OpenAI primary failed: {e}")
            if lock:
                return f"⚠️ OpenAI errore: {str(e).splitlines()[0]}"
            # Condizioni tipiche per fallback: rate limit/quota, auth key mancante
            if _looks_rate_limit(e) or not _has_openai_key() or "api key" in str(e).lower():
                try:
                    txt = _ollama_call(messages, temperature=temperature)
                    if txt and txt.strip():
                        return txt
                except Exception as e2:
                    log.error(f"Ollama fallback failed: {e2}")
                    return f"⚠️ OpenAI errore: {str(e).splitlines()[0]} — Fallback Ollama fallito: {str(e2).splitlines()[0]}"
            else:
                return f"⚠️ OpenAI errore: {str(e).splitlines()[0]}"

    # --- branch Ollama first --------------------------------------------------
    else:
        try:
            txt = _ollama_call(messages, temperature=temperature)
            if txt and txt.strip():
                return txt
            raise RuntimeError("Ollama ha restituito stringa vuota")
        except Exception as e:
            primary_err = e
            log.warning(f"Ollama primary failed: {e}")
            if lock:
                return f"⚠️ Ollama errore: {str(e).splitlines()[0]}"
            # fallback verso OpenAI (se key presente)
            if _has_openai_key():
                try:
                    return _openai_call(messages, temperature=temperature)
                except Exception as e2:
                    log.error(f"OpenAI fallback failed: {e2}")
                    return f"⚠️ Ollama errore: {str(e).splitlines()[0]} — Fallback OpenAI fallito: {str(e2).splitlines()[0]}"
            else:
                return f"⚠️ Ollama errore: {str(e).splitlines()[0]} — Nessuna chiave OpenAI per fallback"

# ==== Helper opzionale: continue-only builder ================================
def build_continue_only_messages(
    *,
    system_once: str,
    history: List[Dict[str, str]],
    final_text_tail: str,
    round_words: int = 600,
    didactic: bool = False,
) -> List[Dict[str, str]]:
    """
    Costruisce i messages per un round di continuazione "pulito":
      - NON ripete RUOLO/DOMANDA in modo ridondante
      - Include la storia reale già avvenuta (history)
      - Chiude con un messaggio utente minimal: continue-only + tail
    """
    msgs: List[Dict[str, str]] = []
    # Assicurati un solo system all'inizio:
    if system_once:
        msgs.append({"role": "system", "content": system_once})
    # Storia: copia i turni esistenti (user/assistant) senza aggiungere altro
    for m in history:
        r = m.get("role")
        if r in ("user", "assistant"):
            msgs.append({"role": r, "content": m.get("content", "")})

    suffix = (
        "\n\n[STILE DIDATTICO] Struttura a sezioni con titoli brevi, definizioni chiare, esempi pratici e, alla fine, 3 domande quiz con risposte."
        if didactic else ""
    )
    user_continue = (
        "Continua esattamente dal seguente frammento. Non ricominciare, non riassumere, non ripetere titoli."
        f" Mantieni lo stesso stile. Lunghezza target ~{round_words} parole. Se completi l'argomento, scrivi esattamente <<FINE>>."
        f"\n\n[Frammento finale]\n{final_text_tail}{suffix}"
    )
    msgs.append({"role": "user", "content": user_continue})
    return msgs

# ==== (opzionale) Fallback generate da messages ==============================
def generate_from_messages_with_fallback(messages: List[Dict[str, str]], *, temperature: float = 0.7) -> str:
    """
    Se per qualche motivo vuoi forzare /api/generate (template instruct), converte i messages in un unico prompt.
    """
    prompt = build_simple_prompt_from_messages(messages)
    try:
        return call_ollama_generate(prompt, temperature=temperature)
    except Exception as e:
        return f"[Errore generate] {e}"
