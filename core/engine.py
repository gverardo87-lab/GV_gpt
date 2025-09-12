# core/engine.py
import os
from core.gpt_clienti import call_gpt_chat, stream_gpt_chat
from core.ollama_client import (
    call_ollama_chat, stream_ollama_chat,
    call_ollama_generate, build_simple_prompt_from_messages
)

def _engine():
    return os.getenv("GV_ENGINE", "openai").lower()

def _locked() -> bool:
    v = os.getenv("GV_ENGINE_LOCK", "0").lower().strip()
    return v in ("1", "true", "yes", "on")

def call_chat(messages, temperature=0.7):
    if _engine() == "ollama":
        return call_ollama_chat(messages, temperature=temperature,
                                timeout=int(os.getenv("GV_OLLAMA_TIMEOUT","60")))
    return call_gpt_chat(messages, temperature=temperature)

def stream_chat(messages, temperature=0.7):
    if _engine() == "ollama":
        return stream_ollama_chat(messages, temperature=temperature)
    return stream_gpt_chat(messages, temperature=temperature)

def call_chat_smart(messages, temperature=0.7):
    """
    Fallback bidirezionale MA rispettando il lock:
    - Se lock attivo: mai cambiare engine, alzare errore.
    - Se engine=ollama: generate → chat → OpenAI (se non locked).
    - Se engine=openai: OpenAI → (solo su 429/quota/ratelimit) Ollama (se non locked).
    """
    eng = _engine()
    timeout = int(os.getenv("GV_OLLAMA_TIMEOUT", "60"))

    # ---- OLLAMA ----
    if eng == "ollama":
        # se lock, NON fare fallback su OpenAI
        try:
            prompt = build_simple_prompt_from_messages(messages)
            r = call_ollama_generate(prompt, temperature=temperature, timeout=timeout)
            if str(r).strip():
                return r
        except Exception:
            pass
        try:
            r = call_ollama_chat(messages, temperature=temperature, timeout=timeout)
            if str(r).strip():
                return r
        except Exception as e:
            if _locked():
                # blocco: niente fallback
                raise
        # fallback su OpenAI SOLO se non locked
        return call_gpt_chat(messages, temperature=temperature)

    # ---- OPENAI ----
    try:
        return call_gpt_chat(messages, temperature=temperature)
    except Exception as e:
        emsg = str(e).lower()
        # se lock, NON fare fallback
        if _locked():
            raise
        # fallback su Ollama SOLO per errori "quota/rate"
        if ("429" in emsg) or ("insufficient_quota" in emsg) or ("rate" in emsg and "limit" in emsg) or ("quota" in emsg):
            try:
                prompt = build_simple_prompt_from_messages(messages)
                r = call_ollama_generate(prompt, temperature=temperature, timeout=timeout)
                if str(r).strip():
                    return r
            except Exception:
                pass
            try:
                r = call_ollama_chat(messages, temperature=temperature, timeout=timeout)
                if str(r).strip():
                    return r
            except Exception:
                pass
        # altri errori: rialza
        raise
