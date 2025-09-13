# core/hugging_client.py
# -----------------------------------------------------------------------------
# Client minimale per Hugging Face Inference API (text-generation):
# - Non-stream (ritorna tutto il testo)
# - Retry con backoff
# - Prompt builder: riusa build_simple_prompt_from_messages
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import requests

# user helper: riuso funzione già presente nel tuo progetto
try:
    from core.ollama_client import build_simple_prompt_from_messages
except Exception:
    def build_simple_prompt_from_messages(messages: Iterable[Dict[str, str]]) -> str:
        parts = []
        sys = "\n".join([m.get("content","") for m in messages if m.get("role")=="system"]).strip()
        if sys:
            parts.append("[SYSTEM]\n" + sys)
        dlg = []
        for m in messages:
            r = (m.get("role") or "").lower()
            c = m.get("content") or ""
            if r == "user":
                dlg.append(f"Utente: {c}")
            elif r == "assistant":
                dlg.append(f"Assistente: {c}")
        if dlg:
            parts.append("[DIALOGO]\n" + "\n".join(dlg))
        parts.append("\n---\nRispondi ora come Assistente, continuando dal contesto.")
        return "\n".join(parts)

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default

def hf_base_url() -> str:
    return _env("HUGGINGFACE_BASE_URL", "https://api-inference.huggingface.co") or "https://api-inference.huggingface.co"

def hf_model() -> str:
    # Esempi: "meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1"
    return _env("HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct") or "meta-llama/Meta-Llama-3.1-8B-Instruct"

def hf_api_key() -> str:
    return _env("HUGGINGFACE_API_KEY", "") or ""

DEFAULT_CONNECT_TIMEOUT = 10
DEFAULT_READ_TIMEOUT = 90

def _with_retry(fn, attempts: int = 3, base_delay: float = 0.8):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    raise last

def call_hugging_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    Chiamata non-stream alla HuggingFace Inference API (text-generation).
    Usa un prompt "semplice" costruito dai messages. Ritorna la stringa generata.
    """
    if not messages:
        return ""

    api_key = hf_api_key()
    if not api_key:
        raise RuntimeError("HUGGINGFACE_API_KEY mancante")

    base = hf_base_url().rstrip("/")
    mdl = (model or hf_model()).strip()

    prompt = build_simple_prompt_from_messages(messages)

    params = {
        "max_new_tokens": int(max_new_tokens or int(os.getenv("HUGGINGFACE_MAX_NEW_TOKENS", "512"))),
        "temperature": float(temperature if temperature is not None else float(os.getenv("HUGGINGFACE_TEMPERATURE", "0.3"))),
        "top_p": float(top_p if top_p is not None else float(os.getenv("HUGGINGFACE_TOP_P", "0.9"))),
        "return_full_text": False,
        # "do_sample": False,  # opzionale, puoi attivarla per risposte più deterministiche
    }
    payload = {
        "inputs": prompt,
        "parameters": params,
        "options": {"wait_for_model": True},  # scarica il modello al primo uso
    }

    url = f"{base}/models/{mdl}"
    read_to = int(timeout or DEFAULT_READ_TIMEOUT)
    connect_to = DEFAULT_CONNECT_TIMEOUT

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    def _do_post():
        return requests.post(url, headers=headers, json=payload, timeout=(connect_to, read_to))

    resp = _with_retry(_do_post)
    resp.raise_for_status()

    data = {}
    try:
        data = resp.json()
    except Exception:
        pass

    # La Inference API spesso ritorna: [{"generated_text": "..."}]
    # In altri casi ritorna direttamente un dict con chiavi diverse.
    text = ""
    if isinstance(data, list) and data:
        # prova le chiavi note nella prima entry
        first = data[0]
        if isinstance(first, dict):
            text = first.get("generated_text") or first.get("generated_texts") or ""
            if not text and "conversation" in first:
                # fallback davvero raro
                text = str(first["conversation"])
    elif isinstance(data, dict):
        text = data.get("generated_text") or data.get("text") or ""

    # Se tutto fallisce, prova a estrarre stringa grezza
    if not isinstance(text, str):
        text = json.dumps(data, ensure_ascii=False)

    return text.strip()
