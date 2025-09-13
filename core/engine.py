# core/engine.py
# -----------------------------------------------------------------------------
# Motori supportati:
#   - OpenAI (chat completions, streaming/non-streaming)
#   - Hugging Face Inference API (mid-tier di test, non-stream; "finto streaming")
#   - Ollama locale (/api/chat, streaming/non-streaming)
#
# Fallback smart (se GV_ENGINE_LOCK=OFF):
#   1) prova engine corrente
#   2) se engine=openai e l'errore indica 429/quota -> passa a hugging
#   3) altrimenti prova ollama come rete di sicurezza
#
# Esporta anche build_continue_only_messages dal modulo core.context.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Generator, Iterable, List, Optional

import requests

# ---- Import motori locali ---------------------------------------------------
from core.ollama_client import (
    call_ollama_chat,
    stream_ollama_chat,
)

# Mid-tier di test (Hugging Face Inference API)
from core.hugging_client import call_hugging_chat  # nuovo client

# Re-export helper per longform "continue-only"
try:
    from core.context import build_continue_only_messages  # re-export
except Exception:
    def build_continue_only_messages(*args, **kwargs):
        raise RuntimeError("build_continue_only_messages non disponibile (core/context.py mancante).")

# ============================== Utilities ====================================

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default

def _is_true(value: Optional[str]) -> bool:
    return (value or "").lower() in ("1", "true", "yes", "on")

def _now_ms() -> int:
    return int(time.time() * 1000)

# ============================== OpenAI client ================================

def _openai_base_url() -> str:
    # Supporta override (es. proxy) via OPENAI_BASE_URL; default API ufficiale
    return (_env("OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1").rstrip("/")

def _openai_model() -> str:
    return _env("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"

def _openai_headers() -> Dict[str, str]:
    key = _env("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY mancante.")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def _with_retry(fn, attempts: int = 2, base_delay: float = 0.6):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    raise last

def _parse_openai_stream_line(raw_line: bytes) -> Optional[Dict[str, Any]]:
    if not raw_line:
        return None
    try:
        line = raw_line.decode("utf-8", errors="ignore").strip()
    except Exception:
        return None
    if not line:
        return None
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return {"done": True}
    try:
        return json.loads(payload)
    except Exception:
        return None

def _call_openai_nonstream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    timeout: Optional[int] = None,
) -> str:
    url = _openai_base_url() + "/chat/completions"
    headers = _openai_headers()
    payload: Dict[str, Any] = {
        "model": (model or _openai_model()),
        "messages": messages,
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    read_to = int(timeout or int(_env("GV_OPENAI_READ_TIMEOUT", "90") or "90"))
    connect_to = int(_env("GV_OPENAI_CONNECT_TIMEOUT", "10") or "10")

    def _do_post():
        return requests.post(url, headers=headers, json=payload, timeout=(connect_to, read_to))

    resp = _with_retry(_do_post)
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    # OpenAI: choices[0].message.content
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        # fallback testuale per debug
        return json.dumps(data, ensure_ascii=False)

def _stream_openai(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    timeout: Optional[int] = None,
) -> Generator[str, None, None]:
    url = _openai_base_url() + "/chat/completions"
    headers = _openai_headers()
    payload: Dict[str, Any] = {
        "model": (model or _openai_model()),
        "messages": messages,
        "stream": True,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    read_to = int(timeout or int(_env("GV_OPENAI_READ_TIMEOUT", "90") or "90"))
    connect_to = int(_env("GV_OPENAI_CONNECT_TIMEOUT", "10") or "10")

    def _do_post():
        return requests.post(url, headers=headers, json=payload, stream=True, timeout=(connect_to, read_to))

    resp = _with_retry(_do_post)
    resp.raise_for_status()

    for raw in resp.iter_lines(decode_unicode=False, delimiter=b"\n"):
        obj = _parse_openai_stream_line(raw)
        if not obj:
            continue
        if obj.get("done") is True:
            break
        try:
            delta = obj["choices"][0]["delta"].get("content")
        except Exception:
            delta = None
        if isinstance(delta, str) and delta:
            yield delta

# ============================== Public API ===================================

def call_chat(
    messages: List[Dict[str, str]],
    **kwargs,
) -> str:
    """
    Chiamata non-stream. Route sull'engine scelto via GV_ENGINE.
    kwargs supportati: temperature, max_tokens, top_p, timeout, model_override
    """
    engine = (_env("GV_ENGINE", "openai") or "openai").lower()
    model_override = kwargs.get("model") or kwargs.get("model_override")

    if engine == "openai":
        return _call_openai_nonstream(
            messages,
            model=model_override or _openai_model(),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            top_p=kwargs.get("top_p"),
            timeout=kwargs.get("timeout"),
        )

    if engine == "hugging":
        # Non-stream: Hugging Face Inference API
        return call_hugging_chat(
            messages,
            model=os.getenv("HUGGINGFACE_MODEL"),
            temperature=kwargs.get("temperature"),
            max_new_tokens=kwargs.get("max_tokens") or int(os.getenv("HUGGINGFACE_MAX_NEW_TOKENS", "512")),
            top_p=kwargs.get("top_p"),
            timeout=kwargs.get("timeout"),
        )

    if engine == "ollama":
        # /api/chat (non-stream)
        options = {}
        # opzionale: parametri di generazione possono essere passati in options
        if kwargs.get("temperature") is not None:
            options["temperature"] = float(kwargs["temperature"])
        if kwargs.get("top_p") is not None:
            options["top_p"] = float(kwargs["top_p"])
        if kwargs.get("max_tokens") is not None:
            options["num_predict"] = int(kwargs["max_tokens"])
        return call_ollama_chat(
            messages=messages,
            model=model_override or None,
            base_url=os.getenv("OLLAMA_BASE_URL"),
            options=options or None,
            timeout=kwargs.get("timeout"),
        )

    raise RuntimeError(f"Engine non supportato: {engine}")

def stream_chat(
    messages: List[Dict[str, str]],
    **kwargs,
) -> Generator[str, None, None]:
    """
    Streaming. Per Hugging Face (no streaming) restituiamo un generatore che
    emette tutto in un solo yield (pseudo-stream).
    """
    engine = (_env("GV_ENGINE", "openai") or "openai").lower()
    model_override = kwargs.get("model") or kwargs.get("model_override")

    if engine == "openai":
        return _stream_openai(
            messages,
            model=model_override or _openai_model(),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            top_p=kwargs.get("top_p"),
            timeout=kwargs.get("timeout"),
        )

    if engine == "hugging":
        txt = call_hugging_chat(
            messages,
            model=os.getenv("HUGGINGFACE_MODEL"),
            temperature=kwargs.get("temperature"),
            max_new_tokens=kwargs.get("max_tokens") or int(os.getenv("HUGGINGFACE_MAX_NEW_TOKENS", "512")),
            top_p=kwargs.get("top_p"),
            timeout=kwargs.get("timeout"),
        )
        def _gen():
            if txt:
                yield txt
        return _gen()

    if engine == "ollama":
        options = {}
        if kwargs.get("temperature") is not None:
            options["temperature"] = float(kwargs["temperature"])
        if kwargs.get("top_p") is not None:
            options["top_p"] = float(kwargs["top_p"])
        if kwargs.get("max_tokens") is not None:
            options["num_predict"] = int(kwargs["max_tokens"])
        return stream_ollama_chat(
            messages=messages,
            model=model_override or None,
            base_url=os.getenv("OLLAMA_BASE_URL"),
            options=options or None,
            timeout=kwargs.get("timeout"),
        )

    def _empty():
        if False:
            yield ""  # pragma: no cover
    return _empty()

def call_chat_smart(
    messages: List[Dict[str, str]],
    **kwargs,
) -> str:
    """
    Chiamata "intelligente" con fallback multi-engine se GV_ENGINE_LOCK è OFF.
    Strategia:
      - Prova engine corrente.
      - Se engine=openai e l'errore indica 429/quota/limit -> prova hugging.
      - Altrimenti prova ollama.
    Se GV_ENGINE_LOCK è ON -> ripropaga l'errore del primo tentativo.
    """
    lock = _is_true(os.getenv("GV_ENGINE_LOCK", "0"))
    current = (_env("GV_ENGINE", "openai") or "openai").lower()

    # 1) tenta engine corrente
    try:
        return call_chat(messages, **kwargs)
    except Exception as e1:
        if lock:
            raise
        err = str(e1).lower()

        # 2) se siamo su OpenAI e l'errore è quota/rate limit -> HuggingFace
        if current == "openai" and any(k in err for k in ("429", "quota", "insufficient", "rate limit")):
            try:
                os.environ["GV_ENGINE"] = "hugging"
                return call_chat(messages, **kwargs)
            except Exception:
                pass
            finally:
                os.environ["GV_ENGINE"] = current

        # 3) ultima rete di sicurezza: Ollama
        try:
            os.environ["GV_ENGINE"] = "ollama"
            return call_chat(messages, **kwargs)
        except Exception:
            # ripristina e rilancia l'errore iniziale
            os.environ["GV_ENGINE"] = current
            raise e1
        finally:
            os.environ["GV_ENGINE"] = current

# ============================== __all__ ======================================

__all__ = [
    "call_chat",
    "stream_chat",
    "call_chat_smart",
    "build_continue_only_messages",  # re-export
]
