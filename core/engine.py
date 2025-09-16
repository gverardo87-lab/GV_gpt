# core/engine.py
# -----------------------------------------------------------------------------
# Motori supportati:
#   - OpenAI (chat completions, streaming/non-stream)
#   - Hugging Face Inference API (non-stream; "finto streaming" opzionale)
#   - Ollama locale (/api/chat, streaming/non-stream)
#
# Fallback smart (se GV_ENGINE_LOCK=OFF):
#   1) prova engine corrente
#   2) se engine=openai e 429/quota -> passa a hugging
#   3) altrimenti prova ollama come rete di sicurezza
#
# Re-export helper per longform "continue-only" (se presente in core.context).
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Generator, List, Optional, Callable

import requests

# ---- Import motori locali ---------------------------------------------------
try:
    from core.ollama_client import call_ollama_chat, stream_ollama_chat
except Exception:
    call_ollama_chat = None  # type: ignore
    stream_ollama_chat = None  # type: ignore

try:
    from core.hugging_client import call_hugging_chat
except Exception:
    call_hugging_chat = None  # type: ignore

# OpenAI client wrapper se presente
try:
    from core import gpt_clienti
except Exception:
    gpt_clienti = None  # type: ignore

# Re-export helper per longform "continue-only" (se esiste)
try:
    from core.context import build_continue_only_messages  # type: ignore
except Exception:
    def build_continue_only_messages(
        *,
        system_once: str,
        history: List[Dict[str, str]],
        final_text_tail: str,
        round_words: int = 600,
        didactic: bool = False,
    ) -> List[Dict[str, str]]:
        tail = (final_text_tail or "").strip()
        guard = (
            "Continua esattamente dal punto in cui il testo si è interrotto, "
            "senza riassunti, senza introduzioni (niente 'Introduzione', 'Capitolo', 'In questa risposta'). "
            f"Scrivi circa {round_words} parole e prosegui lineare. "
        )
        if didactic:
            guard += "Mantieni stile didattico con esempi pratici. "
        guard += "Chiudi con <<FINE>> solo se completi la sezione corrente."
        user_prompt = (tail + ("\n\n" if tail else "") + guard).strip()
        return [{"role": "system", "content": system_once}] + history + [{"role": "user", "content": user_prompt}]

# ============================== Utilities ====================================

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default

def _is_true(value: Optional[str]) -> bool:
    return (value or "").lower() in ("1", "true", "yes", "on")

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

# ============================== OpenAI client ================================

def _openai_base_url() -> str:
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

def _parse_openai_stream_line(raw_line: bytes) -> Optional[Dict[str, Any]]:
    if not raw_line:
        return None
    try:
        line = raw_line.decode("utf-8", errors="ignore").strip()
    except Exception:
        return None
    if not line or not line.startswith("data:"):
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
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    Non-stream: preferisce core.gpt_clienti se disponibile, altrimenti chiama REST.
    """
    if gpt_clienti is not None:
        # API interna tua; tipicamente accetta (messages, temperature)
        return gpt_clienti.call_gpt_chat(messages, temperature=temperature or 0.7)

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
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return json.dumps(data, ensure_ascii=False)

def _stream_openai(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    timeout: Optional[int] = None,
    should_stop: Optional[Callable[[], bool]] = None,  # non può interrompere HTTP, ma lo accettiamo per compat
) -> Generator[str, None, None]:
    """
    Stream SSE OpenAI. Nota: non è possibile interrompere la connessione HTTP già aperta
    senza chiudere la sessione a monte; il callback should_stop è solo per compatibilità.
    """
    if gpt_clienti is not None and hasattr(gpt_clienti, "stream_gpt_chat"):
        # Se esiste la tua implementazione, usala (torna un generator)
        return gpt_clienti.stream_gpt_chat(messages, temperature=temperature or 0.7)

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

    def _gen():
        last = time.time()
        for raw in resp.iter_lines(decode_unicode=False, delimiter=b"\n"):
            # Compat: se il chiamante chiede stop, usciamo dal loop generator
            if should_stop and should_stop():
                break
            obj = _parse_openai_stream_line(raw)
            if not obj:
                # semplice idle/heartbeat control lato client
                now = time.time()
                if now - last > 8:
                    break
                continue
            if obj.get("done") is True:
                break
            try:
                delta = obj["choices"][0]["delta"].get("content")
            except Exception:
                delta = None
            if isinstance(delta, str) and delta:
                last = time.time()
                yield delta
        try:
            resp.close()
        except Exception:
            pass

    return _gen()

# ============================== Public API ===================================

def call_chat(messages: List[Dict[str, str]], **kwargs) -> str:
    """
    Invoca il motore configurato (GV_ENGINE). Supporta:
      - OpenAI: temperature, top_p, max_tokens
      - Hugging: temperature, max_tokens -> mappato su max_new_tokens
      - Ollama: options (temperature, top_p, num_predict)
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
        if call_hugging_chat is None:
            raise RuntimeError("Modulo Hugging Face non disponibile.")
        return call_hugging_chat(
            messages,
            model=os.getenv("HUGGINGFACE_MODEL") if model_override is None else model_override,
            temperature=kwargs.get("temperature"),
            max_new_tokens=kwargs.get("max_tokens") or int(os.getenv("HUGGINGFACE_MAX_NEW_TOKENS", "512")),
            top_p=kwargs.get("top_p"),
            timeout=kwargs.get("timeout"),
        )

    if engine == "ollama":
        if call_ollama_chat is None:
            raise RuntimeError("Modulo Ollama non disponibile.")
        options: Dict[str, Any] = {}
        if kwargs.get("temperature") is not None:
            options["temperature"] = float(kwargs["temperature"])
        if kwargs.get("top_p") is not None:
            options["top_p"] = float(kwargs["top_p"])
        if kwargs.get("max_tokens") is not None:
            options["num_predict"] = int(kwargs["max_tokens"])
        return call_ollama_chat(
            messages=messages,
            model=model_override or None,
            options=options or None,
            timeout=kwargs.get("timeout"),
        )

    raise RuntimeError(f"Engine non supportato: {engine}")

def stream_chat(messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
    """
    Streaming unificato.
    Supporta per Ollama:
      - should_stop (callable) per hard-stop immediato
      - idle_timeout (float, env OLLAMA_IDLE_TIMEOUT default 8s)
      - heartbeat_sec (float, env OLLAMA_HEARTBEAT_SEC default 2s)
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
            should_stop=kwargs.get("should_stop"),
        )

    if engine == "hugging":
        # Non abbiamo stream HF qui; simuliamo un "blocco unico"
        if call_hugging_chat is None:
            raise RuntimeError("Modulo Hugging Face non disponibile.")
        txt = call_hugging_chat(
            messages,
            model=os.getenv("HUGGINGFACE_MODEL") if model_override is None else model_override,
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
        if stream_ollama_chat is None:
            raise RuntimeError("Modulo Ollama non disponibile.")
        options: Dict[str, Any] = {}
        if kwargs.get("temperature") is not None:
            options["temperature"] = float(kwargs["temperature"])
        if kwargs.get("top_p") is not None:
            options["top_p"] = float(kwargs["top_p"])
        if kwargs.get("max_tokens") is not None:
            options["num_predict"] = int(kwargs["max_tokens"])

        should_stop = kwargs.get("should_stop")
        idle_timeout = float(_env("OLLAMA_IDLE_TIMEOUT", "8") or "8")
        heartbeat_sec = float(_env("OLLAMA_HEARTBEAT_SEC", "2") or "2")

        return stream_ollama_chat(
            messages=messages,
            model=model_override or None,
            options=options or None,
            timeout=kwargs.get("timeout"),
            idle_timeout=kwargs.get("idle_timeout", idle_timeout),
            heartbeat_sec=kwargs.get("heartbeat_sec", heartbeat_sec),
            should_stop=should_stop,
        )

    def _empty():
        if False:
            yield ""
    return _empty()

def call_chat_smart(messages: List[Dict[str, str]], **kwargs) -> str:
    """
    Esegue la chiamata con fallback automatico se GV_ENGINE_LOCK non è attivo.
    """
    lock = _is_true(os.getenv("GV_ENGINE_LOCK", "0"))
    current = (_env("GV_ENGINE", "openai") or "openai").lower()

    try:
        return call_chat(messages, **kwargs)
    except Exception as e1:
        if lock:
            raise
        err = str(e1).lower()

        # Se OpenAI quota/429 -> prova HF
        if current == "openai" and any(k in err for k in ("429", "quota", "insufficient", "rate limit")):
            try:
                os.environ["GV_ENGINE"] = "hugging"
                return call_chat(messages, **kwargs)
            except Exception:
                pass
            finally:
                os.environ["GV_ENGINE"] = current

        # Prova Ollama come rete di sicurezza
        try:
            os.environ["GV_ENGINE"] = "ollama"
            return call_chat(messages, **kwargs)
        except Exception:
            os.environ["GV_ENGINE"] = current
            raise e1
        finally:
            os.environ["GV_ENGINE"] = current

# ============================== __all__ ======================================

__all__ = [
    "call_chat",
    "stream_chat",
    "call_chat_smart",
    "build_continue_only_messages",
]
