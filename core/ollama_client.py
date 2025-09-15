# core/ollama_client.py
# -----------------------------------------------------------------------------
# Client robusto per Ollama:
# - /api/chat (consigliato) con streaming stabile e opzioni anti-ripetizione
# - /api/generate (legacy, solo se serve uno style "instruct")
# - Opzioni predefinite: Mirostat v2, penalità ripetizione, keep_alive
# - Helper: build_simple_prompt_from_messages (per fallback generate)
#
# Env utili:
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_MODEL=phi3:3.8b (o llama3:8b / mistral:7b)
#   OLLAMA_NUM_PREDICT=512
#   OLLAMA_NUM_CTX=3072
#   OLLAMA_NUM_THREAD= (opzionale)
#   OLLAMA_CONNECT_TIMEOUT=10
#   OLLAMA_READ_TIMEOUT=180
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, json, time
from typing import Iterable, List, Dict, Any, Optional, Generator, Callable
import requests

# --- Defaults ---------------------------------------------------------------
def _base_url() -> str:
    return (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")

def _model_name(model: Optional[str] = None) -> str:
    return model or os.getenv("OLLAMA_MODEL") or "phi3:3.8b"

def _int_from_env(name: str, default: Optional[int]) -> Optional[int]:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

def _default_options(
    *,
    num_predict: Optional[int] = None,
    num_ctx: Optional[int] = None,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 0.9,
    repeat_penalty: Optional[float] = 1.15,
    repeat_last_n: Optional[int] = 256,
    presence_penalty: Optional[float] = 0.15,
    frequency_penalty: Optional[float] = 0.10,
    mirostat: Optional[int] = 2,
    mirostat_tau: Optional[float] = 5.0,
    mirostat_eta: Optional[float] = 0.1,
    stop: Optional[List[str]] = None,
    keep_alive: Optional[str] = "15s",   # ridotto (prima 5m)
    seed: Optional[int] = None,
    num_thread: Optional[int] = None,
    **kw: Any,
) -> Dict[str, Any]:
    """
    Costruisce il dizionario options per Ollama.
    Valori mancanti ereditano da env (OLLAMA_NUM_PREDICT, OLLAMA_NUM_CTX).
    """
    if num_predict is None:
        num_predict = _int_from_env("OLLAMA_NUM_PREDICT", 512)
    if num_ctx is None:
        num_ctx = _int_from_env("OLLAMA_NUM_CTX", 3072)
    if num_thread is None:
        num_thread = _int_from_env("OLLAMA_NUM_THREAD", None)

    opts: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "repeat_last_n": repeat_last_n,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "mirostat": mirostat,
        "mirostat_tau": mirostat_tau,
        "mirostat_eta": mirostat_eta,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "keep_alive": keep_alive,
    }
    if seed is not None:
        opts["seed"] = seed
    if num_thread is not None:
        opts["num_thread"] = num_thread
    if stop:
        # Unisci alle nostre stop words "anti-introduzione"
        default_stop = ["Introduzione", "Capitolo", "In questa risposta", "<<FINE>>"]
        opts["stop"] = list(dict.fromkeys(default_stop + list(stop)))
    else:
        opts["stop"] = ["Introduzione", "Capitolo", "In questa risposta", "<<FINE>>"]

    # Override extra kwargs (evita override con None)
    for k, v in kw.items():
        if v is not None:
            opts[k] = v
    return opts

# --- HTTP helpers -----------------------------------------------------------
def _mk_timeout(timeout: int | float | tuple[int, int] | tuple[float, float] | None) -> tuple[float, float]:
    """
    Rende sempre (connect_timeout, read_timeout).
    - Se timeout è None → default: 10s connessione, 180s lettura (overridabili via env).
    - Se è numero → (10s, timeout).
    - Se è tupla → cast a float di entrambi gli elementi.
    - Se è stringa convertibile → trattala come numero (read timeout).
    """
    def _envf(name: str, default: str) -> float:
        try:
            return float(os.getenv(name, default))
        except Exception:
            return float(default)

    if timeout is None:
        return (_envf("OLLAMA_CONNECT_TIMEOUT", "10"), _envf("OLLAMA_READ_TIMEOUT", "180"))

    if isinstance(timeout, tuple) and len(timeout) == 2:
        a, b = timeout
        return (float(a), float(b))

    try:
        # int/float o stringa numerica
        return (_envf("OLLAMA_CONNECT_TIMEOUT", "10"), float(timeout))
    except Exception:
        # fallback sicuro
        return (10.0, 180.0)

def _post(path: str, payload: Dict[str, Any], timeout: int | float | tuple[int, int] | None = 120, stream: bool = False) -> requests.Response:
    url = f"{_base_url()}{path}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/x-ndjson" if stream else "application/json",
        "Connection": "close",  # non tenere connessioni appese
    }
    tout = _mk_timeout(timeout)
    # NB: usiamo data=json.dumps per evitare ambiguità su float/None
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=tout, stream=stream)
    return resp

# --- Public API: Chat (recommended) -----------------------------------------
def call_ollama_chat(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.7,
    num_predict: Optional[int] = None,
    num_ctx: Optional[int] = None,
    timeout: int | float | tuple[int, int] | None = 180,
    stop: Optional[List[str]] = None,
    keep_alive: str = "15s",
    options: Optional[Dict[str, Any]] = None,
    **kw: Any,
) -> str:
    """
    Chiamata non-stream a /api/chat. Ritorna il testo finale come stringa.
    """
    mdl = _model_name(model)
    # DEDUP: se una chiave è presente in options/kw, non ripassarla come argomento nominato
    _extra = {k: v for k, v in {**(options or {}), **kw}.items() if v is not None}
    _named = {
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        "temperature": temperature,
        "stop": stop,
        "keep_alive": keep_alive,
    }
    _base = {k: v for k, v in _named.items() if (v is not None and k not in _extra)}
    opts = _default_options(**_base, **_extra)

    payload = {
        "model": mdl,
        "messages": messages,
        "stream": False,
        "options": opts,
    }
    # ⬇️ chiusura garantita della connessione
    with _post("/api/chat", payload, timeout=timeout, stream=False) as resp:
        resp.raise_for_status()
        data = resp.json()
    # formato: {"message":{"role":"assistant","content":"..."}, "done":true, ...}
    msg = data.get("message") or {}
    content = msg.get("content") or ""
    return str(content)

def stream_ollama_chat(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.7,
    num_predict: Optional[int] = None,
    num_ctx: Optional[int] = None,
    timeout: int | float | tuple[int, int] | None = 240,
    stop: Optional[List[str]] = None,
    keep_alive: str = "15s",
    options: Optional[Dict[str, Any]] = None,
    heartbeat_sec: float = 2.0,
    idle_timeout: float = 8.0,                      # tronca se non arrivano token
    should_stop: Optional[Callable[[], bool]] = None,  # stop esterno (facoltativo)
    **kw: Any,
) -> Generator[str, None, None]:
    """
    Streaming generator da /api/chat.
    Emette pezzi di testo (delta). Gestisce heartbeat (vuoti) e idle-timeout.
    Se should_stop è fornito e ritorna True, interrompe immediatamente lo stream.
    """
    mdl = _model_name(model)
    # DEDUP come sopra
    _extra = {k: v for k, v in {**(options or {}), **kw}.items() if v is not None}
    _named = {
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        "temperature": temperature,
        "stop": stop,
        "keep_alive": keep_alive,
    }
    _base = {k: v for k, v in _named.items() if (v is not None and k not in _extra)}
    opts = _default_options(**_base, **_extra)

    payload = {
        "model": mdl,
        "messages": messages,
        "stream": True,
        "options": opts,
    }
    start = time.time()
    last_yield = start
    last_token = start

    try:
        # ⬇️ context manager per chiudere SEMPRE la connessione
        with _post("/api/chat", payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True, delimiter="\n"):
                # stop esterno richiesto?
                if should_stop and should_stop():
                    break

                now = time.time()
                if not line:
                    # heartbeat verso UI
                    if (now - last_yield) > heartbeat_sec:
                        yield ""
                        last_yield = now
                    # idle-timeout: nessun token per un po'
                    if (now - last_token) > idle_timeout:
                        break
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if obj.get("error"):
                    yield f"\n[Errore Ollama] {obj['error']}\n"
                    break

                # nuovo token
                delta = ((obj.get("message") or {}).get("content")) or ""
                if delta:
                    yield delta
                    last_yield = last_token = now

                if obj.get("done") is True:
                    break
    except requests.exceptions.ReadTimeout:
        yield "\n[Errore Ollama] read timeout durante lo streaming.\n"
    except Exception as e:
        yield f"\n[Errore Ollama] {e}\n"

# --- Public API: Generate (legacy/instruct) ----------------------------------
def call_ollama_generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.7,
    num_predict: Optional[int] = None,
    num_ctx: Optional[int] = None,
    timeout: int | float | tuple[int, int] | None = 180,
    stop: Optional[List[str]] = None,
    keep_alive: str = "15s",
    options: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    debug: bool = False,
    **kw: Any,
) -> str | tuple[str, Dict[str, Any]]:
    """
    /api/generate per modelli instruct. Preferire /api/chat per continuazione.
    Se debug=True, ritorna (testo, diagnostica).
    """
    mdl = _model_name(model)
    # DEDUP come sopra
    _extra = {k: v for k, v in {**(options or {}), **kw}.items() if v is not None}
    _named = {
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        "temperature": temperature,
        "stop": stop,
        "keep_alive": keep_alive,
    }
    _base = {k: v for k, v in _named.items() if (v is not None and k not in _extra)}
    opts = _default_options(**_base, **_extra)

    payload = {
        "model": mdl,
        "prompt": prompt,
        "stream": bool(stream),
        "options": opts,
    }
    diag = {"status": None, "t": {}, "model": mdl, "opts": opts}
    try:
        t0 = time.time()
        if stream:
            with _post("/api/generate", payload, timeout=timeout, stream=True) as resp:
                resp.raise_for_status()
                out = []
                for line in resp.iter_lines(decode_unicode=True, delimiter="\n"):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("error"):
                        raise RuntimeError(obj["error"])
                    piece = obj.get("response") or ""
                    if piece:
                        out.append(piece)
                    if obj.get("done"):
                        break
            text = "".join(out)
            diag["status"] = 200
            diag["t"]["elapsed"] = round(time.time() - t0, 3)
        else:
            with _post("/api/generate", payload, timeout=timeout, stream=False) as resp:
                diag["status"] = resp.status_code
                resp.raise_for_status()
                data = resp.json()
            text = data.get("response") or ""
            diag["t"]["elapsed"] = round(time.time() - t0, 3)
    except Exception as e:
        if debug:
            return f"[Errore Ollama] {e}", diag
        raise
    return (text, diag) if debug else text

# --- Helper: build_simple_prompt_from_messages -------------------------------
def build_simple_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    Fallback semplice: trasforma i messages in un prompt unico "instruct".
    Utile quando si vuole usare /api/generate con una storia chat.
    """
    parts: List[str] = []
    for m in messages:
        role = m.get("role","").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"[ISTRUZIONI]\n{content}\n")
        elif role == "user":
            parts.append(f"[UTENTE]\n{content}\n")
        elif role == "assistant":
            parts.append(f"[ASSISTANT]\n{content}\n")
        else:
            parts.append(f"[{role.upper()}]\n{content}\n")
    parts.append("\n[RISPOSTA]\n")
    return "\n".join(parts).strip()
