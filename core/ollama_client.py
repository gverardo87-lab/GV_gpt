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
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, json, time
from typing import Iterable, List, Dict, Any, Optional, Generator
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
    keep_alive: Optional[str] = "5m",
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

    # override extra kwargs
    for k,v in kw.items():
        opts[k] = v
    return opts

# --- HTTP helpers -----------------------------------------------------------
def _post(path: str, payload: Dict[str, Any], timeout: int | float = 120, stream: bool = False) -> requests.Response:
    url = f"{_base_url()}{path}"
    headers = {"Content-Type": "application/json"}
    # timeout può essere float o (conn, read)
    tout = timeout if isinstance(timeout, (int, float)) else 120
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
    timeout: int | float = 180,
    stop: Optional[List[str]] = None,
    keep_alive: str = "5m",
    options: Optional[Dict[str, Any]] = None,
    **kw: Any,
) -> str:
    """
    Chiamata non-stream a /api/chat. Ritorna il testo finale come stringa.
    """
    mdl = _model_name(model)
    opts = _default_options(num_predict=num_predict, num_ctx=num_ctx, temperature=temperature,
                            stop=stop, keep_alive=keep_alive, **(options or {}), **kw)
    payload = {
        "model": mdl,
        "messages": messages,
        "stream": False,
        "options": opts,
    }
    resp = _post("/api/chat", payload, timeout=timeout, stream=False)
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
    timeout: int | float = 240,
    stop: Optional[List[str]] = None,
    keep_alive: str = "5m",
    options: Optional[Dict[str, Any]] = None,
    heartbeat_sec: float = 2.0,
    **kw: Any,
) -> Generator[str, None, None]:
    """
    Streaming generator da /api/chat.
    Emette pezzi di testo (delta). Gestisce heartbeat (spazi) se nessun token arriva per un po'.
    """
    mdl = _model_name(model)
    opts = _default_options(num_predict=num_predict, num_ctx=num_ctx, temperature=temperature,
                            stop=stop, keep_alive=keep_alive, **(options or {}), **kw)
    payload = {
        "model": mdl,
        "messages": messages,
        "stream": True,
        "options": opts,
    }
    start = time.time()
    last_yield = start
    try:
        resp = _post("/api/chat", payload, timeout=timeout, stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                # heartbeat per evitare timeouts UI; emetti uno spazio raramente
                if (time.time() - last_yield) > heartbeat_sec:
                    yield ""
                    last_yield = time.time()
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                yield f"\n[Errore Ollama] {obj['error']}\n"
                return
            delta = ((obj.get("message") or {}).get("content")) or ""
            if delta:
                yield delta
                last_yield = time.time()
            if obj.get("done"):
                return
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
    timeout: int | float = 180,
    stop: Optional[List[str]] = None,
    keep_alive: str = "5m",
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
    opts = _default_options(num_predict=num_predict, num_ctx=num_ctx, temperature=temperature,
                            stop=stop, keep_alive=keep_alive, **(options or {}), **kw)
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
            resp = _post("/api/generate", payload, timeout=timeout, stream=True)
            resp.raise_for_status()
            out = []
            for line in resp.iter_lines(decode_unicode=True):
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
            resp = _post("/api/generate", payload, timeout=timeout, stream=False)
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
