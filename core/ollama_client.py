# core/ollama_client.py
# Client per Ollama (chat + generate + streaming) con parser robusto e diagnostica.
# Env utili:
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_MODEL=phi3:3.8b (o llama3:8b / mistral:7b)
#   OLLAMA_NUM_PREDICT=160
#   OLLAMA_NUM_CTX=1536
#   OLLAMA_NUM_THREAD=   (opzionale: forza i thread CPU)

import os, json, time
from typing import Iterable, List, Dict, Any, Tuple
import requests

BASE  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "phi3:3.8b")

HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

def _format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        raise ValueError("messages deve essere lista di dict con role/content")
    return messages

def _opts(temperature: float) -> Dict[str, Any]:
    opts = {
        "temperature": float(temperature),
        "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "160")),
        "num_ctx":     int(os.getenv("OLLAMA_NUM_CTX", "1536")),
    }
    nt = os.getenv("OLLAMA_NUM_THREAD")
    if nt:
        try:
            opts["num_thread"] = int(nt)
        except Exception:
            pass
    return opts

def _post(path: str, payload: Dict[str, Any], *, timeout: float | int | None) -> requests.Response:
    return requests.post(f"{BASE}{path}", json=payload, headers=HEADERS, timeout=timeout)

def _parse_non_stream_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Risposta vuota da Ollama")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    last_obj = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("data: "):
            s = s[6:].strip()
        try:
            last_obj = json.loads(s)
        except json.JSONDecodeError:
            continue
    if last_obj is not None:
        return last_obj
    i = text.rfind("{")
    if i != -1:
        try:
            return json.loads(text[i:])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Risposta non JSON da Ollama (len={len(text)}): {text[:200]}")

def build_simple_prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Converte i messaggi chat in un prompt minimo (system + ultimo user).
    Utile per /api/generate quando /api/chat fa cilecca.
    """
    system = ""
    last_user = ""
    for m in messages:
        if m.get("role") == "system" and not system:
            system = m.get("content", "")
        if m.get("role") == "user":
            last_user = m.get("content", "")
    parts = []
    if system:
        parts.append(f"[ISTRUZIONI]\n{system}\n")
    if last_user:
        parts.append(f"[UTENTE]\n{last_user}\n\n[RISPOSTA]\n")
    return "\n".join(parts).strip()

def call_ollama_chat(messages: List[Dict[str, Any]], temperature: float = 0.7,
                     timeout: float | int = 90, retries: int = 1,
                     debug: bool = False) -> str | Tuple[str, Dict[str, Any]]:
    """
    Non-stream, endpoint /api/chat con diagnostica opzionale.
    Ritorna stringa o (stringa, diag).
    """
    payload = {"model": MODEL, "messages": _format(messages), "options": _opts(temperature)}
    diag = {"endpoint": "/api/chat", "model": MODEL, "opts": payload["options"], "t": {}}

    last_err = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            resp = _post("/api/chat", payload, timeout=timeout)
            diag["t"]["http"] = round(time.time() - t0, 3)
            diag["status"] = resp.status_code
            resp.raise_for_status()
            data = _parse_non_stream_text(resp.text)
            diag["raw_len"] = len(resp.text or "")
            msg = (data.get("message") or {})
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                if debug:
                    return content, diag
                return content
            # alternativa 'response'
            if isinstance(data.get("response"), str) and data["response"].strip():
                if debug:
                    return data["response"], diag
                return data["response"]
            last_err = ValueError("Ollama ha restituito una risposta vuota")
        except Exception as e:
            last_err = e
            diag["error"] = str(e)
            time.sleep(0.3)

    if debug:
        return "", diag
    raise last_err

def call_ollama_generate(prompt: str, temperature: float = 0.7,
                         timeout: float | int = 90, retries: int = 1,
                         debug: bool = False) -> str | Tuple[str, Dict[str, Any]]:
    """
    Non-stream, endpoint /api/generate (prompt stringa singola).
    Usato come fallback quando /api/chat dà problemi.
    """
    payload = {"model": MODEL, "prompt": prompt, "options": _opts(temperature), "stream": False}
    diag = {"endpoint": "/api/generate", "model": MODEL, "opts": payload["options"], "t": {}}

    last_err = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            resp = _post("/api/generate", payload, timeout=timeout)
            diag["t"]["http"] = round(time.time() - t0, 3)
            diag["status"] = resp.status_code
            resp.raise_for_status()
            data = _parse_non_stream_text(resp.text)
            diag["raw_len"] = len(resp.text or "")
            # schema tipico: {"response":"...","done":true,...}
            text = data.get("response")
            if isinstance(text, str) and text.strip():
                if debug:
                    return text, diag
                return text
            last_err = ValueError("Ollama /generate ha restituito una risposta vuota")
        except Exception as e:
            last_err = e
            diag["error"] = str(e)
            time.sleep(0.3)

    if debug:
        return "", diag
    raise last_err

def stream_ollama_chat(messages: List[Dict[str, Any]], temperature: float = 0.7) -> Iterable[str]:
    """
    Streaming token-by-token (robusto a 'data: ' e più JSON per riga).
    """
    payload = {"model": MODEL, "messages": _format(messages), "options": _opts(temperature), "stream": True}
    with _post("/api/chat", payload, timeout=0) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("data: "):
                line = line[6:].strip()
            for piece in line.splitlines():
                s = piece.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if "error" in obj and obj["error"]:
                    yield f"\n[Errore Ollama] {obj['error']}\n"
                    return
                delta = (obj.get("message") or {}).get("content") or ""
                if delta:
                    yield delta
                if obj.get("done"):
                    return
