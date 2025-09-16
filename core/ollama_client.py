# core/ollama_client.py
# -----------------------------------------------------------------------------
# Client per Ollama locale:
#   - /api/chat   (streaming e non-streaming)
#   - /api/generate (diagnostica legacy)
#
# NOTE:
# - Nessun override della stoplist (gestita dal Modelfile).
# - Streaming robusto con delimiter=b"\n" e decodifica UTF-8 esplicita.
# - Merge opzioni da ENV: OLLAMA_TEMPERATURE, OLLAMA_TOP_P, OLLAMA_NUM_PREDICT, ecc.
# - Strip sentinel finali noti + micro-filtro anti-rumore (mojibake/boilerplate).
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Any, Dict, Generator, List, Optional, Callable, Tuple

import os
import json
import time
import re
import requests

# ============================== Helpers ======================================

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default

def _base_url() -> str:
    return (_env("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434").rstrip("/")

def _model_default() -> str:
    return _env("OLLAMA_MODEL", "gv/phi35-mini-gv:latest") or "gv/phi35-mini-gv:latest"

def _timeouts(timeout: Optional[int] = None) -> Tuple[int, int]:
    """
    Ritorna (connect_timeout, read_timeout).
    """
    connect_to = int(_env("OLLAMA_CONNECT_TIMEOUT", "10") or "10")
    read_to = int(timeout or int(_env("OLLAMA_READ_TIMEOUT", "120") or "120"))
    return connect_to, read_to

def _merge_options(user_opts: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Costruisce il dict options finale:
    - prendi valori da ENV
    - applica/override con user_opts
    - rimuovi chiavi non desiderate (es. 'stop', perché la stoplist è nel Modelfile)
    """
    out: Dict[str, Any] = {}

    # Da ENV (se valorizzati)
    if _env("OLLAMA_TEMPERATURE"):
        try:
            out["temperature"] = float(_env("OLLAMA_TEMPERATURE") or "0.3")
        except Exception:
            pass
    if _env("OLLAMA_TOP_P"):
        try:
            out["top_p"] = float(_env("OLLAMA_TOP_P") or "0.85")
        except Exception:
            pass
    if _env("OLLAMA_TOP_K"):
        try:
            out["top_k"] = int(_env("OLLAMA_TOP_K") or "50")
        except Exception:
            pass
    if _env("OLLAMA_NUM_PREDICT"):
        try:
            out["num_predict"] = int(_env("OLLAMA_NUM_PREDICT") or "900")
        except Exception:
            pass
    if _env("OLLAMA_REPEAT_PENALTY"):
        try:
            out["repeat_penalty"] = float(_env("OLLAMA_REPEAT_PENALTY") or "1.2")
        except Exception:
            pass
    if _env("OLLAMA_REPEAT_LAST_N"):
        try:
            out["repeat_last_n"] = int(_env("OLLAMA_REPEAT_LAST_N") or "256")
        except Exception:
            pass
    if _env("OLLAMA_PRESENCE_PENALTY"):
        try:
            out["presence_penalty"] = float(_env("OLLAMA_PRESENCE_PENALTY") or "0.0")
        except Exception:
            pass
    if _env("OLLAMA_FREQUENCY_PENALTY"):
        try:
            out["frequency_penalty"] = float(_env("OLLAMA_FREQUENCY_PENALTY") or "0.0")
        except Exception:
            pass
    if _env("OLLAMA_NUM_CTX"):
        try:
            out["num_ctx"] = int(_env("OLLAMA_NUM_CTX") or "8192")
        except Exception:
            pass

    # Merge con user options
    if user_opts:
        try:
            out.update({k: v for k, v in user_opts.items() if v is not None})
        except Exception:
            pass

    # Non permettere override della stoplist client-side (governa tutto il Modelfile)
    out.pop("stop", None)

    return out or None

# ============================== Cleaners =====================================

# Meta-markers stile LLM e blocchi tipo [LABEL: ...]
_META_RX = re.compile(r"(?is)(<\|[^>]*\|>|\[[A-Z][A-Z0-9 _-]{2,}:[^\]]*\]|\[[A-Z][A-Z0-9 _-]{2,}\])")

# Prefissi "Instruction:" / scuse inutili
_DRIFT_PREFIX_RX = re.compile(r"(?is)^\s*(instruction[s]?:.*?\n+|\s*i['’]m\s+sorry[^.\n]*[.\n]+\s*)")

# Righe con "Your task:" / "Instruction:" / "Begin by:"
_DRIFT_LINES_RX = re.compile(r"(?im)^\s*(your\s+task|instruction|begin\s+by)\s*:\s.*$")

# Phrases inglesi boilerplate da tagliare se parliamo in it
_EN_BOILER_RX = re.compile(
    r"(?i)\b(ready to help(?: you)?|here (?:are|is)\b|ways we can interact|i can help you with|let'?s (?:work|do) (?:this|it))\b.*"
)

# Sentinels noti
_SENTINELS = {
    "[[END_OF_OUTPUT]]",  # ASCII-clean
    "<<FINE>>",           # usato in alcune routine longform
    "ÔƒéENDÔƒé",          # mojibake visto in 'show'
}

def _strip_sentinels(text: str) -> str:
    if not text:
        return text
    out = text
    for tok in _SENTINELS:
        if tok in out:
            out = out.replace(tok, "")
    return out.strip()

def _sanitize_meta(s: str) -> str:
    if not s:
        return s
    return _META_RX.sub("", s).strip()

def _strip_drift_prefix(text: str) -> str:
    if not text:
        return text
    out = _DRIFT_PREFIX_RX.sub("", text).lstrip()
    return out if out else text

def _strip_drift_lines(text: str) -> str:
    if not text:
        return text
    return _DRIFT_LINES_RX.sub("", text)

def _soft_sentence_case(line: str) -> str:
    """
    Abbassa il "Title Case random" mantenendo acronimi corti (<=3).
    """
    words = line.split()
    if not words:
        return line
    out = []
    for i, w in enumerate(words):
        if w.isupper() and len(w) <= 3:
            out.append(w)
        else:
            out.append(w if i == 0 else w.lower())
    return " ".join(out)

def _looks_titlecased(line: str) -> bool:
    tokens = line.split()
    if len(tokens) < 6:
        return False
    caps = sum(1 for t in tokens if t[:1].isupper())
    return (caps / max(1, len(tokens))) >= 0.6

def _trim_en_boiler(line: str) -> str:
    # taglia dalla prima boiler in poi
    m = _EN_BOILER_RX.search(line)
    if not m:
        return line
    cut = line[:m.start()].rstrip()
    return cut

def _mojibake_soft_fix(s: str) -> str:
    """
    Fix leggerissimo: rimuove simboli di riempimento tipici e doppie decodifiche
    residue (senza mappare aggressivamente gli accenti).
    """
    if not s:
        return s
    # rimuove caratteri di riempimento frequenti
    s = s.replace("\uFFFD", "")  # replacement char
    # elimina sequenze spurie doppie (common artifacts)
    s = s.replace("Â ", " ")
    return s

def _anti_noise(text: str, lang: str = "it") -> str:
    """
    Micro-filtro anti-rumore:
    - rimuove meta, drift prefix/lines
    - trim boilerplate inglese
    - ammorbidisce Title Case casuale in headings troppo lunghe
    - normalizza spazi/punteggiatura leggera
    - strip sentinels
    """
    if not text:
        return text

    t = _sanitize_meta(text)
    t = _strip_drift_prefix(t)
    t = _strip_drift_lines(t)

    lines = t.splitlines()
    cleaned: List[str] = []
    for line in lines:
        l = line.rstrip()

        if not l.strip():
            cleaned.append(l)
            continue

        # taglia boilerplate inglese in righe miste
        if lang == "it":
            before = l
            l = _trim_en_boiler(l)
            # se la riga era solo boiler, salta
            if not l.strip() and before.strip():
                continue

        # riduci Title Case aggressivo su righe molto lunghe (tipo heading "miste")
        if _looks_titlecased(l) and len(l) > 80:
            l = _soft_sentence_case(l)

        # fix leggero mojibake
        l = _mojibake_soft_fix(l)

        # pulizia leggera spazi/punteggiatura ripetuta
        l = re.sub(r"\s{2,}", " ", l)
        l = re.sub(r"([!?])\1{1,}", r"\1", l)

        cleaned.append(l)

    t = "\n".join(cleaned)
    t = _strip_sentinels(t)
    return t.strip()

def _clean_delta_piece(piece: str) -> str:
    """
    Pulizia sicura per pezzi di stream (da applicare on-the-fly).
    - rimuove meta-markers semplici e sentinel se compaiono nel delta
    - elimina solo le righe 'Instruction:/Your task:/Begin by:' intere
    """
    if not piece:
        return piece
    p = _META_RX.sub("", piece)
    p = _DRIFT_LINES_RX.sub("", p)
    for tok in _SENTINELS:
        if tok in p:
            p = p.replace(tok, "")
    return p

# ============================== /api/chat ====================================

def call_ollama_chat(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    Chiamata NON-STREAM a /api/chat.
    Ritorna la stringa completa della risposta (post-processata).
    """
    url = _base_url() + "/api/chat"
    payload: Dict[str, Any] = {
        "model": model or _model_default(),
        "messages": messages,
        "stream": False,
    }
    opts = _merge_options(options)
    if opts:
        payload["options"] = opts

    ct, rt = _timeouts(timeout)
    resp = requests.post(url, json=payload, timeout=(ct, rt))
    resp.raise_for_status()
    data = resp.json() if resp.content else {}

    try:
        content = (data.get("message") or {}).get("content")
        if not isinstance(content, str):
            content = ""
    except Exception:
        content = ""

    # Post-processing anti-rumore
    return _anti_noise(content, lang="it")

def stream_ollama_chat(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    idle_timeout: float = 8.0,
    heartbeat_sec: float = 2.0,
) -> Generator[str, None, None]:
    """
    Chiamata STREAM a /api/chat.
    Genera chunk di testo (stringhe); può emettere heartbeat "" ogni heartbeat_sec.
    Termina se should_stop() True o se idle supera idle_timeout.
    Applica una pulizia leggera per-delta (non distruttiva).
    """
    url = _base_url() + "/api/chat"
    payload: Dict[str, Any] = {
        "model": model or _model_default(),
        "messages": messages,
        "stream": True,
    }
    opts = _merge_options(options)
    if opts:
        payload["options"] = opts

    ct, rt = _timeouts(timeout)
    resp = requests.post(url, json=payload, stream=True, timeout=(ct, rt))
    resp.raise_for_status()

    last_chunk_at = time.time()
    last_heartbeat_at = last_chunk_at

    try:
        # decode_unicode=False + decode manuale UTF-8: evita mojibake
        for raw in resp.iter_lines(decode_unicode=False, delimiter=b"\n"):
            if should_stop and should_stop():
                break

            now = time.time()
            if now - last_chunk_at > idle_timeout:
                break

            if not raw:
                if now - last_heartbeat_at >= heartbeat_sec:
                    last_heartbeat_at = now
                    yield ""  # heartbeat
                continue

            try:
                line = raw.decode("utf-8", errors="replace")
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("done") is True:
                break

            delta = ""
            m = obj.get("message")
            if isinstance(m, dict):
                c = m.get("content")
                if isinstance(c, str) and c:
                    delta = c
            if not delta:
                c2 = obj.get("content")
                if isinstance(c2, str) and c2:
                    delta = c2

            if not delta:
                if now - last_heartbeat_at >= heartbeat_sec:
                    last_heartbeat_at = now
                    yield ""
                continue

            last_chunk_at = last_heartbeat_at = now

            # Pulizia leggera per-delta (sicura su frammenti)
            yield _clean_delta_piece(delta)

    finally:
        try:
            resp.close()
        except Exception:
            pass

# ============================== /api/generate (diagnostica) ==================

def call_ollama_generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    debug: bool = False,
) -> tuple[str, Dict[str, Any]]:
    """
    Chiamata NON-STREAM a /api/generate, utile per diagnostica rapida.
    Ritorna (testo, diagnostica) con post-processing anti-rumore.
    """
    url = _base_url() + "/api/generate"
    payload: Dict[str, Any] = {
        "model": model or _model_default(),
        "prompt": prompt,
        "stream": False,
    }
    opts = _merge_options(options)
    if opts:
        payload["options"] = opts

    ct, rt = _timeouts(timeout)
    resp = requests.post(url, json=payload, timeout=(ct, rt))
    diag: Dict[str, Any] = {"status_code": resp.status_code}
    try:
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
    except Exception as e:
        diag["error"] = str(e)
        return "", diag

    text = ""
    try:
        text = data.get("response") or ""
        if not isinstance(text, str):
            text = ""
    except Exception:
        text = ""

    text = _anti_noise(text, lang="it")

    if debug:
        diag["request"] = payload
        diag["response_raw"] = data

    return text, diag

# ============================== __all__ ======================================

__all__ = [
    "call_ollama_chat",
    "stream_ollama_chat",
    "call_ollama_generate",
]
