# core/memory.py — Memoria persistente leggera (JSON)
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # .../GV_gpt
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
MEM_FILE = DATA_DIR / "memory_default.json"

# quante "righe" di chat manteniamo (user/assistant contano ciascuno 1)
MAX_TURNS = int(os.getenv("GV_MEMORY_TURNS", 30))

def _is_msg_ok(m):
    return isinstance(m, dict) and m.get("role") in {"user", "assistant"} and isinstance(m.get("content"), str)

def load_memory():
    """Ritorna una lista di messaggi [{'role': 'user'/'assistant', 'content': '...'}]"""
    if not MEM_FILE.exists():
        return []
    try:
        data = json.loads(MEM_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list) and all(_is_msg_ok(x) for x in data):
            return data[-MAX_TURNS:]
    except Exception:
        pass
    return []

def save_memory(messages):
    """Salva SOLO gli ultimi MAX_TURNS messaggi."""
    if not isinstance(messages, list):
        return
    trimmed = [m for m in messages if _is_msg_ok(m)][-MAX_TURNS:]
    MEM_FILE.write_text(json.dumps(trimmed, ensure_ascii=False, indent=2), encoding="utf-8")

def clear_memory():
    try:
        if MEM_FILE.exists():
            MEM_FILE.unlink()
    except Exception:
        pass
