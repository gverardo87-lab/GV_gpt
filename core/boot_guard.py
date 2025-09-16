# core/boot_guard.py
# Boot guard minimale: conferma che stiamo girando nella root corretta
# e che le directory chiave esistono. Se vuoi, puoi attivare un controllo
# sul nome progetto via env GV_EXPECTED_PROJECT.

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
markers = [ROOT / "core", ROOT / "server", ROOT / ".env"]

# Verifica presenza marker minimi di progetto
if not all(p.exists() for p in markers):
    raise RuntimeError(f"Boot guard: root incompleta in {ROOT}")

# Controllo opzionale sul nome del progetto (solo se lo imposti)
expected = os.getenv("GV_EXPECTED_PROJECT", "").strip()
if expected:
    # usa il nome della cartella come “nome progetto” attuale
    current = ROOT.name
    if current != expected:
        raise RuntimeError(f"Boot guard: progetto inatteso. Atteso: {expected}, trovato: {current}")

# Se arrivi qui, tutto ok: l’import non solleva eccezioni.
