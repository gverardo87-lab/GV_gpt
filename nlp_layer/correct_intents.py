# -*- coding: utf-8 -*-
"""
correct_intents.py
------------------
Patch A completa:
- Caricamento tollerante di intent_examples.json (rimozione virgole finali, BOM, commenti).
- Lettura robusta di data/nlp_logs.jsonl (salta righe rotte, micro-fix su graffe finali).
- Dedup/normalize esempi per classe.
- Pipeline di correzione supervisionata dai log:
    - Cerca campi di verità/correzione: correct_intent, gold_intent, target_intent,
      fixed_intent, true_intent, label, intent_corrected, intent_label.
    - Se presenti, aggiunge il testo alla classe indicata.
- Dry-run per default (stampa il riepilogo). Usa --apply per scrivere i cambi.
- Crea backup timestamp del file intent_examples.json prima di sovrascrivere.

Uso:
    python nlp_layer/correct_intents.py                 # dry-run (non scrive)
    python nlp_layer/correct_intents.py --apply         # applica le modifiche
    python nlp_layer/correct_intents.py --min-len 6     # ignora testi < 6 caratteri
"""

from __future__ import annotations
from pathlib import Path
import argparse
import datetime as dt
import json
import re
from typing import Dict, Iterable, Tuple, Optional


# ---------------------------
# Path discovery robusto
# ---------------------------

def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Risale massimo 6 livelli finché trova una cartella 'data'.
    """
    start = start or Path(__file__).resolve()
    cur = start if start.is_dir() else start.parent
    for _ in range(6):
        if (cur / "data").exists():
            return cur
        cur = cur.parent
    # Fallback: stessa cartella del file
    return Path(__file__).resolve().parent


ROOT = find_project_root()
DATA = ROOT / "data"
EXAMPLES_FILE = DATA / "intent_examples.json"
LOG_FILE = DATA / "nlp_logs.jsonl"


# ---------------------------
# Sanitizzazione JSON tollerante
# ---------------------------

def sanitize_json_str(s: str) -> str:
    """
    - Rimuove BOM se presente
    - Rimuove commenti //... e /* ... */
    - Rimuove virgole finali prima di '}' o ']'
    - Compatta spazi ripetuti prima di newline
    """
    if s and s[:1] == "\ufeff":
        s = s.lstrip("\ufeff")
    s = re.sub(r"(?m)//.*?$", "", s)                # commenti // ...
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)     # commenti /* ... */
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r",\s*([}\]])", r"\1", s)        # trailing commas
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s


def load_examples(path: Path) -> Dict[str, list]:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads(sanitize_json_str(raw))
    if not isinstance(data, dict):
        raise ValueError("intent_examples.json non è un oggetto JSON { ... }")
    return data


def dedupe_and_normalize_examples(examples: Dict[str, list]) -> Dict[str, list]:
    cleaned: Dict[str, list] = {}
    for label, arr in (examples or {}).items():
        if not isinstance(arr, list):
            continue
        seen = set()
        vals = []
        for x in arr:
            if not isinstance(x, str):
                continue
            t = x.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            vals.append(t)
        if vals:
            cleaned[label] = vals
    return cleaned


def save_examples_with_backup(path: Path, data: Dict[str, list]) -> Path:
    """
    Scrive con backup timestamp prima di sovrascrivere.
    Ritorna il path del backup creato (o None se non necessario).
    """
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak-{ts}")
    if path.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return backup


# ---------------------------
# Lettura robusta JSONL
# ---------------------------

def iter_valid_jsonl(path: Path) -> Iterable[dict]:
    """
    Itera record JSONL validi:
    - Skippa righe vuote o corrotte
    - Micro-fix: chiusura '}' mancante se la riga inizia con '{'
    - Rimozione BOM e virgole finali
    """
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s[:1] == "\ufeff":
                s = s.lstrip("\ufeff")

            fixed = s
            if fixed.startswith("{") and not fixed.endswith("}"):
                fixed = fixed + "}"

            try:
                yield json.loads(fixed)
            except Exception:
                try:
                    yield json.loads(s.rstrip(", \t\r\n"))
                except Exception:
                    # riga irrecuperabile
                    continue


# ---------------------------
# Estrazione correzioni dai log
# ---------------------------

CANDIDATE_LABEL_FIELDS = (
    "correct_intent",
    "gold_intent",
    "target_intent",
    "fixed_intent",
    "true_intent",
    "label",
    "intent_corrected",
    "intent_label",
)

TEXT_FIELDS = (
    "text",
    "user_text",
    "message",
    "input",
)


def extract_text(rec: dict) -> Optional[str]:
    for k in TEXT_FIELDS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def extract_correct_label(rec: dict) -> Optional[str]:
    # priorità alle correzioni esplicite
    for k in CANDIDATE_LABEL_FIELDS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # fallback: se c'è un nested 'correction' con 'intent'
    corr = rec.get("correction") or rec.get("fix")
    if isinstance(corr, dict):
        v = corr.get("intent") or corr.get("label")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def collect_corrections(
    logs: Iterable[dict],
    min_len: int = 6
) -> Dict[str, list]:
    """
    Raccoglie {label: [text,...]} solo dove è presente una label "corretta".
    Ignora testi troppo corti.
    """
    additions: Dict[str, list] = {}
    for rec in logs:
        txt = extract_text(rec)
        lab = extract_correct_label(rec)
        if not txt or not lab:
            continue
        if len(txt) < min_len:
            continue
        bucket = additions.setdefault(lab, [])
        bucket.append(txt)
    return additions


def merge_additions(
    examples: Dict[str, list],
    additions: Dict[str, list]
) -> Tuple[Dict[str, list], Dict[str, int]]:
    """
    Unisce additions negli examples con dedup per classe.
    Ritorna (merged_examples, counts_added_per_label)
    """
    merged = {k: list(v) for k, v in (examples or {}).items()}
    added_counts: Dict[str, int] = {}

    for lab, texts in (additions or {}).items():
        if not isinstance(texts, list) or not texts:
            continue
        dest = merged.setdefault(lab, [])
        seen = set(dest)
        added = 0
        for t in texts:
            t = t.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            dest.append(t)
            added += 1
        if added:
            added_counts[lab] = added

    merged = dedupe_and_normalize_examples(merged)
    return merged, added_counts


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correggi/aggiorna intent_examples.json dai log supervisionati.")
    p.add_argument("--apply", action="store_true", help="Scrive le modifiche (default: dry-run).")
    p.add_argument("--min-len", type=int, default=6, help="Lunghezza minima del testo per essere considerato (default: 6).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"ROOT: {ROOT}")
    print(f"DATA: {DATA}")
    print(f"EXAMPLES_FILE: {EXAMPLES_FILE}")
    print(f"LOG_FILE: {LOG_FILE}")

    # 1) Carica esempi con loader tollerante + dedup
    if not EXAMPLES_FILE.exists():
        raise FileNotFoundError(f"File non trovato: {EXAMPLES_FILE}")
    examples = load_examples(EXAMPLES_FILE)
    examples = dedupe_and_normalize_examples(examples)

    # 2) Leggi log robustamente
    logs = list(iter_valid_jsonl(LOG_FILE))
    print(f"Log letti (validi): {len(logs)}")

    # 3) Raccogli correzioni supervisionate
    additions = collect_corrections(logs, min_len=args.min_len)

    tot_new = sum(len(v) for v in additions.values())
    print(f"Candidate additions (da correzioni esplicite): {tot_new}")
    if not tot_new:
        print("Nessuna correzione esplicita trovata nei log. (Cerca campi come 'correct_intent', 'gold_intent', ...)")
        return

    # 4) Merge & riepilogo
    merged, added_counts = merge_additions(examples, additions)

    print("\n[Riepilogo per etichetta]")
    for lab in sorted(added_counts.keys()):
        print(f"  - {lab}: +{added_counts[lab]} frasi")

    if not args.apply:
        print("\nDry-run completato. Usa --apply per scrivere su intent_examples.json (creerò un backup).")
        return

    # 5) Scrivi con backup
    backup = save_examples_with_backup(EXAMPLES_FILE, merged)
    print(f"\n✔ Modifiche applicate.")
    if backup and backup.exists():
        print(f"Backup creato: {backup.name}")


if __name__ == "__main__":
    main()
