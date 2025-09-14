# nlp_layer/correct_intents.py — correzione manuale dei log
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG_FILE = DATA / "nlp_logs.jsonl"
EXAMPLES_FILE = DATA / "intent_examples.json"

def main():
    if not LOG_FILE.exists():
        print("❌ Nessun log trovato.")
        return

    # Carica dataset esistente
    if EXAMPLES_FILE.exists():
        examples = json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))
    else:
        examples = {}

    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("score", 0) < 0.6 or entry["intent"] == "general":
                print("\n📝 Testo:", entry["text"])
                print("   Predetto:", entry["intent"], "score=", entry["score"])
                label = input("   Correggi intent (ENTER per skip): ").strip()
                if label:
                    examples.setdefault(label, []).append(entry["text"])

    # Salva dataset aggiornato
    EXAMPLES_FILE.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n✔ Dataset aggiornato:", EXAMPLES_FILE)

if __name__ == "__main__":
    main()
