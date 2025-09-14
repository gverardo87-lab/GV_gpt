from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = DATA / "nlp_logs.jsonl"
DST = DATA / "nlp_logs.cleaned.jsonl"

def iter_valid_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s and s[0] == "\ufeff":
                s = s.lstrip("\ufeff")
            fixed = s if s.endswith("}") else (s + "}" if s.startswith("{") else s)
            try:
                yield json.loads(fixed)
            except Exception:
                try:
                    yield json.loads(s.rstrip(", \t\r\n"))
                except Exception:
                    continue

def main():
    if not SRC.exists():
        print("Nessun file di log da riparare:", SRC)
        return
    valid = list(iter_valid_jsonl(SRC))
    with open(DST, "w", encoding="utf-8") as f:
        for obj in valid:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Riparazione completata. Righe valide: {len(valid)} → {DST}")

if __name__ == "__main__":
    main()
