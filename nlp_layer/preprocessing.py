# nlp_layer/preprocessing.py — NLP con continuous learning
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# === Percorsi ===
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

LOG_FILE = DATA / "nlp_logs.jsonl"
EXAMPLES_FILE = DATA / "intent_examples.json"
MODEL_FILE = DATA / "intent_model.pkl"

# === Modello embedding ===
_EMB_MODEL = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# === Carica dataset dinamico ===
def load_examples() -> Dict[str, List[str]]:
    if not EXAMPLES_FILE.exists():
        return {}
    return json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))

# === Carica modello LogisticRegression se esiste ===
def load_model():
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)
    return None

_MODEL = load_model()
_INTENTS = list(load_examples().keys())

def _embed(texts: List[str]):
    return _EMB_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# === Intent detection ===
def _guess_intent(user_input: str) -> Tuple[str, float]:
    if _MODEL is None or not _INTENTS:
        return ("general", 0.0)
    vec = _embed([user_input])
    proba = _MODEL.predict_proba(vec)[0]
    best_idx = proba.argmax()
    return _MODEL.classes_[best_idx], float(proba[best_idx])

# === Analisi NLP ===
def analyze_text(user_input: str) -> Dict[str, Any]:
    intent, score = _guess_intent(user_input)

    # Regex entities
    entities: List[Tuple[str, str]] = []
    if re.search(r"\b\d+(\.\d+)?\s*€", user_input):
        entities.append(("€", "MONEY"))
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_input):
        entities.append(("EMAIL", "CONTACT"))

    result = {
        "text": user_input,
        "intent": intent,
        "score": round(score, 3),
        "entities": entities,
        "tokens": user_input.split(),
    }

    # Log automatico
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"time": datetime.utcnow().isoformat(), **result}, ensure_ascii=False
            )
            + "\n"
        )

    return result
