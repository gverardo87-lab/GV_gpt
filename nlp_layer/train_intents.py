# nlp_layer/train_intents.py — retraining LogisticRegression sugli esempi
import json
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXAMPLES_FILE = DATA / "intent_examples.json"
MODEL_FILE = DATA / "intent_model.pkl"

def main():
    if not EXAMPLES_FILE.exists():
        print("❌ Nessun dataset di esempi trovato.")
        return

    examples = json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))
    X, y = [], []
    for intent, sents in examples.items():
        for s in sents:
            X.append(s)
            y.append(intent)

    print("📊 Totale esempi:", len(X), "Intent:", set(y))

    model_emb = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    X_emb = model_emb.encode(X, convert_to_numpy=True, normalize_embeddings=True)

    clf = LogisticRegression(max_iter=300, solver="lbfgs")
    clf.fit(X_emb, y)

    joblib.dump(clf, MODEL_FILE)
    print("✔ Modello salvato in:", MODEL_FILE)

if __name__ == "__main__":
    main()
