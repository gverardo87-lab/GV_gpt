# nlp_layer/train_intents.py — retraining LogisticRegression sugli esempi (robusto + calibrato)
import json
import os
import joblib
import numpy as np
from collections import Counter
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXAMPLES_FILE = DATA / "intent_examples.json"
MODEL_FILE = DATA / "intent_model.pkl"
EVAL_DIR = DATA / "eval"

RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if not EXAMPLES_FILE.exists():
        print("❌ Nessun dataset di esempi trovato:", EXAMPLES_FILE)
        return

    # Carica esempi in forma {intent: [frasi, ...]}
    examples = json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))
    X, y = [], []
    for intent, sents in examples.items():
        for s in sents:
            X.append(s)
            y.append(intent)

    counts = Counter(y)
    print("📊 Totale esempi:", len(X))
    print("🧩 Intent:", ", ".join(sorted(counts.keys())))
    print("⚖️  Distribuzione per classe:", {k: int(v) for k, v in counts.items()})

    # Embeddings
    model_emb = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    X_emb = model_emb.encode(X, convert_to_numpy=True, normalize_embeddings=True)

    # CV stratificata (se i dati lo consentono)
    min_class = min(counts.values()) if counts else 0
    n_splits = max(2, min(DEFAULT_N_SPLITS, min_class)) if min_class >= 2 else 0

    reports = []
    cms = []
    classes_out = None

    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for fold, (tr, va) in enumerate(skf.split(X_emb, y), start=1):
            base = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
            # calibrazione probabilità (sigmoid); cv ridotta se dataset piccolo
            cv_for_cal = min(3, n_splits)
            clf_cv = CalibratedClassifierCV(base, cv=cv_for_cal, method="sigmoid")
            clf_cv.fit(X_emb[tr], np.array(y)[tr])

            y_pred = clf_cv.predict(X_emb[va])
            rep = classification_report(np.array(y)[va], y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(np.array(y)[va], y_pred, labels=clf_cv.classes_)
            classes_out = clf_cv.classes_
            reports.append(rep)
            cms.append(cm)

            print(f"[fold {fold}/{n_splits}] macro-F1:", round(rep.get("macro avg", {}).get("f1-score", 0.0), 3))

        # Salva sintesi valutazione
        summary = {
            "classes": list(map(str, classes_out)) if classes_out is not None else [],
            "cv_macro_f1_mean": float(np.mean([r["macro avg"]["f1-score"] for r in reports])) if reports else None,
            "cv_macro_f1_std": float(np.std([r["macro avg"]["f1-score"] for r in reports])) if reports else None,
            "n_splits": int(n_splits),
            "counts_per_class": {str(k): int(v) for k, v in counts.items()},
        }
        (EVAL_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if cms:
            np.savetxt(EVAL_DIR / "confusion_matrix_mean.txt", np.mean(cms, axis=0), fmt="%.3f")
        print("[eval] summary salvato in", EVAL_DIR)

    # Fit finale su tutti i dati (modello da esportare)
    base_final = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    # calibrazione finale se possibile
    if min_class >= 3:
        cv_for_cal_final = 3
    elif min_class >= 2:
        cv_for_cal_final = 2
    else:
        cv_for_cal_final = None

    if cv_for_cal_final:
        clf_final = CalibratedClassifierCV(base_final, cv=cv_for_cal_final, method="sigmoid")
        clf_final.fit(X_emb, y)
        model_to_save = clf_final
    else:
        base_final.fit(X_emb, y)
        model_to_save = base_final

    joblib.dump(model_to_save, MODEL_FILE)
    print("✔ Modello salvato in:", MODEL_FILE)


if __name__ == "__main__":
    main()
