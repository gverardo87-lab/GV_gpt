# test_embeddings.py — Demo embedding intent detection con MiniLM
from sentence_transformers import SentenceTransformer, util

# === Setup modello ===
print("⏳ Carico il modello MiniLM multilingua...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Label di intent che usi nel tuo NLP
INTENT_LABELS = [
    "nutrition", "coding", "business", "calendar",
    "study", "health", "motivation", "finance"
]

# Pre-calcoliamo gli embedding delle label
intent_emb = model.encode(INTENT_LABELS, convert_to_tensor=True, normalize_embeddings=True)

# === Frasi di test ===
TEST_SENTENCES = [
    "Voglio imparare a programmare in Python",
    "Come faccio a ridurre il colesterolo?",
    "Domani ho una riunione alle 10",
    "Vorrei capire il bilancio annuale della mia azienda",
    "Devo prepararmi per l’esame di anatomia",
    "Oggi non ho motivazione per studiare",
    "Investire in azioni o obbligazioni conviene?",
]

# === Calcolo similarità ===
for s in TEST_SENTENCES:
    emb = model.encode(s, convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(emb, intent_emb)[0]

    best_idx = int(cos_scores.argmax())
    best_label = INTENT_LABELS[best_idx]
    best_score = float(cos_scores[best_idx])

    print(f"\n📝 Input: {s}")
    print(" → Intent predetto:", best_label, f"(score={best_score:.3f})")
    print(" → Ranking:")
    for i, label in enumerate(INTENT_LABELS):
        print(f"   {label:10s}: {float(cos_scores[i]):.3f}")
