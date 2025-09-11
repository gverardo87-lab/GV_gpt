# nlp_layer/preprocessing.py — NLP locale (spaCy) gratuito e veloce
from typing import Dict, Any, List, Tuple

try:
    import spacy
    _NLP = spacy.load("it_core_news_sm")
    _SPACY_OK = True
except Exception:
    _NLP = None
    _SPACY_OK = False

_INTENT_KEYWORDS = {
    "nutrition": ["calorie", "dieta", "nutrizione", "fibre", "colesterolo"],
    "coding":    ["python", "codice", "bug", "script", "funzione", "api"],
    "business":  ["analisi", "mercato", "kpi", "preventivo", "offerta", "fatturato"],
    "calendar":  ["domani", "alle", "evento", "riunione", "calendar", "appuntamento"],
}

def _guess_intent(text: str) -> str:
    t = text.lower()
    for intent, keys in _INTENT_KEYWORDS.items():
        if any(k in t for k in keys):
            return intent
    return "general"

def analyze_text(user_input: str) -> Dict[str, Any]:
    entities: List[Tuple[str, str]] = []
    tokens: List[str] = user_input.split()

    if _SPACY_OK:
        doc = _NLP(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        tokens = [t.text for t in doc]

    return {
        "intent": _guess_intent(user_input),
        "entities": entities,
        "tokens": tokens,
        "spacy": _SPACY_OK,
    }
