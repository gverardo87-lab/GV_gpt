# orchestrator/orchestrator.py — Composizione del prompt arricchito
from typing import Dict, Any

def compose_prompt(user_text: str, nlp_data: Dict[str, Any]) -> str:
    header = (
        f"[INTENT: {nlp_data.get('intent')}] "
        f"[ENTITIES: {nlp_data.get('entities')}] "
        f"[SPACY: {'ON' if nlp_data.get('spacy') else 'OFF'}]\n"
    )
    instructions = (
        "Istruzioni: capisci la richiesta e rispondi in modo chiaro e operativo. "
        "Se individui date/ore/valori utili (es. 'domani alle 10'), esplicitali; "
        "se servono azioni, elencale in punti.\n"
    )
    return header + instructions + "Utente: " + user_text


# --- system prompt modulare per intent --------------------------------------
def system_prompt_for_intent(intent: str) -> str:
    """
    Restituisce un system prompt conciso e operativo, variato per intent.
    Mantieni il tono sintetico, preciso e orientato all’azione.
    """
    base = (
        "Rispondi in italiano, chiaro e operativo. Usa elenchi puntati dove utile. "
        "Specifica sempre numeri/date/dettagli in modo esplicito. "
        "Evita introduzioni prolisse e ripetizioni."
    )
    i = (intent or "general").lower()

    if i == "coding":
        return base + " Fornisci snippet minimi funzionanti, passi di debug e note su edge cases."
    if i == "business":
        return base + " Offri struttura, KPI, checklist operative e rischi con relative mitigazioni."
    if i == "nutrition":
        return base + " Includi un breve disclaimer non-clinico e suggerimenti generali basati su linee guida."
    if i == "study":
        return base + " Preferisci schemi a punti, esempi rapidi, mnemoniche e (se utile) 3 mini-quiz finali."

    # default
    return base + " Adatta registro e profondità al contesto della richiesta."
