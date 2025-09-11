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
