# orchestrator/orchestrator.py — Composizione del prompt arricchito (definitivo)
from __future__ import annotations
from typing import Any, Dict
import textwrap

# ------------------------------------------------------------
# Helper: formattazione ordinata degli slot estratti dall’NLP
# ------------------------------------------------------------
def _fmt_slots(slots: Dict[str, Any] | None) -> str:
    if not slots:
        return "- (nessuno rilevato)"
    lines = []
    # Ordine stabile e leggibile
    for k in ("platform", "language", "tone", "length", "audience", "age_range", "location", "deadline"):
        if k in slots and slots[k]:
            lines.append(f"- {k}: {slots[k]}")
    return "\n".join(lines) if lines else "- (nessuno rilevato)"

# -----------------------------------------------------------------
# compose_prompt: compone il contenuto utente in base all'intent
# - META_PROMPT: passa uno SPEC chiaro + slot (non svolge il task)
# - ALTRI: conserva header diagnostico + istruzioni operative
# -----------------------------------------------------------------
def compose_prompt(user_text: str, nlp_data: Dict[str, Any]) -> str:
    intent = (nlp_data.get("intent") or "general").lower()

    # ✨ Modalità META-PROMPT
    if intent == "meta_prompt":
        spec = f"""
        ### META_PROMPT_SPEC

        INPUT_GREZZO:
        {user_text or ''}

        SLOTS_RILEVATI:
        {_fmt_slots(nlp_data.get('slots'))}

        NOTE:
        - Mantieni la lingua dell’utente salvo diversa indicazione.
        - Se mancano alcuni slot, non inventarli; lascia placeholders [DA SPECIFICARE].
        - L’obiettivo è produrre un prompt incollabile che un altro GPT possa usare per svolgere il compito.
        - NON svolgere il compito. Restituisci solo il prompt finale.
        """
        return textwrap.dedent(spec).strip()

    # ✅ Default: header diagnostico comodo + istruzioni sintetiche
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
    return header + instructions + "Utente: " + (user_text or "")

# -----------------------------------------------------------------
# system_prompt_for_intent: regole di comportamento per ogni intent
# - META_PROMPT: emetti SOLO un blocco ```PROMPT ...``` strutturato
# - Altri intent: stile operativo, sintetico, orientato all’azione
# -----------------------------------------------------------------
def system_prompt_for_intent(intent: str) -> str:
    base = (
        "Rispondi in italiano, chiaro e operativo. "
        "Usa elenchi puntati dove utile. "
        "Specifica sempre numeri/date/dettagli in modo esplicito. "
        "Evita introduzioni prolisse e ripetizioni."
    )
    i = (intent or "general").lower()

    # ✨ Modalità META-PROMPT (Prompt Optimizer)
    if i == "meta_prompt":
        return (
            base
            + " MODALITÀ META-PROMPT: non svolgere il compito dell'utente."
              " Trasforma il META_PROMPT_SPEC in un *prompt ottimizzato* e restituiscilo"
              " SOLO dentro un blocco di codice con linguetta 'PROMPT'."
              " Struttura il prompt così:"
              " 1) Ruolo/Persona del modello;"
              " 2) Obiettivo;"
              " 3) Contesto/Risorse (usa SLOTS_RILEVATI come guida; lascia [DA SPECIFICARE] se mancano);"
              " 4) Vincoli/Regole;"
              " 5) Stile/Tono;"
              " 6) Output atteso (formato ed esempi);"
              " 7) Parametri (es. temperature, max tokens; niente ragionamento interno se non richiesto esplicitamente)."
              " Vietato aggiungere testo extra fuori dal blocco."
        )

    # Intent specifici già presenti nel tuo flusso
    if i == "coding":
        return base + " Fornisci snippet minimi funzionanti, passi di debug e note su edge cases."
    if i == "business":
        return base + " Offri struttura, KPI, checklist operative e rischi con relative mitigazioni."
    if i == "nutrition":
        return base + " Includi un breve disclaimer non-clinico e suggerimenti generali basati su linee guida."
    if i == "study":
        return base + " Preferisci schemi a punti, esempi rapidi, mnemoniche e (se utile) 3 mini-quiz finali."

    # Default
    return base + " Adatta registro e profondità al contesto della richiesta."
