# core/gpt_client.py — Wrapper OpenAI
import os
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[1]  # .../GV_gpt
load_dotenv(ROOT / ".env")

load_dotenv()

def _get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY non trovato. Crea .env nella root del progetto.")
    return OpenAI(api_key=api_key)

def call_gpt(user_message: str) -> str:
    client = _get_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content":
             "Sei GV Assistant, l'assistente custom di Giacomo Verardo. "
             "Rispondi in italiano con chiarezza, tono caldo-professionale; se utile proponi passi operativi."},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7
    )
    return resp.choices[0].message.content