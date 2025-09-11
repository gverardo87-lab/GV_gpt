# core/gpt_client.py
import os, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Carica .env dalla root del progetto (robusto a working dir diverse)
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def _get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY non trovato (.env in root).")
    return OpenAI(api_key=api_key)

def call_gpt_chat(messages, temperature: float = 0.7, retries: int = 1):
    """
    messages: lista di dict [{"role":"system/user/assistant","content":"..."}]
    """
    client = _get_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    last_err = None
    for _ in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(0.6)
    raise last_err

def stream_gpt_chat(messages, temperature: float = 0.7):
    """
    Generatore che emette testo man mano che arriva (stream=True).
    Usalo con Streamlit: st.write_stream(stream_gpt_chat(messages))
    """
    client = _get_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True
    )
    for chunk in stream:
        # Nuovo SDK: il testo incrementale è in choices[0].delta.content
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            yield delta

# Compat vecchia: singolo messaggio
def call_gpt(user_message: str) -> str:
    system = {"role": "system",
              "content": "Sei GV Assistant. Italiano, chiaro, operativo. Usa elenchi quando aiuta."}
    return call_gpt_chat([system, {"role": "user", "content": user_message}])
