# core/gpt_clienti.py
import os, time
from typing import List, Dict, Any

from openai import OpenAI
from openai import RateLimitError  # sdk 1.x
from openai import APIError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _model():
    return os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

def call_gpt_chat(messages: List[Dict[str, Any]], temperature: float = 0.7, retries: int = 2) -> str:
    delay = 1.5
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=_model(),
                messages=messages,
                temperature=temperature,
                stream=False,
            )
            # sdk 1.x
            return resp.choices[0].message.content
        except RateLimitError as e:
            last_err = e
            time.sleep(delay); delay *= 2
        except APIError as e:
            last_err = e
            # status 429 o codici di quota
            code = getattr(e, "status_code", None)
            text = str(e).lower()
            if code == 429 or "insufficient_quota" in text or "rate" in text and "limit" in text:
                time.sleep(delay); delay *= 2
            else:
                break
        except Exception as e:
            last_err = e
            break
    # errore finale con messaggio pulito
    raise RuntimeError(f"OpenAI error: {str(last_err)}")

def stream_gpt_chat(messages: List[Dict[str, Any]], temperature: float = 0.7):
    # streaming standard; lascia che l'app gestisca il fallback se fallisce
    stream = client.chat.completions.create(
        model=_model(),
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = (chunk.choices[0].delta.content or "")
        if delta:
            yield delta
