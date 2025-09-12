# main.py — CLI engine-aware
from core.engine import call_chat
from nlp_layer.preprocessing import analyze_text
from orchestrator.orchestrator import compose_prompt
try:
    from orchestrator.orchestrator import system_prompt_for_intent
except Exception:
    def system_prompt_for_intent(intent: str) -> str:
        return "Rispondi in italiano, chiaro e operativo."

def main():
    print("GV GPT Middleware — CLI (exit per uscire)")
    history = []
    while True:
        user = input("Tu: ").strip()
        if user.lower() in {"exit", "quit"}: break
        history.append({"role":"user","content": user})
        nlp = analyze_text(user)
        enriched = compose_prompt(user, nlp)
        system = {"role":"system","content": system_prompt_for_intent(nlp.get("intent","general"))}
        messages = [system] + history[:-1] + [{"role":"user","content": enriched}]
        try:
            reply = call_chat(messages)
            print("GV:", reply, "\n")
            history.append({"role":"assistant","content": reply})
        except Exception as e:
            print("Errore modello:", e, "\n")

if __name__ == "__main__":
    main()
