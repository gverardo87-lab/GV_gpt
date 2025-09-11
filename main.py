# main.py — CLI minimale per testare pipeline da terminale
from nlp_layer.preprocessing import analyze_text
from orchestrator.orchestrator import compose_prompt
from core.gpt_clienti import call_gpt

def main():
    print("GV GPT Middleware — CLI (digita 'exit' per uscire).")
    while True:
        user = input("Tu: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        nlp_data = analyze_text(user)
        prompt = compose_prompt(user, nlp_data)
        reply = call_gpt(prompt)
        print("GV:", reply, "\n")

if __name__ == "__main__":
    main()
