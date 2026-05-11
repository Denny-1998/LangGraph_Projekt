import urllib.request
import urllib.error
import json
import sys

# ==========================================
# KONFIGURATION
# ==========================================
OLLAMA_API_URL = "http://localhost:11434/api/chat"
# Trage hier den genauen Namen deines Ollama-Modells ein.
# Z.B. "gemma", "gemma2", oder "gemma4", falls du einen eigenen Tag dafür hast.
MODEL_NAME = "gemma4" 

# ==========================================
# SYSTEM-PROMPT (Anweisung an die KI)
# ==========================================
SYSTEM_PROMPT = """Du bist der Game Master (Spielleiter) eines textbasierten interaktiven Adventure-Spiels. 

DEINE REGELN UND AUFGABEN:
1. GESCHICHTE AUSDENKEN: Überlege dir jetzt heimlich eine kurze, spannende Geschichte mit einem klaren Anfangsszenario, einem überraschenden Hauptevent in der Mitte und einem abschließenden Endszenario.
2. START: Präsentiere dem Spieler zu Beginn nur das Anfangsszenario. Beschreibe die Umgebung atmosphärisch.
3. INTERAKTION: Der Spieler wird simple Befehle eingeben (z.B. "umsehen", "aufheben [Gegenstand]", "ansehen [Objekt]", "gehe zu [Ort]", "inventar prüfen").
4. REAKTION: Reagiere logisch auf die Befehle. Wenn der Spieler etwas ansieht, beschreibe es. Wenn er etwas aufhebt, bestätige, dass er es nun hat. Lass den Spieler Rätsel lösen oder Hindernisse überwinden.
5. LENKUNG: Lenke den Spieler subtil, aber bestimmt in Richtung des Hauptevents und schließlich zum Endszenario. Gib ihm Hinweise in den Beschreibungen, wohin er gehen könnte oder was wichtig erscheint.
6. STIL: Fasse dich relativ kurz (1-3 Absätze pro Antwort), sei aber sehr atmosphärisch. Sprich den Spieler direkt in der 2. Person an ("Du stehst in...").
7. SPRACHE: Antworte IMMER auf Deutsch.
"""

def chat_with_ollama(messages):
    """
    Sendet die Nachrichten an die Ollama API und streamt die Antwort in die Konsole.
    """
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True # Streaming aktivieren für einen Schreibmaschinen-Effekt
    }
    
    req = urllib.request.Request(
        OLLAMA_API_URL, 
        data=json.dumps(data).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    
    full_response = ""
    
    try:
        with urllib.request.urlopen(req) as response:
            for line in response:
                if line:
                    # Ollama sendet NDJSON (Newline Delimited JSON)
                    chunk = json.loads(line.decode('utf-8'))
                    if "message" in chunk and "content" in chunk["message"]:
                        piece = chunk["message"]["content"]
                        print(piece, end='', flush=True)
                        full_response += piece
                        
    except urllib.error.URLError as e:
        print(f"\n[Fehler] Konnte keine Verbindung zu Ollama herstellen.")
        print(f"Details: {e}")
        print("Bitte stelle sicher, dass Ollama läuft (z.B. 'ollama serve' im Terminal).")
        sys.exit(1)
        
    print() # Neue Zeile nach der Antwort
    return full_response

def main():
    print("="*50)
    print("Willkommen beim LLM Text-Adventure!")
    print(f"Modell: {MODEL_NAME}")
    print("Tippe 'ende', 'quit' oder 'exit' um das Spiel zu verlassen.")
    print("="*50)
    print("\nInitialisiere Welt...\n")
    
    # Nachrichten-Verlauf initialisieren
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Das Spiel beginnt jetzt. Bitte präsentiere mir das Anfangsszenario und frage mich, was ich tun möchte."}
    ]
    
    # Initiale Antwort (Start der Geschichte) abrufen
    start_response = chat_with_ollama(messages)
    messages.append({"role": "assistant", "content": start_response})
    
    # Haupt-Spielschleife
    while True:
        try:
            # Spielereingabe
            user_input = input("\n> Was möchtest du tun? ")
            
            # Abbruchbedingungen
            if user_input.lower() in ['ende', 'quit', 'exit', 'verlassen']:
                print("Danke fürs Spielen! Bis zum nächsten Mal.")
                break
            
            if not user_input.strip():
                continue
                
            # Eingabe an den Verlauf anhängen
            messages.append({"role": "user", "content": user_input})
            
            # LLM nach der Konsequenz fragen
            print() # Optischer Abstand
            assistant_response = chat_with_ollama(messages)
            
            # Antwort der KI speichern, damit der Kontext erhalten bleibt
            messages.append({"role": "assistant", "content": assistant_response})
            
        except KeyboardInterrupt:
            print("\nSpiel durch Benutzer abgebrochen.")
            break

if __name__ == "__main__":
    main()