#!/usr/bin/env python3
"""
cli.py — Interaktives CLI-Menü für die RAG-Pipelines.
"""

import json
import os
import subprocess
import sys

# ─────────────────────────────────────────────
# Pfade zu den Pipeline-Skripten (relativ zu cli.py)
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INGESTION_SCRIPT = os.path.join(SCRIPT_DIR, "1 - IngestionPipeline.py")
SLOP_SCRIPT      = os.path.join(SCRIPT_DIR, "2- SlopCreationPipeline.py")
CONFIG_FILE      = os.path.join(SCRIPT_DIR, "rag_config.json")

# ─────────────────────────────────────────────
# Standardwerte
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Ingestion Pipeline
    "dok_ordner":        "./meine_pdfs",
    "datenbank_ordner":  "./chroma_db",
    "token_limit":       400,
    "chunk_overlap":     200,
    "chunking_mode":     "semantic",          # "semantic" oder "fixed"
    "embedding_model":   "all-MiniLM-L6-v2",

    # Slop Creation Pipeline
    "ollama_model":      "gemma4:e4b",
    "llm_temperature":   0.0,
    "retriever_k":       3,
    "max_rewrite_loops": 10,
}

# ─────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            gespeichert = json.load(f)
        config = {**DEFAULT_CONFIG, **gespeichert}
    else:
        config = dict(DEFAULT_CONFIG)
        save_config(config) # Speichert sofort die Initial-Datei ab
    return config


def save_config(config: dict) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def dividing_line(char="─", breite=52):
    print(char * breite)


def header(titel: str):
    clear()
    dividing_line("═")
    print(f"  🤖  RAG Pipeline Manager  │  {titel}")
    dividing_line("═")
    print()


def prompt_input(prompt: str, standard: str = "") -> str:
    anzeige = f"  {prompt}"
    if standard:
        anzeige += f" [{standard}]"
    anzeige += ": "
    wert = input(anzeige).strip()
    return wert if wert else standard


def confirmation(frage: str) -> bool:
    antwort = input(f"  {frage} (j/n): ").strip().lower()
    return antwort in ("j", "ja", "y", "yes")


def wait():
    input("\n  [Eingabe drücken, um fortzufahren...]")


def run_script(skript_pfad: str, args: list = None):
    """Führt ein Pipeline-Skript direkt via Subprocess aus."""
    if not os.path.exists(skript_pfad):
        print(f"  [!] Fehler: Skript wurde nicht gefunden unter: {skript_pfad}")
        return
    try:
        cmd = [sys.executable, skript_pfad]
        if args:
            cmd.extend(args)
        subprocess.run(cmd)
    except Exception as e:
        print(f"  [!] Fehler beim Ausführen des Skripts: {e}")


# ─────────────────────────────────────────────
# Einstellungsmenü
# ─────────────────────────────────────────────

SETTINGS = [
    ("dok_ordner",        "Dokument-Ordner (Input)",          "str", None),
    ("datenbank_ordner",  "Datenbank-Ordner (ChromaDB)",      "str", None),
    ("embedding_model",   "Embedding-Modell (HuggingFace)",   "str", None),
    ("token_limit",       "Token-Limit pro Chunk",            "int", None),
    ("chunk_overlap",     "Chunk-Überlappung (Zeichen)",      "int", None),
    ("chunking_mode",     "Chunking-Modus",                   "opt", ["semantic", "fixed"]),
    ("ollama_model",      "Ollama LLM-Modell",                "str", None),
    ("llm_temperature",   "LLM Temperatur (0.0–1.0)",         "flt", None),
    ("retriever_k",       "Retriever k (Anzahl Chunks)",      "int", None),
    ("max_rewrite_loops", "Max. Umformulierungs-Versuche",    "int", None),
]


def settings_menu(config: dict) -> dict:
    while True:
        header("⚙  Einstellungen")

        for i, (key, name, type, options) in enumerate(SETTINGS, 1):
            value = config[key]
            suffix = ""
            if type == "opt":
                suffix = f"  ({' / '.join(options)})"
            print(f"  [{i:2}] {name:<40} = {value}{suffix}")

        print()
        dividing_line()
        print("  [r]  Auf Standardwerte zurücksetzen")
        print("  [0]  Zurück zum Hauptmenü")
        dividing_line()
        selection = input("\n  Nummer zum Bearbeiten: ").strip().lower()

        if selection == "0":
            break

        if selection == "r":
            if confirmation("Wirklich alle Einstellungen zurücksetzen?"):
                config = dict(DEFAULT_CONFIG)
                save_config(config)
                print("  [OK] Standardwerte wiederhergestellt.")
                wait()
            continue

        try:
            idx = int(selection) - 1
            if not 0 <= idx < len(SETTINGS):
                raise ValueError
        except ValueError:
            print("  [!] Ungültige Eingabe.")
            wait()
            continue

        key, name, type, options = SETTINGS[idx]
        current = config[key]

        print()
        if type == "opt":
            print(f"  {name}")
            for j, opt in enumerate(options, 1):
                markiert = " ◀" if opt == current else ""
                print(f"    [{j}] {opt}{markiert}")
            auswahl = input("  Wahl: ").strip()
            try:
                neuer_wert = options[int(auswahl) - 1]
                config[key] = neuer_wert
                save_config(config)
                print(f"  [OK] '{key}' → {neuer_wert}")
            except (ValueError, IndexError):
                print("  [!] Ungültige Auswahl.")
        else:
            rawInput = prompt_input(f"Neuer Wert für '{name}'", str(current))
            try:
                if   type == "int": config[key] = int(rawInput)
                elif type == "flt": config[key] = float(rawInput)
                else:              config[key] = rawInput
                save_config(config)
                print(f"  [OK] '{key}' → {config[key]}")
            except ValueError:
                print(f"  [!] Ungültiger Wert für Typ '{type}'.")

        wait()

    return config


# ─────────────────────────────────────────────
# Hauptmenü
# ─────────────────────────────────────────────

def main_menu():
    while True:
        config = load_config() # Bei jeder Schleife frisch einlesen

        header("Hauptmenü")
        print("  [1]  🔄  Ingestion Pipeline ausführen")
        print("         (PDFs/TXTs einlesen → ChromaDB aufbauen)")
        print()
        print("  [2]  💬  Slop Creation Pipeline ausführen")
        print("         (RAG-Agent starten → Fragen stellen)")
        print()
        print("  [4]  📊  Batch-Evaluation ausführen")
        print("         (Fragen aus TXT lesen → CSV-Auswertung erstellen)")
        print()
        dividing_line()
        print("  [3]  ⚙   Einstellungen")
        print("  [0]  ✗   Beenden")
        dividing_line()

        print(f"\n  Aktiv: Modell={config['ollama_model']} | "
              f"Chunking={config['chunking_mode']} | "
              f"k={config['retriever_k']} | "
              f"DB={config['datenbank_ordner']}")

        selection = input("\n  Deine Wahl: ").strip()

        if selection == "0":
            clear()
            print("  Tschüss!\n")
            break

        elif selection == "1":
            header("🔄 Ingestion Pipeline")
            print(f"  Dokument-Ordner : {config['dok_ordner']}")
            print(f"  Datenbank-Ordner: {config['datenbank_ordner']}")
            print(f"  Embedding-Modell: {config['embedding_model']}")
            print(f"  Chunking-Modus  : {config['chunking_mode']}")
            if config['chunking_mode'] == 'fixed':
                print(f"  Token-Limit     : {config['token_limit']}")
                print(f"  Chunk-Überlapp  : {config['chunk_overlap']}")
            print()
            if confirmation("Pipeline jetzt starten?"):
                print()
                run_script(INGESTION_SCRIPT)
                wait()

        elif selection == "2":
            header("💬 Slop Creation Pipeline")
            print(f"  LLM-Modell      : {config['ollama_model']}")
            print(f"  Temperatur      : {config['llm_temperature']}")
            print(f"  Retriever k     : {config['retriever_k']}")
            print(f"  Max. Loops      : {config['max_rewrite_loops']}")
            print(f"  Datenbank-Ordner: {config['datenbank_ordner']}")
            print()
            if confirmation("Agent jetzt starten?"):
                print()
                run_script(SLOP_SCRIPT)
                wait()

        elif selection == "3":
            config = settings_menu(config)

        elif selection == "4":
            header("📊 Batch-Evaluation")
            input_pfad = prompt_input("Pfad zur TXT-Datei mit Fragen", "fragen.txt")
            output_pfad = prompt_input("Pfad zur CSV-Ausgabedatei", "eval_ergebnisse.csv")
            
            if not os.path.exists(input_pfad):
                print(f"\n  [!] Fehler: Eingabedatei '{input_pfad}' existiert nicht.")
                wait()
                continue
                
            print()
            if confirmation("Batch-Evaluation jetzt starten?"):
                print()
                run_script(SLOP_SCRIPT, ["--batch_input", input_pfad, "--batch_output", output_pfad])
                wait()

        else:
            print("  [!] Ungültige Eingabe.")
            wait()


if __name__ == "__main__":
    main_menu()