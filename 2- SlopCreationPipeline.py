import os
import json
import time
import csv
import argparse
from typing import List, TypedDict
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

# ==========================================
# 1. SETUP: CONFIGURATION, MODELLE & DATENBANK
# ==========================================
CONFIG_FILE = "rag_config.json"

# Standard-Fallbacks
DATENBANK_ORDNER = "./chroma_db"
OLLAMA_MODEL = "gemma4:e4b"
LLM_TEMPERATURE = 0.0
RETRIEVER_K = 3
MAX_REWRITE_LOOPS = 10
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
            DATENBANK_ORDNER = config.get("datenbank_ordner", DATENBANK_ORDNER)
            OLLAMA_MODEL = config.get("ollama_model", OLLAMA_MODEL)
            LLM_TEMPERATURE = float(config.get("llm_temperature", LLM_TEMPERATURE))
            RETRIEVER_K = int(config.get("retriever_k", RETRIEVER_K))
            MAX_REWRITE_LOOPS = int(config.get("max_rewrite_loops", MAX_REWRITE_LOOPS))
            EMBEDDING_MODEL = config.get("embedding_model", EMBEDDING_MODEL)
            print(f"[INFO] Konfiguration erfolgreich aus '{CONFIG_FILE}' geladen.")
    except Exception as e:
        print(f"[WARNUNG] Fehler beim Lesen der '{CONFIG_FILE}'. Nutze Standardwerte. Fehler: {e}")
else:
    print(f"[INFO] Keine '{CONFIG_FILE}' gefunden. Nutze eingebaute Standardwerte.")

print("\n[SYSTEM] Lade Modelle und Vektordatenbank...")

# Das LLM aus Ollama
llm = ChatOllama(model=OLLAMA_MODEL, temperature=LLM_TEMPERATURE)

# Vektordatenbank laden
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=DATENBANK_ORDNER, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K}) 

# ==========================================
# 2. DER ZUSTAND (State)
# ==========================================
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    loop_count: int

# ==========================================
# 3. DIE KNOTEN (Nodes)
# ==========================================
def retrieve(state: GraphState):
    print("\n-> [KNOTEN: RETRIEVE] Suche in der Vektordatenbank...")
    question = state["question"]
    documents = retriever.invoke(question)
    
    print(f"   [i] {len(documents)} Chunks initial gefunden. Vorschau:")
    for i, d in enumerate(documents):
        snippet = d.page_content.replace('\n', ' ')[:80]
        print(f"       - Chunk {i+1}: '{snippet}...'")
        
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState):
    print("-> [KNOTEN: GRADER] Bewerte die Relevanz der gefundenen Dokumente...")
    question = state["question"]
    documents = state["documents"]
    
    prompt = PromptTemplate(
        template="""Du bist ein strenger Bewerter. Prüfe, ob das Dokument für die Frage relevant ist.
        Antworte NUR mit 'ja' oder 'nein'. Keine weiteren Erklärungen.
        Frage: {question}
        Dokument: {document}
        Antwort:""",
        input_variables=["question", "document"],
    )
    grader_chain = prompt | llm | StrOutputParser()
    
    gefilterte_docs = []
    for d in documents:
        bewertung = grader_chain.invoke({"question": question, "document": d.page_content})
        snippet = d.page_content.replace('\n', ' ')[:80]
        
        if "ja" in bewertung.lower():
            print(f"   [+] BEHALTEN (Relevant): '{snippet}...'")
            gefilterte_docs.append(d)
        else:
            print(f"   [-] VERWORFEN (Irrelevant): '{snippet}...'")
            
    return {"documents": gefilterte_docs, "question": question}

def transform_query(state: GraphState):
    print("-> [KNOTEN: REWRITER] Suche erfolglos. Formuliere die Frage um...")
    question = state["question"]
    loop_count = state.get("loop_count", 0) + 1
    
    prompt = PromptTemplate(
        template="""Du bist ein Experte im Formulieren von Suchanfragen. 
        Die folgende Frage hat keine guten Ergebnisse geliefert. Formuliere sie um, um bessere Treffer in einer Vektordatenbank zu erzielen.
        Gib NUR die neue Frage aus.
        Ursprüngliche Frage: {question}
        Neue Frage:""",
        input_variables=["question"],
    )
    rewriter_chain = prompt | llm | StrOutputParser()
    bessere_frage = rewriter_chain.invoke({"question": question})
    print("\n" + "="*50) 
    print(f"   ❔ Neue Suchanfrage (Versuch {loop_count}): {bessere_frage.strip()}")
    print("="*50)
    return {"documents": state["documents"], "question": bessere_frage, "loop_count": loop_count}

def generate(state: GraphState):
    print("-> [KNOTEN: GENERATOR] Schreibe die finale Antwort...")
    question = state["question"]
    documents = state["documents"]
    
    kontext = "\n\n".join([d.page_content for d in documents])
    
    prompt = PromptTemplate(
        template="""Du bist ein hilfreicher Assistent. Beantworte die Frage NUR basierend auf dem folgenden Kontext. 
        Wenn du die Antwort im Kontext nicht findest, sage, dass du es nicht weißt.
        Kontext: {context}
        Frage: {question}
        Antwort:""",
        input_variables=["context", "question"],
    )
    rag_chain = prompt | llm | StrOutputParser()
    antwort = rag_chain.invoke({"context": kontext, "question": question})
    
    return {"documents": documents, "question": question, "generation": antwort}

def abort_search(state: GraphState):
    print("-> [KNOTEN: ABBRUCH] Limit erreicht. Gebe auf...")
    return {"generation": f"Ich konnte leider auch nach {MAX_REWRITE_LOOPS} Suchläufen und Umformulierungen keine relevanten Informationen in den bereitgestellten PDFs finden."}

# ==========================================
# 4. DIE LOGIK (Edges)
# ==========================================
def decide_to_generate(state: GraphState):
    print("-> [LOGIK: ENTSCHEIDUNG] Prüfe, ob genügend relevanter Kontext vorliegt...")
    gefilterte_docs = state["documents"]
    loop_count = state.get("loop_count", 0)
    
    if not gefilterte_docs:
        if loop_count >= MAX_REWRITE_LOOPS:
            print(f"   [!] {MAX_REWRITE_LOOPS} erfolglose Versuche erreicht. Breche Suche ab.")
            return "abort_search"
        print("   [!] Keine relevanten Dokumente übrig. Gehe zum Rewriter.")
        return "transform_query"
    else:
        print("   [!] Genug Kontext vorhanden. Gehe zum Generator.")
        return "generate"

# ==========================================
# 5. GRAPH BAUEN UND KOMPILIEREN
# ==========================================
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate)
workflow.add_node("abort_search", abort_search)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
        "abort_search": "abort_search",
    }
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END) 
workflow.add_edge("abort_search", END)

app = workflow.compile()

# ==========================================
# 6. INTERAKTIVE CHAT-SCHLEIFE (REPL) & BATCH
# ==========================================

def run_interactive_mode():
    print("\n" + "="*50)
    print(f"🤖 AGENT BEREIT! Modell: {OLLAMA_MODEL} (Tippe 'exit' oder 'quit' zum Beenden)")
    print("="*50)

    while True:
        user_input = input("\nDeine Frage an die PDFs: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Agent wird beendet. Bis bald!")
            break
        
        if not user_input.strip():
            continue

        inputs = {"question": user_input, "loop_count": 0}

        final_state = None
        for output in app.stream(inputs):
            final_state = output
            
        knoten_name = list(final_state.keys())[0]
        finale_antwort = final_state[knoten_name]["generation"]

        print("\n" + "="*50)
        print("🤖 ANTWORT:")
        print("="*50)
        print(finale_antwort)


def run_batch_mode(input_pfad: str, output_pfad: str, start_index: int = 1):
    if not os.path.exists(input_pfad):
        print(f"  [!] Fehler: Eingabedatei '{input_pfad}' existiert nicht.")
        return

    if not output_pfad:
        output_pfad = "eval_ergebnisse.csv"

    print(f"\n[BATCH] Lese Fragen aus '{input_pfad}'...")
    with open(input_pfad, "r", encoding="utf-8") as f:
        fragen = [line.strip() for line in f if line.strip()]

    gesamt_anzahl = len(fragen)
    print(f"[BATCH] Gefunden: {gesamt_anzahl} Fragen.")

    if start_index < 1 or start_index > gesamt_anzahl:
        print(f"  [!] Fehler: Start-Index {start_index} ist ungültig (muss zwischen 1 und {gesamt_anzahl} liegen).")
        return

    # Bestimme Schreibmodus (neu schreiben oder anhängen)
    schreib_modus = "w"
    write_header = True
    
    # Wenn start_index > 1, hängen wir an, vorausgesetzt die Datei existiert bereits und ist nicht leer
    if start_index > 1 and os.path.exists(output_pfad) and os.path.getsize(output_pfad) > 0:
        schreib_modus = "a"
        write_header = False
        print(f"[BATCH] Setze fort ab Frage {start_index}. Ergebnisse werden an '{output_pfad}' angehängt.")
    else:
        print(f"[BATCH] Starte neu ab Frage {start_index}. Ergebnisse werden in '{output_pfad}' gespeichert.")

    # CSV initialisieren (falls write_header, schreiben wir den Header direkt am Anfang)
    fieldnames = ["Question", "Response", "Execution Time (s)", "Source Chunks", "Rewrite Iterations"]
    if write_header:
        try:
            with open(output_pfad, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            print(f"[BATCH] CSV-Datei '{output_pfad}' initialisiert mit Header.")
        except Exception as e:
            print(f"[BATCH] [!] Fehler beim Initialisieren der CSV-Datei: {e}")
            return

    # Nur die Fragen ab dem Start-Index verarbeiten
    verbleibende_fragen = fragen[start_index - 1:]

    for relative_idx, frage in enumerate(verbleibende_fragen):
        aktuelle_frage_nummer = start_index + relative_idx
        print(f"\n" + "="*50)
        print(f"  [BATCH] [{aktuelle_frage_nummer}/{gesamt_anzahl}] Verarbeite Frage: '{frage}'")
        print("="*50)
        
        start_time = time.perf_counter()
        
        inputs = {"question": frage, "loop_count": 0}
        
        loop_count = 0
        retrieved_documents = []
        final_answer = ""
        
        for output in app.stream(inputs):
            node_name = list(output.keys())[0]
            node_output = output[node_name]
            
            if "loop_count" in node_output:
                loop_count = node_output["loop_count"]
            if "documents" in node_output:
                retrieved_documents = node_output["documents"]
            if "generation" in node_output:
                final_answer = node_output["generation"]
                
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        
        # Formatieren der Source Chunks
        chunks_text = []
        for c_idx, doc in enumerate(retrieved_documents, 1):
            source = doc.metadata.get("source", "Unbekannte Quelle")
            chunks_text.append(f"[Chunk {c_idx} (Quelle: {os.path.basename(source)})]:\n{doc.page_content}")
            
        chunks_formatted = "\n---\n".join(chunks_text) if chunks_text else "Keine Chunks verwendet"
        
        print(f"\n  [✔] Fertig in {elapsed_seconds:.3f} Sekunden.")
        print(f"  [✔] Rewriter-Schleifen: {loop_count}")
        
        # Zeile direkt in CSV schreiben
        row_data = {
            "Question": frage,
            "Response": final_answer,
            "Execution Time (s)": f"{elapsed_seconds:.3f}",
            "Source Chunks": chunks_formatted,
            "Rewrite Iterations": loop_count
        }
        
        try:
            with open(output_pfad, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row_data)
            print(f"  [✔] Zeile in CSV geschrieben.")
        except Exception as e:
            print(f"  [!] Fehler beim Schreiben der Zeile in CSV: {e}")

    print(f"\n[BATCH] [FERTIG] Alle verbleibenden Fragen verarbeitet. Ergebnisse in: {os.path.abspath(output_pfad)}")


def main():
    parser = argparse.ArgumentParser(description="LangGraph Corrective RAG Pipeline")
    parser.add_argument("--batch_input", type=str, default=None, help="Pfad zur TXT-Datei mit einer Frage pro Zeile")
    parser.add_argument("--batch_output", type=str, default=None, help="Pfad zur Ausgabedatei (.csv)")
    parser.add_argument("--start_index", type=int, default=1, help="Start-Index (1-basiert) für die Batch-Verarbeitung")
    
    args = parser.parse_args()
    
    if args.batch_input:
        run_batch_mode(args.batch_input, args.batch_output, args.start_index)
    else:
        run_interactive_mode()


if __name__ == "__main__":
    main()