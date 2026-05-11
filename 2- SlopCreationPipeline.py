import os
from typing import List, TypedDict
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

# ==========================================
# 1. SETUP: MODELLE & DATENBANK
# ==========================================
print("\n[SYSTEM] Lade Modelle und Vektordatenbank...")

# Das LLM aus Ollama (Temperatur 0 = sehr sachlich, keine Halluzinationen)
llm = ChatOllama(model="gemma4", temperature=0)

# Vektordatenbank laden
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 

# ==========================================
# 2. DER ZUSTAND (State)
# ==========================================
# Das ist das "Gedächtnis" des Graphen. Diese Variablen werden von Knoten zu Knoten weitergereicht.
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    loop_count: int # Zählt die Anzahl der Umformulierungen

# ==========================================
# 3. DIE KNOTEN (Nodes)
# ==========================================

def retrieve(state: GraphState):
    print("\n-> [KNOTEN: RETRIEVE] Suche in der Vektordatenbank...")
    question = state["question"]
    documents = retriever.invoke(question)
    
    # NEU: Detaillierte Ausgabe, was überhaupt gefunden wurde
    print(f"   [i] {len(documents)} Chunks initial gefunden. Vorschau:")
    for i, d in enumerate(documents):
        snippet = d.page_content.replace('\n', ' ')#[:80] # Vorschau der ersten 80 Zeichen
        print(f"       - Chunk {i+1}: '{snippet}...'")
        
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState):
    # Prüft, ob die gefundenen Dokumente die Frage beantworten können.
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
        
        # NEU: Detaillierte Ausgabe der Entscheidung
        if "ja" in bewertung.lower():
            print(f"   [+] BEHALTEN (Relevant): '{snippet}...'")
            gefilterte_docs.append(d)
        else:
            print(f"   [-] VERWORFEN (Irrelevant): '{snippet}...'")
            
    return {"documents": gefilterte_docs, "question": question}

def transform_query(state: GraphState):
    # Formuliert die Frage um, falls keine guten Dokumente gefunden wurden.
    print("-> [KNOTEN: REWRITER] Suche erfolglos. Formuliere die Frage um...")
    question = state["question"]
    loop_count = state.get("loop_count", 0) + 1 # NEU: Zähler erhöhen
    
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
    print(f"   [!] Neue Suchanfrage (Versuch {loop_count}): {bessere_frage.strip()}")
    
    return {"documents": state["documents"], "question": bessere_frage, "loop_count": loop_count}

def generate(state: GraphState):
    print("-> [KNOTEN: GENERATOR] Schreibe die finale Antwort...")
    question = state["question"]
    documents = state["documents"]
    
     # Fasse die Texte der Dokumente zusammen
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
    """NEU: Fängt den Fall ab, wenn 10 Mal erfolglos gesucht wurde."""
    print("-> [KNOTEN: ABBRUCH] Limit erreicht. Gebe auf...")
    return {"generation": "Ich konnte leider auch nach 10 Suchläufen und Umformulierungen keine relevanten Informationen in den bereitgestellten PDFs finden."}

# ==========================================
# 4. DIE LOGIK (Edges)
# ==========================================
def decide_to_generate(state: GraphState):
    # Entscheidet nach der Bewertung, wie es weitergeht.
    print("-> [LOGIK: ENTSCHEIDUNG] Prüfe, ob genügend relevanter Kontext vorliegt...")
    gefilterte_docs = state["documents"]
    loop_count = state.get("loop_count", 0)
    
    if not gefilterte_docs:
        if loop_count >= 10: # NEU: Abbruchbedingung
            print("   [!] 10 erfolglose Versuche erreicht. Breche Suche ab.")
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

# Knoten hinzufügen
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate)
workflow.add_node("abort_search", abort_search) # NEU

# Ablauf definieren
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Hier kommt die Magie: Eine bedingte Abzweigung!
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query", # Wenn keine guten Docs -> umschreiben
        "generate": "generate",               # Wenn gute Docs -> generieren
        "abort_search": "abort_search",       # Abbruch wenn es zehnmal nix findet
    }
)
workflow.add_edge("transform_query", "retrieve") # Nach dem Umschreiben wieder suchen
workflow.add_edge("generate", END) 
workflow.add_edge("abort_search", END) # Nach der Antwort oder dem Abbruch ist Schluss

app = workflow.compile()

# ==========================================
# 6. INTERAKTIVE CHAT-SCHLEIFE (REPL)
# ==========================================
print("\n" + "="*50)
print("🤖 AGENT BEREIT! (Tippe 'exit' oder 'quit' zum Beenden)")
print("="*50)

while True:
    # NEU: Terminal-Abfrage während der Laufzeit
    user_input = input("\nDeine Frage an die PDFs: ")
    
    if user_input.lower() in ['exit', 'quit']:
        print("Agent wird beendet. Bis bald!")
        break
    
    if not user_input.strip():
        continue

    # Initialisiere den State mit der Frage und loop_count = 0
    inputs = {"question": user_input, "loop_count": 0}

    # Graph ausführen
    final_state = None
    for output in app.stream(inputs):
        # Durchläuft alle Schritte und gibt das Terminal-Feedback aus den Knoten
        final_state = output
        
    # Ergebnis abgreifen (der Schlüssel im Output ändert sich je nach letztem Knoten, 
    # daher holen wir uns dynamisch den Inhalt)
    knoten_name = list(final_state.keys())[0]
    finale_antwort = final_state[knoten_name]["generation"]

    print("\n" + "="*50)
    print("🤖 ANTWORT:")
    print("="*50)
    print(finale_antwort)