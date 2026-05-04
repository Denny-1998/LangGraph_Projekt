import os
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma


# ==========================================
# 1. KONFIGURATION & VARIABLEN
# ==========================================
DOK_ORDNER = "./meine_pdfs"
DATENBANK_ORDNER = "./chroma_db"

TOKEN_LIMIT = 400 
ZEICHEN_LIMIT = TOKEN_LIMIT * 4 

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

print("\n--- [START] Lokale RAG Ingestion Pipeline ---")

# ==========================================
# 2. EINLESEN (Loading)
# ==========================================
print(f"-> Lese Dokumente aus dem Ordner '{DOK_ORDNER}' ein...")
if not os.path.exists(DOK_ORDNER):
    os.makedirs(DOK_ORDNER)
    print(f"   [!] Ordner '{DOK_ORDNER}' existierte nicht und wurde erstellt. Bitte Dokumente einfügen und neu starten.")
    exit()

dokumente = []

for dateiname in os.listdir(DOK_ORDNER):
    dateipfad = os.path.join(DOK_ORDNER, dateiname)

    if not os.path.isfile(dateipfad):
        continue

    endung = os.path.splitext(dateiname)[1].lower()

    if endung not in SUPPORTED_EXTENSIONS:
        print(f"   [~] Überspringe nicht unterstützte Datei: {dateiname}")
        continue

    match endung:
        case ".pdf":
            loader = PyPDFLoader(dateipfad)
        case ".txt":
            loader = TextLoader(dateipfad, encoding="utf-8")

    geladene_docs = loader.load()
    dokumente.extend(geladene_docs)
    print(f"   [OK] {dateiname}: {len(geladene_docs)} Seite(n)/Abschnitt(e) geladen.")

print(f"\n   [OK] Gesamt: {len(dokumente)} Seiten/Dokumente erfolgreich geladen.")




# ==========================================
# 3. NORMALISIEREN (Cleaning)
# ==========================================
print("-> Normalisiere den Text (entferne doppelte Leerzeichen, Zeilenumbrüche etc.)...")
def normalisiere_text(text):
    text = re.sub(r'\s+', ' ', text) # Macht aus mehrfachen Leerzeichen/Umbrüchen ein einzelnes Leerzeichen
    text = text.strip()
    return text

for doc in dokumente:
    doc.page_content = normalisiere_text(doc.page_content)
print("   [OK] Text normalisiert.")

# ==========================================
# 4. VEKTORISIEREN VORBEREITEN (Embedding Model)
# ==========================================
# Wir laden das Embedding-Modell jetzt schon, da das Semantische Chunking es ebenfalls benötigt.
print("-> Lade lokales Embedding-Modell (SBERT: all-MiniLM-L6-v2)...")
# all-MiniLM-L6-v2 ist der absolute Goldstandard für schnelle, lokale SBERT Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
print("   [OK] Modell geladen.")

# ==========================================
# 5. CHUNKEN (Splitting)
# ==========================================
print("-> Beginne mit dem Chunking...")

# OPTION A: Fixes Limit (basiert auf der TOKEN_LIMIT Variable)
# Kommentiere diesen Block ein, wenn du das fixe Limit nutzen willst:
print(f"   [Info] Nutze fixes Limit (~{TOKEN_LIMIT} Tokens / {ZEICHEN_LIMIT} Zeichen pro Chunk).")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=ZEICHEN_LIMIT,
    chunk_overlap=200, # Überlappung, damit keine Sätze mittendrin den Kontext verlieren
    separators=["\n\n", "\n", ".", " ", ""]
)

# OPTION B: Semantisches Chunking
# Kommentiere den TextSplitter oben aus und diesen hier ein, für semantisches Chunking:
# print("   [Info] Nutze Semantisches Chunking (trennt nach Sinnabschnitten).")
# text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

chunks = text_splitter.split_documents(dokumente)
print(f"   [OK] Dokumente wurden in {len(chunks)} Chunks unterteilt.")

# ==========================================
# 6. VEKTOR-DATENBANK SPEICHERN (ChromaDB)
# ==========================================
print("-> Vektorisiere Chunks und speichere sie in ChromaDB...")
vektor_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DATENBANK_ORDNER
)
print(f"   [OK] Datenbank erfolgreich im Ordner '{DATENBANK_ORDNER}' gespeichert.")
print("--- [FERTIG] Die Pipeline lief erfolgreich durch! ---\n")