import os
import re
import json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
import shutil

# ==========================================
# 1. KONFIGURATION & VARIABLEN (Aus JSON oder Defaults)
# ==========================================
CONFIG_FILE = "rag_config.json"

# Standard-Fallbacks, falls keine JSON existiert
DOK_ORDNER = "./meine_pdfs"
DATENBANK_ORDNER = "./chroma_db"
TOKEN_LIMIT = 400 
CHUNK_OVERLAP = 200
CHUNKING_MODE = "semantic"  # "semantic" oder "fixed"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Versuche Einstellungen aus JSON zu laden
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            DOK_ORDNER = config.get("dok_ordner", DOK_ORDNER)
            DATENBANK_ORDNER = config.get("datenbank_ordner", DATENBANK_ORDNER)
            TOKEN_LIMIT = int(config.get("token_limit", TOKEN_LIMIT))
            CHUNK_OVERLAP = int(config.get("chunk_overlap", CHUNK_OVERLAP))
            CHUNKING_MODE = config.get("chunking_mode", CHUNKING_MODE)
            EMBEDDING_MODEL = config.get("embedding_model", EMBEDDING_MODEL)
            print(f"[INFO] Konfiguration erfolgreich aus '{CONFIG_FILE}' geladen.")
    except Exception as e:
        print(f"[WARNUNG] Fehler beim Lesen der '{CONFIG_FILE}'. Nutze Standardwerte. Fehler: {e}")
else:
    print(f"[INFO] Keine '{CONFIG_FILE}' gefunden. Nutze eingebaute Standardwerte.")

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
            encodings_to_try = ["utf-8", "cp1252", "latin-1", "utf-16"]
            geladene_docs = None
            for enc in encodings_to_try:
                try:
                    loader = TextLoader(dateipfad, encoding=enc)
                    geladene_docs = loader.load()
                    break
                except (UnicodeDecodeError, RuntimeError):
                    continue
            if geladene_docs is None:
                print(f"   [!] Konnte '{dateiname}' mit keiner bekannten Kodierung lesen. Überspringe.")
                continue

    geladene_docs = loader.load()
    dokumente.extend(geladene_docs)
    print(f"   [OK] {dateiname}: {len(geladene_docs)} Seite(n)/Abschnitt(e) geladen.")

print(f"\n   [OK] Gesamt: {len(dokumente)} Seiten/Dokumente erfolgreich geladen.")

# ==========================================
# 3. NORMALISIEREN (Cleaning)
# ==========================================
print("-> Normalisiere den Text (entferne doppelte Leerzeichen, Zeilenumbrüche etc.)...")
def normalisiere_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

for doc in dokumente:
    doc.page_content = normalisiere_text(doc.page_content)
print("   [OK] Text normalisiert.")

# ==========================================
# 4. VEKTORISIEREN VORBEREITEN (Embedding Model)
# ==========================================
print(f"-> Lade lokales Embedding-Modell (SBERT: {EMBEDDING_MODEL})...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL) 
print("   [OK] Modell geladen.")

# ==========================================
# 5. CHUNKEN (Splitting)
# ==========================================
print("-> Beginne mit dem Chunking...")

if CHUNKING_MODE == "semantic":
    print("   [Info] Nutze Semantisches Chunking (trennt nach Sinnabschnitten).")
    text_splitter = SemanticChunker(embeddings, 
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=0.75 )
else:
    print(f"   [Info] Nutze fixes Limit (~{TOKEN_LIMIT} Tokens / {ZEICHEN_LIMIT} Zeichen pro Chunk).")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ZEICHEN_LIMIT,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

chunks = text_splitter.split_documents(dokumente)
print(f"   [OK] Dokumente wurden in {len(chunks)} Chunks unterteilt.")

# ==========================================
# 6. VEKTOR-DATENBANK SPEICHERN (ChromaDB)
# ==========================================
print(f"-> Vektorisiere Chunks und speichere sie in ChromaDB unter '{DATENBANK_ORDNER}'...")
if os.path.exists(DATENBANK_ORDNER):
    shutil.rmtree(DATENBANK_ORDNER)
vektor_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DATENBANK_ORDNER
)
print(f"   [OK] Datenbank erfolgreich im Ordner '{DATENBANK_ORDNER}' gespeichert.")
print("--- [FERTIG] Die Pipeline lief erfolgreich durch! ---\n")