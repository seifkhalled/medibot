# store_index.py
import os
from dotenv import load_dotenv

# Choose your vector backend here:
USE_PINECONE = True   # set to False to use Chroma on-disk

from src.helper import (
    load_pdf_file,
    clean_documents,
    split_documents_semantic,
    download_huggingface_embeddings,
)

# ---------- Load env ----------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

DATA_DIR = "Data"
DB_DIR = "chroma_db"            # used if USE_PINECONE=False
INDEX_NAME = "medicalbot-minilm"  # lowercase, dashes only (Pinecone rule)

# ---------- Load + preprocess ----------
print("📂 Loading PDFs...")
docs = load_pdf_file(DATA_DIR)
print(f"Loaded {len(docs)} documents")

print("🧹 Cleaning text...")
docs = clean_documents(docs)

print("✂️ Semantic splitting...")
chunks = split_documents_semantic(
    docs,
    min_chunk_size=400,
    max_chunk_size=800,
    chunk_overlap=60,
)
print(f"✅ Created {len(chunks)} chunks")

# ---------- Embeddings ----------
print("🧠 Loading embeddings (MiniLM - fast & lightweight)...")
# Model: ~22MB, 384 dims, uses safetensors (no torch vulnerability)
embeddings = download_huggingface_embeddings("sentence-transformers/all-MiniLM-L6-v2")

# ---------- Vector store ----------
if USE_PINECONE:
    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY. Set it in .env or switch USE_PINECONE=False.")
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if missing
    existing = [idx["name"] for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print("🔧 Creating Pinecone index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # MiniLM has 384-d embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="disabled",
        )

    print("☁️ Indexing chunks to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    print(f"✅ Done. Pinecone index: {INDEX_NAME}")
else:
    # Local fallback: Chroma
    from langchain_community.vectorstores import Chroma
    print("💾 Using local Chroma DB...")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    vs.persist()
    print(f"✅ Chroma DB persisted at: {DB_DIR}")
