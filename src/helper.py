# src/helper.py
import re
from typing import Iterable, List, Optional
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------
# Loading + Cleaning
# ---------------------------
def load_pdf_file(data_folder: str) -> List[Document]:
    """
    Load all PDFs from a folder into LangChain Documents.
    """
    loader = DirectoryLoader(data_folder, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def _clean_text(text: str) -> str:
    # Remove common noise; tune as needed for your PDF format
    text = re.sub(r"\bPage\s*\d+\b", "", text)          # page numbers
    text = re.sub(r"References?:.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\n{2,}", "\n", text)                # collapse blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)              # collapse spaces
    return text.strip()

def clean_documents(docs: List[Document]) -> List[Document]:
    for d in docs:
        d.page_content = _clean_text(d.page_content)
    return docs

# ---------------------------
# Semantic-aware splitting
# ---------------------------
def split_documents_semantic(
    docs: List[Document],
    min_chunk_size: int = 400,
    max_chunk_size: int = 800,
    chunk_overlap: int = 60,
) -> List[Document]:
    """
    Paragraph-first, then sentence-level splitting with overlap.
    Uses a RecursiveCharacterTextSplitter with custom separators.
    Returns LangChain Documents (not raw strings).
    """
    splitter = RecursiveCharacterTextSplitter(
        # Try to break on section headers / paragraphs / sentences
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    # First pass: create documents by structure
    chunks: List[Document] = splitter.split_documents(docs)

    # Second pass (lightweight): ensure chunks aren’t too tiny
    refined: List[Document] = []
    buffer_text, buffer_meta = "", None

    def flush_buffer():
        nonlocal buffer_text, buffer_meta
        if buffer_text.strip():
            refined.append(Document(page_content=buffer_text.strip(), metadata=buffer_meta or {}))
        buffer_text, buffer_meta = "", None

    for ch in chunks:
        txt = ch.page_content.strip()
        if not txt:
            continue
        if len(buffer_text) == 0:
            buffer_text = txt
            buffer_meta = ch.metadata
        elif len(buffer_text) + 1 + len(txt) <= max_chunk_size:
            buffer_text = f"{buffer_text} {txt}"
        else:
            if len(buffer_text) >= min_chunk_size:
                flush_buffer()
                buffer_text, buffer_meta = txt, ch.metadata
            else:
                # If buffer still small, try to append more before flushing
                buffer_text = f"{buffer_text} {txt}"
                if len(buffer_text) >= min_chunk_size:
                    flush_buffer()
    flush_buffer()
    return refined

# ---------------------------
# Embeddings
# ---------------------------
def download_huggingface_embeddings(model_name: str = "dmis-lab/biobert-base-cased-v1.1"):
    """
    Default to BioBERT (768 dims). You can switch to MiniLM for speed:
    - "sentence-transformers/all-MiniLM-L6-v2" (384 dims)
    """
    return HuggingFaceEmbeddings(model_name=model_name)
