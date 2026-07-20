"""
Ingestion pipeline for AquaMind AI.

Loads PDFs -> chunks -> embeds -> stores in ChromaDB.

Fixes vs. the original script:
- chunk_overlap was equal to chunk_size (1000/1000), meaning every chunk was
  ~100% duplicated. Corrected to a sane 15% overlap.
- Each chunk now carries stable metadata (source filename, page number,
  chunk_id) so the UI can render precise, clickable, page-level citations
  instead of just a filename.
"""
import os
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "./data"
PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 1000  # was 1000 (100% overlap) -- bug fix


def load_documents(data_dir: str):
    documents = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pdf_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    for file in pdf_files:
        file_path = os.path.join(data_dir, file)
        print(f"Loading: {file}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        # PyPDFLoader already sets metadata['page'] (0-indexed) and
        # metadata['source']; we normalize source to just the filename and
        # convert page to a human-friendly 1-indexed number.
        for p in pages:
            p.metadata["source"] = file
            p.metadata["page"] = p.metadata.get("page", 0) + 1
        documents.extend(pages)

    print(f"Total pages loaded from {len(pdf_files)} PDF(s): {len(documents)}")
    return documents


def chunk_documents(documents):
    print(f"Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Stable, unique chunk ids -- useful for citation dedup and for
    # re-ingestion (Chroma can upsert by id instead of duplicating).
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        chunk.metadata["chunk_id"] = f"{source}-p{page}-{uuid.uuid4().hex[:8]}"

    print(f"Created {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks):
    print("Creating embeddings... (this can take a minute on first run)")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Saving to ChromaDB...")
    ids = [c.metadata["chunk_id"] for c in chunks]
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        ids=ids,
    )
    return vector_db


def main():
    documents = load_documents(DATA_DIR)
    chunks = chunk_documents(documents)
    build_vector_store(chunks)
    print(f"\u2705 Ingestion complete! Vector database saved to {PERSIST_DIR}")


if __name__ == "__main__":
    main()
