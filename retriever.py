"""
Advanced hybrid retriever: Vector (Chroma) + BM25 -> Ensemble -> FlashRank rerank.

Unchanged in spirit from the original app.py, just isolated into its own
module so app.py stays focused on UI/orchestration.
"""
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document

PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def setup_advanced_retriever(vector_weight: float = 0.6, bm25_weight: float = 0.4, fetch_k: int = 10):
    # A. Vector retriever (semantic)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": fetch_k})

    # B. BM25 retriever (keyword)
    db_data = vector_db.get(include=["documents", "metadatas"])
    if not db_data or not db_data.get("documents"):
        # إذا كانت قاعدة البيانات فارغة على السيرفر، ننشئ مستنداً وهمياً مؤقتاً لتجنب خطأ الـ zip
        bm25_docs = [
            Document(
                page_content="قاعدة البيانات فارغة حالياً. يرجى تهيئة البيانات أو رفع المستندات.", 
                metadata={"source": "system"}
            )
        ]
    else:
        bm25_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(db_data["documents"], db_data["metadatas"])
        ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = fetch_k

    # C. Hybrid ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )

    # D. Re-ranker (FlashRank -- fast, local, free)
    compressor = FlashrankRerank()

    # E. Final: hybrid -> rerank -> top-N
    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )
    return advanced_retriever
