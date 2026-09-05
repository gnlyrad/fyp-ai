from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS


def build_vector_store(documents, embeddings, index_dir: Path) -> FAISS:
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(str(index_dir / "faiss"))
    return vector_store


def load_vector_store(embeddings, index_dir: Path) -> FAISS:
    faiss_dir = index_dir / "faiss"
    if not faiss_dir.exists():
        raise FileNotFoundError(f"Vector index not found at {faiss_dir}")
    return FAISS.load_local(
        str(faiss_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
