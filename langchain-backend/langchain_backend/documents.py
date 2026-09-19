from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from .settings import settings

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def load_pdf_documents(root: Path | None = None) -> list[Document]:
    corpus_root = (root or settings.corpus_dir).resolve()
    documents: list[Document] = []

    if not corpus_root.exists():
        return documents

    for file_path in sorted(corpus_root.rglob("*")):
        if file_path.suffix.lower() != ".pdf" or not file_path.is_file():
            continue
        # A linked file outside the corpus must not become an ingestion source.
        if not file_path.resolve().is_relative_to(corpus_root):
            continue
        loader = PyPDFLoader(str(file_path))
        pdf_documents = loader.load()
        for pdf_document in pdf_documents:
            pdf_document.metadata["path"] = str(file_path)
            pdf_document.metadata["source"] = file_path.relative_to(corpus_root).as_posix()
            pdf_document.metadata["extension"] = ".pdf"
        documents.extend(pdf_documents)

    return documents
