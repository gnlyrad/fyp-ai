from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from .settings import settings

def load_pdf_documents(root: Path | None = None) -> list[Document]:
    corpus_root = root or settings.corpus_dir
    documents: list[Document] = []

    if not corpus_root.exists():
        return documents

    for file_path in sorted(corpus_root.rglob("*.pdf")):
        if not file_path.is_file():
            continue
        loader = PyPDFLoader(str(file_path))
        pdf_documents = loader.load()
        for pdf_document in pdf_documents:
            pdf_document.metadata["path"] = str(file_path)
            pdf_document.metadata["source"] = str(file_path.relative_to(corpus_root))
            pdf_document.metadata["extension"] = ".pdf"
        documents.extend(pdf_documents)

    return documents
