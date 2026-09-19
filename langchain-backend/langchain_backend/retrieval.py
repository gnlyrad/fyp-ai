"""Dense search over the MAS compliance corpus.

Split out from `rag.py` because the two callers want different things. `/ask`
wants a synthesised answer and needs a chat model; the compliance pipeline wants
the passages themselves and does not. Constructing a Bedrock chat model just to
run a similarity search is wasted setup, so `RegulationSearch` holds only the
embeddings and the index, and `CodeRAGService` composes it.
"""
from __future__ import annotations

from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from .documents import CHUNK_OVERLAP, CHUNK_SIZE, load_pdf_documents
from .settings import settings
from .vectorstore import build_vector_store, load_vector_store


def drop_self(inputs: dict) -> dict:
    """`traceable` on a method would otherwise log the instance as an input."""
    return {k: v for k, v in inputs.items() if k != "self"}


class RegulationSearch:
    """Embeddings plus the FAISS index. No LLM."""

    def __init__(self) -> None:
        self._embeddings = BedrockEmbeddings(
            model_id=settings.bedrock_embedding_model,
            region_name=settings.aws_region,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self._store = None

    def ingest(self, *, target: str = "faiss", dry_run: bool = False) -> dict[str, int | str | bool]:
        """Ingest corpus PDFs into the explicitly selected destination."""
        if target == "s3vectors":
            from .s3_ingestion import ingest_pdf_corpus

            return ingest_pdf_corpus(dry_run=dry_run)
        if target != "faiss":
            raise ValueError("Ingestion target must be 'faiss' or 's3vectors'.")
        documents = load_pdf_documents(settings.corpus_dir)
        chunks = self._splitter.split_documents(documents)
        if not chunks:
            raise ValueError(f"No extractable PDF text found in {settings.corpus_dir}")
        if not dry_run:
            self._store = build_vector_store(chunks, self._embeddings, settings.index_dir)
        return {
            "target": "faiss",
            "dry_run": dry_run,
            "documents": len(documents),
            "pdf_files": len({document.metadata["source"] for document in documents}),
            "chunks": len(chunks),
            "index_dir": str(settings.index_dir),
            "corpus_dir": str(settings.corpus_dir),
        }

    def vector_store(self):
        """The index, built on first use if it is not on disk yet.

        Kept lazy and self-healing so a fresh checkout answers the first query
        without anyone having to remember to call `/ingest`.
        """
        if self._store is None:
            try:
                self._store = load_vector_store(self._embeddings, settings.index_dir)
            except FileNotFoundError:
                self.ingest()
        return self._store

    # `run_type="retriever"` is what makes LangSmith render the result as
    # documents rather than an opaque blob, and puts the embedding + FAISS
    # lookup on the trace as its own step instead of hiding it inside whatever
    # called it.
    @traceable(run_type="retriever", name="regulation_search", process_inputs=drop_self)
    def search(self, query: str, k: int = 5) -> list[Document]:
        """The `k` corpus passages closest to `query`, best first."""
        return self.vector_store().similarity_search(query, k=k)
