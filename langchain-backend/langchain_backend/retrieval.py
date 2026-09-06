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

from .documents import load_pdf_documents
from .settings import settings
from .vectorstore import build_vector_store, load_vector_store

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


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

    def ingest(self) -> dict[str, int | str]:
        """Rebuild the index from every PDF in the corpus folder."""
        documents = load_pdf_documents(settings.corpus_dir)
        chunks = self._splitter.split_documents(documents)
        self._store = build_vector_store(chunks, self._embeddings, settings.index_dir)
        return {
            "documents": len(documents),
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
