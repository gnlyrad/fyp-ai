from __future__ import annotations

from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_aws import BedrockEmbeddings
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .documents import load_pdf_documents
from .settings import settings
from .vectorstore import build_vector_store, load_vector_store


@dataclass
class RagResult:
    answer: str
    sources: list[str]


class CodeRAGService:
    def __init__(self) -> None:
        self._embeddings = BedrockEmbeddings(
            model_id=settings.bedrock_embedding_model,
            region_name=settings.aws_region,
        )
        self._llm = init_chat_model(
            settings.bedrock_chat_model,
            model_provider="bedrock_converse",
            region_name=settings.aws_region,
            temperature=0,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
        )

    def ingest(self) -> dict[str, int | str]:
        documents = load_pdf_documents(settings.corpus_dir)
        chunks = self._splitter.split_documents(documents)
        build_vector_store(chunks, self._embeddings, settings.index_dir)
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "index_dir": str(settings.index_dir),
            "corpus_dir": str(settings.corpus_dir),
        }

    def answer(self, question: str, top_k: int = 5) -> RagResult:
        try:
            vector_store = load_vector_store(self._embeddings, settings.index_dir)
        except FileNotFoundError:
            self.ingest()
            vector_store = load_vector_store(self._embeddings, settings.index_dir)

        matches = vector_store.similarity_search(question, k=top_k)
        context_blocks: list[str] = []
        sources: list[str] = []

        for index, document in enumerate(matches, start=1):
            source = document.metadata.get("source", document.metadata.get("path", "unknown"))
            if source not in sources:
                sources.append(source)
            context_blocks.append(f"[{index}] {source}\n{document.page_content}")

        prompt = (
            "You are a code assistant for this workspace. Answer the question using "
            "only the provided context. If the context does not contain the answer, "
            "say that you do not know. Cite the most relevant source paths when possible.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{'\n\n'.join(context_blocks)}"
        )

        response = self._llm.invoke([HumanMessage(content=prompt)])
        answer_text = getattr(response, "content", str(response))
        return RagResult(answer=str(answer_text).strip(), sources=sources)
