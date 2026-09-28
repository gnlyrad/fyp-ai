from __future__ import annotations

from dataclasses import dataclass

from .models import create_chat_model
from langchain_core.messages import HumanMessage
from langsmith import traceable

from .retrieval import RegulationSearch, drop_self
from .settings import settings


@dataclass
class RagResult:
    answer: str
    sources: list[str]


class CodeRAGService:
    def __init__(self) -> None:
        self._search = RegulationSearch()
        self._llm = create_chat_model()

    def ingest(self, *, target: str = "faiss", dry_run: bool = False) -> dict[str, int | str | bool]:
        return self._search.ingest(target=target, dry_run=dry_run)

    # Groups the retrieval and the generation under one root run, so a
    # LangSmith trace of `/ask` shows the whole RAG pipeline rather than a
    # bare LLM call with a pre-baked prompt and no visible source.
    @traceable(run_type="chain", name="regulation_rag", process_inputs=drop_self)
    def answer(self, question: str, top_k: int = 5) -> RagResult:
        matches = self._search.search(question, k=top_k)
        context_blocks: list[str] = []
        sources: list[str] = []

        for index, document in enumerate(matches, start=1):
            source = document.metadata.get("source", document.metadata.get("path", "unknown"))
            if source not in sources:
                sources.append(source)
            context_blocks.append(f"[{index}] {source}\n{document.page_content}")

        joined = "\n\n".join(context_blocks)
        prompt = (
            "You are a compliance assistant answering questions about the MAS "
            "financial-advisory notices and guidelines in the provided context. "
            "Answer using only that context. If it does not contain the answer, "
            "say that you do not know. Cite the most relevant source documents.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{joined}"
        )

        response = self._llm.invoke([HumanMessage(content=prompt)])
        answer_text = getattr(response, "content", str(response))
        return RagResult(answer=str(answer_text).strip(), sources=sources)
