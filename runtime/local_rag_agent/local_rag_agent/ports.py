from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .agent import answer_question
from .config import Settings
from .index_store import read_index
from .retrieval import rank_chunks


class RetrieverPort(Protocol):
    def retrieve(self, settings: Settings, query: str) -> list[dict[str, object]]:
        ...


class GeneratorPort(Protocol):
    def generate(
        self,
        settings: Settings,
        question: str,
        retrieved_chunks: list[dict[str, object]],
        model_client: object | None = None,
        history: list[dict[str, object]] | None = None,
    ) -> "GeneratedAnswer":
        ...


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    mode: str
    sources: list[dict[str, object]]


class LexicalRetriever:
    def retrieve(self, settings: Settings, query: str) -> list[dict[str, object]]:
        payload = read_index(settings)
        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list):
            raise ValueError(f"Invalid index format: {settings.index_path}")
        return rank_chunks(
            query,
            chunks,
            settings.top_k,
            source_boosts=settings.retrieval_source_boosts,
        )


class RagGenerator:
    def generate(
        self,
        settings: Settings,
        question: str,
        retrieved_chunks: list[dict[str, object]],
        model_client: object | None = None,
        history: list[dict[str, object]] | None = None,
    ) -> GeneratedAnswer:
        result = answer_question(settings, question, retrieved_chunks, model_client, history=history)
        sources = result.get("sources", [])
        return GeneratedAnswer(
            answer=str(result.get("answer", "")),
            mode=str(result.get("mode", "")),
            sources=sources if isinstance(sources, list) else [],
        )


class RetrieverProvider:
    def __init__(self, retriever: RetrieverPort):
        self.retriever = retriever

    @classmethod
    def from_settings(cls, settings: Settings) -> "RetrieverProvider":
        provider = settings.retrieval_provider
        if provider != "lexical":
            raise ValueError(f"Unsupported retriever provider: {provider}")
        return cls(LexicalRetriever())

    def retrieve(self, settings: Settings, query: str) -> list[dict[str, object]]:
        return self.retriever.retrieve(settings, query)


class GeneratorProvider:
    def __init__(self, generator: GeneratorPort):
        self.generator = generator

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeneratorProvider":
        provider = settings.generation_provider
        if provider != "openai_compatible":
            raise ValueError(f"Unsupported generator provider: {provider}")
        return cls(RagGenerator())

    def generate(
        self,
        settings: Settings,
        question: str,
        retrieved_chunks: list[dict[str, object]],
        model_client: object | None = None,
        history: list[dict[str, object]] | None = None,
    ) -> GeneratedAnswer:
        return self.generator.generate(settings, question, retrieved_chunks, model_client, history=history)
