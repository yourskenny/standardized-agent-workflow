from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import Settings


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


class PolicyPort(Protocol):
    def evaluate(
        self,
        message: str,
        intent_decision: object,
        retrieved_chunks: list[dict[str, object]] | None = None,
        require_sources: bool = False,
    ) -> object:
        ...


class ToolPort(Protocol):
    def call(
        self,
        tool_id: str,
        arguments: dict[str, object],
        intent_id: str = "",
    ) -> object:
        ...


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    mode: str
    sources: list[dict[str, object]]


class RetrieverProvider:
    def __init__(self, retriever: RetrieverPort):
        self.retriever = retriever

    @classmethod
    def from_settings(cls, settings: Settings) -> "RetrieverProvider":
        provider = settings.retrieval_provider
        if provider == "lexical":
            from .adapters.retrievers import LexicalRetriever

            return cls(LexicalRetriever())
        raise ValueError(f"Unsupported retriever provider: {provider}")

    def retrieve(self, settings: Settings, query: str) -> list[dict[str, object]]:
        return self.retriever.retrieve(settings, query)


class GeneratorProvider:
    def __init__(self, generator: GeneratorPort):
        self.generator = generator

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeneratorProvider":
        provider = settings.generation_provider
        if provider == "extractive":
            from .adapters.generators import ExtractiveGenerator

            return cls(ExtractiveGenerator())
        if provider == "openai_compatible":
            from .adapters.generators import OpenAICompatibleGenerator

            return cls(OpenAICompatibleGenerator())
        raise ValueError(f"Unsupported generator provider: {provider}")

    def generate(
        self,
        settings: Settings,
        question: str,
        retrieved_chunks: list[dict[str, object]],
        model_client: object | None = None,
        history: list[dict[str, object]] | None = None,
    ) -> GeneratedAnswer:
        return self.generator.generate(settings, question, retrieved_chunks, model_client, history=history)


def __getattr__(name: str) -> object:
    if name == "LexicalRetriever":
        from .adapters.retrievers import LexicalRetriever

        return LexicalRetriever
    if name == "ExtractiveGenerator":
        from .adapters.generators import ExtractiveGenerator

        return ExtractiveGenerator
    if name in {"OpenAICompatibleGenerator", "RagGenerator"}:
        from .adapters.generators import OpenAICompatibleGenerator

        return OpenAICompatibleGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GeneratedAnswer",
    "GeneratorPort",
    "GeneratorProvider",
    "PolicyPort",
    "RetrieverPort",
    "RetrieverProvider",
    "ToolPort",
    "LexicalRetriever",
    "ExtractiveGenerator",
    "OpenAICompatibleGenerator",
    "RagGenerator",
]
