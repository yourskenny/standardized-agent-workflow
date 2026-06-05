from __future__ import annotations

from ..agent import answer_question
from ..config import Settings
from ..ports import GeneratedAnswer, GeneratorPort


class OpenAICompatibleGenerator(GeneratorPort):
    def generate(
        self,
        settings: Settings,
        question: str,
        retrieved_chunks: list[dict[str, object]],
        model_client: object | None = None,
        history: list[dict[str, object]] | None = None,
    ) -> GeneratedAnswer:
        if model_client is None and settings.generation_fallback != "extractive":
            raise ValueError(f"Unsupported generation fallback: {settings.generation_fallback}")
        result = answer_question(settings, question, retrieved_chunks, model_client, history=history)
        sources = result.get("sources", [])
        return GeneratedAnswer(
            answer=str(result.get("answer", "")),
            mode=str(result.get("mode", "")),
            sources=sources if isinstance(sources, list) else [],
        )


class ExtractiveGenerator(GeneratorPort):
    def generate(
        self,
        settings: Settings,
        question: str,
        retrieved_chunks: list[dict[str, object]],
        model_client: object | None = None,
        history: list[dict[str, object]] | None = None,
    ) -> GeneratedAnswer:
        result = answer_question(settings, question, retrieved_chunks, model_client=None, history=history)
        sources = result.get("sources", [])
        return GeneratedAnswer(
            answer=str(result.get("answer", "")),
            mode=str(result.get("mode", "")),
            sources=sources if isinstance(sources, list) else [],
        )


RagGenerator = OpenAICompatibleGenerator
