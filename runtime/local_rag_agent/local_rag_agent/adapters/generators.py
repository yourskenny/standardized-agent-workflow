from __future__ import annotations

from ..agent import answer_question
from ..config import Settings
from ..ports import GeneratedAnswer, GeneratorPort
from ..providers.resolver import ModelProviderResolver


class OpenAICompatibleGenerator(GeneratorPort):
    def generate(
        self,
        settings: Settings,
        question: str,
        retrieved_chunks: list[dict[str, object]],
        model_client: object | None = None,
        history: list[dict[str, object]] | None = None,
    ) -> GeneratedAnswer:
        resolution = ModelProviderResolver.from_settings(settings).resolve(model_client=model_client)
        if resolution.client is None and resolution.fallback != "extractive":
            raise ValueError(f"Unsupported generation fallback: {resolution.fallback}")
        result = answer_question(settings, question, retrieved_chunks, resolution.client, history=history)
        sources = result.get("sources", [])
        return GeneratedAnswer(
            answer=str(result.get("answer", "")),
            mode=str(result.get("mode", "")),
            sources=sources if isinstance(sources, list) else [],
            metadata=resolution.trace_metadata(),
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
            metadata={
                "provider": "extractive",
                "model": "",
                "base_url": "",
                "api_key_env": "",
                "fallback": "extractive",
                "credential_status": "not_required",
            },
        )


RagGenerator = OpenAICompatibleGenerator
