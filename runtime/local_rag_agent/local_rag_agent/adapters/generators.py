from __future__ import annotations

from ..agent import answer_question
from ..config import Settings
from ..ports import GeneratedAnswer, GeneratorPort
from ..prompt.compiler import compile_prompt
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
        compiled_prompt = compile_prompt(settings, question, retrieved_chunks, history=history)
        result = answer_question(
            settings,
            question,
            retrieved_chunks,
            resolution.client,
            history=history,
            compiled_prompt=compiled_prompt if resolution.client is not None else None,
        )
        sources = result.get("sources", [])
        metadata = resolution.trace_metadata()
        metadata["prompt_blocks"] = compiled_prompt.trace_blocks()
        return GeneratedAnswer(
            answer=str(result.get("answer", "")),
            mode=str(result.get("mode", "")),
            sources=sources if isinstance(sources, list) else [],
            metadata=metadata,
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
