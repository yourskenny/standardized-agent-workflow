from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from .blocks import PromptBlock
from .budget import trim_blocks


@dataclass(frozen=True)
class CompiledPrompt:
    blocks: list[PromptBlock]

    def trace_blocks(self) -> list[dict[str, object]]:
        return [
            {
                "source": block.source,
                "type": block.type,
                "token_count": block.token_count,
            }
            for block in self.blocks
        ]


def compile_prompt(
    settings: Settings,
    question: str,
    retrieved_chunks: list[dict[str, object]],
    history: list[dict[str, object]] | None = None,
    max_tokens: int = 4000,
) -> CompiledPrompt:
    blocks: list[PromptBlock] = []
    system_prompt = settings.prompt_path.read_text(encoding="utf-8") if settings.prompt_path.exists() else ""
    if system_prompt.strip():
        blocks.append(PromptBlock(source=str(settings.prompt_path), type="stable", text=system_prompt.strip()))
    for index, chunk in enumerate(retrieved_chunks, start=1):
        source = str(chunk.get("source", "") or f"chunk-{index}")
        title = str(chunk.get("title", ""))
        content = str(chunk.get("content", ""))
        text = f"[{index}] {title}\nSource: {source}\n{content}".strip()
        blocks.append(PromptBlock(source=source, type="context", text=text))
    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            blocks.append(PromptBlock(source=f"history.{role}", type="volatile", text=content.strip()))
    blocks.append(PromptBlock(source="request.message", type="volatile", text=question))
    return CompiledPrompt(trim_blocks(blocks, max_tokens=max_tokens))
