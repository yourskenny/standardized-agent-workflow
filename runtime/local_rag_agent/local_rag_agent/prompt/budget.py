from __future__ import annotations

from .blocks import PromptBlock


def trim_blocks(blocks: list[PromptBlock], max_tokens: int) -> list[PromptBlock]:
    if max_tokens <= 0:
        return list(blocks)
    fixed_tokens = sum(block.token_count for block in blocks if block.type != "context")
    remaining = max_tokens - fixed_tokens
    kept_context_indexes: set[int] = set()
    if remaining > 0:
        for index, block in enumerate(blocks):
            if block.type != "context":
                continue
            if block.token_count <= remaining:
                kept_context_indexes.add(index)
                remaining -= block.token_count
    return [
        block
        for index, block in enumerate(blocks)
        if block.type != "context" or index in kept_context_indexes
    ]
