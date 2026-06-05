from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptBlock:
    source: str
    type: str
    text: str
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            object.__setattr__(self, "token_count", estimate_tokens(self.text))


def estimate_tokens(text: str) -> int:
    compact = text.strip()
    if not compact:
        return 0
    word_count = len(compact.split())
    char_estimate = max(1, len(compact) // 4)
    return max(word_count, char_estimate)
