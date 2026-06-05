from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SkillDefinition:
    id: str
    name: str
    description: str = ""
    keywords: tuple[str, ...] = field(default_factory=tuple)
    text: str = ""
    source: str = ""
