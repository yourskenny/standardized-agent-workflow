from __future__ import annotations

import re
import tomllib
from pathlib import Path

from .definitions import SkillDefinition


class SkillRegistry:
    def __init__(self, skills: list[SkillDefinition] | None = None) -> None:
        self._skills = list(skills or [])

    @property
    def skills(self) -> list[SkillDefinition]:
        return list(self._skills)

    @classmethod
    def from_project(cls, project_root: Path) -> "SkillRegistry":
        return cls(load_skills(Path(project_root) / "skills"))

    def select(self, query: str, limit: int = 3) -> list[SkillDefinition]:
        tokens = _tokens(query)
        query_text = query.casefold()
        scored: list[tuple[int, int, SkillDefinition]] = []
        for index, skill in enumerate(self._skills):
            score = _score_skill(skill, query_text, tokens)
            if score > 0:
                scored.append((score, -index, skill))
        scored.sort(reverse=True)
        return [skill for _, _, skill in scored[:limit]]


def load_skills(skills_root: Path) -> list[SkillDefinition]:
    if not skills_root.exists():
        return []
    skills: list[SkillDefinition] = []
    seen_ids: set[str] = set()
    for skill_dir in sorted(item for item in skills_root.iterdir() if item.is_dir()):
        manifest_path = skill_dir / "manifest.toml"
        skill_path = skill_dir / "SKILL.md"
        if not manifest_path.exists() or not skill_path.exists():
            continue
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8-sig"))
        skill_id = str(data.get("id", skill_dir.name)).strip()
        if not skill_id:
            raise ValueError(f"Skill missing id in {manifest_path}")
        if skill_id in seen_ids:
            raise ValueError(f"Duplicate skill id: {skill_id}")
        seen_ids.add(skill_id)
        skills.append(
            SkillDefinition(
                id=skill_id,
                name=str(data.get("name", skill_id)).strip() or skill_id,
                description=str(data.get("description", "")).strip(),
                keywords=tuple(_string_list(data.get("keywords", []))),
                text=skill_path.read_text(encoding="utf-8").strip(),
                source=(skill_path.relative_to(skills_root.parent)).as_posix(),
            )
        )
    return skills


def _score_skill(skill: SkillDefinition, query_text: str, tokens: set[str]) -> int:
    score = 0
    for keyword in skill.keywords:
        value = keyword.casefold().strip()
        if value and value in query_text:
            score += 10
    searchable = " ".join([skill.id, skill.name, skill.description]).casefold()
    for token in tokens:
        if token and token in searchable:
            score += 1
    return score


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[A-Za-z0-9_]+", text)}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
