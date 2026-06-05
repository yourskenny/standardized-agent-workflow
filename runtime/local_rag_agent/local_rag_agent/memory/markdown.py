from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryEntry:
    source: str
    text: str


def load_memory(project_root: Path) -> list[MemoryEntry]:
    root = Path(project_root).resolve()
    memory_dir = root / "memory"
    if not memory_dir.exists():
        return []
    entries: list[MemoryEntry] = []
    for path in sorted(memory_dir.glob("*.md")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            entries.append(MemoryEntry(source=path.relative_to(root).as_posix(), text=text))
    return entries


def write_memory_proposal(
    project_root: Path,
    filename: str,
    text: str,
    proposal_mode: bool = False,
) -> Path:
    if not proposal_mode:
        raise PermissionError("Memory writes require proposal_mode=True")
    root = Path(project_root).resolve()
    proposal_dir = root / "memory" / "_proposals"
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid memory proposal path: {filename}")
    target = (proposal_dir / relative).resolve()
    if target != proposal_dir and proposal_dir not in target.parents:
        raise ValueError(f"Memory proposal path escapes project root: {filename}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
