from __future__ import annotations

import re
from pathlib import Path

from .config import Settings

EXCLUDED_PARTS = {"_templates", "_manifests", "_pre_ingestion", "archive", "maintenance", "examples"}


def parse_manifest_entries(text: str) -> list[str]:
    entries: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        match = re.search(r"`([^`]+)`", stripped)
        if not match:
            continue
        entry = match.group(1).strip().replace("\\", "/")
        if entry:
            entries.append(entry)
    return entries


def expand_manifest_entries(settings: Settings) -> list[Path]:
    if not settings.manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {settings.manifest_path}")

    files: list[Path] = []
    for entry in parse_manifest_entries(settings.manifest_path.read_text(encoding="utf-8")):
        candidate = _resolve_entry(settings.project_root, entry)
        if _is_excluded(candidate, settings.project_root):
            continue
        if candidate.is_file() and candidate.suffix.lower() == ".md":
            files.append(candidate)
        elif candidate.is_dir():
            for path in sorted(candidate.rglob("*.md")):
                if not _is_excluded(path, settings.project_root):
                    files.append(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise ValueError(f"No Markdown knowledge files found from manifest: {settings.manifest_path}")
    return unique


def _resolve_entry(root: Path, entry: str) -> Path:
    path = Path(entry)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Manifest entry escapes project root: {entry}")
    return resolved


def _is_excluded(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    return any(part in EXCLUDED_PARTS for part in relative.parts)
