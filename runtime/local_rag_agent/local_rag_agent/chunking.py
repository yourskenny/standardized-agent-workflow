from __future__ import annotations

import re
from pathlib import Path


def chunk_markdown(source_path: Path, text: str, chunk_size: int = 1200, chunk_overlap: int = 160) -> list[dict[str, str]]:
    title = _extract_title(text, source_path)
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not normalized:
        return []

    sections = _split_by_heading(normalized)
    chunks: list[dict[str, str]] = []
    for section_title, section_body in sections:
        prefix = section_title if section_title else title
        for piece in _split_text(section_body, chunk_size, chunk_overlap):
            content = f"{prefix}\n\n{piece}".strip() if prefix and prefix not in piece[:80] else piece.strip()
            chunk_index = len(chunks)
            source = source_path.as_posix()
            chunks.append(
                {
                    "chunk_id": f"{source}#{chunk_index:04d}",
                    "source": source,
                    "title": prefix or title,
                    "content": content,
                }
            )
    return chunks


def _extract_title(text: str, source_path: Path) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return source_path.stem


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end == len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return pieces
