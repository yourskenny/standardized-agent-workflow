from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import Settings


def write_index(settings: Settings, chunks: Iterable[dict[str, object]]) -> dict[str, object]:
    payload = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(settings.project_root),
        "chunks": list(chunks),
    }
    settings.index_path.parent.mkdir(parents=True, exist_ok=True)
    settings.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def read_index(settings: Settings) -> dict[str, object]:
    if not settings.index_path.exists():
        raise FileNotFoundError(f"Index not found: {settings.index_path}. Run ingest first.")
    return json.loads(settings.index_path.read_text(encoding="utf-8"))
