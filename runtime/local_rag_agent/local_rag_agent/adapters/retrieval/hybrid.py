from __future__ import annotations

import sqlite3

from ...adapters.retrievers import LexicalRetriever
from ...config import Settings
from ...ports import RetrieverPort
from .sqlite_fts import SQLiteFTSRetriever


class HybridRetriever(RetrieverPort):
    def retrieve(self, settings: Settings, query: str) -> list[dict[str, object]]:
        merged: dict[tuple[str, str], dict[str, object]] = {}
        for chunk in _safe_retrieve(SQLiteFTSRetriever(), settings, query):
            key = (str(chunk.get("source", "")), str(chunk.get("chunk_id", "")))
            item = dict(chunk)
            item["score"] = float(item.get("score", 0)) + 1.0
            merged[key] = item
        for chunk in _safe_retrieve(LexicalRetriever(), settings, query):
            key = (str(chunk.get("source", "")), str(chunk.get("chunk_id", "")))
            if key in merged:
                merged[key]["score"] = float(merged[key].get("score", 0)) + float(chunk.get("score", 0))
            else:
                merged[key] = dict(chunk)
        ranked = list(merged.values())
        ranked.sort(key=lambda item: (-float(item.get("score", 0)), str(item.get("source", "")), str(item.get("chunk_id", ""))))
        return ranked[: settings.top_k]


def _safe_retrieve(provider: RetrieverPort, settings: Settings, query: str) -> list[dict[str, object]]:
    try:
        return provider.retrieve(settings, query)
    except (FileNotFoundError, sqlite3.OperationalError, ValueError, KeyError):
        return []
