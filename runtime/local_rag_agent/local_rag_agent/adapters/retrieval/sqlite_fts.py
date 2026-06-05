from __future__ import annotations

import sqlite3
from contextlib import closing

from ...config import Settings
from ...ports import RetrieverPort


class SQLiteFTSRetriever(RetrieverPort):
    def retrieve(self, settings: Settings, query: str) -> list[dict[str, object]]:
        if not settings.index_path.exists():
            raise FileNotFoundError(f"FTS index not found: {settings.index_path}.")
        with closing(sqlite3.connect(settings.index_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT chunk_id, source, title, content, bm25(chunks_fts) AS rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (_fts_query(query), settings.top_k),
            ).fetchall()
        return [
            {
                "chunk_id": row["chunk_id"],
                "source": row["source"],
                "title": row["title"],
                "content": row["content"],
                "score": round(1.0 / (1.0 + abs(float(row["rank"]))), 6),
            }
            for row in rows
        ]


def _fts_query(query: str) -> str:
    terms = [term.replace('"', "").strip() for term in query.split() if term.strip()]
    return " OR ".join(f'"{term}"' for term in terms) or '""'
