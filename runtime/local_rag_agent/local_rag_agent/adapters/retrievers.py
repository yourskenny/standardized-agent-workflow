from __future__ import annotations

from ..config import Settings
from ..index_store import read_index
from ..ports import RetrieverPort
from ..retrieval import rank_chunks


class LexicalRetriever(RetrieverPort):
    def retrieve(self, settings: Settings, query: str) -> list[dict[str, object]]:
        payload = read_index(settings)
        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list):
            raise ValueError(f"Invalid index format: {settings.index_path}")
        return rank_chunks(
            query,
            chunks,
            settings.top_k,
            source_boosts=settings.retrieval_source_boosts,
        )
