from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

SOURCE_BOOSTS = (
    ("课程关键事务问答核查表", 18.0),
    ("课程核心事实速查", 14.0),
    ("syllabus", 8.0),
    ("课程阅读材料与参考书目", 6.0),
    ("知识库边界与学术诚信说明", 4.0),
    ("课程问法同义词与意图映射", -8.0),
)


def rank_chunks(question: str, chunks: Iterable[dict[str, object]], top_k: int = 5) -> list[dict[str, object]]:
    query_terms = _term_counts(question)
    if not query_terms:
        return []

    ranked: list[dict[str, object]] = []
    for chunk in chunks:
        content = str(chunk.get("content", ""))
        title = str(chunk.get("title", ""))
        source = str(chunk.get("source", ""))
        terms = _term_counts(f"{title}\n{content}")
        score = _score(query_terms, terms)
        if title and any(term in title for term in query_terms):
            score += 0.3
        if source and any(term in source for term in query_terms):
            score += 0.1
        score += _source_boost(source)
        if score <= 0:
            continue
        enriched = dict(chunk)
        enriched["score"] = round(score, 6)
        ranked.append(enriched)

    ranked.sort(key=lambda item: (-float(item["score"]), str(item.get("source", "")), str(item.get("chunk_id", ""))))
    return ranked[:top_k]


def _source_boost(source: str) -> float:
    normalized = source.lower()
    for marker, boost in SOURCE_BOOSTS:
        if marker.lower() in normalized:
            return boost
    return 0.0


def _score(query_terms: Counter[str], chunk_terms: Counter[str]) -> float:
    overlap = set(query_terms) & set(chunk_terms)
    if not overlap:
        return 0.0
    raw = sum(query_terms[term] * (1 + math.log1p(chunk_terms[term])) for term in overlap)
    coverage = len(overlap) / max(len(query_terms), 1)
    return raw * (1 + coverage)


def _term_counts(text: str) -> Counter[str]:
    normalized = text.lower()
    terms: list[str] = []
    terms.extend(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\(\))?", normalized))
    terms.extend(re.findall(r"\d+(?::\d+)?", normalized))
    cjk = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for block in cjk:
        if len(block) == 1:
            terms.append(block)
        else:
            terms.extend(block[index : index + 2] for index in range(len(block) - 1))
            if len(block) >= 3:
                terms.extend(block[index : index + 3] for index in range(len(block) - 2))
    return Counter(term for term in terms if len(term.strip()) > 0)
