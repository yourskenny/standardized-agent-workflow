from __future__ import annotations

from .config import Settings


LOCAL_RUNTIME_INSTRUCTION = """你正在通过本地 RAG 运行时回答问题。
必须优先使用检索片段；如果片段没有依据，不要编造课程事实。
回答后保留简短的来源列表，便于维护者核查。"""


def answer_question(
    settings: Settings,
    question: str,
    retrieved_chunks: list[dict[str, object]],
    model_client: object | None = None,
) -> dict[str, object]:
    sources = [_source_payload(chunk) for chunk in retrieved_chunks]
    if model_client is None:
        return {
            "answer": _retrieval_only_answer(question, retrieved_chunks),
            "sources": sources,
            "mode": "retrieval_only",
        }

    messages = build_messages(settings, question, retrieved_chunks)
    answer = model_client.chat(messages)  # type: ignore[attr-defined]
    return {"answer": answer, "sources": sources, "mode": "model"}


def build_messages(settings: Settings, question: str, retrieved_chunks: list[dict[str, object]]) -> list[dict[str, str]]:
    system_prompt = settings.prompt_path.read_text(encoding="utf-8") if settings.prompt_path.exists() else ""
    context = "\n\n".join(
        f"[{index + 1}] {chunk.get('title', '')}\n来源: {chunk.get('source', '')}\n{chunk.get('content', '')}"
        for index, chunk in enumerate(retrieved_chunks)
    )
    user_prompt = f"检索片段：\n{context}\n\n用户问题：{question}"
    return [
        {"role": "system", "content": f"{system_prompt}\n\n{LOCAL_RUNTIME_INSTRUCTION}".strip()},
        {"role": "user", "content": user_prompt},
    ]


def _retrieval_only_answer(question: str, retrieved_chunks: list[dict[str, object]]) -> str:
    lines = [
        "本地检索结果（未配置模型 API，因此没有生成式回答）：",
        f"问题：{question}",
        "",
    ]
    if not retrieved_chunks:
        lines.append("没有检索到相关片段。")
        return "\n".join(lines)
    for index, chunk in enumerate(retrieved_chunks, start=1):
        snippet = str(chunk.get("content", "")).replace("\n", " ").strip()
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "..."
        lines.append(f"{index}. {chunk.get('source')}#{chunk.get('chunk_id', '').split('#')[-1]} score={chunk.get('score', 0)}")
        lines.append(f"   {snippet}")
    return "\n".join(lines)


def _source_payload(chunk: dict[str, object]) -> dict[str, object]:
    return {
        "source": chunk.get("source", ""),
        "title": chunk.get("title", ""),
        "chunk_id": chunk.get("chunk_id", ""),
        "score": chunk.get("score", 0),
    }
