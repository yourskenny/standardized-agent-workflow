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
    history: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    sources = [_source_payload(chunk) for chunk in retrieved_chunks]
    if model_client is None:
        return {
            "answer": build_extractive_answer(question, retrieved_chunks),
            "sources": sources,
            "mode": "extractive",
        }

    messages = build_messages(settings, question, retrieved_chunks, history=history)
    answer = model_client.chat(messages)  # type: ignore[attr-defined]
    return {"answer": answer, "sources": sources, "mode": "model"}


def build_messages(
    settings: Settings,
    question: str,
    retrieved_chunks: list[dict[str, object]],
    history: list[dict[str, object]] | None = None,
) -> list[dict[str, str]]:
    system_prompt = settings.prompt_path.read_text(encoding="utf-8") if settings.prompt_path.exists() else ""
    context = "\n\n".join(
        f"[{index + 1}] {chunk.get('title', '')}\n来源: {chunk.get('source', '')}\n{chunk.get('content', '')}"
        for index, chunk in enumerate(retrieved_chunks)
    )
    user_prompt = f"检索片段：\n{context}\n\n用户问题：{question}"
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\n{LOCAL_RUNTIME_INSTRUCTION}".strip()},
    ]
    messages.extend(_sanitize_history(history or []))
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _sanitize_history(history: list[dict[str, object]], limit: int = 8) -> list[dict[str, str]]:
    clean: list[dict[str, str]] = []
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        stripped = content.strip()
        if not stripped:
            continue
        clean.append({"role": str(role), "content": stripped[:4000]})
    return clean[-limit:]


def build_extractive_answer(question: str, retrieved_chunks: list[dict[str, object]]) -> str:
    if _is_complete_submission_request(question):
        return (
            "我不能直接替你完成完整论文、完整作业或可直接提交的报告，"
            "但可以帮你把任务拆成研究问题、数据来源、分析方法、R 代码步骤、论文结构和检查清单。"
        )

    if not retrieved_chunks:
        return "根据目前知识库资料，未找到明确说明。建议你向任课教师或助教确认。"

    top_chunk = retrieved_chunks[0]
    answer = _extract_answer_text(str(top_chunk.get("content", "")))
    if not answer:
        answer = _compact_snippet(str(top_chunk.get("content", "")), 420)

    return answer.strip()


def _is_complete_submission_request(question: str) -> bool:
    compact = question.replace(" ", "")
    completion_terms = ("完整论文", "完整作业", "完整报告", "直接提交", "代写", "帮我写完", "直接帮我写")
    return any(term in compact for term in completion_terms)


def _extract_answer_text(content: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("答："):
            return line[2:].strip()
        if line.startswith("答案："):
            return line[3:].strip()
    for marker in ("答：", "答案："):
        if marker in content:
            return content.split(marker, 1)[1].split("\n\n", 1)[0].strip()
    return ""


def _compact_snippet(text: str, limit: int) -> str:
    snippet = text.replace("\n", " ").strip()
    if len(snippet) > limit:
        return snippet[:limit].rstrip() + "..."
    return snippet


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
        snippet = _compact_snippet(str(chunk.get("content", "")), 240)
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
