from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


def parse_regression_questions(markdown: str) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2 or cells[0] in {"编号", "id", "ID"}:
            continue
        expected = cells[2] if len(cells) > 2 else ""
        questions.append({"id": cells[0], "question": cells[1], "expected": expected})
    return questions


def run_regression(question_file: Path, output_file: Path, answer_fn: Callable[[str], dict[str, object]]) -> int:
    questions = parse_regression_questions(question_file.read_text(encoding="utf-8"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_file.open("w", encoding="utf-8") as handle:
        for item in questions:
            response = answer_fn(item["question"])
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "id": item["id"],
                "question": item["question"],
                "expected": item["expected"],
                "answer": response.get("answer", ""),
                "sources": response.get("sources", []),
                "mode": response.get("mode", ""),
                "intent": response.get("intent", ""),
                "workflow": response.get("workflow", ""),
                "trace": response.get("trace", {}),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
