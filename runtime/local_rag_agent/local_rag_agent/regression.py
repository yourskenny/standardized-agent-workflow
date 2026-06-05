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


def summarize_regression_report(path: Path) -> dict[str, object]:
    records = _read_jsonl(path)
    failures: list[dict[str, object]] = []
    missing_source_count = 0
    policy_trace_count = 0
    tool_trace_count = 0
    modes: dict[str, int] = {}
    intents: dict[str, int] = {}
    workflows: dict[str, int] = {}

    for record in records:
        mode = str(record.get("mode", ""))
        intent = str(record.get("intent", ""))
        workflow = str(record.get("workflow", ""))
        modes[mode] = modes.get(mode, 0) + 1
        intents[intent] = intents.get(intent, 0) + 1
        workflows[workflow] = workflows.get(workflow, 0) + 1
        trace = record.get("trace", {})
        steps = trace.get("steps", []) if isinstance(trace, dict) else []
        if any(isinstance(step, dict) and step.get("name") == "apply_policy" for step in steps):
            policy_trace_count += 1
        if any(isinstance(step, dict) and str(step.get("name", "")).startswith("tool.") for step in steps):
            tool_trace_count += 1
        sources = record.get("sources", [])
        has_sources = isinstance(sources, list) and bool(sources)
        if not has_sources and mode not in {"refusal", "no_evidence", "tool", "tool_error"}:
            missing_source_count += 1
            failures.append(
                {
                    "id": record.get("id", ""),
                    "question": record.get("question", ""),
                    "reason": "missing_sources",
                    "mode": mode,
                }
            )

    return {
        "ok": not failures,
        "question_count": len(records),
        "missing_source_count": missing_source_count,
        "policy_trace_count": policy_trace_count,
        "tool_trace_count": tool_trace_count,
        "modes": modes,
        "intents": intents,
        "workflows": workflows,
        "failures": failures,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not path.exists():
        raise FileNotFoundError(f"Regression report not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if isinstance(item, dict):
            records.append(item)
    return records
