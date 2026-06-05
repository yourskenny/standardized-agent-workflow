from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from .chunking import chunk_markdown
from .config import Settings, load_settings
from .index_store import write_index
from .manifest import expand_manifest_entries
from .ports import RetrieverProvider
from .regression import parse_regression_questions, run_regression, summarize_regression_report
from .runtime import AgentRuntime
from .types import AgentRequest
from .validator import validate_project_config
from .workflow import WorkflowRegistry


def ingest_project(settings: Settings) -> dict[str, int | str]:
    files = expand_manifest_entries(settings)
    chunks: list[dict[str, object]] = []
    for file_path in files:
        relative = file_path.resolve().relative_to(settings.project_root).as_posix()
        text = file_path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(Path(relative), text, settings.chunk_size, settings.chunk_overlap))
    write_index(settings, chunks)
    return {"file_count": len(files), "chunk_count": len(chunks), "index_path": str(settings.index_path)}


def retrieve_question(settings: Settings, question: str) -> list[dict[str, object]]:
    return RetrieverProvider.from_settings(settings).retrieve(settings, question)


def chat_question(
    settings: Settings,
    question: str,
    model_client: object | None = None,
    history: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    runtime = AgentRuntime(settings, model_client=model_client)
    request = AgentRequest(message=question, history=history or [])
    return runtime.run(request).to_dict()


def build_retrieval_query(question: str, history: list[dict[str, object]] | None = None) -> str:
    return AgentRuntime.build_retrieval_query(question, history)


def demo_check(settings: Settings, dify_url: str | None = None) -> dict[str, object]:
    questions = _demo_questions(settings)
    checks: list[dict[str, object]] = []
    for question in questions:
        try:
            response = chat_question(settings, question, model_client=None)
            sources = response.get("sources", [])
            top_source = sources[0].get("source", "") if sources else ""
            checks.append(
                {
                    "question": question,
                    "mode": response.get("mode", ""),
                    "top_source": top_source,
                    "answer": response.get("answer", ""),
                    "ok": bool(top_source or response.get("answer")),
                }
            )
        except Exception as error:  # pragma: no cover - exercised through CLI in real runs
            checks.append({"question": question, "ok": False, "error": str(error)})

    return {
        "index_exists": settings.index_path.exists(),
        "index_path": str(settings.index_path),
        "runtime_configs": _runtime_config_status(settings),
        "workflows": _workflow_status(),
        "dify": _check_url(dify_url) if dify_url else None,
        "checks": checks,
    }


def smoke_project(settings: Settings, config_path: Path, questions_path: Path) -> dict[str, object]:
    validation = validate_project_config(settings.project_root, config_path)
    if not validation.ok:
        return {"ok": False, "stage": "validate", "validate": validation.to_dict()}

    ingest_result = ingest_project(settings)
    output_dir = settings.regression_output_dir or settings.project_root / ".local_rag_agent" / "regression"
    regression_output = output_dir / f"{questions_path.stem}-smoke.jsonl"
    question_count = run_regression(
        questions_path,
        regression_output,
        lambda question: chat_question(settings, question),
    )
    release_gate = summarize_regression_report(regression_output)
    http_probe = _probe_http(settings)
    ok = bool(release_gate.get("ok")) and bool(http_probe["healthz"].get("ok")) and bool(http_probe["version"].get("ok"))
    return {
        "ok": ok,
        "validate": validation.to_dict(),
        "ingest": ingest_result,
        "regression": {"question_count": question_count, "output": str(regression_output)},
        "release_gate": release_gate,
        "http": http_probe,
    }


def main(argv: list[str] | None = None) -> int:
    configure_output_stream(sys.stdout)
    configure_output_stream(sys.stderr)
    parser = argparse.ArgumentParser(prog="local_rag_agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in (
        "ingest",
        "retrieve",
        "chat",
        "serve",
        "regression",
        "demo-check",
        "validate",
        "release-gate",
        "smoke",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--project", required=True, type=Path)
        subparser.add_argument("--config", required=True, type=Path)
        if command in {"retrieve", "chat"}:
            subparser.add_argument("question")
        if command == "serve":
            subparser.add_argument("--port", type=int, default=8765)
        if command == "regression":
            subparser.add_argument("--questions", required=True, type=Path)
            subparser.add_argument("--output", type=Path)
        if command == "smoke":
            subparser.add_argument("--questions", required=True, type=Path)
        if command == "demo-check":
            subparser.add_argument("--dify-url")
        if command == "release-gate":
            subparser.add_argument("--report", required=True, type=Path)

    args = parser.parse_args(argv)
    settings = load_settings(args.project, args.config) if args.command not in {"validate", "release-gate"} else None

    if args.command == "ingest":
        print(json.dumps(ingest_project(settings), ensure_ascii=False, indent=2))
        return 0

    if args.command == "retrieve":
        print(json.dumps(retrieve_question(settings, args.question), ensure_ascii=False, indent=2))
        return 0

    if args.command == "chat":
        response = chat_question(settings, args.question)
        print(response["answer"])
        if response.get("sources"):
            print("\n来源：")
            for source in response["sources"]:
                print(f"- {source.get('source')} ({source.get('chunk_id')}, score={source.get('score')})")
        return 0

    if args.command == "serve":
        from .server import run_server

        run_server(settings, args.port)
        return 0

    if args.command == "regression":
        output = args.output
        if output is None:
            output_dir = settings.regression_output_dir or settings.project_root / ".local_rag_agent" / "regression"
            output = output_dir / f"{args.questions.stem}.jsonl"
        count = run_regression(args.questions, output, lambda question: chat_question(settings, question))
        print(json.dumps({"question_count": count, "output": str(output)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "demo-check":
        print(json.dumps(demo_check(settings, args.dify_url), ensure_ascii=False, indent=2))
        return 0

    if args.command == "validate":
        result = validate_project_config(args.project, args.config)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if result.ok else 1

    if args.command == "release-gate":
        validation = validate_project_config(args.project, args.config)
        if not validation.ok:
            print(
                json.dumps(
                    {"stage": "validate", "validation": validation.to_dict()},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1
        summary = summarize_regression_report(args.report)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["ok"] else 1

    if args.command == "smoke":
        result = smoke_project(settings, args.config, args.questions)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def configure_output_stream(stream: object) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    encoding = str(getattr(stream, "encoding", "") or "").lower()
    if callable(reconfigure) and encoding not in {"utf-8", "utf8"}:
        reconfigure(encoding="utf-8", errors="replace")


def _runtime_config_status(settings: Settings) -> dict[str, dict[str, object]]:
    paths = {
        "intent_config": settings.intent_config_path,
        "workflow_config": settings.workflow_config_path,
        "policy_config": settings.policy_config_path,
        "tool_config": settings.tool_config_path,
    }
    return {
        name: {
            "path": str(path) if path is not None else "",
            "exists": bool(path and path.exists()),
        }
        for name, path in paths.items()
    }


def _workflow_status() -> dict[str, bool]:
    registry = WorkflowRegistry.builtins()
    return {
        "rag_qa": registry.has("rag_qa"),
        "retrieval_debug": registry.has("retrieval_debug"),
        "refusal_with_guidance": registry.has("refusal_with_guidance"),
    }


def _demo_questions(settings: Settings) -> list[str]:
    question_path = settings.project_root / "examples" / "core-regression-questions.md"
    if question_path.exists():
        parsed = parse_regression_questions(question_path.read_text(encoding="utf-8"))
        questions = [item["question"] for item in parsed if item.get("question")]
        if questions:
            return questions[:4]
    return [
        "What can this project agent answer?",
        "What are the main boundaries for this project?",
        "Which sources support the answer?",
        "Please produce a complete directly submittable report.",
    ]


def _check_url(url: str) -> dict[str, object]:
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=10) as response:
            return {"url": url, "ok": 200 <= response.status < 400, "status": response.status}
    except urllib.error.HTTPError as error:
        return {"url": url, "ok": False, "status": error.code, "error": str(error)}
    except urllib.error.URLError as error:
        return {"url": url, "ok": False, "error": str(error.reason)}


def _probe_http(settings: Settings) -> dict[str, dict[str, object]]:
    from .server import make_handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(settings))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        return {
            "healthz": _check_url(f"{base_url}/healthz"),
            "version": _check_url(f"{base_url}/version"),
        }
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
