from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import answer_question
from .chunking import chunk_markdown
from .config import Settings, load_settings
from .index_store import read_index, write_index
from .llm import OpenAICompatibleClient
from .manifest import expand_manifest_entries
from .regression import run_regression
from .retrieval import rank_chunks


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
    payload = read_index(settings)
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"Invalid index format: {settings.index_path}")
    return rank_chunks(question, chunks, settings.top_k)


def chat_question(settings: Settings, question: str, model_client: object | None = None) -> dict[str, object]:
    retrieved = retrieve_question(settings, question)
    client = model_client if model_client is not None else OpenAICompatibleClient.from_env()
    return answer_question(settings, question, retrieved, client)


def main(argv: list[str] | None = None) -> int:
    configure_output_stream(sys.stdout)
    configure_output_stream(sys.stderr)
    parser = argparse.ArgumentParser(prog="local_rag_agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("ingest", "retrieve", "chat", "serve", "regression"):
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

    args = parser.parse_args(argv)
    settings = load_settings(args.project, args.config)

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

    parser.error(f"Unknown command: {args.command}")
    return 2


def configure_output_stream(stream: object) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    encoding = str(getattr(stream, "encoding", "") or "").lower()
    if callable(reconfigure) and encoding not in {"utf-8", "utf8"}:
        reconfigure(encoding="utf-8", errors="replace")
