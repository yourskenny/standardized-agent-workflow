import os
import tempfile
import unittest
from pathlib import Path

from local_rag_agent.agent import answer_question
from local_rag_agent.chunking import chunk_markdown
from local_rag_agent.cli import configure_output_stream, ingest_project
from local_rag_agent.config import Settings, load_settings
from local_rag_agent.index_store import read_index
from local_rag_agent.manifest import expand_manifest_entries, parse_manifest_entries
from local_rag_agent.regression import parse_regression_questions
from local_rag_agent.retrieval import rank_chunks


class ConfigAndManifestTests(unittest.TestCase):
    def test_load_config_resolves_project_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge_base").mkdir()
            (root / "dify").mkdir()
            (root / "dify" / "app-prompt.md").write_text("prompt", encoding="utf-8")
            (root / "manifest.md").write_text("- `knowledge_base/a.md`\n", encoding="utf-8")
            config_path = root / "agent.toml"
            config_path.write_text(
                '[project]\n'
                'prompt_path = "dify/app-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "manifest.md"\n',
                encoding="utf-8",
            )

            settings = load_settings(root, config_path)

            self.assertEqual(settings.project_root, root.resolve())
            self.assertEqual(settings.prompt_path, root.resolve() / "dify" / "app-prompt.md")
            self.assertEqual(settings.knowledge_root, root.resolve() / "knowledge_base")
            self.assertEqual(settings.index_path, root.resolve() / ".local_rag_agent" / "index.json")

    def test_load_config_rejects_paths_outside_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root.parent / "outside-prompt.md"
            outside.write_text("prompt", encoding="utf-8")
            config_path = root / "agent.toml"
            config_path.write_text(
                '[project]\n'
                f'prompt_path = "{outside}"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "manifest.md"\n',
                encoding="utf-8",
            )
            self.addCleanup(lambda: outside.unlink(missing_ok=True))

            with self.assertRaises(ValueError):
                load_settings(root, config_path)

    def test_manifest_parser_reads_bulleted_knowledge_paths(self):
        text = "- `knowledge_base/a.md`\n- `knowledge_base/dir/`\n- maintenance is not an entry\n"

        entries = parse_manifest_entries(text)

        self.assertEqual(entries, ["knowledge_base/a.md", "knowledge_base/dir/"])

    def test_expand_manifest_entries_returns_markdown_files_and_skips_excluded_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            kb = root / "knowledge_base"
            (kb / "course").mkdir(parents=True)
            (kb / "course" / "a.md").write_text("a", encoding="utf-8")
            (kb / "course" / "ignore.txt").write_text("ignore", encoding="utf-8")
            (kb / "_pre_ingestion").mkdir()
            (kb / "_pre_ingestion" / "draft.md").write_text("draft", encoding="utf-8")
            manifest = root / "manifest.md"
            manifest.write_text("- `knowledge_base/course/`\n- `knowledge_base/_pre_ingestion/`\n", encoding="utf-8")
            settings = Settings(
                project_root=root,
                prompt_path=root / "prompt.md",
                manifest_path=manifest,
                knowledge_root=kb,
                index_path=root / ".local_rag_agent" / "index.json",
            )

            files = expand_manifest_entries(settings)

            self.assertEqual(files, [kb / "course" / "a.md"])


class ChunkingAndRetrievalTests(unittest.TestCase):
    def test_chunk_markdown_preserves_source_metadata(self):
        chunks = chunk_markdown(Path("knowledge_base/course.md"), "# 标题\n\n上课时间是星期三上午。", 20, 4)

        self.assertEqual(chunks[0]["source"], "knowledge_base/course.md")
        self.assertEqual(chunks[0]["title"], "标题")
        self.assertIn("上课时间", chunks[0]["content"])

    def test_retrieval_ranks_matching_course_fact_first(self):
        chunks = [
            {
                "content": "上课时间是星期三上午 1-2 节。",
                "source": "a.md",
                "title": "课程事实",
                "chunk_id": "a.md#0",
            },
            {
                "content": "R Markdown 可以生成 HTML。",
                "source": "b.md",
                "title": "R Markdown",
                "chunk_id": "b.md#0",
            },
        ]

        results = rank_chunks("这门课周几上课？", chunks, top_k=1)

        self.assertEqual(results[0]["source"], "a.md")
        self.assertGreater(results[0]["score"], 0)


class AgentTests(unittest.TestCase):
    def test_agent_returns_retrieval_only_answer_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("系统提示词", encoding="utf-8")
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
            )
            chunks = [
                {
                    "content": "上课时间是星期三上午。",
                    "source": "course.md",
                    "title": "课程事实",
                    "chunk_id": "course.md#0",
                    "score": 3.0,
                }
            ]

            response = answer_question(settings, "上课时间？", chunks, model_client=None)

            self.assertIn("本地检索结果", response["answer"])
            self.assertEqual(response["sources"][0]["source"], "course.md")


class RegressionTests(unittest.TestCase):
    def test_parse_regression_table_extracts_question_column(self):
        markdown = "| 编号 | 问题 | 预期要点 |\n| --- | --- | --- |\n| C01 | 上课时间？ | 星期三 |\n"

        questions = parse_regression_questions(markdown)

        self.assertEqual(questions[0]["id"], "C01")
        self.assertEqual(questions[0]["question"], "上课时间？")
        self.assertEqual(questions[0]["expected"], "星期三")


class CliWorkflowTests(unittest.TestCase):
    def test_configure_output_stream_uses_utf8_with_replacement(self):
        class FakeStream:
            encoding = "cp936"

            def __init__(self):
                self.calls = []

            def reconfigure(self, **kwargs):
                self.calls.append(kwargs)

        stream = FakeStream()

        configure_output_stream(stream)

        self.assertEqual(stream.calls, [{"encoding": "utf-8", "errors": "replace"}])

    def test_ingest_project_writes_index_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "dify").mkdir()
            (root / "dify" / "app-prompt.md").write_text("prompt", encoding="utf-8")
            (root / "knowledge_base" / "course").mkdir(parents=True)
            (root / "knowledge_base" / "course" / "facts.md").write_text(
                "# 课程核心事实\n\n上课时间是星期三上午 1-2 节。",
                encoding="utf-8",
            )
            (root / "knowledge_base" / "_manifests").mkdir()
            (root / "knowledge_base" / "_manifests" / "current.md").write_text(
                "- `knowledge_base/course/`\n",
                encoding="utf-8",
            )
            config_path = root / "agent.toml"
            config_path.write_text(
                '[project]\n'
                'prompt_path = "dify/app-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current.md"\n'
                '[retrieval]\n'
                'chunk_size = 200\n'
                'top_k = 3\n',
                encoding="utf-8",
            )

            settings = load_settings(root, config_path)
            summary = ingest_project(settings)
            index = read_index(settings)

            self.assertEqual(summary["file_count"], 1)
            self.assertGreaterEqual(summary["chunk_count"], 1)
            self.assertIn("chunks", index)


if __name__ == "__main__":
    unittest.main()
