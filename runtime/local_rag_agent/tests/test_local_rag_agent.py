import os
import tempfile
import unittest
from pathlib import Path

from local_rag_agent.agent import answer_question, build_extractive_answer
from local_rag_agent.chunking import chunk_markdown
from local_rag_agent.cli import configure_output_stream, demo_check, ingest_project
from local_rag_agent.config import Settings, load_settings
from local_rag_agent.index_store import read_index
from local_rag_agent.manifest import expand_manifest_entries, parse_manifest_entries
from local_rag_agent.regression import parse_regression_questions
from local_rag_agent.retrieval import rank_chunks
from local_rag_agent.server import render_chat_page, render_course_site_home


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

    def test_retrieval_prioritizes_fact_tables_over_intent_examples(self):
        chunks = [
            {
                "content": "推荐回答方式：师生会面时间为星期一 16:45-17:30pm。",
                "source": "knowledge_base/semester_specific/2026-spring/course_info/课程问法同义词与意图映射.md",
                "title": "回答示例",
                "chunk_id": "intent.md#0",
            },
            {
                "content": "问：老师的师生会面时间是什么时候？\n\n答：根据课程大纲资料，师生会面时间为星期一 16:45-17:30pm，地点为人文馆 508。",
                "source": "knowledge_base/semester_specific/2026-spring/course_info/课程关键事务问答核查表.md",
                "title": "师生会面时间",
                "chunk_id": "facts.md#0",
            },
        ]

        results = rank_chunks("老师的师生会面时间是什么时候？", chunks, top_k=2)

        self.assertEqual(results[0]["source"], "knowledge_base/semester_specific/2026-spring/course_info/课程关键事务问答核查表.md")


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

            self.assertIn("上课时间是星期三上午", response["answer"])
            self.assertEqual(response["sources"][0]["source"], "course.md")

    def test_extractive_answer_prefers_answer_line_from_top_chunk(self):
        chunks = [
            {
                "content": "## 上课时间和地点\n\n问：课程的上课时间和地点是什么？\n\n答：根据课程大纲资料，课程授课时间为星期三上午 1-2 节，即 8:00-9:35am。授课地点为 1 区教 5-304 或政管院实验室（老物理楼 2 层）。",
                "source": "course.md",
                "title": "上课时间和地点",
                "chunk_id": "course.md#0",
                "score": 9.0,
            }
        ]

        answer = build_extractive_answer("这门课的上课时间和地点是什么？", chunks)

        self.assertIn("根据课程大纲资料", answer)
        self.assertIn("星期三上午 1-2 节", answer)
        self.assertNotIn("来源：course.md", answer)

    def test_extractive_answer_refuses_complete_submission_request(self):
        chunks = [
            {
                "content": "智能体不能提供可直接提交的完整作业、完整论文或完整报告。",
                "source": "boundaries.md",
                "title": "回答边界",
                "chunk_id": "boundaries.md#0",
                "score": 9.0,
            }
        ]

        answer = build_extractive_answer("请直接帮我写完整论文。", chunks)

        self.assertIn("不能直接替你完成", answer)
        self.assertIn("结构", answer)

    def test_extractive_answer_handles_no_retrieval_results(self):
        answer = build_extractive_answer("不存在的问题", [])

        self.assertIn("根据目前知识库资料", answer)
        self.assertIn("未找到明确说明", answer)


class RegressionTests(unittest.TestCase):
    def test_parse_regression_table_extracts_question_column(self):
        markdown = "| 编号 | 问题 | 预期要点 |\n| --- | --- | --- |\n| C01 | 上课时间？ | 星期三 |\n"

        questions = parse_regression_questions(markdown)

        self.assertEqual(questions[0]["id"], "C01")
        self.assertEqual(questions[0]["question"], "上课时间？")
        self.assertEqual(questions[0]["expected"], "星期三")


class ServerPageTests(unittest.TestCase):
    def test_render_course_site_home_matches_server_entry_structure(self):
        page = render_course_site_home()

        self.assertIn("R 课程智能体与往届作品库", page)
        self.assertIn("课程智能体与往届作品查阅入口", page)
        self.assertIn('src="/chatbot"', page)
        self.assertIn("资料使用边界", page)

    def test_render_chat_page_contains_course_agent_demo_ui(self):
        page = render_chat_page("R 课程智能体（自建版）")

        self.assertIn("R 课程智能体（自建版）", page)
        self.assertIn("这门课的上课时间和地点是什么？", page)
        self.assertIn("/api/chat", page)
        self.assertIn("sources", page)


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

    def test_demo_check_reports_top_source_for_core_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("prompt", encoding="utf-8")
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"course.md#0","source":"course.md","title":"上课时间和地点","content":"答：星期三上午 1-2 节，1 区教 5-304。"}]}',
                encoding="utf-8",
            )

            report = demo_check(settings, dify_url=None)

            self.assertTrue(report["index_exists"])
            self.assertEqual(report["checks"][0]["top_source"], "course.md")
            self.assertIn("星期三", report["checks"][0]["answer"])


if __name__ == "__main__":
    unittest.main()
