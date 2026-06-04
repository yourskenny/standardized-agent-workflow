import json
import os
import tempfile
import unittest
from pathlib import Path

from local_rag_agent.agent import answer_question, build_extractive_answer, build_messages
from local_rag_agent.chunking import chunk_markdown
from local_rag_agent.cli import build_retrieval_query, chat_question, configure_output_stream, demo_check, ingest_project
from local_rag_agent.config import Settings, load_settings
from local_rag_agent.index_store import read_index
from local_rag_agent.intent import IntentRouter, load_intents
from local_rag_agent.manifest import expand_manifest_entries, parse_manifest_entries
from local_rag_agent.policy import PolicyGuard, load_policies
from local_rag_agent.regression import parse_regression_questions, run_regression
from local_rag_agent.retrieval import rank_chunks
from local_rag_agent.runtime import AgentRuntime
from local_rag_agent.server import render_chat_page, render_course_site_home
from local_rag_agent.tools import ToolProvider, load_tools
from local_rag_agent.types import AgentRequest, AgentResponse, AgentTrace, SourceReference
from local_rag_agent.workflow import WorkflowContext, WorkflowRegistry, build_retrieval_query as build_workflow_retrieval_query


def load_intents_from_inline(text: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "intents.toml"
        path.write_text(text, encoding="utf-8")
        return load_intents(path)


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

    def test_load_config_accepts_utf8_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge_base").mkdir()
            (root / "agent").mkdir()
            (root / "agent" / "system-prompt.md").write_text("prompt", encoding="utf-8")
            (root / "knowledge_base" / "_manifests").mkdir()
            (root / "knowledge_base" / "_manifests" / "current-upload-manifest.md").write_text(
                "- `knowledge_base/`\n",
                encoding="utf-8",
            )
            config_path = root / "agent.toml"
            config_path.write_text(
                '\ufeff[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n',
                encoding="utf-8",
            )

            settings = load_settings(root, config_path)

            self.assertEqual(settings.prompt_path, root.resolve() / "agent" / "system-prompt.md")

    def test_load_config_resolves_optional_intent_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent").mkdir()
            (root / "agent" / "system-prompt.md").write_text("prompt", encoding="utf-8")
            (root / "agent" / "intents.toml").write_text('[[intents]]\nid = "knowledge_qa"\n', encoding="utf-8")
            (root / "knowledge_base").mkdir()
            (root / "manifest.md").write_text("- `knowledge_base/`\n", encoding="utf-8")
            config_path = root / "agent.toml"
            config_path.write_text(
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "manifest.md"\n'
                '[runtime]\n'
                'intent_config = "agent/intents.toml"\n',
                encoding="utf-8",
            )

            settings = load_settings(root, config_path)

            self.assertEqual(settings.intent_config_path, root.resolve() / "agent" / "intents.toml")

    def test_load_config_resolves_runtime_extension_config_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent").mkdir()
            (root / "agent" / "system-prompt.md").write_text("prompt", encoding="utf-8")
            (root / "agent" / "intents.toml").write_text('[[intents]]\nid = "knowledge_qa"\n', encoding="utf-8")
            (root / "agent" / "workflows.toml").write_text('[[workflows]]\nid = "rag_qa"\n', encoding="utf-8")
            (root / "agent" / "policies.toml").write_text('[[policies]]\nid = "source_required"\n', encoding="utf-8")
            (root / "agent" / "tools.toml").write_text('[[tools]]\nid = "disabled_search"\n', encoding="utf-8")
            (root / "knowledge_base").mkdir()
            (root / "manifest.md").write_text("- `knowledge_base/`\n", encoding="utf-8")
            config_path = root / "agent.toml"
            config_path.write_text(
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "manifest.md"\n'
                '[runtime]\n'
                'default_intent = "knowledge_qa"\n'
                'default_workflow = "rag_qa"\n'
                'intent_config = "agent/intents.toml"\n'
                'workflow_config = "agent/workflows.toml"\n'
                'policy_config = "agent/policies.toml"\n'
                'tool_config = "agent/tools.toml"\n',
                encoding="utf-8",
            )

            settings = load_settings(root, config_path)

            self.assertEqual(settings.default_intent, "knowledge_qa")
            self.assertEqual(settings.default_workflow, "rag_qa")
            self.assertEqual(settings.workflow_config_path, root.resolve() / "agent" / "workflows.toml")
            self.assertEqual(settings.policy_config_path, root.resolve() / "agent" / "policies.toml")
            self.assertEqual(settings.tool_config_path, root.resolve() / "agent" / "tools.toml")

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

    def test_manifest_parser_ignores_do_not_upload_section(self):
        text = (
            "# Manifest\n\n"
            "## 应上传\n\n"
            "- `knowledge_base/public/`\n\n"
            "## 不应上传\n\n"
            "- `PROJECT_BRIEF.md`\n"
            "- `scripts/`\n"
        )

        entries = parse_manifest_entries(text)

        self.assertEqual(entries, ["knowledge_base/public/"])

    def test_template_agent_project_includes_structured_intents(self):
        template_root = Path("templates/agent-project")
        intent_config = template_root / "agent" / "intents.toml"

        intents = load_intents(intent_config)

        self.assertTrue(intent_config.exists())
        self.assertTrue(any(intent.id == "knowledge_qa" for intent in intents))
        self.assertTrue(any(intent.id == "complete_submission_request" for intent in intents))

    def test_template_agent_project_includes_structured_runtime_configs(self):
        template_root = Path("templates/agent-project")
        runtime_config = template_root / "runtime.toml"
        workflow_config = template_root / "agent" / "workflows.toml"
        policy_config = template_root / "agent" / "policies.toml"
        tool_config = template_root / "agent" / "tools.toml"

        settings = load_settings(template_root, runtime_config)
        policies = load_policies(policy_config)
        tools = load_tools(tool_config)

        self.assertTrue(runtime_config.exists())
        self.assertEqual(settings.intent_config_path, template_root.resolve() / "agent" / "intents.toml")
        self.assertEqual(settings.workflow_config_path, workflow_config.resolve())
        self.assertEqual(settings.policy_config_path, policy_config.resolve())
        self.assertEqual(settings.tool_config_path, tool_config.resolve())
        self.assertIn("[[workflows]]", workflow_config.read_text(encoding="utf-8"))
        self.assertTrue(any(policy.id == "academic_integrity" for policy in policies))
        self.assertTrue(any(policy.id == "source_required" for policy in policies))
        self.assertTrue(any(tool.id == "example_disabled_tool" for tool in tools))

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
            self.assertEqual(response["sources"][0]["content"], "上课时间是星期三上午。")

    def test_build_messages_keeps_recent_history_before_current_retrieval_context(self):
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
                    "content": "答：师生会面时间为星期一 16:45-17:30pm。",
                    "source": "course.md",
                    "title": "师生会面时间",
                    "chunk_id": "course.md#0",
                    "score": 9.0,
                }
            ]
            history = [
                {"role": "user", "content": "这门课的上课时间和地点是什么？"},
                {"role": "assistant", "content": "课程在星期三上午 1-2 节上课。"},
                {"role": "system", "content": "不要把这个放入对话历史。"},
            ]

            messages = build_messages(settings, "那老师什么时候可以答疑？", chunks, history=history)

            self.assertEqual(messages[0]["role"], "system")
            self.assertIn("系统提示词", messages[0]["content"])
            self.assertEqual(messages[1], history[0])
            self.assertEqual(messages[2], history[1])
            self.assertNotIn("不要把这个放入对话历史", "\n".join(message["content"] for message in messages))
            self.assertEqual(messages[-1]["role"], "user")
            self.assertIn("检索片段", messages[-1]["content"])
            self.assertIn("那老师什么时候可以答疑？", messages[-1]["content"])

    def test_agent_passes_history_to_model_messages(self):
        class FakeClient:
            def __init__(self):
                self.messages = []

            def chat(self, messages):
                self.messages = messages
                return "模型回答"

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
            client = FakeClient()

            response = answer_question(
                settings,
                "那地点呢？",
                [{"content": "答：地点为 1 区教 5-304。", "source": "course.md", "chunk_id": "course.md#0"}],
                model_client=client,
                history=[{"role": "user", "content": "这门课的上课时间是什么？"}],
            )

            self.assertEqual(response["answer"], "模型回答")
            self.assertIn("这门课的上课时间是什么？", "\n".join(message["content"] for message in client.messages))

    def test_runtime_run_preserves_current_rag_answer_shape(self):
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
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"course.md#0","source":"course.md","title":"课程事实","content":"答：上课时间是星期三上午。"}]}',
                encoding="utf-8",
            )
            runtime = AgentRuntime(settings)

            response = runtime.run(AgentRequest(message="上课时间？"))
            payload = response.to_dict()

            self.assertEqual(payload["intent"], "knowledge_qa")
            self.assertEqual(payload["workflow"], "rag_qa")
            self.assertEqual(payload["mode"], "extractive")
            self.assertIn("上课时间是星期三上午", payload["answer"])
            self.assertEqual(payload["sources"][0]["source"], "course.md")
            self.assertEqual(payload["trace"]["steps"][0]["name"], "route_intent")
            self.assertIn("start_workflow", [step["name"] for step in payload["trace"]["steps"]])
            self.assertEqual(payload["trace"]["steps"][1]["name"], "start_workflow")
            self.assertIn("run_retrieval", [step["name"] for step in payload["trace"]["steps"]])

    def test_runtime_uses_configured_intent_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("系统提示词", encoding="utf-8")
            intent_config = root / "intents.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "submission_boundary"\n'
                'workflow = "refusal_with_guidance"\n'
                'keywords = ["完整论文"]\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                intent_config_path=intent_config,
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"boundaries.md#0","source":"boundaries.md","title":"边界","content":"智能体不能提供可直接提交的完整论文。"}]}',
                encoding="utf-8",
            )
            runtime = AgentRuntime(settings)

            response = runtime.run(AgentRequest(message="请直接帮我写完整论文。"))
            payload = response.to_dict()

            self.assertEqual(payload["intent"], "submission_boundary")
            self.assertEqual(payload["workflow"], "refusal_with_guidance")
            self.assertEqual(payload["trace"]["intent"], "submission_boundary")
            self.assertEqual(payload["trace"]["steps"][0]["name"], "route_intent")

    def test_runtime_runs_refusal_workflow_without_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            intent_config = root / "intents.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "complete_submission_request"\n'
                'workflow = "refusal_with_guidance"\n'
                'keywords = ["完整论文"]\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "missing-index.json",
                intent_config_path=intent_config,
            )

            response = AgentRuntime(settings).run(AgentRequest("请直接帮我写完整论文。")).to_dict()

            self.assertEqual(response["workflow"], "refusal_with_guidance")
            self.assertEqual(response["mode"], "refusal")
            self.assertIn("不能直接替你完成", response["answer"])
            self.assertEqual(response["sources"], [])
            self.assertIn("apply_policy", [step["name"] for step in response["trace"]["steps"]])

    def test_runtime_records_source_required_policy_for_no_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text('{"chunks":[]}', encoding="utf-8")

            response = AgentRuntime(settings).run(AgentRequest("unknown fact?")).to_dict()
            policy_steps = [
                step for step in response["trace"]["steps"] if step["name"] == "apply_policy"
            ]

            self.assertEqual(response["mode"], "no_evidence")
            self.assertEqual(response["sources"], [])
            self.assertEqual(policy_steps[0]["detail"]["policy_id"], "source_required")
            self.assertFalse(policy_steps[0]["detail"]["allowed"])

    def test_runtime_runs_retrieval_debug_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            intent_config = root / "intents.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "debug_retrieval"\n'
                'workflow = "retrieval_debug"\n'
                'keywords = ["debug"]\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                intent_config_path=intent_config,
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                json.dumps(
                    {
                        "chunks": [
                            {
                                "chunk_id": "course.md#0",
                                "source": "course.md",
                                "title": "Course facts",
                                "content": "class time is Wednesday morning.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            response = AgentRuntime(settings).run(AgentRequest("debug retrieval: class time?")).to_dict()

            self.assertEqual(response["workflow"], "retrieval_debug")
            self.assertEqual(response["mode"], "retrieval_debug")
            self.assertIn("course.md", response["answer"])
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


class RuntimeTypeTests(unittest.TestCase):
    def test_agent_response_converts_to_legacy_payload(self):
        trace = AgentTrace(intent="knowledge_qa", workflow="rag_qa")
        trace.add_step("retrieve", {"top_k": 1})
        response = AgentResponse(
            answer="上课时间是星期三上午。",
            mode="extractive",
            intent="knowledge_qa",
            workflow="rag_qa",
            sources=[
                SourceReference(
                    source="course.md",
                    title="课程事实",
                    chunk_id="course.md#0",
                    score=3.0,
                    content="上课时间是星期三上午。",
                )
            ],
            trace=trace,
        )

        payload = response.to_dict()

        self.assertEqual(payload["answer"], "上课时间是星期三上午。")
        self.assertEqual(payload["mode"], "extractive")
        self.assertEqual(payload["intent"], "knowledge_qa")
        self.assertEqual(payload["workflow"], "rag_qa")
        self.assertEqual(payload["sources"][0]["source"], "course.md")
        self.assertEqual(payload["trace"]["steps"][0]["name"], "retrieve")

    def test_agent_request_keeps_history_and_metadata_defaults(self):
        request = AgentRequest(message="那地点呢？")

        self.assertEqual(request.message, "那地点呢？")
        self.assertEqual(request.history, [])
        self.assertEqual(request.metadata, {})


class IntentConfigTests(unittest.TestCase):
    def test_load_intents_reads_project_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intents.toml"
            path.write_text(
                '[[intents]]\n'
                'id = "policy_question"\n'
                'description = "Policy questions"\n'
                'examples = ["迟交政策是什么？"]\n'
                'keywords = ["迟交", "政策"]\n'
                'workflow = "rag_qa"\n'
                'risk_level = "high"\n',
                encoding="utf-8",
            )

            intents = load_intents(path)

            self.assertEqual(len(intents), 1)
            self.assertEqual(intents[0].id, "policy_question")
            self.assertEqual(intents[0].workflow, "rag_qa")
            self.assertEqual(intents[0].keywords, ["迟交", "政策"])

    def test_intent_router_selects_keyword_match(self):
        router = IntentRouter(
            [
                load_intents_from_inline(
                    '[[intents]]\n'
                    'id = "submission_boundary"\n'
                    'workflow = "refusal_with_guidance"\n'
                    'keywords = ["完整论文", "直接提交"]\n'
                )[0],
                load_intents_from_inline(
                    '[[intents]]\n'
                    'id = "knowledge_qa"\n'
                    'workflow = "rag_qa"\n'
                    'keywords = ["上课时间"]\n'
                )[0],
            ]
        )

        decision = router.route("请直接帮我写完整论文。")

        self.assertEqual(decision.intent.id, "submission_boundary")
        self.assertEqual(decision.intent.workflow, "refusal_with_guidance")
        self.assertEqual(decision.source, "config")

    def test_intent_router_falls_back_to_default(self):
        router = IntentRouter([])

        decision = router.route("普通问题")

        self.assertEqual(decision.intent.id, "knowledge_qa")
        self.assertEqual(decision.intent.workflow, "rag_qa")
        self.assertEqual(decision.source, "fallback")


class PolicyGuardTests(unittest.TestCase):
    def test_load_policies_reads_project_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policies.toml"
            path.write_text(
                '[[policies]]\n'
                'id = "academic_integrity"\n'
                'action = "refuse"\n'
                'reason = "complete_submission"\n'
                'message = "Cannot provide directly submitted work."\n'
                'keywords = ["complete paper"]\n',
                encoding="utf-8",
            )

            policies = load_policies(path)

            self.assertEqual(len(policies), 1)
            self.assertEqual(policies[0].id, "academic_integrity")
            self.assertEqual(policies[0].action, "refuse")
            self.assertEqual(policies[0].keywords, ["complete paper"])

    def test_policy_guard_refuses_configured_intent_policy(self):
        intent = load_intents_from_inline(
            '[[intents]]\n'
            'id = "complete_submission_request"\n'
            'workflow = "refusal_with_guidance"\n'
            'policy = "academic_integrity"\n'
        )[0]
        decision = IntentRouter([intent]).route("any message")

        policy = PolicyGuard.builtins().evaluate(
            message="please write my complete paper",
            intent_decision=decision,
            retrieved_chunks=[],
        )

        self.assertFalse(policy.allowed)
        self.assertEqual(policy.policy_id, "academic_integrity")
        self.assertEqual(policy.action, "refuse")
        self.assertTrue(policy.message)


class ToolProviderTests(unittest.TestCase):
    def test_load_tools_reads_project_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tools.toml"
            path.write_text(
                '[[tools]]\n'
                'id = "local_search"\n'
                'description = "Search local data."\n'
                'enabled = false\n',
                encoding="utf-8",
            )

            tools = load_tools(path)

            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].id, "local_search")
            self.assertFalse(tools[0].enabled)

    def test_disabled_tool_provider_returns_structured_failure(self):
        result = ToolProvider.disabled().call("local_search", {"query": "class time"})

        self.assertFalse(result.ok)
        self.assertEqual(result.tool_id, "local_search")
        self.assertEqual(result.output, {})
        self.assertIn("disabled", result.error)


class WorkflowPipelineTests(unittest.TestCase):
    def test_registry_contains_required_builtin_workflows(self):
        registry = WorkflowRegistry.builtins()

        self.assertTrue(registry.has("rag_qa"))
        self.assertTrue(registry.has("retrieval_debug"))
        self.assertTrue(registry.has("refusal_with_guidance"))

    def test_build_retrieval_query_keeps_recent_user_turns(self):
        query = build_workflow_retrieval_query(
            "那地点呢？",
            [
                {"role": "user", "content": "这门课的上课时间是什么？"},
                {"role": "assistant", "content": "星期三上午。"},
                {"role": "user", "content": "老师什么时候答疑？"},
            ],
        )

        self.assertIn("这门课的上课时间是什么？", query)
        self.assertIn("老师什么时候答疑？", query)
        self.assertIn("那地点呢？", query)
        self.assertNotIn("星期三上午", query)

    def test_rag_qa_workflow_produces_answer_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                json.dumps(
                    {
                        "chunks": [
                            {
                                "chunk_id": "course.md#0",
                                "source": "course.md",
                                "title": "Course facts",
                                "content": "Answer: class time is Wednesday morning.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            decision = IntentRouter([]).route("class time?")
            trace = AgentTrace(intent=decision.intent.id, workflow=decision.intent.workflow)
            context = WorkflowContext(settings, AgentRequest("class time?"), decision, trace)

            response = WorkflowRegistry.builtins().get("rag_qa").run(context)
            payload = response.to_dict()

            self.assertEqual(payload["workflow"], "rag_qa")
            self.assertIn("class time is Wednesday morning", payload["answer"])
            self.assertEqual(payload["sources"][0]["source"], "course.md")
            self.assertIn("run_retrieval", [step["name"] for step in payload["trace"]["steps"]])


class RegressionTests(unittest.TestCase):
    def test_parse_regression_table_extracts_question_column(self):
        markdown = "| 编号 | 问题 | 预期要点 |\n| --- | --- | --- |\n| C01 | 上课时间？ | 星期三 |\n"

        questions = parse_regression_questions(markdown)

        self.assertEqual(questions[0]["id"], "C01")
        self.assertEqual(questions[0]["question"], "上课时间？")
        self.assertEqual(questions[0]["expected"], "星期三")

    def test_run_regression_records_runtime_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "questions.md"
            output = root / "results.jsonl"
            questions.write_text(
                "| 编号 | 问题 | 预期要点 |\n"
                "| --- | --- | --- |\n"
                "| C01 | 上课时间？ | 星期三 |\n",
                encoding="utf-8",
            )

            count = run_regression(
                questions,
                output,
                lambda question: {
                    "answer": "星期三上午。",
                    "sources": [],
                    "mode": "extractive",
                    "intent": "knowledge_qa",
                    "workflow": "rag_qa",
                    "trace": {"steps": [{"name": "retrieve", "detail": {"source_count": 1}}]},
                },
            )

            record = json.loads(output.read_text(encoding="utf-8").strip())

            self.assertEqual(count, 1)
            self.assertEqual(record["intent"], "knowledge_qa")
            self.assertEqual(record["workflow"], "rag_qa")
            self.assertEqual(record["trace"]["steps"][0]["name"], "retrieve")


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
        self.assertIn("和 R 课程智能体 聊天", page)
        self.assertIn("/api/chat", page)
        self.assertIn("sources", page)
        self.assertNotIn("{{", page)
        self.assertNotIn("}}", page)
        self.assertIn("dify-shell", page)
        self.assertIn("brand-strip", page)
        self.assertIn("SELF-BUILT AGENT", page)
        self.assertIn("自主架构", page)
        self.assertNotIn("POWERED BY", page)
        self.assertNotIn("dify-word", page)
        self.assertIn("chat-canvas", page)
        self.assertIn("message-list", page)
        self.assertIn("composer-panel", page)
        self.assertIn("knowledge-popover", page)
        self.assertIn("openSourcePopover", page)
        self.assertIn("source.content", page)
        self.assertIn("line-height: 1.34", page)
        self.assertIn("margin: 0 0 6px", page)
        self.assertIn("position: sticky", page)
        self.assertIn("overflow-y: auto", page)
        self.assertIn("conversationHistory", page)
        self.assertIn("history: conversationHistory", page)


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

    def test_build_retrieval_query_includes_recent_user_turns_for_followups(self):
        query = build_retrieval_query(
            "那地点呢？",
            [
                {"role": "user", "content": "这门课的上课时间是什么？"},
                {"role": "assistant", "content": "星期三上午。"},
                {"role": "user", "content": "老师什么时候答疑？"},
            ],
        )

        self.assertIn("这门课的上课时间是什么？", query)
        self.assertIn("老师什么时候答疑？", query)
        self.assertIn("那地点呢？", query)
        self.assertNotIn("星期三上午", query)

    def test_chat_question_returns_runtime_metadata(self):
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
                '{"chunks":[{"chunk_id":"course.md#0","source":"course.md","title":"课程事实","content":"答：上课时间是星期三上午。"}]}',
                encoding="utf-8",
            )

            response = chat_question(settings, "上课时间？", model_client=None)

            self.assertEqual(response["intent"], "knowledge_qa")
            self.assertEqual(response["workflow"], "rag_qa")
            self.assertEqual(response["trace"]["steps"][0]["name"], "route_intent")
            self.assertEqual(response["trace"]["steps"][1]["name"], "start_workflow")
            self.assertIn("run_retrieval", [step["name"] for step in response["trace"]["steps"]])
            self.assertIn("上课时间是星期三上午", response["answer"])

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

    def test_demo_check_reports_runtime_config_and_workflow_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("prompt", encoding="utf-8")
            agent_dir = root / "agent"
            agent_dir.mkdir()
            intent_config = agent_dir / "intents.toml"
            workflow_config = agent_dir / "workflows.toml"
            policy_config = agent_dir / "policies.toml"
            tool_config = agent_dir / "tools.toml"
            intent_config.write_text('[[intents]]\nid = "knowledge_qa"\n', encoding="utf-8")
            workflow_config.write_text('[[workflows]]\nid = "rag_qa"\n', encoding="utf-8")
            policy_config.write_text('[[policies]]\nid = "source_required"\n', encoding="utf-8")
            tool_config.write_text('[[tools]]\nid = "example_disabled_tool"\n', encoding="utf-8")
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                intent_config_path=intent_config,
                workflow_config_path=workflow_config,
                policy_config_path=policy_config,
                tool_config_path=tool_config,
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text('{"chunks":[]}', encoding="utf-8")

            report = demo_check(settings, dify_url=None)

            self.assertTrue(report["runtime_configs"]["intent_config"]["exists"])
            self.assertTrue(report["runtime_configs"]["workflow_config"]["exists"])
            self.assertTrue(report["runtime_configs"]["policy_config"]["exists"])
            self.assertTrue(report["runtime_configs"]["tool_config"]["exists"])
            self.assertTrue(report["workflows"]["rag_qa"])
            self.assertTrue(report["workflows"]["retrieval_debug"])
            self.assertTrue(report["workflows"]["refusal_with_guidance"])


if __name__ == "__main__":
    unittest.main()
