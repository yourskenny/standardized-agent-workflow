import contextlib
import io
import json
import os
import shutil
import tempfile
import threading
import tomllib
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from local_rag_agent.agent import answer_question, build_extractive_answer, build_messages
from local_rag_agent.chunking import chunk_markdown
from local_rag_agent.cli import (
    build_retrieval_query,
    chat_question,
    configure_output_stream,
    demo_check,
    ingest_project,
    main as cli_main,
    retrieve_question,
)
from local_rag_agent.config import Settings, load_settings
from local_rag_agent.index_store import read_index
from local_rag_agent.intent import IntentRouter, load_intent_tests, load_intents
from local_rag_agent.manifest import expand_manifest_entries, parse_manifest_entries
from local_rag_agent.policy import PolicyDefinition, PolicyGuard, load_policies
from local_rag_agent.ports import GeneratorProvider, RetrieverProvider
from local_rag_agent.regression import parse_regression_questions, run_regression, summarize_regression_report
from local_rag_agent.retrieval import rank_chunks
from local_rag_agent.runtime import AgentRuntime
from local_rag_agent.server import ServerHooks, make_handler, render_chat_page, render_workspace_home
from local_rag_agent.tools import ToolProvider, load_tools
from local_rag_agent.types import AgentRequest, AgentResponse, AgentTrace, SourceReference
from local_rag_agent.ui import load_ui_config
from local_rag_agent.validator import (
    ValidationIssue,
    ValidationResult,
    validate_project_config,
    validate_project_contract,
)
from local_rag_agent.workflow import (
    StepRegistry,
    WorkflowContext,
    WorkflowRegistry,
    build_retrieval_query as build_workflow_retrieval_query,
    load_workflows,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = REPO_ROOT / "runtime" / "local_rag_agent"
TEMPLATE_ROOT = REPO_ROOT / "templates" / "agent-project"


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
            (root / "agent" / "ui.toml").write_text('title = "Project Assistant"\n', encoding="utf-8")
            (root / "agent" / "models.toml").write_text('[[models]]\nid = "chat"\n', encoding="utf-8")
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
                'tool_config = "agent/tools.toml"\n'
                'ui_config = "agent/ui.toml"\n'
                'model_config = "agent/models.toml"\n',
                encoding="utf-8",
            )

            settings = load_settings(root, config_path)

            self.assertEqual(settings.default_intent, "knowledge_qa")
            self.assertEqual(settings.default_workflow, "rag_qa")
            self.assertEqual(settings.workflow_config_path, root.resolve() / "agent" / "workflows.toml")
            self.assertEqual(settings.policy_config_path, root.resolve() / "agent" / "policies.toml")
            self.assertEqual(settings.tool_config_path, root.resolve() / "agent" / "tools.toml")
            self.assertEqual(settings.ui_config_path, root.resolve() / "agent" / "ui.toml")
            self.assertEqual(settings.model_config_path, root.resolve() / "agent" / "models.toml")

    def test_load_config_records_schema_version_and_warns_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent").mkdir()
            (root / "agent" / "system-prompt.md").write_text("prompt", encoding="utf-8")
            (root / "knowledge_base").mkdir()
            (root / "manifest.md").write_text("- `knowledge_base/`\n", encoding="utf-8")
            config_path = root / "agent.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "manifest.md"\n'
                'unexpected = true\n',
                encoding="utf-8",
            )

            with self.assertWarnsRegex(UserWarning, "Unknown field"):
                settings = load_settings(root, config_path)

            self.assertEqual(settings.config_schema_versions["runtime"], "runtime.v1")

    def test_load_config_reads_server_safety_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "agent.toml"
            config_path.write_text(
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "manifest.md"\n'
                '[server]\n'
                'request_body_limit_bytes = 128\n'
                'timeout_seconds = 7\n'
                'auth_token = "secret-token"\n'
                'basic_auth_username = "agent"\n'
                'basic_auth_password = "pass"\n'
                'cors_allowlist = ["https://example.test"]\n',
                encoding="utf-8",
            )

            settings = load_settings(root, config_path)

            self.assertEqual(settings.server_request_body_limit_bytes, 128)
            self.assertEqual(settings.server_timeout_seconds, 7)
            self.assertEqual(settings.server_auth_token, "secret-token")
            self.assertEqual(settings.server_basic_auth_username, "agent")
            self.assertEqual(settings.server_basic_auth_password, "pass")
            self.assertEqual(settings.server_cors_allowlist, ["https://example.test"])

    def test_load_config_rejects_unsupported_schema_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "agent.toml"
            config_path.write_text('schema_version = "runtime.v9"\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported schema_version"):
                load_settings(root, config_path)

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
        template_root = TEMPLATE_ROOT
        intent_config = template_root / "agent" / "intents.toml"

        intents = load_intents(intent_config)

        self.assertTrue(intent_config.exists())
        self.assertEqual(intents[0].schema_version, "intent.v2")
        self.assertTrue(any(intent.id == "knowledge_qa" for intent in intents))
        self.assertTrue(any(intent.id == "complete_submission_request" for intent in intents))
        self.assertTrue(any(intent.requires_sources for intent in intents if intent.id == "knowledge_qa"))

    def test_template_agent_project_includes_structured_runtime_configs(self):
        template_root = TEMPLATE_ROOT
        runtime_config = template_root / "runtime.toml"
        workflow_config = template_root / "agent" / "workflows.toml"
        policy_config = template_root / "agent" / "policies.toml"
        tool_config = template_root / "agent" / "tools.toml"
        ui_config = template_root / "agent" / "ui.toml"

        settings = load_settings(template_root, runtime_config)
        policies = load_policies(policy_config)
        tools = load_tools(tool_config)
        ui = load_ui_config(ui_config)

        self.assertTrue(runtime_config.exists())
        self.assertEqual(settings.intent_config_path, template_root.resolve() / "agent" / "intents.toml")
        self.assertEqual(settings.workflow_config_path, workflow_config.resolve())
        self.assertEqual(settings.policy_config_path, policy_config.resolve())
        self.assertEqual(settings.tool_config_path, tool_config.resolve())
        self.assertEqual(settings.ui_config_path, ui_config.resolve())
        self.assertEqual(policies[0].schema_version, "policy.v2")
        self.assertEqual(tools[0].schema_version, "tool.v2")
        self.assertIn("[[workflows]]", workflow_config.read_text(encoding="utf-8"))
        self.assertEqual(load_workflows(workflow_config)[0].schema_version, "workflow.v2")
        self.assertTrue(any(policy.id == "academic_integrity" for policy in policies))
        self.assertTrue(any(policy.id == "source_required" for policy in policies))
        self.assertTrue(any(tool.id == "example_disabled_tool" for tool in tools))
        self.assertEqual(tools[0].adapter, "disabled")
        self.assertEqual(ui.title, "Local Agent")

    def test_template_v2_project_validate_ingest_chat_and_regression_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "agent-project"
            shutil.copytree(TEMPLATE_ROOT, root)
            config_path = root / "runtime.toml"
            questions_path = root / "examples" / "core-regression-questions.md"
            regression_output = root / ".local_rag_agent" / "smoke-results.jsonl"

            with contextlib.redirect_stdout(io.StringIO()):
                validate_exit = cli_main(["validate", "--project", str(root), "--config", str(config_path)])
                ingest_exit = cli_main(["ingest", "--project", str(root), "--config", str(config_path)])
                chat_exit = cli_main(
                    [
                        "chat",
                        "--project",
                        str(root),
                        "--config",
                        str(config_path),
                        "complete paper",
                    ]
                )
                regression_exit = cli_main(
                    [
                        "regression",
                        "--project",
                        str(root),
                        "--config",
                        str(config_path),
                        "--questions",
                        str(questions_path),
                        "--output",
                        str(regression_output),
                    ]
                )

            self.assertEqual(validate_exit, 0)
            self.assertEqual(ingest_exit, 0)
            self.assertEqual(chat_exit, 0)
            self.assertEqual(regression_exit, 0)
            self.assertTrue(regression_output.exists())

    def test_runtime_package_environment_and_container_artifacts_exist(self):
        runtime_root = RUNTIME_ROOT
        pyproject = runtime_root / "pyproject.toml"
        env_example = runtime_root / ".env.example"
        dockerfile = runtime_root / "Dockerfile"

        metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        env_text = env_example.read_text(encoding="utf-8")
        docker_text = dockerfile.read_text(encoding="utf-8")

        self.assertEqual(metadata["project"]["name"], "local-rag-agent")
        self.assertEqual(metadata["project"]["scripts"]["local-rag-agent"], "local_rag_agent.cli:main")
        self.assertIn("LOCAL_AGENT_API_KEY=", env_text)
        self.assertIn("LOCAL_AGENT_BASE_URL=", env_text)
        self.assertIn("LOCAL_AGENT_MODEL=", env_text)
        self.assertIn("python -m local_rag_agent serve", docker_text)

    def test_template_rag_workflow_can_stop_on_policy_decisions(self):
        workflows = load_workflows(TEMPLATE_ROOT / "agent" / "workflows.toml")
        rag_workflow = next(workflow for workflow in workflows if workflow.id == "rag_qa")

        self.assertIn("apply_policy", rag_workflow.steps)
        self.assertIn("build_policy_response", rag_workflow.steps)
        self.assertLess(rag_workflow.steps.index("apply_policy"), rag_workflow.steps.index("build_policy_response"))
        self.assertLess(rag_workflow.steps.index("build_policy_response"), rag_workflow.steps.index("generate_answer"))

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

        results = rank_chunks(
            "老师的师生会面时间是什么时候？",
            chunks,
            top_k=2,
            source_boosts=[("课程关键事务问答核查表", 18.0), ("课程问法同义词与意图映射", -8.0)],
        )

        self.assertEqual(results[0]["source"], "knowledge_base/semester_specific/2026-spring/course_info/课程关键事务问答核查表.md")

    def test_retrieval_has_no_domain_source_boost_without_config(self):
        chunks = [
            {
                "content": "shared answer",
                "source": "knowledge_base/semester_specific/2026-spring/course_info/课程关键事务问答核查表.md",
                "title": "",
                "chunk_id": "facts.md#0",
            },
            {"content": "shared answer", "source": "a.md", "title": "", "chunk_id": "a.md#0"},
        ]

        results = rank_chunks("shared answer", chunks, top_k=2)

        self.assertEqual(results[0]["source"], "a.md")


    def test_retrieval_source_boosts_can_come_from_project_config(self):
        chunks = [
            {"content": "shared answer", "source": "low.md", "title": "", "chunk_id": "low.md#0"},
            {"content": "shared answer", "source": "boosted.md", "title": "", "chunk_id": "boosted.md#0"},
        ]

        results = rank_chunks(
            "shared answer",
            chunks,
            top_k=2,
            source_boosts=[("boosted.md", 5.0)],
        )

        self.assertEqual(results[0]["source"], "boosted.md")

    def test_cli_retrieve_uses_configured_retriever_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            settings = Settings(
                project_root=root,
                prompt_path=root / "prompt.md",
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                retrieval_source_boosts=[("boosted.md", 5.0)],
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"a.md#0","source":"a.md","content":"shared answer"},'
                '{"chunk_id":"boosted.md#0","source":"boosted.md","content":"shared answer"}]}',
                encoding="utf-8",
            )

            results = retrieve_question(settings, "shared answer")

            self.assertEqual(results[0]["source"], "boosted.md")


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
            self.assertNotIn("课程事实", messages[0]["content"])

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
            self.assertEqual(payload["trace"]["config_versions"], {})
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

    def test_runtime_trace_records_all_loaded_config_schema_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            agent_dir = root / "agent"
            agent_dir.mkdir()
            intent_config = agent_dir / "intents.toml"
            workflow_config = agent_dir / "workflows.toml"
            policy_config = agent_dir / "policies.toml"
            tool_config = agent_dir / "tools.toml"
            intent_config.write_text(
                'schema_version = "intent.v1"\n'
                '[[intents]]\n'
                'id = "debug"\n'
                'workflow = "retrieval_debug"\n'
                'keywords = ["debug"]\n',
                encoding="utf-8",
            )
            workflow_config.write_text(
                'schema_version = "workflow.v1"\n'
                '[[workflows]]\n'
                'id = "retrieval_debug"\n'
                'steps = ["prepare_retrieval_query", "run_retrieval", "build_retrieval_debug_response"]\n',
                encoding="utf-8",
            )
            policy_config.write_text(
                'schema_version = "policy.v1"\n'
                '[[policies]]\n'
                'id = "source_required"\n'
                'action = "no_evidence"\n',
                encoding="utf-8",
            )
            tool_config.write_text(
                'schema_version = "tool.v1"\n'
                '[[tools]]\n'
                'id = "lookup"\n'
                'enabled = false\n',
                encoding="utf-8",
            )
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
                config_schema_versions={"runtime": "runtime.v1"},
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"facts.md#0","source":"facts.md","content":"debug fact"}]}',
                encoding="utf-8",
            )

            payload = AgentRuntime(settings).run(AgentRequest("debug")).to_dict()

            self.assertEqual(
                payload["trace"]["config_versions"],
                {
                    "runtime": "runtime.v1",
                    "intent": "intent.v1",
                    "workflow": "workflow.v1",
                    "policy": "policy.v1",
                    "tool": "tool.v1",
                },
            )

    def test_runtime_emits_trace_event_to_registered_sink_with_step_statuses(self):
        from local_rag_agent.components import ComponentRegistry

        class RecordingSink:
            def __init__(self):
                self.events = []

            def emit(self, event):
                self.events.append(event)

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
                config_schema_versions={"runtime": "runtime.v1"},
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"facts.md#0","source":"facts.md","content":"Answer: traced."}]}',
                encoding="utf-8",
            )
            sink = RecordingSink()
            components = ComponentRegistry.builtins()
            components.register_trace_sink("memory", sink)

            response = AgentRuntime(settings, components=components).run(
                AgentRequest("trace this", metadata={"request_id": "req-123"})
            )

            self.assertEqual(response.trace.request_id, "req-123")
            self.assertEqual(len(sink.events), 1)
            event = sink.events[0]
            self.assertEqual(event["request_id"], "req-123")
            self.assertEqual(event["intent"], "knowledge_qa")
            self.assertEqual(event["workflow"], "rag_qa")
            self.assertEqual(event["config_versions"], {"runtime": "runtime.v1"})
            self.assertEqual(event["steps"][0]["name"], "route_intent")
            self.assertEqual(event["steps"][0]["status"], "ok")

    def test_runtime_records_run_id_in_trace_and_optional_store(self):
        from local_rag_agent.stores.sqlite import SQLiteRunStore

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
                '{"chunks":[{"chunk_id":"facts.md#0","source":"facts.md","content":"Answer: stored."}]}',
                encoding="utf-8",
            )
            store = SQLiteRunStore.in_memory()
            self.addCleanup(store.close)

            response = AgentRuntime(settings, run_store=store).run(
                AgentRequest("store this", metadata={"request_id": "req-store", "thread_id": "thread-1"})
            )

            self.assertTrue(response.trace.run_id)
            self.assertEqual(response.trace.request_id, "req-store")
            run = store.get_run(response.trace.run_id)
            self.assertEqual(run["run_id"], response.trace.run_id)
            self.assertEqual(run["thread_id"], "thread-1")
            self.assertEqual(run["intent"], "knowledge_qa")
            self.assertEqual(run["workflow"], "rag_qa")
            self.assertEqual(run["status"], "running")
            self.assertEqual(run["metadata"], {"request_id": "req-store"})

    def test_runtime_writes_workflow_checkpoints_when_store_is_present(self):
        from local_rag_agent.stores.sqlite import SQLiteRunStore

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
                '{"chunks":[{"chunk_id":"facts.md#0","source":"facts.md","content":"Answer: checkpoint this."}]}',
                encoding="utf-8",
            )
            store = SQLiteRunStore.in_memory()
            self.addCleanup(store.close)

            response = AgentRuntime(settings, run_store=store).run(
                AgentRequest("checkpoint this", metadata={"run_id": "run-checkpoint"})
            )

            checkpoints = store.list_checkpoints("run-checkpoint")
            node_ids = [checkpoint["node_id"] for checkpoint in checkpoints]
            self.assertIn("prepare_retrieval_query", node_ids)
            self.assertIn("run_retrieval", node_ids)
            self.assertIn("build_response", node_ids)
            retrieval_checkpoint = next(
                checkpoint for checkpoint in checkpoints if checkpoint["node_id"] == "prepare_retrieval_query"
            )
            self.assertEqual(retrieval_checkpoint["state"]["retrieval_query"], "checkpoint this")
            self.assertEqual(retrieval_checkpoint["trace"]["run_id"], response.trace.run_id)

    def test_runtime_writes_structured_log_for_completed_trace(self):
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
                config_schema_versions={"runtime": "runtime.v1"},
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"facts.md#0","source":"facts.md","content":"Answer: logged."}]}',
                encoding="utf-8",
            )

            with self.assertLogs("local_rag_agent.runtime", level="INFO") as logs:
                AgentRuntime(settings).run(AgentRequest("log this", metadata={"request_id": "req-log"}))

            event = json.loads(logs.output[0].split("runtime_trace ", 1)[1])
            self.assertEqual(event["request_id"], "req-log")
            self.assertEqual(event["intent"], "knowledge_qa")
            self.assertEqual(event["workflow"], "rag_qa")
            self.assertEqual(event["config_versions"], {"runtime": "runtime.v1"})
            self.assertTrue(all("status" in step for step in event["steps"]))

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

    def test_runtime_uses_workflow_requires_sources_for_non_rag_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            intent_config = root / "intents.toml"
            workflow_config = root / "workflows.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "research"\n'
                'workflow = "research_qa"\n'
                'keywords = ["research"]\n',
                encoding="utf-8",
            )
            workflow_config.write_text(
                '[[workflows]]\n'
                'id = "research_qa"\n'
                'requires_sources = true\n'
                'steps = ["prepare_retrieval_query", "run_retrieval", "apply_policy", "build_policy_response", "generate_answer", "build_response"]\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                intent_config_path=intent_config,
                workflow_config_path=workflow_config,
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text('{"chunks":[]}', encoding="utf-8")

            response = AgentRuntime(settings).run(AgentRequest("research unknown fact")).to_dict()
            policy_steps = [
                step for step in response["trace"]["steps"] if step["name"] == "apply_policy"
            ]

            self.assertEqual(response["workflow"], "research_qa")
            self.assertEqual(response["mode"], "no_evidence")
            self.assertEqual(policy_steps[0]["detail"]["policy_id"], "source_required")

    def test_runtime_uses_intent_requires_sources_over_workflow_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            intent_config = root / "intents.toml"
            workflow_config = root / "workflows.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "research"\n'
                'workflow = "research_qa"\n'
                'keywords = ["research"]\n'
                'requires_sources = true\n',
                encoding="utf-8",
            )
            workflow_config.write_text(
                '[[workflows]]\n'
                'id = "research_qa"\n'
                'requires_sources = false\n'
                'steps = ["prepare_retrieval_query", "run_retrieval", "apply_policy", "build_policy_response", "generate_answer", "build_response"]\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                intent_config_path=intent_config,
                workflow_config_path=workflow_config,
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text('{"chunks":[]}', encoding="utf-8")

            response = AgentRuntime(settings).run(AgentRequest("research unknown fact")).to_dict()
            policy_steps = [
                step for step in response["trace"]["steps"] if step["name"] == "apply_policy"
            ]

            self.assertEqual(response["mode"], "no_evidence")
            self.assertEqual(policy_steps[0]["detail"]["policy_id"], "source_required")

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

    def test_runtime_raises_for_unknown_workflow_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("prompt", encoding="utf-8")
            intent_config = root / "intents.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "broken"\n'
                'workflow = "missing_workflow"\n'
                'keywords = ["broken"]\n',
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

            with self.assertRaisesRegex(KeyError, "Unknown workflow"):
                AgentRuntime(settings).run(AgentRequest("broken request"))

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

    def test_extractive_answer_does_not_duplicate_policy_refusal(self):
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

        self.assertIn("智能体不能提供", answer)
        self.assertNotIn("不能直接替你完成", answer)

    def test_extractive_answer_handles_no_retrieval_results(self):
        answer = build_extractive_answer("不存在的问题", [])

        self.assertIn("根据目前知识库资料", answer)
        self.assertIn("未找到明确说明", answer)
        self.assertNotIn("任课教师", answer)


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


class RunStoreTests(unittest.TestCase):
    def test_sqlite_run_store_creates_runs_and_checkpoints(self):
        from local_rag_agent.stores.sqlite import SQLiteRunStore

        store = SQLiteRunStore.in_memory()
        self.addCleanup(store.close)

        store.create_run(
            run_id="run-1",
            thread_id="thread-1",
            intent="knowledge_qa",
            workflow="rag_qa",
            status="running",
            metadata={"request_id": "req-1"},
        )
        store.write_checkpoint(
            run_id="run-1",
            node_id="run_retrieval",
            state={"retrieval_query": "hello"},
            trace={"steps": [{"name": "run_retrieval"}]},
        )

        self.assertEqual(
            store.get_run("run-1"),
            {
                "run_id": "run-1",
                "thread_id": "thread-1",
                "intent": "knowledge_qa",
                "workflow": "rag_qa",
                "status": "running",
                "metadata": {"request_id": "req-1"},
            },
        )
        checkpoints = store.list_checkpoints("run-1")
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["node_id"], "run_retrieval")
        self.assertEqual(checkpoints[0]["state"], {"retrieval_query": "hello"})
        self.assertEqual(checkpoints[0]["trace"], {"steps": [{"name": "run_retrieval"}]})


class IntentConfigTests(unittest.TestCase):
    def test_load_intents_reads_requires_tool_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intents.toml"
            path.write_text(
                'schema_version = "intent.v1"\n'
                '[[intents]]\n'
                'id = "tool_lookup"\n'
                'workflow = "tool_or_rag"\n'
                'requires_tool = true\n',
                encoding="utf-8",
            )

            intents = load_intents(path)

            self.assertTrue(intents[0].requires_tool)

    def test_load_intents_reads_project_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intents.toml"
            path.write_text(
                'schema_version = "intent.v1"\n'
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
            self.assertEqual(intents[0].schema_version, "intent.v1")
            self.assertEqual(intents[0].workflow, "rag_qa")
            self.assertEqual(intents[0].keywords, ["迟交", "政策"])

    def test_load_intents_rejects_bad_schema_and_warns_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intents.toml"
            path.write_text(
                'schema_version = "intent.v1"\n'
                'top_level_extra = true\n'
                '[[intents]]\n'
                'id = "knowledge_qa"\n'
                'workflow = "rag_qa"\n'
                'unexpected = "ignored"\n',
                encoding="utf-8",
            )

            with self.assertWarnsRegex(UserWarning, "Unknown field"):
                load_intents(path)

            path.write_text('schema_version = "intent.v9"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported schema_version"):
                load_intents(path)

    def test_load_intents_reads_v2_routing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intents.toml"
            path.write_text(
                'schema_version = "intent.v2"\n'
                '[[intents]]\n'
                'id = "knowledge_qa"\n'
                'workflow = "rag_qa"\n'
                'priority = 50\n'
                'confidence_threshold = 0.62\n'
                'keywords = ["政策"]\n'
                'negative_keywords = ["代写"]\n'
                'requires_sources = true\n'
                'knowledge_scopes = ["current", "policy"]\n',
                encoding="utf-8",
            )

            intents = load_intents(path)

            self.assertEqual(intents[0].schema_version, "intent.v2")
            self.assertEqual(intents[0].priority, 50)
            self.assertEqual(intents[0].confidence_threshold, 0.62)
            self.assertEqual(intents[0].negative_keywords, ["代写"])
            self.assertTrue(intents[0].requires_sources)
            self.assertEqual(intents[0].knowledge_scopes, ["current", "policy"])

    def test_load_intent_tests_reads_nested_v2_test_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intents.toml"
            path.write_text(
                'schema_version = "intent.v2"\n'
                '[[intents]]\n'
                'id = "knowledge_qa"\n'
                'workflow = "rag_qa"\n'
                'keywords = ["政策"]\n'
                '[[intents.tests]]\n'
                'input = "迟交政策是什么？"\n'
                'expected_intent = "knowledge_qa"\n',
                encoding="utf-8",
            )

            tests = load_intent_tests(path)

            self.assertEqual(len(tests), 1)
            self.assertEqual(tests[0].intent_id, "knowledge_qa")
            self.assertEqual(tests[0].input, "迟交政策是什么？")
            self.assertEqual(tests[0].expected_intent, "knowledge_qa")

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

    def test_intent_router_uses_priority_and_negative_keywords(self):
        router = IntentRouter(
            load_intents_from_inline(
                'schema_version = "intent.v2"\n'
                '[[intents]]\n'
                'id = "high_priority_boundary"\n'
                'workflow = "refusal_with_guidance"\n'
                'priority = 90\n'
                'keywords = ["policy"]\n'
                'negative_keywords = ["class"]\n'
                '[[intents]]\n'
                'id = "knowledge_qa"\n'
                'workflow = "rag_qa"\n'
                'priority = 10\n'
                'keywords = ["policy"]\n'
            )
        )

        high_priority = router.route("policy question")
        negative_excluded = router.route("class policy question")

        self.assertEqual(high_priority.intent.id, "high_priority_boundary")
        self.assertEqual(negative_excluded.intent.id, "knowledge_qa")

    def test_intent_router_honors_confidence_threshold(self):
        router = IntentRouter(
            load_intents_from_inline(
                'schema_version = "intent.v2"\n'
                '[[intents]]\n'
                'id = "strict_intent"\n'
                'workflow = "refusal_with_guidance"\n'
                'keywords = ["strict"]\n'
                'confidence_threshold = 0.95\n'
            )
        )

        decision = router.route("strict")

        self.assertEqual(decision.intent.id, "knowledge_qa")
        self.assertEqual(decision.source, "fallback")

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
                'schema_version = "policy.v1"\n'
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
            self.assertEqual(policies[0].schema_version, "policy.v1")
            self.assertEqual(policies[0].action, "refuse")
            self.assertEqual(policies[0].keywords, ["complete paper"])

    def test_load_policies_rejects_bad_schema_and_warns_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policies.toml"
            path.write_text(
                'schema_version = "policy.v1"\n'
                'extra = true\n'
                '[[policies]]\n'
                'id = "source_required"\n'
                'action = "no_evidence"\n'
                'unknown = "ignored"\n',
                encoding="utf-8",
            )

            with self.assertWarnsRegex(UserWarning, "Unknown field"):
                load_policies(path)

            path.write_text('schema_version = "policy.v9"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported schema_version"):
                load_policies(path)

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
                'schema_version = "tool.v1"\n'
                '[[tools]]\n'
                'id = "local_search"\n'
                'description = "Search local data."\n'
                'enabled = false\n',
                encoding="utf-8",
            )

            tools = load_tools(path)

            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].id, "local_search")
            self.assertEqual(tools[0].schema_version, "tool.v1")
            self.assertFalse(tools[0].enabled)
            self.assertEqual(tools[0].allowed_intents, [])

    def test_load_tools_reads_v2_adapter_contract_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tools.toml"
            path.write_text(
                'schema_version = "tool.v2"\n'
                '[[tools]]\n'
                'id = "lookup"\n'
                'description = "Lookup local records."\n'
                'enabled = true\n'
                'adapter = "mock"\n'
                'allowed_intents = ["tool_lookup"]\n'
                'risk_level = "medium"\n'
                'timeout_seconds = 4\n'
                'max_output_bytes = 100\n'
                'requires_approval = true\n'
                '[tools.input_mapping]\n'
                'query = "$message"\n'
                '[tools.input_schema]\n'
                'required = ["query"]\n'
                '[tools.input_schema.properties.query]\n'
                'type = "string"\n'
                '[tools.output_schema]\n'
                'required = ["answer"]\n'
                '[tools.output_schema.properties.answer]\n'
                'type = "string"\n',
                encoding="utf-8",
            )

            tools = load_tools(path)

            self.assertEqual(tools[0].schema_version, "tool.v2")
            self.assertEqual(tools[0].adapter, "mock")
            self.assertEqual(tools[0].provider, "mock")
            self.assertTrue(tools[0].requires_approval)
            self.assertEqual(tools[0].input_mapping, {"query": "$message"})
            self.assertEqual(tools[0].input_schema["required"], ["query"])
            self.assertEqual(tools[0].output_schema["required"], ["answer"])

    def test_load_tools_rejects_bad_schema_and_warns_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tools.toml"
            path.write_text(
                'schema_version = "tool.v1"\n'
                'top_level_extra = true\n'
                '[[tools]]\n'
                'id = "lookup"\n'
                'enabled = false\n'
                'unexpected = "ignored"\n',
                encoding="utf-8",
            )

            with self.assertWarnsRegex(UserWarning, "Unknown field"):
                load_tools(path)

            path.write_text('schema_version = "tool.v9"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported schema_version"):
                load_tools(path)

    def test_disabled_tool_provider_returns_structured_failure(self):
        result = ToolProvider.disabled().call("local_search", {"query": "class time"})

        self.assertFalse(result.ok)
        self.assertEqual(result.tool_id, "local_search")
        self.assertEqual(result.output, {})
        self.assertIn("disabled", result.error)

    def test_mock_tool_provider_enforces_allowed_intents_and_returns_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tools.toml"
            path.write_text(
                '[[tools]]\n'
                'id = "lookup"\n'
                'enabled = true\n'
                'provider = "mock"\n'
                'allowed_intents = ["tool_lookup"]\n'
                'timeout_seconds = 3\n'
                'max_output_bytes = 200\n'
                '[tools.mock_output]\n'
                'answer = "mocked result"\n',
                encoding="utf-8",
            )
            provider = ToolProvider.from_config(path)

            denied = provider.call("lookup", {"query": "x"}, intent_id="knowledge_qa")
            allowed = provider.call("lookup", {"query": "x"}, intent_id="tool_lookup")

            self.assertFalse(denied.ok)
            self.assertIn("not allowed", denied.error)
            self.assertTrue(allowed.ok)
            self.assertEqual(allowed.output["answer"], "mocked result")


class PolicyToolPortTests(unittest.TestCase):
    def test_policy_and_tool_ports_live_outside_keyword_and_mock_implementations(self):
        from local_rag_agent.adapters.policies import KeywordPolicyGuard
        from local_rag_agent.adapters.tools import ConfiguredToolProvider
        from local_rag_agent.policy import PolicyGuard as LegacyPolicyGuard
        from local_rag_agent.ports import PolicyPort, ToolPort
        from local_rag_agent.tools import ToolProvider as LegacyToolProvider

        self.assertEqual(PolicyPort.__module__, "local_rag_agent.ports")
        self.assertEqual(ToolPort.__module__, "local_rag_agent.ports")
        self.assertEqual(KeywordPolicyGuard.__module__, "local_rag_agent.adapters.policies")
        self.assertEqual(ConfiguredToolProvider.__module__, "local_rag_agent.adapters.tools")
        self.assertIs(LegacyPolicyGuard, KeywordPolicyGuard)
        self.assertIs(LegacyToolProvider, ConfiguredToolProvider)


class ValidatorContractTests(unittest.TestCase):
    def test_validation_result_reports_errors_and_warnings_as_payload(self):
        result = ValidationResult(
            errors=[
                ValidationIssue(
                    code="UNKNOWN_WORKFLOW",
                    path=Path("agent/intents.toml"),
                    detail="intent x references workflow y",
                )
            ],
            warnings=[
                ValidationIssue(
                    code="NO_INTENT_TESTS",
                    path=Path("agent/intents.toml"),
                    detail="intent knowledge_qa has no tests",
                )
            ],
        )

        payload = result.to_dict()

        self.assertFalse(result.ok)
        self.assertEqual(
            payload,
            {
                "ok": False,
                "errors": [
                    {
                        "code": "UNKNOWN_WORKFLOW",
                        "path": "agent/intents.toml",
                        "detail": "intent x references workflow y",
                    }
                ],
                "warnings": [
                    {
                        "code": "NO_INTENT_TESTS",
                        "path": "agent/intents.toml",
                        "detail": "intent knowledge_qa has no tests",
                    }
                ],
            },
        )

    def test_validate_project_contract_returns_structured_success_for_minimal_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            settings = Settings(
                project_root=root,
                prompt_path=root / "agent" / "system-prompt.md",
                manifest_path=root / "knowledge_base" / "_manifests" / "current-upload-manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
            )

            result = validate_project_contract(settings)

            self.assertTrue(result.ok)
            self.assertEqual(result.errors, [])
            self.assertEqual(result.warnings, [])

    def test_validate_project_config_collects_schema_version_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[runtime]\n'
                'intent_config = "agent/intents.toml"\n'
                'workflow_config = "agent/workflows.toml"\n',
                encoding="utf-8",
            )
            agent_dir = root / "agent"
            agent_dir.mkdir()
            (agent_dir / "intents.toml").write_text('schema_version = "intent.v9"\n', encoding="utf-8")
            (agent_dir / "workflows.toml").write_text('schema_version = "workflow.v9"\n', encoding="utf-8")

            result = validate_project_config(root, config_path)

            self.assertFalse(result.ok)
            self.assertEqual(
                [issue.code for issue in result.errors],
                ["UNSUPPORTED_SCHEMA_VERSION", "UNSUPPORTED_SCHEMA_VERSION"],
            )
            self.assertTrue(all("Unsupported schema_version" in issue.detail for issue in result.errors))

    def test_validate_project_config_collects_unknown_field_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                'top_level_extra = true\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[runtime]\n'
                'ui_config = "agent/ui.toml"\n',
                encoding="utf-8",
            )
            agent_dir = root / "agent"
            agent_dir.mkdir()
            (agent_dir / "ui.toml").write_text(
                'title = "Local Agent"\n'
                'extra_ui = true\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertTrue(result.ok)
            warning_details = [issue.detail for issue in result.warnings]
            self.assertTrue(any("top_level_extra" in detail for detail in warning_details))
            self.assertTrue(any("extra_ui" in detail for detail in warning_details))

    def test_validate_project_config_rejects_unknown_intent_workflow_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[runtime]\n'
                'intent_config = "agent/intents.toml"\n'
                'workflow_config = "agent/workflows.toml"\n'
                'policy_config = "agent/policies.toml"\n',
                encoding="utf-8",
            )
            agent_dir = root / "agent"
            agent_dir.mkdir()
            (agent_dir / "intents.toml").write_text(
                'schema_version = "intent.v1"\n'
                '[[intents]]\n'
                'id = "broken"\n'
                'workflow = "missing_workflow"\n'
                'policy = "missing_policy"\n',
                encoding="utf-8",
            )
            (agent_dir / "workflows.toml").write_text(
                'schema_version = "workflow.v1"\n'
                '[[workflows]]\n'
                'id = "rag_qa"\n'
                'steps = ["build_refusal_response"]\n',
                encoding="utf-8",
            )
            (agent_dir / "policies.toml").write_text(
                'schema_version = "policy.v1"\n'
                '[[policies]]\n'
                'id = "source_required"\n'
                'action = "no_evidence"\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertFalse(result.ok)
            self.assertEqual(
                [issue.code for issue in result.errors],
                ["UNKNOWN_WORKFLOW", "UNKNOWN_POLICY"],
            )
            self.assertTrue(any("missing_workflow" in issue.detail for issue in result.errors))
            self.assertTrue(any("missing_policy" in issue.detail for issue in result.errors))

    def test_validate_project_config_accepts_builtin_workflow_and_policy_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[runtime]\n'
                'intent_config = "agent/intents.toml"\n',
                encoding="utf-8",
            )
            agent_dir = root / "agent"
            agent_dir.mkdir()
            (agent_dir / "intents.toml").write_text(
                'schema_version = "intent.v1"\n'
                '[[intents]]\n'
                'id = "submission_boundary"\n'
                'workflow = "refusal_with_guidance"\n'
                'policy = "academic_integrity"\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertTrue(result.ok)

    def test_validate_project_config_reports_unknown_workflow_step_as_structured_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[runtime]\n'
                'workflow_config = "agent/workflows.toml"\n',
                encoding="utf-8",
            )
            agent_dir = root / "agent"
            agent_dir.mkdir()
            (agent_dir / "workflows.toml").write_text(
                'schema_version = "workflow.v1"\n'
                '[[workflows]]\n'
                'id = "broken"\n'
                'steps = ["missing_step"]\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertFalse(result.ok)
            self.assertEqual(result.errors[0].code, "UNKNOWN_WORKFLOW_STEP")
            self.assertIn("missing_step", result.errors[0].detail)

    def test_validate_project_config_accepts_plugin_terminal_step_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            agent_dir = root / "agent"
            plugin_dir = root / "agent_plugins"
            agent_dir.mkdir()
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
            (plugin_dir / "custom_steps.py").write_text(
                "from local_rag_agent.components import StepDefinition\n"
                "from local_rag_agent.types import AgentResponse\n"
                "\n"
                "def plugin_response(context):\n"
                "    context.response = AgentResponse(\n"
                "        answer='plugin response',\n"
                "        mode='plugin',\n"
                "        intent=context.intent_decision.intent.id,\n"
                "        workflow=context.intent_decision.intent.workflow,\n"
                "        sources=[],\n"
                "        trace=context.trace,\n"
                "    )\n"
                "\n"
                "def register(registry):\n"
                "    registry.register_step_definition(\n"
                "        StepDefinition(id='plugin.response', fn=plugin_response, terminal=True)\n"
                "    )\n",
                encoding="utf-8",
            )
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[runtime]\n'
                'workflow_config = "agent/workflows.toml"\n'
                '[plugins]\n'
                'modules = ["agent_plugins.custom_steps"]\n',
                encoding="utf-8",
            )
            (agent_dir / "workflows.toml").write_text(
                'schema_version = "workflow.v1"\n'
                '[[workflows]]\n'
                'id = "plugin_flow"\n'
                'steps = ["plugin.response"]\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertTrue(result.ok, result.to_dict())

    def test_validate_project_config_reports_graph_workflow_contract_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            agent_dir = root / "agent"
            agent_dir.mkdir()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[runtime]\n'
                'workflow_config = "agent/workflows.toml"\n',
                encoding="utf-8",
            )
            (agent_dir / "workflows.toml").write_text(
                'schema_version = "workflow.v3"\n'
                '[[workflows]]\n'
                'id = "broken_graph"\n'
                'type = "graph"\n'
                'start = "missing_start"\n'
                '[[workflows.nodes]]\n'
                'id = "route"\n'
                'step = "tool.select"\n'
                '[[workflows.nodes]]\n'
                'id = "done"\n'
                'step = "response.tool_result"\n'
                'terminal = true\n'
                '[[workflows.edges]]\n'
                'from = "route"\n'
                'to = "missing_node"\n'
                'condition = "unknown.condition"\n'
                '[[workflows.edges]]\n'
                'from = "missing_source"\n'
                'to = "done"\n'
                'condition = "default"\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            codes = [issue.code for issue in result.errors]
            self.assertIn("UNKNOWN_GRAPH_START", codes)
            self.assertIn("UNKNOWN_GRAPH_NODE", codes)
            self.assertIn("UNKNOWN_GRAPH_EDGE_TARGET", codes)
            self.assertIn("UNSUPPORTED_GRAPH_CONDITION", codes)

    def test_validate_project_config_requires_workflow_terminal_response_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[runtime]\n'
                'workflow_config = "agent/workflows.toml"\n',
                encoding="utf-8",
            )
            agent_dir = root / "agent"
            agent_dir.mkdir()
            (agent_dir / "workflows.toml").write_text(
                'schema_version = "workflow.v1"\n'
                '[[workflows]]\n'
                'id = "no_response"\n'
                'steps = ["prepare_retrieval_query", "run_retrieval"]\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertFalse(result.ok)
            self.assertEqual(result.errors[0].code, "NO_TERMINAL_RESPONSE_PATH")
            self.assertIn("no_response", result.errors[0].detail)

    def test_validate_project_config_rejects_terminal_step_not_in_workflow_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[runtime]\n'
                'workflow_config = "agent/workflows.toml"\n',
                encoding="utf-8",
            )
            agent_dir = root / "agent"
            agent_dir.mkdir()
            (agent_dir / "workflows.toml").write_text(
                'schema_version = "workflow.v2"\n'
                '[[workflows]]\n'
                'id = "bad_terminal"\n'
                'requires_sources = true\n'
                'terminal_steps = ["build_response"]\n'
                'steps = ["prepare_retrieval_query", "run_retrieval", "build_retrieval_debug_response"]\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertFalse(result.ok)
            self.assertEqual(result.errors[0].code, "UNKNOWN_TERMINAL_STEP")
            self.assertIn("build_response", result.errors[0].detail)

    def test_validate_project_config_rejects_unknown_runtime_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[retrieval]\n'
                'provider = "vector"\n'
                '[generation]\n'
                'provider = "unknown_model"\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertFalse(result.ok)
            self.assertEqual(
                [issue.code for issue in result.errors],
                ["UNKNOWN_RETRIEVER_PROVIDER", "UNKNOWN_GENERATOR_PROVIDER"],
            )
            self.assertTrue(any("vector" in issue.detail for issue in result.errors))
            self.assertTrue(any("unknown_model" in issue.detail for issue in result.errors))

    def test_validate_project_config_accepts_explicit_extractive_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[generation]\n'
                'provider = "extractive"\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertTrue(result.ok)

    def test_validate_project_config_accepts_retrieval_v2_providers(self):
        for provider in ("sqlite_fts", "hybrid"):
            with self.subTest(provider=provider):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp).resolve()
                    config_path = root / "runtime.toml"
                    config_path.write_text(
                        'schema_version = "runtime.v1"\n'
                        '[project]\n'
                        'prompt_path = "agent/system-prompt.md"\n'
                        'knowledge_root = "knowledge_base"\n'
                        'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                        '[retrieval]\n'
                        f'provider = "{provider}"\n',
                        encoding="utf-8",
                    )

                    result = validate_project_config(root, config_path)

                    self.assertTrue(result.ok, result.to_dict())

    def test_validate_project_config_rejects_unknown_generation_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[generation]\n'
                'provider = "openai_compatible"\n'
                'fallback = "retry_forever"\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertFalse(result.ok)
            self.assertEqual(result.errors[0].code, "UNKNOWN_GENERATION_FALLBACK")
            self.assertIn("retry_forever", result.errors[0].detail)

    def test_validate_project_config_rejects_enabled_tool_unknown_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[runtime]\n'
                'tool_config = "agent/tools.toml"\n',
                encoding="utf-8",
            )
            agent_dir = root / "agent"
            agent_dir.mkdir()
            (agent_dir / "tools.toml").write_text(
                'schema_version = "tool.v1"\n'
                '[[tools]]\n'
                'id = "crm_lookup"\n'
                'enabled = true\n'
                'provider = "crm"\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertFalse(result.ok)
            self.assertEqual(result.errors[0].code, "UNKNOWN_TOOL_PROVIDER")
            self.assertIn("crm_lookup", result.errors[0].detail)
            self.assertIn("crm", result.errors[0].detail)

    def test_validate_template_project_runtime_providers_are_known(self):
        result = validate_project_config(TEMPLATE_ROOT, TEMPLATE_ROOT / "runtime.toml")

        self.assertTrue(result.ok)

    def test_validate_project_config_reports_failed_intent_contract_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[runtime]\n'
                'intent_config = "agent/intents.toml"\n',
                encoding="utf-8",
            )
            agent_dir = root / "agent"
            agent_dir.mkdir()
            (agent_dir / "intents.toml").write_text(
                'schema_version = "intent.v2"\n'
                '[[intents]]\n'
                'id = "high_priority_boundary"\n'
                'workflow = "refusal_with_guidance"\n'
                'priority = 90\n'
                'keywords = ["policy"]\n'
                '[[intents]]\n'
                'id = "knowledge_qa"\n'
                'workflow = "rag_qa"\n'
                'priority = 10\n'
                'keywords = ["policy"]\n'
                '[[intents.tests]]\n'
                'input = "policy question"\n'
                'expected_intent = "knowledge_qa"\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertFalse(result.ok)
            self.assertEqual(result.errors[0].code, "INTENT_TEST_FAILED")
            self.assertIn("expected knowledge_qa", result.errors[0].detail)

    def test_validate_project_config_rejects_empty_manifest_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            manifest = root / "knowledge_base" / "_manifests" / "current-upload-manifest.md"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("## 应上传\n\n- `knowledge_base/_pre_ingestion/`\n", encoding="utf-8")
            (root / "knowledge_base" / "_pre_ingestion").mkdir()
            (root / "knowledge_base" / "_pre_ingestion" / "draft.md").write_text("draft", encoding="utf-8")
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertFalse(result.ok)
            self.assertEqual(result.errors[0].code, "MANIFEST_EMPTY")

    def test_validate_project_config_reports_paths_outside_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            outside = root.parent / "outside-prompt.md"
            outside.write_text("prompt", encoding="utf-8")
            self.addCleanup(lambda: outside.unlink(missing_ok=True))
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                f'prompt_path = "{outside.as_posix()}"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertFalse(result.ok)
            self.assertEqual(result.errors[0].code, "PATH_OUTSIDE_PROJECT")

    def test_validate_project_config_reports_index_manifest_mismatch_when_index_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            knowledge_file = root / "knowledge_base" / "facts.md"
            knowledge_file.parent.mkdir(parents=True)
            knowledge_file.write_text("# Facts\n\nSupported fact.", encoding="utf-8")
            manifest = root / "knowledge_base" / "_manifests" / "current-upload-manifest.md"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("- `knowledge_base/facts.md`\n", encoding="utf-8")
            index_path = root / ".local_rag_agent" / "index.json"
            index_path.parent.mkdir()
            index_path.write_text(
                '{"chunks":[{"chunk_id":"other.md#0","source":"knowledge_base/other.md","content":"stale"}]}',
                encoding="utf-8",
            )
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                'index_path = ".local_rag_agent/index.json"\n',
                encoding="utf-8",
            )

            result = validate_project_config(root, config_path)

            self.assertFalse(result.ok)
            self.assertEqual(result.errors[0].code, "INDEX_MANIFEST_MISMATCH")
            self.assertIn("knowledge_base/facts.md", result.errors[0].detail)


class ComponentPortTests(unittest.TestCase):
    def test_retriever_and_generator_adapters_live_outside_ports_with_compat_exports(self):
        from local_rag_agent.adapters.generators import ExtractiveGenerator, OpenAICompatibleGenerator
        from local_rag_agent.adapters.retrievers import LexicalRetriever
        from local_rag_agent.ports import (
            ExtractiveGenerator as LegacyExtractiveGenerator,
            LexicalRetriever as LegacyLexicalRetriever,
            RagGenerator,
            RetrieverPort,
        )

        self.assertEqual(RetrieverPort.__module__, "local_rag_agent.ports")
        self.assertEqual(LexicalRetriever.__module__, "local_rag_agent.adapters.retrievers")
        self.assertEqual(ExtractiveGenerator.__module__, "local_rag_agent.adapters.generators")
        self.assertEqual(OpenAICompatibleGenerator.__module__, "local_rag_agent.adapters.generators")
        self.assertIs(LegacyLexicalRetriever, LexicalRetriever)
        self.assertIs(LegacyExtractiveGenerator, ExtractiveGenerator)
        self.assertIs(RagGenerator, OpenAICompatibleGenerator)

    def test_retriever_provider_uses_configured_lexical_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            settings = Settings(
                project_root=root,
                prompt_path=root / "prompt.md",
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                retrieval_source_boosts=[("boosted.md", 4.0)],
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"a.md#0","source":"a.md","content":"shared answer"},'
                '{"chunk_id":"boosted.md#0","source":"boosted.md","content":"shared answer"}]}',
                encoding="utf-8",
            )

            chunks = RetrieverProvider.from_settings(settings).retrieve(settings, "shared answer")

            self.assertEqual(chunks[0]["source"], "boosted.md")

    def test_sqlite_fts_retriever_returns_matching_chunks(self):
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            db_path = root / "chunks.db"
            connection = sqlite3.connect(db_path)
            try:
                connection.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id, source, title, content)")
                connection.execute(
                    "INSERT INTO chunks_fts (chunk_id, source, title, content) VALUES (?, ?, ?, ?)",
                    ("policy.md#0", "policy.md", "Refund policy", "Answer: refunds require approval."),
                )
                connection.execute(
                    "INSERT INTO chunks_fts (chunk_id, source, title, content) VALUES (?, ?, ?, ?)",
                    ("other.md#0", "other.md", "Other", "Unrelated content."),
                )
                connection.commit()
            finally:
                connection.close()
            settings = Settings(
                project_root=root,
                prompt_path=root / "prompt.md",
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=db_path,
                retrieval_provider="sqlite_fts",
                top_k=1,
            )

            chunks = RetrieverProvider.from_settings(settings).retrieve(settings, "refund approval")

            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["source"], "policy.md")
            self.assertIn("refunds require approval", chunks[0]["content"])

    def test_generator_provider_returns_extractable_generated_answer(self):
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
            chunks = [{"chunk_id": "facts.md#0", "source": "facts.md", "content": "Answer: generated through port."}]

            answer = GeneratorProvider.from_settings(settings).generate(settings, "question?", chunks, model_client=None)

            self.assertEqual(answer.mode, "extractive")
            self.assertIn("generated through port", answer.answer)
            self.assertEqual(answer.sources[0]["source"], "facts.md")

    def test_generator_provider_uses_openai_compatible_model_client(self):
        class FakeModelClient:
            def __init__(self):
                self.called = False

            def chat(self, messages):
                self.called = True
                return "model generated answer"

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
                generation_provider="openai_compatible",
            )
            chunks = [{"chunk_id": "facts.md#0", "source": "facts.md", "content": "Answer: source answer."}]
            client = FakeModelClient()

            answer = GeneratorProvider.from_settings(settings).generate(settings, "question?", chunks, model_client=client)

            self.assertTrue(client.called)
            self.assertEqual(answer.mode, "model")
            self.assertEqual(answer.answer, "model generated answer")

    def test_generator_provider_extractive_provider_ignores_model_client(self):
        class FakeModelClient:
            def __init__(self):
                self.called = False

            def chat(self, messages):
                self.called = True
                return "model generated answer"

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
                generation_provider="extractive",
            )
            chunks = [{"chunk_id": "facts.md#0", "source": "facts.md", "content": "Answer: explicit fallback answer."}]
            client = FakeModelClient()

            answer = GeneratorProvider.from_settings(settings).generate(settings, "question?", chunks, model_client=client)

            self.assertFalse(client.called)
            self.assertEqual(answer.mode, "extractive")
            self.assertIn("explicit fallback answer", answer.answer)

    def test_generator_provider_uses_configured_extractive_fallback_without_model_client(self):
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
                generation_provider="openai_compatible",
                generation_fallback="extractive",
            )
            chunks = [{"chunk_id": "facts.md#0", "source": "facts.md", "content": "Answer: fallback answer."}]

            answer = GeneratorProvider.from_settings(settings).generate(settings, "question?", chunks, model_client=None)

            self.assertEqual(answer.mode, "extractive")
            self.assertIn("fallback answer", answer.answer)

    def test_load_models_reads_model_provider_contract(self):
        from local_rag_agent.models import load_models

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.toml"
            path.write_text(
                'schema_version = "models.v1"\n'
                '[[models]]\n'
                'id = "primary"\n'
                'provider = "openai_compatible"\n'
                'model = "gpt-4.1-mini"\n'
                'base_url = "https://models.example/v1"\n'
                'api_key_env = "LOCAL_RAG_TEST_MODEL_KEY"\n'
                'fallback = "extractive"\n',
                encoding="utf-8",
            )

            models = load_models(path)

            self.assertEqual(models[0].id, "primary")
            self.assertEqual(models[0].provider, "openai_compatible")
            self.assertEqual(models[0].model, "gpt-4.1-mini")
            self.assertEqual(models[0].base_url, "https://models.example/v1")
            self.assertEqual(models[0].api_key_env, "LOCAL_RAG_TEST_MODEL_KEY")
            self.assertEqual(models[0].fallback, "extractive")
            self.assertEqual(models[0].schema_version, "models.v1")

    def test_generator_provider_falls_back_to_extractive_when_model_api_key_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            models_path = root / "models.toml"
            models_path.write_text(
                'schema_version = "models.v1"\n'
                '[[models]]\n'
                'id = "primary"\n'
                'provider = "openai_compatible"\n'
                'model = "gpt-4.1-mini"\n'
                'base_url = "https://models.example/v1"\n'
                'api_key_env = "LOCAL_RAG_TEST_MISSING_KEY"\n'
                'fallback = "extractive"\n',
                encoding="utf-8",
            )
            os.environ.pop("LOCAL_RAG_TEST_MISSING_KEY", None)
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                generation_provider="openai_compatible",
                generation_fallback="extractive",
                model_config_path=models_path,
            )
            chunks = [{"chunk_id": "facts.md#0", "source": "facts.md", "content": "Answer: no key fallback."}]

            answer = GeneratorProvider.from_settings(settings).generate(settings, "question?", chunks, model_client=None)

            self.assertEqual(answer.mode, "extractive")
            self.assertIn("no key fallback", answer.answer)
            self.assertEqual(answer.metadata["provider"], "openai_compatible")
            self.assertEqual(answer.metadata["model"], "gpt-4.1-mini")
            self.assertEqual(answer.metadata["api_key_env"], "LOCAL_RAG_TEST_MISSING_KEY")
            self.assertEqual(answer.metadata["credential_status"], "missing")

    def test_generator_provider_keeps_no_client_path_extractive_when_model_key_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            models_path = root / "models.toml"
            models_path.write_text(
                'schema_version = "models.v1"\n'
                '[[models]]\n'
                'id = "primary"\n'
                'provider = "openai_compatible"\n'
                'model = "gpt-4.1-mini"\n'
                'base_url = "http://127.0.0.1:9/v1"\n'
                'api_key_env = "LOCAL_RAG_TEST_MODEL_KEY"\n'
                'fallback = "extractive"\n',
                encoding="utf-8",
            )
            old_secret = os.environ.get("LOCAL_RAG_TEST_MODEL_KEY")
            os.environ["LOCAL_RAG_TEST_MODEL_KEY"] = "super-secret-token"
            def restore_secret():
                if old_secret is None:
                    os.environ.pop("LOCAL_RAG_TEST_MODEL_KEY", None)
                else:
                    os.environ["LOCAL_RAG_TEST_MODEL_KEY"] = old_secret

            self.addCleanup(restore_secret)
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                generation_provider="openai_compatible",
                generation_fallback="extractive",
                model_config_path=models_path,
            )
            chunks = [{"chunk_id": "facts.md#0", "source": "facts.md", "content": "Answer: explicit client only."}]

            answer = GeneratorProvider.from_settings(settings).generate(settings, "question?", chunks, model_client=None)

            self.assertEqual(answer.mode, "extractive")
            self.assertIn("explicit client only", answer.answer)
            self.assertEqual(answer.metadata["credential_status"], "present")

    def test_runtime_trace_records_model_resolution_without_secret_values(self):
        class FakeModelClient:
            def chat(self, messages):
                return "model answer"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            models_path = root / "models.toml"
            models_path.write_text(
                'schema_version = "models.v1"\n'
                '[[models]]\n'
                'id = "primary"\n'
                'provider = "openai_compatible"\n'
                'model = "gpt-4.1-mini"\n'
                'base_url = "https://models.example/v1"\n'
                'api_key_env = "LOCAL_RAG_TEST_MODEL_KEY"\n'
                'fallback = "extractive"\n',
                encoding="utf-8",
            )
            old_secret = os.environ.get("LOCAL_RAG_TEST_MODEL_KEY")
            os.environ["LOCAL_RAG_TEST_MODEL_KEY"] = "super-secret-token"
            def restore_secret():
                if old_secret is None:
                    os.environ.pop("LOCAL_RAG_TEST_MODEL_KEY", None)
                else:
                    os.environ["LOCAL_RAG_TEST_MODEL_KEY"] = old_secret

            self.addCleanup(restore_secret)
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                generation_provider="openai_compatible",
                generation_fallback="extractive",
                model_config_path=models_path,
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"facts.md#0","source":"facts.md","content":"Answer: source answer."}]}',
                encoding="utf-8",
            )

            response = AgentRuntime(settings, model_client=FakeModelClient()).run(AgentRequest("source answer"))

            payload = response.to_dict()
            generate_step = next(step for step in payload["trace"]["steps"] if step["name"] == "generate_answer")
            self.assertEqual(generate_step["detail"]["provider"], "openai_compatible")
            self.assertEqual(generate_step["detail"]["model"], "gpt-4.1-mini")
            self.assertEqual(generate_step["detail"]["base_url"], "https://models.example/v1")
            self.assertEqual(generate_step["detail"]["api_key_env"], "LOCAL_RAG_TEST_MODEL_KEY")
            self.assertNotIn("super-secret-token", json.dumps(payload, ensure_ascii=False))

    def test_prompt_compiler_orders_stable_context_and_volatile_blocks(self):
        from local_rag_agent.prompt.compiler import compile_prompt

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

            compiled = compile_prompt(
                settings,
                "What now?",
                [{"chunk_id": "facts.md#0", "source": "facts.md", "title": "Facts", "content": "Answer: use facts."}],
            )

            self.assertEqual([block.type for block in compiled.blocks], ["stable", "context", "volatile"])
            self.assertEqual(compiled.blocks[0].source, str(prompt))
            self.assertEqual(compiled.blocks[1].source, "facts.md")
            self.assertEqual(compiled.blocks[2].source, "request.message")

    def test_prompt_budget_trims_context_blocks_deterministically(self):
        from local_rag_agent.prompt.blocks import PromptBlock
        from local_rag_agent.prompt.budget import trim_blocks

        blocks = [
            PromptBlock(source="system", type="stable", text="one two", token_count=2),
            PromptBlock(source="ctx-1", type="context", text="a b c", token_count=3),
            PromptBlock(source="ctx-2", type="context", text="d e f", token_count=3),
            PromptBlock(source="request", type="volatile", text="question", token_count=1),
        ]

        trimmed = trim_blocks(blocks, max_tokens=6)

        self.assertEqual([block.source for block in trimmed], ["system", "ctx-1", "request"])

    def test_prompt_budget_trims_repeated_context_sources_by_position(self):
        from local_rag_agent.prompt.blocks import PromptBlock
        from local_rag_agent.prompt.budget import trim_blocks

        blocks = [
            PromptBlock(source="system", type="stable", text="one two", token_count=2),
            PromptBlock(source="facts.md", type="context", text="a b c", token_count=3),
            PromptBlock(source="facts.md", type="context", text="d e f", token_count=3),
            PromptBlock(source="request", type="volatile", text="question", token_count=1),
        ]

        trimmed = trim_blocks(blocks, max_tokens=6)

        self.assertEqual([block.text for block in trimmed], ["one two", "a b c", "question"])

    def test_runtime_trace_records_prompt_block_source_and_type_for_model_generation(self):
        class FakeModelClient:
            def __init__(self):
                self.messages = []

            def chat(self, messages):
                self.messages = messages
                return "model answer"

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
                generation_provider="openai_compatible",
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"facts.md#0","source":"facts.md","title":"Facts","content":"Answer: source answer."}]}',
                encoding="utf-8",
            )
            client = FakeModelClient()

            response = AgentRuntime(settings, model_client=client).run(AgentRequest("source answer"))

            payload = response.to_dict()
            generate_step = next(step for step in payload["trace"]["steps"] if step["name"] == "generate_answer")
            prompt_blocks = generate_step["detail"]["prompt_blocks"]
            self.assertEqual(prompt_blocks[0]["type"], "stable")
            self.assertTrue(any(block["source"] == "facts.md" and block["type"] == "context" for block in prompt_blocks))
            self.assertEqual(prompt_blocks[-1]["source"], "request.message")
            self.assertIn("source answer", client.messages[-1]["content"])


class WorkflowPipelineTests(unittest.TestCase):
    def test_workflow_facade_reexports_split_definition_runner_registry_and_steps_modules(self):
        from local_rag_agent.workflow import (
            StepRegistry as LegacyStepRegistry,
            WorkflowContext as LegacyWorkflowContext,
            WorkflowDefinition as LegacyWorkflowDefinition,
            WorkflowPipeline as LegacyWorkflowPipeline,
            WorkflowRegistry as LegacyWorkflowRegistry,
            build_response as legacy_build_response,
            load_workflows as legacy_load_workflows,
            prepare_retrieval_query as legacy_prepare_retrieval_query,
        )
        from local_rag_agent.workflows.definitions import WorkflowDefinition, load_workflows
        from local_rag_agent.workflows.registry import WorkflowRegistry
        from local_rag_agent.workflows.runner import WorkflowContext, WorkflowPipeline
        from local_rag_agent.workflows.steps import StepRegistry, build_response, prepare_retrieval_query

        self.assertEqual(WorkflowDefinition.__module__, "local_rag_agent.workflows.definitions")
        self.assertEqual(load_workflows.__module__, "local_rag_agent.workflows.definitions")
        self.assertEqual(StepRegistry.__module__, "local_rag_agent.workflows.steps")
        self.assertEqual(prepare_retrieval_query.__module__, "local_rag_agent.workflows.steps")
        self.assertEqual(build_response.__module__, "local_rag_agent.workflows.steps")
        self.assertEqual(WorkflowContext.__module__, "local_rag_agent.workflows.runner")
        self.assertEqual(WorkflowPipeline.__module__, "local_rag_agent.workflows.runner")
        self.assertEqual(WorkflowRegistry.__module__, "local_rag_agent.workflows.registry")
        self.assertIs(LegacyWorkflowDefinition, WorkflowDefinition)
        self.assertIs(legacy_load_workflows, load_workflows)
        self.assertIs(LegacyStepRegistry, StepRegistry)
        self.assertIs(legacy_prepare_retrieval_query, prepare_retrieval_query)
        self.assertIs(legacy_build_response, build_response)
        self.assertIs(LegacyWorkflowContext, WorkflowContext)
        self.assertIs(LegacyWorkflowPipeline, WorkflowPipeline)
        self.assertIs(LegacyWorkflowRegistry, WorkflowRegistry)

    def test_registry_contains_required_builtin_workflows(self):
        registry = WorkflowRegistry.builtins()

        self.assertTrue(registry.has("rag_qa"))
        self.assertTrue(registry.has("retrieval_debug"))
        self.assertTrue(registry.has("refusal_with_guidance"))

    def test_workflow_registry_unknown_workflow_requires_explicit_fallback(self):
        registry = WorkflowRegistry.builtins()

        with self.assertRaisesRegex(KeyError, "Unknown workflow"):
            registry.get("missing_workflow")

        self.assertEqual(registry.get("missing_workflow", allow_fallback=True).workflow_id, "rag_qa")

    def test_step_registry_contains_required_builtin_steps(self):
        registry = StepRegistry.builtins()

        self.assertTrue(registry.has("prepare_retrieval_query"))
        self.assertTrue(registry.has("run_retrieval"))
        self.assertTrue(registry.has("apply_policy"))
        self.assertTrue(registry.has("build_retrieval_debug_response"))

    def test_load_workflows_reads_schema_and_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflows.toml"
            path.write_text(
                'schema_version = "workflow.v1"\n'
                '[[workflows]]\n'
                'id = "rag_qa_light"\n'
                'steps = ["prepare_retrieval_query", "run_retrieval", "build_retrieval_debug_response"]\n',
                encoding="utf-8",
            )

            workflows = load_workflows(path)

            self.assertEqual(workflows[0].id, "rag_qa_light")
            self.assertEqual(
                workflows[0].steps,
                ["prepare_retrieval_query", "run_retrieval", "build_retrieval_debug_response"],
            )
            self.assertEqual(workflows[0].schema_version, "workflow.v1")

    def test_load_workflows_reads_v2_requires_sources_and_terminal_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflows.toml"
            path.write_text(
                'schema_version = "workflow.v2"\n'
                '[[workflows]]\n'
                'id = "research_qa"\n'
                'requires_sources = true\n'
                'terminal_steps = ["build_policy_response", "build_response"]\n'
                'steps = ["prepare_retrieval_query", "run_retrieval", "apply_policy", "build_policy_response", "generate_answer", "build_response"]\n',
                encoding="utf-8",
            )

            workflows = load_workflows(path)

            self.assertEqual(workflows[0].id, "research_qa")
            self.assertTrue(workflows[0].requires_sources)
            self.assertEqual(workflows[0].terminal_steps, ["build_policy_response", "build_response"])
            self.assertEqual(workflows[0].schema_version, "workflow.v2")

    def test_load_workflows_reads_v3_graph_nodes_and_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflows.toml"
            path.write_text(
                'schema_version = "workflow.v3"\n'
                '[[workflows]]\n'
                'id = "tool_or_rag"\n'
                'type = "graph"\n'
                'start = "route"\n'
                'requires_sources = false\n'
                '[[workflows.nodes]]\n'
                'id = "route"\n'
                'step = "tool.select"\n'
                'checkpoint_after = true\n'
                '[[workflows.nodes]]\n'
                'id = "tool"\n'
                'step = "response.tool_result"\n'
                'terminal = true\n'
                '[[workflows.edges]]\n'
                'from = "route"\n'
                'to = "tool"\n'
                'condition = "default"\n',
                encoding="utf-8",
            )

            workflows = load_workflows(path)

            self.assertEqual(workflows[0].id, "tool_or_rag")
            self.assertEqual(workflows[0].type, "graph")
            self.assertEqual(workflows[0].start, "route")
            self.assertEqual(workflows[0].steps, ["tool.select", "response.tool_result"])
            self.assertEqual(workflows[0].terminal_steps, ["response.tool_result"])
            self.assertEqual(workflows[0].nodes[0]["id"], "route")
            self.assertEqual(workflows[0].nodes[0]["step"], "tool.select")
            self.assertEqual(workflows[0].edges[0]["condition"], "default")
            self.assertEqual(workflows[0].schema_version, "workflow.v3")

    def test_workflow_registry_from_config_runs_configured_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            workflow_config = root / "workflows.toml"
            workflow_config.write_text(
                'schema_version = "workflow.v1"\n'
                '[[workflows]]\n'
                'id = "rag_qa_light"\n'
                'steps = ["prepare_retrieval_query", "run_retrieval", "build_retrieval_debug_response"]\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                workflow_config_path=workflow_config,
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                json.dumps(
                    {
                        "chunks": [
                            {
                                "chunk_id": "facts.md#0",
                                "source": "facts.md",
                                "title": "Facts",
                                "content": "Answer: configured workflow evidence.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            decision = IntentRouter(
                load_intents_from_inline(
                    '[[intents]]\n'
                    'id = "debug"\n'
                    'workflow = "rag_qa_light"\n'
                    'keywords = ["debug"]\n'
                )
            ).route("debug")
            trace = AgentTrace(intent=decision.intent.id, workflow=decision.intent.workflow)
            context = WorkflowContext(settings, AgentRequest("debug evidence"), decision, trace)

            response = WorkflowRegistry.from_config(workflow_config).get("rag_qa_light").run(context).to_dict()

            self.assertEqual(response["workflow"], "rag_qa_light")
            self.assertEqual(response["mode"], "retrieval_debug")
            self.assertEqual(
                response["trace"]["steps"][0]["detail"]["steps"],
                ["prepare_retrieval_query", "run_retrieval", "build_retrieval_debug_response"],
            )

    def test_workflow_registry_runs_graph_workflow_by_following_default_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            workflow_config = root / "workflows.toml"
            workflow_config.write_text(
                'schema_version = "workflow.v3"\n'
                '[[workflows]]\n'
                'id = "graph_debug"\n'
                'type = "graph"\n'
                'start = "prepare"\n'
                'requires_sources = false\n'
                '[[workflows.nodes]]\n'
                'id = "prepare"\n'
                'step = "prepare_retrieval_query"\n'
                '[[workflows.nodes]]\n'
                'id = "debug"\n'
                'step = "build_retrieval_debug_response"\n'
                'terminal = true\n'
                '[[workflows.edges]]\n'
                'from = "prepare"\n'
                'to = "debug"\n'
                'condition = "default"\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                workflow_config_path=workflow_config,
            )
            decision = IntentRouter(
                load_intents_from_inline(
                    '[[intents]]\n'
                    'id = "debug"\n'
                    'workflow = "graph_debug"\n'
                    'keywords = ["debug"]\n'
                )
            ).route("debug")
            trace = AgentTrace(intent=decision.intent.id, workflow=decision.intent.workflow)
            context = WorkflowContext(settings, AgentRequest("debug evidence"), decision, trace)

            response = WorkflowRegistry.from_config(workflow_config).get("graph_debug").run(context).to_dict()

            self.assertEqual(response["workflow"], "graph_debug")
            self.assertEqual(response["mode"], "retrieval_debug")
            step_names = [step["name"] for step in response["trace"]["steps"]]
            self.assertEqual(step_names[:2], ["start_graph_workflow", "prepare_retrieval_query"])
            self.assertNotIn("run_retrieval", step_names)

    def test_graph_workflow_prefers_policy_blocked_edge_over_default_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            workflow_config = root / "workflows.toml"
            workflow_config.write_text(
                'schema_version = "workflow.v3"\n'
                '[[workflows]]\n'
                'id = "policy_route"\n'
                'type = "graph"\n'
                'start = "policy"\n'
                'requires_sources = false\n'
                '[[workflows.nodes]]\n'
                'id = "policy"\n'
                'step = "apply_policy"\n'
                '[[workflows.nodes]]\n'
                'id = "debug"\n'
                'step = "build_retrieval_debug_response"\n'
                'terminal = true\n'
                '[[workflows.nodes]]\n'
                'id = "blocked"\n'
                'step = "build_policy_response"\n'
                'terminal = true\n'
                '[[workflows.edges]]\n'
                'from = "policy"\n'
                'to = "debug"\n'
                'condition = "default"\n'
                '[[workflows.edges]]\n'
                'from = "policy"\n'
                'to = "blocked"\n'
                'condition = "policy.blocked"\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                workflow_config_path=workflow_config,
            )
            decision = IntentRouter(
                load_intents_from_inline(
                    '[[intents]]\n'
                    'id = "blocked"\n'
                    'workflow = "policy_route"\n'
                    'keywords = ["blocked"]\n'
                    'policy = "deny"\n'
                )
            ).route("blocked")
            trace = AgentTrace(intent=decision.intent.id, workflow=decision.intent.workflow)
            context = WorkflowContext(
                settings,
                AgentRequest("blocked"),
                decision,
                trace,
                policy_guard=PolicyGuard.builtins(
                    [PolicyDefinition(id="deny", action="refuse", message="blocked by policy")]
                ),
            )

            response = WorkflowRegistry.from_config(workflow_config).get("policy_route").run(context).to_dict()

            self.assertEqual(response["mode"], "refusal")
            self.assertEqual(response["answer"], "blocked by policy")

    def test_graph_workflow_follows_intent_requires_tool_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            workflow_config = root / "workflows.toml"
            workflow_config.write_text(
                'schema_version = "workflow.v3"\n'
                '[[workflows]]\n'
                'id = "tool_route"\n'
                'type = "graph"\n'
                'start = "route"\n'
                'requires_sources = false\n'
                '[[workflows.nodes]]\n'
                'id = "route"\n'
                'step = "prepare_retrieval_query"\n'
                '[[workflows.nodes]]\n'
                'id = "debug"\n'
                'step = "build_retrieval_debug_response"\n'
                'terminal = true\n'
                '[[workflows.nodes]]\n'
                'id = "tool"\n'
                'step = "response.tool_result"\n'
                'terminal = true\n'
                '[[workflows.edges]]\n'
                'from = "route"\n'
                'to = "debug"\n'
                'condition = "default"\n'
                '[[workflows.edges]]\n'
                'from = "route"\n'
                'to = "tool"\n'
                'condition = "intent.requires_tool"\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                workflow_config_path=workflow_config,
            )
            decision = IntentRouter(
                load_intents_from_inline(
                    '[[intents]]\n'
                    'id = "tool_lookup"\n'
                    'workflow = "tool_route"\n'
                    'keywords = ["lookup"]\n'
                    'requires_tool = true\n'
                )
            ).route("lookup")
            trace = AgentTrace(intent=decision.intent.id, workflow=decision.intent.workflow)
            context = WorkflowContext(settings, AgentRequest("lookup"), decision, trace)

            response = WorkflowRegistry.from_config(workflow_config).get("tool_route").run(context).to_dict()

            self.assertEqual(response["mode"], "tool_error")
            self.assertEqual(response["answer"], "Tool did not produce output.")

    def test_graph_workflow_writes_checkpoint_after_marked_node(self):
        from local_rag_agent.stores.sqlite import SQLiteRunStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            workflow_config = root / "workflows.toml"
            workflow_config.write_text(
                'schema_version = "workflow.v3"\n'
                '[[workflows]]\n'
                'id = "checkpoint_graph"\n'
                'type = "graph"\n'
                'start = "prepare"\n'
                'requires_sources = false\n'
                '[[workflows.nodes]]\n'
                'id = "prepare"\n'
                'step = "prepare_retrieval_query"\n'
                'checkpoint_after = true\n'
                '[[workflows.nodes]]\n'
                'id = "debug"\n'
                'step = "build_retrieval_debug_response"\n'
                'terminal = true\n'
                '[[workflows.edges]]\n'
                'from = "prepare"\n'
                'to = "debug"\n'
                'condition = "default"\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                workflow_config_path=workflow_config,
            )
            decision = IntentRouter(
                load_intents_from_inline(
                    '[[intents]]\n'
                    'id = "debug"\n'
                    'workflow = "checkpoint_graph"\n'
                    'keywords = ["debug"]\n'
                )
            ).route("debug")
            store = SQLiteRunStore.in_memory()
            self.addCleanup(store.close)
            store.create_run(run_id="run-graph", intent="debug", workflow="checkpoint_graph")
            trace = AgentTrace(intent=decision.intent.id, workflow=decision.intent.workflow, run_id="run-graph")
            context = WorkflowContext(
                settings,
                AgentRequest("debug checkpoint"),
                decision,
                trace,
                run_store=store,
                run_id="run-graph",
            )

            WorkflowRegistry.from_config(workflow_config).get("checkpoint_graph").run(context)

            checkpoints = store.list_checkpoints("run-graph")
            self.assertEqual([checkpoint["node_id"] for checkpoint in checkpoints], ["prepare"])
            self.assertEqual(checkpoints[0]["state"]["retrieval_query"], "debug checkpoint")

    def test_tool_runtime_authorizes_disabled_and_intent_restricted_tools_before_calling_provider(self):
        from local_rag_agent.tool_runtime import ToolRuntime
        from local_rag_agent.tools import ToolDefinition, ToolResult

        class RecordingProvider:
            def __init__(self):
                self.calls = []
                self.tools = {
                    "disabled": ToolDefinition(id="disabled", enabled=False, adapter="mock"),
                    "restricted": ToolDefinition(
                        id="restricted",
                        enabled=True,
                        adapter="mock",
                        allowed_intents=["other_intent"],
                    ),
                }

            def call(self, tool_id, arguments, intent_id=""):
                self.calls.append((tool_id, arguments, intent_id))
                return ToolResult(tool_id=tool_id, ok=True, output={"answer": "called"})

        settings = Settings(
            project_root=REPO_ROOT,
            prompt_path=REPO_ROOT / "prompt.md",
            manifest_path=REPO_ROOT / "manifest.md",
            knowledge_root=REPO_ROOT / "knowledge_base",
            index_path=REPO_ROOT / ".local_rag_agent" / "index.json",
        )
        decision = IntentRouter(
            load_intents_from_inline(
                '[[intents]]\n'
                'id = "tool_lookup"\n'
                'workflow = "tool_lookup"\n'
                'keywords = ["lookup"]\n'
            )
        ).route("lookup")
        context = WorkflowContext(
            settings,
            AgentRequest("lookup this"),
            decision,
            AgentTrace(intent=decision.intent.id, workflow=decision.intent.workflow),
        )
        provider = RecordingProvider()
        runtime = ToolRuntime(provider)

        disabled, _ = runtime.call("disabled", context)
        restricted, _ = runtime.call("restricted", context)

        self.assertFalse(disabled.ok)
        self.assertEqual(disabled.error, "Tool is disabled: disabled")
        self.assertFalse(restricted.ok)
        self.assertEqual(restricted.error, "Tool is not allowed for intent: tool_lookup")
        self.assertEqual(provider.calls, [])

    def test_tool_runtime_blocks_approval_required_tools_and_records_audit_event(self):
        from local_rag_agent.tool_runtime import ToolRuntime
        from local_rag_agent.tools import ToolDefinition, ToolResult

        class RecordingProvider:
            def __init__(self):
                self.calls = []
                self.tools = {
                    "lookup": ToolDefinition(
                        id="lookup",
                        enabled=True,
                        adapter="mock",
                        allowed_intents=["tool_lookup"],
                        requires_approval=True,
                    )
                }

            def call(self, tool_id, arguments, intent_id=""):
                self.calls.append((tool_id, arguments, intent_id))
                return ToolResult(tool_id=tool_id, ok=True, output={"answer": "called"})

        settings = Settings(
            project_root=REPO_ROOT,
            prompt_path=REPO_ROOT / "prompt.md",
            manifest_path=REPO_ROOT / "manifest.md",
            knowledge_root=REPO_ROOT / "knowledge_base",
            index_path=REPO_ROOT / ".local_rag_agent" / "index.json",
        )
        decision = IntentRouter(
            load_intents_from_inline(
                '[[intents]]\n'
                'id = "tool_lookup"\n'
                'workflow = "tool_lookup"\n'
                'keywords = ["lookup"]\n'
            )
        ).route("lookup")
        context = WorkflowContext(
            settings,
            AgentRequest("lookup this"),
            decision,
            AgentTrace(intent=decision.intent.id, workflow=decision.intent.workflow),
        )
        audit_events = []
        provider = RecordingProvider()

        result, arguments = ToolRuntime(provider, audit_sink=audit_events.append).call("lookup", context)

        self.assertFalse(result.ok)
        self.assertEqual(result.error, "Tool requires approval: lookup")
        self.assertEqual(arguments, {"query": "lookup this"})
        self.assertEqual(provider.calls, [])
        self.assertEqual(
            audit_events,
            [
                {
                    "event": "tool.approval_required",
                    "tool_id": "lookup",
                    "intent": "tool_lookup",
                    "arguments": {"query": "lookup this"},
                }
            ],
        )

    def test_workflow_registry_rejects_unknown_configured_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflows.toml"
            path.write_text(
                '[[workflows]]\n'
                'id = "broken"\n'
                'steps = ["missing_step"]\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unknown workflow step"):
                WorkflowRegistry.from_config(path)

    def test_configured_tool_workflow_calls_tool_and_traces_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            agent_dir = root / "agent"
            agent_dir.mkdir()
            intent_config = agent_dir / "intents.toml"
            workflow_config = agent_dir / "workflows.toml"
            tool_config = agent_dir / "tools.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "tool_lookup"\n'
                'workflow = "tool_lookup"\n'
                'keywords = ["lookup"]\n',
                encoding="utf-8",
            )
            workflow_config.write_text(
                'schema_version = "workflow.v1"\n'
                '[[workflows]]\n'
                'id = "tool_lookup"\n'
                'steps = ["tool.call_first", "response.tool_result"]\n',
                encoding="utf-8",
            )
            tool_config.write_text(
                '[[tools]]\n'
                'id = "lookup"\n'
                'enabled = true\n'
                'provider = "mock"\n'
                'allowed_intents = ["tool_lookup"]\n'
                '[tools.mock_output]\n'
                'answer = "mocked tool answer"\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                intent_config_path=intent_config,
                workflow_config_path=workflow_config,
                tool_config_path=tool_config,
            )

            response = AgentRuntime(settings).run(AgentRequest("lookup this")).to_dict()

            self.assertEqual(response["mode"], "tool")
            self.assertIn("mocked tool answer", response["answer"])
            tool_steps = [step for step in response["trace"]["steps"] if step["name"] == "tool.call"]
            self.assertEqual(tool_steps[0]["detail"]["tool_id"], "lookup")
            self.assertTrue(tool_steps[0]["detail"]["ok"])

    def test_configured_tool_workflow_selects_then_calls_tool_and_traces_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            agent_dir = root / "agent"
            agent_dir.mkdir()
            intent_config = agent_dir / "intents.toml"
            workflow_config = agent_dir / "workflows.toml"
            tool_config = agent_dir / "tools.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "tool_lookup"\n'
                'workflow = "tool_lookup"\n'
                'keywords = ["lookup"]\n',
                encoding="utf-8",
            )
            workflow_config.write_text(
                'schema_version = "workflow.v1"\n'
                '[[workflows]]\n'
                'id = "tool_lookup"\n'
                'steps = ["tool.select", "tool.call", "response.tool_result"]\n',
                encoding="utf-8",
            )
            tool_config.write_text(
                '[[tools]]\n'
                'id = "lookup"\n'
                'enabled = true\n'
                'provider = "mock"\n'
                'allowed_intents = ["tool_lookup"]\n'
                '[tools.mock_output]\n'
                'answer = "selected tool answer"\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                intent_config_path=intent_config,
                workflow_config_path=workflow_config,
                tool_config_path=tool_config,
            )

            response = AgentRuntime(settings).run(AgentRequest("lookup this")).to_dict()

            self.assertEqual(response["mode"], "tool")
            self.assertIn("selected tool answer", response["answer"])
            select_steps = [step for step in response["trace"]["steps"] if step["name"] == "tool.select"]
            call_steps = [step for step in response["trace"]["steps"] if step["name"] == "tool.call"]
            self.assertEqual(select_steps[0]["detail"]["tool_id"], "lookup")
            self.assertEqual(call_steps[0]["detail"]["tool_id"], "lookup")
            self.assertTrue(call_steps[0]["detail"]["ok"])

    def test_tool_runtime_maps_request_message_and_metadata_into_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            agent_dir = root / "agent"
            agent_dir.mkdir()
            intent_config = agent_dir / "intents.toml"
            workflow_config = agent_dir / "workflows.toml"
            tool_config = agent_dir / "tools.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "tool_lookup"\n'
                'workflow = "tool_lookup"\n'
                'keywords = ["lookup"]\n',
                encoding="utf-8",
            )
            workflow_config.write_text(
                'schema_version = "workflow.v1"\n'
                '[[workflows]]\n'
                'id = "tool_lookup"\n'
                'steps = ["tool.select", "tool.call", "response.tool_result"]\n',
                encoding="utf-8",
            )
            tool_config.write_text(
                'schema_version = "tool.v2"\n'
                '[[tools]]\n'
                'id = "lookup"\n'
                'enabled = true\n'
                'adapter = "mock"\n'
                'allowed_intents = ["tool_lookup"]\n'
                '[tools.input_mapping]\n'
                'query = "$message"\n'
                'user_id = "$metadata.user_id"\n'
                '[tools.mock_output]\n'
                'answer = "mapped tool answer"\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                intent_config_path=intent_config,
                workflow_config_path=workflow_config,
                tool_config_path=tool_config,
            )

            response = AgentRuntime(settings).run(
                AgentRequest("lookup my calendar", metadata={"user_id": "user-123"})
            ).to_dict()

            self.assertEqual(response["mode"], "tool")
            call_steps = [step for step in response["trace"]["steps"] if step["name"] == "tool.call"]
            self.assertEqual(
                call_steps[0]["detail"]["arguments"],
                {"query": "lookup my calendar", "user_id": "user-123"},
            )

    def test_tool_validate_output_sanitizes_schema_allowed_fields_before_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            agent_dir = root / "agent"
            agent_dir.mkdir()
            intent_config = agent_dir / "intents.toml"
            workflow_config = agent_dir / "workflows.toml"
            tool_config = agent_dir / "tools.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "tool_lookup"\n'
                'workflow = "tool_lookup"\n'
                'keywords = ["lookup"]\n',
                encoding="utf-8",
            )
            workflow_config.write_text(
                '[[workflows]]\n'
                'id = "tool_lookup"\n'
                'steps = ["tool.select", "tool.call", "tool.validate_output", "response.tool_result"]\n',
                encoding="utf-8",
            )
            tool_config.write_text(
                '[[tools]]\n'
                'id = "lookup"\n'
                'enabled = true\n'
                'provider = "mock"\n'
                'allowed_intents = ["tool_lookup"]\n'
                '[tools.mock_output]\n'
                'answer = "schema answer"\n'
                'secret = "drop me"\n'
                '[tools.schema]\n'
                'required = ["answer"]\n'
                '[tools.schema.properties.answer]\n'
                'type = "string"\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                intent_config_path=intent_config,
                workflow_config_path=workflow_config,
                tool_config_path=tool_config,
            )

            response = AgentRuntime(settings).run(AgentRequest("lookup this")).to_dict()

            self.assertEqual(response["mode"], "tool")
            self.assertEqual(response["answer"], "schema answer")
            self.assertEqual(response["metadata"]["tool_results"][0]["output"], {"answer": "schema answer"})
            validate_steps = [step for step in response["trace"]["steps"] if step["name"] == "tool.validate_output"]
            self.assertTrue(validate_steps[0]["detail"]["ok"])

    def test_tool_validate_output_returns_tool_error_for_schema_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            agent_dir = root / "agent"
            agent_dir.mkdir()
            intent_config = agent_dir / "intents.toml"
            workflow_config = agent_dir / "workflows.toml"
            tool_config = agent_dir / "tools.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "tool_lookup"\n'
                'workflow = "tool_lookup"\n'
                'keywords = ["lookup"]\n',
                encoding="utf-8",
            )
            workflow_config.write_text(
                '[[workflows]]\n'
                'id = "tool_lookup"\n'
                'steps = ["tool.select", "tool.call", "tool.validate_output", "response.tool_result"]\n',
                encoding="utf-8",
            )
            tool_config.write_text(
                '[[tools]]\n'
                'id = "lookup"\n'
                'enabled = true\n'
                'provider = "mock"\n'
                'allowed_intents = ["tool_lookup"]\n'
                '[tools.mock_output]\n'
                'text = "missing required answer"\n'
                '[tools.schema]\n'
                'required = ["answer"]\n'
                '[tools.schema.properties.answer]\n'
                'type = "string"\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                intent_config_path=intent_config,
                workflow_config_path=workflow_config,
                tool_config_path=tool_config,
            )

            response = AgentRuntime(settings).run(AgentRequest("lookup this")).to_dict()

            self.assertEqual(response["mode"], "tool_error")
            self.assertIn("missing required field answer", response["answer"])
            validate_steps = [step for step in response["trace"]["steps"] if step["name"] == "tool.validate_output"]
            self.assertFalse(validate_steps[0]["detail"]["ok"])
            self.assertIn("missing required field answer", validate_steps[0]["detail"]["error"])

    def test_tool_v2_adapter_and_output_schema_drive_mock_tool_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("system prompt", encoding="utf-8")
            agent_dir = root / "agent"
            agent_dir.mkdir()
            intent_config = agent_dir / "intents.toml"
            workflow_config = agent_dir / "workflows.toml"
            tool_config = agent_dir / "tools.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "tool_lookup"\n'
                'workflow = "tool_lookup"\n'
                'keywords = ["lookup"]\n',
                encoding="utf-8",
            )
            workflow_config.write_text(
                '[[workflows]]\n'
                'id = "tool_lookup"\n'
                'steps = ["tool.select", "tool.call", "tool.validate_output", "response.tool_result"]\n',
                encoding="utf-8",
            )
            tool_config.write_text(
                'schema_version = "tool.v2"\n'
                '[[tools]]\n'
                'id = "lookup"\n'
                'enabled = true\n'
                'adapter = "mock"\n'
                'allowed_intents = ["tool_lookup"]\n'
                '[tools.mock_output]\n'
                'answer = "v2 adapter answer"\n'
                'extra = "drop me"\n'
                '[tools.output_schema]\n'
                'required = ["answer"]\n'
                '[tools.output_schema.properties.answer]\n'
                'type = "string"\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                intent_config_path=intent_config,
                workflow_config_path=workflow_config,
                tool_config_path=tool_config,
            )

            response = AgentRuntime(settings).run(AgentRequest("lookup this")).to_dict()

            self.assertEqual(response["mode"], "tool")
            self.assertEqual(response["answer"], "v2 adapter answer")
            self.assertEqual(response["metadata"]["tool_results"][0]["output"], {"answer": "v2 adapter answer"})

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


class ComponentRegistryTests(unittest.TestCase):
    def test_component_registry_rejects_duplicate_and_missing_registrations(self):
        from local_rag_agent.components import ComponentRegistry

        def custom_step(context):
            return None

        registry = ComponentRegistry()
        sink = object()

        registry.register_step("custom.response", custom_step)
        registry.register_trace_sink("memory", sink)

        with self.assertRaisesRegex(ValueError, "Duplicate component registration: step custom.response"):
            registry.register_step("custom.response", custom_step)
        with self.assertRaisesRegex(ValueError, "Duplicate component registration: trace_sink memory"):
            registry.register_trace_sink("memory", object())
        with self.assertRaisesRegex(KeyError, "Missing component registration: generator missing"):
            registry.get_generator("missing")
        self.assertIs(registry.get_trace_sink("memory"), sink)

    def test_component_registry_registers_step_definition_metadata(self):
        from local_rag_agent.components import ComponentRegistry, StepDefinition

        def custom_response(context):
            context.response = None

        registry = ComponentRegistry()
        definition = StepDefinition(
            id="custom.response",
            fn=custom_response,
            terminal=True,
            risk_level="low",
            timeout_seconds=5,
        )

        registry.register_step_definition(definition)

        self.assertTrue(registry.has_step("custom.response"))
        self.assertIs(registry.get_step("custom.response"), custom_response)
        self.assertEqual(registry.get_step_definition("custom.response").id, "custom.response")
        self.assertEqual(registry.terminal_steps(), {"custom.response"})
        with self.assertRaisesRegex(ValueError, "Duplicate component registration: step custom.response"):
            registry.register_step_definition(definition)

    def test_runtime_construction_uses_component_registry_for_steps_and_providers(self):
        from local_rag_agent.components import ComponentRegistry

        class FakeRetriever:
            def retrieve(self, settings, query):
                return []

        class FakeGenerator:
            def generate(self, settings, question, retrieved_chunks, model_client=None, history=None):
                return None

        class FakePolicyGuard:
            policies = {}

            def evaluate(self, *args, **kwargs):
                return None

        class FakeToolProvider:
            tools = {}

            def call(self, *args, **kwargs):
                return None

        def custom_response(context):
            context.response = AgentResponse(
                answer="custom registry response",
                mode="custom",
                intent=context.intent_decision.intent.id,
                workflow=context.intent_decision.intent.workflow,
                sources=[],
                trace=context.trace,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            workflow_config = root / "workflows.toml"
            workflow_config.write_text(
                '[[workflows]]\n'
                'id = "custom_flow"\n'
                'steps = ["custom.response"]\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=root / "prompt.md",
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                workflow_config_path=workflow_config,
                default_workflow="custom_flow",
                retrieval_provider="fake_retriever",
                generation_provider="fake_generator",
            )
            policy_guard = FakePolicyGuard()
            tool_provider = FakeToolProvider()
            sink = object()
            registry = ComponentRegistry()
            registry.register_step("custom.response", custom_response)
            registry.register_retriever("fake_retriever", lambda settings: FakeRetriever())
            registry.register_generator("fake_generator", lambda settings: FakeGenerator())
            registry.register_policy_provider("keyword", lambda settings: policy_guard)
            registry.register_tool_provider("configured", lambda settings: tool_provider)
            registry.register_trace_sink("memory", sink)

            runtime = AgentRuntime(settings, components=registry)
            response = runtime.run(AgentRequest(message="hello"))

            self.assertEqual(response.mode, "custom")
            self.assertEqual(response.answer, "custom registry response")
            self.assertIsInstance(runtime.retriever_provider.retriever, FakeRetriever)
            self.assertIsInstance(runtime.generator_provider.generator, FakeGenerator)
            self.assertIs(runtime.policy_guard, policy_guard)
            self.assertIs(runtime.tool_provider, tool_provider)
            self.assertIs(runtime.components.get_trace_sink("memory"), sink)

    def test_runtime_loads_plugin_module_that_registers_workflow_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            agent_dir = root / "agent"
            plugin_dir = root / "agent_plugins"
            agent_dir.mkdir()
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
            (plugin_dir / "custom_steps.py").write_text(
                "from local_rag_agent.types import AgentResponse\n"
                "\n"
                "def plugin_response(context):\n"
                "    context.response = AgentResponse(\n"
                "        answer='plugin response',\n"
                "        mode='plugin',\n"
                "        intent=context.intent_decision.intent.id,\n"
                "        workflow=context.intent_decision.intent.workflow,\n"
                "        sources=[],\n"
                "        trace=context.trace,\n"
                "    )\n"
                "\n"
                "def register(registry):\n"
                "    registry.register_step('plugin.response', plugin_response)\n",
                encoding="utf-8",
            )
            workflow_config = agent_dir / "workflows.toml"
            workflow_config.write_text(
                '[[workflows]]\n'
                'id = "plugin_flow"\n'
                'steps = ["plugin.response"]\n',
                encoding="utf-8",
            )
            config_path = root / "runtime.toml"
            config_path.write_text(
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[runtime]\n'
                'default_workflow = "plugin_flow"\n'
                'workflow_config = "agent/workflows.toml"\n'
                '[plugins]\n'
                'modules = ["agent_plugins.custom_steps"]\n',
                encoding="utf-8",
            )

            runtime = AgentRuntime.from_project(root, config_path)
            response = runtime.run(AgentRequest(message="hello"))

            self.assertEqual(response.mode, "plugin")
            self.assertEqual(response.answer, "plugin response")


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

    def test_summarize_regression_report_flags_missing_sources_and_counts_trace_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            records = [
                {
                    "id": "C01",
                    "question": "source backed?",
                    "answer": "ok",
                    "sources": [{"source": "facts.md"}],
                    "mode": "extractive",
                    "intent": "knowledge_qa",
                    "workflow": "rag_qa",
                    "trace": {
                        "config_versions": {"runtime": "runtime.v1"},
                        "steps": [
                            {"name": "route_intent", "status": "ok", "detail": {}},
                            {"name": "start_workflow", "status": "ok", "detail": {}},
                            {"name": "run_retrieval", "status": "ok", "detail": {"source_count": 1}},
                            {"name": "apply_policy", "status": "ok", "detail": {"policy_id": ""}},
                        ],
                    },
                },
                {
                    "id": "C02",
                    "question": "missing?",
                    "answer": "unsupported",
                    "sources": [],
                    "mode": "extractive",
                    "intent": "knowledge_qa",
                    "workflow": "rag_qa",
                    "trace": {
                        "config_versions": {"runtime": "runtime.v1"},
                        "steps": [
                            {"name": "route_intent", "status": "ok", "detail": {}},
                            {"name": "start_workflow", "status": "ok", "detail": {}},
                            {"name": "run_retrieval", "status": "ok", "detail": {"source_count": 0}},
                            {"name": "apply_policy", "status": "ok", "detail": {"policy_id": ""}},
                            {"name": "tool.call", "status": "ok", "detail": {"ok": True}},
                        ],
                    },
                },
            ]
            path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

            summary = summarize_regression_report(path)

            self.assertFalse(summary["ok"])
            self.assertEqual(summary["question_count"], 2)
            self.assertEqual(summary["missing_source_count"], 1)
            self.assertEqual(summary["policy_trace_count"], 2)
            self.assertEqual(summary["tool_trace_count"], 1)
            self.assertEqual(summary["failures"][0]["id"], "C02")

    def test_summarize_regression_report_fails_missing_release_trace_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            records = [
                {
                    "id": "C01",
                    "question": "missing route?",
                    "answer": "ok",
                    "sources": [{"source": "facts.md"}],
                    "mode": "extractive",
                    "intent": "knowledge_qa",
                    "workflow": "rag_qa",
                    "trace": {
                        "config_versions": {},
                        "steps": [{"name": "run_retrieval", "detail": {"source_count": 1}}],
                    },
                },
                {
                    "id": "C02",
                    "question": "tool trace?",
                    "answer": "ok",
                    "sources": [],
                    "mode": "tool",
                    "intent": "tool_lookup",
                    "workflow": "tool_lookup",
                    "trace": {
                        "config_versions": {"runtime": "runtime.v1"},
                        "steps": [
                            {"name": "route_intent", "status": "ok", "detail": {}},
                            {"name": "start_workflow", "status": "ok", "detail": {}},
                        ],
                    },
                },
            ]
            path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

            summary = summarize_regression_report(path)

            self.assertFalse(summary["ok"])
            failure_reasons = {failure["reason"] for failure in summary["failures"]}
            self.assertIn("missing_config_versions", failure_reasons)
            self.assertIn("missing_route_intent_trace", failure_reasons)
            self.assertIn("missing_start_workflow_trace", failure_reasons)
            self.assertIn("missing_policy_trace", failure_reasons)
            self.assertIn("missing_tool_trace", failure_reasons)

    def test_release_gate_cli_returns_nonzero_for_missing_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n',
                encoding="utf-8",
            )
            report = root / "results.jsonl"
            failure_record = {
                "id": "C01",
                "question": "unsupported?",
                "answer": "unsupported",
                "sources": [],
                "mode": "extractive",
                "intent": "knowledge_qa",
                "workflow": "rag_qa",
                "trace": {
                    "config_versions": {"runtime": "runtime.v1"},
                    "steps": [
                        {"name": "route_intent", "status": "ok", "detail": {}},
                        {"name": "start_workflow", "status": "ok", "detail": {}},
                        {"name": "run_retrieval", "status": "ok", "detail": {"source_count": 0}},
                        {"name": "apply_policy", "status": "ok", "detail": {"policy_id": ""}},
                    ],
                },
            }
            ok_record = {
                "id": "C02",
                "question": "supported?",
                "answer": "ok",
                "sources": [{"source": "facts.md"}],
                "mode": "extractive",
                "intent": "knowledge_qa",
                "workflow": "rag_qa",
                "trace": {
                    "config_versions": {"runtime": "runtime.v1"},
                    "steps": [
                        {"name": "route_intent", "status": "ok", "detail": {}},
                        {"name": "start_workflow", "status": "ok", "detail": {}},
                        {"name": "run_retrieval", "status": "ok", "detail": {"source_count": 1}},
                        {"name": "apply_policy", "status": "ok", "detail": {"policy_id": ""}},
                    ],
                },
            }

            report.write_text(json.dumps(failure_record), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                failed_exit = cli_main(
                    ["release-gate", "--project", str(root), "--config", str(config_path), "--report", str(report)]
                )

            report.write_text(json.dumps(ok_record), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                ok_exit = cli_main(
                    ["release-gate", "--project", str(root), "--config", str(config_path), "--report", str(report)]
                )

            self.assertEqual(failed_exit, 1)
            self.assertEqual(ok_exit, 0)

    def test_release_gate_cli_validates_config_before_reading_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            missing_report = root / "missing-results.jsonl"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[retrieval]\n'
                'provider = "unknown"\n',
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                exit_code = cli_main(
                    [
                        "release-gate",
                        "--project",
                        str(root),
                        "--config",
                        str(config_path),
                        "--report",
                        str(missing_report),
                    ]
                )

            payload = json.loads(stdout.getvalue())

            self.assertEqual(exit_code, 1)
            self.assertEqual(payload["stage"], "validate")
            self.assertEqual(payload["validation"]["errors"][0]["code"], "UNKNOWN_RETRIEVER_PROVIDER")

    def test_smoke_cli_runs_template_runtime_gate_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "agent-project"
            shutil.copytree(TEMPLATE_ROOT, root)
            config_path = root / "runtime.toml"
            questions_path = root / "examples" / "core-regression-questions.md"

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                exit_code = cli_main(
                    [
                        "smoke",
                        "--project",
                        str(root),
                        "--config",
                        str(config_path),
                        "--questions",
                        str(questions_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())

            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["validate"]["ok"])
            self.assertGreater(payload["ingest"]["chunk_count"], 0)
            self.assertGreater(payload["regression"]["question_count"], 0)
            self.assertTrue(payload["release_gate"]["ok"])
            self.assertTrue(payload["http"]["healthz"]["ok"])
            self.assertTrue(payload["http"]["version"]["ok"])

    def test_smoke_cli_accepts_plugin_terminal_workflow_release_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            agent_dir = root / "agent"
            plugin_dir = root / "agent_plugins"
            knowledge_dir = root / "knowledge_base"
            manifest_dir = knowledge_dir / "_manifests"
            examples_dir = root / "examples"
            agent_dir.mkdir()
            plugin_dir.mkdir()
            manifest_dir.mkdir(parents=True)
            examples_dir.mkdir()
            (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
            (plugin_dir / "custom_steps.py").write_text(
                "from local_rag_agent.components import StepDefinition\n"
                "from local_rag_agent.types import AgentResponse\n"
                "\n"
                "def plugin_response(context):\n"
                "    context.response = AgentResponse(\n"
                "        answer='plugin response',\n"
                "        mode='plugin',\n"
                "        intent=context.intent_decision.intent.id,\n"
                "        workflow=context.intent_decision.intent.workflow,\n"
                "        sources=[],\n"
                "        trace=context.trace,\n"
                "    )\n"
                "\n"
                "def register(registry):\n"
                "    registry.register_step_definition(\n"
                "        StepDefinition(id='plugin.response', fn=plugin_response, terminal=True)\n"
                "    )\n",
                encoding="utf-8",
            )
            (knowledge_dir / "facts.md").write_text("# Facts\n\nPlugin smoke fact.\n", encoding="utf-8")
            (manifest_dir / "current-upload-manifest.md").write_text("- `knowledge_base/facts.md`\n", encoding="utf-8")
            (examples_dir / "plugin-questions.md").write_text(
                "| 编号 | 问题 | 预期要点 |\n"
                "| --- | --- | --- |\n"
                "| P01 | hello plugin? | plugin response |\n",
                encoding="utf-8",
            )
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[runtime]\n'
                'default_workflow = "plugin_flow"\n'
                'workflow_config = "agent/workflows.toml"\n'
                '[plugins]\n'
                'modules = ["agent_plugins.custom_steps"]\n',
                encoding="utf-8",
            )
            (agent_dir / "workflows.toml").write_text(
                'schema_version = "workflow.v2"\n'
                '[[workflows]]\n'
                'id = "plugin_flow"\n'
                'requires_sources = false\n'
                'steps = ["plugin.response"]\n',
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                exit_code = cli_main(
                    [
                        "smoke",
                        "--project",
                        str(root),
                        "--config",
                        str(config_path),
                        "--questions",
                        str(examples_dir / "plugin-questions.md"),
                    ]
                )

            payload = json.loads(stdout.getvalue())

            self.assertEqual(exit_code, 0, payload)
            self.assertTrue(payload["validate"]["ok"], payload)
            self.assertEqual(payload["release_gate"]["modes"], {"plugin": 1})
            self.assertEqual(payload["release_gate"]["workflows"], {"plugin_flow": 1})
            self.assertTrue(payload["release_gate"]["ok"], payload["release_gate"])


class ServerPageTests(unittest.TestCase):
    def test_http_interface_split_exports_compatible_entrypoints(self):
        from local_rag_agent.interfaces.http import server as http_server
        from local_rag_agent.interfaces.http.errors import error_payload
        from local_rag_agent.interfaces.http.routes import validate_payload
        from local_rag_agent.services.runtime_service import RuntimeService

        settings = Settings(
            project_root=Path.cwd(),
            prompt_path=Path.cwd() / "prompt.md",
            manifest_path=Path.cwd() / "manifest.md",
            knowledge_root=Path.cwd() / "knowledge_base",
            index_path=Path.cwd() / ".local_rag_agent" / "index.json",
        )

        self.assertIs(http_server.make_handler, make_handler)
        self.assertEqual(error_payload("BAD_REQUEST", "bad"), {"error": {"code": "BAD_REQUEST", "message": "bad"}})
        self.assertTrue(validate_payload(settings)["ok"])
        self.assertIsInstance(RuntimeService(settings), RuntimeService)

    def test_http_health_version_and_validate_endpoints_return_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n',
                encoding="utf-8",
            )
            settings = load_settings(root, config_path)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(settings))
            thread = threading.Thread(target=server.serve_forever)
            thread.daemon = True
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            self.addCleanup(lambda: server.server_close())
            self.addCleanup(lambda: thread.join(timeout=2))
            self.addCleanup(server.shutdown)

            health = self._get_json(f"{base_url}/healthz")
            version = self._get_json(f"{base_url}/version")
            validation = self._get_json(f"{base_url}/api/v1/validate")

            self.assertEqual(health["status"], "ok")
            self.assertEqual(health["service"], "local_rag_agent")
            self.assertIn("version", version)
            self.assertEqual(version["service"], "local_rag_agent")
            self.assertTrue(validation["ok"])
            self.assertEqual(validation["errors"], [])

    def test_http_validate_endpoint_falls_back_to_loaded_settings_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            settings = Settings(
                project_root=root,
                prompt_path=root / "prompt.md",
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(settings))
            thread = threading.Thread(target=server.serve_forever)
            thread.daemon = True
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            self.addCleanup(lambda: server.server_close())
            self.addCleanup(lambda: thread.join(timeout=2))
            self.addCleanup(server.shutdown)

            payload = self._get_json(f"{base_url}/api/v1/validate")

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["errors"], [])

    def test_http_chat_v1_returns_agent_response_contract(self):
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
                '{"chunks":[{"chunk_id":"facts.md#0","source":"facts.md","title":"Facts","content":"Answer: HTTP chat works."}]}',
                encoding="utf-8",
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(settings))
            thread = threading.Thread(target=server.serve_forever)
            thread.daemon = True
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            self.addCleanup(lambda: server.server_close())
            self.addCleanup(lambda: thread.join(timeout=2))
            self.addCleanup(server.shutdown)

            payload = self._post_json(
                f"{base_url}/api/v1/chat",
                {
                    "message": "http chat?",
                    "history": [{"role": "user", "content": "previous"}],
                    "metadata": {"request_id": "http-1"},
                },
            )

            self.assertIn("HTTP chat works.", payload["answer"])
            self.assertEqual(payload["mode"], "extractive")
            self.assertEqual(payload["intent"], "knowledge_qa")
            self.assertEqual(payload["workflow"], "rag_qa")
            self.assertEqual(payload["sources"][0]["source"], "facts.md")
            self.assertEqual(payload["trace"]["request_id"], "http-1")
            self.assertIn("route_intent", [step["name"] for step in payload["trace"]["steps"]])

    def test_http_chat_v1_returns_error_envelope_for_bad_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            settings = Settings(
                project_root=root,
                prompt_path=root / "prompt.md",
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(settings))
            thread = threading.Thread(target=server.serve_forever)
            thread.daemon = True
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            self.addCleanup(lambda: server.server_close())
            self.addCleanup(lambda: thread.join(timeout=2))
            self.addCleanup(server.shutdown)

            with self.assertRaises(urllib.error.HTTPError) as error:
                self._post_json(f"{base_url}/api/v1/chat", {"message": ""})

            self.assertEqual(error.exception.code, 400)
            payload = json.loads(error.exception.read().decode("utf-8"))
            error.exception.close()
            self.assertEqual(payload["error"]["code"], "BAD_REQUEST")
            self.assertIn("message", payload["error"]["message"])

    def test_http_chat_v1_enforces_token_auth_body_limit_and_cors_allowlist(self):
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
                server_request_body_limit_bytes=64,
                server_auth_token="secret-token",
                server_cors_allowlist=["https://example.test"],
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"facts.md#0","source":"facts.md","content":"Answer: secure."}]}',
                encoding="utf-8",
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(settings))
            thread = threading.Thread(target=server.serve_forever)
            thread.daemon = True
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            self.addCleanup(lambda: server.server_close())
            self.addCleanup(lambda: thread.join(timeout=2))
            self.addCleanup(server.shutdown)

            with self.assertRaises(urllib.error.HTTPError) as unauthorized:
                self._post_json(
                    f"{base_url}/api/v1/chat",
                    {"message": "secure?"},
                    headers={"Origin": "https://example.test"},
                )
            self.assertEqual(unauthorized.exception.code, 401)
            self.assertEqual(unauthorized.exception.headers["Access-Control-Allow-Origin"], "https://example.test")
            unauthorized_payload = json.loads(unauthorized.exception.read().decode("utf-8"))
            unauthorized.exception.close()
            self.assertEqual(unauthorized_payload["error"]["code"], "AUTH_REQUIRED")

            with self.assertRaises(urllib.error.HTTPError) as too_large:
                self._post_json(
                    f"{base_url}/api/v1/chat",
                    {"message": "x" * 200},
                    headers={"Authorization": "Bearer secret-token"},
                )
            self.assertEqual(too_large.exception.code, 413)
            too_large_payload = json.loads(too_large.exception.read().decode("utf-8"))
            too_large.exception.close()
            self.assertEqual(too_large_payload["error"]["code"], "REQUEST_TOO_LARGE")

    def test_http_chat_v1_uses_auth_and_rate_limit_hooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            settings = Settings(
                project_root=root,
                prompt_path=root / "prompt.md",
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
            )
            hooks = ServerHooks(
                authenticate=lambda handler, settings: True,
                rate_limit=lambda handler, settings: False,
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(settings, hooks=hooks))
            thread = threading.Thread(target=server.serve_forever)
            thread.daemon = True
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            self.addCleanup(lambda: server.server_close())
            self.addCleanup(lambda: thread.join(timeout=2))
            self.addCleanup(server.shutdown)

            with self.assertRaises(urllib.error.HTTPError) as limited:
                self._post_json(f"{base_url}/api/v1/chat", {"message": "limited"})

            self.assertEqual(limited.exception.code, 429)
            payload = json.loads(limited.exception.read().decode("utf-8"))
            limited.exception.close()
            self.assertEqual(payload["error"]["code"], "RATE_LIMITED")

    def _get_json(self, url: str) -> dict[str, object]:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post_json(
        self,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        data = json.dumps(payload).encode("utf-8")
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers or {})
        request = urllib.request.Request(
            url,
            data=data,
            headers=request_headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_render_workspace_home_matches_server_entry_structure(self):
        page = render_workspace_home()

        self.assertIn("Local Agent Workspace", page)
        self.assertIn("Agent workspace", page)
        self.assertIn('src="/chatbot"', page)
        self.assertIn("Boundaries", page)

    def test_render_chat_page_contains_generic_demo_ui(self):
        page = render_chat_page()

        self.assertIn("Local Agent", page)
        self.assertIn("Ask this agent", page)
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
        self.assertNotIn("R 课程", page)
        self.assertNotIn("tibble", page)
        self.assertNotIn("ggplot2", page)

    def test_render_chat_page_uses_project_ui_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.toml"
            path.write_text(
                'title = "R Course Assistant"\n'
                'placeholder = "Ask the R course assistant"\n'
                'status_text = "Searching course knowledge..."\n'
                '[[welcome_items]]\n'
                'title = "Course logistics"\n'
                'text = "Answer class time and policy questions."\n'
                '[[demo_sources]]\n'
                'label = "syllabus.md"\n'
                'source = "syllabus.md"\n',
                encoding="utf-8",
            )

            page = render_chat_page(ui=load_ui_config(path))

            self.assertIn("R Course Assistant", page)
            self.assertIn("Ask the R course assistant", page)
            self.assertIn("Course logistics", page)
            self.assertIn("syllabus.md", page)


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

    def test_validate_cli_returns_zero_and_structured_payload_for_template_project(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            exit_code = cli_main(
                [
                    "validate",
                    "--project",
                    str(TEMPLATE_ROOT),
                    "--config",
                    str(TEMPLATE_ROOT / "runtime.toml"),
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["errors"], [])

    def test_validate_cli_returns_nonzero_for_contract_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config_path = root / "runtime.toml"
            config_path.write_text(
                'schema_version = "runtime.v1"\n'
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"\n'
                '[retrieval]\n'
                'provider = "vector"\n',
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                exit_code = cli_main(["validate", "--project", str(root), "--config", str(config_path)])

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["errors"][0]["code"], "UNKNOWN_RETRIEVER_PROVIDER")

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
            examples = root / "examples"
            examples.mkdir()
            (examples / "core-regression-questions.md").write_text(
                "| ID | Question | Expected |\n"
                "| --- | --- | --- |\n"
                "| C01 | class logistics? | Wednesday |\n",
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"facts.md#0","source":"facts.md","title":"Project facts","content":"Answer: class logistics are handled on Wednesday."}]}',
                encoding="utf-8",
            )

            report = demo_check(settings, dify_url=None)

            self.assertTrue(report["index_exists"])
            self.assertEqual(report["checks"][0]["top_source"], "facts.md")
            self.assertIn("Wednesday", report["checks"][0]["answer"])

    def test_demo_check_reads_project_regression_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("prompt", encoding="utf-8")
            examples = root / "examples"
            examples.mkdir()
            (examples / "core-regression-questions.md").write_text(
                "| ID | Question | Expected |\n"
                "| --- | --- | --- |\n"
                "| C01 | custom project fact? | custom |\n",
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"facts.md#0","source":"facts.md","title":"Facts","content":"Answer: custom project fact."}]}',
                encoding="utf-8",
            )

            report = demo_check(settings, dify_url=None)

            self.assertEqual(report["checks"][0]["question"], "custom project fact?")
            self.assertNotIn("上课时间", report["checks"][0]["question"])

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
