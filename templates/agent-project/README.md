# Agent Project Template

这是一个可复制的智能体项目模板。使用时先填写 `PROJECT_BRIEF.md`，再整理知识库、运行时配置和系统提示词。

推荐流程：

1. 填写 `PROJECT_BRIEF.md`。
2. 配置 `runtime.toml`，确认 prompt、知识库、manifest、intent、workflow、policy 和 tool 配置路径。
3. 配置 `agent/intents.toml`。新增意图时优先改这里，再同步 `agent/intent-map.md` 作为维护者说明。
4. 配置 `agent/workflows.toml`。先使用内置 workflow：`rag_qa`、`retrieval_debug`、`refusal_with_guidance`。
5. 配置 `agent/policies.toml`，维护拒答、无证据和高风险问题策略。
6. 配置 `agent/tools.toml`。工具默认禁用，只有实现并验证后再启用。
7. 编写 `agent/system-prompt.md` 和 `agent/answer-policies.md`。
8. 整理 `knowledge_base/domain_specific/current/`。
9. 整理 `knowledge_base/stable_materials/`。
10. 整理 `knowledge_base/policy_and_boundaries/`。
11. 更新 `knowledge_base/_manifests/current-upload-manifest.md`。
12. 运行 `examples/core-regression-questions.md`。
13. 将结果写入 `maintenance/test-records/`。
14. 将变更写入 `maintenance/update-log.md`。
