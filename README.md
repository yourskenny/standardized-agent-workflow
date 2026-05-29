# Standardized Agent Workflow

一个可直接复用的智能体项目模板，用于把某个具体场景的智能体建设，抽象成标准化、易维护、可测试、可迁移的工作流。

本模板来自一个课程智能体项目的实践经验，但并不限定于课程场景。它适合用于：

- 课程智能体
- 知识库问答智能体
- 内部制度/流程助手
- 专业学习辅导助手
- 面向特定资料库的 RAG 应用

核心思想：

```text
领域资料
  -> 入库前抽象与审核
  -> 面向检索的知识库
  -> 元信息与风险分级
  -> 智能体行为协议
  -> 发布清单
  -> 回归测试集
  -> 更新记录
  -> 持续维护流程
```

## 快速开始

1. 复制模板项目：

   ```powershell
   Copy-Item -Recurse templates\agent-project my-agent-project
   ```

2. 阅读并填写：

   - `my-agent-project/PROJECT_BRIEF.md`
   - `my-agent-project/agent/system-prompt.md`
   - `my-agent-project/knowledge_base/_templates/`
   - `my-agent-project/examples/core-regression-questions.md`

3. 按清单推进：

   - `checklists/01-agent-design-checklist.md`
   - `checklists/02-knowledge-base-checklist.md`
   - `checklists/03-release-and-test-checklist.md`

4. 发布前确认：

   - 知识库只包含用户应该看到的资料。
   - 高风险事实有明确来源。
   - 系统提示词与知识库事实一致。
   - 核心回归测试已通过并记录。

## 推荐阅读顺序

1. [docs/00-step-by-step-sop.md](docs/00-step-by-step-sop.md)
2. [docs/01-core-principles.md](docs/01-core-principles.md)
3. [docs/02-agent-project-structure.md](docs/02-agent-project-structure.md)
4. [docs/03-knowledge-base-design.md](docs/03-knowledge-base-design.md)
5. [docs/04-agent-behavior-protocol.md](docs/04-agent-behavior-protocol.md)
6. [docs/05-maintenance-and-regression-testing.md](docs/05-maintenance-and-regression-testing.md)
7. [docs/06-from-course-agent-to-general-agent.md](docs/06-from-course-agent-to-general-agent.md)
8. [docs/07-pre-ingestion-abstraction-workflow.md](docs/07-pre-ingestion-abstraction-workflow.md)
9. [docs/08-local-rag-reference-implementation.md](docs/08-local-rag-reference-implementation.md)
10. [docs/09-general-agent-runtime-architecture.md](docs/09-general-agent-runtime-architecture.md)

## 通用自建智能体运行时

本仓库现在包含一个轻量级自建 RAG 运行时：

```text
runtime/local_rag_agent/
```

它可以读取任意符合模板结构的智能体项目：系统提示词、上传清单、Markdown 知识库和回归问题，在本地生成 JSON 索引，并提供 `ingest`、`chat`、`serve`、`regression` 命令。

第一个验证对象是 R 课程智能体，但运行时本身不是课程专用代码。它的目标是成为通用智能体项目的本地验证和演示底座：同一套运行时，通过不同 TOML 配置和不同知识库，可以服务不同课程、内部助手或未来的营销智能体原型。

使用说明见：

- [runtime/local_rag_agent/README.md](runtime/local_rag_agent/README.md)
- [docs/09-general-agent-runtime-architecture.md](docs/09-general-agent-runtime-architecture.md)

## 三层抽象

```text
特定智能体
  解决一个明确场景，例如某门课程、某个团队流程、某类业务资料。

通用领域智能体
  抽出领域内通用结构，例如所有课程都需要课程事实、资料边界、学术诚信、测试问题。

通用智能体工作流
  抽出所有 RAG/知识库智能体共有的工程流程：资料治理、行为边界、发布同步、测试和维护。
```

## 最重要的原则

智能体不是越会编越好，而是：

- 有依据的信息，准确回答。
- 没有依据的信息，明确不知道。
- 用户口语化表达，能映射到正式资料。
- 高风险问题，保守处理。
- 维护者能看懂、能更新、能测试、能迁移。
- 有价值但不适合直接公开的资料，先抽象、审核，再进入正式知识库。
