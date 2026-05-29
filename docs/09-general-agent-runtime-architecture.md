# 通用智能体本地架构与项目结构

本文记录当前版本的通用智能体运行架构。它来自 R 课程智能体的本地复刻验证，但目标不是只服务一门课，而是把“知识库 + 提示词 + 检索 + 模型 + 前端展示 + 回归测试”沉淀成可迁移的通用模板。

当前阶段的定位很清楚：先做到本地可跑、可解释、可测试、可演示。后续再逐步把检索、工作流、评测、观测和部署能力做强。

## 一、设计目标

这个架构要解决四件事：

1. 让一个智能体项目不再依赖 Dify 才能运行。
2. 继续复用现有的知识库、上传清单、系统提示词和测试问题。
3. 让每一步都能被维护者看见：哪些文件被入库、检索到了哪些片段、回答引用了哪些来源。
4. 让同一套运行时可以服务不同课程、不同领域和未来的营销智能体验证。

它不是要立刻替代所有平台能力。Dify 的可视化后台、托管发布、权限管理和工作流设计器仍然有价值。当前版本先替代最核心的问答链路：

```text
项目配置
  -> 上传清单
  -> Markdown 知识库
  -> 本地索引
  -> 检索
  -> 系统提示词 + 检索上下文 + 对话历史
  -> OpenAI-compatible 模型
  -> 带引用的聊天界面
  -> 回归测试证据
```

## 二、仓库层级

本仓库分成三层：

```text
standardized-agent-workflow/
  docs/                         通用方法论、架构说明和维护流程
  checklists/                   项目设计、知识库、发布测试检查清单
  templates/agent-project/      新智能体项目模板
  runtime/local_rag_agent/      通用本地 RAG 运行时
  examples/                     仓库级示例
```

### 1. `templates/agent-project/`

这是创建新智能体项目的内容模板。一个新的课程智能体、业务助手或营销智能体，都可以从这里复制一份。

关键文件：

```text
PROJECT_BRIEF.md
agent/system-prompt.md
agent/answer-policies.md
agent/intent-map.md
knowledge_base/
knowledge_base/_manifests/current-upload-manifest.md
examples/core-regression-questions.md
maintenance/update-log.md
maintenance/test-records/
```

这个目录回答的是：智能体服务谁、能回答什么、不能回答什么、知识库有哪些资料、如何测试和维护。

### 2. `runtime/local_rag_agent/`

这是当前自建架构的运行时。它不是某个课程的代码，而是读取“任意符合模板结构的智能体项目”，并在本地完成入库、检索、模型回答和网页展示。

关键模块：

```text
local_rag_agent/config.py       读取项目配置并限制路径边界
local_rag_agent/manifest.py     解析上传清单
local_rag_agent/chunking.py     Markdown 分块
local_rag_agent/index_store.py  本地 JSON 索引读写
local_rag_agent/retrieval.py    透明 lexical 检索
local_rag_agent/agent.py        构造提示词、上下文和回答
local_rag_agent/llm.py          OpenAI-compatible 模型调用
local_rag_agent/server.py       本地网页和 API 服务
local_rag_agent/regression.py   回归测试输出
local_rag_agent/cli.py          命令入口
```

### 3. 内容项目

内容项目不一定在本仓库内。例如当前验证使用：

```text
C:\coding\syllabus_R\course-agent-r
```

它提供知识库、提示词和上传清单；运行时只读取它，不把课程内容硬编码进通用库。

## 三、通用项目配置

运行时通过一个 TOML 文件把“通用运行时”和“具体智能体项目”连接起来。R 课程示例位于：

```text
runtime/local_rag_agent/examples/r-course-agent.toml
```

结构如下：

```toml
[project]
prompt_path = "dify/app-prompt.md"
knowledge_root = "knowledge_base"
manifest_path = "knowledge_base/_manifests/dify-upload-manifest-2026-spring.md"
index_path = ".local_rag_agent/index.json"

[retrieval]
chunk_size = 1200
chunk_overlap = 160
top_k = 5

[regression]
output_dir = ".local_rag_agent/regression"
```

迁移到新项目时，通常只需要改：

- `prompt_path`
- `knowledge_root`
- `manifest_path`
- `index_path`
- `top_k`

这让运行时保持通用，项目差异留在配置和内容里。

## 四、建构流程

### 第 1 步：创建或选择智能体项目

新项目从模板复制：

```powershell
Copy-Item -Recurse templates\agent-project my-agent-project
```

已有 Dify 项目则先确认它是否具备：

```text
系统提示词
知识库目录
上传清单
核心测试问题
维护记录
```

### 第 2 步：整理知识库边界

所有资料先分为三类：

```text
用户可见资料       可以进入 knowledge_base/
维护者资料         放在 maintenance/、examples/、_templates/
不应入库资料       不进入用户知识库
```

如果资料有参考价值但不适合原文进入知识库，例如学生作品、内部案例、敏感记录，应先走入库前抽象流程：

```text
原始资料
  -> _pre_ingestion/
  -> 抽象成结构、方法、边界、FAQ
  -> 审核
  -> 进入正式 knowledge_base/
```

### 第 3 步：维护上传清单

上传清单是运行时和平台之间的共同契约。它明确哪些文件参与知识库构建。

```text
knowledge_base/_manifests/current-upload-manifest.md
```

本地运行时只读取清单中列出的 Markdown 文件和目录。这样同一份清单既可用于 Dify，也可用于自建运行时。

### 第 4 步：建立本地索引

```powershell
cd C:\coding\standardized-agent-workflow
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"

python -m local_rag_agent ingest `
  --project C:\path\to\agent-project `
  --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\your-agent.toml
```

输出文件默认在内容项目中：

```text
.local_rag_agent/index.json
```

这个 JSON 索引是可检查的。维护者可以直接看到每个 chunk 的来源、标题和内容。

### 第 5 步：本地问答验证

无模型 key 时，运行时会给出确定性的检索式回答，适合先检查召回是否正确。

```powershell
python -m local_rag_agent chat `
  --project C:\path\to\agent-project `
  --config C:\path\to\agent.toml `
  "核心问题"
```

有模型 key 时，运行时使用 OpenAI-compatible API：

```powershell
$env:LOCAL_AGENT_API_KEY="..."
$env:LOCAL_AGENT_BASE_URL="https://api.deepseek.com"
$env:LOCAL_AGENT_MODEL="deepseek-v4-flash"
```

也支持：

```text
OPENAI_API_KEY
OPENAI_BASE_URL
OPENAI_MODEL
```

### 第 6 步：启动本地网页

```powershell
python -m local_rag_agent serve `
  --project C:\path\to\agent-project `
  --config C:\path\to\agent.toml `
  --port 8765
```

打开：

```text
http://127.0.0.1:8765/
```

当前网页包含：

- 类 Dify 的嵌入式聊天界面；
- 自建架构品牌标识；
- 多轮上下文；
- 固定底部输入框；
- 回答来源 chips；
- 可点击的知识库片段弹窗；
- 课程站入口式外壳，便于演示。

### 第 7 步：回归测试

```powershell
python -m local_rag_agent regression `
  --project C:\path\to\agent-project `
  --config C:\path\to\agent.toml `
  --questions C:\path\to\agent-project\examples\core-regression-questions.md
```

输出：

```text
.local_rag_agent/regression/*.jsonl
```

这些文件用于记录本地行为证据。默认不提交，除非项目明确需要归档。

## 五、当前运行时的数据流

```text
用户问题
  -> 如果有历史对话，拼接最近用户问题形成检索 query
  -> rank_chunks() 在 index.json 中找 top_k chunks
  -> build_messages()
       system prompt
       local runtime instruction
       最近历史消息
       当前问题 + 检索片段
  -> OpenAICompatibleClient.chat()
  -> answer + sources
  -> 前端渲染回答
  -> 点击 source chip 查看 chunk 内容
```

这种设计的好处是：模型回答和引用来源之间有明确连接，维护者可以追查某个回答到底依赖了哪些片段。

## 六、当前能力边界

当前版本已经具备：

- 读取任意项目配置；
- 解析上传清单；
- Markdown 分块；
- 本地 JSON 索引；
- 透明 lexical 检索；
- 无模型时的确定性兜底回答；
- OpenAI-compatible 模型调用；
- 多轮上下文；
- 本地网页演示；
- 来源展示和知识库片段弹窗；
- 单元测试和回归测试命令。

当前版本尚未包含：

- 嵌入向量检索；
- 向量数据库；
- 复杂工作流编排；
- 工具调用；
- 用户账户和权限；
- 在线部署脚本；
- 管理后台；
- 完整观测系统。

这些不是缺陷，而是后续迭代方向。当前优先级是把最小闭环做清楚、做稳、做成模板。

## 七、相比 Dify 和 LangGraph 的改造方向

短期内不应宣称已经超过 Dify 或 LangGraph。更有价值的方向是把它们的优势拆解出来，逐步做成更适合教学和领域智能体项目的架构。

### 相比 Dify

Dify 优势是快速搭建、可视化配置、托管发布方便。自建架构的发力点应是：

- 配置全部代码化，便于版本控制；
- 上传清单、知识库、提示词、测试记录统一管理；
- 来源片段可追溯；
- 回归测试成为发布流程的一部分；
- 同一运行时服务多个课程或领域；
- 可按教学场景定制前端，而不是受平台 Web App 约束。

### 相比 LangGraph

LangGraph 优势是复杂状态机和多节点工作流。自建架构的发力点应是：

- 先把单智能体 RAG 闭环做稳定；
- 再把检索、生成、校验、拒答、工具调用拆成可替换节点；
- 用项目配置描述节点，而不是过早写复杂图；
- 保持维护者能看懂和测试。

## 八、下一轮值得做的通用化改造

建议按以下顺序迭代：

1. 多项目配置目录：`runtime/local_rag_agent/examples/` 下增加通用模板配置。
2. 检索器接口化：把 lexical retriever 抽象成可替换模块。
3. Embedding 检索：增加可选 embedding index，但保留 lexical debug 模式。
4. 引用质量评测：检查回答是否引用了 top sources，是否出现无依据事实。
5. 前端主题参数化：允许项目配置名称、颜色、欢迎语和入口外壳。
6. 回归报告可视化：把 JSONL 测试结果生成 HTML 报告。
7. 部署方案：把本地服务迁移到可控服务器或轻量容器。
8. 工作流节点化：把回答前检查、边界判断、工具调用逐步纳入统一协议。

## 九、当前版本的复用标准

一个新项目如果想接入当前通用运行时，至少应满足：

- 有一个明确的项目根目录；
- 有系统提示词文件；
- 有 `knowledge_base/`；
- 有上传清单；
- 知识文件使用 Markdown；
- 有核心回归问题；
- 有一个 TOML 配置指向上述文件；
- 能通过 `ingest`、`chat`、`serve`、`regression` 四个命令。

达到这个标准后，它就不再只是一个平台配置，而是一个可迁移、可测试、可版本化的智能体项目。
