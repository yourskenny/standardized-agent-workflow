# Step-by-Step SOP

这份 SOP 用于从零创建一个新的智能体项目。

## 第 1 步：复制模板

```powershell
Copy-Item -Recurse templates\agent-project my-agent-project
```

## 第 2 步：填写项目简报

编辑：

```text
my-agent-project/PROJECT_BRIEF.md
```

必须先写清楚：

- 智能体服务谁
- 主要回答什么
- 明确不能做什么
- 哪些问题高风险
- 没有依据时怎么回答
- 谁负责维护

## 第 3 步：整理资料边界

把资料分成三类：

```text
用户可见资料
维护者资料
不能进入知识库的资料
```

用户可见资料进入 `knowledge_base/`。

维护者资料留在 `maintenance/`、`examples/`、`_templates/`、`_manifests/`。

隐私、个人作品、内部敏感资料不进入任何用户知识库。

## 第 4 步：整理核心事实

复制：

```text
knowledge_base/_templates/core-facts-template.md
```

放到：

```text
knowledge_base/domain_specific/current/core-facts.md
```

填写最高频、最高风险、最容易被模型编错的信息。

## 第 5 步：整理问答核查表

复制：

```text
knowledge_base/_templates/faq-table-template.md
```

把常见问题写成“问题、答案、依据、边界”。

## 第 6 步：整理意图映射

编辑：

```text
agent/intent-map.md
```

把用户口语表达映射到正式字段。

## 第 7 步：编写系统提示词

编辑：

```text
agent/system-prompt.md
```

至少包含：

- 身份边界
- 知识库范围
- 回答优先级
- 核心事实
- 高风险问题处理
- 拒答与替代帮助方式
- 回答风格

## 第 8 步：更新上传清单

编辑：

```text
knowledge_base/_manifests/current-upload-manifest.md
```

明确哪些文件上传，哪些文件绝不上传。

## 第 9 步：发布到平台

在 Dify 或其他平台中：

1. 上传清单中的知识文件。
2. 等待索引完成。
3. 绑定知识库。
4. 复制系统提示词。
5. 保存并发布。
6. 使用新会话测试。

## 第 10 步：运行回归测试

编辑并执行：

```text
examples/core-regression-questions.md
```

测试必须覆盖：

- 高频事实
- 高风险事实
- 同义词问法
- 没有资料的问题
- 拒答边界
- 领域专业问题

## 第 11 步：填写记录

填写：

```text
maintenance/test-records/<version>-regression.md
maintenance/update-log.md
```

没有记录的更新，视为没有完成。

