# 智能体项目结构

推荐每个智能体项目采用以下结构：

```text
agent-project/
  PROJECT_BRIEF.md
  README.md

  agent/
    system-prompt.md
    intent-map.md
    answer-policies.md

  knowledge_base/
    _templates/
    _manifests/
    domain_specific/
      current/
    stable_materials/
    policy_and_boundaries/
    archive/

  maintenance/
    maintenance-workflow.md
    update-checklist.md
    update-log.md
    test-records/

  examples/
    core-regression-questions.md
    extended-test-questions.md

  scripts/
```

## 目录职责

### `PROJECT_BRIEF.md`

定义这个智能体是什么、服务谁、回答什么、不回答什么、谁维护。

这是新项目的第一份文件。

### `agent/`

保存智能体的行为协议：

- `system-prompt.md`：系统提示词源文件。
- `intent-map.md`：用户常见问法到正式字段的映射。
- `answer-policies.md`：拒答、保守回答、引用来源、代码回答等规则。

### `knowledge_base/`

保存知识库源文件。平台中的知识库只是发布版本，本目录才是维护源头。

### `knowledge_base/_templates/`

新增知识文件时复制这里的模板。模板不上传给用户端知识库。

### `knowledge_base/_manifests/`

记录每次发布应上传哪些文件。清单本身不上传给用户端知识库。

### `knowledge_base/domain_specific/current/`

存放当前版本、当前学期、当前业务周期会变化的资料。

课程智能体中，这里可对应：

```text
semester_specific/<semester>/
```

企业流程智能体中，这里可对应：

```text
current_policy/
current_workflow/
current_product_docs/
```

### `knowledge_base/stable_materials/`

存放长期稳定的学习资料、背景知识、通用方法和示例。

### `knowledge_base/policy_and_boundaries/`

存放长期有效的边界规则，例如：

- 不能编造
- 不能代写
- 不能泄露内部资料
- 不能处理个人隐私
- 没有依据时如何回答

### `maintenance/`

维护者使用，不上传给用户端知识库。

### `examples/`

测试问题和预期要点。每次更新后必须运行核心回归问题。

## 用户可见资料和维护资料分离

判断规则：

- 如果文件回答“用户可以问什么、事实是什么、如何学习或操作”，通常可进入用户知识库。
- 如果文件回答“维护者怎么上传、怎么测试、怎么更新”，只留在本地。
- 如果文件包含个人资料、隐私、内部敏感信息，不进入任何公共知识库。

