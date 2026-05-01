# Maintenance Workflow

## 初始化

1. 填写 `PROJECT_BRIEF.md`。
2. 整理当前周期资料到 `knowledge_base/domain_specific/current/`。
3. 整理长期资料到 `knowledge_base/stable_materials/`。
4. 整理边界规则到 `knowledge_base/policy_and_boundaries/`。
5. 编写 `agent/system-prompt.md`。
6. 更新上传清单。
7. 发布到平台并运行核心回归测试。
8. 填写测试记录和更新记录。

## 周期中更新

1. 判断更新类型。
2. 检查资料是否可进入用户知识库。
3. 更新本地知识库。
4. 更新上传清单。
5. 同步平台知识库并等待索引。
6. 更新系统提示词并发布。
7. 使用新会话运行回归测试。
8. 填写记录。

## 归档

1. 锁定当前周期资料。
2. 复制或移动到 `knowledge_base/archive/`。
3. 保留上传清单、测试记录和更新记录。
4. 基于当前结构准备下一周期。

