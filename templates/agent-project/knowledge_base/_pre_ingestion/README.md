# Pre-Ingestion Workspace

本目录用于处理有价值但不适合直接进入用户可见知识库的资料。

## 用途

- 保存原始资料的处理记录。
- 保存逐份摘要卡片。
- 生成主题地图。
- 暂存候选知识条目。

## 边界

本目录默认不上传到智能体知识库。只有经过抽象、审核并移入正式知识库目录的条目，才可以加入上传清单。

## 推荐结构

```text
_pre_ingestion/
  <source_batch>/
    README.md
    source_inventory.md
    item_digests/
    theme_map.md
    candidate_kb_entries/
```

