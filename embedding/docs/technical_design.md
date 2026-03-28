# 技术设计：索引向量化

## 1. 目标

定义用于从基准数据集生成索引输入工件的、与实现对齐的架构。

当前范围：

- 单一 CLI 入口：`main.py --dataset-path --config-path model_config.yaml`
- 在策略构建之前初始化共享的本地模型缓存
- 遍历 `model_config.yaml` 中所有已配置的模型
- 遍历固定的数据集候选项："hotpotqa_distractor_v1", "msmarco_v1", "scifact_v1", "trec_car_v1"
- 一次运行一个 `(dataset, model)` 的构建
- 在 `output/<dataset_name>/<model_id>/` 下生成确定的输出结构

## 2. 不可协商的约束

1. 公共接口必须遵循 [contract.md](../contract.md)。
2. 文档和查询的向量维度必须严格为 768。
3. 输出位置固定为：`output/<dataset_name>/<model_id>/`。
4. 任何 `(dataset, model)` 组合的失败都必须在进程退出时返回非零状态码。

## 3. 模块与职责

### 3.1 `main.py`（入口 + 编排器）

职责：

1. 解析 CLI（ `--dataset-path` 和 --config-path）。
2. 通过 `ConfigLoader.load_configs()` 加载所有模型构建器配置。
3. 使用固定的数据集候选项列表。
4. 执行嵌套循环：
   - 外层：按 YAML 中的顺序遍历模型配置
   - 内层：按固定顺序遍历数据集
5. 调用 `run_once(dataset_path, dataset_name, builder_cfg)`。
6. 当一次运行失败时立即停止并返回非零状态码。

CLI 接口：

- `--dataset-path`（必需）
- --config-path （必需）

内部固定源：

- 输出根目录：`output`
- 数据集："hotpotqa_distractor_v1", "msmarco_v1", "scifact_v1", "trec_car_v1"

### 3.2 支持组件

- `infra/config_loader.py`
  - 加载 YAML 配置
  - 从单个入口点构建每个模型的构建器配置
- `infra/dataset_loader.py`
  - 通过精确匹配后不区分大小写的唯一匹配来解析子数据集目录
  - 从 `dataset.json` 加载数据集上下文和文件路径
- `infra/embedding_strategies/*`
  - 根据 YAML 配置选择本地/HTTP 嵌入策略
- `infra/model_cache.py`
  - 为下载的模型初始化稳定的缓存目录
- `app/runner.py`
  - 执行文档/查询嵌入流水线
  - 写入工件、元数据和日志

## 4. 运行时序列

1. 解析 `--dataset-path`和--config-path。
2. 初始化共享的模型缓存目录。
3. 从 `model_config.yaml` 加载模型注册表。
4. 对于每个模型和每个固定数据集候选项：
   - 创建 `output/<dataset_name>/<model_id>/`
   - 使用加载的配置映射中的当前构建器配置
   - 在 `--dataset-path` 下解析数据集上下文
   - 构建嵌入策略
   - 运行 `BuilderRunner`
5. 如果所有运行成功则返回 `0`；否则返回非零状态码。

缓存行为：

- 对于给定模型，首次成功运行可能会将工件下载到共享缓存目录中。
- 后续运行会重用相同的本地缓存，并且不应重新下载相同的模型文件。
- 可以使用 `RETRIEVAL_SURVEY_MODEL_CACHE_DIR` 来固定跨机器或会话的缓存位置。

## 5. 数据模型（锁定的接口）

### 5.1 输入记录

输入文件从以下路径加载：

`<dataset_path>/<resolved_dataset_dir>/`

- `docs.jsonl`：`doc_id`，`doc_text`
- `train/queries.jsonl`：`query_id`，`query_text`
- `train/qrels.jsonl`：`query_id`，`doc_id`，`relevance`（用于验证兼容性）

### 5.2 输出记录

- docs parquet 行：
  - `doc_id： string`
  - `chunk_text： string`
  - `chunk_embedding： list<float>[<= embedding_dim]`
- queries parquet 行：
  - `query_id： string`
  - `query_text： string`
  - `query_embedding： list<float>[<= embedding_dim]`
- run metadata json：
  - 运行时间戳
  - model_id/提供商标识
  - 文档/查询计数器

## 6. 故障模式与可观测性

1. 业务异常在传播前于源层记录日志。
2. 日志条目包含函数名、行号、原因文本和上下文标识符。
3. 禁止静默吞没异常。
4. 编排返回码按组合快速失败：任何失败都返回非零状态码。

## 7. 验证计划

1. 契约一致性：
   - CLI、迭代顺序和退出语义与 `contract.md` 一致。
2. 接口一致性：
   - README 命令与 argparse 签名一致。
3. 回归检查：
   - 单元测试套件运行时无接口回归。

## 8. 需求可追溯性矩阵

| 需求 | 契约参考 | 验证证据 |
|---|---|---|
| 固定模型配置源 | 契约第 2.1 节 | 配置加载器路径检查 |
| 固定数据集候选项 | 契约第 2.1/2.3 节 | 编排器循环检查 |
| 维度不超过 embedding_dim | 契约第 2.1/4 节 | 运行时维度断言 + 故障路径测试 |
| 仅通过 YAML 选择本地/HTTP 策略 | 契约第 2.3 节 | 策略选择测试 |
| 模型输出日志路径和异常详情 | 契约第 5 节 | 日志检查测试 |
