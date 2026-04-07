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

1. 解析 CLI（`--dataset-path` 和 `--config-path`）。
2. 通过 `ConfigLoader.load_configs()` 加载所有模型构建器配置。
3. 使用固定的数据集候选项列表。
4. 执行嵌套循环：
   - 外层：按 YAML 中的顺序遍历模型配置
   - 内层：按固定顺序遍历数据集
5. 调用 `run_once(dataset_path, dataset_name, builder_cfg)`。
6. 当一次运行失败时立即停止并返回非零状态码。

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
- `services/chunk_selector.py`
  - 完成候选切分、chunk embedding、文档中心向量计算和 MMR TopN 选择


## 4. 运行时序列

1. 解析 `--dataset-path` 和 `--config-path`。
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

## 5. 数据模型

### 5.1 输入记录

输入文件从以下路径加载：

`<dataset_path>/<resolved_dataset_dir>/`

- `docs.jsonl`：`doc_id`，`doc_text`
- `train/queries.jsonl`：`query_id`，`query_text`
- `train/qrels.jsonl`：`query_id`，`doc_id`，`relevance`

### 5.2 输出记录

- docs parquet 行：
  - `doc_id: string`
  - `chunk_id: string`
  - `chunk_text: string`
  - `chunk_vector: list<float>[<= embedding_dim]`
  - `chunk_score: float`
  - `chunk_rank: int`
- queries parquet 行：
  - `query_id: string`
  - `query_text: string`
  - `query_embedding: list<float>[<= embedding_dim]`

## 6. 文档 chunk 选择

`chunk_selector.py` 中的候选构建与选择函数对单篇文档执行如下步骤：

1. 将 `doc_text` 统一视为有序文本片段序列；单元素输入按连续双换行做初切，列表项优先并入前一块。
2. 每个逻辑块独立处理；块未超过 `target_tokens` 时直接保留。
3. 块超过 `target_tokens` 时优先按句边界继续切，并按顺序 greedily 合并到目标上限内。
4. 若句切失败或单句超过 `hard_max_tokens`，退化到字符级切分以保证上限约束。
5. `process_doc` 按固定文档窗口聚合输入，为每个文档记录候选块及其在窗口 chunk 列表中的 offset。
   - 服务层先把原始 `doc_text` 归一化为文本片段序列：`str -> [str]`，`list[str] -> 过滤空项后的有序片段序列`
   - 该归一化仅是输入适配，不声明 `list[str]` 元素等于 sentence
6. 窗口内全部 chunk 文本一次性交给 embedding 策略，策略内部再按 `inference.batch_size` 完成推理层分批。
   - 运行日志默认记录 chunk 的估算 token 统计，而不是字符数统计
   - `ChunkSplitter` 的 token 长度是启发式估算，不等同于模型 tokenizer 的精确长度
   - 本地 provider 可在可访问 tokenizer 时额外记录 prepared input 的真实 token 统计；HTTP provider 默认不提供该统计
7. 返回的大矩阵按文档 offset 切回单篇文档，对每篇文档的候选向量统一做 L2 归一化。
8. 对单篇文档的全部候选向量做均值 pooling，再归一化，得到 `doc_centroid`。
9. 计算 `rep_score = cosine(chunk_emb, doc_centroid)`。
10. 用 MMR 进行 TopN 选择：
   - 第 1 个 chunk 取最高 `rep_score`
   - 后续 chunk 取 `mmr_lambda * rep_score - (1 - mmr_lambda) * max_sim_to_selected`
11. 若候选数不超过 `top_n`，则全部保留，并按输入文档顺序直接写出结果。

默认参数：

- `hard_max_tokens = 8092`
- `target_tokens = min(1200, max(400, int(hard_max_tokens * 0.5)))`
- `top_n = 3`
- `mmr_lambda = 0.7`

## 6.1 Query 向量化

`process_query` 按 `collect -> encode -> save` 的固定窗口流水线处理查询：

1. 从 `queries.jsonl` 中按输入顺序收集 query，服务层固定窗口大小为 `5000`。
2. 每个窗口内完成 `query_id` 和 `query_text` 的非空校验，并保留原始行号用于错误日志。
3. 窗口内全部 `query_text` 一次性交给 embedding 策略。
4. embedding 策略内部继续按 `inference.batch_size` 执行推理层分批。
5. 返回向量矩阵后，按窗口顺序组装为 `query_id/query_text/query_embedding` 记录并立即写出。
6. 最后一个不足 `2000` 条的窗口按相同流程处理。

## 7. 故障模式与可观测性

1. 业务异常在传播前于源层记录日志。
2. 日志条目包含函数名、行号、原因文本和上下文标识符。
3. 禁止静默吞没异常。
4. 编排返回码按组合快速失败：任何失败都返回非零状态码。
5. 运行开始和结束时打印模型、数据集路径、时间戳、文档数、chunk 数和查询数。

## 8. 验证计划

1. 契约一致性：
   - CLI、迭代顺序和退出语义与 `contract.md` 一致。
2. 接口一致性：
   - docs 输出字段、日志摘要和查询输出与设计一致。
3. 回归检查：
   - 单元测试和集成测试覆盖 MMR 选择、全量重写输出和 metadata 移除后的行为。
