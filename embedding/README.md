# 索引向量化

## 目的

本仓库包含用于构建检索基准测试所使用的索引输入工件的治理基线。

当前运行时入口：

`python embedding/main.py --dataset-path <datasets_root> --config-path model_config.yaml`

该入口脚本会遍历所有已配置的模型以及固定的数据集（"hotpotqa_distractor_v1", "msmarco_v1", "scifact_v1", "trec_car_v1"），并生成可用于索引和评估的嵌入工件。

## 范围

本治理包定义了：

- 必需的接口和运行时行为
- 输入和输出工件的数据契约
- 代理级别的实现约束
- 设计/契约质量的阶段关卡

## 文档优先策略

- 契约：[contract.md](./contract.md)
- 代理约束：[agent.md](./agent.md)
- 技术设计：[docs/technical_design.md](./docs/technical_design.md)

## 运行时工作流程

1. 使用 `--dataset-path` 参数运行 `main.py`。
2. 从 `model_config.yaml` 加载构建器配置。
3. 初始化一个共享的本地模型缓存，路径为 `~/.cache/retrieval_survey/models`（除非通过环境变量覆盖）。
4. 遍历固定的数据集列表："hotpotqa_distractor_v1", "msmarco_v1", "scifact_v1", "trec_car_v1"。
5. 对每个数据集和模型的组合执行 `run_once(dataset, model_id)`。
6. 当任何组合失败时，以非零状态退出。

## 前提条件

- Python 3.10+
- `PyYAML`
- `pyproject.toml` 或 `requirements/base.txt` 中定义的其他运行时依赖

## 命令行接口

```bash
python embedding/main.py --dataset-path datasets --config-path model_config.yaml
```

安装后：

```bash
retrieval-embedding --dataset-path datasets --config-path model_config.yaml
```

参数契约：

- `--dataset-path`（必需）：数据集根目录
- `--config-path` （必需）：配置参数文件

内部固定源：

- 输出根目录：`output`
- 数据集候选项："hotpotqa_distractor_v1", "msmarco_v1", "scifact_v1", "trec_car_v1"

模型缓存行为：

- 首次运行时，模型会下载到共享的缓存目录中。
- 后续运行会重用相同的本地缓存，避免重复下载。
- 设置 `RETRIEVAL_SURVEY_MODEL_CACHE_DIR` 可更改缓存根目录。

预热工作流程：

1. 在具备网络访问权限的情况下运行流水线一次，以便将每个配置的模型拉取到本地缓存中。
2. 后续评估时重新运行相同命令，流程应重用磁盘上的缓存文件。
3. 如需指定专用的缓存位置，请在首次运行前设置 `RETRIEVAL_SURVEY_MODEL_CACHE_DIR`，并在之后持续使用同一路径。

## 预期的数据集目录结构

```text
datasets/
├── hotpotqa_distractor_v1/
│   ├── dataset.json
│   ├── docs.jsonl
│   └── train/
│       ├── queries.jsonl
│       └── qrels.jsonl
├── msmarco_v1/
├── scifact_v1/
└── trec_car_v1/
```

每个已解析的子数据集目录必须包含 `dataset.json`、文档文件以及 `dataset.json` 中定义的训练查询文件。

## 输出目录结构契约

```text
output/
└── <dataset_name>/
    └── <model_id>/
        ├── docs_dim<embedding_dim>.parquet
        └── queries_dim<embedding_dim>.parquet
```

输出路径和文件名为固定的契约接口，未经契约修订批准不得更改。

## 测试命令

```bash
python -m py_compile $(rg --files embedding -g"*.py")
```
