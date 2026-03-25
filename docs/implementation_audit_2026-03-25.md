# Index Input Pipeline 实现审计报告（2026-03-25）

## 1. 审计结论摘要

- 审计基准：`docs/technical_design.md` + `contract.md`
- 审计范围：`index_scheduler.py`、`build_index_inputs.py`、`index_builder/`、`chunk_selector/`、`tests/`
- 总体判断：**核心流程可运行，但尚未达到“契约严格一致 + 工业级组织”标准**。
- 关键结论：
  - `P0`（阻断级）缺口 3 项：`failures.jsonl` 契约不一致、错误分类未闭环、测试可发现性/可执行性不稳定。
  - `P1`（高优先级）缺口 5 项：scheduler 验证边界、768 维强约束、异常映射一致性、模块耦合方式、关键契约测试缺口。
  - `P2`（优化级）缺口 4 项：注释语言一致性、重复 HTTP 客户端逻辑、日志结构化增强、包化边界清晰化。

---

## 2. 需求追踪矩阵（Technical Design / Contract -> 实现）

| Requirement | 状态 | 证据 | 测试覆盖 | 风险 |
|---|---|---|---|---|
| Builder CLI 参数签名与默认值 | 已实现 | `build_index_inputs.py:11-17` | 间接覆盖（无 CLI E2E） | 低 |
| Scheduler CLI 参数签名与默认值 | 已实现 | `index_scheduler.py:22-28` | 缺失 | 中 |
| Scheduler 单模型/全模型模式 | 已实现 | `index_scheduler.py:35-40` | 缺失 | 中 |
| Scheduler 单模型失败不提前退出 | 已实现 | `index_scheduler.py:43-85` | 缺失 | 中 |
| 全模型执行顺序 deterministic | 部分实现 | `index_scheduler.py:40`（依赖 YAML 键顺序） | 缺失 | 中 |
| dataset 子目录 exact -> case-insensitive -> ambiguous error | 已实现 | `index_builder/dataset.py:18-37` | `tests/index_builder/test_dataset.py:9-34` | 低 |
| 必需输入文件存在性校验（dataset.json/docs/train queries） | 已实现 | `index_builder/dataset.py:40-63` | 间接覆盖 | 低 |
| Embedding strategy 仅由 YAML 决定（local/http） | 已实现 | `index_builder/config.py`（读取）、`index_builder/embedding.py:169-172`（路由） | `tests/index_builder/test_config.py`、`tests/index_builder/test_embedding.py` | 低 |
| HTTP payload 契约 `{"chunks":[...]}`/`{"vectors":[...]}` | 已实现 | `index_builder/embedding.py:124-146`, `chunk_selector/embedding_client.py:24-33` | `tests/index_builder/test_embedding.py` | 低 |
| Top1 chunk（每 doc 一条） | 已实现 | `index_builder/processors.py:117-136` + `build_chunk_selector(... chunk_num=1)` at `:47-50` | 部分覆盖 | 低 |
| Query 仅 train split | 已实现 | `index_builder/dataset.py:55-62`, `index_builder/pipeline.py:91-97` | 间接覆盖 | 低 |
| 输出向量维度严格检查 | 部分实现 | `ensure_embedding_shape` + `index_builder/processors.py:125-129,220` | 覆盖存在 | 中（未强制固定 768） |
| 固定 768 维（不可配置为其它） | 未实现 | 运行时读 `embedding_dim`，无 `==768` 断言 | 缺失 | 高 |
| 增量重试与 merge overwrite 语义 | 已实现 | `index_builder/pipeline.py:65-87`, `index_builder/io_utils.py:75-92` | 间接覆盖 | 中 |
| `failures.jsonl` schema（doc_id/error_type/error_message/timestamp_utc） | **未实现** | 当前写入 `record_type/record_id/stage` at `index_builder/processors.py:31-38` | 测试固化了当前非契约格式（`tests/index_builder/test_pipeline.py:108`） | **高** |
| `run_metadata.json` 必填顶层与关键字段 | 已实现 | `index_builder/metadata.py:28-70` | `tests/index_builder/test_pipeline.py:110-118`（部分） | 低 |
| 日志路径 `output/<dataset>/<model>/app.log` | 已实现 | `index_builder/pipeline.py:43-46` | 间接覆盖 | 低 |
| 失败日志包含 exception class/message/doc_id | 已实现 | `index_builder/processors.py:140-145,230-235` | 间接覆盖 | 低 |
| canonical error taxonomy 完整闭环 | **未实现** | 缺少 `SerializationError`、`UnexpectedRuntimeError`（`index_builder/errors.py:1-14`）且未统一映射 | 缺失 | **高** |

说明：本表以“契约可验证一致性”为判定标准；“部分实现”表示功能存在但对契约的严格性/可验证性不足。

---

## 3. 工业级组织评分卡

评分标准：1（差）- 5（优）

| 维度 | 得分 | 结论 |
|---|---:|---|
| 模块边界清晰度 | 2.5/5 | `index_builder` 通过动态 `sys.path` 注入耦合 `chunk_selector`（`index_builder/processors.py:42-45`），跨包边界脆弱。 |
| 错误模型与治理 | 2/5 | taxonomy 不完整、异常类型未统一映射、输出失败记录 schema 漂移。 |
| 可测试性 | 2/5 | 默认测试发现路径不稳定（`unittest discover` 直接失败）；大量契约级行为无测试。 |
| 可运维性/可观测性 | 3/5 | 有 start/end + failure 日志与 metadata，但失败文件格式与契约不一致影响下游治理。 |
| 可扩展性 | 3/5 | 配置驱动与策略模式已具备，但重复 HTTP 客户端逻辑、边界未包化限制演进。 |
| 总分 | **2.5/5** | 可作为原型/内部工具运行，离工业级上线标准仍有明显缺口。 |

---

## 4. 分级问题单（Findings First）

### P0（阻断契约/线上风险）

1. `failures.jsonl` 输出 schema 与契约不一致
- 影响：下游重试工具、审计工具、合同验收失败。
- 触发：任意失败记录落盘时。
- 证据：`index_builder/processors.py:31-38` 写入 `record_type/record_id/stage`；契约要求 `doc_id` 主键。
- 测试现状：`tests/index_builder/test_pipeline.py:108` 反向固化了非契约字段。
- 建议修复：对外落盘严格输出契约字段（至少 doc 失败），内部扩展字段转 metadata/log；加 schema 契约测试。

2. 错误分类体系未闭环（canonical taxonomy 未实现完整）
- 影响：治理报表、故障归因与 SLA 指标不可比。
- 触发：序列化错误、未知运行时错误、输入缺失字段等。
- 证据：`index_builder/errors.py:1-14` 仅 4 类，缺 `SerializationError`、`UnexpectedRuntimeError`；`error_type` 直接用 `type(exc).__name__`（`index_builder/processors.py:138,229`）。
- 建议修复：补齐 taxonomy + 建立统一 `map_exception_to_error_type()`；在 parquet/json 写入点显式捕获序列化错误。

3. 测试可发现性/可执行性不稳定，无法作为 CI 准入门
- 影响：回归不可控，契约漂移无法被自动阻断。
- 触发：标准 `unittest discover -s tests` 或默认导入路径执行。
- 证据：本次执行出现大量 `ModuleNotFoundError`（需要手动 `PYTHONPATH=.` 才能跑 `tests/index_builder`）；`chunk_selector` 测试依赖顶层导入风格（如 `from chunk_scorer import ...`）。
- 建议修复：统一包导入与测试入口（`python -m pytest` 无需手动 PYTHONPATH）。

### P1（高优先级）

1. “严格 768 维”只做了“等于配置维度”校验
- 证据：`index_builder/processors.py:125-129,220` 校验 against `runtime.embedding_dim`；无 `embedding_dim == 768` 的强约束。
- 风险：配置误改可绕过合同硬约束。

2. Scheduler 未复用 builder 级 dataset 校验，失败前移能力弱
- 证据：`index_scheduler.py` 仅读取模型配置并循环子进程；dataset 校验完全后置到 builder。
- 风险：多模型模式下重复失败、成本放大。

3. 模块耦合通过运行时 `sys.path` 注入
- 证据：`index_builder/processors.py:42-45`。
- 风险：IDE、测试、打包、部署环境行为不一致。

4. `chunk_selector` 异常处理过宽且吞异常
- 证据：`chunk_selector/chunk_selector.py:88` 捕获 `(RequestException, Exception)` 并返回空结果。
- 风险：真实缺陷被误判为“选块为空”，导致静默质量退化。

5. 契约关键场景测试缺失
- 缺口：scheduler 顺序/容错、failure schema、taxonomy 映射、输出路径规范。
- 证据：`tests/` 中无 scheduler 测试，无 failures 契约字段断言。

### P2（组织优化）

1. `index_builder` 与 `chunk_selector` 重复 HTTP 嵌入调用逻辑（重试/载荷校验重复）。
2. 部分 docstring/注释中英混杂，协作可读性不一致。
3. 运行日志为自由文本，建议补充结构化键（或 JSON logger）便于检索。
4. `README.md` 仍声明“does not implement runtime code yet”，与仓库现状不一致。

---

## 5. 最小改造路线图（不做架构重写）

### 阶段 A（先止血，1-2 天）
- 统一 `failures.jsonl` 对外 schema 到契约字段。
- 补齐错误 taxonomy 类与异常映射函数；序列化点补显式异常处理。
- 增加契约回归测试：failure schema + taxonomy + scheduler 容错。

### 阶段 B（提稳态，2-3 天）
- 去除 `sys.path` 注入，改为包内可解析导入（`chunk_selector` 包化）。
- 统一测试入口，保证 `python -m pytest` 在默认环境可发现并执行。
- 为 scheduler 加入顺序与 continue-on-failure 的单测/集成测试。

### 阶段 C（工程化提升，1-2 天）
- 去重 HTTP embedding client 逻辑，收敛重试/超时/校验策略。
- 规范日志字段与 README 叙述，提升可维护性和交接效率。

---

## 6. 测试证据

已执行：
1. `PYTHONPATH=. /Users/zhangjie/anaconda3/envs/stock_analysis/bin/python -m unittest tests.index_builder.test_config tests.index_builder.test_dataset tests.index_builder.test_embedding tests.index_builder.test_io_utils tests.index_builder.test_pipeline tests.index_builder.test_processors -v`
- 结果：`OK (skipped=1)`。

2. `/Users/zhangjie/anaconda3/envs/stock_analysis/bin/python -m unittest discover -s tests -v`
- 结果：大量 `ModuleNotFoundError`（测试导入路径配置问题）。

3. `PYTHONPATH=chunk_selector ... tests.chunk_selector...`
- 结果：受运行环境 OMP SHM 限制中断，未获得完整通过性结论。

备注：本报告中的结论以“静态审计 + 可执行证据”联合得出；对无法稳定执行的测试项已明确标注限制。
