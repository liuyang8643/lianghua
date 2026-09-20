# WBR 目标架构迁移计划

## 2026-09-17 当前单一实现收敛

本轮按ai/报告、env/factor、data/offline_data/trade并行审查，删除无调用包装、旧发行资料升级、固定全轴rank模式与重复黑名单。报告统一显式路径，只读当前报告/trace契约；取消历史源码执行和自动生成回放入口。非运行中源码树先归档逐文件验证再删除展开副本，当前训练的唯一冻结目录保留以维持运行身份。历史模型和指标不删除，不通过重新解释旧schema伪装成当前结果。

执行边界、具体删除与测试见 [本轮计划](artifacts/code_cleanup_20260917/plan.md) 和 [完成报告](artifacts/code_cleanup_20260917/report.md)。以下旧阶段记录只说明迁移历史；已被本节替代的旧回放入口不再存在。

本文只记录现有目录向目标五模块架构的迁移关系。目标职责、公共契约和开发准则见 [AGENTS.md](AGENTS.md)。

## 4. 现有目录迁移关系

| 当前内容 | 目标位置 | 整理方式 |
|---|---|---|
| `data/update_*.py`、`data/kline_mootdx.py`、下载脚本 | `offline_data/sources/`、`offline_data/update.py` | 统一外部数据源、全量/增量更新入口 |
| `data/financial_pit.py`、覆盖率审计和数据诊断 | `offline_data` 的 PIT/quality 实现 | 与数据更新和快照版本统一 |
| `data/build_runtime.py`、`data/build_deep_fin_runtime.py` | `offline_data/runtime` | 迁移 runtime 构建代码 |
| `data/runtime/*.npz` 及现有 parquet 数据目录 | `offline_data` 管理的可配置本地数据根目录 | 作为数据产物管理；不混入 Python 包迁移 |
| `dashboard` 数据覆盖率页面 | `offline_data/quality` | 作为离线数据质量工具，不作为独立业务模块 |
| `factor_db/factors` 中已接纳的生产因子 | `factor/library` | 进入固定因子词表和版本管理 |
| `factor_db/factors` 中尚未接纳的生成候选 | `ai/factor_discovery` 工作区 | 验证通过后才注册到 `factor/library` |
| `factor_db/db.py`、records、signatures、扫描维护脚本 | `ai/factor_discovery` | 保留因子发现、血缘、相似度和评估逻辑 |
| `factor_db/registry.db`、扫描记录和报告 | `artifacts/factor_discovery` | 作为 AI 研究产物，不进入市场数据仓库 |
| `testback/reportor` | `env` 输出之上的只读报告适配器 | 不得复制指标、账户或回测语义 |
| factor report、仍在使用的研究脚本 | `ai/factor_discovery` | 作为候选因子研究和离线评价入口；新扫描器按统一因子契约实现 |
| `utils/stock/time.py` | `offline_data` 交易日历契约 | 由数据层统一提供交易日历 |
| `utils/stock/info.py` | `env/strategy.py` | 板块、涨跌停和申报数量规则归合法性领域逻辑 |
| 股票显示格式 | `trade/broker/helper.py` | 已迁移唯一使用的股票描述；无调用日期格式和旧`utils/stock/format.py`已删除 |
| 进程和唤醒工具 | 实际调用方模块 | 无消费者的`utils/recorder.py`已删除；现有进程和唤醒工具仍由真实入口使用 |
| 日志配置 | `utils/logger.py`及调用方入口 | 已删除`BaseLogger`转发层和`testback/logger.py`；直接使用Loguru与公共控制台配置 |
| `configs` | `configs` | 保留声明式配置；动态参数默认值迁入 schema |
| `results` 及生成报告 | `artifacts` | 统一不可变运行产物和报告目录 |
| `tests` | `tests` | 按目标模块和公共契约重新分组 |

## 迁移顺序

1. 建立首个完整可运行垂直切片：本地快照 -> 因子 -> Observation -> 固定 DayConfig -> OrderPlan -> SimExecutor -> `settle_next_open` -> StepResult；随实际调用建立所需最小契约和目录。
2. 迁移数据读取与因子实现，保持当前回测结果不变；迁移完成即移除对应旧入口。
3. 将评分、合法性、调仓、账户和回测收敛到同一个 env 领域核。
4. 将 GA 接入统一 `Policy -> DayConfig` 契约。
5. 接入 Gymnasium 和 Stable-Baselines3 PPO，复用相同 Observation、ActionSchema 和 env。
6. 将实盘改为调用同一 env 生成 `OrderPlan`，券商层只负责执行。
7. 验证新路径后删除旧入口和重复实现，完成架构收敛。

每个阶段单独保持可运行、可回测、可对比；不得通过一次性整体搬迁掩盖行为变化。

## 2026-09-14 分模块精简

本轮按AI、env、数据/因子和实盘/报告/入口四组审查，先删除有调用证据的不可达分支和无人使用闭包，再合并原子文件I/O、候选窗口计算及共享数组。模块审查、实施顺序与逐项验证见 [执行计划](artifacts/code_cleanup_20260914/plan.md) 和 [结果报告](artifacts/code_cleanup_20260914/report.md)。

`data`仍承担下载/构建、`offline_data`承担封存读取，不是两个等价runtime；`factor_db`仍有生产导入及动态登记候选。它们的目录迁移保留上表的渐进顺序，不通过新的兼容层来制造表面上的目录完成。旧冻结模型/源码和研究产物不并入活动代码清理。

## GA / PPO 报告复用

活动训练看板统一由 `ai/reporting.py` 适配既有GA/PPO运行记录，`ai/report_server.py` 提供同一个HTTP/CSV接口，`ai/report_assets` 维护唯一页面和图表。报告层不持有模型、市场快照、选模或收益计算。算法专属部分限定为输入格式转换和可选诊断字段。`testback/reportor`继续服务单次回测逐笔交易明细，不另建训练指标实现。

旧实验的monitor/source仅保留作冻结审计材料；唯一报告服务只读取已保存的报告与逐日轨迹，没有解压或执行历史源码的入口。数据层质量页面仍归离线数据职责，不与训练指标混用。实施与验收见 [报告统一计划](artifacts/report_unification_20260914/plan.md)。

## 2026-09-17 原始输入替换

env负责唯一原始Observation、开盘合法性与共享RawMarketStore；ai负责可训练的逐股时序网络，日期引用在可学习层前移除。训练、冻结回放、live Policy显式绑定同一输入契约；删除旧统计encoder与缓存，不提供跨schema兼容路径。GA复用原有执行核，不创建不使用的actor输入。共有报告契约保留，详细诊断记录原始表身份和compact样本。字段、工程验收和性能见[改造报告](artifacts/raw_observation_20260917/report.md)。

PPO CUDA 唯一执行约束归 `ai/rl/device.py`，冻结 actor 的 CUDA launch graph 归 `ai/rl/inference.py`；二者服务训练、冻结回放和 live 的同一 typed policy，不实现第二套决策或网络。共享 patch/梯度与冻结市场缓存归现有 `raw_features.py`，GA 索引归现有 GA 训练器，因子共同中间量归 `factor`。性能重排不迁移 env 的领域语义；旧实现只作为隔离测量材料保留在 artifacts。验收和未达目标项见[性能报告](artifacts/performance_optimization_20260917/report.md)。

因子和选股排序共享 `utils/stable_sort.py` 的唯一稳定数值排序核，领域键构造与校验留在各自模块，避免两个 radix 实现。该基础核必须计入模型 source identity。原始窗口成员 CSR 归 `ai/rl/raw_features.py`，不改变可学习网络、股票轴或时间长度。

第四轮保留固定前缀激活的执行额度仍归 `raw_features.py`，不引入第二个 encoder；Bili 状态链归原因子模块，styles 的缺失计数前缀/后缀核归既有 `completed_windows.py`。旧 styles 数值函数与64列分块被删除，其他不同语义且有真实调用方的窗口核保留。planner 结构化输入打包只减少适配开销，复用原交易数值核。生产最终验收与隔离候选测速分别记录在性能目录。
