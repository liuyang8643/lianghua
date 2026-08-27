# WBR 目标架构迁移计划

本文只记录现有目录向目标五模块架构的迁移关系。目标职责、公共契约和开发准则见 [AGENTS.md](AGENTS.md)。

## 4. 现有目录迁移关系

| 当前内容 | 目标位置 | 整理方式 |
|---|---|---|
| `data/update_*.py`、`data/kline_mootdx.py`、下载脚本 | `offline_data/sources/`、`offline_data/update.py` | 统一外部数据源、全量/增量更新入口 |
| `data/financial_pit.py`、覆盖率审计和数据诊断 | `offline_data` 的 PIT/quality 实现 | 与数据更新和快照版本统一 |
| `data/build_runtime.py`、`data/build_deep_fin_runtime.py` | `offline_data/runtime` | 迁移 runtime 构建代码 |
| `core/runtime.py` | `offline_data/runtime` | 与构建端合并为唯一 runtime 读取契约 |
| `data/runtime/*.npz` 及现有 parquet 数据目录 | `offline_data` 管理的可配置本地数据根目录 | 作为数据产物管理；不混入 Python 包迁移 |
| `dashboard` 数据覆盖率页面 | `offline_data/quality` | 作为离线数据质量工具，不作为独立业务模块 |
| `core/factors/*` | `factor/base.py`、`factor/registry.py` | 迁移因子接口和生产注册表 |
| `factor_db/factors` 中已接纳的生产因子 | `factor/library` | 进入固定因子词表和版本管理 |
| `factor_db/factors` 中尚未接纳的生成候选 | `ai/factor_discovery` 工作区 | 验证通过后才注册到 `factor/library` |
| `factor_db/db.py`、records、signatures、扫描维护脚本 | `ai/factor_discovery` | 保留因子发现、血缘、相似度和评估逻辑 |
| `factor_db/registry.db`、扫描记录和报告 | `artifacts/factor_discovery` | 作为 AI 研究产物，不进入市场数据仓库 |
| `llm_ga` | `ai/factor_discovery` | 与上述发现数据库合并为一个候选因子流程 |
| `core/ga/*`、`testback/run_ga.py` | `ai/ga` | 迁移 GA profile、采样、主循环和统一 Policy 导出 |
| `testback` 中 GA 收敛、邻域、landscape 分析 | `ai/ga` | 作为 GA 训练评估工具 |
| 新 PPO 模块 | `ai/rl` | 承载 SB3 PPO 训练、评估和推理 |
| `core/scoring.py`、`core/prefilter.py` | `env/strategy.py` | 与选股流程统一 |
| `core/legality.py`、`core/rebalance.py` | `env/strategy.py` | 统一合法性和 `OrderPlan` 生成 |
| `core/strategy.py`、timing、trend_timing | `env/observation.py`、`env/strategy.py` | 拆开决策数据准备与纯策略计划 |
| `core/strategy_config.py` | `env/action_schema.py`、`configs` | 动态参数约束归 schema，部署选择归配置 |
| `core/sim/*`、`core/fees.py` | `env/simulator.py` | 统一模拟成交、费用和账户转换 |
| `core/backtest.py`、`testback/run_backtest*.py` | `env/backtest.py` | 迁移单日 transition 和回测 runner |
| `core/metrics.py`、`testback/metrics.py`、`testback/reportor` | `env` 的 metrics/reporting 实现 | 回测评价和报告复用 env 输出 |
| `testback/scan_factors.py`、factor report、研究脚本 | `ai/factor_discovery` | 作为候选因子研究和离线评价入口 |
| `trading/executor.py`、`qmt.py`、`trader.py` | `trade/executor.py`、`trade/broker/` | 实现 env 定义的 `ExecutionPort` |
| `trading/main.py`、scheduler、watcher、watchdog、manual_confirm、Lark | `trade/runtime.py` 及内部适配文件 | 统一实盘调度、人工确认和通知 |
| `trading/persistence.py`、post_close、replay、report | `trade/journal.py` 及内部报告实现 | 统一实盘记录、盘后对账和 replay |
| `trading/test_*` | `tests/trade` | 测试跟随目标模块整理 |
| `utils/stock/time.py` | `offline_data` 交易日历契约 | 由数据层统一提供交易日历 |
| `utils/stock/info.py` | `env/strategy.py` | 板块、涨跌停和申报数量规则归合法性领域逻辑 |
| `utils/stock/format.py` | `trade/broker` | 保留 QMT 代码和日期格式适配 |
| `utils/recorder.py`、进程和唤醒工具 | 实际调用方模块 | recorder 归 `trade`，进程控制归 `trade`，GA 唤醒归 `ai/ga` |
| `utils/logger.py`、`core/logger.py`、`testback/logger.py` | 调用方的统一日志配置 | 合并重复包装；不保留泛化业务 `utils` |
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
