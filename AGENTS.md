参考@CLAUDE.md

- `python -m trading.main --skip YYYYMMDDHHMM --update` 使用 `--skip` 日期作为数据锚点；历史模拟只更新该交易日 K 线并重建 runtime，不更新系统当天的股票列表、财务等基础数据。
- 单日策略目标的唯一入口是 `core.strategy.build_strategy_day()`；盘前实盘和盘后回放必须直接消费其 `StrategyDayResult`，`trading/` 不得自行拼装 prefilter、因子过滤、合法性或调仓目标。
- T 日选股、择时、合法性和成交只允许使用 `open[T]`；`high/low/close/volume/amount[T]` 不得用于任何交易决策，IPO 首日也不例外。前收统一使用官方 `preClose[T]`。
- 老规则 IPO 首日若 `open[T]` 已达到 `floor(发行价×1.20)` 的集合竞价上限，视为开盘无法可靠成交并禁买；该规则只能使用已知上市日、发行价和 `open[T]`，不得用当日 HLC 反推封板。
- 策略每个调仓日都在完整股票池上进行因子排名；`prefilter_n` 不是策略参数，当前持仓不得改变候选池，只能用于估值、可卖数量和调仓差额。
- 市场趋势与策略目标组合趋势的仓位倍率由 `core.trend_timing` 负责；GA 与单回测入口调用同一倍率构建函数，正式账户回测只消费最终倍率数组且不运行影子账户。单回测可加 `--no-live-sim` 跳过分钟价格模拟。
- V9 双趋势 GA 使用 `python -m testback.run_ga --mode ga --profile v9_dual_shadow`；训练期为 2010-2018，验证期为 2019-2022，测试期为 2023 至 runtime 最新交易日；`sell_m` 固定等于当次 `buy_n`，中证500/中证1000同期 Calmar 仅作日志诊断，不参与适应度或候选选择。
- GA 与单回测的候选掩码只合并非零权重因子的有效区间，并直接共享过滤因子的原始布尔结果；权重为零的因子不得通过缺失值过滤继续影响股票池。
- 因子截面排名的分母只能包含当前策略股票池；runtime 中股票池外的代码不得改变候选股票的排名分数。
- GA 的遗传适应度、hall-of-fame 和 `best_individual_config.json` 只能按训练集指标选择；验证集与测试集允许记录诊断结果，但不得参与排序、候选保存或参数淘汰。
- V9 使用训练期连续三折稳健适应度（50%全训练Calmar + 50%最差训练折Calmar），并要求训练期每日实际仓位的平均值不低于45%；需要逐个体保留三周期诊断时使用 `--split-period-results`，训练、验证、测试分别写入 `training_results.jsonl`、`validation_results.jsonl`、`test_results.jsonl`，三者按 generation 与 config 一一对齐。
- GA 代数、种群和随机种子临时覆盖使用 `--generations N --population-size N --seed S`，不得为一次研究运行修改 YAML 全局默认值；正式消融必须使用相同种子，种子与封存状态记录在结果目录的 `run_metadata.json`。
- 训练稳健目标只能恢复包含 `fitness` 与 `fold_calmars` 的同口径训练 JSONL，禁止使用带外部评分的 `--warm-start`；验证与测试结果只写各自的周期 JSONL，不得回写训练 JSONL。
- GA `--resume` 必须与 `run_metadata.json` 的 profile、随机种子、训练目标、sealed 状态和 split-period-results 状态完全一致；遗传适应度、hall-of-fame 与 `best_individual_config.json` 始终只能读取训练结果，验证/测试文件仅作诊断留存。
- `v10_intermediate_momentum` 是 V9 的单变量训练消融，只新增固定公式 `IntermediateMomentum12Minus1`；不得同时加入其他新因子后再归因其效果。
- 完成的 V9/V10 sealed 运行使用 `python -m testback.analyze_train_ablation <v9目录> <v10目录> --output <报告.json>` 比较；该入口只读取训练 JSONL，并强制两边随机种子一致且训练记录不含 holdout 指标。
- 冻结候选的单参数稳定性使用 `python -m testback.generate_ga_neighborhood <best_config.json> --output <neighbors.json>` 生成邻域及完整择时关闭对照，再用 `python -m testback.run_ga --mode debug --profile <profile> --candidate-configs <neighbors.json> --generations 1 --sealed-holdout` 仅在训练集批量评估；禁止用该入口读取holdout。GA 缓存键必须覆盖全部回测语义字段，仅可忽略报告标签。
