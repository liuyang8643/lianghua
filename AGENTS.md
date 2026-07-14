参考@CLAUDE.md

- `python -m trading.main --skip YYYYMMDDHHMM --update` 使用 `--skip` 日期作为数据锚点；历史模拟只更新该交易日 K 线并重建 runtime，不更新系统当天的股票列表、财务等基础数据。
- 单日策略目标的唯一入口是 `core.strategy.build_strategy_day()`；盘前实盘和盘后回放必须直接消费其 `StrategyDayResult`，`trading/` 不得自行拼装 prefilter、因子过滤、合法性或调仓目标。
- Prefilter 固定为 `T-1 全量股票排名 -> top prefilter_n -> T 日候选池内排名`；当前持仓不得进入候选池，只能用于估值、可卖数量和调仓差额。
