# WBR 项目说明

> 所有实施完成后，由 verify-agent 单独验收。

## 1. 核心目标
1. **数据获取模块 data/ 获取所有数据源**。回测/ga除此之外全部离线。
1. **回测速度快/回测完全离线**。
2. **回测避免任何形式上的数据泄露**，如回测使用当前股票列表（会排除历史退市）、买卖合法性检查（不同板块不同时间段规则完全不同）、因子前视检查。
3. **回测和实盘完全对齐**。策略逻辑（冻结模型、因子、TopK、合法性与股数规划）完全复用，仅实际买卖调用接口不同。并配套开发实盘回测 diff 模块，实盘期间记录所有 diff 需要的信息，盘后再跑一遍当天回测看 diff。
4、**代码精简**。项目尽量模块化复用，避免冗余代码、冗余文件。代码理论上不允许防御性编程和容错，如try\get等（除非是逻辑需要）。

## 2. 整体架构

数据下载 → 已完成历史Runtime与时点资料 → strategy_release.runtime统一因子/预测/分配 → env.DayPlanner。
离线用env.DaySimulator执行；实盘发布绝对目标JSON，由客户端Python3.6执行唯一trading.execution_engine。
模型与策略只有一套实现，禁止兼容旧策略或在客户端复制选股逻辑。具体启动/数据/执行契约以AGENTS.md为准。

### 2.2 红线规则

- **数据源红线（按是否联网判断）**：除 `data/update_*.py`、`data/kline_mootdx.py` 预下载入口、`trading/` 买卖模块（QMT/xtdata）外，**所有其它模块禁止任何形式的网络获取**（akshare / requests / xtdata / CNINFO 等）。NPZ + 预下载产物 parquet 均可读，因为它们不联网。历史 K 线统一由 data/update_kline.py 预下载：优先 mootdx，失败后使用已登录券商 QMT 不复权日线及官方 preClose。
- **T 日价格红线（最高优先级，防数据泄露）**：信号触发、买卖合法性检查、账户成交价 **只允许使用 `open[T]`**。当日的 `high[T] / low[T] / close[T] / volume[T] / amount[T]` 全部视为前视野泄露，禁止出现在选股 / 风控 / 估值路径。需要"前收"时统一使用 `close[T-1]`。
- **因子 `calc_batch` 纯 numpy 向量化，禁止逐股票遍历**。5000+ 股票 × 20 年耗时应 < 1s，超过必有 bug。
- **多进程/多线程红线**：仅 `_run_ga` 中 GA 个体评估可用 `multiprocessing.Pool`。其他所有地方（因子计算、数据加载、实盘路径）禁止多进程/多线程。
- **回测/实盘对齐红线**：选股、买卖合法性检查、Top-N 排序等逻辑必须由 `strategy_release` / `env` 提供唯一实现，回测和实盘共同调用；**禁止任何相同逻辑出现两份实现**。

## 3. 数据与验证

历史不复权K线、官方preClose、ST及财务事件均从本地已验证数据读取；盘前腾讯开盘价仅内存overlay。
Runtime由data/build_runtime.py完整写临时文件后原子发布，完整股票轴保留历史退市股。
收益与合法性沿用研究env统一口径；禁止混用旧core或以旧基础Runtime替代新模型所需PIT/行业/月度资料。

运行前执行 `uv sync --locked`。离线入口：
`python testback/run_backtest.py --start-date 2022-01-01 --end-date 2026-10-31 --output-dir results/new_model_check`。
所有实施完成后由verify-agent独立验收；未通过不得启动新实盘计划。
