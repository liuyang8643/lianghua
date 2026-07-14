# WBR 项目说明

> 所有实施完成后，由 verify-agent 单独验收。

## 1. 核心目标
1. **数据获取模块 data/ 获取所有数据源**。回测/ga除此之外全部离线。
1. **回测速度快/回测完全离线**。
2. **回测避免任何形式上的数据泄露**，如回测使用当前股票列表（会排除历史退市）、买卖合法性检查（不同板块不同时间段规则完全不同）、因子前视检查。
3. **回测和实盘完全对齐**。策略逻辑（因子策略 topn 等）完全复用，仅实际买卖调用接口不同。并配套开发实盘回测 diff 模块，实盘期间记录所有 diff 需要的信息，盘后再跑一遍当天回测看 diff。
4、**代码精简**。项目尽量模块化复用，避免冗余代码、冗余文件。代码理论上不允许防御性编程和容错，如try\get等（除非是逻辑需要）。

## 2. 整体架构

### 2.1 数据与计算流

```mermaid
flowchart LR
    PreDownload["预下载<br/>parquet"] --> BuildRuntime["Runtime 构建<br/>np.savez"]
    BuildRuntime --> LoadNpz["加载 NPZ<br/>~6s"]
    LoadNpz --> ComputeFactor["因子矩阵计算<br/>numpy 向量化<br/>~3.5s / 因子"]
    ComputeFactor --> LegalCheck["买卖合法性检查<br/>numpy"]
    LegalCheck --> Backtest["账户收益 numpy 回测<br/>~12s (~4000 调仓日)"]
    Backtest --> Report["报告生成<br/>~2s"]
```

因子矩阵维度为 `[回测天数, 股票个数, 因子历史需要天数]`，单因子计算耗时在毫秒级。

### 2.2 红线规则
- **数据源红线（按是否联网判断）**：除 `data/update_*.py`、`data/kline_mootdx.py` 预下载入口、`trading/` 买卖模块（QMT/xtdata）外，**所有其它模块禁止任何形式的网络获取**（akshare / requests / xtdata / CNINFO 等）。NPZ + 预下载产物 parquet 均可读，因为它们不联网。mootdx（腾讯通达信）为 K 线唯一数据源，无需本地 QMT。
- **T 日价格红线（最高优先级，防数据泄露）**：信号触发、买卖合法性检查、账户成交价 **只允许使用 `open[T]`**。当日的 `high[T] / low[T] / close[T] / volume[T] / amount[T]` 全部视为前视野泄露，禁止出现在选股 / 风控 / 估值路径。需要"前收"时统一使用 `close[T-1]`。
- **因子 `calc_batch` 纯 numpy 向量化，禁止逐股票遍历**。5000+ 股票 × 20 年耗时应 < 1s，超过必有 bug。
- **多进程/多线程红线**：仅 `_run_ga` 中 GA 个体评估可用 `multiprocessing.Pool`。其他所有地方（因子计算、数据加载、实盘路径）禁止多进程/多线程。
- **回测/实盘对齐红线**：选股、买卖合法性检查、Top-N 排序等逻辑必须由 `core/` 提供唯一实现，回测和实盘共同调用；**禁止任何相同逻辑出现两份实现**。

## 3. 数据管线

### 3.1 预下载

**全量更新流程**：执行全部预下载脚本 → 先删光昨天 parquet → 再全量拉取到今天。

**原因**：实盘开盘时触发预下载，此时获取到的日线 close/high/low 是盘中快照而非收盘值，数据错误。因此次日必须先删除昨天的不完整数据，再重新拉取完整日线覆盖到今天。

| 数据 | 来源 | 产物 |
|---|---|---|
| K线日线 · 不复权 | mootdx `fq=0`（OHLCV）+ `xdxr()` 自算 preClose | `data/k-line/{code}.parquet` |
| 股票列表 | xtdata | `data/stock_list/` |
| 退市列表 | akshare | `data/delist/` |
| 名称/ST 历史 | CNINFO API | `data/stock_name/` |
| 财务面板 | akshare | `data/financial/` |
| 股本 | akshare | `data/financial/` |
| 发行价 | akshare `stock_ipo_info` | `data/issue_price/` |

目录：`data/{k-line, runtime, db, financial, stock_list, stock_name, ...}`。

K线下载唯一入口 `data/kline_mootdx.py`（腾讯通达信，免 QMT）：`update_full()` 全量、`update_recent(days)` 增量合并最近 N 个交易日；`update_all._update_kline` 与 `update_live._download_kline_all` 均复用它。

### 3.2 Runtime 构建

入口：`python data/build_runtime.py`，产出 `data/runtime/runtime_{start}_{end}.npz`，回测时 `load_runtime_npz` 自动加载。

```python
np.savez_compressed('runtime_{start}_{end}.npz',
  stock_codes=np.array(U12),  trade_dates=np.array(datetime64[D]),
  open/high/low/close/volume/amount   # (n_dates, n_stocks)  原始不复权(真实价)
  preClose                             # (n_dates, n_stocks)  官方前收(除权除息参考价)
  issue_price                          # (n_stocks,)  发行价
  st_mask                              # (n_dates, n_stocks) bool
  total_share                          # (n_dates, n_stocks)
  eps/roe/profit_yoy/revenue_yoy/operating_cf_ps/gross_margin
)
```

### 3.3 复权口径

全部使用不复权真实价。

- **K线唯一源 = mootdx（腾讯通达信），一套不复权 parquet**：`data/kline_mootdx.py` 取 `fq=0` 的不复权 OHLCV，preClose 由 `xdxr()` 除权除息数据按交易所公式自算（99.4% 除权日与官方一致），→ `data/k-line/`。
- **收益计算用 preClose**：个股日收益 `r[t]=close[t]/preClose[t]-1`（普通日 preClose=昨收，除权日=除权参考价，已吸收分红送转配股），除权日不产生假跳空。不需要后复权价格序列。
- **涨跌停 / 合法性判断（`legality`）一律用「原始 OHLC + 官方 preClose」**：`涨停价=preClose×(1+板块涨跌幅)`、`一字板=open≥涨停价`。preClose 已是除权参考价，除权日 open/preClose 天然不假跳空——研究与实盘对账口径完全一致。
- **所有成交量/金额/市值/财务因子用原始真实价**（`TrueMarketCap`/`AmountBasedSmallCap`/`VolumeCV`/`deep_value` 及全部 `Factor_*` 等，绝对规模口径）。


## 5. 验收命令

```bash
uv run python run_backtest --start 2024-01-01 --end 2024-12-31
```
