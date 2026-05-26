# AI 维护守则



### 回测 vs 实盘 严格分离

| 层面 | 回测 (`core/backtest.py` + `testback/single.py` + `testback/ga_run.py`) | 实盘 (`trading/main.py`) |
|---|---|---|
| 数据 | `load_runtime_npz()` 加载 npz → numpy | `load_runtime_npz(max_lookback)` 裁剪 → numpy |
| 因子 | `f.calc_batch(panel)` 全量向量化 | 同回测，共享 `_compute_factor_scores` 逻辑 |
| 选股 | `np.argsort(-final_score)[:n]` | 同回测 |
| 进程 | **GA 评估多进程并行，其余单进程串行** | 单进程实时 |

- **所有数据源有且仅有 runtime NPZ**，禁止 xtdata/mootdx/S3/CNINFO 等外部数据源出现在因子/选股/交易路径
- **因子 calc_batch 纯 numpy 向量化，禁止逐股票遍历**。5000+ 股票×20 年耗时应 < 1s，超过必有 bug。
- **多进程/多线程红线**：仅 `_run_ga` 中 GA 个体评估可用 `multiprocessing.Pool`。其他所有地方（因子计算、数据加载、实盘路径）禁止多进程/多线程。


## 数据管线

### 预下载

**全量更新流程**：执行全部预下载脚本 → 先删光昨天 parquet → 再全量拉取到今天。

**原因**：实盘开盘时触发预下载，此时获取到的日线 close/high/low 是盘中快照而非收盘值，数据错误。因此次日必须先删除昨天的不完整数据，再重新拉取完整日线覆盖到今天。

| 数据 | 来源 | 产物 |
|---|---|---|
| K线日线 | akshare | data/k-line/{code}.parquet |
| 股票列表 | xtdata | data/stock_list/ |
| 退市列表 | akshare | data/delist/ |
| 名称/ST历史 | CNINFO API | data/stock_name/ |
| 财务面板 | akshare | data/financial/ |
| 股本 | akshare | data/financial/ |
| 发行价 | akshare stock_ipo_info | data/issue_price/ |
目录：`data/{k-line, runtime, db, financial, stock_list, stock_name, ...}`

### Runtime 构建

入口：`python data/build_runtime.py`，产出 `data/runtime/runtime_{start}_{end}.npz`，回测时 `load_runtime_npz` 自动加载。

```python
np.savez_compressed('runtime_{start}_{end}.npz',
  stock_codes=np.array(U12),  trade_dates=np.array(datetime64[D]),
  open/high/low/close/volume/amount   # (n_dates, n_stocks)
  issue_price                          # (n_stocks,)  发行价
  st_mask                              # (n_dates, n_stocks) bool
  total_share                          # (n_dates, n_stocks)
  eps/roe/profit_yoy/revenue_yoy/operating_cf_ps/gross_margin
)
```

### 回测核心 `core/backtest.py`

```
run_single_mode (单回测入口)
  → _compute_factor_scores (加载npz → 逐因子 calc_batch → scores_to_ranks)
  → _backtest_direct (权重+温度 → argsort 取 topN → 查 open → 先卖后买)
  → compute_strategy_metrics → generate_single_report
```

| 函数 | 位置 | 职责 |
|---|---|---|
| `_compute_factor_scores` | `core/backtest.py` | 加载npz → 逐因子 calc_batch → scores_to_ranks 排名 → 返回 (data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices) |
| `_backtest_direct` | `core/backtest.py` | 权重+温度加权 → argsort 取 topN → 查 open → 先卖后买 |
| `run_single_mode` | `core/backtest.py` | 解析 config → compute → backtest → 报告 |
| `_run_ga` | `testback/ga_run.py` | GA/调试模式，个体评估 `multiprocessing.Pool` 并行 |

### 入口架构

| 入口 | 职责 |
|---|---|
| `testback/single.py` | 单回测入口，零 GA 依赖，`run_single_mode` |
| `testback/ga_run.py` | GA 引擎入口（私有），`ga_optimizer` + `_worker_*` 并行 |

`testback/single.py` 直接调用 `core/backtest.py` 的 `run_single_mode`，不依赖 `core/ga/`。

## 性能基准

| 指标 | 值 |
|---|---|
| NPZ加载 (1文件, ~4000d×~5200s) | ~6s |
| 因子计算 (单因子) | ~3.5s |
| 回测循环 (~4000调仓日) | ~12s |
| 报告生成 | ~2s |
| **20年全量总耗时** | **~27s** |

## 验收命令

```bash
# 单回测（core profile, TrueMarketCap 因子）
python -u -m testback.main --mode single \
  --start-date 20240101 --end-date 20241231 \
  --individual-config configs/single_tmc_pure.json

# WBR 精简入口
python run_backtest.py --start 2024-01-01 --end 2024-12-31
```
