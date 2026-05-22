# AI 维护守则

## 核心规则

1. **验收规则**：完成实质性产出后，必须调用 verify agent 做独立验收，不得自行报告完成。
2. **调试规则**：先小量（10~50只/7~30天）→ 全量 → 长周期。>20s 无日志 → 卡死，立即 kill。运行用 `python -u`。
3. **代码精简**：避免防御性编程、冗余 try/except、无意义抽象。写完回头删废代码。
4. **执行优先级**：agent-team > subagent > main-agent + verify-agent。优先用 team 并行分发任务。
5. **GA 运行前强制清理**：`powershell Stop-Process -Name python -Force` 杀后台 → sleep 3 → 二次确认进程数=0 且空闲内存>30GB。不跳过，否则必报 PermissionError/WinError 5 拒绝访问。

## 架构红线（不可违反）

### 回测 vs 实盘 严格分离

| 层面 | 回测 (`testback/main.py`) | 实盘 (`trading/main.py`) |
|---|---|---|
| 数据 | `load_runtime_npz()` 合并 npz → 全量 numpy | `TopN` + `get_market_data_batch` 逐股查 |
| 因子 | `f.calc_batch(panel)` 全量向量化 | 通过 `_precomputed_scores` 注入 |
| 归一化 | `_backtest_direct` 内联 rank 归一化 | `BatchNormFactor.batch_normalize()` |
| 选股 | `np.argsort(-final_score)[:n]` | `TopN.get_ordered_stocks(n)` |
| 进程 | **GA 评估多进程并行，其余单进程串行** | 单进程实时 |

- **回测路径绝不创建 TopN 对象**
- **回测路径不调 xtdata/mootdx/S3/CNINFO 等外部数据源**
- **因子 calc_batch 纯 numpy 向量化，禁止逐股票遍历**。5000+ 股票×20 年耗时应 < 1s，超过必有 bug。
- **多进程/多线程红线**：仅 `_run_ga` 中 GA 个体评估可用 `multiprocessing.Pool`。其他所有地方（因子计算、数据加载、实盘路径）禁止多进程/多线程。

### 已删除的组件（禁止恢复）

- `SharedMemoryCache` — 回测路径已移除（仅 `core/database/data.py` 实盘缓存保留）
- `joblib.Parallel` — 已移除，使用 `multiprocessing.Pool`（仅 GA 评估使用）
- `_wrap_process_worker` / `_prepare_shared_topn` / `_put_topn_window_slice` — 共享内存 TopN 预计算
- `_sample_topn_window` — GA 窗口采样（现全量回测）
- `compute_topn_range` — 回测不再调用（实盘仍用）
- `testback_cache` 全局变量

## 数据管线

### 预下载

**全量更新流程**：执行全部预下载脚本 → 先删光昨天 parquet → 再全量拉取到今天。

**原因**：实盘开盘时触发预下载，此时获取到的日线 close/high/low 是盘中快照而非收盘值，数据错误。因此次日必须先删除昨天的不完整数据，再重新拉取完整日线覆盖到今天。

| 数据 | 来源 | 产物 |
|---|---|---|
| K线日线 | mootdx | parquet |
| 股票列表 | xtdata | parquet |
| 退市列表 | akshare | parquet |
| 名称/ST历史 | CNINFO API | parquet |
| 财务面板 | PershareIndex | parquet |
| 股本 | balance.parquet cap_stk | parquet |
| 发行价 | akshare stock_ipo_info | parquet |
目录：`data/{数据源}/{parquet&runtime}/{数据&构建脚本}`

### Runtime 构建

入口：`python data/build_runtime.py`，产出按年分段 npz，回测时 `load_runtime_npz` 自动合并。

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

### 回测入口 `testback/main.py`

```
导入 → ga_optimizer → 工具函数
     → _compute_factor_scores (npz+因子)
     → _backtest_direct (核心：纯numpy逐日rank→排序→交易)
     → run_single_mode → _run_ga(GA/调试) → main
```

| 函数 | 职责 |
|---|---|
| `_compute_factor_scores` | 合并npz → 逐因子 calc_batch → 返回 (data, all_scores, ...) |
| `_backtest_direct` | 逐日 rank 归一化 → argsort 取 topN → 查 open → 先卖后买 |
| `run_single_mode` | 解析 config → compute → backtest → 报告 |
| `_run_ga` | GA/调试模式，个体评估 `multiprocessing.Pool` 并行，支持 is_debug 串行 |

## 性能基准

| 指标 | 值 |
|---|---|
| NPZ加载合并 (9文件, 3969d×5201s) | ~6s |
| 因子计算 (单因子) | ~3.5s |
| 回测循环 (3968调仓日) | ~12s |
| 报告生成 | ~2s |
| **20年全量总耗时** | **~27s** |

## 验收命令

```bash
python -u -m testback.main --mode single --profile smallcap_g2a_roe2x_bottom10 \
  --start-date 20100104 --end-date 20260514 \
  --individual-config configs/single_smallcap_g2a_config.json
```
