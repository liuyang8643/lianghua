# WBR Copilot 指引

## 环境与命令

- **仅支持 Windows** — QMT（`xtquant`）只能在 Windows 上运行
- **包管理器：** `uv`（读取 `pyproject.toml`）
- **安装依赖：** `uv sync --locked`
- **启动实盘：** `.\run.ps1`（自动 git pull，然后启动 `trading\watchdog.py`）
- **启动实盘（手动）：** `python trading/main.py --individual-config configs/best_individual_config.json`
- **运行回测 + GA 优化：** `python testback/ga.py`
- **运行任何脚本前**，需将 `PYTHONPATH` 设置为仓库根目录

本项目没有配置测试套件或代码检查工具。

## 架构概览

### 因子流水线

```
allow_buy_stock_code_list()
    → TopN(stock_list, base_date)                          # 多线程（cpu_count-1）并行计算所有因子
    → get_ordered_stocks(n, weights, temperatures)
        → BatchNormFactor.batch_normalize()                # Rank 归一化 + 温度调节
        → 加权求和 → 降序排序 → Top N 股票列表
```

`TopN`（`core/strategies/top_n.py`）以**因子为粒度**调度线程（每个因子一个线程，每线程遍历全部股票），并在批次级别缓存整个因子×股票列表的计算结果。

### 调仓逻辑（buy_n / sell_m）

`trading/main.py` 的 `before_trade()` 完整实现了调仓：
1. 计算 Top `sell_m` → 持仓中**不在**列表的股票全部清仓
2. 计算 Top `buy_n` → `Sizer.allocate()` 分配资金 → 下单买入（跳过已持仓）

`buy_n`、`sell_m`、各因子 `weights` 和 `temperatures` 均来自 `--individual-config` 指定的 JSON 文件（GA 优化产出，默认为 `configs/best_individual_config.json`）。

### Watchdog 进程管理

`trading/watchdog.py` 是实盘真正的入口：
- 交易日 **09:00–16:00** → 自动启动 QMT 进程 + `main.py`
- 任一子进程崩溃 → 指数退避重启（最大 5 分钟间隔）
- 收盘后自动停止所有子进程

### GA 优化 → 实盘闭环

```
testback/ga.py
    → StockAccountMocker (testback/account.py)  # 模拟现金/持仓，含佣金和滑点
    → ga_optimizer()                             # DEAP 遗传算法，搜索 weights/temperatures/buy_n/sell_m
    → configs/best_individual_config.json        # 保存最优参数
        ↓
trading/main.py --individual-config configs/best_individual_config.json
```

GA 使用 Joblib + Loky 多进程并行评估种群，通过 `SharedMemoryCache` 跨进程共享 `TopN` 数据避免重复序列化。

### 因子设计模式

```python
class MyFactor(BaseFactor):
    def calc(self, ctx: FactorCtx) -> FactorResult:
        # 推荐公式：base_score（50%）+ signal_bonus（50%）
        # base_score:   连续值，基于指标当前数值
        # signal_bonus: 离散信号（金叉/背离等），配合 time_decay() 做时效衰减
        return FactorResult(score=final_score, err=None)
```

- 因子输出**原始未归一化分数**，归一化在 `TopN` 层统一处理
- `FactorCtx` 提供所有数据入口：`ctx.get_daily_data(n)`、`ctx.get_macd()`、`ctx.get_rsi()` 等（详见 `core/factors/helpers/indicators.py`）
- 信号辅助工具在 `core/factors/helpers/signal_detection.py`：背离检测、交叉检测、`time_decay()`

### 当前已启用因子

`TopN.factors` 中默认加载：`MACD`、`BBI`、`CCI`、`TRIXFactor`、`MOMFactor`、`ADXFactor`、`RetailFlow`、`RetailFlowMomentum`

已实现但默认禁用：`Fundamental`（API 慢）、`SmallCap`、`Unpopular`、`MainFundVolatility`、`RSI`、`KDJ`、`BollingerBands`

### 缓存系统

均存储于 `core/factors/helpers/.cache/`：

| 装饰器 | 格式 | 缓存键 |
|--------|------|--------|
| `@cached_factor(name)` | `.pkl` | `factor-{name}-{func_hash}_{datetime}_{code}` |
| `@cached_dataframe(name)` | `.parquet`（Snappy） | `{name}_{datetime}_{code}` |
| `@cached_value(name)` | `.pkl` | `{name}_{datetime}_{code}` |

函数源码变化时通过 `hash_function_code()` **自动失效**。手动清理：`core/factors/helpers/cache_clear.ps1`。

### 配置体系

| 文件 | 说明 |
|------|------|
| `configs/env.py` | 密钥（`TRADE_ACCOUNT`、飞书 Token）—— 从 `env.template.py` 复制，**禁止提交** |
| `configs/best_individual_config.json` | GA 产出的最优参数，实盘直接读取 |

温度参数：`T < 1.0` 压缩排名差异（稳定），`T > 1.0` 放大差异（灵敏）。

## 关键约定

- **股票代码**使用 QMT 格式：`000001.SZ`、`600000.SH`
- **日期**以 `datetime` 对象传递；字符串转换用 `utils/stock/format.py` 中的 `format_qmt_date()` / `format_qmt_datetime()`
- **各模块独立 logger**：从各自 `logger.py` 导入（`core_logger`、`testback_logger`、`trading_logger`），底层均为 `loguru`
- **添加新因子**：`core/factors/MyFactor.py` → `__init__.py` 导出 → 加入 `TopN.factors` → `_weights.py` 加默认值 → `base.yaml` 加配置块（含 `search_range`）
- **性能分析**：用 `line_profiler` 的 `@profile` 装饰器
