# WBR 量化交易系统 - 项目架构文档

## 项目概述

WBR 是一个基于 QMT（国金量化交易平台）的 Python 量化交易系统，采用多因子选股 + 遗传算法参数优化的全自动交易策略。

**技术栈：**
- Python 3.12 / uv 包管理
- QMT (xtquant) — 仅支持 Windows
- TA-Lib — 技术指标计算
- DEAP — 遗传算法优化
- Joblib + Loky — 多进程并行
- NumPy / Pandas / PyArrow
- loguru — 日志

## 运行方式

```powershell
# 安装依赖
uv sync --locked

# 实盘（run.ps1 会先 git pull 再启动 watchdog）
.\run.ps1

# 回测 + GA 优化（直接运行，无需额外参数）
python testback/ga.py

# 运行前设置 PYTHONPATH
$env:PYTHONPATH = "项目根目录绝对路径"
```

无配置的测试套件或代码检查工具。

## 整体架构

### 选股流水线

```
allow_buy_stock_code_list()          ← 全市场可买入股票池
    ↓
TopN(stock_list, base_date)
    ↓  多线程（最多 cpu_count-1 线程）并行计算每个因子
    factor_scores: {因子名: {股票代码: FactorResult}}
    ↓
get_ordered_stocks(n, weights, temperatures)
    ↓  BatchNormFactor.batch_normalize()   ← Rank归一化 + 温度调节
    ↓  加权求和
    ↓  降序排序
    → Top N 股票列表
```

### 调仓逻辑（buy_n / sell_m 机制）

`trading/main.py` 在 `before_trade()` 阶段执行：
1. 计算 Top `sell_m` 股票列表 → 持仓中**不在**此列表的股票全部清仓
2. 计算 Top `buy_n` 股票列表 → 通过 `Sizer.allocate()` 分配资金后下单买入（跳过已持仓）
3. 以上参数来自 `--individual-config` 指定的 JSON 文件（由 GA 优化产出）

**`trading/main.py` 接收命令行参数：**
```powershell
python trading/main.py --individual-config configs/best_individual_config.json
```

### Watchdog 进程管理

`trading/watchdog.py` 是真正的入口（`run.ps1` 启动它）：
- 交易时段（09:00–16:00）+ 交易日 → 自动启动 QMT 和 `main.py`
- 任一子进程崩溃 → 指数退避后重启（最大 5 分钟）
- 收盘后自动停止所有子进程

### GA 优化

`testback/ga.py` 搜索最优 `individual_config`：
- **搜索参数：** `buy_n`、`sell_m`、各因子的 `weights`（正负均可）和 `temperatures`
- **搜索空间：** 离散值（定义在 `GA_SEARCH_SPACES`）
- **并行：** Joblib + Loky 多进程，`SharedMemoryCache` 跨进程共享 `TopN` 数据避免重复序列化
- **回测模拟：** `testback/account.py`（`StockAccountMocker`）模拟现金/持仓，计入佣金和滑点
- **最优结果** 保存至 `configs/best_individual_config.json`，供实盘直接读取

## 因子库

### 因子设计模式

```python
class MyFactor(BaseFactor):
    def calc(self, ctx: FactorCtx) -> FactorResult:
        # 评分公式（推荐）：base_score（50%）+ signal_bonus（50%）
        # base_score:    连续值，基于指标当前数值
        # signal_bonus:  离散信号（金叉/背离等），配合 time_decay() 衰减
        return FactorResult(score=final_score, err=None)
```

- 因子输出**原始未归一化分数**，归一化在 `TopN` 层统一处理
- `FactorCtx` 提供所有数据访问：`ctx.get_daily_data(n)`、`ctx.get_macd()`、`ctx.get_rsi()` 等
- 全部技术指标通过 `core/factors/helpers/indicators.py` 的 `FactorCtx` 方法暴露，底层使用 TA-Lib

### 添加新因子

1. `core/factors/MyFactor.py` — 继承 `BaseFactor`，实现 `calc()`
2. `core/factors/__init__.py` — 添加 `from .MyFactor import *`
3. `core/strategies/top_n.py` — 在 `self.factors` 列表中添加实例
4. `core/strategies/_weights.py` — 添加默认权重和温度参数
5. `configs/strategies/base.yaml` — 添加因子配置块（含 `search_range`）

## 缓存系统

**存储位置：** `core/factors/helpers/.cache/`（`.parquet` / `.pkl`）

| 装饰器 | 适用场景 | 缓存键 |
|--------|---------|--------|
| `@cached_factor(name)` | 单股票因子结果 | `factor-{name}-{func_hash}_{datetime}_{code}` |
| `@cached_dataframe(name)` | DataFrame 数据 | `{name}_{datetime}_{code}` |
| `@cached_value(name)` | 任意 pickle 值 | `{name}_{datetime}_{code}` |

- 函数源码变化时通过 `hash_function_code()` **自动失效**
- `TopN._calculate_factor_score()` 在**批次级别**额外缓存（整个因子×股票列表），键包含股票列表哈希
- 手动清理：`core/factors/helpers/cache_clear.ps1`

## 配置体系

| 文件 | 说明 |
|------|------|
| `configs/env.py` | 密钥（`TRADE_ACCOUNT`、飞书 Token）—— 从 `env.template.py` 复制，**禁止提交** |
| `configs/best_individual_config.json` | GA 搜索出的最优参数，实盘直接读取 |

**归一化温度参数说明：**
- `T < 1.0` — 压缩排名差异（适合长期稳定因子）
- `T = 1.0` — 保持原始差异
- `T > 1.0` — 放大排名差异（适合短期敏感因子）

## 关键约定

- **股票代码** 使用 QMT 格式：`000001.SZ`、`600000.SH`
- **日期** 以 `datetime` 对象传递；字符串转换用 `utils/stock/format.py` 中的 `format_qmt_date()` / `format_qmt_datetime()`
- **各模块独立 logger**：从各自 `logger.py` 导入（`core_logger`、`testback_logger`、`trading_logger`），底层均为 `loguru`
- **性能分析**：用 `line_profiler` 的 `@profile` 装饰器

---

**最后更新：** 2026-02-28
