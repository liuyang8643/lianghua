# WBR 量化交易系统 - 项目架构文档

## 项目概述

WBR 是一个基于 QMT (国金量化交易平台) 的 Python 量化交易系统，采用多因子选股策略进行自动化交易。

**技术栈:**
- Python 3.12
- QMT (xtquant) - 国金量化交易平台
- TA-Lib - 技术指标计算
- NumPy/Pandas - 数据处理
- Joblib - 并行计算

## 目录结构

```
WBR/
├── core/                  # 核心模块
│   ├── database/         # 数据访问层
│   ├── factors/          # 因子库
│   │   └── helpers/     # 因子工具
│   └── strategies/       # 策略模块
│       └── sizers/      # 仓位管理
├── trading/              # 实盘交易模块
│   └── lark/            # 飞书通知
├── testback/             # 回测模块
├── utils/                # 工具函数
│   └── stock/           # 股票相关工具
└── configs/              # 配置文件

```

## 核心模块详解

### 1. core/database - 数据访问层

**主要文件:**
- `data.py` - 行情数据获取
- `detail.py` - 股票详情信息
- `history.py` - 历史数据管理
- `allow_list.py` - 股票池管理（可买入股票列表）
- `type.py` - 数据类型定义

**职责:**
- 封装 QMT 数据接口
- 提供统一的数据访问接口
- 管理可交易股票池

### 2. core/factors - 因子库

**因子设计架构:**

所有因子继承自 `BaseFactor` 接口，实现 `calc(ctx: FactorCtx) -> FactorResult` 方法。

**已实现因子:**

#### 传统因子
- `MACD.py` - MACD 指标
- `BBI.py` - 多空指数
- `CCI.py` - 顺势指标

#### 连续因子（新迁移）
- `RSI.py` - RSI 超卖因子
- `ADX.py` - ADX 趋势强度因子
- `KDJ.py` - KDJ 超卖因子
- `BollingerBands.py` - 布林带因子
- `TRIX.py` - TRIX 三重指数平滑因子
- `MOM.py` - 动量因子

**因子设计原则（加法混合）:**

```python
final_score = base_score(50%) + signal_bonus(50%)
```

- `base_score`: 连续值评分，基于指标当前状态
- `signal_bonus`: 离散信号评分，基于技术形态（金叉、背离等）

**因子工具（helpers/）:**

1. `interface.py` - 因子接口定义
   - `BaseFactor` - 因子基类
   - `FactorResult` - 因子计算结果（score, err）

2. `indicators.py` - 技术指标计算（FactorCtx）
   - 基础指标: MA, EMA, MACD, BBI, CCI
   - 高级指标: RSI, ADX, KDJ, Bollinger Bands, WILLR, SAR, TRIX, MOM, ROC
   - 使用 TA-Lib 库进行高性能计算

3. `cache.py` - 因子缓存装饰器
   - `@cached_factor(factor_name)` - 避免重复计算
   - 基于 (股票代码, 日期, 因子名) 的缓存机制

4. `batch_norm.py` - 批量归一化工具
   - `BatchNormFactor.batch_normalize()` - 跨股票归一化
   - 支持 rank（排名归一化）和 minmax（最小-最大值归一化）
   - 输出: norm_score, raw_value, batch_mean, batch_std, z_score

5. `signal_detection.py` - 技术信号检测
   - 背离检测: `detect_bullish_divergence()`, `detect_bearish_divergence()`
   - 交叉检测: `detect_golden_cross()`, `detect_death_cross()`
   - 突破检测: `detect_price_breakout()`, `detect_sar_flip()`
   - 时效衰减: `time_decay()` - 将离散信号转换为连续值

### 3. core/strategies - 策略模块

**核心文件:**

1. `top_n.py` - TopN 多因子选股策略

   **工作流程:**
   ```python
   TopN(stock_list, base_date)
   │
   ├─> 多线程并行计算每只股票的因子分数
   │   └─> _calculate_factor_score(stock_code)
   │       └─> 计算所有因子: MACD, BBI, CCI, ...
   │
   └─> 汇总因子分数到 factor_scores 字典
   ```

   **因子列表（当前）:**
   ```python
   self.factors = [
       MACD(),
       BBI(),
       CCI(),
   ]
   ```

2. `_weights.py` - 因子权重配置

   ```python
   FactorWights = {
       'MACD': 1.0,
       'BBI': 0.5,
       'CCI': 0.5,
   }
   ```

   **TODO:** 该权重配置当前未被使用，需要在策略中实现加权逻辑。

3. `sizers/sizer.py` - 仓位管理与资金分配

   **核心参数:**
   ```python
   HAND_SIZE = 100           # 一手 = 100 股
   MIN_BUY_AMOUNT = 10,000   # 最小买入金额
   MAX_BUY_AMOUNT = 25,000   # 最大买入金额
   PRESERVE_AMOUNT = 0       # 保留金额
   ```

   **分配策略:**
   - 均等分配: `budget / len(stocks)`
   - 约束条件: MIN_BUY_AMOUNT ≤ 分配金额 ≤ MAX_BUY_AMOUNT
   - 整手买入: 向上取整到 100 股的倍数

### 4. trading - 实盘交易模块

**核心文件:**

1. `main.py` - 交易主程序

   **交易流程:**
   ```python
   TradingScheduler
   │
   ├─> before_trade()        # 盘前准备
   │   └─> 订阅全市场行情
   │
   ├─> while_trade()         # 盘中循环
   │   └─> buy_task()
   │       ├─> 14:40-14:55 尾盘买入
   │       ├─> TopN 选股
   │       └─> TODO: 实现买入逻辑
   │
   └─> after_trade()         # 盘后清理
       └─> 取消订阅
   ```

2. `trader.py` - QMT 交易接口封装

   **主要方法:**
   - `order()` - 下单（买入/卖出）
   - `cancel_order()` - 撤单
   - `query_asset()` - 查询资产
   - `query_positions()` - 查询持仓
   - `clear_position()` - 清仓
   - `query_buy_trades()` - 查询当日成交

3. `scheduler.py` - 交易调度器
   - 管理交易时间
   - 盘前/盘中/盘后任务调度

4. `lark/` - 飞书通知模块
   - `sender.py` - 发送交易通知
   - `receiver.py` - 接收飞书指令

### 5. testback - 回测模块

**核心文件:**

1. `main.py` - 回测主程序

   **回测架构（基于多进程 + 共享内存）:**
   ```python
   多线程获取 TopN 实例（每个交易日）
   │
   ├─> 序列化到共享内存
   │
   └─> 多进程并行回测（Joblib + Loky）
       └─> 每个进程使用不同的因子权重
           └─> TODO: 计算收益并返回
   ```

   **设计亮点:**
   - 共享内存: 避免重复序列化大对象
   - 多进程: 绕过 Python GIL，充分利用 CPU
   - 批量计算: 一次性获取所有日期的因子数据

   **TODO:**
   - 实现收益计算逻辑
   - 实现遗传算法搜索最优权重

### 6. utils - 工具函数

**主要模块:**

1. `parallel.py` - 并行计算工具
   - `batch_run_threads()` - 多线程批量执行

2. `stock/` - 股票工具
   - `format.py` - 格式化工具（股票代码、日期等）
   - `time.py` - 交易时间处理
   - `holiday.py` - 交易日历
   - `info.py` - 股票信息判断（可转债等）

3. `recorder.py` - 日志记录器
4. `hash.py` - 哈希工具

## 数据流

### 选股流程

```
allow_buy_stock_code_list(date)  # 获取股票池
    ↓
TopN(stock_list, base_date)      # 多因子选股
    ↓
多线程并行计算因子分数
    ├─> MACD
    ├─> BBI
    └─> CCI
    ↓
汇总: factor_scores = {
    'MACD': {...},
    'BBI': {...},
    'CCI': {...}
}
    ↓
TODO: 应用权重 + BatchNorm
    ↓
TODO: 排序并选取 Top N
```

### 交易流程（实盘）

```
TradingScheduler.start_check_trading()
    ↓
盘中循环（每秒检查）
    ↓
14:40-14:55 触发 buy_task()
    ↓
TopN 选股
    ↓
TODO: Sizer 计算仓位
    ↓
TODO: Trader.order() 下单
    ↓
TODO: 订单跟踪与风控
```

## 当前架构的关键问题

### 1. 策略层缺失

**问题:**
- `TopN.factor_scores` 只是存储了因子分数，**没有实现加权、归一化、排序逻辑**
- `_weights.py` 中的权重配置未被使用
- 无法输出最终的股票排名列表

**解决方案:**
需要在 `TopN` 中实现 `get_ordered_stocks()` 方法:
```python
def get_ordered_stocks(self, n: int = 30) -> List[str]:
    # 1. 对每个因子应用 BatchNorm
    # 2. 加权求和
    # 3. 按综合得分排序
    # 4. 返回 Top N
```

### 2. 交易执行逻辑缺失

**问题:**
- `trading/main.py` 中的 `buy_task()` 只调用了选股，**未实现买入逻辑**
- 缺少卖出逻辑（何时卖出？如何调仓？）

**TODO:**
- 实现 buy_n sell_m 或 rank_n 调仓机制（见架构设计分析）
- 实现订单跟踪和异常处理

### 3. 回测逻辑缺失

**问题:**
- `testback/main.py` 只创建了多进程框架，**未实现收益计算**
- 无法验证策略有效性

**TODO:**
- 实现回测收益计算
- 实现遗传算法优化权重

## 缓存系统

**位置:** `testback/.cache/.cache/`

**结构:**
```
{stock_code}/
    {stock_code}_{date}_factor_{factor_name}_{hash}.pkl
```

**示例:**
```
000010.SZ/
    000010.SZ_20251205_factor_MACD_hc58cdfdc.pkl
    000010.SZ_20251205_factor_BBI_h8de59753.pkl
```

**缓存机制:**
- 使用 `@cached_factor` 装饰器
- 基于 (股票代码, 日期, 因子名, 参数哈希) 缓存
- 避免重复计算相同因子

## 配置管理

**位置:** `configs/`

**主要配置:**
- `env.py` - 环境配置（从 `env.template.py` 复制）
  - `TRADE_ACCOUNT` - 交易账号

## 依赖包

详见 `pyproject.toml`:
- `xtquant` - QMT 量化接口
- `ta-lib` - 技术指标库
- `joblib` / `loky` - 并行计算
- `numpy` / `pandas` - 数据处理
- `lark-oapi` - 飞书 API
- `loguru` - 日志管理

## 开发指引

### 添加新因子

1. 在 `core/factors/` 创建新文件（如 `MyFactor.py`）
2. 继承 `BaseFactor` 并实现 `calc()` 方法
3. 使用 `@cached_factor` 装饰器
4. 在 `TopN.factors` 列表中添加
5. 在 `_weights.py` 中配置权重

### 性能优化

- 使用 `@profile` 装饰器（来自 `line_profiler`）
- 并行计算: `batch_run_threads()` 或 `Joblib`
- 缓存: `@cached_factor` 自动缓存

### 日志系统

- `core/logger.py` - 核心模块日志
- `trading/logger.py` - 交易日志
- `testback/logger.py` - 回测日志

## 实盘运行流程

1. 配置 `configs/env.py`
2. 登录 QMT 客户端（极简模式）
3. 运行 `.\run.ps1`
4. 系统自动在 14:40-14:55 执行选股和买入

## Git 状态（当前）

**新增文件（未提交）:**
- `core/factors/ADX.py`
- `core/factors/BollingerBands.py`
- `core/factors/KDJ.py`
- `core/factors/RSI.py`
- `core/factors/TRIX.py`
- `core/factors/MOM.py`
- `core/factors/MIGRATION_README.md`
- `core/factors/helpers/batch_norm.py`
- `core/factors/helpers/signal_detection.py`

**修改文件:**
- `core/factors/helpers/__init__.py`
- `core/factors/helpers/indicators.py`

**删除文件:**
- `configs/env.template.py`

## 后续改进方向

1. 实现完整的选股策略（BatchNorm + 加权 + 排序）
2. 实现交易执行逻辑（buy_n sell_m 或 rank_n）
3. 实现回测逻辑并优化因子权重
4. 添加风控模块（止损、仓位控制等）
5. 完善飞书通知（交易结果、异常告警）
6. 实现因子平滑参数搜索（遗传算法）

---

**最后更新:** 2025-12-13
