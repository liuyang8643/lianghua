# GA多目标遗传算法优化器

参考 qmt-trade 项目的遗传算法实现，适配到 WBR 量化交易系统。

## 特性

1. **多目标优化**：同时优化年化超额收益、夏普比率、Calmar比率
2. **历史最优个体选择**：每代从历史记录中选取最优个体参与进化
3. **3k评估策略**：每代评估 k父代 + k子代 + k历史最优（共3k个体）
4. **多进程并行**：充分利用多核CPU加速评估
5. **进度保存与恢复**：支持中断后继续运行
6. **NSGA-II算法**：使用DEAP框架实现多目标优化

## 架构设计

### 模块结构

```
testback/
├── ga/
│   ├── __init__.py           # 模块导出
│   ├── ga_optimizer.py       # GA优化器核心类
│   ├── fitness.py            # 适应度函数
│   └── utils.py              # 工具函数（进度保存/加载等）
└── ga_main.py                # 主程序入口
```

### 核心流程

```
准备数据
  └─> 获取TopN因子数据（多线程）
  └─> 写入共享内存
       ↓
初始化GA
  └─> 定义因子权重范围
  └─> 设置种群大小、代数等参数
       ↓
第0代：生成3k随机个体 → 评估 → 选择k个父代
       ↓
后续代（循环）：
  ├─> 保留k个父代
  ├─> 交叉+变异生成k个子代
  ├─> 从历史记录中选择k个最优个体
  ├─> 并行评估3k个体
  └─> NSGA-II选择k个作为下一代父代
       ↓
输出Pareto前沿
  └─> 保存最终结果
```

### 适应度评估流程

```
evaluate_individual(weights)
  ↓
从共享内存加载TopN数据
  ↓
加权求和计算综合得分
  score = Σ(weight_i × factor_score_i)
  ↓
选取Top N股票
  ↓
模拟回测（差集调仓）
  ├─> 卖出不在Top N中的持仓
  └─> 买入新的Top N股票
  ↓
计算收益指标
  ├─> 年化超额收益率
  ├─> 夏普比率
  └─> Calmar比率
  ↓
返回 (超额, 夏普, Calmar, 详细指标)
```

## 使用方法

### 1. 基本用法

```bash
# 新运行（默认参数）
python -m testback.ga_main

# 自定义参数
python -m testback.ga_main \
    --population 24 \
    --generations 100 \
    --backtest-days 180 \
    --initial-capital 500000 \
    --rank-n 20 \
    --max-workers 24
```

### 2. 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--population` | 种群大小（k） | 24 |
| `--generations` | 进化代数 | 100 |
| `--backtest-days` | 回测天数 | 180 |
| `--initial-capital` | 初始资金 | 500000 |
| `--rank-n` | 选股数量 | 20 |
| `--stock-limit` | 股票池数量限制 | None（全部）|
| `--max-workers` | 最大进程数 | = population |
| `--resume-file` | 恢复文件路径 | None |
| `--resume-best-only` | 只恢复最佳个体 | False |

### 3. 恢复运行

```bash
# 恢复整个种群（推荐）
python -m testback.ga_main \
    --resume-file results/ga_20250118_120000/progress.json

# 只恢复最佳个体（热启动）
python -m testback.ga_main \
    --resume-best-only \
    --resume-file results/ga_20250118_120000/final_result.json
```

### 4. 快速测试

```bash
# 小规模测试（10个个体，10代，30天）
python -m testback.ga_main \
    --population 10 \
    --generations 10 \
    --backtest-days 30 \
    --stock-limit 50 \
    --max-workers 10
```

## 输出文件

运行后会在 `results/ga_YYYYMMDD_HHMMSS/` 目录下生成：

### 1. progress.json

实时进度文件，包含每代的详细信息：

```json
{
  "current_generation": 10,
  "total_time": 3600.5,
  "last_update": "20250118 14:30:00",
  "generations": [
    {
      "generation": 0,
      "population_size": 72,
      "selected_size": 24,
      "generation_time": 120.5,
      "memory_mb": 2048.3,
      "pareto_front": [...],
      "population_stats": {
        "mean_obj1": 0.15,
        "std_obj1": 0.05,
        "pareto_front_size": 5
      },
      "all_individuals": [
        {
          "individual_id": 0,
          "genes": {"MACD": 1.2, "BBI": 0.8, ...},
          "objective1_annualized_excess": 0.18,
          "objective2_sharpe_ratio": 1.5,
          "objective3_calmar_ratio": 2.3,
          "annualized_return": 0.25,
          "max_drawdown": 0.12,
          ...
        },
        ...
      ]
    },
    ...
  ]
}
```

### 2. final_result.json

最终结果文件，包含Pareto前沿：

```json
{
  "timestamp": "20250118_120000",
  "mode": "multi_objective",
  "optimization_targets": ["annualized_excess", "sharpe_ratio", "calmar_ratio"],
  "config": {
    "population": 24,
    "generations": 100,
    "backtest_days": 180,
    ...
  },
  "pareto_front": [
    {
      "genes": {"MACD": 1.5, "BBI": 0.9, ...},
      "objective1_annualized_excess": 0.22,
      "objective2_sharpe_ratio": 2.1,
      "objective3_calmar_ratio": 3.5,
      ...
    },
    ...
  ],
  "pareto_front_size": 8,
  "total_time_seconds": 7200.0
}
```

## 核心算法

### 1. NSGA-II选择

使用DEAP库的 `tools.selNSGA2()` 实现多目标优化选择：

- 非支配排序（Pareto前沿分层）
- 拥挤距离计算（保持多样性）
- 精英保留策略

### 2. 历史最优个体选择

每代从 `progress.json` 中提取历史个体：

1. 收集所有历史个体
2. 按基因去重
3. 计算每个基因的平均超额收益
4. 排序并选取前k个（剔除与父代+子代重复的）
5. 如果不足k个，用随机个体补充

### 3. 遗传操作

- **交叉**：单点交叉（交叉率80%）
- **变异**：均匀变异（变异率20%）
- **约束**：因子权重范围 [0, 2]

## 性能优化

1. **多进程并行**：使用 `ProcessPoolExecutor` 并行评估个体
2. **共享内存**：TopN数据通过共享内存传递，避免重复序列化
3. **缓存机制**：因子计算结果缓存（继承自TopN策略）
4. **增量评估**：每代只评估新个体，不重复计算

## 注意事项

1. **内存占用**：每代需要加载3k个体的回测数据，建议至少16GB内存
2. **进程数**：建议 `max_workers = population`，充分利用多核
3. **回测周期**：建议至少180天，太短的周期容易过拟合
4. **股票池**：测试时可以用 `--stock-limit 100` 限制股票数量加速
5. **恢复运行**：Resume时会继承输出目录，不会创建新目录

## 与参考项目的差异

| 特性 | qmt-trade | WBR |
|------|-----------|-----|
| 数据源 | TradingEnv（预计算因子） | TopN（实时计算） |
| 共享方式 | 共享内存（因子缓存） | 共享内存（TopN对象） |
| 评估方式 | 直接加权求和 | 加权求和 + 回测 |
| 优化目标 | 年化超额 | 超额+夏普+Calmar |
| 项目耦合 | 深度集成 | 独立模块 |

## 后续改进

1. **动态回测周期**：每代随机选择回测周期（避免过拟合）
2. **多策略集成**：支持不同的调仓策略（buy_n_sell_m、rank_n等）
3. **约束优化**：添加最大回撤约束、交易次数约束等
4. **可视化**：集成 wandb 实时监控优化过程
5. **超参数优化**：自动搜索最优的变异率、交叉率等

## 示例结果

```
[第10代] 个体权重:
  [ 0] 超额=+0.2250 夏普=2.15 Calmar=3.80 | {"MACD": 1.52, "BBI": 0.88, ...}
  [ 1] 超额=+0.2100 夏普=2.45 Calmar=3.20 | {"MACD": 1.38, "BBI": 0.95, ...}
  ...

📊 Pareto前沿: 5个个体
  [0] 超额=+0.2250 夏普=2.15 Calmar=3.80
  [1] 超额=+0.2100 夏普=2.45 Calmar=3.20
  [2] 超额=+0.1980 夏普=2.30 Calmar=3.50
  ...

时间统计:
  总耗时: 7200.5秒 (120.0分钟)
  平均每代: 72.0秒
  最快一代: 45.2秒
  最慢一代: 98.7秒
```

## 常见问题

### Q1: 为什么第0代评估3k个体？

A: 为了充分探索搜索空间，第0代生成3k个随机个体，评估后选择k个最优作为初始种群。

### Q2: 历史最优个体从哪里来？

A: 从 `progress.json` 的 `all_individuals` 字段中提取，包含所有历史代的所有评估个体。

### Q3: 为什么不直接修改项目代码？

A: 为了保持模块独立性，GA作为独立模块，通过TopN和共享内存与现有代码交互，不影响原有架构。

### Q4: 如何选择最佳权重？

A: Pareto前沿包含多个"最优"解，根据实际需求选择：
- 追求高收益 → 选择超额收益最高的
- 追求稳定性 → 选择夏普比率最高的
- 控制风控 → 选择Calmar比率最高的

## 许可

参考 qmt-trade 项目的GA实现，适配到WBR项目。
