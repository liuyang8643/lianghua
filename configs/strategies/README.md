# 策略配置文件说明

## 目录结构

```
configs/strategies/
├── base.yaml           # 基础配置（所有策略的默认值）
├── strategy_v1.yaml    # 保守型策略
├── strategy_v2.yaml    # 激进型策略
└── README.md           # 本文档
```

## 配置文件设计理念

### 1. 继承机制

所有策略配置都基于 `base.yaml`，只需覆盖需要修改的部分：

```yaml
# base.yaml 中定义了完整的默认配置
factors:
  MACD:
    weight: 1.0
    temperature: 1.0

# strategy_v1.yaml 只覆盖需要修改的部分
factors:
  MACD:
    temperature: 0.8  # 只修改温度，weight继承base
```

### 2. 配置层级

```
base.yaml (基础默认值)
    ↓
strategy_vX.yaml (策略特定配置)
    ↓
运行时参数 (命令行参数，可选)
```

## 配置文件结构

### meta - 策略元信息

```yaml
meta:
  name: "策略名称"
  version: "v1.0"
  description: "策略描述"
  created_at: "2025-12-14"
  author: "作者"
  base_config: "base.yaml"  # 继承的基础配置
  changelog:                 # 变更日志
    - "2025-12-14: 初始版本"
```

### factors - 因子配置

```yaml
factors:
  FactorName:
    enabled: true           # 是否启用
    weight: 1.0            # 因子权重（用于加权求和）
    temperature: 1.0       # 温度参数（控制归一化的离散程度）
    search_range:          # GA搜索范围
      weight: [0.5, 2.0]
      temperature: [0.5, 2.0]
    params:                # 因子特定参数
      period: 14
```

**温度参数说明:**
- **低温度 (0.2-0.8)**: 压缩分数差异，排名稳定，适合长期因子
- **中温度 (0.8-1.5)**: 平衡，适合大多数因子
- **高温度 (1.5-3.0)**: 放大分数差异，排名灵敏，适合短期因子

**权重说明:**
- 权重决定该因子对最终排名的贡献度
- 所有启用因子的加权分数求和得到最终得分
- 建议主导因子权重=1.0，辅助因子权重=0.3-0.8

### selection - 选股策略配置

```yaml
selection:
  strategy: "TopN"         # 策略类型
  rank_n: 30              # 选取前N只股票
  stock_pool: "allow_buy" # 股票池来源
```

### position - 仓位管理配置

```yaml
position:
  hand_size: 100              # 一手股数
  min_buy_amount: 10000       # 最小买入金额（元）
  max_buy_amount: 25000       # 最大买入金额（元）
  preserve_amount: 0          # 保留金额（元）
  allocation_method: "equal"  # 分配方法
```

### capital - 资金配置

```yaml
capital:
  initial_capital: 1000000    # 初始资金（元）
  max_position_ratio: 0.95    # 最大持仓比例
```

### trading - 交易时间配置

```yaml
trading:
  buy_time_start: "14:40:00"
  buy_time_end: "14:55:00"
  sell_time_start: "09:35:00"
  sell_time_end: "09:45:00"
```

### backtest - 回测配置

```yaml
backtest:
  start_date: "20240101"
  end_date: "20241231"
  commission_rate: 0.0003     # 万三
  slippage_rate: 0.001        # 千一
  min_commission: 5
```

### genetic_algorithm - 遗传算法配置

```yaml
genetic_algorithm:
  enabled: false
  population_size: 50
  generations: 100
  mutation_rate: 0.1
  crossover_rate: 0.8
  objective:
    metric: "sharpe_ratio"    # 优化指标
    direction: "maximize"
```

### risk_control - 风控配置

```yaml
risk_control:
  max_drawdown_ratio: 0.2
  stop_loss_ratio: 0.1
  stop_profit_ratio: 0.5
```

## 使用方法

### 1. 创建新策略配置

```bash
# 复制一个现有配置作为模板
cp strategy_v1.yaml strategy_v3.yaml

# 编辑新配置
vim strategy_v3.yaml
```

### 2. 在代码中加载配置

```python
# 加载配置文件（伪代码，需要实现）
from core.config import StrategyConfig

# 加载特定策略配置
config = StrategyConfig.load("strategy_v1.yaml")

# 访问配置
print(config.factors["MACD"].weight)
print(config.capital.initial_capital)

# 获取启用的因子列表
enabled_factors = config.get_enabled_factors()
```

### 3. 版本对比

```bash
# 使用git diff对比两个策略配置
git diff --no-index strategy_v1.yaml strategy_v2.yaml

# 或使用其他diff工具
diff -u strategy_v1.yaml strategy_v2.yaml
```

## 策略配置建议

### 保守型策略

- **因子数量**: 2-4个
- **温度参数**: 0.5-1.0（低温度，稳定排名）
- **rank_n**: 30-50（分散持仓）
- **适用场景**: 长期持有，降低交易频率

### 激进型策略

- **因子数量**: 5-8个
- **温度参数**: 1.5-2.5（高温度，灵敏排名）
- **rank_n**: 20-30（集中持仓）
- **适用场景**: 短期交易，追求高收益

### 平衡型策略

- **因子数量**: 4-6个
- **温度参数**: 0.8-1.5（中等温度）
- **rank_n**: 25-35
- **适用场景**: 中期持有，平衡收益与风险

## 注意事项

1. **配置验证**: 修改配置后，建议先运行回测验证
2. **版本管理**: 每次重要修改都应创建新版本并记录changelog
3. **参数搜索**: 使用GA优化时，合理设置search_range
4. **风险控制**: 不要过度拟合历史数据，注意前瞻性偏差

## 配置文件更新日志

- **2025-12-14**: 初始版本，创建base.yaml和两个策略示例
