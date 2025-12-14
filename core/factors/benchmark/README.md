# 因子相关性分析模块（日度截面分析）

## 功能说明

本模块用于分析因子得分排名与T+M日收益率的相关性，采用日度截面相关性分析方法，使用皮尔逊相关系数和斯皮尔曼等级相关系数进行统计分析。

## 分析原理

### 日度截面相关性

不同于传统的单股票时间序列相关性，本模块采用**日度截面相关性**分析方法：

1. 对于每个交易日（如9月1日）：
   - 计算所有股票（如A、B、C）的因子得分（A1, B1, C1）
   - 计算每只股票从当日到T+M日（如T+5日，即9月6日）的收益率（Aa1, Ba1, Ca1）
   - 计算因子得分与T+M日收益率的相关系数

2. 重复以上步骤计算每个交易日的相关性

3. 最终汇总所有交易日的平均相关性

这种方法能够评估因子在**横截面上**的预测能力，即因子能否区分同一时间点不同股票的未来表现。

## 主要特性

- **日度截面分析**: 每日计算股票池的因子排名与未来收益率的相关性
- **T+M日收益率**: 可自定义持有期（默认T+5日）
- **双重相关系数**: 同时计算Pearson和Spearman相关系数
- **并行计算**: 利用多核CPU实现高效的批量计算
- **可视化报告**: 生成交互式HTML报告，包含时间序列图、分布图和详细数据
- **股票详情展开**: 点击每日数据可查看当日所有股票的因子得分和收益率

## 使用方法

### 基本用法

```python
from datetime import date
from core.factors.benchmark import calculate_factor_correlation, generate_html_report
from core.factors import MACD
from core.database import allow_buy_stock_code_list

# 获取股票列表
stock_list = allow_buy_stock_code_list()

# 计算MACD因子与T+5日收益率的相关性
report = calculate_factor_correlation(
  factor_cls=MACD,
  start_date=date(2024, 1, 1),
  end_date=date(2024, 12, 31),
  m_days=5,  # T+5日收益率
  stock_codes=stock_list  # 可选，默认使用允许买入的股票列表
)

# 生成HTML报告
html_file = generate_html_report(report)
print(f"报告已生成: {html_file}")
```

### 直接运行

```bash
cd C:\Users\Sleaf\PycharmProjects\WBR
python -m core.factors.benchmark.main
```

## 报告内容

生成的HTML报告包含以下内容：

1. **汇总统计**
   - 分析时间范围
   - 收益率周期（T+M日）
   - 股票总数和有效交易日数
   - 平均相关系数和中位数
   - 正负相关天数分布

2. **每日相关系数时间序列图**
   - 展示Pearson和Spearman相关系数的时间变化趋势
   - 支持缩放和数据选择

3. **相关系数分布图**
   - 直方图展示相关系数的分布情况
   - 绿色表示正相关，红色表示负相关

4. **每日相关性详情表**
   - 交易日期
   - Pearson相关系数
   - Spearman等级相关系数
   - 可视化条形图
   - P值（显著性检验，P < 0.05表示统计学显著）
   - 有效股票数量
   - 点击可展开查看当日所有股票的因子得分和收益率

5. **股票明细（可展开）**
   - 股票代码和名称
   - 因子得分
   - 当日收盘价
   - T+M日收盘价
   - 收益率

6. **交互功能**
   - 过滤：按相关性类型（正/负/无效）筛选
   - 展开/收起：查看每日股票详情

## 依赖项

- `scipy`: 相关系数计算（pearsonr, spearmanr）
- `joblib`: 并行计算
- `jinja2`: HTML模板渲染
- `numpy`: 数值计算
- `pandas`: 数据处理

## 技术说明

### 相关性计算方法

1. 获取指定时间范围内的所有交易日
2. 对每个交易日并行计算：
   - 确定T+M日（如T+5日）
   - 计算所有股票的因子得分
   - 获取当日和T+M日的收盘价
   - 计算T+M日收益率 = (T+M日收盘价 - 当日收盘价) / 当日收盘价
   - 计算因子得分与收益率的Pearson和Spearman相关系数
3. 汇总统计所有交易日的平均相关性

### 性能优化

- 使用`joblib`并行处理多个交易日
- 利用项目现有的历史数据缓存机制
- 工作进程数自动适配CPU核心数和交易日数量

### 数据质量

- 至少需要10只股票有效数据才计算相关性
- 自动过滤NaN和无效数据
- 自动处理停牌等特殊情况

## 参数说明

### calculate_factor_correlation

- `factor_cls`: 因子类（如MACD, KDJ等）
- `start_date`: 分析开始日期
- `end_date`: 分析结束日期
- `m_days`: T+M日收益率的M值（默认5，表示T+5日）
- `stock_codes`: 股票代码列表（可选，默认使用允许买入列表）

## 示例场景

### 场景1：评估MACD因子的预测能力

```python
# 计算MACD因子能否预测未来5日的收益
report = calculate_factor_correlation(
  factor_cls=MACD,
  start_date=date(2024, 1, 1),
  end_date=date(2024, 12, 31),
  m_days=5
)
# 如果平均相关系数为正且显著，说明MACD高的股票未来5日表现更好
```

### 场景2：对比不同持有期

```python
# 对比T+1, T+5, T+10日的预测能力
for m in [1, 5, 10]:
  report = calculate_factor_correlation(
    factor_cls=KDJ,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    m_days=m
  )
  print(f"T+{m}日平均相关性: {report.avg_correlation:.4f}")
```
- 详细记录计算失败原因

## 注意事项

1. 首次运行可能需要下载大量历史数据，耗时较长
2. 建议在非交易时间运行，避免数据更新影响
3. 相关性不等于因果关系，需结合其他指标综合判断
4. P值 < 0.05 表示相关性具有统计学意义

## 示例输出

报告文件保存在 `./logs` 目录下，文件名格式：

```
factor-correlation-{因子名称}-{时间戳}.html
```

报告生成后会自动在默认浏览器中打开。

