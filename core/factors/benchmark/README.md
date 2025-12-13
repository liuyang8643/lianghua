# 因子相关性分析模块

## 功能说明

本模块用于分析因子得分与股票收盘价的相关性，使用皮尔逊相关系数进行统计分析。

## 主要特性

- **并行计算**: 利用多核CPU实现高效的批量计算
- **统计分析**: 使用皮尔逊相关系数评估因子与股价的线性关系
- **可视化报告**: 生成交互式HTML报告，包含分布图和详细数据表
- **缓存优化**: 复用历史数据缓存，提高计算效率

## 使用方法

### 基本用法

```python
from datetime import date
from core.factors.benchmark import calculate_factor_correlation, generate_html_report
from core.factors import MACD
from core.database import allow_buy_stock_code_list

# 获取股票列表
stock_list = allow_buy_stock_code_list()

# 计算MACD因子与股价的相关性
report = calculate_factor_correlation(
  factor_cls=MACD,
  start_date=date(2014, 1, 1),
  end_date=date(2025, 1, 1),
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
    - 股票总数（有效/无效）
    - 平均相关系数和中位数
    - 正负相关股票分布

2. **相关系数分布图**
    - 直方图展示相关系数的分布情况
    - 绿色表示正相关，红色表示负相关

3. **股票详情表**
    - 股票代码和名称
    - 相关系数（Pearson correlation coefficient）
    - P值（显著性检验，P < 0.05表示统计学显著）
    - 有效样本数量
    - 错误信息（如有）

4. **交互功能**
    - 搜索：按股票代码或名称筛选
    - 过滤：按相关性类型（正/负/失败）筛选
    - 排序：点击表头进行多维度排序

## 依赖项

- `scipy`: 皮尔逊相关系数计算
- `joblib`: 并行计算
- `jinja2`: HTML模板渲染
- `numpy`: 数值计算
- `pandas`: 数据处理

## 技术说明

### 相关性计算方法

1. 获取指定时间范围内的所有交易日
2. 对每只股票的每个交易日：
    - 计算因子得分（FactorResult.score）
    - 获取当日收盘价
3. 使用有效数据点计算皮尔逊相关系数
4. 进行显著性检验（p-value）

### 性能优化

- 使用`joblib`并行处理多只股票
- 利用项目现有的历史数据缓存机制
- 工作进程数自动适配CPU核心数

### 数据质量

- 至少需要10个有效样本才进行统计
- 自动过滤NaN和无效数据
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

