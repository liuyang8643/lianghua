# 因子 Benchmark 模块

本模块用于分析因子分数与未来收益率之间的相关性，核心是按交易日做横截面相关性统计，并可选地按股票做时间序列相关性统计。

## 公开接口

```python
from core.factors.benchmark import calculate_factor_correlation, generate_html_report
```

- `calculate_factor_correlation(...)` 计算相关性并返回 `FactorCorrelationReport`
- `generate_html_report(report)` 把结果渲染成 HTML

## 分析类型

### 类型 1：同一天、不同股票

默认路径。对每个交易日：

1. 计算股票池里所有股票的因子分数
2. 计算这些股票在 `T+M` 日的收益率
3. 对“因子分数 vs. 未来收益率”计算 Pearson / Spearman 相关系数

最终输出每个持有期的：

- 加权平均相关系数
- 中位数相关系数
- IC / IR / ICIR
- 正负相关分布

### 类型 2：同一股票、不同日期

设置 `show_stock_correlation=True` 后启用。它会对每只股票在不同买入日上的因子分数与未来收益率做相关性统计。

## 使用示例

```python
from datetime import date

from core.database import get_all_stock_code_list
from core.factors import WMACross
from core.factors.benchmark import calculate_factor_correlation, generate_html_report

stock_list = get_all_stock_code_list()

report = calculate_factor_correlation(
    factor_cls=WMACross,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    m_days=[1, 3, 5, 10, 20],
    stock_codes=stock_list,
    save_stock_scores=False,
    show_stock_correlation=False,
)

html_file = generate_html_report(report)
print(html_file)
```

## 当前入口脚本

### 1. 运行当前 benchmark 入口

```powershell
python -m core.factors.benchmark.benchmark
```

`benchmark.py` 当前会：

- 读取全部股票列表，然后按环境变量决定是否抽样
- 用 `WMACross` 作为默认 benchmark 因子
- 默认分析 `2024-05-01` 到 `2025-05-01`
- 计算 `T+1/3/5/10/20/30/60`
- 总是保存 `.pkl` 报告到 `reports/`
- 在控制台打印每个持有期的汇总统计
- 仅在 `BENCHMARK_GENERATE_HTML=1/true/yes` 时生成 HTML

可用环境变量：

- `BENCHMARK_START_DATE`
- `BENCHMARK_END_DATE`
- `BENCHMARK_SAMPLE_STEP`
- `BENCHMARK_SAMPLE_SIZE`
- `BENCHMARK_GENERATE_HTML`

### 2. 从现有 `.pkl` 重新生成 HTML

```powershell
python -m core.factors.benchmark.calc_correlation
```

注意：`calc_correlation.py` 底部当前使用的是硬编码 `pkl_file` 路径，运行前需要先改成目标文件。

### 3. 启动因子可视化 Web 页面

```powershell
python -m core.factors.benchmark.web_chart
python -m core.factors.benchmark.web_chart --port 9090
python -m core.factors.benchmark.web_chart --code 600000.SH
```

`web_chart.py` 当前会：

- 默认随机选择一只 `allow_buy_stock_code_list()` 中的股票并启动本地 HTTP 服务
- 在页面中展示 K 线、成交量、MA5、MA20 和因子得分
- 当前仅计算并展示 `WMACross`
- 当前把图表时间范围限制在 `2021-01-01` 到 `2022-12-31`
- 使用浏览器端 `ECharts` 渲染图表

## 输出位置

`benchmark.py` 和 `calc_correlation.py` 的报告输出写入当前工作目录下的 `reports/` 目录，而不是固定写到模块目录。

常见产物：

- `factor-correlation-<factor>-<timestamp>.pkl`
- `factor-correlation-<factor>-T+<m_days>-<timestamp>.html`

`web_chart.py` 在 `--code` 模式下会把 HTML 临时写到系统临时目录，然后自动打开浏览器；服务模式下不会固定落库到仓库目录。

## 参数说明

`calculate_factor_correlation(...)` 主要参数：

- `factor_cls`: 因子类
- `start_date`: 开始日期
- `end_date`: 结束日期
- `m_days`: 单个持有期或持有期列表
- `stock_codes`: 股票代码列表
- `save_stock_scores`: 是否把单日股票明细保存在报告对象里
- `show_stock_correlation`: 是否计算类型 2 结果

## 实现要点

- 会在内部调用 `init_stock_detail_cache()` 和 `init_full_data()`
- 使用 `joblib` 多进程并行
- 因子分数会复用 `core/factors/helpers/.cache/` 中的磁盘缓存
- HTML 由 `report.py` 使用 Jinja2 模板生成

## 注意事项

- 股票池过大、持有期过多时，计算会很慢
- `BENCHMARK_SAMPLE_STEP` 和 `BENCHMARK_SAMPLE_SIZE` 可以先把股票池缩小，再做试跑
- `save_stock_scores=True` 会明显增加内存和 pickle 体积
- `benchmark.py` 默认不会生成 HTML，需要显式设置 `BENCHMARK_GENERATE_HTML`
- `web_chart.py` 当前模块注释写的是更宽的时间范围，但实际代码只展示 `2021-01-01` 到 `2022-12-31`
- 相关性只表示统计关系，不表示因果关系
