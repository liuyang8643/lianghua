# reportor

`testback/reportor/` 是当前单次回测 HTML 报告的实现目录。

## 入口

- `report.py`
  - 当前单次回测报告的主实现
- `__init__.py`
  - 聚合导出报告相关 API

当前主要对外函数：

- `generate_single_report(report_data, output_dir)`
- `get_hs300_daily_returns(trade_dates)`
- `calc_metrics(...)`
- `calc_monthly_stats(...)`

## 模板与前端

当前报告页面由以下前端资源驱动：

- `testback/reportor/templates/single_report_styles.css`
  - 页面样式
- `testback/reportor/templates/single_report_module.js`
  - ECharts 图表
  - Tippy tooltip
  - TanStack Table 虚拟滚动表格

当前页面包含：

- 核心指标网格
- 净值曲线
- 收益率分布与盈亏分布图
- 月度收益表
- 交易记录、持仓、清仓、每日资金快照、退市归零事件表

## 数据流

1. `core/backtest.py:run_single_mode()` 在 single 模式下构造 `report_data`
2. `generate_single_report()` 从 `report_data` 计算：
   - 核心指标
   - 月度统计
   - 净值与收益率图表数据
   - 各类表格数据
3. 报告生成函数把前端模板资源内联进最终 HTML
4. 最终 HTML 会先做一层轻量 minify 再落盘

## 已知约束

- 调用方应统一从 `testback.reportor` 导入
- 历史的 `testback.report` 路径已不再保留
