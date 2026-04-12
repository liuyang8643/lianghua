---
name: factor-correlation
description: 在 WBR 中测试单因子的相关性 / IC 表现并保存报告产物。用于对指定因子类在给定日期区间、股票池和持有期上运行 `core/factors/benchmark/calc_correlation.py` 的 `calculate_factor_correlation(...)`，生成 `.pkl` 与 HTML 报告，或基于已有 `.pkl` 重新生成 HTML。
---

# 单因子相关性

## 概述

把 [`core/factors/benchmark/calc_correlation.py`](C:/Users/Sleaf/PycharmProjects/WBR/core/factors/benchmark/calc_correlation.py) 当成单因子相关性检查的主入口。新跑一次相关性时，优先导入 `calculate_factor_correlation(...)` 和 `generate_html_report(...)`；不要把 [`core/factors/benchmark/benchmark.py`](C:/Users/Sleaf/PycharmProjects/WBR/core/factors/benchmark/benchmark.py) 当成通用单因子入口，因为它当前固定使用 `WMACross`。

## 工作流

1. 从仓库根目录运行，优先使用 `uv run python`。
2. 从 `core.factors` 解析目标因子类；如果类名不存在，先停止并修正因子名。
3. 用 `get_all_stock_code_list()` 构造股票池；试跑时先做采样，再跑全量。
4. 始终显式传 `start_date`、`end_date` 和 `m_days`，不要依赖默认日期。
5. 调用 `calculate_factor_correlation(...)`，默认保持 `save_stock_scores=False`、`show_stock_correlation=False`。
6. 把返回的 `report` 保存为 `reports/factor-correlation-<factor>-<timestamp>.pkl`，再调用 `generate_html_report(report)` 生成 HTML。
7. 如果出现 `xtquant` 或本地 QMT 连接问题，再使用 `$qmt-launch`。
8. 只在用户明确需要“同一股票、不同日期”的时间序列相关性时，把 `show_stock_correlation` 设为 `True`。

## 运行模板

```powershell
@'
from datetime import date, datetime
import os
import pickle

from core.database import get_all_stock_code_list
import core.factors as factor_module
from core.factors.benchmark import calculate_factor_correlation, generate_html_report

factor_name = "WMACross"
start_date = date(2024, 1, 1)
end_date = date(2024, 12, 31)
m_days = [1, 3, 5, 10, 20]
sample_step = 5
sample_size = 500
show_stock_correlation = False

factor_cls = getattr(factor_module, factor_name)

stock_codes = get_all_stock_code_list()
if sample_step > 1:
    stock_codes = stock_codes[::sample_step]
if sample_size > 0:
    stock_codes = stock_codes[:sample_size]

report = calculate_factor_correlation(
    factor_cls=factor_cls,
    start_date=start_date,
    end_date=end_date,
    m_days=m_days,
    stock_codes=stock_codes,
    save_stock_scores=False,
    show_stock_correlation=show_stock_correlation,
)

os.makedirs("reports", exist_ok=True)
pickle_path = os.path.join(
    "reports",
    f"factor-correlation-{report.factor_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
)
with open(pickle_path, "wb") as f:
    pickle.dump(report, f, protocol=pickle.HIGHEST_PROTOCOL)

html_path = generate_html_report(report)
print(f"Pickle报告已保存: {pickle_path}")
print(f"HTML报告已生成: {html_path}")
'@ | uv run python -
```

把模板里的 `factor_name`、日期、持有期和采样参数替换成任务实际值。先用采样做 smoke test，确认能跑通后再放大全量股票池。

## 现有报告转 HTML

如果用户已经给了 `.pkl` 报告，只需要重生 HTML，可以直接改 [`core/factors/benchmark/calc_correlation.py`](C:/Users/Sleaf/PycharmProjects/WBR/core/factors/benchmark/calc_correlation.py) 底部的 `pkl_file`，然后运行：

```powershell
uv run python -m core.factors.benchmark.calc_correlation
```

这个模块入口当前只适合“从现有 `.pkl` 生成 HTML”，不适合作为带参数的新相关性 CLI。

## 注意事项

- `factor_name` 必须是 `core.factors` 下可直接访问的类名。
- 股票池过大、持有期过多时会很慢；先小样本试跑，再跑全量。
- `save_stock_scores=True` 会明显增大内存和 pickle 体积，默认不要打开。
- 输出目录是当前工作目录下的 `reports/`，不是模块目录。
