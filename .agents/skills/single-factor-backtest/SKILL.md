---
name: single-factor-backtest
description: 在 WBR 中用 `testback/ga.py --mode single` 回测某个单因子在指定日期区间内的表现。用于基于 `results/single_smallcap_30d_smoketest_config.json` 准备单次回测配置、覆盖 `--start-date` 和 `--end-date`、生成 `results/single_*` HTML 报告，或排查某个单因子在 single 模式下的收益、换仓与持仓表现。
---

# 单因子回测

## 概述

使用 [`testback/ga.py`](C:/Users/Sleaf/PycharmProjects/WBR/testback/ga.py) 的 `single` 模式，对一个因子在指定区间内做一次完整回测。把 [`results/single_smallcap_30d_smoketest_config.json`](C:/Users/Sleaf/PycharmProjects/WBR/results/single_smallcap_30d_smoketest_config.json) 当成最小模板，而不是日期来源。

## 工作流

1. 复制样例配置到新的 JSON 文件，不要直接覆盖仓库里的 smoketest 文件，除非用户明确要求。
2. 在新配置里只保留一个目标因子权重，并同步更新 `temperatures`。
3. 如果任务指定了时间段，始终显式传 `--start-date` 和 `--end-date`；不要依赖默认区间。
4. 从仓库根目录运行 `uv run python testback/ga.py --mode single ...`。
5. 检查输出目录下的 HTML 报告和日志。
6. 如果在数据准备阶段出现 `xtquant` 或本地行情连接问题，再使用 `$qmt-launch`。

## 配置模板

以 [`results/single_smallcap_30d_smoketest_config.json`](C:/Users/Sleaf/PycharmProjects/WBR/results/single_smallcap_30d_smoketest_config.json) 为起点。单因子最小结构如下：

```json
{
  "ga_profile": "smallcap_only",
  "individual_config": {
    "weights": {
      "SmallCap": 1.0
    },
    "buy_n": 30,
    "sell_m": 30,
    "temperatures": {
      "SmallCap": 1.0
    }
  }
}
```

按任务修改这些字段：

- `weights` 里只保留目标因子名。
- `temperatures` 的 key 要和 `weights` 保持一致。
- `buy_n` / `sell_m` 控制持仓数，通常单因子回测保持相等。
- 需要锁仓时再加 `freeze_days`。

## 运行命令

```powershell
uv run python testback/ga.py `
  --mode single `
  --individual-config results\single_my_factor.json `
  --start-date 2024-01-01 `
  --end-date 2024-03-31 `
  --output-dir results\single_my_factor_20240101_20240331
```

如果不传 `--output-dir`，脚本会自动生成 `results/single_<timestamp>`。single 模式默认会生成 HTML，并尝试自动打开浏览器。

## 日期与 Profile

- `results/single_smallcap_30d_smoketest_config.json` 这个文件名里的 `30d` 只是样例名称，不会自动限制回测区间。
- [`testback/ga.py`](C:/Users/Sleaf/PycharmProjects/WBR/testback/ga.py) 在主流程里先按 CLI `--profile` 计算因子历史需求和共享数据预加载窗口，再进入 `single` 模式。
- 如果目标因子不在当前默认 profile `smallcap_only` 里，不要假设预加载窗口一定正确。优先传一个包含该因子的 `--profile`；如果仓库里还没有对应 profile，就先更新 [`testback/ga_config.py`](C:/Users/Sleaf/PycharmProjects/WBR/testback/ga_config.py)。

## 注意事项

- `individual_config.weights` 里的因子名必须能在 `core.factors` 里直接找到同名类，否则脚本会退出。
- 不传日期时，当前默认区间是 `2020-06-30` 到 `2024-12-31`。任务只要提到“某段时间”，就显式传日期，避免误跑全区间。
- single 模式会生成换仓、持仓、收益和指标报告；先看 HTML，再决定是否深入翻日志。
