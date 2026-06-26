---
name: single-factor-backtest
description: 在 WBR 中用 `testback/run_backtest.py` 回测某个单因子在指定日期区间内的表现。用于基于 `configs/config.json` 准备临时配置、覆盖 `--start-date` 和 `--end-date`、生成 `results/single_*` HTML 报告，或排查某个单因子在 single 模式下的收益、换仓与持仓表现。
---

# 单因子回测

## 概述

使用 `testback/run_backtest.py` 对一个因子在指定区间内做一次完整回测。以 `configs/config.json` 的最终运行结构为最小模板，而不是日期来源。

## 工作流

1. 复制 `configs/config.json` 到临时 JSON 文件，不要直接覆盖最终实盘配置，除非用户明确要求。
2. 在新配置里只保留一个目标因子权重。
3. 如果任务指定了时间段，始终显式传 `--start-date` 和 `--end-date`；不要依赖默认区间。
4. 从仓库根目录运行 `uv run python testback/run_backtest.py ...`，优先显式传 `--output-dir`，并把程序输出重定向到临时日志文件，不要让完整 stdout/stderr 直接进入对话上下文。
5. 先根据退出码判断是否成功；成功时只确认输出目录、HTML 报告路径和临时日志路径，不要读取或转述程序输出内容。
6. 只有在命令失败时，才读取临时日志的最新 200 行，结合报错堆栈分析原因并提出修复方案。
7. 如果在数据准备阶段出现 `xtquant` 或本地行情连接问题，再使用 `$qmt-launch`。

## 配置模板

以 `configs/config.json` 为起点。单因子最小结构如下：

```json
{
  "ga_profile": "core",
  "individual_config": {
    "weights": {
      "AmountBasedSmallCap": 1.0
    },
    "buy_n": 30,
    "sell_m": 30
  }
}
```

按任务修改这些字段：

- `weights` 里只保留目标因子名。
- `buy_n` / `sell_m` 控制持仓数，通常单因子回测保持相等。
- 需要锁仓时再加 `freeze_days`。

## 运行命令

```powershell
$logPath = Join-Path $env:TEMP ("single-factor-backtest-" + (Get-Date -Format 'yyyyMMdd_HHmmss') + ".log")
$outputDir = "results\\single_my_factor_20240101_20240331"

uv run python testback/run_backtest.py `
  --individual-config results\single_my_factor.json `
  --start-date 2024-01-01 `
  --end-date 2024-03-31 `
  --output-dir $outputDir `
  > $logPath 2>&1

$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
  Get-Content $logPath -Tail 200
}
```

如果不传 `--output-dir`，脚本会自动生成 `results/single_<timestamp>`。single 模式默认会生成 HTML，并尝试自动打开浏览器。为了避免成功后还要回读日志，默认更推荐显式传 `--output-dir`。

推荐执行策略：

- 成功时，不读取 `Get-Content $logPath`，只汇报命令成功、`$outputDir` 或通过文件系统定位到的最新 `results/single_*` 输出目录、以及对应 HTML 报告位置。
- 失败时，读取 `Get-Content $logPath -Tail 200`，优先抓取最后一个 traceback、异常类型、缺失数据/配置项和触发阶段，再给出修复建议。
- 如果需要保留日志供后续排查，直接把临时日志路径记在回复里，不要把整份日志贴进对话。

## 日期与 Profile

- 单次回测的因子类直接由 `individual_config.weights` 推导；profile 只用于标识策略模板。
- 如果要让 GA 搜索这个因子，再更新 `configs/strategy.yaml` 中目标 profile 的 `factor_classes`。

## 注意事项

- `individual_config.weights` 里的因子名必须能被 `core/factors/registry.py` 在 `factor_db/factors` 中发现，否则脚本会退出。
- 不传日期时，当前默认区间是 `2020-06-30` 到 `2024-12-31`。任务只要提到“某段时间”，就显式传日期，避免误跑全区间。
- single 模式会生成换仓、持仓、收益和指标报告；成功时先看 HTML，再决定是否需要打开日志。
- 排查失败时只读取最新 200 行日志，除非这 200 行完全不包含异常上下文；默认不要展开更长输出，避免浪费上下文。
