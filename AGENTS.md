# WBR Agent Guide

## Overview

WBR is a Windows-only Python 3.12 quantitative trading project built around QMT (`xtquant`). The repository currently covers:

- Live trading entrypoints under `trading/`
- Data access, factor implementations, and benchmark helpers under `core/`
- GA optimization and result analysis under `testback/`
- Shared utilities under `utils/`
- Runtime configuration under `configs/`
- Generated reports under `reports/`

Read this file before making changes. Tool-specific notes should defer to this file.

## Agent Files

- Primary repository instructions live in this root `AGENTS.md`.
- `CLAUDE.md` should be a symlink to `AGENTS.md`.
- Shared project skills live under `.agents/skills/`.
- `.claude/skills` should be a symlink to `.agents/skills`.
- `.github/copilot-instructions.md` is a pointer note and should not diverge from this file.

## Environment

- OS: Windows only
- Python: 3.12+
- Package manager: `uv`
- External dependency: QMT client must be installed locally

Install dependencies with:

```powershell
uv sync --locked
```

Create local runtime config by copying `configs/env.template.py` to `configs/env.py` and filling in account, Feishu, and QMT paths.

## Common Commands

Run live trading:

```powershell
python -m trading.main --individual-config configs/best_individual_config.json
```

Run one-shot manual trading with interactive confirmation:

```powershell
python -m trading.main --individual-config configs/best_individual_config.json --buy
python -m trading.main --individual-config configs/best_individual_config.json --sell
python -m trading.main --individual-config configs/best_individual_config.json --buy --sell
```

Run GA:

```powershell
python testback/ga.py
python testback/ga.py --individual-config configs/best_individual_config.json --period-span 30
```

Analyze GA output:

```powershell
python testback/ga_analyzer.py results/ga_<timestamp>
```

Run factor benchmark with the current default `WMACross` setup:

```powershell
python -m core.factors.benchmark.benchmark
```

Override benchmark date range, sample size, or HTML generation through environment variables:

```powershell
$env:BENCHMARK_START_DATE = '2024-01-01'
$env:BENCHMARK_END_DATE = '2024-12-31'
$env:BENCHMARK_SAMPLE_STEP = '5'
$env:BENCHMARK_SAMPLE_SIZE = '1000'
$env:BENCHMARK_GENERATE_HTML = '1'
python -m core.factors.benchmark.benchmark
```

Launch the factor visualization tool:

```powershell
python -m core.factors.benchmark.web_chart
python -m core.factors.benchmark.web_chart --port 9090
python -m core.factors.benchmark.web_chart --code 600000.SH
```

## Architecture Notes

- `core/database/` contains market, financial, and money-flow data access layers.
- `core/factors/` contains factor implementations plus benchmark helpers. The current benchmark focus is `WMACross`, a short-horizon 2/28 WMA mean-reversion factor with a retail money-flow amplifier.
- `core/factors/benchmark/benchmark.py` currently defaults to `WMACross`, supports benchmark date/sample overrides via environment variables, always saves a pickle report, and only generates HTML when `BENCHMARK_GENERATE_HTML` is enabled.
- `core/factors/benchmark/web_chart.py` provides an ECharts-based browser view for K-line, moving averages, volume, and factor scores. It currently visualizes `WMACross` only and filters chart data to `2021-01-01` through `2022-12-31`.
- `core/strategies/` contains position-sizing and strategy selection logic.
- `trading/main.py` is the explicit live-trading CLI entrypoint and currently requires `--individual-config`. It also supports one-shot manual `--buy` / `--sell` execution with terminal confirmation.
- `trading/watchdog.py` does not currently match the `trading.main` CLI contract and should be treated carefully.
- `testback/` contains GA search and analysis scripts rather than a separate packaged service.
- `testback/reportor/` contains the single-run report implementation and frontend templates. Import report generation from `testback.reportor`.

## Dependency Notes

Key runtime dependencies declared in `pyproject.toml` include:

- Trading and data: `xtquant`, `akshare`, `pandas`, `numpy`, `scipy`, `ta-lib`
- Optimization and parallelism: `deap`, `joblib`, `loky`, `filelock`
- Reporting and integration: `plotly`, `jinja2`, `boto3`, `lark-oapi`, `pyyaml`, `loguru`, `psutil`, `pyarrow`

Additional runtime behavior to remember:

- `WMACross` now depends on retail money-flow data from `core/database/money_flow/`.
- `core.factors.benchmark.web_chart` loads `echarts@5` from a CDN in the browser.

When these dependencies change, keep this section aligned with `pyproject.toml` and the runtime workflow.

## Maintenance Rule

If a change affects project architecture, directory ownership, entrypoints, workflows, or dependency relationships, update this `AGENTS.md` in the same change.

Examples that require an `AGENTS.md` update:

- Adding, removing, or repurposing top-level modules or major subpackages
- Changing the main way live trading, GA, benchmark, or visualization flows are started
- Introducing, removing, or materially reclassifying runtime dependencies
- Changing configuration locations or required local setup steps
- Changing where shared agent instructions or skills live

The goal is to keep future AI agents aligned with the current repository shape without re-discovering the whole project from scratch.

## Current Risks

- `run.ps1` performs `git fetch origin main` followed by `git reset --hard origin/main`, which discards local uncommitted work.
- The checked-in `configs/best_individual_config.json` reflects an older factor set and may not match the current `TopN` and GA assumptions.
- `core/factors/benchmark/web_chart.py` advertises a wider date span in its module docstring, but the implementation currently filters to `2021-01-01` through `2022-12-31`.
- The repository currently has no configured automated test or lint pipeline.
