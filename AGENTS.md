# Repository Guidelines

## Project Structure & Module Organization
- `core/` contains shared market-data access, factor implementations, ranking logic, and sizing utilities.
- `trading/` holds live-trading entrypoints such as `main.py`, `watchdog.py`, and Lark notification handlers.
- `testback/` contains backtesting and GA optimization code.
- `utils/` stores reusable helpers plus small test modules like `utils/stock/test_time.py`.
- `configs/` holds runtime configuration. Copy `configs/env.template.py` to `configs/env.py`; treat `best_individual_config.json` as generated output.
- `reports/` and `core/factors/helpers/.cache/` contain generated artifacts and should not be treated as source.

## Build, Test, and Development Commands
- `uv sync --locked`: create or update the local `.venv` from `pyproject.toml` and `uv.lock`.
- `$env:PYTHONPATH = (Get-Location).Path`: set the repo root before running scripts directly.
- `python testback/ga.py`: run backtests and GA parameter search.
- `python trading/main.py --individual-config configs/best_individual_config.json`: start trading logic manually.
- `.\run.ps1`: production helper that fetches `origin/main`, hard-resets to it, installs deps, and starts `trading/watchdog.py`. Do not use it for normal feature development.

## Coding Style & Naming Conventions
- Target Python 3.12 on Windows. Follow the existing 2-space indentation style.
- Use `snake_case` for functions, modules, and variables; use `PascalCase` for classes such as `TopN` and factor classes.
- Reuse module loggers (`core_logger`, `trading_logger`, `testback_logger`) instead of `print`.
- New factors belong under `core/factors/`; when adding one, export it in `core/factors/__init__.py` and register defaults in `core/strategies/`.

## Testing Guidelines
- Tests use `unittest` and live near the code as `test_*.py`.
- Run targeted tests with commands such as `python -m unittest utils.stock.test_time` and `python -m unittest core.database.test_stock_list_new`.
- No coverage gate is configured. Add focused tests when changing date handling, factor scoring, cache behavior, or order-sizing logic.

## Commit & Pull Request Guidelines
- Recent commits use short, specific, imperative subjects, often in Chinese, for example `优化WMACross` or `修正有效股票列表不包含st`.
- Keep the first line concise and scoped to one change.
- PRs should explain trading or data impact, list local test commands, link the related task, and include screenshots or report paths for visual or benchmark output changes.

## Security & Configuration Tips
- Never commit `configs/env.py`, account credentials, QMT login artifacts, or temporary logs.
- Treat `reports/`, `.cache/`, `tmp_*.log`, and generated JSON/HTML/PKL files as disposable unless the change explicitly requires them.
