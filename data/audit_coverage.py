"""Audit daily K-line and configured-factor coverage in the runtime panel."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np

from core.backtest import _compute_factor_scores
from core.runtime import latest_runtime_npz_path, load_runtime_npz, load_runtime_stock_codes
from core.strategy_config import load_strategy_config
from utils.stock.time import get_trading_date_span


ROOT = Path(__file__).resolve().parents[1]


def _parse_date(value: str) -> date:
    return datetime.strptime(value.replace("-", ""), "%Y%m%d").date()


def _internal_gap_counts(open_panel: np.ndarray) -> tuple[np.ndarray, list[int]]:
    available = np.isfinite(open_panel)
    has_bar = available.any(axis=0)
    first = np.where(has_bar, available.argmax(axis=0), open_panel.shape[0])
    last = np.where(has_bar, open_panel.shape[0] - 1 - available[::-1].argmax(axis=0), -1)
    row_numbers = np.arange(open_panel.shape[0])[:, None]
    internal = (~available) & (row_numbers > first) & (row_numbers < last)
    return internal.sum(axis=1).astype(int), np.flatnonzero(internal.any(axis=0)).tolist()


def _max_true_run(values: np.ndarray) -> int:
    max_run = run = 0
    for value in values:
        run = run + 1 if value else 0
        max_run = max(max_run, run)
    return max_run


def audit(config_path: str, start: date, end: date, output_dir: Path) -> Path:
    strategy = load_strategy_config(config_path)
    factor_classes = strategy["factor_classes"]
    filter_classes = strategy["filter_factor_classes"]
    weights = strategy["individual_config"]["weights"]
    requested_dates = [datetime.combine(day, datetime.min.time()) for day in get_trading_date_span(start, end)]
    max_lookback = max(cls.hist_days for cls in factor_classes + filter_classes)
    data = load_runtime_npz(requested_dates, max_lookback=max_lookback)
    if data is None:
        raise FileNotFoundError(f"runtime does not cover {start} to {end}")

    missing: dict[str, list[int]] = {}
    result = _compute_factor_scores(
        requested_dates,
        load_runtime_stock_codes(),
        weights=weights,
        factor_classes=factor_classes,
        data=data,
        filter_factor_classes=filter_classes,
        factor_missing_counts=missing,
    )
    if result is None:
        raise RuntimeError("no valid runtime dates")
    _, _, _, valid_dates, date_indices, valid_stocks, stock_indices = result
    cols = np.array([stock_indices[code] for code in valid_stocks], dtype=np.intp)
    open_panel = data["open"][np.ix_(date_indices, cols)]
    kline_missing = np.isnan(open_panel).sum(axis=1).astype(int)
    internal_missing, internal_stock_cols = _internal_gap_counts(open_panel)

    for factor_class in filter_classes:
        raw = factor_class().calc_batch(data)
        values = np.isnan(raw[np.ix_(date_indices, cols)]).sum(axis=1).astype(int)
        missing[factor_class.__name__] = values.tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, dt in enumerate(valid_dates):
        row = {
            "date": dt.date().isoformat(),
            "kline_missing": int(kline_missing[i]),
            "kline_internal_gap": int(internal_missing[i]),
        }
        for name, values in missing.items():
            row[name] = int(values[i])
            row[f"{name}_extra_vs_kline"] = int(values[i] - kline_missing[i])
        rows.append(row)

    csv_path = output_dir / "daily_coverage.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    available = np.isfinite(open_panel)
    gap_rows = []
    for col in internal_stock_cols:
        first = int(available[:, col].argmax())
        last = int(len(available) - 1 - available[::-1, col].argmax())
        internal = (~available[:, col]) & (np.arange(len(available)) > first) & (np.arange(len(available)) < last)
        gap_rows.append({
            "stock_code": valid_stocks[col],
            "internal_missing_days": int(internal.sum()),
            "max_consecutive_internal_gap_days": _max_true_run(internal),
            "first_kline_date": valid_dates[first].date().isoformat(),
            "last_kline_date": valid_dates[last].date().isoformat(),
        })
    gap_rows.sort(key=lambda row: (row["max_consecutive_internal_gap_days"], row["internal_missing_days"]), reverse=True)
    gap_csv_path = output_dir / "internal_kline_gaps_by_stock.csv"
    with gap_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(gap_rows[0]))
        writer.writeheader()
        writer.writerows(gap_rows)

    summary = {
        "period": {"start": rows[0]["date"], "end": rows[-1]["date"], "days": len(rows)},
        "stocks": len(valid_stocks),
        "kline": {
            "total_missing_cells": int(kline_missing.sum()),
            "total_internal_gap_cells": int(internal_missing.sum()),
            "stocks_with_internal_gaps": len(internal_stock_cols),
            "internal_gap_stock_csv": str(gap_csv_path),
        },
        "factors": {
            name: {
                "total_missing_cells": int(sum(values)),
                "extra_vs_kline_cells": int(sum(values) - kline_missing.sum()),
                "max_daily_extra_vs_kline": int(max(np.asarray(values) - kline_missing)),
            }
            for name, values in missing.items()
        },
        "daily_csv": str(csv_path),
    }
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default="results/coverage_audit")
    args = parser.parse_args()
    if args.end is None:
        with np.load(latest_runtime_npz_path(), allow_pickle=False) as runtime:
            args.end = str(runtime["trade_dates"][-1])
    print(audit(args.config, _parse_date(args.start), _parse_date(args.end), Path(args.output_dir)))


if __name__ == "__main__":
    main()
