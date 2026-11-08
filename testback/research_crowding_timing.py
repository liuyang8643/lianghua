"""Reproducible, causal screening of market amount crowding overlays.

This is a research screen, not an execution-engine backtest.  It scales the
stored daily returns of an already completed baseline backtest.  A candidate
must still pass a full engine backtest before it can enter a live config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PERIOD_RECORDS = {
    "train": "current_baseline_train/record.json",
    "validation": "current_baseline_validation/record.json",
    "test": "current_baseline_test/record.json",
}


def compute_amount_crowding(amount: np.ndarray, block_size: int = 64) -> np.ndarray:
    """Return the share of amount traded by the largest 5% of valid stocks.

    The universe is every SH/SZ stock column in the runtime.  A stock is valid
    on a row when its amount is finite and strictly positive.  ST stocks are
    retained because the requested definition is whole-market turnover.  The
    number selected is ``ceil(valid_count * 0.05)``.
    """
    result = np.full(amount.shape[0], np.nan, dtype=np.float64)
    for start in range(0, amount.shape[0], block_size):
        stop = min(start + block_size, amount.shape[0])
        values = np.asarray(amount[start:stop], dtype=np.float64)
        valid = np.isfinite(values) & (values > 0.0)
        values = np.where(valid, values, 0.0)
        counts = valid.sum(axis=1)
        ordered = np.sort(values, axis=1)[:, ::-1]
        cumulative = np.cumsum(ordered, axis=1)
        top_counts = np.maximum(1, np.ceil(counts * 0.05).astype(np.intp))
        top_amount = cumulative[np.arange(stop - start), top_counts - 1]
        total_amount = values.sum(axis=1)
        result[start:stop] = np.divide(
            top_amount,
            total_amount,
            out=np.full(stop - start, np.nan),
            where=total_amount > 0.0,
        )
    return result


def _rolling_mean_std(values: np.ndarray, window: int, minimum: int):
    mean = np.full(values.shape, np.nan)
    std = np.full(values.shape, np.nan)
    for row in range(len(values)):
        start = max(0, row - window)
        history = values[start:row]
        history = history[np.isfinite(history)]
        if len(history) >= minimum:
            mean[row] = history.mean()
            std[row] = history.std(ddof=1)
    return mean, std


def causal_features(crowding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Features at open T use crowding only through completed row T-1."""
    level = np.full(crowding.shape, np.nan)
    level[1:] = crowding[:-1]
    delta5 = np.full(crowding.shape, np.nan)
    delta5[6:] = crowding[5:-1] - crowding[:-6]
    level_mean, level_std = _rolling_mean_std(level, 252, 126)
    delta_mean, delta_std = _rolling_mean_std(delta5, 252, 126)
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((level - level_mean) / level_std,
                (delta5 - delta_mean) / delta_std)


def metrics(returns: np.ndarray) -> dict:
    values = np.nan_to_num(np.asarray(returns, dtype=np.float64))
    nav = np.cumprod(1.0 + values)
    annualized = nav[-1] ** (252.0 / len(values)) - 1.0
    peak = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    max_drawdown = np.min(nav / peak - 1.0)
    volatility = values.std(ddof=1)
    sharpe = values.mean() / volatility * np.sqrt(252.0)
    return {
        "annualized_pct": float(annualized * 100.0),
        "max_drawdown_pct": float(max_drawdown * 100.0),
        "sharpe": float(sharpe),
        "calmar": float(annualized / abs(max_drawdown)),
    }


def overlay(level_z: np.ndarray, delta_z: np.ndarray,
            family: str, threshold: float) -> np.ndarray:
    """Return exposure relative to the baseline's fixed 75% exposure."""
    if family == "level_reverse":
        absolute = np.where(level_z > threshold, 0.50, 0.75)
    elif family == "derivative":
        absolute = np.where(
            delta_z > threshold, 0.90,
            np.where(delta_z < -threshold, 0.50, 0.75),
        )
    elif family == "lifecycle":
        absolute = np.where(
            (delta_z > threshold) & (level_z < 1.0), 0.90,
            np.where((level_z > 1.0) & (delta_z < 0.0), 0.375, 0.75),
        )
    else:
        raise ValueError(f"unknown family: {family}")
    relative = absolute / 0.75
    relative[~np.isfinite(level_z) | ~np.isfinite(delta_z)] = 1.0
    return relative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--result-dir", default="results/crowding_research")
    args = parser.parse_args()
    result_dir = Path(args.result_dir)

    with np.load(args.runtime, mmap_mode="r") as runtime:
        dates = runtime["trade_dates"].astype("datetime64[D]")
        codes = runtime["stock_codes"].astype(str)
        stock_mask = np.array([
            code.endswith((".SH", ".SZ")) for code in codes
        ])
        start = int(np.searchsorted(dates, np.datetime64("2008-01-01")))
        research_dates = dates[start:]
        crowding = compute_amount_crowding(
            runtime["amount"][start:, stock_mask],
        )

    level_z, delta_z = causal_features(crowding)
    date_to_row = {str(value): row for row, value in enumerate(research_dates)}
    candidates = [("baseline", None)] + [
        (family, threshold)
        for family in ("level_reverse", "derivative", "lifecycle")
        for threshold in (0.0, 0.5, 1.0)
    ]
    summary = {
        "contract": {
            "universe": "all runtime SH/SZ stocks",
            "valid_stock": "finite amount > 0; ST retained",
            "top_fraction": 0.05,
            "top_count_rounding": "ceil",
            "signal_lag": "trade row T uses crowding only through T-1",
            "derivative": "crowding[T-1] - crowding[T-6]",
            "normalization": "prior 252 rows only; minimum 126",
            "warning": "daily-return exposure screen, not execution-engine backtest",
        },
        "periods": {},
    }

    for period, relative_record in PERIOD_RECORDS.items():
        record = json.loads(
            (result_dir / relative_record).read_text(encoding="utf-8")
        )
        rows = np.array([date_to_row[value] for value in record["dates"]])
        returns = np.asarray(record["daily_returns"], dtype=np.float64) / 100.0
        results = {}
        for family, threshold in candidates:
            if family == "baseline":
                adjusted = returns
                label = "baseline"
            else:
                adjusted = returns * overlay(
                    level_z[rows], delta_z[rows], family, float(threshold),
                )
                label = f"{family}_{threshold:g}"
            results[label] = metrics(adjusted)
        summary["periods"][period] = results

    train_results = summary["periods"]["train"]
    selected = max(train_results, key=lambda name: train_results[name]["calmar"])
    summary["selection"] = {
        "metric": "train calmar",
        "selected": selected,
        "promote_crowding": selected != "baseline",
    }

    np.savez_compressed(
        result_dir / "crowding_series.npz",
        dates=research_dates,
        crowding=crowding,
    )
    (result_dir / "crowding_screening_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["selection"], ensure_ascii=False))


if __name__ == "__main__":
    main()
