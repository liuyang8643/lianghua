"""Leakage-controlled A/H premium mean-reversion research.

This module deliberately does not use :mod:`core.backtest`.  That engine has
one CNY cash account and A-share trading rules; treating an H share as another
A-share column would silently apply the wrong calendar, lot, fee, currency,
and shorting rules.  The executable result here is therefore a conservative
``long_only_switch`` event backtest.  A dollar-neutral pair return is reported
only as ``theoretical_only`` until point-in-time borrow/short eligibility is
available.

For a completed common close T, the signal is the current adjusted log A/H
premium relative to the *previous* N common observations.  It executes at the
next row with valid A open, H open, and official Stock Connect settlement FX.
No field from that execution row other than the two opens and FX is used to
choose the side.  Train/validation/test are contiguous and derived solely from
the common data availability; a candidate is selected from training results
before validation and test are evaluated.

The current Eastmoney A/H list is a current-live cohort, not a historical PIT
constituent file.  Results from that cohort are explicitly labelled
``conditional_survivor_cohort`` and are not an unbiased historical-universe
portfolio estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.update_ah_history import load_ah_history


DEFAULT_DATA_DIR = ROOT / "data" / "ah_history"
DEFAULT_OUTPUT = ROOT / "results" / "stat_arb_ah" / "summary.json"


@dataclass(frozen=True)
class AHConfig:
    """Frozen signal and holding-period parameters."""

    lookback: int
    entry_z: float
    horizon: int

    def __post_init__(self) -> None:
        if self.lookback < 5:
            raise ValueError("lookback must be at least 5")
        if not math.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError("entry_z must be finite and positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")


@dataclass(frozen=True)
class CostModel:
    """Round-trip proportional cost assumptions in basis points."""

    a_round_trip_bps: float = 25.0
    h_round_trip_bps: float = 45.0
    fx_round_trip_bps: float = 20.0

    def __post_init__(self) -> None:
        values = (
            self.a_round_trip_bps,
            self.h_round_trip_bps,
            self.fx_round_trip_bps,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("costs must be finite and non-negative")


@dataclass(frozen=True)
class Period:
    start: pd.Timestamp
    end: pd.Timestamp

    def contains(self, values: pd.Series) -> pd.Series:
        return (values >= self.start) & (values <= self.end)


def _date_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.tz_localize(None).dt.normalize()


def causal_adjusted_a_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Build split/rights-continuous A-share open and close indices.

    The raw execution prices are linked only with that day's official
    ``preClose``.  For row d after the first valid row::

        adj_close[d] = adj_close[d-1] * close[d] / preClose[d]
        adj_open[d]  = adj_close[d-1] * open[d]  / preClose[d]

    This does not use a future back-adjustment factor.  An invalid row breaks
    the chain rather than being filled from the future.
    """

    required = {"date", "open", "close", "preClose"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"A history missing columns: {sorted(missing)}")
    values = frame.loc[:, ["date", "open", "close", "preClose"]].copy()
    values["date"] = _date_series(values["date"])
    values = values.dropna(subset=["date"]).sort_values("date", kind="stable")
    values = values.drop_duplicates("date", keep="last").reset_index(drop=True)
    for column in ("open", "close", "preClose"):
        values[column] = pd.to_numeric(values[column], errors="coerce")

    raw_open = values["open"].to_numpy(dtype=np.float64)
    raw_close = values["close"].to_numpy(dtype=np.float64)
    pre_close = values["preClose"].to_numpy(dtype=np.float64)
    adj_open = np.full(len(values), np.nan, dtype=np.float64)
    adj_close = np.full(len(values), np.nan, dtype=np.float64)
    segment = np.full(len(values), -1, dtype=np.int64)
    previous_close_index = np.nan
    segment_id = -1
    for row in range(len(values)):
        valid = (
            np.isfinite(raw_open[row])
            and np.isfinite(raw_close[row])
            and np.isfinite(pre_close[row])
            and raw_open[row] > 0
            and raw_close[row] > 0
            and pre_close[row] > 0
        )
        if not valid:
            previous_close_index = np.nan
            continue
        if not np.isfinite(previous_close_index):
            segment_id += 1
            adj_open[row] = raw_open[row]
            adj_close[row] = raw_close[row]
        else:
            adj_open[row] = previous_close_index * raw_open[row] / pre_close[row]
            adj_close[row] = previous_close_index * raw_close[row] / pre_close[row]
        previous_close_index = adj_close[row]
        segment[row] = segment_id
    values["a_adj_open"] = adj_open
    values["a_adj_close"] = adj_close
    values["a_segment_id"] = segment
    return values


def causal_adjusted_h_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Build an H-share index from raw prices and updater-published preClose.

    ``h_pre_close`` is reconstructed by the explicit data updater from a
    same-day raw/qfq four-price affine corporate-action map.  Research never
    takes logs or percentage changes of Tencent qfq/hfq levels.  An
    underdetermined affine row is skipped; a later identifiable row can link to
    the last identifiable completed close without any future fill.
    """

    required = {
        "date",
        "raw_open",
        "raw_close",
        "qfq_affine_valid",
        "h_pre_close",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"H history missing columns: {sorted(missing)}")
    values = frame.loc[:, list(required)].copy()
    values["date"] = _date_series(values["date"])
    values = values.dropna(subset=["date"]).sort_values("date", kind="stable")
    values = values.drop_duplicates("date", keep="last").reset_index(drop=True)
    for column in ("raw_open", "raw_close", "h_pre_close"):
        values[column] = pd.to_numeric(values[column], errors="coerce")
    affine_valid = values["qfq_affine_valid"].fillna(False).astype(bool).to_numpy()
    raw_open = values["raw_open"].to_numpy(dtype=np.float64)
    raw_close = values["raw_close"].to_numpy(dtype=np.float64)
    pre_close = values["h_pre_close"].to_numpy(dtype=np.float64)
    adj_open = np.full(len(values), np.nan, dtype=np.float64)
    adj_close = np.full(len(values), np.nan, dtype=np.float64)
    segment = np.full(len(values), -1, dtype=np.int64)
    previous_close_index = np.nan
    segment_id = -1
    for row in range(len(values)):
        raw_valid = (
            affine_valid[row]
            and np.isfinite(raw_open[row])
            and np.isfinite(raw_close[row])
            and raw_open[row] > 0.0
            and raw_close[row] > 0.0
        )
        if not raw_valid:
            continue
        if not np.isfinite(previous_close_index):
            segment_id += 1
            adj_open[row] = raw_open[row]
            adj_close[row] = raw_close[row]
        elif np.isfinite(pre_close[row]) and pre_close[row] > 0.0:
            adj_open[row] = previous_close_index * raw_open[row] / pre_close[row]
            adj_close[row] = previous_close_index * raw_close[row] / pre_close[row]
        else:
            segment_id += 1
            adj_open[row] = raw_open[row]
            adj_close[row] = raw_close[row]
        if not (
            np.isfinite(adj_open[row])
            and np.isfinite(adj_close[row])
            and adj_open[row] > 0.0
            and adj_close[row] > 0.0
        ):
            previous_close_index = np.nan
            continue
        previous_close_index = adj_close[row]
        segment[row] = segment_id
    values["h_adj_open"] = adj_open
    values["h_adj_close"] = adj_close
    values["h_segment_id"] = segment
    return values


def prior_rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Z-score each current value against exactly N *prior* observations."""

    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if window < 2:
        raise ValueError("window must be at least 2")
    result = np.full(series.shape, np.nan, dtype=np.float64)
    if len(series) <= window:
        return result
    finite = np.isfinite(series)
    safe = np.where(finite, series, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(safe, dtype=np.float64)))
    csq = np.concatenate(([0.0], np.cumsum(safe * safe, dtype=np.float64)))
    count = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    rows = np.arange(window, len(series), dtype=np.intp)
    starts = rows - window
    hist_count = count[rows] - count[starts]
    hist_sum = csum[rows] - csum[starts]
    hist_sq = csq[rows] - csq[starts]
    mean = hist_sum / window
    variance = (hist_sq - hist_sum * hist_sum / window) / (window - 1)
    valid = (
        (hist_count == window)
        & finite[rows]
        & np.isfinite(variance)
        & (variance > 0.0)
    )
    output = np.full(rows.shape, np.nan, dtype=np.float64)
    output[valid] = (series[rows[valid]] - mean[valid]) / np.sqrt(variance[valid])
    result[rows] = output
    return result


def build_pair_panel(
    a_history: pd.DataFrame,
    h_history: pd.DataFrame,
    fx_history: pd.DataFrame,
    *,
    share_ratio: float = 1.0,
) -> pd.DataFrame:
    """Return the strict A/H/FX common calendar for one economic pair.

    ``share_ratio`` is the number of H-share economic units represented by one
    A-share economic unit in the denominator ``H * FX * share_ratio``.  It is
    one for standard ordinary A and H shares.  The updater records this as an
    assumption; an unknown or non-positive ratio is rejected, never inferred
    from Eastmoney's displayed price ratio.
    """

    if not math.isfinite(share_ratio) or share_ratio <= 0:
        raise ValueError("share_ratio must be finite and positive")
    a = causal_adjusted_a_prices(a_history)
    h = causal_adjusted_h_prices(h_history)
    required_fx = {"date", "mid_rate"}
    missing_fx = required_fx.difference(fx_history.columns)
    if missing_fx:
        raise ValueError(f"FX history missing columns: {sorted(missing_fx)}")
    fx = fx_history.loc[:, ["date", "mid_rate"]].copy()
    fx["date"] = _date_series(fx["date"])
    fx = fx.dropna(subset=["date"]).drop_duplicates("date", keep="last")

    panel = a.merge(h, on="date", how="inner", validate="one_to_one")
    panel = panel.merge(fx, on="date", how="inner", validate="one_to_one")
    numeric = (
        "a_adj_open",
        "a_adj_close",
        "h_adj_open",
        "h_adj_close",
        "mid_rate",
    )
    for column in numeric:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    valid = np.logical_and.reduce(
        [np.isfinite(panel[column]) & (panel[column] > 0.0) for column in numeric]
    )
    panel = panel.loc[valid].sort_values("date", kind="stable").reset_index(drop=True)
    boundaries = (
        panel["a_segment_id"].ne(panel["a_segment_id"].shift())
        | panel["h_segment_id"].ne(panel["h_segment_id"].shift())
    )
    panel["segment_id"] = boundaries.cumsum().astype(np.int64) - 1
    panel["spread_close"] = np.log(panel["a_adj_close"]) - np.log(
        panel["h_adj_close"] * panel["mid_rate"] * share_ratio
    )
    panel["spread_open"] = np.log(panel["a_adj_open"]) - np.log(
        panel["h_adj_open"] * panel["mid_rate"] * share_ratio
    )
    return panel


def _segment_prior_zscore(
    values: np.ndarray,
    segments: np.ndarray,
    window: int,
) -> np.ndarray:
    """Apply the prior-only z-score independently inside contiguous segments."""

    output = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) == 0:
        return output
    starts = np.r_[0, np.flatnonzero(segments[1:] != segments[:-1]) + 1]
    stops = np.r_[starts[1:], len(values)]
    for start, stop in zip(starts, stops, strict=True):
        output[start:stop] = prior_rolling_zscore(values[start:stop], window)
    return output


def generate_pair_events(
    panel: pd.DataFrame,
    config: AHConfig,
    costs: CostModel,
    *,
    a_code: str,
    h_code: str,
    non_overlapping: bool = True,
) -> pd.DataFrame:
    """Generate next-common-open events for one pair."""

    required = {
        "date",
        "a_adj_open",
        "h_adj_open",
        "mid_rate",
        "spread_close",
        "spread_open",
        "segment_id",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"pair panel missing columns: {sorted(missing)}")
    segments = panel["segment_id"].to_numpy(dtype=np.int64)
    zscore = _segment_prior_zscore(
        panel["spread_close"].to_numpy(dtype=np.float64),
        segments,
        config.lookback,
    )
    dates = _date_series(panel["date"])
    a_open = panel["a_adj_open"].to_numpy(dtype=np.float64)
    h_open_cny = (
        panel["h_adj_open"].to_numpy(dtype=np.float64)
        * panel["mid_rate"].to_numpy(dtype=np.float64)
    )
    spread_close = panel["spread_close"].to_numpy(dtype=np.float64)
    spread_open = panel["spread_open"].to_numpy(dtype=np.float64)
    rows: list[dict] = []
    next_allowed_signal = 0
    for signal_row in range(config.lookback, len(panel)):
        z = zscore[signal_row]
        if not np.isfinite(z) or abs(z) < config.entry_z:
            continue
        if non_overlapping and signal_row < next_allowed_signal:
            continue
        entry_row = signal_row + 1
        exit_row = entry_row + config.horizon
        if exit_row >= len(panel):
            break
        if segments[signal_row] != segments[exit_row]:
            continue
        a_return = a_open[exit_row] / a_open[entry_row] - 1.0
        h_return = h_open_cny[exit_row] / h_open_cny[entry_row] - 1.0
        if not np.isfinite(a_return) or not np.isfinite(h_return):
            continue
        cheap_leg = "H" if z > 0.0 else "A"
        if cheap_leg == "A":
            long_only_gross = a_return
            long_only_cost = costs.a_round_trip_bps / 10_000.0
            pair_gross = a_return - h_return
        else:
            long_only_gross = h_return
            long_only_cost = (
                costs.h_round_trip_bps + costs.fx_round_trip_bps
            ) / 10_000.0
            pair_gross = h_return - a_return
        pair_cost = (
            costs.a_round_trip_bps
            + costs.h_round_trip_bps
            + costs.fx_round_trip_bps
        ) / 10_000.0
        rows.append(
            {
                "a_code": a_code,
                "h_code": h_code,
                "signal_date": dates.iloc[signal_row],
                "entry_date": dates.iloc[entry_row],
                "exit_date": dates.iloc[exit_row],
                "zscore": float(z),
                "cheap_leg": cheap_leg,
                "a_gross_return": float(a_return),
                "h_cny_gross_return": float(h_return),
                "long_only_net_return": float(long_only_gross - long_only_cost),
                "theoretical_pair_net_return": float(pair_gross - pair_cost),
                "signed_spread_reversion": float(
                    -np.sign(z) * (spread_open[exit_row] - spread_close[signal_row])
                ),
            }
        )
        if non_overlapping:
            next_allowed_signal = exit_row
    return pd.DataFrame(rows)


def require_executable_mode(
    mode: str,
    *,
    has_pit_borrow_data: bool = False,
) -> None:
    """Reject an executable short claim without PIT borrow evidence."""

    if mode == "long_only_switch":
        return
    if mode == "long_short_pair" and has_pit_borrow_data:
        return
    if mode == "long_short_pair":
        raise ValueError(
            "long_short_pair is theoretical_only without point-in-time borrow "
            "availability, short eligibility, recall, and borrow-fee data"
        )
    raise ValueError(f"unknown mode: {mode}")


def availability_dates(
    panels: Mapping[tuple[str, str], pd.DataFrame],
    *,
    warmup: int,
    horizon: int,
    min_pairs: int = 5,
) -> np.ndarray:
    """Dates with enough pairs after parameter-independent warmup/exit room."""

    counts: dict[np.datetime64, int] = {}
    for panel in panels.values():
        for _, segment in panel.groupby("segment_id", sort=False):
            dates = _date_series(segment["date"]).to_numpy(dtype="datetime64[D]")
            if len(dates) <= warmup + horizon:
                continue
            for value in dates[warmup : len(dates) - horizon]:
                counts[value] = counts.get(value, 0) + 1
    selected = sorted(value for value, count in counts.items() if count >= min_pairs)
    return np.asarray(selected, dtype="datetime64[D]")


def derive_contiguous_periods(dates: Iterable[object]) -> dict[str, Period]:
    """Derive deterministic 60/20/20 chronological splits from availability."""

    values = np.unique(np.asarray(list(dates), dtype="datetime64[D]"))
    values = values[~np.isnat(values)]
    values.sort()
    if len(values) < 30:
        raise ValueError("at least 30 availability dates are required")
    train_stop = max(1, min(len(values) - 2, int(math.floor(len(values) * 0.60))))
    validation_stop = max(
        train_stop + 1,
        min(len(values) - 1, int(math.floor(len(values) * 0.80))),
    )
    ts = pd.to_datetime(values)
    return {
        "train": Period(ts[0], ts[train_stop - 1]),
        "validation": Period(ts[train_stop], ts[validation_stop - 1]),
        "test": Period(ts[validation_stop], ts[-1]),
    }


def _sample_t_stat(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return float("nan")
    std = float(np.std(finite, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(finite) / std * math.sqrt(len(finite)))


def _signal_month_returns(events: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight concurrent events within each signal month.

    This removes the most direct source of pseudo-replication from treating
    many A/H pairs observed in the same market regime as independent samples.
    It is deliberately based on ``signal_date`` and must be called only after
    :func:`period_events` has sealed both signal and exit inside a period.
    """

    return_columns = (
        "long_only_net_return",
        "theoretical_pair_net_return",
        "signed_spread_reversion",
    )
    if events.empty:
        return pd.DataFrame(columns=return_columns, dtype=np.float64)
    required = {"signal_date", *return_columns}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")
    values = events.loc[:, ["signal_date", *return_columns]].copy()
    values["signal_month"] = _date_series(values.pop("signal_date")).dt.to_period(
        "M"
    )
    for column in return_columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    return (
        values.dropna(subset=["signal_month"])
        .groupby("signal_month", sort=True)[list(return_columns)]
        .mean()
    )


def event_metrics(events: pd.DataFrame) -> dict:
    """Return event statistics without pretending they are daily NAV metrics."""

    if events.empty:
        return {
            "events": 0,
            "pairs": 0,
            "signal_months": 0,
            "long_only_mean_pct": None,
            "long_only_hit_rate": None,
            "long_only_t_stat": None,
            "long_only_monthly_mean_pct": None,
            "long_only_monthly_hit_rate": None,
            "long_only_monthly_t_stat": None,
            "theoretical_pair_mean_pct": None,
            "theoretical_pair_hit_rate": None,
            "theoretical_pair_t_stat": None,
            "theoretical_pair_monthly_mean_pct": None,
            "theoretical_pair_monthly_hit_rate": None,
            "theoretical_pair_monthly_t_stat": None,
            "spread_reversion_mean_pct": None,
            "spread_reversion_hit_rate": None,
            "spread_reversion_t_stat": None,
            "spread_reversion_monthly_mean_pct": None,
            "spread_reversion_monthly_hit_rate": None,
            "spread_reversion_monthly_t_stat": None,
        }

    def stats(
        values: np.ndarray,
    ) -> tuple[float | None, float | None, float | None]:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return None, None, None
        t_stat = _sample_t_stat(values)
        return (
            float(np.mean(values) * 100.0),
            float(np.mean(values > 0.0)),
            float(t_stat) if np.isfinite(t_stat) else None,
        )

    monthly = _signal_month_returns(events)

    def event_stats(
        column: str,
    ) -> tuple[float | None, float | None, float | None]:
        return stats(events[column].to_numpy(dtype=np.float64))

    def monthly_stats(
        column: str,
    ) -> tuple[float | None, float | None, float | None]:
        return stats(monthly[column].to_numpy(dtype=np.float64))

    long_mean, long_hit, long_t = event_stats("long_only_net_return")
    pair_mean, pair_hit, pair_t = event_stats("theoretical_pair_net_return")
    rev_mean, rev_hit, rev_t = event_stats("signed_spread_reversion")
    long_month_mean, long_month_hit, long_month_t = monthly_stats(
        "long_only_net_return"
    )
    pair_month_mean, pair_month_hit, pair_month_t = monthly_stats(
        "theoretical_pair_net_return"
    )
    rev_month_mean, rev_month_hit, rev_month_t = monthly_stats(
        "signed_spread_reversion"
    )
    return {
        "events": int(len(events)),
        "pairs": int(events[["a_code", "h_code"]].drop_duplicates().shape[0]),
        "signal_months": int(len(monthly)),
        "long_only_mean_pct": long_mean,
        "long_only_hit_rate": long_hit,
        "long_only_t_stat": long_t,
        "long_only_monthly_mean_pct": long_month_mean,
        "long_only_monthly_hit_rate": long_month_hit,
        "long_only_monthly_t_stat": long_month_t,
        "theoretical_pair_mean_pct": pair_mean,
        "theoretical_pair_hit_rate": pair_hit,
        "theoretical_pair_t_stat": pair_t,
        "theoretical_pair_monthly_mean_pct": pair_month_mean,
        "theoretical_pair_monthly_hit_rate": pair_month_hit,
        "theoretical_pair_monthly_t_stat": pair_month_t,
        "spread_reversion_mean_pct": rev_mean,
        "spread_reversion_hit_rate": rev_hit,
        "spread_reversion_t_stat": rev_t,
        "spread_reversion_monthly_mean_pct": rev_month_mean,
        "spread_reversion_monthly_hit_rate": rev_month_hit,
        "spread_reversion_monthly_t_stat": rev_month_t,
    }


def period_events(events: pd.DataFrame, period: Period) -> pd.DataFrame:
    """Keep only events whose signal and exit are both sealed in a period."""

    if events.empty:
        return events.copy()
    signal = _date_series(events["signal_date"])
    exit_date = _date_series(events["exit_date"])
    mask = period.contains(signal) & period.contains(exit_date)
    return events.loc[mask].copy()


def training_fitness(events: pd.DataFrame, period: Period) -> tuple[float, list[float]]:
    """Robust train-only score over executable long-only signal-month returns."""

    sealed = period_events(events, period)
    if len(sealed) < 30:
        return float("-inf"), []
    values = _signal_month_returns(sealed)["long_only_net_return"]
    if len(values) < 12:
        return float("-inf"), []
    full = _sample_t_stat(values)
    split = len(values) // 2
    fold_values = (values.iloc[:split], values.iloc[split:])
    if any(len(value) < 6 for value in fold_values):
        return float("-inf"), []
    fold_scores = [_sample_t_stat(value.to_numpy()) for value in fold_values]
    if not np.isfinite(full) or not np.all(np.isfinite(fold_scores)):
        return float("-inf"), fold_scores
    return float(0.5 * full + 0.5 * min(fold_scores)), [float(x) for x in fold_scores]


def select_training_candidate(
    candidate_events: Mapping[AHConfig, pd.DataFrame],
    train_period: Period,
) -> tuple[AHConfig, dict]:
    """Select deterministically using no validation/test observations."""

    rankings: list[tuple[tuple, AHConfig, dict]] = []
    for config, events in candidate_events.items():
        fitness, folds = training_fitness(events, train_period)
        metrics = event_metrics(period_events(events, train_period))
        record = {"fitness": fitness, "fold_monthly_t_stats": folds, **metrics}
        key = (
            fitness,
            metrics["long_only_monthly_t_stat"]
            if metrics["long_only_monthly_t_stat"] is not None
            else float("-inf"),
            metrics["signal_months"],
            -config.lookback,
            -config.entry_z,
            -config.horizon,
        )
        rankings.append((key, config, record))
    if not rankings:
        raise ValueError("no candidates")
    _, selected, record = max(rankings, key=lambda value: value[0])
    if not np.isfinite(record["fitness"]):
        raise ValueError("no candidate has enough sealed training events")
    return selected, record


def _load_local_a_history(path: Path) -> pd.DataFrame:
    values = pd.read_parquet(path, columns=["time", "open", "close", "preClose"])
    milliseconds = pd.to_numeric(values.pop("time"), errors="coerce")
    values.insert(
        0,
        "date",
        pd.to_datetime(milliseconds, unit="ms", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
        .dt.normalize(),
    )
    return values


def load_research_panels(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    kline_dir: Path = ROOT / "data" / "k-line",
) -> tuple[dict[tuple[str, str], pd.DataFrame], dict]:
    """Load ignored updater artifacts and local A histories entirely offline."""

    try:
        pairs, hk, fx, metadata = load_ah_history(data_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"missing A/H artifacts in {data_dir}; run "
            "`python data/update_ah_history.py update` first"
        ) from exc
    cohort = metadata.get("universe", {})
    if cohort.get("point_in_time_complete") is not False:
        raise RuntimeError("AH metadata must explicitly declare PIT completeness")
    panels: dict[tuple[str, str], pd.DataFrame] = {}
    exclusions: list[dict[str, str]] = []
    for pair in pairs.itertuples(index=False):
        a_code = str(pair.a_code)
        h_code = str(pair.h_code)
        a_path = kline_dir / f"{a_code}.parquet"
        if not a_path.is_file():
            exclusions.append(
                {"a_code": a_code, "h_code": h_code, "reason": "missing_a_history"}
            )
            continue
        h_part = hk.loc[hk["h_code"].astype(str) == h_code]
        if h_part.empty:
            exclusions.append(
                {"a_code": a_code, "h_code": h_code, "reason": "missing_h_history"}
            )
            continue
        exchange = a_code.rsplit(".", maxsplit=1)[-1]
        fx_part = fx
        if "exchange" in fx.columns:
            exact = fx.loc[fx["exchange"].astype(str).str.upper() == exchange]
            if exact.empty:
                # Never value an SZ pair with the SSE settlement rate (or the
                # reverse) when an updater was explicitly published SH-only.
                exclusions.append(
                    {
                        "a_code": a_code,
                        "h_code": h_code,
                        "reason": "missing_exchange_specific_fx",
                    }
                )
                continue
            fx_part = exact
        share_ratio = float(getattr(pair, "share_ratio", 1.0))
        panel = build_pair_panel(
            _load_local_a_history(a_path),
            h_part,
            fx_part,
            share_ratio=share_ratio,
        )
        if not panel.empty:
            panel.attrs["name"] = str(getattr(pair, "name", ""))
            panels[(a_code, h_code)] = panel
        else:
            exclusions.append(
                {
                    "a_code": a_code,
                    "h_code": h_code,
                    "reason": "no_valid_causal_h_panel",
                }
            )
    metadata["research_loading"] = {
        "loaded_pairs": len(panels),
        "excluded_pairs": exclusions,
    }
    return panels, metadata


def _all_events(
    panels: Mapping[tuple[str, str], pd.DataFrame],
    config: AHConfig,
    costs: CostModel,
) -> pd.DataFrame:
    chunks = [
        generate_pair_events(
            panel,
            config,
            costs,
            a_code=pair[0],
            h_code=pair[1],
        )
        for pair, panel in panels.items()
    ]
    chunks = [chunk for chunk in chunks if not chunk.empty]
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).sort_values(
        ["signal_date", "a_code"], kind="stable"
    )


def current_pair_signals(
    panels: Mapping[tuple[str, str], pd.DataFrame],
    config: AHConfig,
) -> pd.DataFrame:
    """Threshold diagnostics at each pair's latest completed common close.

    These are raw signals, not trades: no next-common-open execution row exists
    yet, and the event backtest's historical non-overlap state is not applied.
    Each pair retains its own latest common date so stale panels stay visible.
    """

    columns = (
        "a_code",
        "h_code",
        "name",
        "signal_date",
        "zscore",
        "abs_zscore",
        "cheap_leg",
    )
    rows: list[dict] = []
    for (a_code, h_code), source in panels.items():
        if len(source) <= config.lookback:
            continue
        panel = source.loc[:, ["date", "spread_close", "segment_id"]].copy()
        panel["date"] = _date_series(panel["date"])
        panel["spread_close"] = pd.to_numeric(
            panel["spread_close"], errors="coerce"
        )
        panel = (
            panel.dropna(subset=["date"])
            .sort_values("date", kind="stable")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True)
        )
        latest_segment = panel["segment_id"].iloc[-1]
        panel = panel.loc[panel["segment_id"] == latest_segment].reset_index(
            drop=True
        )
        if len(panel) <= config.lookback:
            continue
        zscore = prior_rolling_zscore(
            panel["spread_close"].to_numpy(dtype=np.float64),
            config.lookback,
        )[-1]
        if not np.isfinite(zscore) or abs(zscore) < config.entry_z:
            continue
        rows.append(
            {
                "a_code": a_code,
                "h_code": h_code,
                "name": str(source.attrs.get("name", "")),
                "signal_date": panel["date"].iloc[-1],
                "zscore": float(zscore),
                "abs_zscore": float(abs(zscore)),
                "cheap_leg": "H" if zscore > 0.0 else "A",
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["abs_zscore", "a_code", "h_code"],
            ascending=[False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _json_safe(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_research(
    panels: Mapping[tuple[str, str], pd.DataFrame],
    metadata: dict,
    *,
    costs: CostModel = CostModel(),
    configs: Iterable[AHConfig] | None = None,
    progress: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """Run a sealed grid search and return summary plus selected events."""

    candidate_configs = list(configs or (
        AHConfig(lookback, entry_z, horizon)
        for lookback in (20, 60, 120, 250)
        for entry_z in (1.0, 1.5, 2.0)
        for horizon in (5, 10, 20)
    ))
    if not candidate_configs:
        raise ValueError("configs must not be empty")
    max_lookback = max(config.lookback for config in candidate_configs)
    max_horizon = max(config.horizon for config in candidate_configs)
    available = availability_dates(
        panels,
        warmup=max_lookback,
        horizon=max_horizon,
        min_pairs=min(5, max(1, len(panels))),
    )
    periods = derive_contiguous_periods(available)

    candidate_events: dict[AHConfig, pd.DataFrame] = {}
    training_rows = []
    started = time.perf_counter()
    for candidate_number, config in enumerate(candidate_configs, start=1):
        events = _all_events(panels, config, costs)
        candidate_events[config] = events
        fitness, folds = training_fitness(events, periods["train"])
        training_rows.append(
            {
                "config": asdict(config),
                "fitness": fitness,
                "fold_monthly_t_stats": folds,
                **event_metrics(period_events(events, periods["train"])),
            }
        )
        if progress and (
            candidate_number == 1
            or candidate_number % 6 == 0
            or candidate_number == len(candidate_configs)
        ):
            print(
                f"A/H grid {candidate_number}/{len(candidate_configs)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    selected, selected_train = select_training_candidate(
        candidate_events,
        periods["train"],
    )
    selected_events = candidate_events[selected]
    period_results = {
        name: event_metrics(period_events(selected_events, period))
        for name, period in periods.items()
    }
    current = current_pair_signals(panels, selected)
    universe = metadata.get("universe", {})
    summary = {
        "research_status": (
            "conditional_survivor_cohort"
            if universe.get("point_in_time_complete") is False
            else "point_in_time_universe"
        ),
        "promotable_to_unbiased_historical_portfolio": bool(
            universe.get("point_in_time_complete") is True
        ),
        "contract": {
            "signal": (
                "completed common close T premium z-score versus exactly N prior "
                "common observations"
            ),
            "execution": "next common valid A/H/official-FX open; no execution-row HLCV",
            "a_adjustment": "causal close/preClose chain; no future adjustment factor",
            "h_adjustment": (
                "causal raw-price index linked by updater-published h_pre_close; "
                "preClose comes from a fail-closed raw/qfq four-price affine map; "
                "qfq/hfq levels and percentage returns are never consumed"
            ),
            "fx_unit": "CNY per HKD; official Stock Connect settlement midpoint",
            "share_unit": metadata.get("share_unit_assumption"),
            "executable_mode": "long_only_switch event backtest",
            "long_short_mode": "theoretical_only; PIT borrow data absent",
            "event_overlap": "suppressed within each pair until prior exit",
            "selection": (
                "training signal-month aggregate executable long-only t-stat only; "
                "validation/test sealed diagnostics"
            ),
            "dependence_adjustment": (
                "same-signal-month events are equal-weighted before t-stat and "
                "training selection; this mitigates within-month cross-sectional "
                "dependence but is not a HAC/Newey-West inference"
            ),
            "split": "availability-derived contiguous 60/20/20",
            "known_limitations": [
                "current-live A/H cohort is not a historical PIT constituent universe",
                "ticker-only history may splice a prior issuer after H-code reuse",
                "daily A and H closes are asynchronous",
                "event returns are not an account-level NAV with capital contention",
                "historical board lots, borrow, and short eligibility are unavailable",
            ],
        },
        "data_metadata": metadata,
        "pairs_loaded": len(panels),
        "availability": {
            "start": str(pd.Timestamp(available[0]).date()),
            "end": str(pd.Timestamp(available[-1]).date()),
            "dates": len(available),
        },
        "periods": {
            name: {"start": str(period.start.date()), "end": str(period.end.date())}
            for name, period in periods.items()
        },
        "costs": asdict(costs),
        "training_grid": training_rows,
        "selected_config": asdict(selected),
        "selected_training_record": selected_train,
        "selected_period_results": period_results,
        "current_pair_signals": {
            "definition": (
                "raw threshold diagnostics at each pair's latest completed common "
                "close; not yet executable and historical non-overlap is not applied"
            ),
            "count": int(len(current)),
            "signals": current.to_dict("records"),
        },
    }
    return _json_safe(summary), selected_events


def _write_outputs(summary: dict, events: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)
    events_path = output.with_name("selected_events.parquet")
    events.to_parquet(events_path, index=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": digest,
                "status": summary["research_status"],
                "selected": summary["selected_config"],
                "period_results": summary["selected_period_results"],
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--a-cost-bps", type=float, default=25.0)
    parser.add_argument("--h-cost-bps", type=float, default=45.0)
    parser.add_argument("--fx-cost-bps", type=float, default=20.0)
    args = parser.parse_args()
    panels, metadata = load_research_panels(args.data_dir)
    summary, events = run_research(
        panels,
        metadata,
        costs=CostModel(
            a_round_trip_bps=args.a_cost_bps,
            h_round_trip_bps=args.h_cost_bps,
            fx_round_trip_bps=args.fx_cost_bps,
        ),
        progress=True,
    )
    _write_outputs(summary, events, args.output)


if __name__ == "__main__":
    main()
