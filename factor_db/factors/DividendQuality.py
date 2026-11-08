"""Completed cash-dividend factors for an independent dividend style.

The event snapshots are downloaded only by ``data/update_research_*``.  This
module is offline: it reads the immutable local parquet snapshots and makes an
implemented dividend usable on the first exchange row strictly after every
relevant source date is known.  Runtime ``total_share`` is stored in 10,000
shares, so market capitalisation is converted back to yuan before calculating
yield.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parents[2]
_DIVIDEND_SOURCES = (
    _ROOT / "results" / "strategy_opt_20260730"
    / "eastmoney_dividends_2010_2022.parquet",
    _ROOT / "results" / "strategy_opt_20260730"
    / "eastmoney_dividends_2023_2026.parquet",
)
_SOURCE_START = np.datetime64("2010-01-01", "D")
_POOL_PREFIXES = ("60", "00", "30")


def _normalized_codes(codes: np.ndarray) -> np.ndarray:
    values = np.asarray(codes).astype("U16")
    return np.char.zfill(np.char.partition(values, ".")[:, 0], 6)


@lru_cache(maxsize=1)
def _implemented_cash_dividends() -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in _DIVIDEND_SOURCES]
    frame = pd.concat(frames, ignore_index=True)
    frame["security_code"] = (
        frame["SECURITY_CODE"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    for source, target in (
        ("EX_DIVIDEND_DATE", "ex_date"),
        ("NOTICE_DATE", "notice_date"),
        ("PUBLISH_DATE", "publish_date"),
    ):
        frame[target] = pd.to_datetime(frame[source], errors="coerce")
    frame["known_date"] = frame[
        ["ex_date", "notice_date", "publish_date"]
    ].max(axis=1)
    frame["cash_per_ten"] = pd.to_numeric(
        frame["PRETAX_BONUS_RMB"], errors="coerce"
    )
    frame["event_total_shares"] = pd.to_numeric(
        frame["TOTAL_SHARES"], errors="coerce"
    )
    valid = (
        frame["security_code"].str.startswith(_POOL_PREFIXES, na=False)
        & frame["known_date"].notna()
        & frame["ex_date"].notna()
        & frame["ASSIGN_PROGRESS"].eq("实施分配")
        & (frame["cash_per_ten"] > 0.0)
        & (frame["event_total_shares"] > 0.0)
    )
    result = frame.loc[
        valid,
        [
            "security_code",
            "ex_date",
            "known_date",
            "cash_per_ten",
            "event_total_shares",
        ],
    ].copy()
    result["total_cash_yuan"] = (
        result["cash_per_ten"] / 10.0 * result["event_total_shares"]
    )
    return result


def _dividend_slot_counts3(panel: dict) -> np.ndarray:
    """Count occupied completed 365-day ex-date slots for each decision row."""
    trade_dates = np.asarray(panel["trade_dates"], dtype="datetime64[D]")
    stock_codes = np.asarray(panel["stock_codes"])
    normalized = _normalized_codes(stock_codes)
    code_map = pd.Series(np.arange(stock_codes.size), index=normalized)
    events = _implemented_cash_dividends()
    events = events[
        events["security_code"].isin(code_map.index)
        & (events["known_date"] <= pd.Timestamp(str(trade_dates[-1])))
    ].copy()
    events["stock_col"] = events["security_code"].map(code_map).astype(np.intp)
    known_start = np.searchsorted(
        trade_dates,
        events["known_date"].to_numpy(dtype="datetime64[D]"),
        side="right",
    )
    ex_dates = events["ex_date"].to_numpy(dtype="datetime64[D]")
    columns = events["stock_col"].to_numpy(dtype=np.intp)
    counts = np.zeros((trade_dates.size, stock_codes.size), dtype=np.int8)
    for slot in range(3):
        lower = ex_dates + np.timedelta64(365 * slot, "D")
        upper = ex_dates + np.timedelta64(365 * (slot + 1), "D")
        start = np.maximum(
            known_start,
            np.searchsorted(trade_dates, lower, side="right"),
        )
        end = np.searchsorted(trade_dates, upper, side="right")
        usable = (start < end) & (start < trade_dates.size)
        difference = np.zeros(
            (trade_dates.size + 1, stock_codes.size),
            dtype=np.int16,
        )
        np.add.at(difference, (start[usable], columns[usable]), 1)
        np.add.at(difference, (end[usable], columns[usable]), -1)
        occupied = np.cumsum(difference[:-1], axis=0) > 0
        counts += occupied.astype(np.int8)
    history_complete = (
        trade_dates - np.timedelta64(3 * 365, "D") >= _SOURCE_START
    )
    counts[~history_complete] = -1
    return counts


def _trailing_cash_yield(panel: dict, window: int) -> np.ndarray:
    trade_dates = np.asarray(panel["trade_dates"], dtype="datetime64[D]")
    stock_codes = np.asarray(panel["stock_codes"])
    normalized = _normalized_codes(stock_codes)
    code_map = pd.Series(np.arange(stock_codes.size), index=normalized)

    events = _implemented_cash_dividends()
    events = events[
        events["security_code"].isin(code_map.index)
        & (events["known_date"] <= pd.Timestamp(str(trade_dates[-1])))
    ].copy()
    events["stock_col"] = events["security_code"].map(code_map).astype(np.intp)
    events["effective_row"] = np.searchsorted(
        trade_dates,
        events["known_date"].to_numpy(dtype="datetime64[D]"),
        side="right",
    )
    events = events[events["effective_row"] < trade_dates.size]

    cash = np.zeros((trade_dates.size, stock_codes.size), dtype=np.float64)
    np.add.at(
        cash,
        (
            events["effective_row"].to_numpy(dtype=np.intp),
            events["stock_col"].to_numpy(dtype=np.intp),
        ),
        events["total_cash_yuan"].to_numpy(dtype=np.float64),
    )
    cumulative = np.cumsum(cash, axis=0, dtype=np.float64)
    trailing = cumulative.copy()
    if trade_dates.size > window:
        trailing[window:] -= cumulative[:-window]

    pre_close = np.asarray(panel["preClose"], dtype=np.float64)
    total_share_wan = np.asarray(panel["total_share"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        market_cap_yuan = pre_close * total_share_wan * 10_000.0
        result = trailing / market_cap_yuan
    valid = (
        np.isfinite(market_cap_yuan)
        & (market_cap_yuan > 0.0)
        & np.isfinite(result)
    )
    result = np.where(valid, result, np.nan)

    history_complete = (
        trade_dates - np.timedelta64(window * 2, "D") >= _SOURCE_START
    )
    result[~history_complete] = np.nan
    return result


class CompletedDividendYield252PIT:
    """Rank implemented cash paid over the prior 252 exchange rows."""

    hist_days = 252
    scores_are_ranks = False
    requires_full_history = False

    @classmethod
    def raw_yield(cls, panel: dict) -> np.ndarray:
        return _trailing_cash_yield(panel, 252)

    def calc_batch(self, panel: dict) -> np.ndarray:
        return self.raw_yield(panel).astype(np.float32)


class CompletedDividendYield504PIT:
    """Rank implemented cash paid over the prior 504 exchange rows."""

    hist_days = 504
    scores_are_ranks = False
    requires_full_history = False

    @classmethod
    def raw_yield(cls, panel: dict) -> np.ndarray:
        return _trailing_cash_yield(panel, 504)

    def calc_batch(self, panel: dict) -> np.ndarray:
        return self.raw_yield(panel).astype(np.float32)


class _FilterDividendYieldTopPositive252PIT:
    """Keep a top fraction of stocks with observed positive 252-row yield."""

    hist_days = 252
    top_fraction: float

    def calc_batch(self, panel: dict) -> np.ndarray:
        dividend_yield = CompletedDividendYield252PIT.raw_yield(panel)
        positive = np.isfinite(dividend_yield) & (dividend_yield > 0.0)
        values = np.where(positive, dividend_yield, np.nan)
        with np.errstate(invalid="ignore"):
            threshold = np.nanquantile(
                values,
                1.0 - self.top_fraction,
                axis=1,
            )
        allowed = positive & (dividend_yield >= threshold[:, None])
        return np.where(allowed, 1.0, np.nan).astype(np.float32)


class FilterDividendYieldTop50Positive252PIT(
    _FilterDividendYieldTopPositive252PIT
):
    """Keep the higher-yielding half of observed cash-dividend payers."""

    top_fraction = 0.50


class FilterDividendYieldTop30Positive252PIT(
    _FilterDividendYieldTopPositive252PIT
):
    """Keep the highest-yielding 30% of observed cash-dividend payers."""

    top_fraction = 0.30


class FilterDividendYieldTop20Positive252PIT(
    _FilterDividendYieldTopPositive252PIT
):
    """Keep the highest-yielding 20% of observed cash-dividend payers."""

    top_fraction = 0.20


class FilterDividendYieldTop10Positive252PIT(
    _FilterDividendYieldTopPositive252PIT
):
    """Keep the highest-yielding 10% of observed cash-dividend payers."""

    top_fraction = 0.10


class FilterDividendYieldPositive252PIT:
    """Require at least one observed positive cash payment in 252 rows."""

    hist_days = 252

    def calc_batch(self, panel: dict) -> np.ndarray:
        dividend_yield = CompletedDividendYield252PIT.raw_yield(panel)
        allowed = np.isfinite(dividend_yield) & (dividend_yield > 0.0)
        return np.where(allowed, 1.0, np.nan).astype(np.float32)


class CompletedDividendConsistency3YPIT:
    """Score how many of the last three 365-day slots contain a cash payout."""

    hist_days = 756
    scores_are_ranks = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        counts = _dividend_slot_counts3(panel)
        return np.where(counts >= 0, counts / 3.0, np.nan).astype(np.float32)


class FilterDividendPaidAtLeast2Of3YearsPIT:
    """Require cash payouts in at least two of three completed annual slots."""

    hist_days = 756

    def calc_batch(self, panel: dict) -> np.ndarray:
        counts = _dividend_slot_counts3(panel)
        return np.where(counts >= 2, 1.0, np.nan).astype(np.float32)


class FilterDividendPaidEachOfLast3YearsPIT:
    """Require cash payouts in all three completed annual slots."""

    hist_days = 756

    def calc_batch(self, panel: dict) -> np.ndarray:
        counts = _dividend_slot_counts3(panel)
        return np.where(counts == 3, 1.0, np.nan).astype(np.float32)
