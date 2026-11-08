"""Video factor research using completed bars and an IPO-anchored return chain.

All scores are oriented so larger means preferred. ``Low`` and discount
scores therefore negate the displayed video statistic. A decision at T uses
only bars strictly before T. Corporate-action adjustment is causal: the IPO
bar has scale one; later scale is the previous adjusted close / preClose.

Lifetime factors require the actual IPO bar, identified by issue_date and
listing_age == 0. An entirely absent OHLC row is skipped without inventing a
bar. This is an explicit no-bar convention, NOT evidence that every missing
row is a legitimate suspension. A missing close/preClose in a present bar
permanently invalidates the price chain; a malformed high/low invalidates
the lifetime extremes only. Missing initial history cannot be repaired
by anchoring at the first row of a later runtime slice.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numba import njit

from factor.base import FactorDefinition, FactorMetadata
from offline_data import RuntimeSlice


BILIBILI_FACTOR_NAMES = (
    "BiliRawIssueDiscount",
    "BiliAdjustedIssueDiscount",
    "BiliLowLifetimeRangeRatio",
    "BiliHighLifetimeRangeRatio",
    "BiliLowLifetimeAmplitude",
)
LIFETIME_STATE_NAMES = ("ipo_adjusted_close", "ipo_lifetime_high", "ipo_lifetime_low")
_CHAIN_OUTPUT_NAMES = BILIBILI_FACTOR_NAMES + LIFETIME_STATE_NAMES
REQUIRED_FIELDS = (
    "open", "high", "low", "close", "preClose", "issue_price",
    "issue_date", "listing_age", "delisted_mask",
)
SEMANTICS_VERSION = "bilibili-completed-ipo-return-chain-v1"


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _lifetime_chain(dates, issue_dates, issue_prices, opening, high, low, close,
                    pre_close, listing_age, delisted, slots, phase_shift, result):
    rows, stocks = close.shape
    adjusted_close = np.full(stocks, np.nan)
    lifetime_high = np.full(stocks, np.nan)
    lifetime_low = np.full(stocks, np.nan)
    alive = np.zeros(stocks, dtype=np.bool_)
    range_alive = np.zeros(stocks, dtype=np.bool_)
    nat = np.iinfo(np.int64).min
    for row in range(rows - phase_shift):
        output_row = row + phase_shift
        for stock in range(stocks):
            # Match the public NumPy chain's per-row float64 conversions even
            # when the immutable runtime uses float32 source fields.
            op = np.float64(opening[row, stock])
            hi = np.float64(high[row, stock])
            lo = np.float64(low[row, stock])
            cl = np.float64(close[row, stock])
            pc = np.float64(pre_close[row, stock])
            issue = np.float64(issue_prices[stock])
            valid_issue = np.isfinite(issue) and issue > 0
            member = listing_age[row, stock] >= 0 and not delisted[row, stock]
            output_member = (listing_age[output_row, stock] >= 0
                             and not delisted[output_row, stock])
            known_issue = issue_dates[stock] != nat and issue_dates[stock] <= dates[row]
            if (slots[0] >= 0 and member and output_member and known_issue
                    and valid_issue and np.isfinite(cl) and cl > 0):
                result[slots[0], output_row, stock] = -cl / issue
            absent = np.isnan(op) and np.isnan(hi) and np.isnan(lo) and np.isnan(cl)
            complete = np.isfinite(cl) and cl > 0 and np.isfinite(pc) and pc > 0
            complete_range = (complete and np.isfinite(hi) and np.isfinite(lo)
                              and lo > 0 and pc > 0 and hi >= lo and cl >= lo and cl <= hi)
            initial = issue_dates[stock] == dates[row] and listing_age[row, stock] == 0 and member
            alive[stock] = member and ((alive[stock] and (absent or complete)) or (initial and complete))
            range_alive[stock] = member and ((range_alive[stock] and (absent or complete_range))
                                            or (initial and complete_range))
            update = alive[stock] and complete
            scale = np.nan
            if update and not initial:
                scale = adjusted_close[stock] / pc
            if initial and update:
                scale = 1.0
            adjusted_bar_close = cl * scale
            adjusted_bar_high = hi * scale
            adjusted_bar_low = lo * scale
            finite_adjusted = np.isfinite(adjusted_bar_close) and adjusted_bar_close > 0
            finite_range = (np.isfinite(adjusted_bar_high) and adjusted_bar_high > 0
                            and np.isfinite(adjusted_bar_low) and adjusted_bar_low > 0)
            alive[stock] = alive[stock] and (not update or finite_adjusted)
            range_alive[stock] = (range_alive[stock] and alive[stock]
                                  and (not update or finite_range))
            update = update and alive[stock]
            if update:
                adjusted_close[stock] = adjusted_bar_close
            range_update = update and range_alive[stock]
            if initial and range_update:
                lifetime_high[stock] = adjusted_bar_high
                lifetime_low[stock] = adjusted_bar_low
            if range_update:
                lifetime_high[stock] = np.maximum(lifetime_high[stock], adjusted_bar_high)
                lifetime_low[stock] = np.minimum(lifetime_low[stock], adjusted_bar_low)
            valid = alive[stock] and output_member
            if slots[5] >= 0 and valid:
                result[slots[5], output_row, stock] = adjusted_close[stock]
            if valid and range_alive[stock]:
                if slots[6] >= 0:
                    result[slots[6], output_row, stock] = lifetime_high[stock]
                if slots[7] >= 0:
                    result[slots[7], output_row, stock] = lifetime_low[stock]
            if slots[1] >= 0 and valid and update and valid_issue:
                result[slots[1], output_row, stock] = -adjusted_close[stock] / issue
            if (slots[2] >= 0 or slots[3] >= 0) and valid and range_alive[stock]:
                ratio = lifetime_high[stock] / lifetime_low[stock]
                if slots[3] >= 0:
                    result[slots[3], output_row, stock] = ratio
                if slots[2] >= 0:
                    result[slots[2], output_row, stock] = -ratio
            if slots[4] >= 0 and valid and range_alive[stock] and valid_issue:
                result[slots[4], output_row, stock] = (lifetime_low[stock] - lifetime_high[stock]) / issue
    # Same final canonicalization for all five public outputs, including NaN
    # signs from the negative-ratio branch and infinity after finite overflow.
    for slot in range(result.shape[0]):
        for row in range(rows):
            for stock in range(stocks):
                if not np.isfinite(result[slot, row, stock]):
                    result[slot, row, stock] = np.nan


def _native_chain_field(values: np.ndarray) -> np.ndarray:
    """Lossless storage adaptation before the chain's float64 arithmetic."""
    values = np.asarray(values)
    dtype = values.dtype.newbyteorder("=")
    if dtype.kind == "f":
        dtype = np.promote_types(dtype, np.dtype(np.float32))
    return values.astype(dtype, copy=False)


def _calculate_chain_outputs(
    trade_dates: np.ndarray, panel: Mapping[str, np.ndarray],
    *, output_phase: str = "next_open",
    factor_names: tuple[str, ...] = BILIBILI_FACTOR_NAMES,
) -> Mapping[str, np.ndarray]:
    """Compute requested scores together, retaining one common date chain.

    Output matrices are read-only float64, date by stock. ``next_open`` is
    the production default: row T contains strictly prior completed bars.
    ``completed_close`` is an explicit research phase whose row T includes
    that day's completed bar and uses only that day's membership. Lifetime min/max
    and adjusted close are only stock-sized state arrays, shared across all
    five signals. No full adjusted OHLC copies or Python per-stock loops are used.
    """
    if output_phase not in ("next_open", "completed_close"):
        raise ValueError("output_phase must be next_open or completed_close")
    if (not factor_names or len(set(factor_names)) != len(factor_names)
            or set(factor_names).difference(_CHAIN_OUTPUT_NAMES)):
        raise ValueError("factor_names must be a nonempty unique Bilibili vocabulary")
    dates = np.asarray(trade_dates, dtype="datetime64[D]")
    close = np.asarray(panel["close"])
    if close.ndim != 2 or dates.shape != (close.shape[0],):
        raise ValueError("trade_dates and OHLC must have matching date axes")
    if np.isnat(dates).any() or np.any(dates[1:] <= dates[:-1]):
        raise ValueError("trade_dates must be finite and strictly increasing")
    rows, stocks = close.shape
    for name in REQUIRED_FIELDS:
        expected = (stocks,) if name in ("issue_date", "issue_price") else close.shape
        if np.asarray(panel[name]).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    issue_date = np.asarray(panel["issue_date"], dtype="datetime64[D]")
    issue_price = np.asarray(panel["issue_price"], dtype=np.float64)
    slots = np.full(len(_CHAIN_OUTPUT_NAMES), -1, dtype=np.int64)
    for slot, name in enumerate(factor_names):
        slots[_CHAIN_OUTPUT_NAMES.index(name)] = slot
    output = np.full((len(factor_names), rows, stocks), np.nan, dtype=np.float64)
    _lifetime_chain(
        dates.view(np.int64), issue_date.view(np.int64), issue_price,
        *(_native_chain_field(panel[name]) for name in (
            "open", "high", "low", "close", "preClose", "listing_age", "delisted_mask",
        )), slots, int(output_phase == "next_open"), output,
    )
    output.flags.writeable = False
    return MappingProxyType({name: output[slot] for slot, name in enumerate(factor_names)})


def calculate_bilibili_scores(trade_dates, panel, *, output_phase="next_open", factor_names=BILIBILI_FACTOR_NAMES):
    if set(factor_names).difference(BILIBILI_FACTOR_NAMES):
        raise ValueError("factor_names must be a nonempty unique Bilibili vocabulary")
    return _calculate_chain_outputs(trade_dates, panel, output_phase=output_phase, factor_names=factor_names)


def calculate_lifetime_state(trade_dates, panel):
    """IPO-anchored prices/extrema available at each opening, never factor scores."""
    return _calculate_chain_outputs(trade_dates, panel, factor_names=LIFETIME_STATE_NAMES)


def prepare_bilibili_factor_definitions(runtime: RuntimeSlice) -> tuple[FactorDefinition, ...]:
    """Bind a single shared calculation to one immutable full-history slice.

    Call this once per sealed split, then pass the returned public tuple to
    ``factor.precompute_factors``. Definitions are deliberately bound to this
    source, date axis and stock axis, and reject any other input objects,
    including same-valued copies. Original read-only views make binding
    verification O(number of fields), without full-panel scans or copies.
    The caller must request sufficient preload to include the runtime's first
    date. IPOs predating that date remain invalid for lifetime factors.
    """
    if runtime.manifest.actual_preload_rows >= runtime.manifest.requested_preload_rows:
        raise ValueError("lifetime factors require preload extending to the runtime's first row")
    source = inspect.getsource(sys.modules[__name__])
    identity = {
        "source": source, "semantics_version": SEMANTICS_VERSION,
        "runtime_source_sha256": runtime.manifest.source_sha256,
        "dates": [str(runtime.trade_dates[0]), str(runtime.trade_dates[-1])],
        "stocks": runtime.stock_codes,
    }
    scores = calculate_bilibili_scores(runtime.trade_dates, runtime.data)

    def implementation(name: str) -> type:
        class BoundFactor:
            def calc_batch(self, panel):
                # This definition explicitly opts into original immutable
                # views. No dtype conversion or panel-wide comparison is
                # needed to verify its binding to this sealed calculation.
                for field in REQUIRED_FIELDS:
                    if panel[field] is not runtime.field(field) or panel[field].flags.writeable:
                        raise ValueError("bound Bilibili factor received a different runtime panel")
                return scores[name]
        BoundFactor.__name__ = name
        return BoundFactor

    return tuple(
        FactorDefinition(
            metadata=FactorMetadata(
                name=name, version=SEMANTICS_VERSION,
                hist_days=runtime.n_dates - 1,
                required_fields=REQUIRED_FIELDS,
                implementation_hash=hashlib.sha256(json.dumps(
                    {**identity, "factor": name}, sort_keys=True,
                    separators=(",", ":"),
                ).encode()).hexdigest(),
            ),
            implementation=implementation(name),
            raw_runtime_view=True,
        )
        for name in BILIBILI_FACTOR_NAMES
    )


__all__ = [
    "BILIBILI_FACTOR_NAMES", "SEMANTICS_VERSION", "calculate_bilibili_scores",
    "prepare_bilibili_factor_definitions",
    "calculate_lifetime_state", "LIFETIME_STATE_NAMES",
]


class BiliAdjustedIssueDiscount:
    """IPO-anchored discount; the canonical batch path shares its price chain."""
    hist_days = 100000

    def calc_batch(self, panel):
        name = "BiliAdjustedIssueDiscount"
        return calculate_bilibili_scores(panel["trade_dates"], panel, factor_names=(name,))[name]


class BiliHighLifetimeRangeRatio:
    """Completed lifetime range; the canonical batch path shares its price chain."""
    hist_days = 100000

    def calc_batch(self, panel):
        name = "BiliHighLifetimeRangeRatio"
        return calculate_bilibili_scores(panel["trade_dates"], panel, factor_names=(name,))[name]
