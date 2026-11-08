"""Frozen former NumPy chain checks the one compiled production path."""
from types import MappingProxyType
from typing import Mapping
import numpy as np
import pytest
from factor.library.bilibili import (
    BILIBILI_FACTOR_NAMES, REQUIRED_FIELDS, calculate_bilibili_scores,
)
from test_rl_runtime_slice import _runtime_arrays


def _numpy_lifetime(
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
    five signals. No full adjusted OHLC copies or per-stock loops are used.
    """
    if output_phase not in ("next_open", "completed_close"):
        raise ValueError("output_phase must be next_open or completed_close")
    if (not factor_names or len(set(factor_names)) != len(factor_names)
            or set(factor_names).difference(BILIBILI_FACTOR_NAMES)):
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
    valid_issue_price = np.isfinite(issue_price) & (issue_price > 0)
    result = {name: np.full(close.shape, np.nan) for name in factor_names}
    adjusted_close = np.full(stocks, np.nan)
    lifetime_high = np.full(stocks, np.nan)
    lifetime_low = np.full(stocks, np.nan)
    alive = np.zeros(stocks, dtype=bool)
    range_alive = np.zeros(stocks, dtype=bool)

    for row in range(rows - (output_phase == "next_open")):
        output_row = row + (output_phase == "next_open")
        opening, high, low, closing, pre_close = (
            np.asarray(panel[name][row], dtype=np.float64)
            for name in ("open", "high", "low", "close", "preClose")
        )
        member = (panel["listing_age"][row] >= 0) & ~panel["delisted_mask"][row]
        output_member = (panel["listing_age"][output_row] >= 0) & ~panel["delisted_mask"][output_row]
        known_issue = ~np.isnat(issue_date) & (issue_date <= dates[row])
        if BILIBILI_FACTOR_NAMES[0] in result:
            raw_valid = member & known_issue & valid_issue_price & np.isfinite(closing) & (closing > 0)
            np.divide(-closing, issue_price, out=result[BILIBILI_FACTOR_NAMES[0]][output_row], where=raw_valid & output_member)
        absent = np.isnan(opening) & np.isnan(high) & np.isnan(low) & np.isnan(closing)
        complete = (
            np.isfinite(closing) & (closing > 0)
            & np.isfinite(pre_close) & (pre_close > 0)
        )
        complete_range = (
            complete & np.isfinite(high) & np.isfinite(low)
            & (low > 0) & (pre_close > 0)
            & (high >= low)
            & (closing >= low) & (closing <= high)
        )
        initial = (issue_date == dates[row]) & (panel["listing_age"][row] == 0) & member
        # The equality date condition can occur only once. No later good bar
        # may restart a chain that lost its initial bar or valid preClose.
        alive = member & ((alive & (absent | complete)) | (initial & complete))
        range_alive = member & (
            (range_alive & (absent | complete_range)) | (initial & complete_range)
        )
        update = alive & complete
        scale = np.full(stocks, np.nan)
        np.divide(adjusted_close, pre_close, out=scale, where=update & ~initial)
        scale[initial & update] = 1.0
        with np.errstate(over="ignore", invalid="ignore"):
            adjusted_bar_close = closing * scale
            adjusted_bar_high = high * scale
            adjusted_bar_low = low * scale
        finite_adjusted = np.isfinite(adjusted_bar_close) & (adjusted_bar_close > 0)
        finite_range = (
            np.isfinite(adjusted_bar_high) & (adjusted_bar_high > 0)
            & np.isfinite(adjusted_bar_low) & (adjusted_bar_low > 0)
        )
        alive &= ~update | finite_adjusted
        range_alive &= alive & (~update | finite_range)
        update &= alive
        adjusted_close[update] = adjusted_bar_close[update]
        range_update = update & range_alive
        lifetime_high[initial & range_update] = adjusted_bar_high[initial & range_update]
        lifetime_low[initial & range_update] = adjusted_bar_low[initial & range_update]
        lifetime_high[range_update] = np.maximum(lifetime_high[range_update], adjusted_bar_high[range_update])
        lifetime_low[range_update] = np.minimum(lifetime_low[range_update], adjusted_bar_low[range_update])
        valid = alive & output_member
        if BILIBILI_FACTOR_NAMES[1] in result:
            np.divide(-adjusted_close, issue_price,
                out=result[BILIBILI_FACTOR_NAMES[1]][output_row],
                where=valid & update & valid_issue_price)
        high_name, low_name = BILIBILI_FACTOR_NAMES[3], BILIBILI_FACTOR_NAMES[2]
        if high_name in result or low_name in result:
            target = result[high_name if high_name in result else low_name][output_row]
            np.divide(lifetime_high, lifetime_low, out=target, where=valid & range_alive)
            if low_name in result:
                np.negative(target, out=result[low_name][output_row])
        if BILIBILI_FACTOR_NAMES[4] in result:
            np.divide(lifetime_low - lifetime_high, issue_price,
                out=result[BILIBILI_FACTOR_NAMES[4]][output_row],
                where=valid & range_alive & valid_issue_price)
    for values in result.values():
        values[~np.isfinite(values)] = np.nan
        values.flags.writeable = False
    return MappingProxyType(result)


def _case(dtype):
    data = _runtime_arrays(rows=73, stocks=28)
    for name in ('open', 'high', 'low', 'close', 'preClose', 'issue_price'):
        data[name] = data[name].astype(dtype)
    data['issue_date'][0] = np.datetime64('NaT')
    data['issue_date'][1] = data['trade_dates'][0] - 1
    data['issue_date'][2] = data['trade_dates'][3]
    data['listing_age'][:3, 2] = -1
    data['listing_age'][3:, 2] = np.arange(70)
    data['delisted_mask'][11:, 3] = True
    data['listing_age'][17:, 4] = -1
    if np.dtype(dtype).kind == 'f':
        native = np.dtype(dtype).newbyteorder('=')
        data['issue_price'][5:10] = [np.nan, np.inf, -np.inf, 0, -0.0]
        for name in ('open', 'high', 'low', 'close', 'preClose'):
            data[name][13:16, 10:15] = np.nan
        for index, name in enumerate(('open', 'high', 'low', 'close', 'preClose')):
            data[name][20 + index, 15:23] = [np.nan, np.inf, -np.inf, 0, -0.0, -1,
                                           np.finfo(native).max, np.finfo(native).tiny]
        data['preClose'][23, 24] = np.finfo(native).tiny
    for values in data.values():
        values.flags.writeable = False
    return data


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.int64,
                                  np.dtype('>f4'), np.dtype('>f8'), np.dtype('>i8')])
@pytest.mark.parametrize('phase', ['next_open', 'completed_close'])
@pytest.mark.parametrize('names', [BILIBILI_FACTOR_NAMES, BILIBILI_FACTOR_NAMES[1:4:2],
                                   *((name,) for name in BILIBILI_FACTOR_NAMES)])
def test_compiled_chain_preserves_numpy_public_values_and_storage(dtype, phase, names):
    data = _case(dtype)
    before = {name: values.tobytes() for name, values in data.items()}
    with np.errstate(all='ignore'):
        expected = _numpy_lifetime(data['trade_dates'], data, output_phase=phase, factor_names=names)
        actual = calculate_bilibili_scores(data['trade_dates'], data, output_phase=phase, factor_names=names)
    assert tuple(actual) == names
    for name in names:
        assert actual[name].dtype == np.float64
        assert not actual[name].flags.writeable
        assert actual[name].tobytes() == expected[name].tobytes()
    assert all(values.tobytes() == before[name] for name, values in data.items())


@pytest.mark.parametrize('rows,stocks', [(0, 0), (0, 3), (1, 3), (7, 0)])
@pytest.mark.parametrize('phase', ['next_open', 'completed_close'])
def test_compiled_chain_empty_axes(rows, stocks, phase):
    data = _runtime_arrays(rows=max(3, rows), stocks=stocks)
    data['trade_dates'] = data['trade_dates'][:rows]
    for name in ('open', 'high', 'low', 'close', 'preClose', 'listing_age', 'delisted_mask'):
        data[name] = data[name][:rows]
    expected = _numpy_lifetime(data['trade_dates'], data, output_phase=phase)
    actual = calculate_bilibili_scores(data['trade_dates'], data, output_phase=phase)
    assert all(actual[name].tobytes() == expected[name].tobytes() for name in expected)


def test_compiled_chain_noncontiguous_inputs_preserve_numpy_bytes():
    data = _case(np.float64)
    for name in REQUIRED_FIELDS:
        data[name] = data[name][::2] if data[name].ndim == 1 else data[name][:, ::2]
    with np.errstate(all='ignore'):
        expected = _numpy_lifetime(data['trade_dates'], data)
        actual = calculate_bilibili_scores(data['trade_dates'], data)
    assert all(actual[name].tobytes() == expected[name].tobytes() for name in expected)
