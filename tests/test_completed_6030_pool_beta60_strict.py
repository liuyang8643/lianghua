from __future__ import annotations

import ast
import inspect
import time
from pathlib import Path

import numpy as np
import pytest

from factor_db.factors.Completed6030PoolBeta60Strict import (
    Completed6030PoolBeta60Strict,
    _compute_6030_market_returns,
    _pool_mask,
)


WINDOW = 60
MIN_MARKET_STOCKS = 30


def _codes(pool_stocks: int = 36, outside_stocks: int = 4) -> np.ndarray:
    prefixes = ("60", "00", "30")
    pool = [
        f"{prefixes[index % len(prefixes)]}{index:04d}.SZ"
        for index in range(pool_stocks)
    ]
    outside_prefixes = ("68", "43", "83", "90")
    outside = [
        f"{outside_prefixes[index % len(outside_prefixes)]}{index:04d}.XX"
        for index in range(outside_stocks)
    ]
    return np.asarray(pool + outside)


def _panel(
    rows: int = 143,
    pool_stocks: int = 36,
    outside_stocks: int = 4,
) -> dict:
    codes = _codes(pool_stocks, outside_stocks)
    stocks = len(codes)
    row = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    official_return = (
        0.003 * np.sin((row + 1.0) / 8.0 + stock / 11.0)
        + 0.00017 * (stock - (stocks - 1.0) / 2.0)
        + 0.0002 * np.cos(row / 19.0 - stock / 7.0)
    )
    pre_close = 7.0 + 0.009 * row + 0.31 * stock
    close = pre_close * (1.0 + official_return)
    shape = close.shape
    return {
        "close": close,
        "preClose": pre_close,
        "stock_codes": codes,
        "open": pre_close * 1.001,
        "high": np.maximum(close, pre_close) * 1.01,
        "low": np.minimum(close, pre_close) * 0.99,
        "volume": np.full(shape, 2.0e6, dtype=np.float64),
        "amount": np.full(shape, 2.0e7, dtype=np.float64),
        "st_mask": np.zeros(shape, dtype=bool),
    }


def _copy_panel(panel: dict) -> dict:
    return {
        name: value.copy() if hasattr(value, "copy") else value
        for name, value in panel.items()
    }


def _official_returns(panel: dict) -> tuple[np.ndarray, np.ndarray]:
    close = np.asarray(panel["close"], dtype=np.float64)
    pre_close = np.asarray(panel["preClose"], dtype=np.float64)
    valid = (
        np.isfinite(close)
        & (close > 0.0)
        & np.isfinite(pre_close)
        & (pre_close > 0.0)
    )
    returns = np.full(close.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(close, pre_close, out=returns, where=valid)
        returns[valid] -= 1.0
    valid &= np.isfinite(returns)
    returns[~valid] = np.nan
    return returns, valid


def _oracle(panel: dict) -> tuple[np.ndarray, np.ndarray]:
    returns, valid = _official_returns(panel)
    pool = _pool_mask(np.asarray(panel["stock_codes"]))
    market = np.full(returns.shape[0], np.nan, dtype=np.float64)
    for row in range(returns.shape[0]):
        row_valid = valid[row] & pool
        if np.count_nonzero(row_valid) >= MIN_MARKET_STOCKS:
            market[row] = np.mean(returns[row, row_valid], dtype=np.float64)

    expected = np.full(returns.shape, np.nan, dtype=np.float32)
    for output_row in range(WINDOW, returns.shape[0]):
        market_window = market[output_row - WINDOW : output_row]
        if not np.all(np.isfinite(market_window)):
            continue
        centered_market = market_window - np.mean(market_window)
        denominator = float(np.dot(centered_market, centered_market))
        if not np.isfinite(denominator) or denominator <= 0.0:
            continue
        for stock in range(returns.shape[1]):
            stock_window = returns[
                output_row - WINDOW : output_row,
                stock,
            ]
            if not np.all(np.isfinite(stock_window)):
                continue
            centered_stock = stock_window - np.mean(stock_window)
            numerator = float(np.dot(centered_stock, centered_market))
            beta = numerator / denominator
            if np.isfinite(beta):
                expected[output_row, stock] = -beta
    return expected, market


def _assert_same_raw_scores(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(
        np.isfinite(actual),
        np.isfinite(expected),
    )
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite], expected[finite])


def test_metadata_prefix_pool_and_independent_oracle_match():
    factor = Completed6030PoolBeta60Strict()
    assert factor.hist_days == WINDOW
    assert factor.update_frequency == "daily"
    assert factor.pre_ranked is False
    assert factor.requires_full_history is False
    np.testing.assert_array_equal(
        _pool_mask(
            np.asarray(
                [
                    "600000.SH",
                    "000001.SZ",
                    "300001.SZ",
                    "688001.SH",
                    "430001.BJ",
                    "830001.BJ",
                ]
            )
        ),
        np.asarray([True, True, True, False, False, False]),
    )

    panel = _panel()
    actual = factor.calc_batch(panel)
    expected, expected_market = _oracle(panel)
    actual_market = _compute_6030_market_returns(
        panel["close"],
        panel["preClose"],
        panel["stock_codes"],
    )
    assert actual.dtype == np.float32
    np.testing.assert_allclose(
        actual_market,
        expected_market,
        rtol=0.0,
        atol=2e-16,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        np.isfinite(actual),
        np.isfinite(expected),
    )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-6,
        atol=2e-7,
        equal_nan=True,
    )


def test_candidate_is_in_market_mean_and_beta_is_not_clipped():
    rows, stocks = WINDOW + 1, MIN_MARKET_STOCKS
    codes = _codes(pool_stocks=stocks, outside_stocks=0)
    stock_return = np.linspace(-0.03, 0.03, WINDOW)
    returns = np.zeros((rows, stocks), dtype=np.float64)
    returns[:WINDOW, 0] = stock_return
    pre_close = np.full((rows, stocks), 100.0, dtype=np.float64)
    close = pre_close * (1.0 + returns)

    market = _compute_6030_market_returns(close, pre_close, codes)
    np.testing.assert_allclose(
        market[:WINDOW],
        stock_return / MIN_MARKET_STOCKS,
        rtol=0.0,
        atol=2e-17,
    )
    actual = Completed6030PoolBeta60Strict().calc_batch(
        {
            "close": close,
            "preClose": pre_close,
            "stock_codes": codes,
        }
    )
    assert actual[WINDOW, 0] == pytest.approx(
        -float(MIN_MARKET_STOCKS),
        rel=2e-6,
    )
    np.testing.assert_array_equal(
        actual[WINDOW, 1:],
        np.zeros(stocks - 1, dtype=np.float32),
    )
    assert actual[WINDOW, 0] < -1.0


def test_centered_covariance_is_stable_with_large_return_offset():
    rows, stocks = WINDOW + 1, MIN_MARKET_STOCKS
    day_signal = np.linspace(-0.02, 0.02, WINDOW)
    returns = np.full((rows, stocks), 1.0e8, dtype=np.float64)
    returns[:WINDOW] += day_signal[:, None]
    returns[:WINDOW, 0] += 2.0 * day_signal
    pre_close = np.ones((rows, stocks), dtype=np.float64)
    close = pre_close * (1.0 + returns)
    actual = Completed6030PoolBeta60Strict().calc_batch(
        {
            "close": close,
            "preClose": pre_close,
            "stock_codes": _codes(stocks, 0),
        }
    )[WINDOW]

    market_loading = (3.0 + stocks - 1.0) / stocks
    assert actual[0] == pytest.approx(
        -3.0 / market_loading,
        rel=2e-6,
    )
    np.testing.assert_allclose(
        actual[1:],
        -1.0 / market_loading,
        rtol=2e-6,
        atol=2e-7,
    )


def test_first_valid_row_and_exact_completed_window_boundaries():
    panel = _panel(rows=WINDOW + 3, pool_stocks=31, outside_stocks=0)
    factor = Completed6030PoolBeta60Strict()
    baseline = factor.calc_batch(panel)
    assert np.isnan(baseline[:WINDOW]).all()
    assert np.isfinite(baseline[WINDOW:]).all()

    changed = _copy_panel(panel)
    changed["close"][0, 0] *= 1.03
    shifted = factor.calc_batch(changed)
    assert shifted[WINDOW, 0] != baseline[WINDOW, 0]
    np.testing.assert_array_equal(
        shifted[WINDOW + 1 :, 0],
        baseline[WINDOW + 1 :, 0],
    )


def test_market_requires_at_least_30_valid_pool_returns_each_day():
    panel = _panel(
        rows=WINDOW * 2 + 2,
        pool_stocks=MIN_MARKET_STOCKS,
        outside_stocks=2,
    )
    bad_row = 37
    panel["close"][bad_row, 0] = np.nan
    market = _compute_6030_market_returns(
        panel["close"],
        panel["preClose"],
        panel["stock_codes"],
    )
    assert np.isnan(market[bad_row])
    assert np.isfinite(market[np.arange(len(market)) != bad_row]).all()

    actual = Completed6030PoolBeta60Strict().calc_batch(panel)
    assert np.isnan(
        actual[bad_row + 1 : bad_row + WINDOW + 1]
    ).all()
    assert np.isfinite(actual[bad_row + WINDOW + 1]).all()


@pytest.mark.parametrize("field", ["close", "preClose"])
@pytest.mark.parametrize(
    "invalid",
    [np.nan, np.inf, -np.inf, 0.0, -1.0],
)
def test_invalid_stock_return_strictly_poisons_its_affected_windows_only(
    field: str,
    invalid: float,
):
    panel = _panel(
        rows=WINDOW * 2 + 3,
        pool_stocks=MIN_MARKET_STOCKS + 1,
        outside_stocks=0,
    )
    bad_row = 60
    panel[field][bad_row, 0] = invalid
    actual = Completed6030PoolBeta60Strict().calc_batch(panel)
    assert np.isnan(
        actual[bad_row + 1 : bad_row + WINDOW + 1, 0]
    ).all()
    assert np.isfinite(actual[bad_row + WINDOW + 1, 0])
    assert np.isfinite(actual[WINDOW:, 1:]).all()


def test_zero_market_variance_has_no_beta_fallback():
    rows, stocks = WINDOW + 4, MIN_MARKET_STOCKS + 5
    pre_close = np.full((rows, stocks), 10.0, dtype=np.float64)
    close = pre_close * 1.01
    actual = Completed6030PoolBeta60Strict().calc_batch(
        {
            "close": close,
            "preClose": pre_close,
            "stock_codes": _codes(stocks, 0),
        }
    )
    assert np.isnan(actual).all()


def test_randomized_missing_panels_match_independent_brute_force_oracle():
    factor = Completed6030PoolBeta60Strict()
    for seed in range(6):
        rng = np.random.default_rng(603060 + seed)
        rows = int(rng.integers(WINDOW + 2, WINDOW * 2 + 40))
        pool_stocks = int(rng.integers(30, 39))
        outside_stocks = int(rng.integers(1, 6))
        panel = _panel(rows, pool_stocks, outside_stocks)
        locations = rng.choice(
            rows * (pool_stocks + outside_stocks),
            size=min(45, rows),
            replace=False,
        )
        invalid_values = (np.nan, np.inf, -np.inf, 0.0, -1.0)
        for ordinal, location in enumerate(locations):
            row, stock = divmod(
                int(location),
                pool_stocks + outside_stocks,
            )
            field = ("close", "preClose")[ordinal % 2]
            panel[field][row, stock] = invalid_values[
                ordinal % len(invalid_values)
            ]

        actual = factor.calc_batch(panel)
        expected, expected_market = _oracle(panel)
        actual_market = _compute_6030_market_returns(
            panel["close"],
            panel["preClose"],
            panel["stock_codes"],
        )
        np.testing.assert_array_equal(
            np.isfinite(actual_market),
            np.isfinite(expected_market),
        )
        np.testing.assert_allclose(
            actual_market,
            expected_market,
            rtol=0.0,
            atol=2e-16,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            np.isfinite(actual),
            np.isfinite(expected),
        )
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2e-6,
            atol=2e-7,
            equal_nan=True,
        )


def test_modifying_adding_or_removing_outside_pool_columns_is_inert():
    panel = _panel(rows=181, pool_stocks=33, outside_stocks=5)
    factor = Completed6030PoolBeta60Strict()
    pool = _pool_mask(panel["stock_codes"])
    base_market = _compute_6030_market_returns(
        panel["close"],
        panel["preClose"],
        panel["stock_codes"],
    )
    base_scores = factor.calc_batch(panel)[:, pool]

    modified = _copy_panel(panel)
    outside = ~pool
    modified["close"][:, outside] = np.resize(
        np.asarray([np.nan, np.inf, 0.0, -1.0, 1.0e200]),
        modified["close"][:, outside].shape,
    )
    modified["preClose"][:, outside] = np.resize(
        np.asarray([np.inf, np.nan, -1.0, 0.0, 1.0e-200]),
        modified["preClose"][:, outside].shape,
    )
    modified_market = _compute_6030_market_returns(
        modified["close"],
        modified["preClose"],
        modified["stock_codes"],
    )
    modified_scores = factor.calc_batch(modified)[:, pool]
    np.testing.assert_array_equal(modified_market, base_market)
    _assert_same_raw_scores(modified_scores, base_scores)

    appended_count = 7
    appended = {
        "close": np.concatenate(
            (
                panel["close"],
                np.full(
                    (len(panel["close"]), appended_count),
                    np.nan,
                    dtype=np.float64,
                ),
            ),
            axis=1,
        ),
        "preClose": np.concatenate(
            (
                panel["preClose"],
                np.full(
                    (len(panel["close"]), appended_count),
                    -np.inf,
                    dtype=np.float64,
                ),
            ),
            axis=1,
        ),
        "stock_codes": np.concatenate(
            (
                panel["stock_codes"],
                np.asarray(
                    [f"68{index:04d}.SH" for index in range(appended_count)]
                ),
            )
        ),
    }
    appended_market = _compute_6030_market_returns(
        appended["close"],
        appended["preClose"],
        appended["stock_codes"],
    )
    appended_scores = factor.calc_batch(appended)[:, : len(pool)][:, pool]
    np.testing.assert_array_equal(appended_market, base_market)
    _assert_same_raw_scores(appended_scores, base_scores)

    removed = {
        "close": panel["close"][:, pool],
        "preClose": panel["preClose"][:, pool],
        "stock_codes": panel["stock_codes"][pool],
    }
    removed_market = _compute_6030_market_returns(
        removed["close"],
        removed["preClose"],
        removed["stock_codes"],
    )
    removed_scores = factor.calc_batch(removed)
    np.testing.assert_array_equal(removed_market, base_market)
    _assert_same_raw_scores(removed_scores, base_scores)


def test_current_t_row_and_all_unread_fields_cannot_change_existing_scores():
    panel = _panel(rows=149, pool_stocks=32, outside_stocks=3)
    factor = Completed6030PoolBeta60Strict()
    expected = factor.calc_batch(panel)
    changed = _copy_panel(panel)
    stocks = len(panel["stock_codes"])
    replacements = {
        "close": np.resize(
            np.asarray([np.nan, np.inf, -1.0, 0.0, 1e-200, 1e200]),
            stocks,
        ),
        "preClose": np.resize(
            np.asarray([np.inf, np.nan, 0.0, -1.0, 1e200, 1e-200]),
            stocks,
        ),
        "open": np.linspace(1e-200, 1e200, stocks),
        "high": np.linspace(1e200, 1e-200, stocks),
        "low": np.linspace(-1e10, 1e10, stocks),
        "volume": np.linspace(0.0, 1e200, stocks),
        "amount": np.linspace(1e200, 0.0, stocks),
        "st_mask": np.resize(
            np.asarray([True, False]),
            stocks,
        ),
    }
    for name, values in replacements.items():
        changed[name][-1] = values
    actual = factor.calc_batch(changed)
    np.testing.assert_array_equal(actual, expected)


def test_appending_arbitrary_future_row_preserves_full_prefix_bit_for_bit():
    panel = _panel(rows=137, pool_stocks=32, outside_stocks=3)
    factor = Completed6030PoolBeta60Strict()
    expected = factor.calc_batch(panel)
    extended = {
        "stock_codes": panel["stock_codes"].copy(),
    }
    for name, values in panel.items():
        if name == "stock_codes":
            continue
        future = values[-1:].copy()
        if name == "close":
            future[:] = np.resize(
                np.asarray([np.nan, np.inf, 0.0, -1.0]),
                future.shape,
            )
        elif name == "preClose":
            future[:] = np.resize(
                np.asarray([np.inf, np.nan, -1.0, 0.0]),
                future.shape,
            )
        elif name == "st_mask":
            future[:] = True
        else:
            future[:] = 0
        extended[name] = np.concatenate((values, future), axis=0)
    actual = factor.calc_batch(extended)
    np.testing.assert_array_equal(actual[: len(expected)], expected)


def test_completed_corporate_action_price_rescaling_is_invariant():
    panel = _panel(rows=WINDOW + 7, pool_stocks=32, outside_stocks=3)
    factor = Completed6030PoolBeta60Strict()
    expected_market = _compute_6030_market_returns(
        panel["close"],
        panel["preClose"],
        panel["stock_codes"],
    )
    expected = factor.calc_batch(panel)
    scaled = _copy_panel(panel)
    row_scale = np.linspace(0.2, 5.0, len(panel["stock_codes"]))
    for completed_row in (0, 17, WINDOW - 1, WINDOW + 2):
        scaled["close"][completed_row] *= row_scale
        scaled["preClose"][completed_row] *= row_scale
    actual_market = _compute_6030_market_returns(
        scaled["close"],
        scaled["preClose"],
        scaled["stock_codes"],
    )
    actual = factor.calc_batch(scaled)
    np.testing.assert_allclose(
        actual_market,
        expected_market,
        rtol=0.0,
        atol=2e-16,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-6,
        atol=2e-7,
        equal_nan=True,
    )


def test_only_close_preclose_and_stock_codes_are_accessed():
    class RecordingPanel(dict):
        def __init__(self, values):
            super().__init__(values)
            self.accesses = []

        def __getitem__(self, key):
            self.accesses.append(key)
            return super().__getitem__(key)

    panel = RecordingPanel(_panel(rows=WINDOW + 1))
    Completed6030PoolBeta60Strict().calc_batch(panel)
    assert panel.accesses == ["close", "preClose", "stock_codes"]


def test_missing_shape_and_code_alignment_contracts_fail_loudly():
    panel = _panel(rows=WINDOW + 1)
    for missing in ("close", "preClose", "stock_codes"):
        changed = _copy_panel(panel)
        changed.pop(missing)
        with pytest.raises(KeyError):
            Completed6030PoolBeta60Strict().calc_batch(changed)

    changed = _copy_panel(panel)
    changed["close"] = changed["close"][:, 0]
    with pytest.raises(ValueError, match="two-dimensional"):
        Completed6030PoolBeta60Strict().calc_batch(changed)

    changed = _copy_panel(panel)
    changed["preClose"] = changed["preClose"][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        Completed6030PoolBeta60Strict().calc_batch(changed)

    changed = _copy_panel(panel)
    changed["stock_codes"] = changed["stock_codes"][:-1]
    with pytest.raises(ValueError, match="align"):
        Completed6030PoolBeta60Strict().calc_batch(changed)

    changed = _copy_panel(panel)
    changed["stock_codes"] = changed["stock_codes"][:, None]
    with pytest.raises(ValueError, match="one-dimensional"):
        Completed6030PoolBeta60Strict().calc_batch(changed)


def test_factor_source_has_no_per_stock_loop_or_forbidden_fill_path():
    source_path = Path(
        inspect.getsourcefile(Completed6030PoolBeta60Strict)
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loop_targets = {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.comprehension))
        and isinstance(node.target, ast.Name)
    }
    assert not (
        loop_targets
        & {"stock", "stocks", "column", "columns", "stock_index"}
    )
    assert "range(stocks)" not in source
    assert "np.nan_to_num" not in source
    assert "np.clip" not in source
    assert "sliding_window" not in source
    assert "panel.get(" not in source
    assert "try:" not in source


def test_vectorized_large_panel_performance_smoke():
    rows, stocks = 1601, 1200
    row = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    pre_close = 8.0 + 0.001 * row + 0.0001 * stock
    returns = (
        0.002 * np.sin(row / 17.0 + stock / 31.0)
        + 0.00001 * np.cos(row / 7.0 - stock / 13.0)
    )
    close = pre_close * (1.0 + returns)
    codes = _codes(pool_stocks=900, outside_stocks=300)

    started = time.perf_counter()
    actual = Completed6030PoolBeta60Strict().calc_batch(
        {
            "close": close,
            "preClose": pre_close,
            "stock_codes": codes,
        }
    )
    elapsed = time.perf_counter() - started
    print(f"large-panel factor elapsed: {elapsed:.3f}s")
    assert actual.shape == (rows, stocks)
    assert np.isfinite(actual[WINDOW:]).all()
    assert elapsed < 6.0
