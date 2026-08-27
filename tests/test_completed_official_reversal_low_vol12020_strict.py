from __future__ import annotations

import ast
import inspect
import time
from pathlib import Path

import numpy as np
import pytest

from factor_db.factors.CompletedOfficialReversalLowVol12020Strict import (
    CompletedOfficialReversalLowVol12020Strict,
)


WINDOW = 120
VOL_WINDOW = 20


def _panel(rows: int = 287, stocks: int = 7) -> dict:
    row = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    pre_close = 7.0 + 0.013 * row + 0.61 * stock
    daily_return = (
        0.012 * np.sin(row / 13.0 + stock / 7.0)
        + 0.003 * np.cos(row / 29.0 - stock / 11.0)
    )
    close = pre_close * (1.0 + daily_return)
    shape = close.shape
    return {
        "close": close,
        "preClose": pre_close,
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


def _oracle(panel: dict) -> np.ndarray:
    close = np.asarray(panel["close"], dtype=np.float64)
    pre_close = np.asarray(panel["preClose"], dtype=np.float64)
    expected = np.full(close.shape, np.nan, dtype=np.float32)
    for row in range(WINDOW, close.shape[0]):
        for stock in range(close.shape[1]):
            completed_close = close[row - WINDOW : row, stock]
            completed_pre_close = pre_close[
                row - WINDOW : row,
                stock,
            ]
            if not (
                np.all(np.isfinite(completed_close))
                and np.all(completed_close > 0.0)
                and np.all(np.isfinite(completed_pre_close))
                and np.all(completed_pre_close > 0.0)
            ):
                continue
            with np.errstate(
                divide="ignore",
                invalid="ignore",
                over="ignore",
                under="ignore",
            ):
                daily_return = (
                    completed_close / completed_pre_close - 1.0
                )
                one_plus_return = 1.0 + daily_return
                log_return = np.log1p(daily_return)
                squared_return = daily_return * daily_return
            if not (
                np.all(np.isfinite(daily_return))
                and np.all(np.isfinite(one_plus_return))
                and np.all(one_plus_return > 0.0)
                and np.all(np.isfinite(log_return))
                and np.all(np.isfinite(squared_return))
            ):
                continue
            with np.errstate(
                invalid="ignore",
                over="ignore",
                under="ignore",
            ):
                momentum120 = (
                    np.exp(np.sum(log_return, dtype=np.float64)) - 1.0
                )
                sigma20 = np.sqrt(
                    np.mean(
                        squared_return[-VOL_WINDOW:],
                        dtype=np.float64,
                    )
                )
                score = (
                    0.25
                    * np.clip(
                        (0.40 - momentum120) / 0.80,
                        0.0,
                        1.0,
                    )
                    + 0.75
                    * np.clip(
                        (0.06 - sigma20) / 0.05,
                        0.0,
                        1.0,
                    )
                )
            if np.isfinite(momentum120) and np.isfinite(sigma20) and np.isfinite(score):
                expected[row, stock] = score
    return expected


def test_metadata_fixed_formula_and_independent_oracle_match():
    factor = CompletedOfficialReversalLowVol12020Strict()
    assert factor.hist_days == WINDOW
    assert factor.update_frequency == "daily"
    assert factor.pre_ranked is False
    assert factor.requires_full_history is False

    panel = _panel()
    actual = factor.calc_batch(panel)
    expected = _oracle(panel)

    assert actual.dtype == np.float32
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


def test_first_valid_row_and_both_window_boundaries_are_exact():
    rows = WINDOW + 3
    pre_close = np.full((rows, 2), 10.0, dtype=np.float64)
    close = pre_close.copy()
    close[:WINDOW, 0] *= 1.001
    close[: WINDOW - VOL_WINDOW, 1] *= 1.002
    panel = {"close": close, "preClose": pre_close}

    actual = CompletedOfficialReversalLowVol12020Strict().calc_batch(panel)
    expected = _oracle(panel)

    assert np.isnan(actual[:WINDOW]).all()
    assert np.isfinite(actual[WINDOW:]).all()
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-6,
        atol=2e-7,
        equal_nan=True,
    )

    changed = _copy_panel(panel)
    changed["close"][0, 0] *= 1.1
    shifted = CompletedOfficialReversalLowVol12020Strict().calc_batch(
        changed
    )
    assert shifted[WINDOW, 0] != actual[WINDOW, 0]
    np.testing.assert_array_equal(
        shifted[WINDOW + 1 :, 0],
        actual[WINDOW + 1 :, 0],
    )


def test_modifying_output_row_cannot_affect_its_score():
    panel = _panel(rows=251, stocks=6)
    factor = CompletedOfficialReversalLowVol12020Strict()
    expected = factor.calc_batch(panel)
    row = 207
    changed = _copy_panel(panel)
    changed["close"][row] = np.asarray(
        [np.nan, np.inf, -1.0, 0.0, 1e-250, 1e250]
    )
    changed["preClose"][row] = np.asarray(
        [np.inf, np.nan, 0.0, -1.0, 1e250, 1e-250]
    )
    actual = factor.calc_batch(changed)

    np.testing.assert_array_equal(actual[: row + 1], expected[: row + 1])


def test_appending_arbitrary_future_rows_preserves_prefix_bit_exactly():
    panel = _panel(rows=277, stocks=5)
    factor = CompletedOfficialReversalLowVol12020Strict()
    expected = factor.calc_batch(panel)
    future_rows = 91
    rng = np.random.default_rng(12020)
    future_pre_close = rng.lognormal(
        2.0,
        1.5,
        size=(future_rows, 5),
    )
    future_close = future_pre_close * rng.lognormal(
        0.0,
        0.5,
        size=(future_rows, 5),
    )
    future_close[0] = [np.nan, np.inf, 0.0, -1.0, 1e300]
    future_pre_close[0] = [1.0, 1.0, 1.0, 1.0, 1e-300]
    extended = {
        "close": np.concatenate((panel["close"], future_close), axis=0),
        "preClose": np.concatenate(
            (panel["preClose"], future_pre_close),
            axis=0,
        ),
    }

    actual = factor.calc_batch(extended)
    np.testing.assert_array_equal(actual[: len(expected)], expected)


@pytest.mark.parametrize("field", ["close", "preClose"])
@pytest.mark.parametrize(
    "invalid",
    [np.nan, np.inf, -np.inf, 0.0, -1.0],
)
def test_one_invalid_completed_return_poisons_exact_affected_windows_only(
    field: str,
    invalid: float,
):
    panel = _panel(rows=WINDOW * 3 + 9, stocks=2)
    bad_row = 151
    panel[field][bad_row, 0] = invalid
    actual = CompletedOfficialReversalLowVol12020Strict().calc_batch(
        panel
    )

    assert np.isfinite(actual[WINDOW : bad_row + 1, 0]).all()
    assert np.isnan(
        actual[bad_row + 1 : bad_row + WINDOW + 1, 0]
    ).all()
    assert np.isfinite(actual[bad_row + WINDOW + 1 :, 0]).all()
    assert np.isfinite(actual[WINDOW:, 1]).all()


def test_simultaneous_close_and_preclose_scaling_is_invariant():
    panel = _panel(rows=263, stocks=6)
    factor = CompletedOfficialReversalLowVol12020Strict()
    expected = factor.calc_batch(panel)
    scaled = _copy_panel(panel)
    row_scale = np.power(
        2.0,
        (np.arange(len(expected), dtype=np.int64) % 17) - 8,
    )[:, None]
    stock_scale = np.power(
        2.0,
        np.arange(6, dtype=np.int64) - 3,
    )[None, :]
    scale = row_scale * stock_scale
    scaled["close"] *= scale
    scaled["preClose"] *= scale
    actual = factor.calc_batch(scaled)

    np.testing.assert_array_equal(actual, expected)


def test_nonpositive_and_float64_extremes_are_strictly_missing_then_recover():
    rows, stocks = WINDOW + 2, 7
    close = np.full((rows, stocks), 10.0, dtype=np.float64)
    pre_close = np.full((rows, stocks), 10.0, dtype=np.float64)
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    largest = np.finfo(np.float64).max
    close[0, 1] = 0.0
    close[0, 2] = -1.0
    pre_close[0, 3] = 0.0
    pre_close[0, 4] = -1.0
    close[0, 5], pre_close[0, 5] = largest, smallest
    close[0, 6], pre_close[0, 6] = smallest, largest

    actual = CompletedOfficialReversalLowVol12020Strict().calc_batch(
        {"close": close, "preClose": pre_close}
    )

    assert np.isfinite(actual[WINDOW, 0])
    assert np.isnan(actual[WINDOW, 1:]).all()
    assert np.isfinite(actual[WINDOW + 1]).all()


def test_only_close_and_preclose_are_accessed():
    class RecordingPanel(dict):
        def __init__(self, values):
            super().__init__(values)
            self.accesses = []

        def __getitem__(self, key):
            self.accesses.append(key)
            return super().__getitem__(key)

    panel = RecordingPanel(_panel())
    CompletedOfficialReversalLowVol12020Strict().calc_batch(panel)
    assert panel.accesses == ["close", "preClose"]


@pytest.mark.parametrize("rows", [0, 1, 19, 20, 119, 120])
def test_short_panels_are_entirely_missing(rows: int):
    panel = _panel(rows=rows, stocks=4)
    actual = CompletedOfficialReversalLowVol12020Strict().calc_batch(
        panel
    )
    assert actual.shape == (rows, 4)
    assert actual.dtype == np.float32
    assert np.isnan(actual).all()


def test_missing_shape_and_dimensional_contracts_fail_loudly():
    panel = _panel(rows=WINDOW + 1, stocks=2)
    for missing in ("close", "preClose"):
        changed = _copy_panel(panel)
        changed.pop(missing)
        with pytest.raises(KeyError, match=missing):
            CompletedOfficialReversalLowVol12020Strict().calc_batch(
                changed
            )

    changed = _copy_panel(panel)
    changed["close"] = changed["close"][:, 0]
    with pytest.raises(ValueError, match="two-dimensional"):
        CompletedOfficialReversalLowVol12020Strict().calc_batch(changed)

    changed = _copy_panel(panel)
    changed["preClose"] = changed["preClose"][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        CompletedOfficialReversalLowVol12020Strict().calc_batch(changed)


def test_source_has_no_per_stock_loop_fill_tolerance_or_fallback():
    source_path = Path(
        inspect.getsourcefile(
            CompletedOfficialReversalLowVol12020Strict
        )
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
        & {"stock", "stocks", "column", "columns", "security", "securities"}
    )
    assert "range(stocks)" not in source
    assert "nan_to_num" not in source
    assert "try:" not in source
    assert "panel.get(" not in source
    assert "tolerance" not in source.lower()
    assert "fallback" not in source.lower()


def test_vectorized_large_panel_completes_far_below_six_seconds():
    rows, stocks = 1801, 1200
    row = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    pre_close = 8.0 + 0.001 * row + 0.0001 * stock
    close = pre_close * (
        1.0 + 0.015 * np.sin(row / 17.0 + stock / 31.0)
    )

    started = time.perf_counter()
    actual = CompletedOfficialReversalLowVol12020Strict().calc_batch(
        {"close": close, "preClose": pre_close}
    )
    elapsed = time.perf_counter() - started

    assert actual.shape == (rows, stocks)
    assert actual.dtype == np.float32
    assert np.isnan(actual[:WINDOW]).all()
    assert np.isfinite(actual[WINDOW:]).all()
    assert elapsed < 6.0
