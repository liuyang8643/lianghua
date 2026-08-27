from __future__ import annotations

import ast
import gc
import inspect
import time
from pathlib import Path

import numpy as np
import pytest

from factor_db.factors.CompletedSmallCapTrendConsistency60Strict import (
    CompletedSmallCapTrendConsistency60Strict,
)


WINDOW = 60
PIVOT_RMB = 1.0e10
RUNTIME_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "runtime"
    / "runtime_1990-12-19_2026-07-24.npz"
)


def _panel(rows: int = 143, stocks: int = 7) -> dict:
    row = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    log_return = (
        0.0025 * np.sin((row + 1.0) / 9.0 + stock / 5.0)
        + 0.0004 * (stock - (stocks - 1.0) / 2.0)
    )
    pre_close = 8.0 + 0.015 * row + 0.7 * stock
    close = pre_close * np.exp(log_return)
    total_share = np.broadcast_to(
        1.5e4 + 2.0e3 * stock,
        close.shape,
    ).copy()
    return {
        "close": close,
        "preClose": pre_close,
        "total_share": total_share,
        "open": pre_close * 1.001,
        "high": np.maximum(close, pre_close) * 1.01,
        "low": np.minimum(close, pre_close) * 0.99,
        "volume": np.full(close.shape, 2.0e6, dtype=np.float64),
        "amount": np.full(close.shape, 2.0e7, dtype=np.float64),
        "st_mask": np.zeros(close.shape, dtype=bool),
    }


def _copy_panel(panel: dict) -> dict:
    return {
        name: value.copy() if hasattr(value, "copy") else value
        for name, value in panel.items()
    }


def _oracle(panel: dict) -> np.ndarray:
    close = np.asarray(panel["close"], dtype=np.float64)
    pre_close = np.asarray(panel["preClose"], dtype=np.float64)
    total_share = np.asarray(panel["total_share"], dtype=np.float64)
    result = np.full(close.shape, np.nan, dtype=np.float32)
    for row in range(WINDOW, close.shape[0]):
        for stock in range(close.shape[1]):
            completed_close = close[row - WINDOW : row, stock]
            completed_pre_close = pre_close[row - WINDOW : row, stock]
            if not (
                np.all(np.isfinite(completed_close))
                and np.all(completed_close > 0.0)
                and np.all(np.isfinite(completed_pre_close))
                and np.all(completed_pre_close > 0.0)
            ):
                continue
            last_share = total_share[row - 1, stock]
            if not np.isfinite(last_share) or last_share <= 0.0:
                continue
            log_returns = np.log(completed_close) - np.log(
                completed_pre_close
            )
            squared_sum = float(np.sum(log_returns * log_returns))
            trend = (
                0.0
                if squared_sum == 0.0
                else float(np.sum(log_returns))
                / np.sqrt(WINDOW * squared_sum)
            )
            cap = completed_close[-1] * last_share * 1.0e4
            smallness = 1.0 / (1.0 + cap / PIVOT_RMB)
            result[row, stock] = trend * smallness
    return result


def test_metadata_formula_and_independent_oracle_match():
    factor = CompletedSmallCapTrendConsistency60Strict()
    assert factor.hist_days == WINDOW
    assert factor.update_frequency == "daily"
    assert factor.pre_ranked is False
    assert factor.requires_full_history is False

    panel = _panel()
    actual = factor.calc_batch(panel)
    expected = _oracle(panel)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-6,
        atol=2e-7,
        equal_nan=True,
    )


def test_first_valid_row_is_60_and_window_is_exactly_completed_rows():
    panel = _panel(rows=62, stocks=3)
    factor = CompletedSmallCapTrendConsistency60Strict()
    actual = factor.calc_batch(panel)
    assert np.isnan(actual[:WINDOW]).all()
    assert np.isfinite(actual[WINDOW:]).all()

    changed = _copy_panel(panel)
    changed["close"][0, 0] *= 1.1
    changed_actual = factor.calc_batch(changed)
    assert changed_actual[WINDOW, 0] != actual[WINDOW, 0]
    np.testing.assert_array_equal(
        changed_actual[WINDOW + 1, 0],
        actual[WINDOW + 1, 0],
    )


def test_positive_and_negative_trend_interact_with_size_in_opposite_order():
    rows, stocks = WINDOW + 1, 4
    pre_close = np.full((rows, stocks), 10.0, dtype=np.float64)
    signs = np.asarray([1.0, 1.0, -1.0, -1.0])[None, :]
    close = pre_close * np.exp(0.01 * signs)
    target_caps = np.asarray([1.0e9, 1.0e11, 1.0e9, 1.0e11])
    total_share = np.broadcast_to(
        target_caps[None, :] / close / 1.0e4,
        close.shape,
    ).copy()
    actual = CompletedSmallCapTrendConsistency60Strict().calc_batch(
        {
            "close": close,
            "preClose": pre_close,
            "total_share": total_share,
        }
    )[WINDOW]

    expected_smallness = PIVOT_RMB / (PIVOT_RMB + target_caps)
    np.testing.assert_allclose(
        actual,
        np.asarray(
            [
                expected_smallness[0],
                expected_smallness[1],
                -expected_smallness[2],
                -expected_smallness[3],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
    )
    assert actual[0] > actual[1] > 0.0
    assert actual[2] < actual[3] < 0.0


def test_runtime_total_share_unit_places_100yi_cap_at_half_smallness():
    rows = WINDOW + 1
    pre_close = np.full((rows, 1), 10.0, dtype=np.float64)
    close = pre_close * np.exp(0.01)
    total_share = np.full(
        (rows, 1),
        PIVOT_RMB / (close[-1, 0] * 1.0e4),
        dtype=np.float64,
    )

    actual = CompletedSmallCapTrendConsistency60Strict().calc_batch(
        {
            "close": close,
            "preClose": pre_close,
            "total_share": total_share,
        }
    )[WINDOW, 0]

    np.testing.assert_allclose(actual, 0.5, rtol=1e-6)
    wrong_share_unit_smallness = 1.0 / (
        1.0 + close[-1, 0] * total_share[-1, 0] / PIVOT_RMB
    )
    assert wrong_share_unit_smallness > 0.999


def test_exactly_flat_official_return_window_is_neutral_not_missing():
    panel = _panel(rows=WINDOW + 2, stocks=5)
    panel["close"][:] = panel["preClose"]
    actual = CompletedSmallCapTrendConsistency60Strict().calc_batch(panel)
    assert np.isnan(actual[:WINDOW]).all()
    np.testing.assert_array_equal(
        actual[WINDOW:],
        np.zeros_like(actual[WINDOW:]),
    )


@pytest.mark.parametrize("field", ["close", "preClose"])
@pytest.mark.parametrize(
    "invalid",
    [np.nan, np.inf, -np.inf, 0.0, -1.0],
)
def test_invalid_official_return_input_poisons_every_affected_window(
    field: str,
    invalid: float,
):
    panel = _panel(rows=WINDOW * 2 + 3, stocks=2)
    bad_row = 60
    panel[field][bad_row, 0] = invalid
    actual = CompletedSmallCapTrendConsistency60Strict().calc_batch(panel)
    assert np.isnan(actual[bad_row + 1 : bad_row + WINDOW + 1, 0]).all()
    assert np.isfinite(actual[bad_row + WINDOW + 1, 0])
    assert np.isfinite(actual[WINDOW:, 1]).all()


@pytest.mark.parametrize(
    "invalid",
    [np.nan, np.inf, -np.inf, 0.0, -1.0],
)
def test_last_completed_share_is_strict_and_is_not_filled(
    invalid: float,
):
    panel = _panel(rows=WINDOW + 3, stocks=2)
    panel["total_share"][WINDOW - 1, 0] = invalid
    actual = CompletedSmallCapTrendConsistency60Strict().calc_batch(panel)
    assert np.isnan(actual[WINDOW, 0])
    assert np.isfinite(actual[WINDOW + 1, 0])
    assert np.isfinite(actual[WINDOW:, 1]).all()


def test_current_row_all_fields_cannot_change_any_existing_output():
    panel = _panel(rows=149, stocks=6)
    factor = CompletedSmallCapTrendConsistency60Strict()
    expected = factor.calc_batch(panel)
    changed = _copy_panel(panel)
    replacements = {
        "close": np.asarray([np.nan, np.inf, -1.0, 0.0, 1e-200, 1e200]),
        "preClose": np.asarray([np.inf, np.nan, 0.0, -1.0, 1e200, 1e-200]),
        "total_share": np.asarray([0.0, -1.0, np.nan, np.inf, 1e-200, 1e200]),
        "open": np.linspace(1e-200, 1e200, 6),
        "high": np.linspace(1e200, 1e-200, 6),
        "low": np.linspace(-1e10, 1e10, 6),
        "volume": np.linspace(0.0, 1e200, 6),
        "amount": np.linspace(1e200, 0.0, 6),
        "st_mask": np.asarray([True, False, True, False, True, False]),
    }
    for name, values in replacements.items():
        changed[name][-1] = values

    actual = factor.calc_batch(changed)
    np.testing.assert_array_equal(actual, expected)


def test_appending_future_rows_cannot_change_any_existing_output():
    panel = _panel(rows=137, stocks=5)
    factor = CompletedSmallCapTrendConsistency60Strict()
    expected = factor.calc_batch(panel)
    extended = {}
    for name, values in panel.items():
        future = values[-1:].copy()
        if name == "close":
            future[:] = np.linspace(1e-100, 1e100, 5)
        elif name == "preClose":
            future[:] = np.linspace(1e100, 1e-100, 5)
        elif name == "total_share":
            future[:] = np.linspace(1e-100, 1e100, 5)
        elif name == "st_mask":
            future[:] = True
        else:
            future[:] = 0
        extended[name] = np.concatenate((values, future), axis=0)

    actual = factor.calc_batch(extended)
    np.testing.assert_array_equal(actual[: len(expected)], expected)


def test_completed_corporate_action_rescaling_is_invariant():
    panel = _panel(rows=WINDOW + 2, stocks=4)
    factor = CompletedSmallCapTrendConsistency60Strict()
    expected = factor.calc_batch(panel)[WINDOW]
    scaled = _copy_panel(panel)
    scale = np.asarray([0.25, 0.5, 2.0, 4.0])
    scaled["close"][WINDOW - 1] *= scale
    scaled["preClose"][WINDOW - 1] *= scale
    scaled["total_share"][WINDOW - 1] /= scale
    actual = factor.calc_batch(scaled)[WINDOW]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


def test_only_three_preregistered_panel_fields_are_accessed():
    class RecordingPanel(dict):
        def __init__(self, values):
            super().__init__(values)
            self.accesses = []

        def __getitem__(self, key):
            self.accesses.append(key)
            return super().__getitem__(key)

    panel = RecordingPanel(_panel(rows=WINDOW + 1, stocks=2))
    CompletedSmallCapTrendConsistency60Strict().calc_batch(panel)
    assert panel.accesses == ["close", "preClose", "total_share"]


def test_missing_shape_and_dimensional_contracts_fail_loudly():
    panel = _panel(rows=WINDOW + 1, stocks=2)
    for missing in ("close", "preClose", "total_share"):
        changed = _copy_panel(panel)
        changed.pop(missing)
        with pytest.raises(KeyError):
            CompletedSmallCapTrendConsistency60Strict().calc_batch(changed)

    changed = _copy_panel(panel)
    changed["close"] = changed["close"][:, 0]
    with pytest.raises(ValueError, match="two-dimensional"):
        CompletedSmallCapTrendConsistency60Strict().calc_batch(changed)

    changed = _copy_panel(panel)
    changed["total_share"] = changed["total_share"][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        CompletedSmallCapTrendConsistency60Strict().calc_batch(changed)


def test_float64_extremes_remain_finite_without_clipping():
    rows, stocks = WINDOW + 1, 4
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    largest = np.finfo(np.float64).max
    close = np.empty((rows, stocks), dtype=np.float64)
    pre_close = np.empty_like(close)
    close[:, 0] = largest
    pre_close[:, 0] = smallest
    close[:, 1] = smallest
    pre_close[:, 1] = largest
    close[:, 2] = largest
    pre_close[:, 2] = largest
    close[:, 3] = smallest
    pre_close[:, 3] = smallest
    total_share = np.broadcast_to(
        np.asarray([smallest, largest, largest, smallest])[None, :],
        close.shape,
    ).copy()

    actual = CompletedSmallCapTrendConsistency60Strict().calc_batch(
        {
            "close": close,
            "preClose": pre_close,
            "total_share": total_share,
        }
    )[WINDOW]
    assert np.isfinite(actual).all()
    assert actual.dtype == np.float32
    assert np.all(actual >= -1.0)
    assert np.all(actual <= 1.0)
    assert actual[0] > 0.0
    assert actual[1] <= 0.0
    assert actual[2] == 0.0
    assert actual[3] == 0.0


def test_factor_source_has_no_per_stock_loop_or_forbidden_fill_path():
    source_path = Path(inspect.getsourcefile(
        CompletedSmallCapTrendConsistency60Strict
    ))
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loop_targets = {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.comprehension))
        and isinstance(node.target, ast.Name)
    }
    assert not (loop_targets & {"stock", "stocks", "column", "columns"})
    assert "range(stocks)" not in source
    assert "nan_to_num" not in source
    assert "sliding_window" not in source
    assert "panel.get(" not in source
    assert "try:" not in source
    assert "_SIZE_PIVOT_RMB = 1.0e10" in source
    assert "not a searched parameter" in source


def test_full_runtime_calc_batch_median_is_under_one_second():
    assert RUNTIME_PATH.exists()
    with np.load(RUNTIME_PATH) as runtime:
        panel = {
            "close": runtime["close"],
            "preClose": runtime["preClose"],
            "total_share": runtime["total_share"],
        }
        factor = CompletedSmallCapTrendConsistency60Strict()
        warmup = factor.calc_batch(panel)
        assert warmup.shape == panel["close"].shape
        del warmup
        gc.collect()

        elapsed = []
        for _ in range(3):
            started = time.perf_counter()
            output = factor.calc_batch(panel)
            elapsed.append(time.perf_counter() - started)
            assert output.shape == panel["close"].shape
            assert output.dtype == np.float32
            del output
            gc.collect()

    assert float(np.median(elapsed)) < 1.0, elapsed
