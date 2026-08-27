from __future__ import annotations

import ast
import gc
import inspect
import time
from pathlib import Path

import numpy as np
import pytest

from factor_db.factors.CompletedTrendConsistency60Strict import (
    CompletedTrendConsistency60Strict,
)


WINDOW = 60
RUNTIME_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "runtime"
    / "runtime_1990-12-19_2026-07-24.npz"
)


def _panel(rows: int = 137, stocks: int = 6) -> dict:
    row = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    pre_close = 8.0 + 0.017 * row + 0.3 * stock
    log_return = (
        0.002
        + 0.003 * np.sin(row / 13.0 + stock / 7.0)
        - 0.0001 * stock
    )
    close = pre_close * np.exp(log_return)
    return {
        "close": close,
        "preClose": pre_close,
        "open": pre_close * 1.001,
        "high": np.maximum(close, pre_close) * 1.01,
        "low": np.minimum(close, pre_close) * 0.99,
        "volume": np.full(close.shape, 2.0e6, dtype=np.float64),
        "amount": np.full(close.shape, 2.0e7, dtype=np.float64),
        "st_mask": np.zeros(close.shape, dtype=bool),
    }


def _copy_panel(panel: dict) -> dict:
    return {name: value.copy() for name, value in panel.items()}


def _oracle(panel: dict) -> np.ndarray:
    close = np.asarray(panel["close"], dtype=np.float64)
    pre_close = np.asarray(panel["preClose"], dtype=np.float64)
    output = np.full(close.shape, np.nan, dtype=np.float32)
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
            log_return = (
                np.log(completed_close) - np.log(completed_pre_close)
            )
            energy = float(np.sum(log_return * log_return))
            if energy <= 0.0:
                continue
            score = float(np.sum(log_return)) / np.sqrt(WINDOW * energy)
            if np.isfinite(score) and score > 0.0:
                output[row, stock] = score
    return output


def test_metadata_and_formula_match_independent_oracle():
    factor = CompletedTrendConsistency60Strict()
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


def test_window_boundary_and_one_row_shift_are_exact():
    panel = _panel(rows=WINDOW + 2, stocks=2)
    factor = CompletedTrendConsistency60Strict()
    baseline = factor.calc_batch(panel)
    assert np.isnan(baseline[:WINDOW]).all()
    assert np.isfinite(baseline[WINDOW:]).all()

    changed = _copy_panel(panel)
    changed["close"][0, 0] *= 1.2
    actual = factor.calc_batch(changed)
    assert actual[WINDOW, 0] != baseline[WINDOW, 0]
    np.testing.assert_array_equal(
        actual[WINDOW + 1, 0],
        baseline[WINDOW + 1, 0],
    )


def test_current_row_hlcva_changes_cannot_change_row_t():
    panel = _panel(rows=121, stocks=6)
    factor = CompletedTrendConsistency60Strict()
    row = 83
    expected = factor.calc_batch(panel)[row].copy()
    changed = _copy_panel(panel)
    replacements = {
        "high": np.asarray([np.nan, np.inf, -1.0, 0.0, 1e-200, 1e200]),
        "low": np.asarray([np.inf, np.nan, 0.0, -1.0, 1e200, 1e-200]),
        "close": np.asarray([np.nan, np.inf, -1.0, 0.0, 1e-200, 1e200]),
        "volume": np.linspace(-1e200, 1e200, 6),
        "amount": np.linspace(1e200, -1e200, 6),
    }
    for field, values in replacements.items():
        changed[field][row] = values
    actual = factor.calc_batch(changed)[row]
    np.testing.assert_array_equal(actual, expected)


def test_last_completed_close_changes_row_t():
    panel = _panel(rows=WINDOW + 2, stocks=3)
    factor = CompletedTrendConsistency60Strict()
    expected = factor.calc_batch(panel)[WINDOW].copy()
    panel["close"][WINDOW - 1, 0] *= 0.8
    actual = factor.calc_batch(panel)[WINDOW]
    assert actual[0] != expected[0]
    np.testing.assert_array_equal(actual[1:], expected[1:])


@pytest.mark.parametrize("field", ("close", "preClose"))
@pytest.mark.parametrize(
    "invalid",
    (np.nan, np.inf, -np.inf, 0.0, -1.0),
)
def test_invalid_pair_poisons_every_exact_window(
    field: str,
    invalid: float,
):
    bad_row = 70
    panel = _panel(rows=bad_row + WINDOW + 2, stocks=2)
    panel[field][bad_row, 0] = invalid
    actual = CompletedTrendConsistency60Strict().calc_batch(panel)
    assert np.isnan(actual[bad_row + 1 : bad_row + WINDOW + 1, 0]).all()
    assert np.isfinite(actual[bad_row + WINDOW + 1, 0])
    assert np.isfinite(actual[WINDOW:, 1]).all()


def test_non_positive_or_zero_energy_trend_is_nan():
    rows = WINDOW + 1
    pre_close = np.full((rows, 4), 10.0, dtype=np.float64)
    daily = np.asarray([0.01, -0.01, 0.0, -0.01])[None, :]
    close = pre_close * np.exp(daily)
    actual = CompletedTrendConsistency60Strict().calc_batch(
        {"close": close, "preClose": pre_close}
    )[WINDOW]
    assert actual[0] == pytest.approx(1.0)
    assert np.isnan(actual[1:]).all()


def test_only_close_and_official_preclose_are_accessed():
    class RecordingPanel(dict):
        def __init__(self, values):
            super().__init__(values)
            self.accesses = []

        def __getitem__(self, key):
            self.accesses.append(key)
            return super().__getitem__(key)

    panel = RecordingPanel(_panel(rows=WINDOW + 1, stocks=2))
    CompletedTrendConsistency60Strict().calc_batch(panel)
    assert panel.accesses == ["close", "preClose"]


def test_missing_dimension_and_shape_errors_fail_loudly():
    panel = _panel(rows=WINDOW + 1, stocks=2)
    for missing in ("close", "preClose"):
        changed = _copy_panel(panel)
        changed.pop(missing)
        with pytest.raises(KeyError):
            CompletedTrendConsistency60Strict().calc_batch(changed)

    changed = _copy_panel(panel)
    changed["close"] = changed["close"][:, 0]
    with pytest.raises(ValueError, match="two-dimensional"):
        CompletedTrendConsistency60Strict().calc_batch(changed)

    changed = _copy_panel(panel)
    changed["preClose"] = changed["preClose"][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        CompletedTrendConsistency60Strict().calc_batch(changed)


def test_extreme_valid_price_pairs_preserve_mathematical_log_return():
    rows = WINDOW + 1
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    largest = np.finfo(np.float64).max
    close = np.empty((rows, 4), dtype=np.float64)
    pre_close = np.empty_like(close)
    close[:, 0] = largest
    pre_close[:, 0] = smallest
    close[:, 1] = smallest
    pre_close[:, 1] = largest
    close[:, 2] = largest
    pre_close[:, 2] = largest
    close[:, 3] = smallest
    pre_close[:, 3] = smallest
    actual = CompletedTrendConsistency60Strict().calc_batch(
        {"close": close, "preClose": pre_close}
    )[WINDOW]
    assert actual[0] == pytest.approx(1.0)
    assert np.isnan(actual[1:]).all()


def test_source_has_no_stock_loop_fill_or_forbidden_data_dependency():
    source_path = Path(
        inspect.getsourcefile(CompletedTrendConsistency60Strict)
    )
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


def test_full_runtime_calc_batch_median_is_under_one_second():
    assert RUNTIME_PATH.exists()
    with np.load(RUNTIME_PATH) as runtime:
        panel = {
            "close": runtime["close"],
            "preClose": runtime["preClose"],
        }
        elapsed = []
        for _ in range(3):
            started = time.perf_counter()
            output = CompletedTrendConsistency60Strict().calc_batch(panel)
            elapsed.append(time.perf_counter() - started)
            assert output.shape == panel["close"].shape
            assert output.dtype == np.float32
            del output
            gc.collect()

    assert float(np.median(elapsed)) < 1.0, elapsed
