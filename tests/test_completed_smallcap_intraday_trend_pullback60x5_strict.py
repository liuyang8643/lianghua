from __future__ import annotations

import ast
import gc
import inspect
import time
from pathlib import Path

import numpy as np
import pytest

from factor_db.factors.CompletedSmallCapIntradayTrendPullback60x5Strict import (
    CompletedSmallCapIntradayTrendPullback60x5Strict,
)


LONG_WINDOW = 60
SHORT_WINDOW = 5
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
    intraday = (
        0.0015
        + 0.003 * np.sin(row / 8.0 + stock / 5.0)
        - 0.00015 * stock
    )
    open_price = 8.0 + 0.013 * row + 0.4 * stock
    close = open_price * np.exp(intraday)
    total_share = np.broadcast_to(
        2.0e4 + 1.5e3 * stock,
        close.shape,
    ).copy()
    pre_close = open_price * np.exp(
        0.04 * np.cos(row / 11.0 + stock / 3.0)
    )
    return {
        "open": open_price,
        "close": close,
        "total_share": total_share,
        "preClose": pre_close,
        "high": np.maximum(open_price, close) * 1.01,
        "low": np.minimum(open_price, close) * 0.99,
        "volume": np.full(close.shape, 2.0e6, dtype=np.float64),
        "amount": np.full(close.shape, 2.0e7, dtype=np.float64),
        "st_mask": np.zeros(close.shape, dtype=bool),
    }


def _copy_panel(panel: dict) -> dict:
    return {name: values.copy() for name, values in panel.items()}


def _oracle(panel: dict) -> np.ndarray:
    open_price = np.asarray(panel["open"], dtype=np.float64)
    close = np.asarray(panel["close"], dtype=np.float64)
    total_share = np.asarray(panel["total_share"], dtype=np.float64)
    output = np.full(close.shape, np.nan, dtype=np.float32)
    for row in range(LONG_WINDOW, close.shape[0]):
        for stock in range(close.shape[1]):
            long_open = open_price[row - LONG_WINDOW : row, stock]
            long_close = close[row - LONG_WINDOW : row, stock]
            last_share = total_share[row - 1, stock]
            valid = (
                np.all(np.isfinite(long_open))
                and np.all(long_open > 0.0)
                and np.all(np.isfinite(long_close))
                and np.all(long_close > 0.0)
                and np.isfinite(last_share)
                and last_share > 0.0
            )
            if not valid:
                continue
            returns = np.log(long_close) - np.log(long_open)
            short_returns = returns[-SHORT_WINDOW:]
            long_energy = float(np.sum(returns * returns))
            short_energy = float(np.sum(short_returns * short_returns))
            q60 = (
                0.0
                if long_energy == 0.0
                else float(np.sum(returns))
                / np.sqrt(LONG_WINDOW * long_energy)
            )
            q5 = (
                0.0
                if short_energy == 0.0
                else float(np.sum(short_returns))
                / np.sqrt(SHORT_WINDOW * short_energy)
            )
            cap = long_close[-1] * last_share * 1.0e4
            smallness = 1.0 / (1.0 + cap / PIVOT_RMB)
            output[row, stock] = (
                smallness * ((1.0 + q60) / 2.0) * ((1.0 - q5) / 2.0)
            )
    return output


def test_metadata_and_randomized_panel_match_independent_oracle():
    factor = CompletedSmallCapIntradayTrendPullback60x5Strict()
    assert factor.hist_days == LONG_WINDOW
    assert factor.update_frequency == "daily"
    assert factor.pre_ranked is False
    assert factor.requires_full_history is False

    panel = _panel()
    actual = factor.calc_batch(panel)
    expected = _oracle(panel)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-6,
        atol=2e-7,
        equal_nan=True,
    )


def test_hand_formula_long_uptrend_and_five_day_pullback():
    rows = LONG_WINDOW + 1
    open_price = np.full((rows, 1), 10.0)
    returns = np.full((rows, 1), 0.01)
    returns[LONG_WINDOW - SHORT_WINDOW : LONG_WINDOW] = -0.02
    close = open_price * np.exp(returns)
    total_share = np.full((rows, 1), 20_000.0)
    panel = {
        "open": open_price,
        "close": close,
        "total_share": total_share,
    }

    actual = CompletedSmallCapIntradayTrendPullback60x5Strict().calc_batch(
        panel
    )[LONG_WINDOW, 0]
    completed = returns[:LONG_WINDOW, 0]
    q60 = np.sum(completed) / np.sqrt(
        LONG_WINDOW * np.sum(completed * completed)
    )
    q5 = -1.0
    smallness = 1.0 / (
        1.0 + close[LONG_WINDOW - 1, 0] * 20_000.0 * 1.0e4 / PIVOT_RMB
    )
    expected = smallness * ((1.0 + q60) / 2.0) * ((1.0 - q5) / 2.0)
    assert actual == pytest.approx(expected, rel=2e-6)


def test_first_output_and_t_minus_60_boundary_are_exact():
    rows, stocks = LONG_WINDOW + 2, 2
    open_price = np.full((rows, stocks), 10.0)
    returns = np.full((rows, stocks), 0.01)
    returns[LONG_WINDOW - SHORT_WINDOW :] = -0.02
    panel = {
        "open": open_price,
        "close": open_price * np.exp(returns),
        "total_share": np.full((rows, stocks), 20_000.0),
    }
    factor = CompletedSmallCapIntradayTrendPullback60x5Strict()
    baseline = factor.calc_batch(panel)
    assert np.isnan(baseline[:LONG_WINDOW]).all()
    assert np.isfinite(baseline[LONG_WINDOW:]).all()

    changed = _copy_panel(panel)
    changed["close"][0, 0] *= 0.8
    actual = factor.calc_batch(changed)
    assert actual[LONG_WINDOW, 0] != baseline[LONG_WINDOW, 0]
    np.testing.assert_array_equal(
        actual[LONG_WINDOW + 1, 0],
        baseline[LONG_WINDOW + 1, 0],
    )


def test_t_row_hlcva_and_required_current_values_cannot_change_row_t():
    panel = _panel(rows=131, stocks=6)
    factor = CompletedSmallCapIntradayTrendPullback60x5Strict()
    target = 91
    expected = factor.calc_batch(panel)[target].copy()
    changed = _copy_panel(panel)
    replacements = {
        "open": np.asarray([np.nan, np.inf, -1.0, 0.0, 1e-100, 1e100]),
        "high": np.asarray([np.inf, np.nan, 0.0, -1.0, 1e100, 1e-100]),
        "low": np.linspace(-1e100, 1e100, 6),
        "close": np.asarray([np.nan, np.inf, -1.0, 0.0, 1e-100, 1e100]),
        "volume": np.linspace(0.0, 1e100, 6),
        "amount": np.linspace(1e100, 0.0, 6),
        "total_share": np.asarray(
            [np.nan, np.inf, -1.0, 0.0, 1e-100, 1e100]
        ),
    }
    for field, values in replacements.items():
        changed[field][target] = values
    actual = factor.calc_batch(changed)[target]
    np.testing.assert_array_equal(actual, expected)


def test_t_minus_one_close_and_share_change_current_score():
    rows = LONG_WINDOW + 1
    open_price = np.full((rows, 3), 10.0)
    returns = np.full((rows, 3), 0.01)
    returns[LONG_WINDOW - SHORT_WINDOW : LONG_WINDOW] = -0.02
    close = open_price * np.exp(returns)
    share = np.full((rows, 3), 20_000.0)
    factor = CompletedSmallCapIntradayTrendPullback60x5Strict()
    baseline = factor.calc_batch(
        {"open": open_price, "close": close, "total_share": share}
    )[LONG_WINDOW]

    close[LONG_WINDOW - 1, 0] *= 0.99
    share[LONG_WINDOW - 1, 1] *= 4.0
    actual = factor.calc_batch(
        {"open": open_price, "close": close, "total_share": share}
    )[LONG_WINDOW]
    assert actual[0] != baseline[0]
    assert actual[1] != baseline[1]
    assert actual[2] == baseline[2]


@pytest.mark.parametrize("field", ("open", "close"))
@pytest.mark.parametrize("invalid", (np.nan, np.inf, -np.inf, 0.0, -1.0))
def test_invalid_intraday_observation_poisons_every_affected_window(
    field: str,
    invalid: float,
):
    bad_row = 67
    panel = _panel(rows=bad_row + LONG_WINDOW + 2, stocks=2)
    panel[field][bad_row, 0] = invalid
    actual = CompletedSmallCapIntradayTrendPullback60x5Strict().calc_batch(
        panel
    )
    assert np.isnan(
        actual[bad_row + 1 : bad_row + LONG_WINDOW + 1, 0]
    ).all()
    assert np.isfinite(actual[bad_row + LONG_WINDOW + 1, 0])
    assert np.isfinite(actual[LONG_WINDOW:, 1]).all()


@pytest.mark.parametrize(
    "invalid",
    (np.nan, np.inf, -np.inf, 0.0, -1.0),
)
def test_invalid_last_completed_share_is_exposed_without_forward_fill(
    invalid: float,
):
    panel = _panel(rows=LONG_WINDOW + 3, stocks=2)
    panel["total_share"][LONG_WINDOW - 1, 0] = invalid
    actual = CompletedSmallCapIntradayTrendPullback60x5Strict().calc_batch(
        panel
    )
    assert np.isnan(actual[LONG_WINDOW, 0])
    assert np.isfinite(actual[LONG_WINDOW + 1, 0])
    assert np.isfinite(actual[LONG_WINDOW:, 1]).all()


def test_complete_zero_energy_windows_use_q_zero_not_missing():
    panel = _panel(rows=LONG_WINDOW + 3, stocks=4)
    panel["close"][:] = panel["open"]
    actual = CompletedSmallCapIntradayTrendPullback60x5Strict().calc_batch(
        panel
    )
    expected = 0.25 / (
        1.0
        + panel["close"][LONG_WINDOW - 1 : -1]
        * panel["total_share"][LONG_WINDOW - 1 : -1]
        * 1.0e4
        / PIVOT_RMB
    )
    assert np.isnan(actual[:LONG_WINDOW]).all()
    np.testing.assert_allclose(
        actual[LONG_WINDOW:],
        expected,
        rtol=2e-6,
        atol=0.0,
    )


def test_100yi_runtime_share_unit_pivot_has_smallness_one_half():
    rows = LONG_WINDOW + 1
    open_price = np.full((rows, 1), 10.0)
    returns = np.full((rows, 1), 0.01)
    returns[LONG_WINDOW - SHORT_WINDOW : LONG_WINDOW] = -0.02
    close = open_price * np.exp(returns)
    total_share = np.full((rows, 1), 1.0)
    total_share[-2, 0] = PIVOT_RMB / (close[-2, 0] * 1.0e4)
    panel = {
        "open": open_price,
        "close": close,
        "total_share": total_share,
    }
    actual = CompletedSmallCapIntradayTrendPullback60x5Strict().calc_batch(
        panel
    )[LONG_WINDOW, 0]
    completed = returns[:LONG_WINDOW, 0]
    q60 = np.sum(completed) / np.sqrt(
        LONG_WINDOW * np.sum(completed * completed)
    )
    np.testing.assert_allclose(
        actual,
        0.5 * ((1.0 + q60) / 2.0),
        rtol=2e-6,
    )


def test_non_target_q_values_remain_continuous_without_zero_score_plateau():
    rows, stocks = LONG_WINDOW + 1, 128
    row = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    open_price = np.full((rows, stocks), 10.0)
    returns = (
        -0.004
        + 0.0015 * np.sin(row / 7.0 + stock / 19.0)
        + 0.000002 * stock
    )
    returns[LONG_WINDOW - SHORT_WINDOW : LONG_WINDOW] += (
        0.005 + 0.000003 * stock
    )
    close = open_price * np.exp(returns)
    total_share = np.full((rows, stocks), 20_000.0)

    actual = CompletedSmallCapIntradayTrendPullback60x5Strict().calc_batch(
        {
            "open": open_price,
            "close": close,
            "total_share": total_share,
        }
    )[LONG_WINDOW]

    assert np.isfinite(actual).all()
    assert np.count_nonzero(actual) == stocks
    assert np.unique(actual).size > int(0.95 * stocks)


def test_preclose_changes_do_not_change_intraday_signal():
    panel = _panel(rows=151, stocks=5)
    factor = CompletedSmallCapIntradayTrendPullback60x5Strict()
    expected = factor.calc_batch(panel)
    row = np.arange(panel["preClose"].shape[0], dtype=np.float64)[:, None]
    stock = np.arange(panel["preClose"].shape[1], dtype=np.float64)[None, :]
    panel["preClose"][:] = np.exp(
        -300.0 + 600.0 * ((row + stock) % 17.0) / 16.0
    )
    actual = factor.calc_batch(panel)
    np.testing.assert_array_equal(actual, expected)


def test_only_open_close_and_total_share_are_accessed():
    class RecordingPanel(dict):
        def __init__(self, values):
            super().__init__(values)
            self.accesses = []

        def __getitem__(self, key):
            self.accesses.append(key)
            return super().__getitem__(key)

    panel = RecordingPanel(_panel(rows=LONG_WINDOW + 1, stocks=2))
    CompletedSmallCapIntradayTrendPullback60x5Strict().calc_batch(panel)
    assert panel.accesses == ["open", "close", "total_share"]


def test_missing_dimension_and_shape_contracts_fail_loudly():
    panel = _panel(rows=LONG_WINDOW + 1, stocks=2)
    for missing in ("open", "close", "total_share"):
        changed = _copy_panel(panel)
        changed.pop(missing)
        with pytest.raises(KeyError, match=missing):
            CompletedSmallCapIntradayTrendPullback60x5Strict().calc_batch(
                changed
            )

    for field in ("open", "close", "total_share"):
        changed = _copy_panel(panel)
        changed[field] = changed[field][:, 0]
        with pytest.raises(ValueError, match="two-dimensional"):
            CompletedSmallCapIntradayTrendPullback60x5Strict().calc_batch(
                changed
            )

    changed = _copy_panel(panel)
    changed["total_share"] = changed["total_share"][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        CompletedSmallCapIntradayTrendPullback60x5Strict().calc_batch(changed)

    for field in ("open", "close", "total_share"):
        changed = _copy_panel(panel)
        changed[field] = changed[field].astype(np.complex128)
        with pytest.raises(ValueError, match="real numeric dtypes"):
            CompletedSmallCapIntradayTrendPullback60x5Strict().calc_batch(
                changed
            )


def test_source_has_no_tolerance_stock_loop_or_unregistered_dependency():
    source_path = Path(
        inspect.getsourcefile(
            CompletedSmallCapIntradayTrendPullback60x5Strict
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
    assert not (loop_targets & {"stock", "stocks", "column", "columns"})
    assert "range(stocks)" not in source
    assert "nan_to_num" not in source
    assert "sliding_window" not in source
    assert "panel.get(" not in source
    assert "try:" not in source
    assert "np.maximum" not in source
    assert "_SIZE_PIVOT_RMB = 1.0e10" in source
    assert "not a searched parameter" in source


def test_full_runtime_calc_batch_median_is_under_one_second():
    assert RUNTIME_PATH.exists()
    with np.load(RUNTIME_PATH) as runtime:
        panel = {
            "open": runtime["open"],
            "close": runtime["close"],
            "total_share": runtime["total_share"],
        }
        factor = CompletedSmallCapIntradayTrendPullback60x5Strict()
        warmup = factor.calc_batch(panel)
        assert warmup.shape == panel["open"].shape
        del warmup
        gc.collect()

        elapsed = []
        for _ in range(3):
            started = time.perf_counter()
            output = factor.calc_batch(panel)
            elapsed.append(time.perf_counter() - started)
            assert output.shape == panel["open"].shape
            assert output.dtype == np.float32
            assert np.isnan(output[:LONG_WINDOW]).all()
            del output
            gc.collect()

    assert float(np.median(elapsed)) < 1.0, elapsed
