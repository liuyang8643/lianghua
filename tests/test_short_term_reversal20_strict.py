import time

import numpy as np
import pytest

from factor_db.factors.ShortTermReversal20Strict import (
    ShortTermReversal20Strict,
)


def _panel(rows=26, stocks=4):
    day = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    pre_close = 8.0 + 0.02 * day + stock
    official_return = -0.01 + 0.0005 * day + 0.0002 * stock
    close = pre_close * (1.0 + official_return)
    return {
        "open": np.full((rows, stocks), 5.0, dtype=np.float64),
        "close": close,
        "preClose": pre_close,
        "st_mask": np.zeros((rows, stocks), dtype=bool),
        "high": close * 1.02,
        "low": close * 0.98,
        "volume": np.full((rows, stocks), 1e6, dtype=np.float64),
        "amount": np.full((rows, stocks), 1e8, dtype=np.float64),
    }


def _manual_score(panel, row, stock):
    gross = (
        panel["close"][row - 20 : row, stock]
        / panel["preClose"][row - 20 : row, stock]
    )
    return -np.expm1(np.sum(np.log(gross), dtype=np.float64))


def test_uses_exactly_twenty_completed_days_and_matches_manual_reference():
    panel = _panel()
    result = ShortTermReversal20Strict().calc_batch(panel)

    assert result.shape == panel["close"].shape
    assert result.dtype == np.float32
    assert np.isnan(result[:20]).all()
    for row in (20, 21, 25):
        for stock in range(result.shape[1]):
            np.testing.assert_allclose(
                result[row, stock],
                _manual_score(panel, row, stock),
                rtol=1e-6,
                atol=0.0,
            )


@pytest.mark.parametrize(
    ("field", "value"),
    (("close", np.nan), ("close", np.inf), ("close", 0.0),
     ("preClose", np.nan), ("preClose", np.inf), ("preClose", -1.0)),
)
def test_invalid_completed_observation_strictly_poison_entire_window(field, value):
    panel = _panel(rows=43, stocks=1)
    panel[field][7, 0] = value
    result = ShortTermReversal20Strict().calc_batch(panel)[:, 0]

    assert np.isnan(result[20:28]).all()
    assert np.isfinite(result[28])


def test_current_day_hlcva_and_preclose_cannot_change_current_score():
    panel = _panel()
    factor = ShortTermReversal20Strict()
    expected = factor.calc_batch(panel)[20]
    changed = {name: values.copy() for name, values in panel.items()}
    for name in ("high", "low", "close", "volume", "amount", "preClose"):
        changed[name][20] = np.array([np.nan, np.inf, -1.0, 1e30])

    np.testing.assert_array_equal(factor.calc_batch(changed)[20], expected)


def test_completed_day_equal_price_scale_is_corporate_action_invariant():
    panel = _panel()
    factor = ShortTermReversal20Strict()
    expected = factor.calc_batch(panel)[20]
    scaled = {name: values.copy() for name, values in panel.items()}
    scaled["close"][9] *= 0.2
    scaled["preClose"][9] *= 0.2

    np.testing.assert_allclose(
        factor.calc_batch(scaled)[20],
        expected,
        rtol=1e-6,
        atol=0.0,
    )


def test_current_open_and_st_are_legality_gates_only():
    panel = _panel()
    factor = ShortTermReversal20Strict()
    expected = factor.calc_batch(panel)[20]
    assert np.isfinite(expected).all()

    panel["open"][20] = np.array([2.0, 2000.0, 1.99, np.nan])
    panel["st_mask"][20, 1] = True
    actual = factor.calc_batch(panel)[20]

    np.testing.assert_allclose(actual[0], expected[0])
    assert np.isnan(actual[1:]).all()


@pytest.mark.parametrize(
    "field",
    ("open", "close", "preClose", "st_mask"),
)
def test_rejects_shape_mismatch(field):
    panel = _panel()
    panel[field] = panel[field][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        ShortTermReversal20Strict().calc_batch(panel)


def test_rejects_non_matrix_price_panel():
    panel = _panel()
    panel["close"] = panel["close"][:, 0]
    with pytest.raises(ValueError, match="two-dimensional"):
        ShortTermReversal20Strict().calc_batch(panel)


def test_medium_panel_is_vectorized_across_stocks():
    rows, stocks = 420, 1100
    rng = np.random.default_rng(20260722)
    pre_close = rng.uniform(2.1, 100.0, size=(rows, stocks))
    panel = {
        "open": pre_close.copy(),
        "close": pre_close * (1.0 + rng.normal(0.0002, 0.02, (rows, stocks))),
        "preClose": pre_close,
        "st_mask": np.zeros((rows, stocks), dtype=bool),
    }

    started = time.perf_counter()
    result = ShortTermReversal20Strict().calc_batch(panel)
    elapsed = time.perf_counter() - started

    assert result.shape == (rows, stocks)
    assert np.isfinite(result[20:]).all()
    assert elapsed < 1.0
