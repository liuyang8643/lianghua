import numpy as np
import pytest

from factor_db.factors.CompletedRangeVolatility20 import (
    CompletedRangeVolatility20,
)


def _panel(days=24, stocks=3):
    low = np.full((days, stocks), 10.0)
    high = low * np.array([1.01, 1.03, 1.08])[None, :]
    close = (high + low) / 2.0
    return {
        "open": close.copy(),
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full((days, stocks), 1_000_000.0),
        "amount": np.full((days, stocks), 10_000_000.0),
        "st_mask": np.zeros((days, stocks), dtype=bool),
    }


def test_uses_exactly_20_completed_ranges_and_prefers_lower_range():
    actual = CompletedRangeVolatility20().calc_batch(_panel())

    assert np.isnan(actual[:20]).all()
    np.testing.assert_allclose(
        actual[20],
        -np.log(np.array([1.01, 1.03, 1.08])),
        rtol=1e-6,
    )
    assert actual[20, 0] > actual[20, 1] > actual[20, 2]


def test_current_day_hlcva_cannot_change_current_score():
    panel = _panel()
    factor = CompletedRangeVolatility20()
    expected = factor.calc_batch(panel)[20].copy()

    for name in ("high", "low", "close", "volume", "amount"):
        panel[name][20] = np.array([np.nan, np.inf, -1.0])

    actual = factor.calc_batch(panel)[20]
    np.testing.assert_array_equal(actual, expected)


def test_incomplete_completed_window_is_exposed_as_nan():
    panel = _panel()
    panel["high"][7, 1] = np.nan

    actual = CompletedRangeVolatility20().calc_batch(panel)

    assert np.isfinite(actual[20, 0])
    assert np.isnan(actual[20, 1])
    assert np.isfinite(actual[20, 2])


def test_current_open_and_st_status_are_legality_gates_only():
    panel = _panel()
    expected = CompletedRangeVolatility20().calc_batch(panel)[20].copy()
    panel["open"][20] = np.array([2.0, 200.0, 1.99])
    panel["st_mask"][20, 1] = True

    actual = CompletedRangeVolatility20().calc_batch(panel)[20]

    assert actual[0] == pytest.approx(expected[0])
    assert np.isnan(actual[1:]).all()


def test_rejects_mismatched_panel_shapes():
    panel = _panel()
    panel["low"] = panel["low"][:, :2]

    with pytest.raises(ValueError, match="matching shapes"):
        CompletedRangeVolatility20().calc_batch(panel)
