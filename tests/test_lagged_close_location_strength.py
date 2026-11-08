import numpy as np

from factor_db.factors.LaggedCloseLocationStrength import (
    LaggedCloseLocationStrength,
)


def _panel() -> dict:
    return {
        "open": np.full((4, 2), 10.0),
        "high": np.array([[12.0, 14.0], [15.0, 12.0], [20.0, 16.0], [18.0, 15.0]]),
        "low": np.array([[8.0, 10.0], [5.0, 8.0], [10.0, 12.0], [10.0, 11.0]]),
        "close": np.array([[11.0, 11.0], [10.0, 9.0], [12.0, 15.0], [16.0, 12.0]]),
        "st_mask": np.zeros((4, 2), dtype=bool),
    }


def test_close_location_uses_the_completed_bar_only():
    panel = _panel()
    expected = LaggedCloseLocationStrength().calc_batch(panel)
    changed = {key: value.copy() for key, value in panel.items()}
    changed["high"][2] *= 100.0
    changed["low"][2] *= 0.01
    changed["close"][2] *= 50.0

    actual = LaggedCloseLocationStrength().calc_batch(changed)

    np.testing.assert_array_equal(actual[2], expected[2])
    assert not np.array_equal(actual[3], expected[3])
    np.testing.assert_allclose(expected[1], [0.75, 0.25])


def test_close_location_exposes_missing_and_zero_range_bars():
    panel = _panel()
    panel["high"][1, 0] = panel["low"][1, 0]
    panel["close"][1, 1] = np.nan

    actual = LaggedCloseLocationStrength().calc_batch(panel)

    assert np.isnan(actual[2, 0])
    assert np.isnan(actual[2, 1])


def test_close_location_uses_current_open_and_st_only_for_legality():
    panel = _panel()
    expected = LaggedCloseLocationStrength().calc_batch(panel)
    changed = {key: value.copy() for key, value in panel.items()}
    changed["open"][2] = [200.0, 1.99]
    changed["st_mask"][3, 0] = True

    actual = LaggedCloseLocationStrength().calc_batch(changed)

    assert actual[2, 0] == expected[2, 0]
    assert np.isnan(actual[2, 1])
    assert np.isnan(actual[3, 0])
