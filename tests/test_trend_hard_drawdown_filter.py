import numpy as np

from factor_db.factors.TrendHardDrawdownFilter import TrendHardDrawdownFilter


def _panel(days=30, stocks=3):
    open_ = np.full((days, stocks), 10.0, dtype=np.float64)
    high = np.full((days, stocks), 10.0, dtype=np.float64)
    st_mask = np.zeros((days, stocks), dtype=bool)
    high[days - 10, 0] = 20.0
    return {"open": open_, "high": high, "st_mask": st_mask}


def test_hard_drawdown_filter_maps_only_known_risk_to_nan():
    panel = _panel()
    panel["open"][-1] = [15.0, 10.0, 10.0]
    panel["st_mask"][-1, 2] = True

    result = TrendHardDrawdownFilter().calc_batch(panel)

    assert np.isnan(result[-1, 0])
    assert result[-1, 1] == 1.0
    assert np.isnan(result[-1, 2])


def test_hard_drawdown_filter_uses_current_open_but_not_current_high():
    panel = _panel()
    expected = TrendHardDrawdownFilter().calc_batch(panel)[-1]

    changed_high = {key: value.copy() for key, value in panel.items()}
    changed_high["high"][-1] = [1_000.0, 2_000.0, 3_000.0]
    np.testing.assert_array_equal(
        TrendHardDrawdownFilter().calc_batch(changed_high)[-1],
        expected,
    )

    changed_open = {key: value.copy() for key, value in panel.items()}
    changed_open["open"][-1, 0] = 19.0
    actual = TrendHardDrawdownFilter().calc_batch(changed_open)[-1]
    assert np.isnan(expected[0])
    assert actual[0] == 1.0


def test_hard_drawdown_filter_prefix_is_independent_of_future_rows():
    panel = _panel(days=40)
    expected = TrendHardDrawdownFilter().calc_batch(panel)[:32]
    truncated = {key: value[:32].copy() for key, value in panel.items()}

    np.testing.assert_array_equal(
        TrendHardDrawdownFilter().calc_batch(truncated),
        expected,
    )


def test_hard_drawdown_filter_explicitly_fails_open_without_window_history():
    panel = _panel(days=30)
    result = TrendHardDrawdownFilter().calc_batch(panel)

    assert np.all(result[:19] == 1.0)

