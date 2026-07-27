import numpy as np

from factor_db.factors.TrendIndividualV5Strict import TrendIndividualV5Strict


def _panel(days=260, stocks=4):
    base = np.linspace(8.0, 12.0, days)[:, None]
    close = base * np.linspace(0.95, 1.05, stocks)[None, :]
    return {
        "open": close * 1.001,
        "close": close,
        "st_mask": np.zeros(close.shape, dtype=bool),
    }


def _copy(panel):
    return {name: values.copy() for name, values in panel.items()}


def test_strict_v5_exposes_incomplete_history_instead_of_neutral_fill():
    panel = _panel()
    actual = TrendIndividualV5Strict().calc_batch(panel)

    assert np.all(np.isnan(actual[:121]))
    assert np.all(np.isfinite(actual[121:]))


def test_strict_v5_current_close_does_not_change_current_signal():
    panel = _panel()
    expected = TrendIndividualV5Strict().calc_batch(panel)
    changed = _copy(panel)
    changed["close"][-1] *= np.array([0.1, 0.5, 2.0, 10.0])

    actual = TrendIndividualV5Strict().calc_batch(changed)

    np.testing.assert_array_equal(actual[-1], expected[-1])


def test_strict_v5_current_open_only_changes_legal_gate():
    panel = _panel()
    expected = TrendIndividualV5Strict().calc_batch(panel)
    changed = _copy(panel)
    changed["open"][-1] *= np.array([0.8, 0.9, 1.1, 1.2])
    actual = TrendIndividualV5Strict().calc_batch(changed)
    np.testing.assert_array_equal(actual[-1], expected[-1])

    changed["open"][-1, 0] = 1.99
    actual = TrendIndividualV5Strict().calc_batch(changed)
    assert np.isnan(actual[-1, 0])


def test_strict_v5_missing_completed_close_remains_missing():
    panel = _panel()
    panel["close"][200, 0] = np.nan

    actual = TrendIndividualV5Strict().calc_batch(panel)

    assert np.isnan(actual[201, 0])
    assert np.isfinite(actual[201, 1])


def test_strict_v5_prefix_is_independent_of_future_rows():
    panel = _panel(days=300)
    expected = TrendIndividualV5Strict().calc_batch(panel)[:260]
    truncated = {
        name: values[:260].copy() for name, values in panel.items()
    }

    actual = TrendIndividualV5Strict().calc_batch(truncated)

    np.testing.assert_array_equal(actual, expected)
