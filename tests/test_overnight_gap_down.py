import numpy as np

from factor_db.factors.OvernightGapDown import OvernightGapDown


def _panel(days=4):
    open_ = np.array(
        [[9.8, 10.0, 10.2], [9.7, 10.0, 10.3], [9.6, 10.0, 10.4], [9.5, 10.0, 10.5]],
        dtype=np.float64,
    )[:days]
    shape = open_.shape
    return {
        "open": open_,
        "preClose": np.full(shape, 10.0, dtype=np.float64),
        "st_mask": np.zeros(shape, dtype=bool),
        "high": np.full(shape, 99.0, dtype=np.float64),
        "low": np.full(shape, 1.0, dtype=np.float64),
        "close": np.full(shape, 50.0, dtype=np.float64),
        "volume": np.full(shape, 1_000.0, dtype=np.float64),
        "amount": np.full(shape, 5_000.0, dtype=np.float64),
    }


def test_gap_down_score_is_exact_open_to_official_preclose_gap():
    panel = _panel()
    result = OvernightGapDown().calc_batch(panel)

    expected = 1.0 - panel["open"] / panel["preClose"]
    np.testing.assert_allclose(result, expected.astype(np.float32))
    assert result[-1, 0] > result[-1, 1] > result[-1, 2]


def test_gap_down_ignores_trade_day_hlcva():
    panel = _panel()
    expected = OvernightGapDown().calc_batch(panel)[-1]
    changed = {key: value.copy() for key, value in panel.items()}
    for key in ("high", "low", "close", "volume", "amount"):
        changed[key][-1] = [0.01, 500.0, 9_999.0]

    np.testing.assert_array_equal(
        OvernightGapDown().calc_batch(changed)[-1],
        expected,
    )


def test_gap_down_prefix_is_independent_of_future_rows():
    panel = _panel()
    expected = OvernightGapDown().calc_batch(panel)[:3]
    truncated = {key: value[:3].copy() for key, value in panel.items()}

    np.testing.assert_array_equal(
        OvernightGapDown().calc_batch(truncated),
        expected,
    )


def test_gap_down_exposes_invalid_open_preclose_and_st_as_missing():
    panel = _panel()
    panel["open"][-1, 0] = 1.99
    panel["preClose"][-1, 1] = np.nan
    panel["st_mask"][-1, 2] = True

    assert np.isnan(OvernightGapDown().calc_batch(panel)[-1]).all()

