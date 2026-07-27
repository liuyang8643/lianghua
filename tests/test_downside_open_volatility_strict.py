import numpy as np
import pytest

from factor_db.factors.DownsideOpenVolatilityStrict import (
    DownsideOpenVolatilityStrict,
)


def _panel(days=30, stocks=2):
    price = np.full((days, stocks), 10.0)
    return {
        "open": price.copy(),
        "high": price.copy() * 1.01,
        "low": price.copy() * 0.99,
        "close": price.copy(),
        "preClose": price.copy(),
        "volume": np.full((days, stocks), 1_000_000.0),
        "amount": np.full((days, stocks), 10_000_000.0),
        "st_mask": np.zeros((days, stocks), dtype=bool),
    }


def test_requires_exactly_twenty_finite_open_path_returns():
    panel = _panel()
    actual = DownsideOpenVolatilityStrict().calc_batch(panel)

    assert np.isnan(actual[:20]).all()
    np.testing.assert_array_equal(actual[20:], 0.0)


def test_current_hlcva_cannot_change_current_score():
    panel = _panel()
    expected = DownsideOpenVolatilityStrict().calc_batch(panel)
    changed = {key: value.copy() for key, value in panel.items()}
    for key in ("high", "low", "close", "volume", "amount"):
        changed[key][24] *= 50.0

    actual = DownsideOpenVolatilityStrict().calc_batch(changed)

    np.testing.assert_array_equal(actual[24], expected[24])


def test_current_open_and_preclose_are_part_of_the_known_open_return():
    panel = _panel()
    changed_open = {key: value.copy() for key, value in panel.items()}
    changed_open["open"][24, 0] = 9.0
    changed_preclose = {key: value.copy() for key, value in panel.items()}
    changed_preclose["preClose"][24, 0] = 12.0

    open_score = DownsideOpenVolatilityStrict().calc_batch(changed_open)
    preclose_score = DownsideOpenVolatilityStrict().calc_batch(changed_preclose)

    assert open_score[24, 0] == pytest.approx(-np.sqrt(0.1 ** 2 / 20))
    assert preclose_score[24, 0] < open_score[24, 0]


def test_official_preclose_neutralizes_a_split_in_the_raw_price_path():
    panel = _panel(stocks=1)
    panel["open"][10:] = 5.0
    panel["close"][10:] = 5.0
    panel["preClose"][10:] = 5.0

    actual = DownsideOpenVolatilityStrict().calc_batch(panel)

    assert actual[20, 0] == pytest.approx(0.0)


def test_missing_return_propagates_until_it_leaves_the_strict_window():
    panel = _panel(days=45, stocks=1)
    panel["preClose"][10, 0] = np.nan

    actual = DownsideOpenVolatilityStrict().calc_batch(panel)

    assert np.isnan(actual[20:30, 0]).all()
    assert actual[30, 0] == pytest.approx(0.0)


def test_current_legality_is_exposed_as_nan_without_filling():
    panel = _panel(stocks=2)
    panel["open"][24, 0] = 1.99
    panel["st_mask"][24, 1] = True

    actual = DownsideOpenVolatilityStrict().calc_batch(panel)

    assert np.isnan(actual[24]).all()
