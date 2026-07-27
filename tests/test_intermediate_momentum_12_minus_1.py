import numpy as np
import pytest

from factor_db.factors.IntermediateMomentum12Minus1 import (
    IntermediateMomentum12Minus1,
)
from core.ga import get_profile, get_profile_factor_names


def _panel(days=280, stocks=2):
    close = np.arange(1, days + 1, dtype=np.float64)[:, None]
    close = np.repeat(close, stocks, axis=1)
    pre_close = close.copy()
    pre_close[1:] = close[:-1]
    return {
        "open": np.full((days, stocks), 10.0),
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "preClose": pre_close,
        "volume": np.ones_like(close),
        "amount": np.ones_like(close),
        "st_mask": np.zeros((days, stocks), dtype=bool),
    }


def test_intermediate_momentum_uses_fixed_12_minus_1_window():
    panel = _panel()

    actual = IntermediateMomentum12Minus1().calc_batch(panel)

    expected = panel["close"][258, 0] / panel["close"][27, 0] - 1.0
    assert actual[280 - 1, 0] == pytest.approx(expected)
    assert np.isnan(actual[250, 0])
    expected_first = panel["close"][230, 0] / panel["preClose"][0, 0] - 1.0
    assert actual[251, 0] == pytest.approx(expected_first)


def test_intermediate_momentum_uses_official_preclose_across_split():
    panel = _panel()
    expected = IntermediateMomentum12Minus1().calc_batch(panel)

    split = {key: value.copy() for key, value in panel.items()}
    split["close"][100:] *= 0.5
    split["preClose"][100:] *= 0.5
    actual = IntermediateMomentum12Minus1().calc_batch(split)

    assert actual[-1, 0] == pytest.approx(expected[-1, 0])


def test_intermediate_momentum_requires_complete_return_window():
    panel = _panel()
    panel["preClose"][100, 0] = np.nan

    actual = IntermediateMomentum12Minus1().calc_batch(panel)

    assert np.isnan(actual[-1, 0])
    assert np.isfinite(actual[-1, 1])


def test_intermediate_momentum_ignores_current_hlc_and_future_rows():
    panel = _panel()
    expected = IntermediateMomentum12Minus1().calc_batch(panel)

    changed = {key: value.copy() for key, value in panel.items()}
    for key in ("high", "low", "close", "volume", "amount"):
        changed[key][260] *= 100.0
    actual = IntermediateMomentum12Minus1().calc_batch(changed)

    assert actual[260, 0] == expected[260, 0]
    np.testing.assert_array_equal(actual[:260], expected[:260])


def test_intermediate_momentum_uses_current_open_only_for_eligibility():
    panel = _panel()
    panel["open"][260, 0] = 1.99
    panel["st_mask"][261, 1] = True

    actual = IntermediateMomentum12Minus1().calc_batch(panel)

    assert np.isnan(actual[260, 0])
    assert np.isnan(actual[261, 1])


def test_v10_profile_changes_only_the_factor_set():
    baseline = get_profile('v9_dual_shadow')
    experiment = get_profile('v10_intermediate_momentum')

    assert get_profile_factor_names('v10_intermediate_momentum') == [
        *get_profile_factor_names('v9_dual_shadow'),
        'IntermediateMomentum12Minus1',
    ]
    assert experiment['search_spaces'] == baseline['search_spaces']
    assert experiment['training_objective'] == baseline['training_objective']
