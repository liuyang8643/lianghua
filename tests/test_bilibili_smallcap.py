import numpy as np
import pytest

from factor import get_factor_definition, precompute_factors
from factor.library.bilibili_smallcap import BiliSmallCapRisingMA20, SMALLCAP_FACTOR_DEFINITIONS
from offline_data import load_runtime_slice
from test_rl_runtime_slice import _runtime_arrays


def _panel(rows=24, stocks=3):
    return {
        "open": np.full((rows, stocks), 10.0),
        "total_share": np.full((rows, stocks), 100_000_000.0),
        "close": np.full((rows, stocks), 10.0),
    }


def test_exact_ma20_hand_calculation_and_strict_flat_boundary():
    panel = _panel()
    panel["close"][20] = [12, 10, 8]
    old = panel["close"][:20].mean(axis=0)
    new = panel["close"][1:21].mean(axis=0)
    np.testing.assert_allclose(old, [10, 10, 10])
    np.testing.assert_allclose(new, [10.1, 10, 9.9])
    score = BiliSmallCapRisingMA20().calc_batch(panel)
    assert np.isnan(score[:21]).all()
    assert score[21, 0] == -10
    assert np.isnan(score[21, 1:]).all()


@pytest.mark.parametrize("row,bad", [(0, np.nan), (10, 0), (19, -1), (20, np.inf)])
def test_all_twenty_one_completed_closes_are_required(row, bad):
    panel = _panel(stocks=1)
    panel["close"][20] = 12
    panel["close"][row] = bad
    assert np.isnan(BiliSmallCapRisingMA20().calc_batch(panel)[21, 0])


def test_invalid_close_leaving_window_recovers_without_extra_history():
    panel = _panel(stocks=1)
    panel["close"][0] = np.nan
    panel["close"][20:22] = 12
    score = BiliSmallCapRisingMA20().calc_batch(panel)
    assert np.isnan(score[21, 0])
    assert score[22, 0] == -10


def test_completed_close_and_market_cap_causality_through_public_precompute(tmp_path):
    data = _runtime_arrays(rows=26, stocks=2)
    data["close"][:] = 10
    data["close"][20:] = 12

    def compute(name, arrays):
        path = tmp_path / name
        np.savez(path, **arrays)
        runtime = load_runtime_slice(path, arrays["trade_dates"][21], arrays["trade_dates"][-1], preload_rows=1000)
        return precompute_factors(runtime, definitions=SMALLCAP_FACTOR_DEFINITIONS)

    original = compute("original.npz", data)
    changed = {key: value.copy() for key, value in data.items()}
    changed["close"][21:] = 1
    changed["total_share"][21:] *= 100
    changed["open"][22:] *= 100
    altered = compute("changed.npz", changed)
    np.testing.assert_array_equal(original.raw[:22], altered.raw[:22])
    expected = -data["open"][21] * data["total_share"][20] / 1e8
    np.testing.assert_allclose(original.raw[21, 0], expected, rtol=1e-6)
    np.testing.assert_array_equal(original.raw[21, 0], original.raw[21, 1])
    changed["open"][21] *= 2  # The current opening price is explicitly available.
    opening = compute("opening.npz", changed)
    np.testing.assert_allclose(opening.raw[21], original.raw[21] * 2)


def test_definition_reuses_production_formula_and_share_lag():
    original, rising = SMALLCAP_FACTOR_DEFINITIONS
    assert original is get_factor_definition("TrueMarketCap")
    assert issubclass(rising.implementation, original.implementation)
    assert rising.metadata.lagged_fields == original.metadata.lagged_fields == ("total_share",)
    assert rising.metadata.hist_days == 21
    assert rising.metadata.required_fields == ("open", "total_share", "close")
