import time

import numpy as np
import pytest

from factor_db.factors.VolumeContraction5v15Strict import (
    VolumeContraction5v15Strict,
)


def _panel(rows=26, stocks=4):
    day = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    volume = (1_000_000.0 + 25_000.0 * day) * (1.0 + 0.1 * stock)
    open_ = np.full((rows, stocks), 5.0, dtype=np.float64)
    return {
        "open": open_,
        "high": open_ * 1.02,
        "low": open_ * 0.98,
        "close": open_ * 1.01,
        "volume": volume,
        "amount": volume * open_,
        "st_mask": np.zeros((rows, stocks), dtype=bool),
    }


def _manual_score(panel, row, stock):
    recent = np.mean(panel["volume"][row - 5 : row, stock])
    prior = np.mean(panel["volume"][row - 20 : row - 5, stock])
    return -np.log(recent / prior)


def test_uses_exact_completed_windows_and_matches_manual_reference():
    panel = _panel(rows=260)
    result = VolumeContraction5v15Strict().calc_batch(panel)

    assert result.shape == panel["volume"].shape
    assert result.dtype == np.float32
    assert np.isnan(result[:20]).all()
    for row in (20, 21, 25, 255, 256, 257, 259):
        for stock in range(result.shape[1]):
            np.testing.assert_allclose(
                result[row, stock],
                _manual_score(panel, row, stock),
                rtol=1e-6,
                atol=0.0,
            )


def test_window_boundary_moves_one_completed_day_at_a_time():
    panel = _panel(rows=22, stocks=1)
    factor = VolumeContraction5v15Strict()
    baseline = factor.calc_batch(panel)[:, 0]

    changed_oldest = {name: values.copy() for name, values in panel.items()}
    changed_oldest["volume"][0, 0] *= 10.0
    oldest_result = factor.calc_batch(changed_oldest)[:, 0]
    assert oldest_result[20] != pytest.approx(baseline[20])
    assert oldest_result[21] == pytest.approx(baseline[21])

    changed_current = {name: values.copy() for name, values in panel.items()}
    changed_current["volume"][20, 0] *= 10.0
    current_result = factor.calc_batch(changed_current)[:, 0]
    assert current_result[20] == pytest.approx(baseline[20])
    assert current_result[21] != pytest.approx(baseline[21])


@pytest.mark.parametrize("value", (np.nan, np.inf, 0.0, -1.0))
def test_invalid_completed_observation_strictly_poisons_entire_window(value):
    panel = _panel(rows=43, stocks=1)
    panel["volume"][7, 0] = value
    result = VolumeContraction5v15Strict().calc_batch(panel)[:, 0]

    assert np.isnan(result[20:28]).all()
    assert np.isfinite(result[28])


def test_current_day_hlcva_cannot_change_current_score():
    panel = _panel()
    factor = VolumeContraction5v15Strict()
    expected = factor.calc_batch(panel)[20]
    changed = {name: values.copy() for name, values in panel.items()}
    for name in ("high", "low", "close", "volume", "amount"):
        changed[name][20] = np.array([np.nan, np.inf, -1.0, 1e30])

    np.testing.assert_array_equal(factor.calc_batch(changed)[20], expected)


def test_volume_scale_does_not_change_scores():
    panel = _panel(rows=40)
    factor = VolumeContraction5v15Strict()
    expected = factor.calc_batch(panel)
    scaled = {name: values.copy() for name, values in panel.items()}
    scaled["volume"] *= 37.5

    np.testing.assert_allclose(
        factor.calc_batch(scaled)[20:],
        expected[20:],
        rtol=1e-6,
        atol=1e-7,
    )


def test_current_open_and_st_are_legality_gates_only():
    panel = _panel()
    factor = VolumeContraction5v15Strict()
    expected = factor.calc_batch(panel)[20]
    assert np.isfinite(expected).all()

    panel["open"][20] = np.array([2.0, 2000.0, 1.99, np.nan])
    panel["st_mask"][20, 1] = True
    actual = factor.calc_batch(panel)[20]

    np.testing.assert_allclose(actual[0], expected[0])
    assert np.isnan(actual[1:]).all()


@pytest.mark.parametrize("field", ("volume", "open", "st_mask"))
def test_rejects_shape_mismatch(field):
    panel = _panel()
    panel[field] = panel[field][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        VolumeContraction5v15Strict().calc_batch(panel)


def test_rejects_non_matrix_volume_panel():
    panel = _panel()
    panel["volume"] = panel["volume"][:, 0]
    with pytest.raises(ValueError, match="two-dimensional"):
        VolumeContraction5v15Strict().calc_batch(panel)


def test_does_not_shrink_window_when_only_twenty_rows_exist():
    panel = _panel(rows=20, stocks=2)
    result = VolumeContraction5v15Strict().calc_batch(panel)
    assert np.isnan(result).all()


def test_medium_panel_is_vectorized_across_stocks():
    rows, stocks = 420, 1100
    rng = np.random.default_rng(20260722)
    panel = {
        "open": rng.uniform(2.0, 100.0, size=(rows, stocks)),
        "volume": rng.lognormal(14.0, 0.8, size=(rows, stocks)),
        "st_mask": np.zeros((rows, stocks), dtype=bool),
    }

    started = time.perf_counter()
    result = VolumeContraction5v15Strict().calc_batch(panel)
    elapsed = time.perf_counter() - started

    assert result.shape == (rows, stocks)
    assert np.isfinite(result[20:]).all()
    assert elapsed < 1.0
