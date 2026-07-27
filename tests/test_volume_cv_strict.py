import numpy as np

from factor_db.factors.VolumeCVStrict import VolumeCVStrict


def test_uses_exactly_twenty_completed_finite_days():
    volume = np.arange(1.0, 23.0)[:, None]
    result = VolumeCVStrict().calc_batch({"volume": volume})

    assert np.isnan(result[:20]).all()
    expected = -np.std(volume[:20, 0]) / np.mean(volume[:20, 0])
    np.testing.assert_allclose(result[20, 0], expected, rtol=1e-6)


def test_rejects_incomplete_non_finite_and_non_positive_mean_windows():
    volume = np.tile(np.arange(1.0, 24.0)[:, None], (1, 4))
    volume[0, 0] = np.nan
    volume[5, 1] = np.inf
    volume[:20, 2] = 0.0
    volume[:20, 3] = np.tile([-1.0, 1.0], 10)

    result = VolumeCVStrict().calc_batch({"volume": volume})

    assert np.isnan(result[20]).all()
    assert np.isfinite(result[21, 0])
    assert np.isnan(result[21, 1])


def test_current_day_volume_cannot_change_current_score():
    volume = np.arange(1.0, 26.0)[:, None]
    factor = VolumeCVStrict()
    expected = factor.calc_batch({"volume": volume})[20, 0]

    changed_current = volume.copy()
    changed_current[20, 0] = np.nan
    actual = factor.calc_batch({"volume": changed_current})[20, 0]
    np.testing.assert_array_equal(actual, expected)

    changed_history = volume.copy()
    changed_history[19, 0] = 1000.0
    historical_actual = factor.calc_batch({"volume": changed_history})[20, 0]
    assert historical_actual != expected


def test_matches_direct_twenty_day_windows_across_missing_values():
    rng = np.random.default_rng(20260722)
    volume = rng.lognormal(mean=12.0, sigma=1.0, size=(620, 7))
    volume[17, 1] = np.nan
    volume[255, 2] = np.inf
    volume[510:530, 3] = 0.0

    result = VolumeCVStrict().calc_batch({"volume": volume})
    expected = np.full_like(result, np.nan)
    for row in range(20, len(volume)):
        history = volume[row - 20:row]
        finite = np.isfinite(history).all(axis=0)
        means = np.mean(history, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.std(history, axis=0) / means
        valid = finite & np.isfinite(means) & (means > 0.0) & np.isfinite(cv)
        expected[row, valid] = -cv[valid]

    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-7, equal_nan=True)
