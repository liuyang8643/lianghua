import numpy as np

from factor_db.factors.PreCloseMarketCap import PreCloseMarketCap


def test_preclose_market_cap_does_not_use_t_day_open():
    panel = {
        "preClose": np.array([[10.0, 20.0], [11.0, 21.0]], dtype=np.float64),
        "total_share": np.full((2, 2), 1e8, dtype=np.float64),
        "open": np.array([[9.0, 19.0], [10.0, 22.0]], dtype=np.float64),
    }
    expected = PreCloseMarketCap().calc_batch(panel)
    panel["open"][1] = [1e-9, 1e9]
    np.testing.assert_array_equal(PreCloseMarketCap().calc_batch(panel), expected)


def test_preclose_market_cap_invalid_reference_is_nan():
    panel = {
        "preClose": np.array([[10.0, np.nan, -1.0]], dtype=np.float64),
        "total_share": np.array([[1e8, 1e8, 1e8]], dtype=np.float64),
    }
    result = PreCloseMarketCap().calc_batch(panel)
    assert np.isfinite(result[0, 0])
    assert np.isnan(result[0, 1])
    assert np.isnan(result[0, 2])
