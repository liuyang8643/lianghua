import numpy as np

from factor_db.factors.TrueMarketCapT1 import TrueMarketCapT1


def test_t1_market_cap_ignores_current_open_and_close():
    close = np.array([[10.0, 20.0], [11.0, 18.0], [12.0, 17.0]])
    shares = np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]) * 1e8
    panel = {"close": close.copy(), "total_share": shares.copy(),
             "open": close.copy()}
    expected = TrueMarketCapT1().calc_batch(panel)

    changed = {key: value.copy() for key, value in panel.items()}
    changed["open"][2] = [1.0, 100.0]
    changed["close"][2] = [1000.0, 0.1]
    actual = TrueMarketCapT1().calc_batch(changed)

    np.testing.assert_array_equal(actual[2], expected[2])
