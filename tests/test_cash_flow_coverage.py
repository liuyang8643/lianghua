import numpy as np

from factor_db.factors.EarningsQuality import CashFlowCoverage


def _panel() -> dict:
    return {
        "open": np.full((4, 2), 10.0),
        "st_mask": np.zeros((4, 2), dtype=bool),
        "eps": np.array([
            [1.0, -2.0],
            [2.0, -4.0],
            [4.0, -8.0],
            [8.0, -16.0],
        ]),
        "operating_cf_ps": np.array([
            [1.5, 1.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [4.0, 8.0],
        ]),
    }


def test_cash_flow_coverage_uses_only_lagged_financial_values():
    panel = _panel()
    expected = CashFlowCoverage().calc_batch(panel)
    changed = {key: value.copy() for key, value in panel.items()}
    changed["eps"][2] *= 100.0
    changed["operating_cf_ps"][2] *= -100.0

    actual = CashFlowCoverage().calc_batch(changed)

    np.testing.assert_array_equal(actual[2], expected[2])
    assert not np.array_equal(actual[3], expected[3])
    np.testing.assert_allclose(expected[1], [1.5, 0.5])


def test_cash_flow_coverage_exposes_missing_lagged_inputs():
    panel = _panel()
    panel["eps"][1, 0] = np.nan
    panel["operating_cf_ps"][1, 1] = np.nan

    actual = CashFlowCoverage().calc_batch(panel)

    assert np.isnan(actual[2, 0])
    assert np.isnan(actual[2, 1])


def test_cash_flow_coverage_uses_current_open_only_for_legality():
    panel = _panel()
    expected = CashFlowCoverage().calc_batch(panel)
    changed = {key: value.copy() for key, value in panel.items()}
    changed["open"][2] = [200.0, 1.99]

    actual = CashFlowCoverage().calc_batch(changed)

    assert actual[2, 0] == expected[2, 0]
    assert np.isnan(actual[2, 1])

