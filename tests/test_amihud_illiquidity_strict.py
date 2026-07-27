import numpy as np

from factor_db.factors.AmihudIlliquidityStrict import AmihudIlliquidityStrict


def _panel(rows=24, stocks=3):
    pre_close = np.full((rows, stocks), 10.0)
    returns = np.linspace(-0.02, 0.03, rows)[:, None]
    return {
        "close": pre_close * (1.0 + returns),
        "preClose": pre_close,
        "amount": np.full((rows, stocks), 2e8),
    }


def test_uses_exactly_twenty_completed_official_returns():
    panel = _panel()
    result = AmihudIlliquidityStrict().calc_batch(panel)

    expected = np.mean(np.abs(panel["close"][:20, 0] / 10.0 - 1.0) / 2.0)
    assert np.isnan(result[:20]).all()
    np.testing.assert_allclose(result[20], expected, rtol=1e-6)


def test_rejects_any_incomplete_or_invalid_window():
    panel = _panel()
    panel["amount"][5, 0] = np.nan
    panel["preClose"][7, 1] = 0.0
    panel["close"][9, 2] = np.inf
    result = AmihudIlliquidityStrict().calc_batch(panel)

    assert np.isnan(result[20]).all()


def test_current_day_fields_cannot_change_current_score():
    panel = _panel()
    factor = AmihudIlliquidityStrict()
    expected = factor.calc_batch(panel)[20]

    changed = {key: value.copy() for key, value in panel.items()}
    changed["close"][20] = np.nan
    changed["preClose"][20] = np.inf
    changed["amount"][20] = -1.0
    np.testing.assert_array_equal(factor.calc_batch(changed)[20], expected)


def test_official_preclose_absorbs_a_completed_day_price_scale_change():
    panel = _panel()
    factor = AmihudIlliquidityStrict()
    expected = factor.calc_batch(panel)[20]

    scaled = {key: value.copy() for key, value in panel.items()}
    scaled["close"][10] *= 0.25
    scaled["preClose"][10] *= 0.25
    np.testing.assert_allclose(factor.calc_batch(scaled)[20], expected)
