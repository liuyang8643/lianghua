"""Corrected Amihud illiquidity: finite scale, validity gate, completed windows only."""
import numpy as np

from factor.library import CompletedAmihudIlliquidity20
from factor.registry import PRODUCTION_FACTOR_NAMES, get_factor_definition


def _panel(rows: int, stocks: int, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    pre_close = np.full((rows, stocks), 10.0)
    close = pre_close * (1.0 + rng.normal(0.0, 0.01, size=(rows, stocks)))
    amount = np.full((rows, stocks), 5e7)
    return {"close": close, "preClose": pre_close, "amount": amount}


def test_zero_and_tiny_amount_days_are_missing_not_infinite():
    panel = _panel(40, 3)
    panel["amount"][:, 1] = 0.0                      # suspended-like prints: legacy version produced inf
    panel["amount"][5:12, 2] = 1e3                   # below the 1e5 CNY gate: excluded from the window
    result = CompletedAmihudIlliquidity20().calc_batch(panel)
    assert result.shape == panel["close"].shape
    assert np.all(np.isnan(result[:20]))             # needs 20 completed days
    assert np.isfinite(result[20:, 0]).all()
    assert np.isnan(result[:, 1]).all()              # never a valid day -> never a score, never inf
    finite = result[np.isfinite(result)]
    assert finite.max() < 1e3                         # |ret| ~1% over 5e7 CNY -> ~0.02 per 1e8, no explosion
    # stock 2: the gated days simply shrink the count; 13 valid of 20 (< 15) -> missing, then recovers
    assert np.isnan(result[20, 2]) and np.isfinite(result[32, 2])


def test_window_mean_matches_direct_definition_and_uses_only_completed_days():
    panel = _panel(30, 1, seed=11)
    close, pre_close, amount = panel["close"][:, 0], panel["preClose"][:, 0], panel["amount"][:, 0]
    daily = np.abs(close / pre_close - 1.0) / (amount / 1e8)
    result = CompletedAmihudIlliquidity20().calc_batch(panel)[:, 0]
    for t in range(20, 30):
        np.testing.assert_allclose(result[t], daily[t - 20:t].mean(), rtol=1e-12)
    # day T itself never enters the score at T
    panel["close"][25, 0] = 1e6
    changed = CompletedAmihudIlliquidity20().calc_batch(panel)[:, 0]
    np.testing.assert_allclose(changed[:26], result[:26], rtol=1e-12)
    assert changed[26] != result[26]


def test_registered_as_twelfth_production_factor():
    assert PRODUCTION_FACTOR_NAMES[-1] == "CompletedAmihudIlliquidity20"
    assert len(PRODUCTION_FACTOR_NAMES) == 12
    definition = get_factor_definition("CompletedAmihudIlliquidity20")
    assert definition.metadata.required_fields == ("close", "preClose", "amount")
    assert definition.metadata.hist_days == 20
    assert definition.metadata.score_semantics == "continuous"
