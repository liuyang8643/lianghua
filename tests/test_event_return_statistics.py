import numpy as np
import pytest

from env.metrics import summarize_event_forward_returns
from env.simulator import settlement_economics


def test_event_weights_missing_chains_and_split_right_censoring():
    events = np.array([[True, True], [True, False], [True, True]])
    gross = np.array([[1.1, 1.3], [0.9, np.nan]])
    stats = summarize_event_forward_returns(events, gross, horizons=(1, 2, 4))
    assert stats['1']['observed_event_count'] == 3
    assert stats['1']['right_censored_count'] == 2
    assert stats['1']['mean_return'] == pytest.approx(0.1)
    assert stats['1']['signal_day_equal_weight_mean_return'] == pytest.approx(0.05)
    assert stats['2']['observed_event_count'] == 1
    assert stats['2']['missing_price_chain_count'] == 1
    assert stats['2']['right_censored_count'] == 3
    assert stats['2']['mean_return'] == pytest.approx(-0.01)
    assert stats['4']['mean_return'] is None
    assert stats['4']['right_censored_count'] == 5


def test_event_returns_reuse_corporate_action_economics():
    # Two-for-one split halves prices without an economic loss.
    economics = settlement_economics(current_mark=[[10.]], current_close=[[10.]],
                                      next_preclose=[[5.]], next_open=[[5.5]], diagnostics=False)
    stats = summarize_event_forward_returns(np.array([[True], [False]]),
                                            economics.gross_return, horizons=(1,))
    assert stats['1']['mean_return'] == pytest.approx(0.1)


def test_reject_misaligned_returns_and_invalid_horizons():
    with pytest.raises(ValueError):
        summarize_event_forward_returns(np.ones((2, 2), dtype=bool), np.ones((2, 2)), horizons=(1,))
    with pytest.raises(ValueError):
        summarize_event_forward_returns(np.ones((2, 2), dtype=bool), np.ones((1, 2)), horizons=(0,))
