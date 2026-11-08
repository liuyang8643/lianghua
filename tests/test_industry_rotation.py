import numpy as np
import pytest
from env.industry_rotation import run_industry_rotation


def panel():
    dates = np.arange("2012-01-01", "2012-01-17", dtype="datetime64[D]")
    prices = np.full((16, 3), 100.)
    return dates, np.array(["a", "b", "c"]), prices.copy(), prices.copy()


def run(data, **kwargs):
    return run_industry_rotation(*data, signal_start="2012-01-11", end="2012-01-16", **kwargs)


def test_staggered_chain_hand_calculation_includes_overnight_and_cash():
    d, c, op, cl = panel()
    op[11:] = np.array([100, 120, 90, 100, 100])[:, None]
    cl[11:] = np.array([110, 130, 100, 110, 120])[:, None]
    result = run((d, c, op, cl))
    # A buys 0.5/100 on day 11, exits day 12 at 130; B enters day 12.
    expected_a = [.5, .55, .65, .65/90*100, .65/90*110, .65/90*110/100*120]
    expected_b = [.5, .5, .5/120*130, .5/120*100, .5/120*100/100*110, .5/120*100/100*120]
    np.testing.assert_allclose(result.chain_nav, np.array([expected_a, expected_b]).T)
    assert result.trades[6]["date"] == "2012-01-13"
    assert result.trades[6]["phase"] == "close"


@pytest.mark.parametrize("strategy", ["original_top1", "staggered_top3"])
def test_future_perturbation_preserves_prior_actions_and_nav(strategy):
    d, c, op, cl = panel()
    original = run((d, c, op, cl), strategy=strategy)
    op[14:, 1] *= 7
    cl[14:, 1] *= 13
    changed = run((d, c, op, cl), strategy=strategy)
    np.testing.assert_array_equal(original.nav[:4], changed.nav[:4])
    assert original.signals[:4] == changed.signals[:4]  # day 14 entry uses day 13 close
    assert [t for t in original.trades if t["date"] < "2012-01-15"] == [t for t in changed.trades if t["date"] < "2012-01-15"]


def test_original_same_index_continues_without_extra_cost():
    result = run(panel(), strategy="original_top1", cost_rate=.001)
    assert len(result.trades) == 1
    np.testing.assert_allclose(result.nav[1:], 1/1.001)


def test_missing_future_execution_price_fails_not_reranks():
    d, c, op, cl = panel()
    op[11, 0] = np.nan
    with pytest.raises(ValueError, match="missing open"):
        run((d, c, op, cl), strategy="original_top1")


def test_momentum_uses_ten_returns_not_ten_closes():
    d, c, op, cl = panel()
    cl[0, 1] = 50
    result = run((d, c, op, cl), strategy="original_top1")
    assert result.signals[0]["codes"] == ("b",)
    assert result.signals[0]["momentum"] == (1.,)


def test_staggered_costs_debit_each_independent_chain():
    result = run(panel(), cost_rate=.001)
    buy = 1 / 1.001
    sell = .999
    np.testing.assert_allclose(result.chain_nav[2], [.5*buy*sell, .5*buy])
    np.testing.assert_allclose(result.chain_nav[3], [.5*buy*sell*buy, .5*buy*sell])


def test_current_close_changes_next_open_signal_only():
    d, c, op, cl = panel()
    original = run((d, c, op, cl), strategy="original_top1")
    cl[12, 0] = 50  # Marking changes today, but today's open already happened.
    cl[12, 1] = 200
    changed = run((d, c, op, cl), strategy="original_top1")
    assert original.signals[:2] == changed.signals[:2]
    assert [t for t in original.trades if t["date"] <= "2012-01-13"] == [t for t in changed.trades if t["date"] <= "2012-01-13"]
    assert changed.nav[2] == .5
    assert changed.signals[2]["codes"] == ("b",)


def test_calendar_rejects_whole_missing_date_including_warmup():
    import pandas as pd
    from ai.factor_discovery.industry_rotation_backtest import align_industry_calendar
    d, c, op, cl = panel()
    frame = pd.DataFrame(op, index=pd.DatetimeIndex(d), columns=c)
    for missing in (2, 12):
        truncated = frame.drop(frame.index[missing])
        with pytest.raises(ValueError, match="entirely missing required trading dates"):
            align_industry_calendar(truncated, truncated, d, "2012-01-11", "2012-01-16")
    aligned, _, audit = align_industry_calendar(frame, frame, d, "2012-01-11", "2012-01-16")
    assert len(aligned) == 16
    assert audit["warmup_session_count"] == 10
