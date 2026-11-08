from datetime import datetime
import numpy as np
import pytest
from core.scoring import FactorScoreMatrices, scores_to_ranks, select_topn
from core.prefilter import select_prefilter_candidates, prefilter_quote_codes


def matrices(raw):
    scores = FactorScoreMatrices()
    for name, values in raw.items():
        scores.raw_scores[name] = np.array(values, dtype=np.float32)
        scores[name] = scores_to_ranks(scores.raw_scores[name])
    return scores


def test_outside_quotes_do_not_change_candidate_ranks_or_ties():
    # Same candidate raw values, radically different outside market coverage.
    sparse = matrices({'A': [[4, 4, 1, np.nan, np.nan]],
                       'B': [[1, 5, 4, np.nan, np.nan]]})
    full = matrices({'A': [[4, 4, 1, 100, 2]], 'B': [[1, 5, 4, 3, 0]]})
    args = (0, ['a', 'b', 'c'], np.array([0, 1, 2]), {'A': .9, 'B': .6}, 3)
    a, sa = select_topn(sparse, *args, rank_within_candidates=True)
    b, sb = select_topn(full, *args, rank_within_candidates=True)
    assert a == b
    np.testing.assert_array_equal(sa, sb)


def test_nan_remains_invalid_and_does_not_become_low_positive_rank():
    scores = matrices({'A': [[np.nan, 2, 1, 999]]})
    _, values = select_topn(scores, 0, ['a','b','c'], np.array([0,1,2]),
                           {'A': 1}, 3, rank_within_candidates=True)
    np.testing.assert_array_equal(values, [0, 1, .5])


def test_shortlist_uses_previous_full_scores_and_adds_held_quotes_only():
    scores = matrices({'A': [[3,2,1,9], [1,2,3,9]]})
    si = {s:i for i,s in enumerate(['a','b','c','held'])}
    candidates = select_prefilter_candidates(scores,0,['a','b','c'],si,{'A':1},2,{'held':100})
    assert candidates == ['a','b']
    assert prefilter_quote_codes(candidates,{'held':100}) == ['a','b','held']
    assert candidates == ['a','b']  # quote expansion must not mutate buy eligibility
    assert select_prefilter_candidates(scores,0,['a','b','c'],si,{'A':1},1,{'c':100}) == ['a','c']


@pytest.mark.parametrize('idx', [1,2])
def test_first_requested_backtest_day_uses_actual_previous_session(monkeypatch, idx):
    from core import backtest
    scores = matrices({'A': [[3,2,1], [1,2,3], [3,1,2]]})
    monkeypatch.setattr(backtest,'get_delist_stock_info',lambda:{})
    monkeypatch.setattr(backtest,'LegalityChecker',lambda *a,**kw:None)
    class ReachedPlanner(Exception): pass
    def capture(**kw):
        assert kw['valid_stocks'] == (['a','b'] if idx == 1 else ['b','c'])
        assert kw['rank_within_candidates']
        raise ReachedPlanner()
    monkeypatch.setattr(backtest,'build_rebalance_day',capture)
    with pytest.raises(ReachedPlanner):
        backtest._backtest_direct({},scores,[datetime(2026,9,30)],[idx],
            ['a','b','c'],{'a':0,'b':1,'c':2},weights={'A':1},buy_n=1,sell_m=1,prefilter_n=2)


def test_missing_previous_session_is_not_silently_full_market():
    with pytest.raises(ValueError,match='previous trading session'):
        select_prefilter_candidates({},-1,[],{}, {},300,{})


@pytest.mark.parametrize('sellable', [0,100])
def test_outside_holding_quote_used_for_sale_without_buy_eligibility(sellable):
    from core.strategy import build_rebalance_day
    from datetime import date
    class Checker:
        def check(self, columns, trade_idx, signal_date, is_buy):
            return np.ones(len(columns),dtype=bool), columns
    data={'open':np.array([[10.,20.]]),'close':np.array([[10.,20.]])}
    scores=matrices({'F':[[1.,999.]]})
    plan=build_rebalance_day(data=data,all_scores=scores,date_idx=0,trade_idx=0,
        signal_date=date(2026,9,30),valid_stocks=['000001.SZ'],valid_cols=np.array([0]),
        stock_indices={'000001.SZ':0,'600000.SH':1},weights={'F':1},buy_n=1,sell_m=1,
        checker=Checker(),positions={'600000.SH':100},sellable_volumes={'600000.SH':sellable},
        cash=0,rebalance=False,market_order_freeze=False,rank_within_candidates=True)
    assert plan.prices['600000.SH']==20.
    assert '600000.SH' not in plan.buy_n_stocks
    assert '600000.SH' not in plan.buy_orders
    assert bool(plan.sell_orders)==bool(sellable)
