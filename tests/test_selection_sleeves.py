import inspect
from datetime import date, datetime

import numpy as np
import pytest

from core.backtest import _backtest_direct, _compute_factor_scores
from core.scoring import (
    FactorScoreMatrices,
    select_selection_sleeves_legal,
    select_topn_legal,
    validate_selection_sleeves,
)
from core.strategy import build_rebalance_day, build_strategy_day


class _AllowAllChecker:
    def check(self, stock_indices, *args, **kwargs):
        return np.ones(len(stock_indices), dtype=bool), {}


def _matrices(core, trend, core_valid=None, trend_valid=None):
    core = np.asarray([core], dtype=np.float32)
    trend = np.asarray([trend], dtype=np.float32)
    if core_valid is None:
        core_valid = np.ones_like(core, dtype=bool)
    else:
        core_valid = np.asarray([core_valid], dtype=bool)
    if trend_valid is None:
        trend_valid = np.ones_like(trend, dtype=bool)
    else:
        trend_valid = np.asarray([trend_valid], dtype=bool)
    return FactorScoreMatrices(
        {'Core': core, 'Trend': trend},
        factor_validity={'Core': core_valid, 'Trend': trend_valid},
    )


def _select(scores, stocks, sleeves, buy_n):
    cols = np.arange(len(stocks), dtype=np.intp)
    return select_selection_sleeves_legal(
        all_scores=scores,
        score_idx=0,
        valid_stocks=stocks,
        valid_cols=cols,
        selection_sleeves=sleeves,
        buy_n=buy_n,
        sell_m=buy_n,
        checker=_AllowAllChecker(),
        trade_idx=0,
        signal_date=date(2024, 1, 2),
        day_open=np.full(len(stocks), 10.0),
    )


def test_sparse_trend_validity_does_not_filter_core_sleeve():
    stocks = ['A', 'B', 'C', 'D']
    sleeves = [
        {'name': 'core', 'slots': 2, 'weights': {'Core': 1.0}},
        {'name': 'trend', 'slots': 1, 'weights': {'Trend': 1.0}},
    ]
    scores = _matrices(
        core=[4.0, 3.0, 2.0, 1.0],
        trend=[4.0, 3.0, 2.0, 1.0],
        trend_valid=[False, False, True, False],
    )

    buy, sell, _, _ = _select(scores, stocks, sleeves, buy_n=3)

    assert buy == ['A', 'B', 'C']
    assert sell == buy


def test_later_sleeve_skips_already_selected_core_code():
    stocks = ['A', 'B', 'C']
    sleeves = [
        {'name': 'core', 'slots': 1, 'weights': {'Core': 1.0}},
        {'name': 'trend', 'slots': 1, 'weights': {'Trend': 1.0}},
    ]
    scores = _matrices(
        core=[3.0, 2.0, 1.0],
        trend=[3.0, 2.0, 1.0],
    )

    buy, sell, _, _ = _select(scores, stocks, sleeves, buy_n=2)

    assert buy == ['A', 'B']
    assert sell == buy


def test_common_filter_is_applied_to_every_sleeve():
    stocks = ['A', 'B', 'C', 'D']
    scores = _matrices(
        core=[4.0, 3.0, 2.0, 1.0],
        trend=[4.0, 3.0, 2.0, 1.0],
    )
    cols = np.arange(len(stocks), dtype=np.intp)
    buy, _, _, _ = select_selection_sleeves_legal(
        all_scores=scores,
        score_idx=0,
        valid_stocks=stocks,
        valid_cols=cols,
        selection_sleeves=[
            {'name': 'core', 'slots': 1, 'weights': {'Core': 1.0}},
            {'name': 'trend', 'slots': 1, 'weights': {'Trend': 1.0}},
        ],
        buy_n=2,
        sell_m=2,
        checker=_AllowAllChecker(),
        trade_idx=0,
        signal_date=date(2024, 1, 2),
        day_open=np.full(len(stocks), 10.0),
        common_filter_mask=np.array([False, True, False, True]),
    )

    assert buy == ['B', 'D']


def test_sparse_trend_leaves_cash_and_target_denominator_at_buy_n():
    stocks = [f'{index:06d}.SZ' for index in range(31)]
    cols = np.arange(len(stocks), dtype=np.intp)
    core_valid = np.zeros((1, len(stocks)), dtype=bool)
    core_valid[0, :24] = True
    trend_valid = np.zeros((1, len(stocks)), dtype=bool)
    trend_valid[0, 24] = True
    scores = FactorScoreMatrices(
        {
            'Core': np.asarray(
                [np.arange(len(stocks), 0, -1)], dtype=np.float32
            ),
            'Trend': np.asarray(
                [np.arange(len(stocks), 0, -1)], dtype=np.float32
            ),
        },
        factor_validity={'Core': core_valid, 'Trend': trend_valid},
    )
    data = {
        'open': np.full((1, len(stocks)), 10.0),
        'close': np.full((1, len(stocks)), 10.0),
    }

    plan = build_rebalance_day(
        data=data,
        all_scores=scores,
        date_idx=0,
        trade_idx=0,
        signal_date=date(2024, 1, 2),
        valid_stocks=stocks,
        valid_cols=cols,
        stock_indices={code: index for index, code in enumerate(stocks)},
        weights={'Core': 1.0, 'Trend': 0.0},
        buy_n=30,
        sell_m=30,
        checker=_AllowAllChecker(),
        positions={},
        sellable_volumes={},
        cash=300_000.0,
        market_order_freeze=False,
        selection_sleeves=[
            {'name': 'core', 'slots': 24, 'weights': {'Core': 1.0}},
            {'name': 'trend', 'slots': 6, 'weights': {'Trend': 1.0}},
        ],
    )

    assert len(plan.buy_n_stocks) == 25
    assert plan.sell_m_stocks == plan.buy_n_stocks
    assert plan.base_target == pytest.approx(10_000.0)
    assert sum(plan.buy_orders.values()) < 30 * 1_000


def test_normal_selection_path_is_unchanged_without_sleeves():
    stocks = ['A', 'B', 'C']
    cols = np.arange(3, dtype=np.intp)
    scores = {'Core': np.array([[0.5, 1.0, 0.25]], dtype=np.float32)}
    checker = _AllowAllChecker()
    expected = select_topn_legal(
        scores,
        0,
        stocks,
        cols,
        {'Core': 1.0},
        2,
        2,
        checker,
        0,
        date(2024, 1, 2),
        np.full(3, 10.0),
        {code: index for index, code in enumerate(stocks)},
    )

    plan = build_rebalance_day(
        data={'open': np.full((1, 3), 10.0), 'close': np.full((1, 3), 10.0)},
        all_scores=scores,
        date_idx=0,
        trade_idx=0,
        signal_date=date(2024, 1, 2),
        valid_stocks=stocks,
        valid_cols=cols,
        stock_indices={code: index for index, code in enumerate(stocks)},
        weights={'Core': 1.0},
        buy_n=2,
        sell_m=2,
        checker=checker,
        positions={},
        sellable_volumes={},
        cash=20_000.0,
        market_order_freeze=False,
    )

    assert plan.buy_n_stocks == expected[0]
    assert plan.sell_m_stocks == expected[1]
    np.testing.assert_array_equal(plan.final_score, expected[2])
    assert plan.t1_ranking == expected[3]


@pytest.mark.parametrize(
    ('sleeves', 'buy_n', 'message'),
    [
        (
            [
                {'name': 'same', 'slots': 1, 'weights': {'Core': 1.0}},
                {'name': 'same', 'slots': 1, 'weights': {'Trend': 1.0}},
            ],
            2,
            'duplicate',
        ),
        (
            [{'name': 'core', 'slots': 0, 'weights': {'Core': 1.0}}],
            1,
            'positive integer',
        ),
        (
            [{'name': 'core', 'slots': 1, 'weights': {'Missing': 1.0}}],
            1,
            'does not exist',
        ),
        (
            [{'name': 'core', 'slots': 1, 'weights': {'Core': 0.0}}],
            1,
            'all be zero',
        ),
        (
            [{'name': 'core', 'slots': 1, 'weights': {'Core': 1.0}}],
            2,
            'sum to buy_n',
        ),
    ],
)
def test_invalid_selection_sleeves_fail_closed(sleeves, buy_n, message):
    with pytest.raises(ValueError, match=message):
        validate_selection_sleeves(
            sleeves,
            buy_n,
            available_factor_names={'Core', 'Trend'},
        )


def test_sleeve_factor_is_computed_when_top_level_weight_is_zero():
    class Core:
        hist_days = 0

        def calc_batch(self, panel):
            return panel['core']

    class Trend:
        hist_days = 0

        def calc_batch(self, panel):
            return panel['trend']

    data = {
        'stock_codes': np.array(['A', 'B']),
        'trade_dates': np.array(['2024-01-02'], dtype='datetime64[D]'),
        'open': np.array([[10.0, 11.0]]),
        'core': np.array([[2.0, 1.0]]),
        'trend': np.array([[np.nan, 3.0]]),
    }
    result = _compute_factor_scores(
        [datetime(2024, 1, 2)],
        ['A', 'B'],
        {'Core': 1.0, 'Trend': 0.0},
        [Core, Trend],
        data=data,
        selection_sleeves=[
            {'name': 'core', 'slots': 1, 'weights': {'Core': 1.0}},
            {'name': 'trend', 'slots': 1, 'weights': {'Trend': 1.0}},
        ],
        buy_n=2,
    )

    all_scores, filter_masks = result[1], result[2]
    assert set(all_scores) == {'Core', 'Trend'}
    assert all_scores.factor_validity['Trend'].tolist() == [[False, True]]
    assert '_nan_union' not in filter_masks


def test_strategy_day_and_direct_backtest_share_core_selection_path():
    strategy_source = inspect.getsource(build_strategy_day)
    direct_source = inspect.getsource(_backtest_direct)
    rebalance_source = inspect.getsource(build_rebalance_day)

    assert 'build_rebalance_day(' in strategy_source
    assert 'build_rebalance_day(' in direct_source
    assert 'select_selection_sleeves_legal(' in rebalance_source
