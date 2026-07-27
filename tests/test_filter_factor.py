from datetime import datetime

import numpy as np

from core.backtest import _compute_factor_scores
from core.prefilter import apply_prefilter
from core.scoring import select_topn
from testback.run_ga import _build_worker_filter_masks
from factor_db.factors.AmihudIlliquidity import AmihudIlliquidity
from factor_db.factors.AmountBasedSmallCap import AmountBasedSmallCap
from factor_db.factors.TrueMarketCap import TrueMarketCap
from factor_db.factors.VolumeCV import VolumeCV
from factor_db.factors.filter import FilterLowPrice, FilterST, FilterStarST


def _panel(rows=80, cols=3):
    day = np.arange(rows, dtype=float)[:, None]
    col = np.arange(cols, dtype=float)[None, :]
    return {
        'open': 10.0 + day * 0.01 + col,
        'close': 9.8 + day * 0.01 + col,
        'amount': 1e8 + day * 1e5 + col * 1e6,
        'volume': 1e6 + day * 100.0 + col * 1e4,
        'total_share': np.full((rows, cols), 1e8),
        'st_mask': np.zeros((rows, cols), dtype=bool),
    }


def test_filter_conditions_are_independent_factors():
    panel = _panel()
    panel['open'][-1, 0] = 1.99
    panel['st_mask'][-1, 1] = True

    assert np.isnan(FilterLowPrice().calc_batch(panel)[-1, 0])
    assert np.isfinite(FilterLowPrice().calc_batch(panel)[-1, 1])
    assert np.isnan(FilterST().calc_batch(panel)[-1, 1])
    assert np.isfinite(FilterST().calc_batch(panel)[-1, 0])
    assert np.isnan(FilterStarST().calc_batch(panel)[-1, 1])

    with np.errstate(divide='ignore', invalid='ignore'):
        for factor_cls in (AmihudIlliquidity, AmountBasedSmallCap, TrueMarketCap, VolumeCV):
            scores = factor_cls().calc_batch(panel)
            assert np.isfinite(scores[-1, 0])
            assert np.isfinite(scores[-1, 1])


class Score:
    hist_days = 0

    def calc_batch(self, panel):
        return panel['open']


def test_filter_factors_become_buy_masks_in_factor_pipeline():
    data = {
        'stock_codes': np.array(['A', 'B', 'C']),
        'trade_dates': np.array(['2024-01-02'], dtype='datetime64[D]'),
        'open': np.array([[1.99, 5.0, 5.0]]),
        'st_mask': np.array([[False, True, False]]),
    }

    result = _compute_factor_scores(
        [datetime(2024, 1, 2)], ['A', 'B', 'C'], {'Score': 1.0}, [Score],
        data=data, filter_factor_classes=[FilterST, FilterLowPrice],
    )

    _, _, filter_masks, *_ = result
    assert filter_masks['FilterST'].tolist() == [[True, False, True]]
    assert filter_masks['FilterLowPrice'].tolist() == [[False, True, True]]


def test_factor_missing_counts_follow_backtest_dates():
    class SparseScore:
        hist_days = 0

        def calc_batch(self, panel):
            return np.array([[1.0, np.nan], [np.nan, np.nan]])

    data = {
        'stock_codes': np.array(['A', 'B']),
        'trade_dates': np.array(['2024-01-02', '2024-01-03'], dtype='datetime64[D]'),
    }
    counts = {}
    _compute_factor_scores(
        [datetime(2024, 1, 2), datetime(2024, 1, 3)], ['A', 'B'],
        {'SparseScore': 1.0}, [SparseScore], data=data,
        factor_missing_counts=counts,
    )

    assert counts == {'SparseScore': [1, 2]}


def test_filter_valid_counts_require_an_available_t_day_open():
    class DenseScore:
        hist_days = 0

        def calc_batch(self, panel):
            return np.ones_like(panel['open'])

    data = {
        'stock_codes': np.array(['A', 'B', 'C']),
        'trade_dates': np.array(['2000-01-04', '2000-01-05'], dtype='datetime64[D]'),
        'open': np.array([[10.0, np.nan, np.nan], [10.0, 11.0, np.nan]]),
        'st_mask': np.zeros((2, 3), dtype=bool),
    }
    counts = {}
    _compute_factor_scores(
        [datetime(2000, 1, 4), datetime(2000, 1, 5)], ['A', 'B', 'C'],
        {'DenseScore': 1.0}, [DenseScore], data=data,
        filter_factor_classes=[FilterStarST], factor_missing_counts=counts,
    )

    assert counts == {
        'DenseScore': [2, 1],
        'FilterStarST': [2, 1],
    }


def test_prefilter_uses_the_latest_full_ranking():
    universe = ['A', 'B', 'C', 'D']
    assert apply_prefilter(['D', 'C', 'B', 'A'], 2, universe) == ['C', 'D']
    assert apply_prefilter(['A', 'B', 'C', 'D'], 2, universe) == ['A', 'B']


def test_candidate_ranks_ignore_runtime_stocks_outside_universe():
    class RawFactor:
        hist_days = 0

        def calc_batch(self, panel):
            return panel['raw']

    dt = [datetime(2024, 1, 2)]
    base = {
        'stock_codes': np.array(['A', 'B']),
        'trade_dates': np.array(['2024-01-02'], dtype='datetime64[D]'),
        'raw': np.array([[2.0, 1.0]]),
    }
    extended = {
        'stock_codes': np.array(['A', 'B', 'X']),
        'trade_dates': base['trade_dates'],
        'raw': np.array([[2.0, 1.0, 1000.0]]),
    }
    args = (dt, ['A', 'B'], {'RawFactor': 1.0}, [RawFactor])
    base_scores = _compute_factor_scores(*args, data=base)[1]['RawFactor'][:, :2]
    extended_scores = _compute_factor_scores(*args, data=extended)[1]['RawFactor'][:, :2]

    np.testing.assert_array_equal(base_scores, extended_scores)


def test_filter_mask_never_backfills_filtered_stocks():
    topn, _ = select_topn(
        {'Score': np.array([[1.0, 0.5]], dtype=np.float32)},
        0, ['A', 'B'], np.array([0, 1]), {'Score': 1.0}, 2,
        filter_mask=np.array([False, False]),
    )

    assert topn == []


def test_ga_filter_masks_preserve_raw_filters_and_ignore_zero_weight_factors():
    arrays = {
        '_factor_valid_Active': np.array([[True, False]]),
        '_factor_valid_Disabled': np.array([[False, False]]),
        'FilterST': np.array([[True, False]]),
    }
    config = {
        'weights': {'Active': 1.0, 'Disabled': 0.0},
        'filter_factors': {'FilterST': True},
    }

    masks = _build_worker_filter_masks(
        arrays, {'Active', 'Disabled'}, config,
    )

    assert masks['_active_factor_intersection'].tolist() == [[True, False]]
    assert masks['FilterST'].tolist() == [[True, False]]
