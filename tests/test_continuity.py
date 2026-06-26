"""连续分数硬约束 check_continuity 的单元测试。

用真实 runtime npz（离线）在一小段区间上验证：
- 连续、覆盖全市场的因子 -> 通过；
- 离散因子（np.sign 大量重复值）-> 因 tie 被拒；
- 过滤型因子（eps>0 才打分，覆盖率低）-> 因覆盖率被拒。
"""
from datetime import date

import numpy as np
import pytest

from llm_ga import evaluator
from utils.stock.time import get_trading_date_span

_DATES = [
    np.datetime64(d).astype('datetime64[s]').item()
    for d in get_trading_date_span(date(2023, 1, 1), date(2023, 6, 30))
]


class _Continuous:
    """全市场连续打分：log(open) 叠加极小权重的 log(总股本) -> 几乎无 tie，覆盖率≈100%。"""
    hist_days = 0

    def calc_batch(self, panel):
        open_ = panel['open'].astype(float)
        ts = panel['total_share'].astype(float)
        med = np.nanmedian(np.where(np.isfinite(ts), ts, np.nan), axis=1, keepdims=True)
        ts = np.where(np.isfinite(ts) & (ts > 0), ts, med)
        ts = np.where(np.isfinite(ts) & (ts > 0), ts, 1.0)
        score = np.log(np.maximum(open_, 1e-6)) + 1.731e-4 * np.log(ts)
        base_valid = np.isfinite(open_) & (open_ >= 2.0) & ~panel['st_mask']
        return np.where(base_valid & np.isfinite(score), score, np.nan)


class _Discrete:
    """离散打分：np.sign(eps) 只有 -1/0/1，大量重复值。"""
    hist_days = 0

    def calc_batch(self, panel):
        open_ = panel['open'].astype(float)
        base_valid = np.isfinite(open_) & (open_ >= 2.0) & ~panel['st_mask']
        score = np.sign(np.where(np.isnan(panel['eps']), 0.0, panel['eps']))
        return np.where(base_valid, score, np.nan)


class _OverFilter:
    """过滤型：只给 eps>0 的股票打分，覆盖率远低于 base_valid。"""
    hist_days = 0

    def calc_batch(self, panel):
        open_ = panel['open'].astype(float)
        base_valid = np.isfinite(open_) & (open_ >= 2.0) & ~panel['st_mask']
        ey = np.where(panel['eps'] > 0, panel['eps'] / open_, np.nan)
        return np.where(base_valid & np.isfinite(ey), ey, np.nan)


def test_continuous_factor_passes():
    cover, tie = evaluator.check_continuity(_Continuous, _DATES)
    assert cover >= evaluator.MIN_COVERAGE
    assert tie <= evaluator.MAX_TIE_RATIO


def test_discrete_factor_rejected_by_tie():
    with pytest.raises(ValueError, match='离散分数'):
        evaluator.check_continuity(_Discrete, _DATES)


def test_overfilter_factor_rejected_by_coverage():
    with pytest.raises(ValueError, match='覆盖率'):
        evaluator.check_continuity(_OverFilter, _DATES)
