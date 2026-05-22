import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import numpy as np

from core.factors.SmallCap import SmallCap
from utils.stock.format import format_qmt_date
from ..logger import core_logger
from .runtime import load_runtime_npz

DEFAULT_FACTOR_CLASSES = [SmallCap]

_load_runtime_npz = load_runtime_npz


class FactorResult(TypedDict):
    score: Optional[float]
    err: Optional[str]


def _rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Cross-sectional rank normalize per row. No Python loops.
    Higher score → rank closer to 1.0. NaN → 0.0."""
    nans = np.isnan(scores)
    filled = np.where(nans, np.inf, scores)
    order = np.argsort(filled, axis=1)
    positions = np.empty(order.shape, dtype=np.float32)
    row_idx = np.arange(order.shape[0])[:, None]
    positions[row_idx, order] = np.arange(order.shape[1], dtype=np.float32)
    n_valid = np.sum(~nans, axis=1)
    return np.where(nans, 0.0, positions / np.maximum(n_valid[:, None] - 1, 1))


class TopN:
    """多因子选股策略 - 选取综合得分排名前N的股票"""

    def __init__(
        self,
        stock_list: List[str],
        base_date: datetime,
        weights: Dict[str, float] = None,
        factor_classes: List[type] = None,
        dividend_type: str = 'back',
        _precomputed_scores: Dict[str, Dict[str, FactorResult]] = None,
    ):
        self.stock_list = stock_list
        self.base_date = base_date
        self.weights = weights
        self.dividend_type = dividend_type

        self._factor_classes: List[type] = factor_classes or DEFAULT_FACTOR_CLASSES

        self.factors = []
        self._factor_names: List[str] = []
        for f_cls in self._factor_classes:
            f = f_cls()
            name = f.__class__.__name__
            if self.weights is None or self.weights.get(name, 0.0) != 0:
                self.factors.append(f)
                self._factor_names.append(name)

        self.max_hist_days = max((f.hist_days for f in self.factors), default=0)

        if _precomputed_scores is not None:
            self.factor_scores = _precomputed_scores
        else:
            self.factor_scores: Dict[str, Dict[str, FactorResult]] = {
                name: {} for name in self._factor_names
            }

    def get_normalized_scores(
        self,
        temperatures: Dict[str, float],
        method: str = 'rank'
    ) -> Dict[str, Dict[str, float]]:
        """Rank normalize + temperature. Uses raw per-stock scores from self.factor_scores."""
        normalized = {}
        for factor_name, raw_scores in self.factor_scores.items():
            temp = temperatures.get(factor_name, 1.0)
            if not raw_scores:
                normalized[factor_name] = {}
                continue

            stocks, scores = [], []
            for stock, result in raw_scores.items():
                s = result.get('score')
                e = result.get('err')
                if e is None and s is not None:
                    stocks.append(stock)
                    scores.append(s)

            if not stocks:
                normalized[factor_name] = {}
                continue

            n = len(stocks)
            arr = np.array(scores)
            order = np.argsort(arr)[::-1]
            norm = np.empty(n, dtype=np.float64)
            norm[order] = 1.0 - (np.arange(n) / n)
            if temp != 1.0:
                norm = norm ** (1.0 / temp)
            normalized[factor_name] = dict(zip(stocks, norm.tolist()))
        return normalized

    def _get_full_ordered_stocks(
        self,
        weights: Dict[str, float] = None,
        temperatures: Dict[str, float] = None,
        norm_method: str = 'rank',
    ) -> List[str]:
        weights = self.weights if weights is None else weights
        temperatures = temperatures or {}

        normed_temps = frozenset(
            (k, v) for k, v in (temperatures or {}).items() if v != 1.0
        )
        tag = (frozenset(weights.items()) if weights else frozenset(), normed_temps, norm_method)
        if tag in getattr(self, '_ordered_cache', {}):
            return self._ordered_cache[tag]

        normalized_scores = self.get_normalized_scores(temperatures, norm_method)

        stock_scores = []
        for stock_code in self.stock_list:
            score = 0.0
            valid_factors = 0
            for factor_name in self.factor_scores.keys():
                if stock_code in normalized_scores.get(factor_name, {}):
                    weight = weights.get(factor_name, 0.0)
                    if weight == 0:
                        continue
                    score += normalized_scores[factor_name][stock_code] * weight
                    valid_factors += 1
            if valid_factors > 0:
                stock_scores.append((stock_code, score))

        if not stock_scores:
            core_logger.warning(f"没有股票有有效因子分数！日期: {format_qmt_date(self.base_date)}")
            return []

        stock_scores.sort(key=lambda x: x[1], reverse=True)
        ordered = [s for s, _ in stock_scores]

        if not hasattr(self, '_ordered_cache'):
            self._ordered_cache = {}
        self._ordered_cache[tag] = ordered

        core_logger.info(
            f"{format_qmt_date(self.base_date)} TopN选股完成: "
            f"候选{len(ordered)}只"
        )
        if ordered:
            top_5 = stock_scores[:min(5, len(stock_scores))]
            core_logger.debug(
                "Top 5: " + ", ".join([f"{s}({sc:.6f})" for s, sc in top_5])
            )

        return ordered

    def get_ordered_stocks(
        self,
        n: int,
        weights: Dict[str, float] = None,
        temperatures: Dict[str, float] = None,
        norm_method: str = 'rank',
    ) -> List[str]:
        ordered = self._get_full_ordered_stocks(weights, temperatures, norm_method)
        return ordered[:n] if n < len(ordered) else ordered


def compute_topn_range(
    backtest_datetime_list: List[datetime],
    stock_list: List[str],
    weights: Dict[str, float] = None,
    factor_classes: List[type] = None,
    dividend_type: str = 'back',
) -> List["TopN"]:
    """计算指定日期范围的 TopN 实例列表。

    加载 runtime npz，批量计算所有因子分数，构建每日 TopN 实例。
    """
    if not backtest_datetime_list:
        return []

    factor_classes = factor_classes or DEFAULT_FACTOR_CLASSES
    first_d = backtest_datetime_list[0].strftime('%Y%m%d')
    last_d = backtest_datetime_list[-1].strftime('%Y%m%d')
    core_logger.info(f"TopN 范围 {first_d}~{last_d}，加载 runtime npz...")

    data = _load_runtime_npz(backtest_datetime_list)
    if data is None:
        raise FileNotFoundError(
            f"未找到覆盖 {first_d}~{last_d} 的 runtime npz 文件，"
            f"请先运行 python data/build_runtime.py"
        )

    npz_stocks = [str(s) for s in data['stock_codes']]
    stock_indices = {c: i for i, c in enumerate(npz_stocks)}
    valid_stocks = [s for s in stock_list if s in stock_indices]

    npz_dates = data['trade_dates']
    date_to_idx = {}
    for i, d in enumerate(npz_dates):
        date_to_idx[d.astype('datetime64[D]').item()] = i

    n_npz_dates = len(npz_dates)
    date_indices = []
    valid_dates = []
    for dt in backtest_datetime_list:
        d = dt.date() if hasattr(dt, 'date') else dt
        di = date_to_idx.get(d)
        if di is None:
            continue
        # 需要下一交易日作为执行日（trade_idx = di + 1），超出 npz 范围则跳过
        if di + 1 >= n_npz_dates:
            continue
        date_indices.append(di)
        valid_dates.append(dt)

    if not valid_dates:
        core_logger.warning("没有交易日落在 runtime npz 日期范围内")
        return []

    # 日期列表（Python date，用于 DataFrame index）
    py_dates = []
    for d in npz_dates:
        ts = d.astype('datetime64[D]').item()
        py_dates.append(ts if hasattr(ts, 'date') else ts)

    t0 = time.time()

    # 构建因子元数据
    factor_meta = []
    for f_cls in factor_classes:
        f = f_cls()
        name = f.__class__.__name__
        if weights is not None and weights.get(name, 0.0) == 0:
            continue
        factor_meta.append((name, f))

    # 批量计算所有因子分数
    all_scores: dict[str, np.ndarray] = {}
    for name, f in factor_meta:
        panel = {
            'stock_codes': npz_stocks,
            'trade_dates': py_dates,
            'open': data['open'],
            'high': data.get('high', data['open']),
            'low': data.get('low', data['close']),
            'close': data['close'],
            'volume': data['volume'],
            'amount': data['amount'],
            'issue_price': data['issue_price'],
            'st_mask': data['st_mask'],
            'total_share': data['total_share'],
            'eps': data['eps'],
            'roe': data['roe'],
            'profit_yoy': data['profit_yoy'],
            'revenue_yoy': data['revenue_yoy'],
            'operating_cf_ps': data['operating_cf_ps'],
            'gross_margin': data['gross_margin'],
        }
        raw = f.calc_batch(panel)
        all_scores[name] = _rank_normalize(raw.astype(np.float32, copy=False))

    core_logger.info(f"因子批量计算+归一化完成 ({time.time() - t0:.1f}s)")

    # 构建每日 TopN 实例
    t1 = time.time()
    # 预计算列索引数组，避免逐股 .iloc[] 调用
    valid_stock_cols = np.array([stock_indices[s] for s in valid_stocks], dtype=np.intp)
    result = []
    for i, dt in enumerate(valid_dates):
        date_idx = date_indices[i]
        precomputed: dict[str, dict[str, FactorResult]] = {}
        for name in all_scores:
            scores_row = all_scores[name][date_idx, valid_stock_cols]
            batch: dict[str, FactorResult] = {
                s: FactorResult(score=float(scores_row[j]), err=None)
                for j, s in enumerate(valid_stocks)
            }
            precomputed[name] = batch

        topn = TopN(valid_stocks, dt, weights=weights, factor_classes=factor_classes,
                   dividend_type=dividend_type, _precomputed_scores=precomputed)

        # 存储 trade_date 价格数据供回测执行层使用（signal_date + 1）
        trade_idx = date_idx + 1
        if trade_idx < n_npz_dates:
            topn._trade_arrays = {
                'open': data['open'][trade_idx],
                'high': data['high'][trade_idx],
                'low': data['low'][trade_idx],
                'close': data['close'][trade_idx],
                'pre_close': data['close'][trade_idx - 1],
                'volume': data['volume'][trade_idx],
                'st_mask': data['st_mask'][trade_idx],
            }
            topn._trade_stock_idx = stock_indices

            topn.get_ordered_stocks(1, weights=weights, norm_method='rank')
            result.append(topn)

    core_logger.info(f"TopN 实例构建完成 ({time.time() - t1:.1f}s), 共 {len(result)} 天")
    return result
