import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from core.factors.helpers.interface import FactorResult, BaseFactor
from core.factors.helpers.batch_norm import BatchNormFactor
from core.factors.SmallCap import SmallCap
from utils.stock.format import format_qmt_date
from ..logger import core_logger

DEFAULT_FACTOR_CLASSES = [SmallCap]


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

        self.factors: List[BaseFactor] = []
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
        method: str = 'softmax'
    ) -> Dict[str, Dict[str, float]]:
        normalized = {}
        for factor_name, raw_scores in self.factor_scores.items():
            temperature = temperatures.get(factor_name, 1.0)
            norm_scores = BatchNormFactor.batch_normalize(
                raw_scores=raw_scores,
                temperature=temperature,
                method=method
            )
            normalized[factor_name] = norm_scores
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


_RUNTIME_DIR = Path(__file__).resolve().parents[2] / "data" / "runtime"


def _load_runtime_npz(dates: List[datetime]) -> dict | None:
    """加载覆盖指定日期范围的 runtime npz 文件（支持多文件合并）。"""
    if not _RUNTIME_DIR.exists():
        return None

    min_date = np.datetime64(min(dt.date() for dt in dates))
    # 为最后一个 signal_date 的 trade_date（次日）预留缓冲
    max_date = np.datetime64(max(dt.date() for dt in dates)) + np.timedelta64(7, 'D')

    npz_files = sorted(_RUNTIME_DIR.glob("runtime_*.npz"))
    parts = []
    for npz_path in npz_files:
        try:
            data = dict(np.load(npz_path, allow_pickle=False))
            d0, d1 = data['trade_dates'][0], data['trade_dates'][-1]
            if d0 <= max_date and d1 >= min_date:
                parts.append(data)
                core_logger.info(f"  {npz_path.name}: {len(data['trade_dates'])}d x {len(data['stock_codes'])}s")
        except Exception:
            continue

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    core_logger.info(f"合并 {len(parts)} 个 npz 文件...")

    # 所有 NPZ 构建时使用相同的 stock_codes（全量 K线文件），只需拼接
    first_codes = parts[0]['stock_codes']
    codes_match = all(np.array_equal(p['stock_codes'], first_codes) for p in parts[1:])

    all_dates = np.concatenate([p['trade_dates'] for p in parts])
    all_dates = np.unique(all_dates)
    all_dates.sort()

    if codes_match:
        n_stocks = len(first_codes)
        merged = {
            'stock_codes': first_codes,
            'trade_dates': all_dates,
        }
        offsets_list = [np.searchsorted(all_dates, p['trade_dates']) for p in parts]
        _2D_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'amount',
                      'total_share', 'eps', 'roe', 'profit_yoy', 'revenue_yoy',
                      'operating_cf_ps', 'gross_margin', 'retail_net_flow', 'st_mask']
        for field in _2D_FIELDS:
            if field not in parts[0]:
                continue
            dtype = np.bool_ if field == 'st_mask' else np.float64
            fill = False if field == 'st_mask' else np.nan
            arr = np.full((len(all_dates), n_stocks), fill, dtype=dtype)
            for pi, p in enumerate(parts):
                arr[offsets_list[pi]] = p[field]
            merged[field] = arr
        # 1D fields: take from last part (most complete)
        for field in ['issue_price']:
            if field in parts[0]:
                merged[field] = parts[-1][field]
    else:
        # 慢路径：NPZ 股票列表不一致，需要 union（理论上不会走到这里）
        all_stocks = []
        seen = set()
        for p in parts:
            for s in p['stock_codes']:
                s_str = str(s)
                if s_str not in seen:
                    seen.add(s_str)
                    all_stocks.append(s_str)
        n_stocks = len(all_stocks)
        stock_to_idx = {s: i for i, s in enumerate(all_stocks)}
        merged = {
            'stock_codes': np.array(all_stocks, dtype='U12'),
            'trade_dates': all_dates,
        }
        offsets_list = [np.searchsorted(all_dates, p['trade_dates']) for p in parts]
        _2D_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'amount',
                      'total_share', 'eps', 'roe', 'profit_yoy', 'revenue_yoy',
                      'operating_cf_ps', 'gross_margin', 'retail_net_flow', 'st_mask']
        for field in _2D_FIELDS:
            if field not in parts[0]:
                continue
            dtype = np.bool_ if field == 'st_mask' else np.float64
            fill = False if field == 'st_mask' else np.nan
            arr = np.full((len(all_dates), n_stocks), fill, dtype=dtype)
            for pi, p in enumerate(parts):
                p_stocks = [str(s) for s in p['stock_codes']]
                col_idx = np.array([stock_to_idx.get(s, -1) for s in p_stocks])
                valid = col_idx >= 0
                if not valid.any():
                    continue
                for di in range(len(offsets_list[pi])):
                    arr[offsets_list[pi][di], col_idx[valid]] = p[field][di, valid]
            merged[field] = arr
        if 'issue_price' in parts[0]:
            arr = np.full(n_stocks, np.nan, dtype=np.float64)
            for pi, p in enumerate(parts):
                p_stocks = [str(s) for s in p['stock_codes']]
                for j, s in enumerate(p_stocks):
                    t = stock_to_idx.get(s, -1)
                    if t >= 0 and np.isnan(arr[t]) and not np.isnan(p['issue_price'][j]):
                        arr[t] = p['issue_price'][j]
            merged['issue_price'] = arr

    core_logger.info(f"合并完成: {len(all_dates)}d x {len(merged['stock_codes'])}s")
    return merged


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
    all_scores: dict[str, pd.DataFrame] = {}
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
        scores_df = f.calc_batch(panel)
        all_scores[name] = scores_df

    core_logger.info(f"因子批量计算完成 ({time.time() - t0:.1f}s)")

    # 构建每日 TopN 实例
    t1 = time.time()
    # 预计算列索引数组，避免逐股 .iloc[] 调用
    valid_stock_cols = np.array([stock_indices[s] for s in valid_stocks], dtype=np.intp)
    result = []
    for i, dt in enumerate(valid_dates):
        date_idx = date_indices[i]
        precomputed: dict[str, dict[str, FactorResult]] = {}
        for name in all_scores:
            scores_row = all_scores[name].iloc[date_idx, valid_stock_cols].values
            isnan = np.isnan(scores_row)
            batch: dict[str, FactorResult] = {}
            for j, s in enumerate(valid_stocks):
                if not isnan[j]:
                    batch[s] = FactorResult(score=float(scores_row[j]), err=None)
                else:
                    batch[s] = FactorResult(score=None, err="nan")
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
