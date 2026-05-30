"""深历史价值/质量/成长财务因子（基于 deep_fin_pit.npz，覆盖 1990s~今）。

高分=优先买入。所有比率使用 open[T]（T 日唯一允许价格），财务量为 PIT 滞后值，
不构成数据泄露。辅助 NPZ 由 data/build_deep_fin_runtime.py 生成。
"""
import numpy as np
from pathlib import Path

MIN_RAW_PRICE = 2.0
_AUX_PATH = Path(__file__).resolve().parents[2] / 'data' / 'runtime' / 'deep_fin_pit.npz'
_cache: dict = {}


def _load() -> dict:
    if not _cache:
        d = np.load(_AUX_PATH, allow_pickle=False)
        _cache['dates'] = d['trade_dates'].astype('datetime64[D]')
        for k in d.files:
            if k not in ('trade_dates', 'stock_codes'):
                _cache[k] = d[k]
    return _cache


def _aligned(panel: dict, field: str) -> np.ndarray:
    c = _load()
    pdates = np.array([np.datetime64(dt) for dt in panel['trade_dates']], dtype='datetime64[D]')
    start = int(np.searchsorted(c['dates'], pdates[0]))
    return c[field][start:start + len(pdates)]


def _base_valid(panel: dict) -> np.ndarray:
    raw_open = panel['open']
    return ~np.isnan(raw_open) & (raw_open >= MIN_RAW_PRICE) & ~panel['st_mask']


def _pct_rank(x: np.ndarray) -> np.ndarray:
    """逐日截面百分位排名（高值=高分，1=最优），NaN 保留。纯向量化双 argsort。"""
    nan = np.isnan(x)
    xx = np.where(nan, -np.inf, x.astype(np.float64))
    order = np.argsort(np.argsort(-xx, axis=1), axis=1).astype(np.float32)
    n = (~nan).sum(axis=1, keepdims=True).astype(np.float32)
    r = 1.0 - order / np.where(n > 0, n, 1.0)
    r[nan] = np.nan
    return r


class BookToMarket:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        bps = _aligned(panel, 'bps')
        with np.errstate(divide='ignore', invalid='ignore'):
            score = bps / panel['open']
        valid = _base_valid(panel) & np.isfinite(bps) & (bps > 0)
        return np.where(valid, score, np.nan)


class EarningsYield:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        eps = _aligned(panel, 'eps')
        with np.errstate(divide='ignore', invalid='ignore'):
            score = eps / panel['open']
        valid = _base_valid(panel) & np.isfinite(eps)
        return np.where(valid, score, np.nan)


class CashFlowYield:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        ocfps = _aligned(panel, 'ocfps')
        with np.errstate(divide='ignore', invalid='ignore'):
            score = ocfps / panel['open']
        valid = _base_valid(panel) & np.isfinite(ocfps)
        return np.where(valid, score, np.nan)


class ROEQuality:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        roe = _aligned(panel, 'roe')
        valid = _base_valid(panel) & np.isfinite(roe)
        return np.where(valid, roe, np.nan)


class ProfitGrowth:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        g = _aligned(panel, 'profit_yoy')
        valid = _base_valid(panel) & np.isfinite(g)
        return np.where(valid, g.astype(np.float64), np.nan)


class DeepValueComposite:
    """价值复合：bps/open、eps/open、ocfps/open 三路截面排名平均。高分=便宜。"""
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_ = panel['open']
        with np.errstate(divide='ignore', invalid='ignore'):
            bm = _aligned(panel, 'bps') / open_
            ep = _aligned(panel, 'eps') / open_
            cf = _aligned(panel, 'ocfps') / open_
        bv = _base_valid(panel)
        bm = np.where(bv & np.isfinite(bm), bm, np.nan)
        ep = np.where(bv & np.isfinite(ep), ep, np.nan)
        cf = np.where(bv & np.isfinite(cf), cf, np.nan)
        comp = np.nanmean(np.stack([_pct_rank(bm), _pct_rank(ep), _pct_rank(cf)]), axis=0)
        valid = bv & np.isfinite(comp)
        return np.where(valid, comp, np.nan)


class BookToMarketQuality:
    """账面市值比，但仅在盈利（eps>0）股票中打分，规避价值陷阱。"""
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        bps = _aligned(panel, 'bps')
        eps = _aligned(panel, 'eps')
        with np.errstate(divide='ignore', invalid='ignore'):
            score = bps / panel['open']
        valid = _base_valid(panel) & np.isfinite(bps) & (bps > 0) & np.isfinite(eps) & (eps > 0)
        return np.where(valid, score, np.nan)
