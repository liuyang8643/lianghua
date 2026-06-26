"""因子评测器：动态加载因子 .py → 复用 core 回测 → 训练期夏普（fitness）。

只调用 core.backtest 暴露的函数（传动态类对象，不经 registry），不复制任何回测逻辑。
回测口径与现框架一致：Top-N 归一化打分、默认 top20、每日均匀持仓、多退少补。
"""
import importlib.util
from datetime import date, datetime

import numpy as np

from core.backtest import (
    _backtest_direct,
    _compute_factor_scores,
    _compute_list_dates,
)
from core.metrics import compute_core_metrics
from core.runtime import load_runtime_npz, load_runtime_stock_codes
from utils.stock.time import get_trading_date_span

_NEG_INF = float('-inf')

# 连续分数硬约束（fitness 前的闸门）：
#   每只 base_valid 股票必须拿到“连续、无重复”的有限分数，否则 TopN 退化、各因子结果趋同。
MIN_COVERAGE = 0.90   # 有限分股票 / base_valid 的每日中位覆盖率下限
MAX_TIE_RATIO = 0.05  # base_valid 内有限分数的每日中位重复值比例上限（连续因子≈0-3%，离散因子≈90%+）
_MIN_DAY_STOCKS = 30  # 只在 base_valid 足够多的交易日上评估


def _parse_date(s: str) -> date:
    s = s.replace('-', '')
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _load_valid_issue_price_stocks() -> set:
    """加载 issue_price 有效的股票代码集合（用于统一股票池过滤）。"""
    from pathlib import Path
    import pandas as pd
    ip_path = Path(__file__).resolve().parent.parent / 'data' / 'issue_price' / 'issue_price.parquet'
    if not ip_path.exists():
        return set()
    df = pd.read_parquet(ip_path)
    return {str(c) for c in df['stock_code']}


def build_universe(start: str, end: str, pool_prefixes=None):
    """构建回测日期序列与股票池（一次性，可在多个体评测间复用）。

    自动过滤 issue_price 缺失的股票，确保所有因子在同构股票池上竞争。
    """
    dts = [
        datetime.combine(d, datetime.min.time())
        for d in get_trading_date_span(_parse_date(start), _parse_date(end))
    ]
    stocks = load_runtime_stock_codes()
    if pool_prefixes:
        stocks = [s for s in stocks if s.startswith(tuple(pool_prefixes))]
    ip_stocks = _load_valid_issue_price_stocks()
    if ip_stocks:
        stocks = [s for s in stocks if s[:6] in ip_stocks]
    return dts, stocks


def load_factor_class(path, class_name: str):
    spec = importlib.util.spec_from_file_location(f'_llmga_{class_name}', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def _raw_scores(factor_cls, dates, data=None):
    """复用 _compute_factor_scores 同款 panel 构造，返回 (raw 分数矩阵, base_valid 掩码)。

    data 可传入预加载面板（避免重复加载 NPZ）。
    """
    if data is None:
        max_lookback = factor_cls.hist_days or None
        data = load_runtime_npz(dates, max_lookback=max_lookback)
    if data is None:
        raise FileNotFoundError('runtime npz 未覆盖回测区间')
    panel = {
        **data,
        'stock_codes': [str(s) for s in data['stock_codes']],
        'trade_dates': [d.astype('datetime64[D]').item() for d in data['trade_dates']],
    }
    raw = np.asarray(factor_cls().calc_batch(panel), dtype=np.float64)
    open_ = np.asarray(data['open'], dtype=np.float64)
    base_valid = np.isfinite(open_) & (open_ >= 2.0) & ~np.asarray(data['st_mask'], dtype=bool)
    return raw, base_valid


def check_continuity(factor_cls, dates, data=None) -> tuple[float, float]:
    """连续分数硬约束校验。返回 (覆盖率中位数, tie比例中位数)；不达标抛 ValueError。"""
    raw, base_valid = _raw_scores(factor_cls, dates, data=data)
    finite = np.isfinite(raw) & base_valid
    n_bv = base_valid.sum(axis=1)
    day_idx = np.where(n_bv >= _MIN_DAY_STOCKS)[0]
    if day_idx.size == 0:
        raise ValueError('无足够 base_valid 交易日，无法评估连续性')

    cover_med = float(np.median(finite.sum(axis=1)[day_idx] / n_bv[day_idx]))
    ties = []
    for i in day_idx:
        vals = raw[i][finite[i]]
        if vals.size >= _MIN_DAY_STOCKS:
            ties.append(1.0 - np.unique(vals).size / vals.size)
    tie_med = float(np.median(ties)) if ties else 1.0

    if cover_med < MIN_COVERAGE:
        raise ValueError(
            f'分数覆盖率过低 median={cover_med:.1%} < {MIN_COVERAGE:.0%}：'
            f'因子把过多 base_valid 股票打成 NaN（应给全市场连续打分，勿用基本面门槛过滤）')
    if tie_med > MAX_TIE_RATIO:
        raise ValueError(
            f'离散分数 tie比例 median={tie_med:.1%} > {MAX_TIE_RATIO:.0%}：'
            f'base_valid 内出现大量重复分数（要求连续、唯一，勿离散化/分桶/置常数）')
    return cover_med, tie_med


def evaluate_detailed(factor_cls, name: str, dates, all_stocks, buy_n: int = 20, *,
                      check: bool = True, want_sig: bool = False, data=None) -> dict:
    """跑一次回测，返回训练期指标 + 完整明细（每日日期/收益/topN 持仓）。

    check=True（GA 路径）：回测前做连续分数硬约束校验（覆盖率 + 无 tie），不达标抛 ValueError。
    check=False（回填路径）：跳过闸门，为已存在的离散/连续因子都落明细。
    want_sig=True：顺手用已算好的 rank 矩阵生成全截面指纹（多样性用），返回 signature/sig_shape。
    data：可传入预加载 NPZ 面板，整轮 GA 复用同一份，避免重复加载。
    计算失败（无有效交易日）返回 sharpe=-inf 且明细为空。
    """
    if check:
        check_continuity(factor_cls, dates, data=data)

    weights = {name: 1.0}

    scored = _compute_factor_scores(dates, all_stocks, weights, [factor_cls], data=data)
    if scored is None:
        return {'sharpe': _NEG_INF, 'annualized': 0.0, 'max_dd': 0.0,
                'n_trades': 0, 'dates': [], 'daily_returns': [], 'topn': [],
                'signature': None, 'sig_shape': None}

    data, all_scores, _, valid_dates, date_indices, valid_stocks, stock_indices = scored
    list_dates_map = _compute_list_dates(data['stock_codes'], data['open'], data['trade_dates'])

    bt = _backtest_direct(
        data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
        weights=weights, buy_n=buy_n, sell_m=buy_n,
        list_dates_map=list_dates_map, lightweight=True,
    )
    m = compute_core_metrics(bt['daily_returns'])

    sig, sig_shape = None, None
    if want_sig:
        from factor_db import similarity
        rank_aligned = all_scores[name][date_indices]              # (n_valid_dates, n_full_stocks)
        sig = similarity.signature(rank_aligned)
        sig_shape = (int(rank_aligned.shape[0]), int(rank_aligned.shape[1]))

    return {
        'sharpe': m['sharpe'],
        'annualized': m['annualized'],
        'max_dd': m['max_drawdown'],
        'n_trades': bt.get('cleared_positions_count', 0),
        'dates': [d.date().isoformat() for d in valid_dates],
        'daily_returns': bt.get('daily_returns', []),
        'topn': bt.get('daily_topn', []),
        'signature': sig,
        'sig_shape': sig_shape,
    }


def evaluate(factor_cls, name: str, dates, all_stocks, buy_n: int = 20) -> dict:
    """跑一次回测，返回训练期夏普等指标（不含明细）。计算失败返回 sharpe=-inf。"""
    d = evaluate_detailed(factor_cls, name, dates, all_stocks, buy_n)
    return {'sharpe': d['sharpe'], 'annualized': d['annualized'],
            'max_dd': d['max_dd'], 'n_trades': d['n_trades']}
