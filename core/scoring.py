"""共享打分与风控模块 — 回测和实盘共用，纯 numpy 无外部依赖。

红线（CLAUDE.md §2.2）：T 日开盘契约——选股/合法性检查 **只允许使用 open[T]**，
当日的 close/high/low/volume/amount 全部视为前视野泄露。需要"前收"统一用 close[T-1]。
"""
import numpy as np
from datetime import date
from typing import Optional

_EPS = 0.001


def _round_half_up_np(values):
    return np.floor(values * 100.0 + 0.5 + 1e-9) / 100.0


def select_topn(
    all_scores: dict,
    score_idx: int,
    valid_stocks: list[str],
    valid_cols: np.ndarray,
    weights: dict[str, float],
    temperatures: dict[str, float],
    top_n: int,
    force_codes: Optional[list[str]] = None,
) -> tuple[list[str], np.ndarray]:
    """加权打分 + 截面排序 + 取前 N。回测和实盘必须共用此函数。

    Args:
        all_scores: {factor_name: ranks_mat (n_dates, n_full_stocks)} 归一化排名矩阵
        score_idx: 信号日索引（回测=date_idx, 实盘=score_date_idx, 都等于 T 日 NPZ 索引）
        valid_stocks: 候选股 codes（已经过 stock_indices 过滤）
        valid_cols: 候选股在 ranks_mat 中的列索引 np.intp 数组
        weights: {factor_name: weight}
        temperatures: {factor_name: temperature}
        top_n: 取前 N 只
        force_codes: 强制纳入的 codes（保留位次，前置）；None 表示不启用

    Returns:
        (topn_stocks, final_score_arr)
            topn_stocks: 选出的 top-N 股票代码列表
            final_score_arr: 全候选股的加权打分（按 valid_stocks 顺序，用于落地 plan）
    """
    final_score = np.zeros(len(valid_stocks))
    for name, ranks_mat in all_scores.items():
        w = weights.get(name, 0.0)
        if w == 0:
            continue
        ranks = ranks_mat[score_idx][valid_cols]
        temp = temperatures.get(name, 1.0)
        if temp != 1.0:
            np.power(ranks, 1.0 / temp, out=ranks)
        final_score += ranks * w

    top_idx = np.argsort(-final_score)
    topn = [valid_stocks[i] for i in top_idx[:top_n]]

    if force_codes:
        ordered, seen = [], set()
        for code in force_codes:
            if code and code not in seen:
                seen.add(code)
                ordered.append(code)
        for code in topn:
            if code not in seen:
                seen.add(code)
                ordered.append(code)
        topn = ordered[:top_n]

    return topn, final_score


def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """每日截面排名归一化 (0~1, 1=最优), NaN → 0。原地修改。"""
    n_days = scores.shape[0]
    ranks = np.empty_like(scores, dtype=np.float32)
    for d in range(n_days):
        row = scores[d]
        nans = np.isnan(row)
        valid_mask = ~nans
        n_valid = valid_mask.sum()
        if n_valid == 0:
            ranks[d] = 0.0
            continue
        order = np.argsort(row[valid_mask])[::-1]
        ranks[d, nans] = 0.0
        col_idx = np.where(valid_mask)[0]
        ranks[d, col_idx[order]] = 1.0 - np.arange(n_valid, dtype=np.float32) / n_valid
    return ranks


def batch_limit_check(candidates, candidates_idx, trade_idx, signal_date,
                      board_type, base_ratio, list_tidx,
                      open_all, close_all, st_all, issue_price_all, is_buy):
    """向量化涨跌停检查（T 日开盘成交契约）。

    数据使用：
      - open_all[trade_idx]       ← T 日开盘价（合法，9:30 集合竞价后即可知）
      - close_all[trade_idx - 1]  ← 前一交易日收盘价作为 preclose
      - st_all[trade_idx]         ← T 日 ST 状态（盘前已知，合法）
      - issue_price_all[..]       ← IPO 首日 preclose 兜底

    禁止使用：T 日的 high/low/close/volume/amount，全部是数据泄露。
    实盘 9:30 下单时不可能拿到这些当日 LHC 信息。
    """
    if len(candidates) == 0:
        return np.array([], dtype=bool), {}
    idx = np.asarray(candidates_idx, dtype=np.intp); n = len(idx)
    opens = open_all[trade_idx, idx].astype(np.float64)
    valid_open = ~np.isnan(opens) & (opens > 0)
    if not np.any(valid_open):
        return np.zeros(n, dtype=bool), {'suspended': n}

    precloses = close_all[trade_idx - 1, idx].astype(np.float64) if trade_idx > 0 else np.full(n, np.nan)
    valid_preclose = (trade_idx > 0) & ~np.isnan(precloses) & (precloses > 0)

    st_arr = st_all[trade_idx, idx] if st_all is not None else np.zeros(n, dtype=bool)
    ratios = np.where(st_arr, 0.05, base_ratio[idx])
    boards = board_type[idx]
    cyb_pre = (boards == 1) & (signal_date < date(2020, 8, 24))
    ratios[cyb_pre] = 0.10
    lti = list_tidx[idx]

    is_ipo_first = np.zeros(n, dtype=bool)
    exempt = np.zeros(n, dtype=bool)
    for i in range(n):
        lt = lti[i]
        if lt < 0 or trade_idx < lt: continue
        ds = trade_idx - lt; b = boards[i]
        if b == 3: exempt[i] = (ds == 0)
        elif b == 2: exempt[i] = (signal_date >= date(2019, 7, 22) and ds <= 4)
        elif b == 1:
            if signal_date >= date(2020, 8, 24) and ds <= 4: exempt[i] = True
            elif signal_date < date(2014, 1, 1) and ds == 0: exempt[i] = True
        else:
            if signal_date >= date(2023, 4, 10) and ds <= 4: exempt[i] = True
            elif signal_date < date(2014, 1, 1) and ds == 0: exempt[i] = True
        if not exempt[i] and lt >= 0 and trade_idx == lt and signal_date >= date(2014, 1, 1):
            is_ipo_first[i] = True

    if issue_price_all is not None and np.any(is_ipo_first):
        need_fb = is_ipo_first & ~valid_preclose
        if np.any(need_fb):
            ips = issue_price_all[idx].astype(np.float64)
            vip = need_fb & ~np.isnan(ips) & (ips > 0)
            precloses[vip] = ips[vip]; valid_preclose[vip] = True

    ratios = np.where(is_ipo_first & ~exempt, 0.44, ratios)
    has_limit = valid_preclose & ~exempt

    if is_buy:
        up_limits = np.where(has_limit, _round_half_up_np(precloses * (1.0 + ratios)), np.nan)
        limit_up = valid_open & has_limit & (opens >= up_limits - _EPS)
        # 注：旧版用 T 日 high/low 判断 IPO 首日"开盘没涨停但盘中触及涨停"的秒封场景。
        # 因 high/low 是 T 日盘中数据、实盘 9:30 不可知，移除该判断以避免数据泄露。
        # 副作用：回测会乐观接受 IPO 首日开盘价 < 涨停价的成交，实盘也是同样下单（QMT 撮合决定是否成交）。
        tradable = valid_open & ~limit_up
    else:
        down_limits = np.where(has_limit, _round_half_up_np(precloses * (1.0 - ratios)), np.nan)
        limit_down = valid_open & has_limit & (opens <= down_limits + _EPS)
        tradable = valid_open & ~limit_down

    suspended = np.sum(~valid_open)
    reasons = {'suspended': int(suspended)} if suspended > 0 else {}
    return tradable, reasons


def precompute_limit_helpers(data, stock_indices, list_dates_map=None):
    """从 NPZ 数据预计算涨跌停检查辅助数组。"""
    codes = [str(s) for s in data['stock_codes']]
    n = len(codes)
    bt = np.zeros(n, dtype=np.int8); br = np.full(n, 0.10, dtype=np.float64)
    for i, c in enumerate(codes):
        if c.startswith('300') or c.startswith('301'): bt[i] = 1; br[i] = 0.20
        elif c.startswith('688'): bt[i] = 2; br[i] = 0.20
        elif c.startswith('83') or c.startswith('87') or c.startswith('43') or c.startswith('92'): bt[i] = 3; br[i] = 0.30
    tdp = [d.astype('datetime64[D]').item() for d in data['trade_dates']]
    d2t = {d: i for i, d in enumerate(tdp)}
    lt = np.full(n, -1, dtype=np.int32)
    if list_dates_map:
        for code, ld in list_dates_map.items():
            si = stock_indices.get(code)
            if si is None: continue
            ldi = d2t.get(ld)
            if ldi is None:
                for d in tdp:
                    if d >= ld: ldi = d2t[d]; break
            if ldi is not None: lt[si] = ldi
    return bt, br, lt


def build_legality_context(data, stock_indices, list_dates_map=None):
    """构建 batch_limit_check 所需的全部上下文。回测和实盘必须共用此工厂。

    Args:
        data: load_runtime_npz 返回的 dict
        stock_indices: {code: col_idx}
        list_dates_map: {code: list_date} 可选；回测从 db.delist 等渠道传入

    Returns:
        dict 含 batch_limit_check 所需全部参数：
            open_all, close_all, st_all, issue_price_all,
            board_type, base_ratio, list_tidx
    """
    board_type, base_ratio, list_tidx = precompute_limit_helpers(data, stock_indices, list_dates_map)
    return {
        'open_all': data['open'],
        'close_all': data['close'],
        'st_all': data.get('st_mask'),
        'issue_price_all': data.get('issue_price'),
        'board_type': board_type,
        'base_ratio': base_ratio,
        'list_tidx': list_tidx,
    }
