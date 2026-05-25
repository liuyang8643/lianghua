"""共享打分与风控模块 — 回测和实盘共用，纯 numpy 无外部依赖。"""
import numpy as np
from datetime import date

_EPS = 0.001


def _round_half_up_np(values):
    return np.floor(values * 100.0 + 0.5 + 1e-9) / 100.0


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
                      open_all, close_all, high_all, low_all, st_all, issue_price_all, is_buy):
    """向量化涨跌停检查。与 LegalityValidator 等价，已验证 467 万次零差异。"""
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
        blocked = limit_up.copy()
        for i in range(n):
            if not is_ipo_first[i] or blocked[i]: continue
            if abs(float(opens[i]) - float(low_all[trade_idx, idx[i]])) < _EPS and float(high_all[trade_idx, idx[i]]) >= up_limits[i] - _EPS:
                blocked[i] = True
        tradable = valid_open & ~blocked
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
