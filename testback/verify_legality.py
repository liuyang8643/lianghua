"""快速对比+debug：短周期全量，打印详细差异"""
import numpy as np
from datetime import date, datetime
from utils.stock.time import get_trading_date_span
from utils.stock.legality import LegalityValidator
from core.strategies.runtime import load_runtime_npz
from testback.main import _compute_list_dates

EPSILON = 0.001

def _round_half_up(values):
    return np.floor(values * 100.0 + 0.5 + 1e-9) / 100.0

def _precompute_helpers(data):
    codes = [str(s) for s in data['stock_codes']]
    n = len(codes)
    bt = np.zeros(n, dtype=np.int8); br = np.full(n, 0.10, dtype=np.float64)
    for i, c in enumerate(codes):
        if c.startswith('300') or c.startswith('301'): bt[i] = 1; br[i] = 0.20
        elif c.startswith('688'): bt[i] = 2; br[i] = 0.20
        elif c.startswith('83') or c.startswith('87') or c.startswith('43') or c.startswith('92'): bt[i] = 3; br[i] = 0.30
    tdp = [d.astype('datetime64[D]').item() for d in data['trade_dates']]
    d2t = {d: i for i, d in enumerate(tdp)}
    ldm = _compute_list_dates(data['stock_codes'], data['open'], data['trade_dates'])
    lt = np.full(n, -1, dtype=np.int32)
    if ldm:
        idx_map = {c: i for i, c in enumerate(codes)}
        for code, ld in ldm.items():
            si = idx_map.get(code)
            if si is None: continue
            ldi = d2t.get(ld)
            if ldi is None:
                for d in tdp:
                    if d >= ld: ldi = d2t[d]; break
            if ldi is not None: lt[si] = ldi
    return bt, br, lt, tdp

def batch_limit_check(tidx, signal_date, ci, bt, br, lt, o, c, h, l, st, ip, is_buy):
    idx = np.asarray(ci, dtype=np.intp); n = len(idx)
    opens = o[tidx, idx].astype(np.float64)
    valid_open = ~np.isnan(opens) & (opens > 0)
    if not np.any(valid_open):
        return np.zeros(n, dtype=bool), {}

    precloses = c[tidx - 1, idx].astype(np.float64) if tidx > 0 else np.full(n, np.nan)
    valid_preclose = (tidx > 0) & ~np.isnan(precloses) & (precloses > 0)

    st_arr = st[tidx, idx] if st is not None else np.zeros(n, dtype=bool)
    ratios = np.where(st_arr, 0.05, br[idx])
    boards = bt[idx]
    cyb_pre = (boards == 1) & (signal_date < date(2020, 8, 24))
    ratios[cyb_pre] = 0.10
    lti = lt[idx]

    is_ipo_first = np.zeros(n, dtype=bool)
    exempt = np.zeros(n, dtype=bool)
    for i in range(n):
        lti_i = lti[i]
        if lti_i < 0 or tidx < lti_i: continue
        ds = tidx - lti_i; b = boards[i]
        if b == 3: exempt[i] = (ds == 0)
        elif b == 2: exempt[i] = (signal_date >= date(2019, 7, 22) and ds <= 4)
        elif b == 1:
            if signal_date >= date(2020, 8, 24) and ds <= 4: exempt[i] = True
            elif signal_date < date(2014, 1, 1) and ds == 0: exempt[i] = True
        else:
            if signal_date >= date(2023, 4, 10) and ds <= 4: exempt[i] = True
            elif signal_date < date(2014, 1, 1) and ds == 0: exempt[i] = True
        if not exempt[i] and lti_i >= 0 and tidx == lti_i and signal_date >= date(2014, 1, 1):
            is_ipo_first[i] = True

    # issuePrice fallback 仅 IPO 首日（与原版一致）
    if ip is not None and np.any(is_ipo_first):
        need_fb = is_ipo_first & ~valid_preclose
        if np.any(need_fb):
            ips = ip[idx].astype(np.float64)
            vip = need_fb & ~np.isnan(ips) & (ips > 0)
            precloses[vip] = ips[vip]; valid_preclose[vip] = True

    ratios = np.where(is_ipo_first & ~exempt, 0.44, ratios)
    has_limit = valid_preclose & ~exempt

    if is_buy:
        up_limits = np.where(has_limit, _round_half_up(precloses * (1.0 + ratios)), np.nan)
        limit_up = valid_open & has_limit & (opens >= up_limits - EPSILON)
        blocked = limit_up.copy()
        for i in range(n):
            if not is_ipo_first[i] or blocked[i]: continue
            if abs(float(opens[i]) - float(l[tidx, idx[i]])) < EPSILON and float(h[tidx, idx[i]]) >= up_limits[i] - EPSILON:
                blocked[i] = True
        return valid_open & ~blocked, {}
    else:
        down_limits = np.where(has_limit, _round_half_up(precloses * (1.0 - ratios)), np.nan)
        limit_down = valid_open & has_limit & (opens <= down_limits + EPSILON)
        return valid_open & ~limit_down, {}

# ── 主流程 ──
print("加载...")
data = load_runtime_npz([datetime.combine(d, datetime.min.time()) for d in get_trading_date_span(date(2022, 1, 1), date(2022, 2, 28))])
npz_codes = [str(s) for s in data['stock_codes']]
stock_indices = {c: i for i, c in enumerate(npz_codes)}
open_all = data['open']; close_all = data['close']
high_all = data['high']; low_all = data['low']
st_all = data.get('st_mask'); issue_price_all = data.get('issue_price')
list_dates_map = _compute_list_dates(data['stock_codes'], data['open'], data['trade_dates'])
board_type, base_ratio, list_tidx, trade_dates_py = _precompute_helpers(data)

validator = LegalityValidator(
    st_mask=data.get('st_mask'), stock_codes=data.get('stock_codes'),
    trade_dates=data.get('trade_dates'), list_dates=list_dates_map)

n_dates = len(trade_dates_py)
# 只测 2022-01-01 ~ 2022-02-28
start_d = date(2022, 1, 1); end_d = date(2023, 12, 31)
test_range = [(i, d) for i, d in enumerate(trade_dates_py) if start_d <= d <= end_d]
print(f"日期范围: {test_range[0][1]} ~ {test_range[-1][1]}, {len(test_range)}天")

total = 0; mismatches = []
for tidx, trade_date in test_range:
    if tidx == 0: continue
    trade_datetime = datetime.combine(trade_date, datetime.min.time())
    valid = np.where(~np.isnan(open_all[tidx]) & (open_all[tidx] > 0))[0]
    if len(valid) == 0: continue

    for is_buy in [True, False]:
        batch_ok, _ = batch_limit_check(tidx, trade_date, valid, board_type, base_ratio, list_tidx,
            open_all, close_all, high_all, low_all, st_all, issue_price_all, is_buy)
        for j, si in enumerate(valid):
            code = npz_codes[si]
            bar = {'open': float(open_all[tidx, si]), 'high': float(high_all[tidx, si]),
                   'low': float(low_all[tidx, si]), 'close': float(close_all[tidx, si]),
                   'preClose': float(close_all[tidx - 1, si]),
                   'issuePrice': float(issue_price_all[si]) if issue_price_all is not None else np.nan,
                   'suspendFlag': 0}
            res = validator.check_buy(code, trade_datetime, bar=bar) if is_buy else validator.check_sell(code, trade_datetime, bar=bar)
            orig_allowed = res.allowed
            batch_allowed = bool(batch_ok[j])
            total += 1
            if orig_allowed != batch_allowed:
                mismatches.append((code, trade_date, is_buy, orig_allowed, batch_allowed, float(open_all[tidx, si]), float(close_all[tidx - 1, si]), si))
                if len(mismatches) <= 30:
                    st_val = bool(st_all[tidx, si]) if st_all is not None else False
                    print(f"#{len(mismatches)} {code} {trade_date} {'BUY' if is_buy else 'SELL'} orig={orig_allowed} batch={batch_allowed}")
                    print(f"   open={bar['open']:.2f} preclose={bar['preClose']} st={st_val} board={board_type[si]} ratio={base_ratio[si]} reason={res.reason}")
                    print(f"   list_tidx={list_tidx[si]} tidx={tidx}")

print(f"\n总计: {total} 对比, {len(mismatches)} 差异 ({100*len(mismatches)/max(1,total):.4f}%)")
