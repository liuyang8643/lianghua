"""多周期多配置验证：batch合法性 + lightweight收益 与 原版一致性"""
import numpy as np
from datetime import date, datetime
from pathlib import Path
from data.db.delist import get_delist_stock_info
from core.strategies.runtime import load_runtime_npz
from utils.stock.time import get_trading_date_span
from utils.stock.legality import LegalityValidator
from testback.main import (
    _backtest_direct, _compute_factor_scores, _compute_list_dates,
    _precompute_limit_helpers, _batch_limit_check,
)
import core.factors as _all_factors

# ── 测试场景 ──
# (start, end, weights, buy_n, sell_m, hp, label)
SCENARIOS = [
    # 短周期不同年份
    (date(2020,1,1), date(2020,12,31), {'TrueMarketCap': 1.0}, 10, 10, 1, '2020_全年_top10'),
    (date(2021,1,1), date(2021,12,31), {'TrueMarketCap': 1.0}, 10, 10, 1, '2021_全年_top10'),
    (date(2020,1,1), date(2021,12,31), {'TrueMarketCap': 1.0}, 10, 10, 1, '2020-21_两年_top10'),
    # 不同持仓数
    (date(2022,1,1), date(2022,12,31), {'TrueMarketCap': 1.0}, 25, 25, 1, '2022_top25'),
    (date(2022,1,1), date(2022,12,31), {'TrueMarketCap': 1.0}, 5, 5, 1, '2022_top5'),
    # holding_period > 1
    (date(2020,1,1), date(2022,12,31), {'TrueMarketCap': 1.0}, 20, 20, 20, '2020-22_hp20'),
    # 多因子
    (date(2024,1,1), date(2024,12,31), {'TrueMarketCap': 0.5, 'CloseMom21D': 0.5}, 15, 15, 1, '2024_双因子'),
    # 更多年份+多因子
    (date(2018,1,1), date(2020,12,31), {'ROE': 0.4, 'TrueMarketCap': 0.3, 'ProfitYoy': 0.3}, 20, 20, 1, '2018-20_三因子'),
    (date(2023,1,1), date(2025,12,31), {'TMC_GARP_Mult': 1.0}, 15, 15, 5, '2023-25_hp5'),
]

def run_scenario(start, end, weights, buy_n, sell_m, hp, label):
    bt_list = [datetime.combine(d, datetime.min.time()) for d in get_trading_date_span(start, end)]
    npz_dir = Path('data/runtime')
    npz_files = sorted(npz_dir.glob('runtime_*.npz'))
    all_stocks = [str(s) for s in np.load(npz_files[0], allow_pickle=False)['stock_codes']]

    factor_classes = []
    for fname in weights:
        cls = getattr(_all_factors, fname, None)
        if cls is None:
            raise ValueError(f"因子 {fname} 不存在")
        factor_classes.append(cls)

    temps = {k: 1.0 for k in weights}
    scores_result = _compute_factor_scores(bt_list, all_stocks, weights=weights, factor_classes=factor_classes)
    data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices = scores_result
    list_dates_map = _compute_list_dates(data['stock_codes'], data['open'], data['trade_dates'])

    r_normal = _backtest_direct(data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
        weights=weights, buy_n=buy_n, sell_m=sell_m, temperatures=temps, holding_period=hp,
        list_dates_map=list_dates_map, lightweight=False)
    r_light = _backtest_direct(data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
        weights=weights, buy_n=buy_n, sell_m=sell_m, temperatures=temps, holding_period=hp,
        list_dates_map=list_dates_map, lightweight=True)

    dr_n, dr_l = r_normal['daily_returns'], r_light['daily_returns']
    diffs = [abs(a - b) for a, b in zip(dr_n, dr_l) if abs(a - b) > 0.0001]
    return_allowed = abs(r_normal['total_return'] - r_light['total_return']) < 0.005
    cleared_ok = r_normal['cleared_positions_count'] == r_light['cleared_positions_count']
    ok = len(diffs) == 0 and return_allowed and cleared_ok

    print(f"[{'OK' if ok else 'FAIL'}] {label}: {len(valid_dates)}天 {len(valid_stocks)}股")
    print(f"      收益 normal={r_normal['total_return']:.4f}% light={r_light['total_return']:.4f}%")
    print(f"      daily_returns: {len(dr_n)}条, max_diff={max(diffs) if diffs else 0:.6f}")
    print(f"      清仓 normal={r_normal['cleared_positions_count']} light={r_light['cleared_positions_count']}")
    if not ok:
        if diffs:
            print(f"      !! daily_returns差异: {len(diffs)}/{len(dr_n)}")
        if not cleared_ok:
            print(f"      !! 清仓数不一致")
    return ok, len(dr_n), len(valid_stocks)

# ── 合法性逐股验证 ──
def verify_legality_batch(data, start_d, end_d):
    """在指定日期范围内，每只股票每天都做 buy/sell 比对 batch vs LegalityValidator"""
    npz_codes = [str(s) for s in data['stock_codes']]
    stock_indices = {c: i for i, c in enumerate(npz_codes)}
    open_all, close_all = data['open'], data['close']
    high_all, low_all = data['high'], data['low']
    st_all = data.get('st_mask')
    issue_price_all = data.get('issue_price')
    list_dates_map = _compute_list_dates(data['stock_codes'], data['open'], data['trade_dates'])
    board_type, base_ratio, list_tidx = _precompute_limit_helpers(data, stock_indices, list_dates_map)

    validator = LegalityValidator(
        st_mask=data.get('st_mask'), stock_codes=data.get('stock_codes'),
        trade_dates=data.get('trade_dates'), list_dates=list_dates_map)

    tdp = [d.astype('datetime64[D]').item() for d in data['trade_dates']]
    test_range = [(i, d) for i, d in enumerate(tdp) if start_d <= d <= end_d]

    total, mismatches = 0, 0
    for tidx, trade_date in test_range:
        if tidx == 0:
            continue
        valid = np.where(~np.isnan(open_all[tidx]) & (open_all[tidx] > 0))[0]
        if len(valid) == 0:
            continue
        for is_buy in [True, False]:
            batch_ok, _ = _batch_limit_check(
                [npz_codes[i] for i in valid], valid, tidx, trade_date,
                board_type, base_ratio, list_tidx,
                open_all, close_all, high_all, low_all, st_all, issue_price_all, is_buy)
            for j, si in enumerate(valid):
                code = npz_codes[si]
                bar = {
                    'open': float(open_all[tidx, si]), 'high': float(high_all[tidx, si]),
                    'low': float(low_all[tidx, si]), 'close': float(close_all[tidx, si]),
                    'preClose': float(close_all[tidx - 1, si]),
                    'issuePrice': float(issue_price_all[si]) if issue_price_all is not None else np.nan,
                    'suspendFlag': 0,
                }
                res = validator.check_buy(code, datetime.combine(trade_date, datetime.min.time()), bar=bar) if is_buy \
                     else validator.check_sell(code, datetime.combine(trade_date, datetime.min.time()), bar=bar)
                total += 1
                if res.allowed != bool(batch_ok[j]):
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"  MISMATCH {code} {trade_date} {'BUY' if is_buy else 'SELL'} orig={res.allowed} batch={batch_ok[j]} reason={res.reason}")
    return total, mismatches

if __name__ == '__main__':
    print("=" * 60)
    print("验证 1: 合法性 batch vs LegalityValidator")
    print("=" * 60)
    data = load_runtime_npz([datetime.combine(d, datetime.min.time()) for d in get_trading_date_span(date(1991,1,1), date(2026,5,21))])
    periods = [
        (date(2010,1,1), date(2010,3,31), '2010年(创业板10%,科创未开,北交未开)'),
        (date(2015,1,1), date(2015,3,31), '2015年(创业板10%, IPO44%)'),
        (date(2020,7,1), date(2020,9,30), '2020年(创业板转20%前后)'),
        (date(2021,1,1), date(2021,3,31), '2021年(主板+科创板+创业板20%)'),
        (date(2023,4,1), date(2023,6,30), '2023年(主板注册制豁免窗口)'),
        (date(2024,1,1), date(2024,3,31), '2024年(北交所30%+全面注册制)'),
    ]
    for sd, ed, desc in periods:
        total, mismatches = verify_legality_batch(data, sd, ed)
        pct = 100 * mismatches / max(1, total)
        status = "OK" if mismatches == 0 else "FAIL"
        print(f"  [{status}] {desc}: {total}次对比, {mismatches}差异 ({pct:.4f}%)")

    print()
    print("=" * 60)
    print("验证 2: lightweight vs normal 收益计算")
    print("=" * 60)
    all_ok = True
    total_days, total_stocks = 0, 0
    for start, end, weights, buy_n, sell_m, hp, label in SCENARIOS:
        ok, nd, ns = run_scenario(start, end, weights, buy_n, sell_m, hp, label)
        all_ok = all_ok and ok
        total_days += nd
        total_stocks += ns

    print()
    print(f"总计: {len(SCENARIOS)} 个场景, {total_days} 调仓日")
    print("全部通过!" if all_ok else "有失败!")
