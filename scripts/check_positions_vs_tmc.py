"""用实际策略因子 + 实际股票池 对比持仓"""
import os
import sys
import glob
import re
from collections import defaultdict
from datetime import date, datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_buy_dates(current_positions):
    log_dir = 'D:/coding/WBR/configs/logs'
    log_files = sorted(glob.glob(os.path.join(log_dir, 'qmt-*.log')))
    buy_records = {}
    pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}).*已提交买入委托:\s*(\d{6}\.[A-Z]{2})\s*\*\s*\d+\s*股'
    )
    for fp in log_files:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    trade_date = datetime.strptime(m.group(1), '%Y-%m-%d').date()
                    code = m.group(2)
                    if code not in buy_records or trade_date < buy_records[code]:
                        buy_records[code] = trade_date
    result = {}
    for code in current_positions:
        if code in buy_records:
            result[code] = buy_records[code]
        else:
            print(f"  ⚠ {code} 未在日志中找到买入记录")
    return result


def load_npz():
    npz_dir = 'D:/coding/WBR/data/runtime'
    files = sorted(glob.glob(os.path.join(npz_dir, 'runtime_*.npz')))
    all_2d = defaultdict(list)
    all_dates = []
    stock_codes = None
    for fp in files:
        d = dict(np.load(fp, allow_pickle=True))
        if stock_codes is None:
            stock_codes = d['stock_codes']
        all_dates.append(d['trade_dates'])
        for key, arr in d.items():
            if key in ('stock_codes', 'trade_dates'):
                continue
            all_2d[key].append(arr)
    result = {
        'stock_codes': stock_codes,
        'trade_dates': np.concatenate(all_dates),
    }
    for key, arrays in all_2d.items():
        result[key] = np.concatenate(arrays, axis=0)
    return result


def get_listed_before(npz_codes, cutoff_date):
    """返回在 cutoff_date 前已上市的股票码"""
    from core.database.stock_list import _get_stock_date_range
    valid = []
    for code in npz_codes:
        dr = _get_stock_date_range(code)
        if dr and dr[0] and dr[0] <= cutoff_date:
            valid.append(code)
    return set(valid)


def compute_g2a_scores(npz_data, target_date, pool_filter=None):
    """SmallCapDailyMVMaskRoe2xBottom10
    score = (1 - size_pct_rank) + 0.20 * roe_pct_rank
    pool_filter: set of allowed stock codes
    """
    trade_dates = npz_data['trade_dates']
    target_dt = np.datetime64(target_date)

    date_idx = None
    for i, d in enumerate(trade_dates):
        if d >= target_dt:
            date_idx = i
            break
    if date_idx is None:
        return None, None, None

    actual_date = trade_dates[date_idx]
    all_stocks = npz_data['stock_codes']
    raw_open = npz_data['open'][date_idx]
    total_share = npz_data['total_share'][date_idx]
    st_mask = npz_data['st_mask'][date_idx]
    roe = npz_data['roe'][date_idx]

    MIN_RAW_PRICE = 2.0
    valid = (
        ~np.isnan(raw_open)
        & (raw_open >= MIN_RAW_PRICE)
        & ~np.isnan(total_share)
        & (total_share > 0)
        & ~st_mask
    )

    # Apply pool filter
    if pool_filter is not None:
        pool_mask = np.array([str(c) in pool_filter for c in all_stocks])
        valid = valid & pool_mask

    valid_idx = np.where(valid)[0]
    n_valid = len(valid_idx)
    if n_valid == 0:
        return None, None, None

    # size percentile rank
    total_mv = np.where(valid, (raw_open * total_share) / 1e8, np.nan)
    mv_sorted = np.argsort(total_mv[valid_idx])
    size_rank = np.full(len(all_stocks), np.nan)
    for pos, arr_i in enumerate(mv_sorted):
        size_rank[valid_idx[arr_i]] = pos / (n_valid - 1) if n_valid > 1 else 0.5

    # ROE percentile rank
    roe_arr = np.array(roe)
    roe_mask = valid & ~np.isnan(roe_arr)
    roe_valid_idx = np.where(roe_mask)[0]
    n_roe = len(roe_valid_idx)
    roe_rank = np.full(len(all_stocks), np.nan)
    if n_roe > 0:
        roe_sorted = np.argsort(roe_arr[roe_valid_idx])
        for pos, arr_i in enumerate(roe_sorted):
            roe_rank[roe_valid_idx[arr_i]] = pos / (n_roe - 1) if n_roe > 1 else 0.5

    roe_filled = np.nan_to_num(roe_rank, nan=0.0)
    raw_score = np.where(valid, (1.0 - size_rank) + 0.20 * roe_filled, np.nan)

    # Rank normalization (same as BatchNormFactor)
    score_arr = raw_score[valid_idx]
    norm_order = np.argsort(score_arr)[::-1]
    norm_score = np.full(len(all_stocks), np.nan)
    for pos, arr_i in enumerate(norm_order):
        norm_score[valid_idx[arr_i]] = 1.0 - (pos / n_valid)

    result = {}
    for i in valid_idx:
        code = str(all_stocks[i])
        result[code] = {
            'raw_score': raw_score[i],
            'norm_score': norm_score[i],
            'size_rank': size_rank[i],
            'roe_rank': roe_filled[i],
            'mv_yi': total_mv[i],
            'open': raw_open[i],
        }

    sorted_by_norm = sorted(result.items(), key=lambda x: x[1]['norm_score'], reverse=True)
    for rank, (code, info) in enumerate(sorted_by_norm, 1):
        info['rank'] = rank

    return actual_date, result


def main():
    current_positions = [
        '002316.SZ', '300163.SZ', '300169.SZ', '300405.SZ', '300417.SZ',
        '300500.SZ', '300535.SZ', '300605.SZ', '300614.SZ', '300621.SZ',
        '300635.SZ', '300665.SZ', '300929.SZ', '301098.SZ', '301167.SZ',
        '600455.SH', '600493.SH', '600561.SH', '600697.SH', '600778.SH',
        '600883.SH', '603717.SH', '603860.SH', '603879.SH', '603880.SH',
    ]

    print("=" * 70)
    print("因子: SmallCapDailyMVMaskRoe2xBottom10 (1 - size_rank + 0.20 * ROE_rank)")
    print("配置: buy_n=25, sell_m=25, temperature=1.0, norm=rank")
    print("股票池: 上证+深证 上市早于买入日")
    print("=" * 70)

    buy_dates = get_buy_dates(current_positions)
    print(f"\n找到 {len(buy_dates)}/{len(current_positions)} 只持仓的买入记录")

    print("\n加载NPZ...")
    npz_data = load_npz()
    print(f"{len(npz_data['trade_dates'])} 交易日, {npz_data['trade_dates'][0]} ~ {npz_data['trade_dates'][-1]}")

    npz_stocks = [str(c) for c in npz_data['stock_codes']]

    date_groups = defaultdict(list)
    for code, bd in buy_dates.items():
        date_groups[bd].append(code)
    for d in sorted(date_groups):
        print(f"  {d}: {len(date_groups[d])} 只")

    from core.database.stock_name import get_stock_name_at_date

    cache = {}
    all_results = []

    for buy_date in sorted(date_groups):
        codes = date_groups[buy_date]
        cache_key = str(buy_date)

        if cache_key in cache:
            actual_date, result, pool_size = cache[cache_key]
        else:
            # 用上市日期过滤池子（模拟当天xtdata实际可用的股票）
            pool = get_listed_before(npz_stocks, buy_date)
            actual_date, result = compute_g2a_scores(npz_data, buy_date, pool_filter=pool)
            cache[cache_key] = (actual_date, result, len(pool))

        actual_date, result, pool_size = cache[cache_key]
        print(f"\n--- 买入日期: {buy_date}, {len(codes)}只 | 池子: {pool_size}只 ---")

        if result is None:
            print(f"  ❌ 无有效数据")
            for c in codes:
                all_results.append({'code': c, 'buy_date': str(buy_date), 'in_top25': False,
                                    'rank': None, 'norm_score': None})
            continue

        valid_in_pool = len(result)
        print(f"  实际交易日: {actual_date}, 有效股票: {valid_in_pool}")

        for code in codes:
            info = result.get(code)
            if info is None:
                name = get_stock_name_at_date(code, actual_date.astype(object) if hasattr(actual_date, 'astype') else actual_date)
                in_top = False
                print(f"  ❌ {code} {name} | 不在有效池（停牌/ST/未上市/低价）")
            else:
                in_top = info['rank'] <= 25
                name = get_stock_name_at_date(code, actual_date.astype(object) if hasattr(actual_date, 'astype') else actual_date)
                flag = "✅" if in_top else "❌"
                print(f"  {flag} {code} {name:8s} | G2A #{info['rank']}/{valid_in_pool} | "
                      f"raw={info['raw_score']:.4f} size_pct={info['size_rank']:.3f} "
                      f"ROE_pct={info['roe_rank']:.3f} MV={info['mv_yi']:.1f}亿")

            all_results.append({
                'code': code,
                'name': name,
                'buy_date': str(buy_date),
                'signal_date': str(actual_date) if actual_date else '',
                'in_top25': in_top,
                'rank': info['rank'] if info else None,
                'norm_score': info['norm_score'] if info else None,
            })

    import pandas as pd
    df = pd.DataFrame(all_results)
    in_count = df['in_top25'].sum()
    total = len(df)
    print(f"\n{'=' * 70}")
    print(f"汇总: {total} 只, G2A Top25命中 {in_count}/{total} ({in_count/total*100:.1f}%)")
    print(f"{'=' * 70}")

    # Top25完整列表（最近买入日）
    last_buy_date = max(date_groups.keys())
    actual_date, result, _ = cache[str(last_buy_date)]
    if result:
        sorted_stocks = sorted(result.items(), key=lambda x: x[1]['norm_score'], reverse=True)
        print(f"\n{actual_date} G2A Top25 完整列表 ({len(result)} 池中):")
        print(f"{'-' * 60}")
        for i, (code, info) in enumerate(sorted_stocks[:25]):
            name = get_stock_name_at_date(code, actual_date.astype(object) if hasattr(actual_date, 'astype') else actual_date)
            held = "🔵" if code in buy_dates else "  "
            print(f"  {held} #{i+1:2d} {code} {name:8s} | raw={info['raw_score']:.4f} "
                  f"size={info['size_rank']:.3f} ROE={info['roe_rank']:.3f} MV={info['mv_yi']:.1f}亿")


if __name__ == '__main__':
    main()
