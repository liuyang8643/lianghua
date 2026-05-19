"""
退市股票 K 线数据源完整测试

结论：
1. mootdx (通达信): 退市股 → 无数据
2. xtdata get_market_data_ex: 退市股 → OHLC 全为 0（仅日期+成交量）
3. xtdata download_history_data + get_local_data: 退市股 → ✅ 完整真实数据
"""
import pandas as pd
from datetime import date

from core.database.delist import get_delist_stock_info

# 按退市年份分层抽样
# 早期(1990s-2005) + 中期(2006-2019) + 近期(2020-2025) + 活跃对照组
SAMPLE = [
    # 早期退市（需先 download_history_data 才能拿到）
    ('000003.SZ', True),   # PT金田A, 上市1991, 退市2002
    ('000013.SZ', True),   # *ST石化A, 上市1992, 退市2004
    ('000015.SZ', True),   # PT中浩A, 上市1992, 退市2001
    # 中期退市
    ('000024.SZ', False),  # 招商地产, 上市1993, 退市2015
    ('000033.SZ', False),  # 新都退, 上市1994, 退市2017
    ('000018.SZ', False),  # 神城A退, 上市1992, 退市2020
    # 近期退市
    ('000038.SZ', False),  # 大通退, 上市1994, 退市2023
    ('000005.SZ', False),  # ST星源, 上市1990, 退市2024
    ('000040.SZ', False),  # *ST旭蓝, 上市1994, 退市2025
    # 对照组
    ('000001.SZ', False),  # 平安银行（活跃）
]


def test_mootdx(code: str):
    """mootdx direct bars"""
    from mootdx.quotes import Quotes
    from core.database.history import _to_mootdx_code

    market = Quotes.factory()
    bare = _to_mootdx_code(code)
    MAX_OFFSET = 800
    all_bars = []
    offset = 0
    while True:
        try:
            bars = market.bars(symbol=bare, frequency=9, start=offset, offset=MAX_OFFSET)
        except Exception:
            break
        if bars is None or bars.empty:
            break
        all_bars.append(bars)
        if len(bars) < MAX_OFFSET:
            break
        offset += MAX_OFFSET
    if not all_bars:
        return None
    combined = pd.concat(all_bars)
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined


def test_xtdata_local(code: str, need_download: bool = False):
    """xtdata get_local_data (with optional download)"""
    from xtquant import xtdata

    if need_download:
        xtdata.download_history_data(code, '1d', '19900101', '')

    local = xtdata.get_local_data(
        stock_list=[code], period='1d',
        field_list=['open', 'high', 'low', 'close', 'volume', 'amount'],
        start_time='19900101', end_time='', count=-1)

    if local and code in local and local[code] is not None and not local[code].empty:
        df = local[code]
        dts = pd.to_datetime(df.index, format='%Y%m%d')
        return df, dts
    return None, None


def fmt_date(d):
    return d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)


def main():
    delist_info = get_delist_stock_info()
    print(f"{'代码':<12} {'名称':<10} {'上市':>10} {'退市':>10} {'mootdx':>20} {'xtdata(local)':>25} {'覆盖评价'}")
    print("=" * 120)

    for code, need_dl in SAMPLE:
        d = delist_info.get(code)
        list_date = d.list_date if d else date(1990, 1, 1)
        delist_date = d.delist_date if d else date(2026, 12, 31)
        name = d.name if d else '活跃'

        # mootdx
        mdf = test_mootdx(code)
        mootdx_str = '无数据'
        if mdf is not None and not mdf.empty:
            mootdx_str = f"{mdf.index.min().date()}~{mdf.index.max().date()}"

        # xtdata local
        xdf, xdts = test_xtdata_local(code, need_download=need_dl)
        xt_str = '无数据'
        coverage = ''
        if xdf is not None and xdts is not None:
            c_ok = (xdf['close'] > 0).sum()
            xt_str = f"{fmt_date(xdts.min())}~{fmt_date(xdts.max())} ({c_ok}有效)"
            # 覆盖评价
            start_gap = (xdts.min().date() - list_date).days if hasattr(xdts.min(), 'date') else (xdts.min() - pd.Timestamp(list_date)).days
            end_gap = (delist_date - xdts.max().date()).days if hasattr(xdts.max(), 'date') else (pd.Timestamp(delist_date) - xdts.max()).days
            if abs(start_gap) < 10 and abs(end_gap) < 10:
                coverage = '✅ 完整'
            elif abs(start_gap) < 365 and abs(end_gap) < 30:
                coverage = '⚠️ 基本完整'
            else:
                coverage = f'❌ 缺口(前{start_gap}d,后{end_gap}d)'

        print(f"{code:<12} {name:<10} {str(list_date):>10} {str(delist_date):>10} {mootdx_str:>20} {xt_str:>25} {coverage}")

    # 总结
    print("\n" + "=" * 60)
    print("结论:")
    print("  1. mootdx(通达信): 退市股拿不到任何K线")
    print("  2. xtdata get_market_data_ex: 退市股OHLC全为0（仅日期+量）")
    print("  3. xtdata download_history_data + get_local_data: ✅ 可拿到完整K线")
    print("     - 1990年代退市股需先 download_history_data")
    print("     - 2005年后退市股通常已有本地缓存，直接 get_local_data 即可")
    print("  4. 现有 data/k-line/*.parquet (退市股): 用错了 get_market_data_ex → 全是0")


if __name__ == '__main__':
    main()
