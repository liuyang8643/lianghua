"""构建 runtime npz 文件：将所有 parquet 中间数据合并为 stocks × dates 维度的 numpy 数组。

输出格式（按 CLAUDE.md 规范）：
  np.savez_compressed(
    'runtime_{start}_{end}.npz',
    stock_codes=np.array(..., dtype='U12'),
    trade_dates=np.array(..., dtype='datetime64[D]'),
    open=ndarray(n_dates, n_stocks),
    high=ndarray(n_dates, n_stocks),
    low=ndarray(n_dates, n_stocks),
    close=ndarray(n_dates, n_stocks),
    volume=ndarray(n_dates, n_stocks),
    amount=ndarray(n_dates, n_stocks),
    issue_price=ndarray(n_stocks,),        # 每股发行价（元），从 akshare 新浪财经获取
    stock_names=ndarray(n_stocks,),        # 股票最新简称，U16，从 CNINFO 缓存获取
    st_mask=ndarray(bool, n_dates, n_stocks),
    total_share=ndarray(n_dates, n_stocks),
    eps=ndarray(n_dates, n_stocks),
    roe=ndarray(n_dates, n_stocks),
    profit_yoy=ndarray(n_dates, n_stocks),
    revenue_yoy=ndarray(n_dates, n_stocks),
    operating_cf_ps=ndarray(n_dates, n_stocks),
    gross_margin=ndarray(n_dates, n_stocks),
  )

用法:
  uv run python data/build_runtime.py
  uv run python data/build_runtime.py --start 2020-01-01 --end 2024-12-31
  uv run python data/build_runtime.py --max-stocks 100  # 调试模式
"""
import argparse
import time
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "runtime"
KLINE_DIR = DATA_DIR / "k-line"


def load_kline_panel(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
) -> dict[str, np.ndarray]:
    """读取所有 k-line parquet 文件，构建 OHLCV 面板。

    每只股票一个 parquet 文件（{code}.parquet），
    通过二分查找将 trade_dates 对齐到 k-line 的 time 列。
    """
    n_dates = len(trade_dates)
    n_stocks = len(stock_codes)

    arrays = {
        'open': np.full((n_dates, n_stocks), np.nan, dtype=np.float64),
        'high': np.full((n_dates, n_stocks), np.nan, dtype=np.float64),
        'low': np.full((n_dates, n_stocks), np.nan, dtype=np.float64),
        'close': np.full((n_dates, n_stocks), np.nan, dtype=np.float64),
        'volume': np.full((n_dates, n_stocks), np.nan, dtype=np.float64),
        'amount': np.full((n_dates, n_stocks), np.nan, dtype=np.float64),
    }

    # trade_dates 转为 date 列表方便比较
    trade_date_list = trade_dates.astype('datetime64[D]')

    t0 = time.time()
    last_log = t0
    loaded = 0
    missing = 0

    for j, code in enumerate(stock_codes):
        parquet_path = KLINE_DIR / f"{code}.parquet"
        if not parquet_path.exists():
            missing += 1
            continue

        df = pd.read_parquet(parquet_path)
        if df.empty:
            missing += 1
            continue

        # k-line time 是 ms 时间戳，转为 numpy 日期（去除时分秒只留日期）
        time_ms = df['time'].values.astype(np.int64)
        kline_dates = time_ms.astype('datetime64[ms]').astype('datetime64[D]')

        # mootdx 返回降序（最新在前），sort 为升序后对齐
        sort_idx = np.argsort(kline_dates)
        kline_dates = kline_dates[sort_idx]
        open_arr = df['open'].values[sort_idx]
        high_arr = df['high'].values[sort_idx]
        low_arr = df['low'].values[sort_idx]
        close_arr = df['close'].values[sort_idx]
        vol_arr = df['volume'].values[sort_idx]
        amt_arr = df['amount'].values[sort_idx]

        # 二分查找对齐: trade_dates ∈ kline_dates
        indices = np.searchsorted(kline_dates, trade_date_list, side='left')
        valid = (indices >= 0) & (indices < len(kline_dates))
        if not valid.any():
            loaded += 1
            continue
        indices = np.clip(indices, 0, len(kline_dates) - 1)
        match = valid & (kline_dates[indices] == trade_date_list)

        if not match.any():
            loaded += 1
            continue

        idx = indices[match]
        date_mask = match

        arrays['open'][date_mask, j] = open_arr[idx]
        arrays['high'][date_mask, j] = high_arr[idx]
        arrays['low'][date_mask, j] = low_arr[idx]
        arrays['close'][date_mask, j] = close_arr[idx]
        arrays['volume'][date_mask, j] = vol_arr[idx]
        arrays['amount'][date_mask, j] = amt_arr[idx]

        loaded += 1

        now = time.time()
        if now - last_log >= 5 or j == n_stocks - 1:
            elapsed = now - t0
            speed = (j + 1) / elapsed if elapsed > 0 else 0
            eta = (n_stocks - j - 1) / speed if speed > 0 else 0
            print(f"[{time.strftime('%H:%M:%S')}] K线面板: {j+1}/{n_stocks} "
                  f"(耗时 {elapsed:.0f}s, 速度 {speed:.0f}只/s, 预计剩余 {eta:.0f}s)")
            last_log = now

    print(f"K线面板完成: 加载 {loaded} 只, 缺失 {missing} 只")
    return arrays


def build_st_mask(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
) -> np.ndarray:
    """从 st_changes.parquet 构建 ST 掩码。

    Returns:
        bool ndarray (n_dates, n_stocks), True = ST/*ST/退市状态
    """
    n_dates = len(trade_dates)
    n_stocks = len(stock_codes)
    st_path = DATA_DIR / "stock_name" / "st_changes.parquet"

    if not st_path.exists():
        print("警告: st_changes.parquet 不存在，ST 掩码全部为 False")
        return np.zeros((n_dates, n_stocks), dtype=bool)

    df_st = pd.read_parquet(st_path)
    if df_st.empty:
        print("警告: st_changes.parquet 为空，ST 掩码全部为 False")
        return np.zeros((n_dates, n_stocks), dtype=bool)

    trade_date_arr = trade_dates.astype('datetime64[ns]')
    result = np.zeros((n_dates, n_stocks), dtype=bool)

    _KEEP_ST_KEYWORDS = ("披*", "退市整理", "戴帽", "暂停上市", "终止上市")
    _CLEAR_KEYWORDS = ("摘帽", "恢复上市", "新股上市", "重新上市", "转板上市", "摘*摘帽", "发行失败", "拟上市")

    for j, code in enumerate(stock_codes):
        bare = str(code).split('.')[0]
        sc_records = df_st[df_st['bare_code'] == bare]
        if sc_records.empty:
            continue

        dates = sc_records['date'].values.astype('datetime64[ns]')
        events = sc_records['event'].values

        # _KEEP_ST_KEYWORDS 优先：命中则保持 ST
        is_keep = np.array([any(kw in str(e) for kw in _KEEP_ST_KEYWORDS) for e in events])
        is_clear = np.array([any(kw in str(e) for kw in _CLEAR_KEYWORDS) for e in events])
        # keep 优先级最高, clear 次之, 其余保持 ST
        status_changes = np.where(is_keep, True, ~is_clear)

        indices = np.searchsorted(dates, trade_date_arr, side='right') - 1
        valid = indices >= 0
        if valid.any():
            result[valid, j] = status_changes[indices[valid]]

    print(f"ST掩码: {result.sum()} 个 True / {result.size} ({result.sum()/result.size*100:.1f}%)")
    return result


# 报告期末 -> 生效日的滞后天数（防前视野泄露：报告期末后约 120 天财报才公开披露，
# 年报 12-31 通常次年 4-30 才披露；与 build_deep_fin_runtime 口径一致）。
_FIN_LAG_DAYS = 120


def _ffill_axis0(mat: np.ndarray) -> np.ndarray:
    """沿时间轴(行)向前填充：每个生效值持续到下一期覆盖；首个有效值之前保持 NaN。"""
    n_dates = mat.shape[0]
    mask = ~np.isnan(mat)
    idx = np.where(mask, np.arange(n_dates)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = np.take_along_axis(mat, idx, axis=0)
    out[np.cumsum(mask, axis=0) == 0] = np.nan
    return out


def build_financial_arrays(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
) -> dict[str, np.ndarray]:
    """从 deep_indicators.parquet（同花顺深历史，唯一财务源）构建财务指标数组。

    PIT：报告期末 + _FIN_LAG_DAYS 天才生效（防前视野泄露），生效后向前填充至下一期覆盖。
    单一数据源、无兜底/无合并——有数据用数据，无数据为 NaN。
    """
    n_dates = len(trade_dates)
    n_stocks = len(stock_codes)

    # 输出字段 -> deep_indicators 列名
    colmap = {
        'bps': 'bps',
        'eps': 'eps',
        'roe': 'roe',
        'operating_cf_ps': 'ocfps',
        'profit_yoy': 'profit_yoy',
        'revenue_yoy': 'revenue_yoy',
        'gross_margin': 'gross_margin',
    }
    results = {k: np.full((n_dates, n_stocks), np.nan, dtype=np.float64) for k in colmap}

    deep_path = DATA_DIR / "financial" / "deep_indicators.parquet"
    if not deep_path.exists():
        print("警告: deep_indicators.parquet 不存在，财务指标全部为 NaN")
        return results

    df = pd.read_parquet(deep_path)
    code_to_col = {str(c): i for i, c in enumerate(stock_codes)}
    td = trade_dates.astype('datetime64[D]')

    period_end = pd.to_datetime(df['report_period'].astype(int).astype(str), format='%Y%m%d')
    eff_date = (period_end + pd.Timedelta(days=_FIN_LAG_DAYS)).values.astype('datetime64[D]')
    eff_row = np.searchsorted(td, eff_date, side='left')
    col = df['stock_code'].map(code_to_col).to_numpy()

    valid = np.isfinite(col.astype(np.float64)) & (eff_row < n_dates)
    order = np.argsort(df['report_period'].to_numpy()[valid], kind='stable')  # 升序，后期覆盖前期
    eff_row_v = eff_row[valid][order]
    col_v = col[valid][order].astype(np.intp)

    for out_name, src_col in colmap.items():
        if src_col not in df.columns:
            continue
        vals = pd.to_numeric(df[src_col], errors='coerce').to_numpy()[valid][order]
        mat = results[out_name]
        place = np.isfinite(vals)
        mat[eff_row_v[place], col_v[place]] = vals[place]
        results[out_name] = _ffill_axis0(mat)

    cov = {k: int(np.isfinite(v).any(axis=0).sum()) for k, v in results.items()}
    print(f"财务面板完成（deep_indicators, 报告期+{_FIN_LAG_DAYS}d PIT, 单一源）: 覆盖股票 {cov}")
    return results


def build_total_share(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
) -> np.ndarray:
    """从 balance.parquet 的 cap_stk 构建历史总股本数组 (n_dates, n_stocks)。

    使用 m_anntime（披露日期）对齐交易日期，避免未来信息泄露。
    """
    n_dates = len(trade_dates)
    n_stocks = len(stock_codes)
    result = np.zeros((n_dates, n_stocks), dtype=np.float64)

    balance_path = DATA_DIR / "financial" / "balance.parquet"
    if not balance_path.exists():
        print("警告: balance.parquet 不存在，总股本全部为 0")
        return result

    trade_date_ts = trade_dates.astype('datetime64[ns]')
    df_all = pd.read_parquet(balance_path)

    t0 = time.time()
    last_log = t0

    for j, code in enumerate(stock_codes):
        df_code = df_all[df_all['stock_code'] == code]
        if df_code.empty:
            continue

        anntimes = df_code['m_anntime'].values.astype('datetime64[ns]')
        if len(anntimes) == 0:
            continue

        indices = np.searchsorted(anntimes, trade_date_ts, side='right') - 1
        valid = indices >= 0
        if not valid.any():
            continue

        result[valid, j] = df_code['cap_stk'].values[indices[valid]]

        now = time.time()
        if now - last_log >= 5 or j == n_stocks - 1:
            elapsed = now - t0
            speed = (j + 1) / elapsed if elapsed > 0 else 0
            eta = (n_stocks - j - 1) / speed if speed > 0 else 0
            print(f"[{time.strftime('%H:%M:%S')}] 总股本: {j+1}/{n_stocks} "
                  f"(耗时 {elapsed:.0f}s, 速度 {speed:.0f}只/s, 预计剩余 {eta:.0f}s)")
            last_log = now

    nonzero = (result > 0).sum()
    print(f"总股本: {nonzero}/{result.size} 非零 ({nonzero/result.size*100:.1f}%)")
    return result


def build_stock_names(stock_codes: np.ndarray) -> np.ndarray:
    """从 CNINFO pickle 缓存构建股票名称数组 (n_stocks,)，dtype U16。"""
    from data.db.stock_name import get_stock_name_at_date

    today = date.today()
    names = []
    t0 = time.time()
    for j, code in enumerate(stock_codes):
        try:
            name = get_stock_name_at_date(str(code), today)
        except Exception:
            name = None
        names.append(name or '')
        if (j + 1) % 1000 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] 股票名称: {j+1}/{len(stock_codes)} "
                  f"(耗时 {time.time()-t0:.0f}s)")
    result = np.array(names, dtype='U16')
    nonempty = (result != '').sum()
    print(f"股票名称: {nonempty}/{len(result)} 非空 ({nonempty/len(result)*100:.1f}%)")
    return result


def get_trade_dates_from_kline() -> np.ndarray:
    """从所有 k-line parquet 文件中提取并集交易日列表。

    Returns:
        sorted unique numpy array of datetime64[D] dates
    """
    parquet_files = sorted(KLINE_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"未找到 k-line parquet 文件: {KLINE_DIR}")

    all_dates = set()
    t0 = time.time()

    for i, path in enumerate(parquet_files):
        df = pd.read_parquet(path, columns=['time'])
        if df.empty:
            continue
        # time 是 ms 时间戳 → date
        dates = pd.to_datetime(df['time'], unit='ms').dt.date
        all_dates.update(dates)

        if (i + 1) % 500 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] 交易日收集: {i+1}/{len(parquet_files)}")

    dates_arr = np.array(sorted(all_dates), dtype='datetime64[D]')
    print(f"交易日收集完成: {len(all_dates)} 个, 耗时 {time.time()-t0:.1f}s")
    return dates_arr


def build_issue_price(stock_codes: np.ndarray) -> np.ndarray:
    """从 issue_price.parquet 构建发行价数组 (n_stocks,)。

    未匹配到的股票发行价为 NaN。
    """
    n_stocks = len(stock_codes)
    result = np.full(n_stocks, np.nan, dtype=np.float64)

    ip_path = DATA_DIR / "issue_price" / "issue_price.parquet"
    if not ip_path.exists():
        print("警告: issue_price.parquet 不存在，发行价全部为 NaN")
        return result

    df = pd.read_parquet(ip_path)
    ip_map = {}
    for _, row in df.iterrows():
        ip_map[str(row['stock_code'])] = float(row['issue_price'])

    matched = 0
    for j, code in enumerate(stock_codes):
        bare = code[:6]
        price = ip_map.get(bare)
        if price is not None and price > 0:
            result[j] = price
            matched += 1

    print(f"发行价: {matched}/{n_stocks} 匹配 ({matched/n_stocks*100:.1f}%)")
    return result


def build_runtime(
    start_date: date | None = None,
    end_date: date | None = None,
    max_stocks: int | None = None,
):
    """主构建流程。"""
    overall_t0 = time.time()

    print(f"[{time.strftime('%H:%M:%S')}] ===== 1/7 收集交易日 =====")
    all_trade_dates = get_trade_dates_from_kline()

    # 剔除1970脏数据（最早A股交易在1990-12-19）
    all_trade_dates = all_trade_dates[all_trade_dates >= np.datetime64('1990-12-01')]

    # 按日期范围裁剪
    if start_date:
        start_dt = np.datetime64(start_date)
        all_trade_dates = all_trade_dates[all_trade_dates >= start_dt]
    if end_date:
        end_dt = np.datetime64(end_date)
        all_trade_dates = all_trade_dates[all_trade_dates <= end_dt]

    print(f"交易日范围: {all_trade_dates[0]} ~ {all_trade_dates[-1]}, 共 {len(all_trade_dates)} 天")

    # 确定 stock_codes: allow_buy_stock_code_list ∩ 有K线parquet的股票
    from data.db import allow_buy_stock_code_list
    allowed = set(allow_buy_stock_code_list())
    kline_stocks = sorted([
        f.stem for f in KLINE_DIR.glob("*.parquet") if f.stem in allowed
    ])
    stock_codes = np.array(kline_stocks, dtype='U12')

    if max_stocks:
        stock_codes = stock_codes[:max_stocks]

    print(f"可买池: {len(allowed)} 只, 有K线: {len(kline_stocks)} 只, 使用: {len(stock_codes)} 只")

    print(f"[{time.strftime('%H:%M:%S')}] ===== 2/7 构建K线面板 =====")
    kline_arrays = load_kline_panel(stock_codes, all_trade_dates)

    # issue_price: 每股发行价，用于 IPO 首日涨跌停基准
    issue_price = build_issue_price(stock_codes)

    print(f"[{time.strftime('%H:%M:%S')}] ===== 3/7 构建ST掩码 =====")
    st_mask = build_st_mask(stock_codes, all_trade_dates)

    print(f"[{time.strftime('%H:%M:%S')}] ===== 4/7 构建总股本 =====")
    total_share = build_total_share(stock_codes, all_trade_dates)

    print(f"[{time.strftime('%H:%M:%S')}] ===== 5/7 构建财务面板 =====")
    fin_arrays = build_financial_arrays(stock_codes, all_trade_dates)

    print(f"[{time.strftime('%H:%M:%S')}] ===== 6/7 构建股票名称 =====")
    stock_names = build_stock_names(stock_codes)

    print(f"[{time.strftime('%H:%M:%S')}] ===== 7/7 保存 npz =====")
    output_name = f"runtime_{all_trade_dates[0]}_{all_trade_dates[-1]}.npz"
    output_path = OUT_DIR / output_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        stock_codes=stock_codes,
        trade_dates=all_trade_dates,
        open=kline_arrays['open'],
        high=kline_arrays['high'],
        low=kline_arrays['low'],
        close=kline_arrays['close'],
        volume=kline_arrays['volume'],
        amount=kline_arrays['amount'],
        issue_price=issue_price,
        stock_names=stock_names,
        st_mask=st_mask,
        total_share=total_share,
        bps=fin_arrays['bps'],
        eps=fin_arrays['eps'],
        roe=fin_arrays['roe'],
        profit_yoy=fin_arrays['profit_yoy'],
        revenue_yoy=fin_arrays['revenue_yoy'],
        operating_cf_ps=fin_arrays['operating_cf_ps'],
        gross_margin=fin_arrays['gross_margin'],
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - overall_t0
    print(f"构建完成: {output_path} ({file_size_mb:.1f} MB, 耗时 {elapsed:.0f}s)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="构建 runtime npz 文件")
    parser.add_argument("--start", type=str, default=None, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--max-stocks", type=int, default=None, help="调试模式：只处理前N只股票")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start) if args.start else None
    end_date = date.fromisoformat(args.end) if args.end else None

    build_runtime(start_date=start_date, end_date=end_date, max_stocks=args.max_stocks)


if __name__ == "__main__":
    main()
