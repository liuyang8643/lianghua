"""mootdx 日线下载 —— WBR 唯一 K 线数据源，产出一套不复权 parquet：

    data/k-line/{code}.parquet  列: time/open/high/low/close/volume/amount/preClose

mootdx 只取 fq=0 的不复权真实价 OHLCV；preClose 由 mootdx xdxr() 除权除息数据
自行计算（交易所官方公式），99.4% 除权日与 QMT 官方 preClose 对齐（200 只×250 日验证）。
转配股/缩股等非标准事件日期（约 0.6%）preClose=NaN，legality 模块自动跳过不交易。

复权序列由 build_runtime 用 `r = close/preClose - 1` 连乘自建（等比后复权，数学上恒正）。

用法:
    python data/kline_mootdx.py                      # 全量拉取到今天
    python data/kline_mootdx.py --codes 000001.SZ    # 指定代码
    python data/kline_mootdx.py --recent 3           # 只刷新最近 N 个交易日
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "k-line"

# 除权除息类别：1=除权除息 5=股本变化(不调价)。其他(9转配股/15等)标记为不可计算
XD_CAT_STANDARD = 1       # 除权除息 — 可用公式

START_DEFAULT = '19901219'

PAGE_SIZE = 800            # 单次 API 最大返回条数
MAX_HISTORY_BARS = 10000   # 全量拉取上限（覆盖 1990-至今约 8500 交易日）
RECENT_BARS = 400          # 增量拉取条数上限


def _log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _connect_mootdx():
    from mootdx.quotes import Quotes
    return Quotes.factory(market='std')


def _ms_to_date_ymd(ts_ms):
    """毫秒时间戳 → YYYYMMDD int64 array。"""
    dt = pd.to_datetime(ts_ms, unit='ms')
    if isinstance(dt, pd.Series):
        return (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day).to_numpy(np.int64)
    return (dt.year * 10000 + dt.month * 100 + dt.day).to_numpy(np.int64)


def _datetime_in_index(bars: pd.DataFrame) -> bool:
    """检查 datetime 是否为 index 中任一层级名称（含 MultiIndex）。"""
    if bars.index.name == 'datetime':
        return True
    if isinstance(bars.index, pd.MultiIndex):
        return 'datetime' in (n for n in bars.index.names if n is not None)
    return False


def _bars_to_date_ymd(bars: pd.DataFrame) -> np.ndarray:
    """mootdx bars → YYYYMMDD int64 array (时间倒序)。"""
    if 'datetime' in bars.columns:
        if _datetime_in_index(bars):
            bars = bars.reset_index(drop=True)
        dt = pd.to_datetime(bars['datetime'])
        return (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day).to_numpy(np.int64)
    return _ms_to_date_ymd(bars.index.astype('int64') // 10 ** 6)


def _mootdx_bars_to_df(bars: pd.DataFrame, *, xdxr_df: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """mootdx 不复权日线 → 标准 raw DataFrame (时间倒序)。

    preClose 由 xdxr 除权数据计算；无法计算的日期（非标事件）preClose=NaN。
    """
    if bars is None or bars.empty:
        return None

    # mootdx 可能返回 datetime 既是列名又是 index 层名（含 MultiIndex），导致后续 bars['datetime'] 歧义
    if 'datetime' in bars.columns and _datetime_in_index(bars):
        bars = bars.reset_index(drop=True)

    # 先按时间升序排列，保证 _compute_preclose 按时间顺序计算
    if 'datetime' in bars.columns:
        bars = bars.sort_values('datetime')
        times = pd.to_datetime(bars['datetime'])
    else:
        times = bars.index
    time_ms = (times.astype('int64') // 10 ** 6).to_numpy()

    keep = (bars['open'].to_numpy(float) > 0) | (bars['close'].to_numpy(float) > 0)
    if not keep.any():
        return None

    cl = bars['close'].to_numpy(float)[keep]
    op = bars['open'].to_numpy(float)[keep]
    hi = bars['high'].to_numpy(float)[keep]
    lo = bars['low'].to_numpy(float)[keep]
    vo = bars['volume'].to_numpy(float)[keep]
    am = bars['amount'].to_numpy(float)[keep]
    tm = time_ms[keep]

    preclose = _compute_preclose(cl, tm, xdxr_df)

    # 时间升序存 parquet，下游 build_runtime 依赖时间升序
    raw = pd.DataFrame({
        'time': tm, 'open': op, 'high': hi, 'low': lo,
        'close': cl, 'volume': vo, 'amount': am, 'preClose': preclose,
    }).reset_index(drop=True)

    return raw


def _compute_preclose(closes: np.ndarray, time_ms: np.ndarray,
                      xdxr_df: pd.DataFrame | None) -> np.ndarray:
    """从 close + xdxr 计算官方 preClose。

    - 非除权日: preClose[t] = close[t-1]
    - 除权日(cat=1): 交易所公式

    closes / time_ms: 时间升序
    """
    n = len(closes)
    preclose = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return preclose
    preclose[0] = closes[0]

    # 构建 xdxr 除权日 → record 映射: YYYYMMDD int → row
    xd_map = {}
    if xdxr_df is not None and not xdxr_df.empty:
        xd = xdxr_df.copy()
        xd['ymd'] = (xd['year'].astype(int) * 10000
                     + xd['month'].astype(int) * 100
                     + xd['day'].astype(int))
        for _, row in xd.iterrows():
            d = int(row['ymd'])
            if int(row['category']) == XD_CAT_STANDARD:
                xd_map[d] = row

    dates_ymd = _ms_to_date_ymd(time_ms)

    for i in range(1, n):
        d = int(dates_ymd[i])
        if d in xd_map:
            row = xd_map[d]
            fh = float(row['fenhong']) if pd.notna(row['fenhong']) else 0.0
            sg = float(row['songzhuangu']) if pd.notna(row['songzhuangu']) else 0.0
            pg = float(row['peigu']) if pd.notna(row['peigu']) else 0.0
            pgj = float(row['peigujia']) if pd.notna(row['peigujia']) else 0.0

            div_per_share = fh / 10.0
            bonus_rate = sg / 10.0
            rights_rate = pg / 10.0

            numerator = closes[i - 1] - div_per_share + pgj * rights_rate
            denominator = 1.0 + bonus_rate + rights_rate
            preclose[i] = numerator / denominator
        else:
            preclose[i] = closes[i - 1]

    return preclose


def _fetch_bars_all(mdx, code: str) -> pd.DataFrame | None:
    """分页拉取全量历史日线（1990-至今）。mootdx 单次最多返回 800 条。"""
    all_parts = []
    for start_pos in range(0, MAX_HISTORY_BARS, PAGE_SIZE):
        bars = mdx.bars(symbol=code[:6], frequency=9, start=start_pos, offset=PAGE_SIZE, fq=0)
        if bars is None or bars.empty:
            break
        all_parts.append(bars)
        if len(bars) < PAGE_SIZE:
            break
    if not all_parts:
        return None
    df = pd.concat(all_parts, ignore_index=True)
    if df.index.name == 'datetime':
        df = df.reset_index(drop=True)
    return df.drop_duplicates(subset=['datetime'], keep='first')


def _fetch_bars_recent(mdx, code: str) -> pd.DataFrame | None:
    """拉取最近 ~1.5 年日线（增量用，单次调用）。"""
    return mdx.bars(symbol=code[:6], frequency=9, start=0, offset=RECENT_BARS, fq=0)


def fetch_one(mdx, code: str, start: str | None = None, end: str | None = None) -> pd.DataFrame | None:
    """取单只股票全量 K 线 + preClose（调试用）。"""
    xd = mdx.xdxr(symbol=code[:6])
    bars = _fetch_bars_all(mdx, code)
    if bars is None:
        return None
    return _mootdx_bars_to_df(bars, xdxr_df=xd)


def download(mdx, codes: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """全量下载（分页拉取全历史）并写 parquet。返回 {code: combined_bar_dict}。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total = len(codes)
    t0 = time.time()
    written = 0
    empty = 0
    out: dict = {}

    # 预拉取 xdxr 数据
    xdxr_cache: dict[str, pd.DataFrame | None] = {}
    for code in codes:
        try:
            xdxr_cache[code] = mdx.xdxr(symbol=code[:6])
        except Exception:
            xdxr_cache[code] = None

    for i, code in enumerate(codes):
        bars = _fetch_bars_all(mdx, code)
        raw = _mootdx_bars_to_df(bars, xdxr_df=xdxr_cache.get(code)) if bars is not None else None
        if raw is None:
            empty += 1
        else:
            raw.to_parquet(RAW_DIR / f'{code}.parquet', index=False)
            out[code] = _combined_bar_dict(raw)
            written += 1

        if (i + 1) % 500 == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            _log(f"进度 {i+1}/{total} 写入 {written} 无数据 {empty} | {elapsed:.0f}s ETA {eta:.0f}s")

    _log(f"完成: 写入 {written} 无数据 {empty} 用时 {time.time() - t0:.0f}s")
    return out


def _combined_bar_dict(raw: pd.DataFrame) -> dict:
    return {
        'time': raw['time'].to_numpy(np.int64),
        'open': raw['open'].to_numpy(float),
        'high': raw['high'].to_numpy(float),
        'low': raw['low'].to_numpy(float),
        'close': raw['close'].to_numpy(float),
        'volume': raw['volume'].to_numpy(float),
        'amount': raw['amount'].to_numpy(float),
        'preClose': raw['preClose'].to_numpy(float),
    }


def _all_codes() -> list[str]:
    from data.db.stock_list import get_all_stock_code_list
    return sorted(get_all_stock_code_list())


def resolve_recent_range(days: int, anchor_date: date | None = None) -> tuple[str, str, date]:
    """解析最近 N 个交易日的 [start, end]（YYYYMMDD）及 end 交易日。"""
    from utils.stock.time import get_last_trading_day
    base = anchor_date or date.today()
    end_d = get_last_trading_day(base)
    start_d = end_d
    for _ in range(max(1, days) - 1):
        start_d = get_last_trading_day(start_d - pd.Timedelta(days=1).to_pytimedelta())
    return start_d.strftime('%Y%m%d'), end_d.strftime('%Y%m%d'), end_d


def _merge_recent_into(path: Path, df_new: pd.DataFrame) -> pd.DataFrame:
    """用 df_new 覆盖 path 中 time 落在 df_new 区间内的旧行。"""
    if path.exists() and path.stat().st_size > 0:
        df_old = pd.read_parquet(path)
        df_old = df_old[df_old['time'] < int(df_new['time'].min())]
        df = pd.concat([df_new, df_old], ignore_index=True)
    else:
        df = df_new
    df = df.sort_values('time').reset_index(drop=True)
    df.to_parquet(path, index=False)
    return df


def update_full(start: str = START_DEFAULT, end: str | None = None,
                codes: list[str] | None = None):
    """全量拉取到 end（默认今天）。"""
    mdx = _connect_mootdx()
    end = end or datetime.now().strftime('%Y%m%d')
    codes = codes or _all_codes()
    _log(f"全量 {len(codes)} 只 → {RAW_DIR} (start={start} end={end})")
    download(mdx, codes, start, end)


def update_recent(days: int, *, anchor_date: date | None = None, collect: bool = False) -> dict:
    """刷新最近 days 个交易日并合并进 parquet；新股全量补齐。"""
    mdx = _connect_mootdx()
    start, end, end_d = resolve_recent_range(days, anchor_date)

    all_codes = _all_codes()
    existing = {f.stem for f in RAW_DIR.glob('*.parquet')}
    new_codes = [c for c in all_codes if c not in existing]
    upd_codes = [c for c in all_codes if c in existing]

    _log(f"增量刷新最近 {days} 日 [{start}~{end}] 锚定={end_d.isoformat()}: "
         f"更新 {len(upd_codes)} 只, 新股全量 {len(new_codes)} 只")

    out = {}
    if new_codes:
        _log(f"  处理新股 {len(new_codes)} 只...")
        out.update(download(mdx, new_codes, START_DEFAULT, end))

    # 增量：只拉最近 bars 覆盖 + 合并
    if upd_codes:
        _log(f"  增量合并最近 {len(upd_codes)} 只...")
        t0 = time.time()
        written = empty = 0

        for i, code in enumerate(upd_codes):
            try:
                bars = mdx.bars(symbol=code[:6], frequency=9, start=0, offset=min(days * 5, RECENT_BARS), fq=0)
            except Exception:
                empty += 1
                continue
            raw = _mootdx_bars_to_df(bars, xdxr_df=None)
            if raw is None:
                empty += 1
                continue
            df_new = raw[raw['time'] >= int(pd.Timestamp(start).timestamp() * 1000)]
            if df_new.empty:
                written += 1
                continue
            _merge_recent_into(RAW_DIR / f'{code}.parquet', df_new)
            if collect:
                out[code] = _combined_bar_dict(raw)
            written += 1
            if (i + 1) % 2000 == 0:
                _log(f"  增量 {i+1}/{len(upd_codes)} 写入 {written} | {time.time()-t0:.0f}s")
        _log(f"  增量完成: 写入 {written} 无数据 {empty} 用时 {time.time()-t0:.0f}s")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=START_DEFAULT)
    parser.add_argument('--end', default=None)
    parser.add_argument('--codes', nargs='*', default=None, help='只拉指定代码')
    parser.add_argument('--recent', type=int, default=None, help='只刷新最近 N 个交易日')
    args = parser.parse_args()

    if args.recent:
        update_recent(args.recent)
    else:
        update_full(start=args.start, end=args.end, codes=args.codes)


if __name__ == '__main__':
    main()
