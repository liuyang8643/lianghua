"""下载全市场1分钟K线数据，仅保留09:31-09:35的5根bar，用于开盘滑点分析。

数据源: mootdx (腾讯通达信)，频率=8 (1分钟)，约覆盖最近4个月（~91交易日）。
产物: data/minute/{code}.parquet
列: time, open, close, high, low, volume, amount

09:31 bar 的 open = 当日开盘价(09:30集合竞价结果)。
用法: uv run python -m data.download_minute
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "minute"

TARGET_MINUTES = {f"09:{m:02d}" for m in range(31, 36)}  # 09:31-09:35

PAGE_SIZE = 800
MAX_BARS = 22000   # ~91交易日×240分钟，留余量


def _log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _connect():
    from mootdx.quotes import Quotes
    return Quotes.factory(market='std')


def _all_codes() -> list[str]:
    from data.db.stock_list import get_all_stock_code_list
    return sorted(get_all_stock_code_list())


def _parse_minute(dt_str: str) -> str | None:
    try:
        return dt_str[11:16]
    except (IndexError, TypeError):
        return None


def fetch_minutes(client, code: str) -> pd.DataFrame | None:
    """分页拉取全量1分钟K线，仅保留09:31-09:35。"""
    parts = []
    for start in range(0, MAX_BARS, PAGE_SIZE):
        bars = client.bars(symbol=code[:6], frequency=8, start=start, offset=PAGE_SIZE, fq=0)
        if bars is None or bars.empty:
            break
        raw_len = len(bars)
        if bars.index.name == 'datetime':
            bars = bars.reset_index(drop=True)
        if 'datetime' in bars.columns:
            bars['_minute'] = bars['datetime'].astype(str).str[11:16]
            bars = bars[bars['_minute'].isin(TARGET_MINUTES)]
            if not bars.empty:
                parts.append(bars[['datetime', 'open', 'close', 'high', 'low', 'volume' if 'volume' in bars.columns else 'vol', 'amount']].copy())
        if raw_len < PAGE_SIZE:
            break

    if not parts:
        return None

    df = pd.concat(parts, ignore_index=True)
    df.columns = ['time', 'open', 'close', 'high', 'low', 'volume', 'amount']
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    return df


def download_all():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = _connect()
    codes = _all_codes()
    total = len(codes)
    t0 = time.time()
    written = 0
    empty = 0
    errors = 0

    for i, code in enumerate(codes):
        try:
            df = fetch_minutes(client, code)
            if df is not None and not df.empty:
                df.to_parquet(DATA_DIR / f'{code}.parquet', index=False)
                written += 1
            else:
                empty += 1
        except Exception:
            errors += 1

        if (i + 1) % 200 == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            _log(f"进度 {i+1}/{total} 写入 {written} 空 {empty} 错 {errors} | {elapsed:.0f}s ETA {eta:.0f}s")

    _log(f"完成: 写入 {written} 空 {empty} 错 {errors} 用时 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    download_all()
