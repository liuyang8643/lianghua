"""大盘择时调仓 — 指数均线偏离度仓位调节"""
import numpy as np
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

INDEX_INFO = {
    'sh000300': '沪深300',
    'sh000905': '中证500',
    'sh000852': '中证1000',
}


def _index_parquet_path(symbol):
    return DATA_DIR / f"index_{symbol}_daily.parquet"


def _download_index(symbol):
    """从 akshare 下载指数日线并缓存为 parquet"""
    import akshare as ak
    df = ak.stock_zh_index_daily(symbol=symbol)
    df = df.rename(columns={'date': 'trade_date'})
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df[['trade_date', 'open']].sort_values('trade_date')
    path = _index_parquet_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_index_open(symbol, trade_dates=None):
    """加载指数开盘价，首次自动下载缓存。返回 (dates_array, open_array)。"""
    path = _index_parquet_path(symbol)
    if path.exists():
        df = pd.read_parquet(path)
        last_date = pd.Timestamp(df['trade_date'].max())
        if (pd.Timestamp.now() - last_date).days > 7:
            df = _download_index(symbol)
    else:
        df = _download_index(symbol)

    dates = pd.to_datetime(df['trade_date'].values)
    open_arr = df['open'].values.astype(np.float64)

    if trade_dates is not None:
        date_to_val = dict(zip(dates, open_arr))
        aligned = []
        for d in trade_dates:
            dt = pd.Timestamp(d) if hasattr(d, 'strftime') else pd.Timestamp(d)
            aligned.append(date_to_val.get(dt, np.nan))
        aligned = pd.Series(aligned).ffill().values
        return np.array(trade_dates), aligned

    return dates, open_arr


def compute_position_multiplier(index_open, ma_period=60, sensitivity=1.0, direction=1, floor=0.0, cap=1.0):
    """仓位乘数: multiplier = 0.5 + direction * Zscore * sensitivity * 0.25

    direction=+1 顺势(涨→加仓), direction=-1 逆势(涨→减仓)
    sensitivity=1: ±2σ→0%/100%
    """
    arr = np.asarray(index_open, dtype=np.float64)
    n = len(arr)

    ma = np.full(n, np.nan)
    std = np.full(n, np.nan)
    with np.errstate(all='ignore'):
        for i in range(ma_period - 1, n):
            window = arr[i - ma_period + 1:i + 1]
            ma[i] = np.nanmean(window)
            std[i] = np.nanstd(window)

    deviation = np.full(n, 0.0)
    valid = ~np.isnan(ma) & (ma > 0) & ~np.isnan(std) & (std > 0)
    deviation[valid] = (arr[valid] - ma[valid]) / std[valid]

    multiplier = 0.5 + direction * deviation * sensitivity * 0.25
    multiplier = np.clip(multiplier, floor, cap)
    return multiplier
