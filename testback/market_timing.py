"""大盘择时调仓 — 窗口涨跌比仓位调节（网格思路）"""
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

INDEX_INFO = {
    'sh000300': '沪深300',
    'sh000905': '中证500',
    'sh000852': '中证1000',
}


def _index_parquet_path(symbol):
    return DATA_DIR / f"index_{symbol}_daily.parquet"


def load_index_open(symbol, trade_dates=None):
    """加载指数开盘价（从预下载 parquet，不联网）。返回 (dates_array, open_array)。"""
    import pyarrow.parquet as pq
    path = _index_parquet_path(symbol)

    if not path.exists():
        raise FileNotFoundError(f"指数数据缺失: {path}，请先运行 data/update_all.py 预下载")

    table = pq.read_table(path)
    dates_arr = table.column('trade_date').to_numpy().astype('datetime64[D]')
    open_arr = table.column('open').to_numpy().astype(np.float64)

    if trade_dates is not None:
        date_to_val = {}
        for i in range(len(dates_arr)):
            date_to_val[dates_arr[i].item()] = open_arr[i]

        aligned = np.array([
            date_to_val.get(d.date() if hasattr(d, 'date') else d, np.nan)
            for d in trade_dates
        ], dtype=np.float64)

        mask = np.isnan(aligned)
        if mask.any():
            idx = np.where(~mask, np.arange(len(aligned)), 0)
            np.maximum.accumulate(idx, out=idx)
            aligned = aligned[idx]

        return np.array(trade_dates), aligned

    return dates_arr, open_arr


def compute_position_multiplier(index_open, window=20, base=0.5, leverage=10, direction=1, floor=0.0, cap=1.0):
    """仓位乘数: 窗口收益率 × 杠杆 + 基准仓位

    multiplier = base + direction * ret * leverage, clip到[floor, cap]
    ret = (今日开盘 - window日前开盘) / window日前开盘

    direction=-1 逆势(涨→减仓), direction=+1 顺势(涨→加仓)
    例: base=0.5, leverage=10, direction=-1 → 大盘跌5% → 0.5+0.5=满仓, 涨5%→0.5-0.5=空仓
    """
    arr = np.asarray(index_open, dtype=np.float64)
    n = len(arr)
    ret = np.full(n, np.nan)

    for i in range(window, n):
        ret[i] = (arr[i] - arr[i - window]) / arr[i - window]

    multiplier = np.full(n, base)
    valid = ~np.isnan(ret)
    multiplier[valid] = base + direction * ret[valid] * leverage
    multiplier = np.clip(multiplier, floor, cap)
    return multiplier
