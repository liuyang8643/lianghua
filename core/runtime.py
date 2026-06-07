"""Runtime NPZ 数据加载 — 无因子依赖，回测和实盘共用。"""
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np

from .logger import core_logger

_RUNTIME_DIR = Path(__file__).resolve().parents[1] / "data" / "runtime"

_2D_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'amount',
              'preClose',
              'total_share', 'eps', 'roe', 'profit_yoy', 'revenue_yoy',
              'operating_cf_ps', 'gross_margin', 'st_mask', 'bps']
_1D_FIELDS = ['issue_price', 'stock_names']


def load_runtime_npz(dates: List[datetime], max_lookback: Optional[int] = None) -> dict | None:
    """加载 runtime NPZ 数据。

    Args:
        dates: 信号日期列表（回测=全部调仓日, 实盘=当日）
        max_lookback: 可选，裁剪数据只保留 min(dates)-max_lookback 到 max(dates)+5 个交易日。
                      用于实盘/单回测减少内存；GA 不传此参数加载全量。
    """
    if not _RUNTIME_DIR.exists():
        return None

    min_date = np.datetime64(min(dt.date() for dt in dates))
    max_date = np.datetime64(max(dt.date() for dt in dates)) + np.timedelta64(7, 'D')

    trim_start = None
    if max_lookback is not None and max_lookback > 0:
        trim_start = min_date - np.timedelta64(int(max_lookback * 1.5) + 10, 'D')

    npz_files = sorted(_RUNTIME_DIR.glob("runtime_*.npz"))
    parts = []
    for npz_path in npz_files:
        data = dict(np.load(npz_path, allow_pickle=False))
        d0, d1 = data['trade_dates'][0], data['trade_dates'][-1]
        if d0 <= max_date and d1 >= min_date:
            if trim_start is not None:
                td = data['trade_dates']
                si = max(0, int(np.searchsorted(td, trim_start)))
                ei = min(len(td), int(np.searchsorted(td, max_date)) + 5)
                data['trade_dates'] = td[si:ei]
                for f in _2D_FIELDS:
                    if f in data:
                        data[f] = data[f][si:ei]
            parts.append(data)
            core_logger.info(f"  {npz_path.name}: {len(data['trade_dates'])}d x {len(data['stock_codes'])}s")

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    core_logger.info(f"合并 {len(parts)} 个 npz 文件...")

    first_codes = parts[0]['stock_codes']
    codes_match = all(np.array_equal(p['stock_codes'], first_codes) for p in parts[1:])

    all_dates = np.concatenate([p['trade_dates'] for p in parts])
    all_dates = np.unique(all_dates)
    all_dates.sort()

    if codes_match:
        n_stocks = len(first_codes)
        merged = {'stock_codes': first_codes, 'trade_dates': all_dates}
        offsets_list = [np.searchsorted(all_dates, p['trade_dates']) for p in parts]
        for field in _2D_FIELDS:
            if field not in parts[0]:
                continue
            dtype = np.bool_ if field == 'st_mask' else np.float64
            fill = False if field == 'st_mask' else np.nan
            arr = np.full((len(all_dates), n_stocks), fill, dtype=dtype)
            for pi, p in enumerate(parts):
                arr[offsets_list[pi]] = p[field]
            merged[field] = arr
        for field in _1D_FIELDS:
            if field in parts[0]:
                merged[field] = parts[-1][field]
    else:
        all_stocks = []
        seen = set()
        for p in parts:
            for s in p['stock_codes']:
                s_str = str(s)
                if s_str not in seen:
                    seen.add(s_str)
                    all_stocks.append(s_str)
        n_stocks = len(all_stocks)
        stock_to_idx = {s: i for i, s in enumerate(all_stocks)}
        merged = {
            'stock_codes': np.array(all_stocks, dtype='U12'),
            'trade_dates': all_dates,
        }
        offsets_list = [np.searchsorted(all_dates, p['trade_dates']) for p in parts]
        for field in _2D_FIELDS:
            if field not in parts[0]:
                continue
            dtype = np.bool_ if field == 'st_mask' else np.float64
            fill = False if field == 'st_mask' else np.nan
            arr = np.full((len(all_dates), n_stocks), fill, dtype=dtype)
            for pi, p in enumerate(parts):
                p_stocks = [str(s) for s in p['stock_codes']]
                col_idx = np.array([stock_to_idx.get(s, -1) for s in p_stocks])
                valid = col_idx >= 0
                if not valid.any():
                    continue
                for di in range(len(offsets_list[pi])):
                    arr[offsets_list[pi][di], col_idx[valid]] = p[field][di, valid]
            merged[field] = arr
        if 'issue_price' in parts[0]:
            arr = np.full(n_stocks, np.nan, dtype=np.float64)
            for pi, p in enumerate(parts):
                p_stocks = [str(s) for s in p['stock_codes']]
                for j, s in enumerate(p_stocks):
                    t = stock_to_idx.get(s, -1)
                    if t >= 0 and np.isnan(arr[t]) and not np.isnan(p['issue_price'][j]):
                        arr[t] = p['issue_price'][j]
            merged['issue_price'] = arr
        if 'stock_names' in parts[0]:
            arr = np.empty(n_stocks, dtype='U16')
            for pi, p in enumerate(parts):
                p_stocks = [str(s) for s in p['stock_codes']]
                for j, s in enumerate(p_stocks):
                    t = stock_to_idx.get(s, -1)
                    if t >= 0 and p['stock_names'][j]:
                        arr[t] = p['stock_names'][j]
            merged['stock_names'] = arr

    core_logger.info(f"合并完成: {len(all_dates)}d x {len(merged['stock_codes'])}s")
    return merged
