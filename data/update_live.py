"""09:25 / 15:00 快速 K 线更新 — 只写 parquet；NPZ 仅内存覆盖（不落盘）。

- 09:25 / 15:00：update_live_quick(patch_npz=False) → parquet + 可选内存 overlay
- 16:00 update_all：重拉 3 日 K 线 + build_runtime 全量写 NPZ（权威落盘）

锚定日：
- 未传 anchor：用 date.today() 在交易日历中的最近交易日（0605 凌晨常是 0604）
- --skip 202606040925：anchor=2026-06-04 → 拉 0604 的 K 线
"""
import time
import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("update_live")

DATA_DIR = Path(__file__).resolve().parent
DOWNLOAD_TRADING_DAYS = 1

_RAW_PATCH_FIELDS = ('open', 'high', 'low', 'close', 'volume', 'amount', 'preClose')
_KLINE_NANFILL_FIELDS = _RAW_PATCH_FIELDS


def _info(msg: str, *args):
    logger.info(msg, *args)
    try:
        from trading.logger import trading_logger
        trading_logger.info(msg % args if args else msg)
    except Exception as e:
        logger.debug(f"trading_logger 转发失败 (非实盘环境可忽略): {e}")


def _download_kline_all(download_trading_days: int = DOWNLOAD_TRADING_DAYS,
                        anchor_date: date | None = None) -> dict:
    from data.kline_mootdx import update_recent
    return update_recent(download_trading_days, anchor_date=anchor_date, collect=True)


def apply_kline_overlay(data: dict, kline_data: dict) -> tuple[dict, int, list]:
    """在已加载的 runtime dict 上覆盖 K 线（内存）— 不写 NPZ 文件。"""
    if not kline_data:
        return data, 0, []

    stock_to_idx = {str(c): i for i, c in enumerate(data['stock_codes'])}
    old_dates = data['trade_dates']
    n_old = len(old_dates)
    old_date_set = {d.astype('datetime64[D]').item() for d in old_dates}

    all_kline_dates = set()
    for bars in kline_data.values():
        for ts in bars['time']:
            dt = pd.Timestamp(int(ts), unit='ms').to_datetime64().astype('datetime64[D]').item()
            all_kline_dates.add(dt)

    new_dates = sorted(all_kline_dates - old_date_set)
    if new_dates:
        n_new = len(new_dates)
        new_dt = np.array(new_dates, dtype='datetime64[D]')
        data['trade_dates'] = np.concatenate([old_dates, new_dt])
        for key in list(data.keys()):
            arr = data[key]
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                continue
            if arr.shape[0] != n_old:
                continue
            is_kline = key in _KLINE_NANFILL_FIELDS
            filler = np.full((1, arr.shape[1]), np.nan, dtype=arr.dtype) if is_kline else arr[-1:].copy()
            new_rows = np.tile(filler, (n_new, 1))
            data[key] = np.concatenate([arr, new_rows], axis=0)

    trade_dates = data['trade_dates']
    date_to_idx = {
        trade_dates[i].astype('datetime64[D]').item(): i for i in range(len(trade_dates))
    }

    updated_cells = 0
    patched_rows: dict[int, list[int]] = {}
    for code, bars in kline_data.items():
        si = stock_to_idx.get(code)
        if si is None:
            continue
        rows = []
        for k in range(len(bars['time'])):
            dt = pd.Timestamp(int(bars['time'][k]), unit='ms').to_datetime64().astype('datetime64[D]').item()
            di = date_to_idx.get(dt)
            if di is None:
                continue
            for f in _RAW_PATCH_FIELDS:
                data[f][di, si] = bars[f][k]
            rows.append(di)
            updated_cells += 1
        if rows:
            patched_rows[si] = sorted(set(rows))

    return data, updated_cells, new_dates


def _patch_npz_incremental(kline_data: dict):
    """增量修补 NPZ 并落盘（仅 16:00 全量 build 或显式 patch_npz=True 时使用）。"""
    OUT_DIR = DATA_DIR / "runtime"
    npz_files = sorted(OUT_DIR.glob("runtime_*.npz"))
    if not npz_files:
        _info("[NPZ增量] 无现有 NPZ，执行全量构建")
        from data.build_runtime import build_runtime
        return build_runtime()

    t0 = time.time()
    data = dict(np.load(npz_files[-1], allow_pickle=False))
    data, updated_cells, new_dates = apply_kline_overlay(data, kline_data)
    _info("[NPZ增量] 覆盖 %d 个 (date,stock) 单元格 + 连乘复权 (%.0fs)",
          updated_cells, time.time() - t0)

    if updated_cells == 0 and not new_dates:
        _info("[NPZ增量] 无变更，跳过 1GB 级重写 → %s", npz_files[-1].name)
        return npz_files[-1]

    for f in npz_files:
        f.unlink()
    td = data['trade_dates']
    output_path = OUT_DIR / f"runtime_{str(td[0])}_{str(td[-1])}.npz"
    np.savez_compressed(output_path, **data)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    _info("[NPZ增量] 保存: %s (%.1f MB, %.0fs)", output_path.name, file_size_mb, time.time() - t0)
    return output_path


def update_live_quick(download_trading_days: int = DOWNLOAD_TRADING_DAYS, *,
                      patch_npz: bool = False,
                      anchor_date: date | None = None) -> dict:
    """快速 K 线更新。默认只写 parquet；patch_npz=True 时才落盘 NPZ。"""
    from data.kline_mootdx import resolve_recent_range

    t0 = time.time()
    _, _, end_d = resolve_recent_range(download_trading_days, anchor_date)
    anchor_note = f"锚定={end_d.isoformat()}" + (
        f" (来自 --skip {anchor_date})" if anchor_date else " (日历最近交易日)")
    _info("=" * 60)
    _info("快速K线: 最近 %d 日 parquet%s | %s",
          download_trading_days, " + NPZ落盘" if patch_npz else " only", anchor_note)
    _info("=" * 60)

    _info("--- Phase 1: K线下载 → parquet ---")
    kline_data = _download_kline_all(download_trading_days, anchor_date=anchor_date)

    if patch_npz:
        _info("--- Phase 2: NPZ 落盘 ---")
        _patch_npz_incremental(kline_data)
    else:
        _info("--- Phase 2: 跳过 NPZ 落盘（16:00 update_all 再全量写）---")

    _info("=" * 60)
    _info("快速K线完成! 耗时 %.0fs", time.time() - t0)
    _info("=" * 60)
    return kline_data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    update_live_quick()
