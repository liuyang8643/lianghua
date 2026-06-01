"""09:25 快速实盘数据更新 — K线增量下载 → 增量修补 NPZ。

开盘只拉 K 线 + 直接覆盖 NPZ 对应日期的 K 线数组，不重建整个 NPZ。
其余数据（ST/名称/财务等）盘后全量更新——缺一天不影响选股。

每次开盘统一重拉「最新 N 个交易日」（DOWNLOAD_TRADING_DAYS，默认 3）的 K 线，
并覆盖（删除旧值后写入）本地 parquet 与 NPZ 中这 N 天的对应行：
开盘拉到的当日 bar 是盘中快照（high/low/close 不完整），若当天 16:00 的
update_all 又没跑成，旧的不完整数据会一直留在本地。重拉最近 N 天可保证
之后任意一个开盘窗口把前几天的不完整 OHLC 用收盘后的完整值覆盖回来。

用法:
  uv run python data/update_live.py
  uv run python -c "from data.update_live import update_live_quick; update_live_quick()"
"""
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 注意：不要在模块层面调用 logging.basicConfig — 它会抢占 Python stdlib 的 root logger，
# 导致 lark_oapi / xtquant 等三方库的日志被重复输出。
# 独立运行时（__main__）才配置 root logger；被 trading.main 等入口 import 时静默使用本地 logger。
logger = logging.getLogger("update_live")

DATA_DIR = Path(__file__).resolve().parent
TODAY = date.today()
# 每次开盘重拉并覆盖「最新 N 个交易日」的 K 线（parquet + NPZ）。
# 取 3 而非 1，是为了用收盘后的完整 OHLC 覆盖掉之前开盘抓到的盘中快照。
DOWNLOAD_TRADING_DAYS = 3


def _merge_overwrite_by_time(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """用 df_new 覆盖 df_old 中 time 落在 df_new 时间区间内的行，按 time 降序返回。

    df_new 是最近 N 个交易日重拉的 bar，可能与 df_old 的最后 N 行重叠（旧值是开盘
    盘中快照，需要用收盘完整值覆盖）。删除 df_old 中 time >= df_new 最小 time 的旧行，
    再拼接 df_new，保证每个交易日只保留最新一份。
    """
    min_new_time = df_new['time'].min()
    df_old = df_old[df_old['time'] < min_new_time]
    df_merged = pd.concat([df_new, df_old], ignore_index=True)
    df_merged.sort_values('time', ascending=False, inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
    return df_merged


def _info(msg: str, *args):
    logger.info(msg, *args)
    try:
        from trading.logger import trading_logger
        trading_logger.info(msg % args if args else msg)
    except Exception:
        pass


def _warning(msg: str, *args):
    logger.warning(msg, *args)
    try:
        from trading.logger import trading_logger
        trading_logger.warning(msg % args if args else msg)
    except Exception:
        pass


def _download_kline_all(download_trading_days: int = DOWNLOAD_TRADING_DAYS):
    """下载所有股票最近若干交易日 K 线，写 parquet 并返回内存数据。

    Returns:
        {code: {time, open, high, low, close, volume, amount}}  (code with .SZ/.SH suffix)
    """
    from data.db.stock_list import get_all_stock_code_list
    from mootdx.quotes import Quotes
    from data.db.history import _to_mootdx_code, _mootdx_frequency, _convert_to_wbr, _filter_and_clip_by_time

    KLINE_DIR = DATA_DIR / "k-line"
    KLINE_DIR.mkdir(parents=True, exist_ok=True)

    codes = sorted(get_all_stock_code_list())
    existing = {f.stem: f for f in KLINE_DIR.glob("*.parquet")}
    new_stocks = [c for c in codes if c not in existing]
    existing_codes = list(existing.keys())

    _info("[K线] 已有 %d 只, 新下载 %d 只", len(existing), len(new_stocks))

    fetch_offset = download_trading_days + 5
    now = datetime.now()
    kline_data = {}  # {code: bar_dict}
    lock = threading.Lock()

    def _process_batch(batch_codes):
        market = Quotes.factory()
        freq = _mootdx_frequency('1d')
        results = {'updated': 0, 'skipped': 0}
        for code in batch_codes:
            filepath = existing[code]
            bare = _to_mootdx_code(code)
            try:
                bars = market.bars(symbol=bare, frequency=freq, start=0, offset=fetch_offset)
                if bars is None or bars.empty:
                    results['skipped'] += 1
                    continue

                part = _convert_to_wbr(bars)
                if part is None:
                    results['skipped'] += 1
                    continue

                new_data = _filter_and_clip_by_time(part, now, download_trading_days)
                if not new_data:
                    results['skipped'] += 1
                    continue

                # 写 parquet：用最近 N 天重拉值覆盖本地对应日期的旧行
                df_new = pd.DataFrame(new_data)
                df_old = pd.read_parquet(filepath)
                df_merged = _merge_overwrite_by_time(df_old, df_new)
                df_merged.to_parquet(filepath, index=False)

                # 收集内存数据供 NPZ 修补
                with lock:
                    kline_data[code] = new_data
                results['updated'] += 1
            except Exception:
                results['skipped'] += 1
        return results

    batch_size = 50
    batches = [existing_codes[i:i + batch_size] for i in range(0, len(existing_codes), batch_size)]
    _info("[K线] 处理: %d 只, %d 批, 最近 %d 日", len(existing_codes), len(batches), download_trading_days)
    t1 = time.time()
    total_updated = 0

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(_process_batch, b): b for b in batches}
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            total_updated += r['updated']
            if i % 20 == 0:
                done = min(i * batch_size, len(existing_codes))
                elapsed = time.time() - t1
                _info("[K线] %d/%d (%.0f只/s), 更新=%d",
                      done, len(existing_codes), done / elapsed, total_updated)

    _info("[K线] 完成: %d 更新 / %d 只 (%.0fs), 内存 %d 只",
          total_updated, len(existing_codes), time.time() - t1, len(kline_data))

    if new_stocks:
        from data.db.history import get_history_data
        _info("[K线] 新股票全量下载: %d 只", len(new_stocks))
        t2 = time.time()
        new_ok = 0
        for code in new_stocks:
            try:
                result = get_history_data([code], count=None, base_time=datetime.now(), period="1d")
                nd = result.get(code)
                if nd is not None:
                    pd.DataFrame(nd).to_parquet(KLINE_DIR / f"{code}.parquet", index=False)
                    kline_data[code] = {
                        'time': np.array([r['time'] for r in nd], dtype=np.int64),
                        'open': np.array([r['open'] for r in nd], dtype=np.float64),
                        'high': np.array([r['high'] for r in nd], dtype=np.float64),
                        'low': np.array([r['low'] for r in nd], dtype=np.float64),
                        'close': np.array([r['close'] for r in nd], dtype=np.float64),
                        'volume': np.array([r['volume'] for r in nd], dtype=np.float64),
                        'amount': np.array([r['amount'] for r in nd], dtype=np.float64),
                    }
                    new_ok += 1
            except Exception as e:
                _warning("[K线] 新股票 %s 下载失败: %s", code, e)
        _info("[K线] 新股票完成: %d/%d 只 (%.0fs)", new_ok, len(new_stocks), time.time() - t2)

    return kline_data


def _patch_npz_incremental(kline_data: dict):
    """增量修补 NPZ：加载旧 NPZ → 用 K 线覆盖/追加对应日期行 → 保存。

    不重建财务/股本/ST 等未变更字段，直接继承旧值或 forward-fill。
    """
    OUT_DIR = DATA_DIR / "runtime"
    npz_files = sorted(OUT_DIR.glob("runtime_*.npz"))
    if not npz_files:
        _info("[NPZ增量] 无现有 NPZ，执行全量构建")
        from data.build_runtime import build_runtime
        return build_runtime()

    t0 = time.time()

    # 1. 加载旧 NPZ
    data = dict(np.load(npz_files[-1], allow_pickle=False))
    old_dates = data['trade_dates']
    stock_codes = data['stock_codes']
    n_old = len(old_dates)
    _info("[NPZ增量] 已加载: %d 日 x %d 股 (%.0fs)", n_old, len(stock_codes), time.time() - t0)

    stock_to_idx = {str(c): i for i, c in enumerate(stock_codes)}
    old_date_set = {d.astype('datetime64[D]').item() for d in old_dates}

    # 2. 收集新 K 线中的所有日期
    all_kline_dates = set()
    for bars in kline_data.values():
        for ts in bars['time']:
            dt = pd.Timestamp(int(ts), unit='ms').to_datetime64().astype('datetime64[D]').item()
            all_kline_dates.add(dt)

    new_dates = sorted(all_kline_dates - old_date_set)
    _info("[NPZ增量] 新增交易日: %d 个 (已有 %d 个)",
          len(new_dates), len(all_kline_dates - set(new_dates)))

    # 3. 如果有新日期，扩展所有 2D 数组（向前填充非 K 线字段）
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
            # 非 K 线字段：forward-fill 最后一行；K 线字段：NaN（会被步骤 4 覆盖）
            is_kline = key in ('open', 'high', 'low', 'close', 'volume', 'amount')
            filler = np.full((1, arr.shape[1]), np.nan, dtype=arr.dtype) if is_kline else arr[-1:].copy()
            new_rows = np.tile(filler, (n_new, 1))
            data[key] = np.concatenate([arr, new_rows], axis=0)

    # 4. 重建日期索引（数组可能已扩展）
    trade_dates = data['trade_dates']
    date_to_idx = {}
    for i in range(len(trade_dates)):
        date_to_idx[trade_dates[i].astype('datetime64[D]').item()] = i

    # 5. 逐条覆盖 K 线数据
    updated_cells = 0
    for code, bars in kline_data.items():
        si = stock_to_idx.get(code)
        if si is None:
            continue
        times = bars['time']
        for k in range(len(times)):
            dt = pd.Timestamp(int(times[k]), unit='ms').to_datetime64().astype('datetime64[D]').item()
            di = date_to_idx.get(dt)
            if di is None:
                continue
            data['open'][di, si] = bars['open'][k]
            data['high'][di, si] = bars['high'][k]
            data['low'][di, si] = bars['low'][k]
            data['close'][di, si] = bars['close'][k]
            data['volume'][di, si] = bars['volume'][k]
            data['amount'][di, si] = bars['amount'][k]
            updated_cells += 1

    _info("[NPZ增量] 覆盖 %d 个 (date,stock) 单元格 (%.0fs)",
          updated_cells, time.time() - t0)

    # 6. 删除旧 NPZ，保存新 NPZ
    for f in npz_files:
        f.unlink()

    td = data['trade_dates']
    output_path = OUT_DIR / f"runtime_{str(td[0])}_{str(td[-1])}.npz"
    np.savez_compressed(output_path, **data)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    _info("[NPZ增量] 保存: %s (%.1f MB, %.0fs)", output_path.name, file_size_mb, elapsed)
    return output_path


def update_live_quick(download_trading_days: int = DOWNLOAD_TRADING_DAYS):
    t0 = time.time()
    _info("=" * 60)
    _info("开盘快速更新: 重拉最近 %d 个交易日 K线 → 覆盖 parquet + 增量修补 NPZ",
          download_trading_days)
    _info("=" * 60)

    # Phase 1: 下载 K 线 + 写 parquet + 收集内存数据
    _info("--- Phase 1: K线下载 ---")
    kline_data = _download_kline_all(download_trading_days)

    # Phase 2: 增量修补 NPZ（只改 K 线字段，不动财务/股本/ST）
    _info("--- Phase 2: 增量修补 NPZ ---")
    _patch_npz_incremental(kline_data)

    _info("=" * 60)
    _info("开盘更新完成! 总耗时 %.0fs", time.time() - t0)
    _info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    update_live_quick()
