"""09:25 / 15:00 T-open 快照与 K 线更新。

- 09:25 / 15:00：update_live_quick(patch_npz=False) → 腾讯 T-open 纯内存 overlay
- 16:00 update_all：通过 mootdx 重拉 K 线 + build_runtime 全量写 NPZ（权威落盘）

锚定日：
- 未传 anchor：用 date.today() 在交易日历中的最近交易日（0605 凌晨常是 0604）
- --skip 202606040925：anchor=2026-06-04 → 只接受腾讯返回 0604 的开盘快照
"""
import time
import logging
import math
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("update_live")

DATA_DIR = Path(__file__).resolve().parent
DOWNLOAD_TRADING_DAYS = 1
TENCENT_OPEN_CHUNK_SIZE = 250
TENCENT_OPEN_REQUEST_TIMEOUT = 10
# 实盘因子是全市场截面排名；缺口超过 1% 已足以改变 Top-N，必须中止。
TENCENT_OPEN_MIN_COVERAGE = 0.99
TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="

_RAW_PATCH_FIELDS = ('open', 'high', 'low', 'close', 'volume', 'amount', 'preClose')
_KLINE_NANFILL_FIELDS = _RAW_PATCH_FIELDS
_TENCENT_QUOTE_RE = re.compile(r'v_([a-z]{2}\d{6})="([^"]*)"')


def _info(msg: str, *args):
    logger.info(msg, *args)
    try:
        from trading.logger import trading_logger
        trading_logger.info(msg % args if args else msg)
    except Exception as e:
        logger.debug(f"trading_logger 转发失败 (非实盘环境可忽略): {e}")


def _download_kline_all(download_trading_days: int = DOWNLOAD_TRADING_DAYS,
                        anchor_date: date | None = None,
                        codes: list[str] | None = None) -> dict:
    from data.update_kline import update_recent
    return update_recent(download_trading_days, anchor_date=anchor_date, collect=True,
                         codes=codes)


def _tencent_quote_symbol(stock_code: str) -> tuple[str, str]:
    """把 WBR 股票代码转成腾讯行情 symbol，并返回规范化的请求代码。"""
    code = str(stock_code).strip().upper()
    match = re.fullmatch(r'(\d{6})(?:\.(SH|SZ|BJ))?', code)
    if match is None:
        raise ValueError(f"腾讯 T-open 股票代码非法: {stock_code!r}")

    bare, exchange = match.groups()
    if exchange is None:
        if bare.startswith(('43', '83', '87', '92')):
            exchange = 'BJ'
        elif bare.startswith(('5', '6', '9')):
            exchange = 'SH'
        else:
            exchange = 'SZ'
    canonical = f'{bare}.{exchange}'
    return f'{exchange.lower()}{bare}', canonical


def _requested_tencent_symbols(codes: list[str]) -> tuple[list[str], dict[str, str]]:
    """建立有序 symbol 映射；同一市场代码重复请求会立即拒绝。"""
    symbols: list[str] = []
    symbol_to_code: dict[str, str] = {}
    for code in codes:
        symbol, canonical = _tencent_quote_symbol(code)
        if symbol in symbol_to_code:
            raise ValueError(
                f"腾讯 T-open 请求含重复代码: "
                f"{symbol_to_code[symbol]!r}, {canonical!r} -> {symbol}"
            )
        symbols.append(symbol)
        symbol_to_code[symbol] = canonical
    if not symbols:
        raise ValueError("腾讯 T-open 请求代码列表为空")
    return symbols, symbol_to_code


def _parse_tencent_open_batch(
        payload: bytes, *, expected_symbols: set[str], trade_date: date,
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    """解析一批腾讯行情，只取 open/preClose，其他当日字段不读取。"""
    try:
        text = payload.decode('gbk')
    except UnicodeDecodeError as exc:
        raise RuntimeError("腾讯 T-open 响应不是有效 GBK") from exc

    parsed: dict[str, tuple[float, float]] = {}
    rejected: dict[str, str] = {}
    statements = [item.strip() for item in text.split(';') if item.strip()]
    for statement in statements:
        match = _TENCENT_QUOTE_RE.fullmatch(statement)
        if match is None:
            raise RuntimeError(f"腾讯 T-open 响应畸形: {statement[:160]!r}")
        symbol, packed = match.groups()
        if symbol not in expected_symbols:
            raise RuntimeError(f"腾讯 T-open 返回未请求代码: {symbol}")
        if symbol in parsed or symbol in rejected:
            raise RuntimeError(f"腾讯 T-open 响应代码重复: {symbol}")

        values = packed.split('~')
        if len(values) <= 30:
            raise RuntimeError(
                f"腾讯 T-open 响应畸形: {symbol} 只有 {len(values)} 个字段"
            )
        if values[2].strip().zfill(6) != symbol[2:]:
            raise RuntimeError(
                f"腾讯 T-open 代码不一致: symbol={symbol}, field={values[2]!r}"
            )

        date_text = values[30].strip()[:8]
        try:
            quote_date = datetime.strptime(date_text, '%Y%m%d').date()
        except ValueError as exc:
            raise RuntimeError(
                f"腾讯 T-open 日期畸形: {symbol} field30={values[30]!r}"
            ) from exc
        if quote_date != trade_date:
            raise RuntimeError(
                f"腾讯 T-open 错日: {symbol} 返回 {quote_date.isoformat()}, "
                f"请求 {trade_date.isoformat()}"
            )

        try:
            open_price = float(values[5])
            pre_close = float(values[4])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"腾讯 T-open 价格畸形: {symbol} "
                f"open={values[5]!r}, preClose={values[4]!r}"
            ) from exc
        if not (math.isfinite(open_price) and math.isfinite(pre_close)):
            raise RuntimeError(
                f"腾讯 T-open 价格非有限数: {symbol} "
                f"open={open_price!r}, preClose={pre_close!r}"
            )
        if open_price < 0 or pre_close <= 0:
            rejected[symbol] = f'open={open_price:g},preClose={pre_close:g}'
            continue
        parsed[symbol] = (open_price, pre_close)
    return parsed, rejected


def fetch_tencent_open_overlay(
        codes: list[str], trade_date: date, *,
        min_coverage: float = TENCENT_OPEN_MIN_COVERAGE,
) -> dict[str, dict[str, np.ndarray]]:
    """Fetch a causal T-open-only in-memory overlay from Tencent quotes.

    The returned row deliberately sets ``high/low/close/volume/amount`` to NaN.
    It is never merged into parquet or written to NPZ by this function.
    """
    if not (0 < min_coverage <= 1):
        raise ValueError(f'min_coverage 必须在 (0, 1] 内: {min_coverage!r}')
    symbols, symbol_to_code = _requested_tencent_symbols(codes)
    parsed: dict[str, tuple[float, float]] = {}
    rejected: dict[str, str] = {}
    headers = {'User-Agent': 'Mozilla/5.0'}

    for start in range(0, len(symbols), TENCENT_OPEN_CHUNK_SIZE):
        chunk = symbols[start:start + TENCENT_OPEN_CHUNK_SIZE]
        response = requests.get(
            TENCENT_QUOTE_URL + ','.join(chunk),
            headers=headers,
            timeout=TENCENT_OPEN_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        batch_parsed, batch_rejected = _parse_tencent_open_batch(
            response.content,
            expected_symbols=set(chunk),
            trade_date=trade_date,
        )
        overlap = (set(parsed) | set(rejected)) & (
            set(batch_parsed) | set(batch_rejected)
        )
        if overlap:
            raise RuntimeError(
                f"腾讯 T-open 跨批次响应代码重复: {sorted(overlap)}"
            )
        parsed.update(batch_parsed)
        rejected.update(batch_rejected)

    requested = set(symbols)
    missing = sorted(requested - set(parsed) - set(rejected))
    valid_count = len(parsed)
    if parsed and not any(price > 0 for price, _ in parsed.values()):
        raise RuntimeError('腾讯 T-open 尚未就绪: 全部响应均无有效开盘价')
    required_count = math.ceil(len(symbols) * min_coverage)
    if valid_count < required_count:
        rejected_text = ','.join(
            f'{symbol}({reason})' for symbol, reason in sorted(rejected.items())
        ) or '-'
        missing_text = ','.join(missing) or '-'
        raise RuntimeError(
            f"腾讯 T-open 覆盖率不足: {valid_count}/{len(symbols)}="
            f"{valid_count / len(symbols):.2%}, 要求>={min_coverage:.2%}; "
            f"无响应=[{missing_text}]; 非正价格=[{rejected_text}]"
        )
    if missing or rejected:
        _info(
            "[腾讯T-open] 有效 %d/%d，跳过无响应 %d、非正价格 %d",
            valid_count, len(symbols), len(missing), len(rejected),
        )

    time_ms = np.int64(pd.Timestamp(trade_date).value // 1_000_000)
    overlay: dict[str, dict[str, np.ndarray]] = {}
    for symbol in symbols:
        quote = parsed.get(symbol)
        if quote is None:
            continue
        open_price, pre_close = quote
        overlay[symbol_to_code[symbol]] = {
            'time': np.asarray([time_ms], dtype=np.int64),
            'open': np.asarray([open_price], dtype=float),
            'high': np.asarray([np.nan], dtype=float),
            'low': np.asarray([np.nan], dtype=float),
            'close': np.asarray([np.nan], dtype=float),
            'volume': np.asarray([np.nan], dtype=float),
            'amount': np.asarray([np.nan], dtype=float),
            'preClose': np.asarray([pre_close], dtype=float),
        }
    return overlay


def load_local_open_overlay(
        codes: list[str], trade_date: date, *,
        min_coverage: float = TENCENT_OPEN_MIN_COVERAGE,
) -> dict[str, dict[str, np.ndarray]]:
    """Build a causal historical T-open overlay from local immutable K-lines."""
    if not (0 < min_coverage <= 1):
        raise ValueError(f'min_coverage 必须在 (0, 1] 内: {min_coverage!r}')
    target_ms = np.int64(pd.Timestamp(trade_date).value // 1_000_000)
    overlay: dict[str, dict[str, np.ndarray]] = {}
    for code in codes:
        path = DATA_DIR / 'k-line' / f'{code}.parquet'
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=['time', 'open', 'preClose'])
        bar_dates = pd.to_datetime(frame['time'].astype('int64'), unit='ms').dt.date
        row = frame.loc[bar_dates == trade_date]
        if row.empty:
            continue
        open_price = float(row['open'].iloc[-1])
        pre_close = float(row['preClose'].iloc[-1])
        if not (
            math.isfinite(open_price) and open_price >= 0
            and math.isfinite(pre_close) and pre_close > 0
        ):
            continue
        overlay[code] = {
            'time': np.asarray([target_ms], dtype=np.int64),
            'open': np.asarray([open_price], dtype=float),
            'high': np.asarray([np.nan], dtype=float),
            'low': np.asarray([np.nan], dtype=float),
            'close': np.asarray([np.nan], dtype=float),
            'volume': np.asarray([np.nan], dtype=float),
            'amount': np.asarray([np.nan], dtype=float),
            'preClose': np.asarray([pre_close], dtype=float),
        }
    required_count = math.ceil(len(codes) * min_coverage)
    if overlay and not any(bars['open'][0] > 0 for bars in overlay.values()):
        raise RuntimeError('本地 T-open 尚未就绪: 全部响应均无有效开盘价')
    if len(overlay) < required_count:
        raise RuntimeError(
            f"本地历史 T-open 覆盖率不足: {len(overlay)}/{len(codes)}="
            f"{len(overlay) / len(codes):.2%}, 要求>={min_coverage:.2%}"
        )
    _info("[本地T-open] 有效 %d/%d，日期=%s", len(overlay), len(codes), trade_date)
    return overlay


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
    if 'st_mask' in data:
        from data.db.stock_name import build_st_mask
        status_dates = np.array(sorted(all_kline_dates), dtype='datetime64[D]')
        status_rows = np.searchsorted(trade_dates, status_dates)
        data['st_mask'][status_rows] = build_st_mask(data['stock_codes'], status_dates)
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

    td = data['trade_dates']
    output_path = OUT_DIR / f"runtime_{str(td[0])}_{str(td[-1])}.npz"
    from data.build_runtime import save_runtime_npz_atomic
    save_runtime_npz_atomic(output_path, **data)
    for f in npz_files:
        if f != output_path:
            f.unlink()
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    _info("[NPZ增量] 保存: %s (%.1f MB, %.0fs)", output_path.name, file_size_mb, time.time() - t0)
    return output_path


def update_live_quick(download_trading_days: int = DOWNLOAD_TRADING_DAYS, *,
                      patch_npz: bool = False,
                      anchor_date: date | None = None,
                      codes: list[str] | None = None) -> dict:
    """快速获取当日开盘价。

    默认使用腾讯行情构造 T-open 纯内存 overlay，不写 parquet/NPZ；
    只有显式 ``patch_npz=True`` 才调用 mootdx 并落盘。
    ``codes`` 非空时只请求指定股票（用于实盘 prefilter 加速）。
    """
    from data.kline_mootdx import resolve_recent_range

    t0 = time.time()
    _, _, end_d = resolve_recent_range(download_trading_days, anchor_date)
    anchor_note = f"锚定={end_d.isoformat()}" + (
        f" (来自 --skip {anchor_date})" if anchor_date else " (日历最近交易日)")
    _info("=" * 60)
    _info("快速K线: 最近 %d 日 %s | %s%s",
          download_trading_days,
          "mootdx parquet + NPZ落盘" if patch_npz else "腾讯 T-open 纯内存",
          anchor_note,
          f" | 子集 {len(codes)}只" if codes else "")
    _info("=" * 60)

    if patch_npz:
        _info("--- Phase 1: mootdx K线下载 → parquet ---")
        kline_data = _download_kline_all(
            download_trading_days,
            anchor_date=anchor_date,
            codes=codes,
        )
        _info("--- Phase 2: NPZ 落盘 ---")
        _patch_npz_incremental(kline_data)
    else:
        if download_trading_days != 1:
            raise ValueError(
                "腾讯 T-open 纯内存路径只支持 1 个交易日; "
                "历史更新请使用 patch_npz=True"
            )
        if codes is None:
            from data.db.stock_list import get_all_stock_code_list
            codes = get_all_stock_code_list(target_date=end_d)
        _info("--- Phase 1: 腾讯批量开盘快照 → 内存 overlay ---")
        kline_data = fetch_tencent_open_overlay(codes, end_d)
        _info("--- Phase 2: 跳过 parquet/NPZ（T-open 内存契约）---")

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
