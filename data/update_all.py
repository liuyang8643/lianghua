"""全量数据预下载更新脚本

update_offline_toNow()  — 下载并完整校验新快照，再原子替换本地数据

用法:
  uv run python data/update_all.py

原则（来自 CLAUDE.md）：
  实盘开盘时触发预下载，此时获取到的日线 close/high/low 是盘中快照而非收盘值。
  K 线增量入口会按时间键覆盖最近窗口；其它快照禁止预先破坏旧正式文件。
"""
import time
import logging
import os
import queue
import re
import threading
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import json as _json_lib
import requests  # noqa: F401

# 注意：不要在模块层面调用 logging.basicConfig — 它会抢占 Python stdlib 的 root logger，
# 导致 lark_oapi / xtquant 等三方库的日志被重复输出（飞书/QMT 日志会打两遍）。
# 独立运行时（__main__）才配置 root logger。


def _post_json_with_retry(url, max_attempts=3, **kwargs):
    """对 requests.post 做 3 次重试，仅对网络/JSON 解析错误 retry，其他直接 raise。

    网络抖动是预下载里最常见的临时错误；缺字段/缺数据应直接报错让人工排查。
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(url, **kwargs)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, _json_lib.JSONDecodeError) as e:
            last_exc = e
            if attempt < max_attempts:
                time.sleep(min(attempt * 2, 5))
                continue
    raise RuntimeError(f"POST {url} 重试 {max_attempts} 次均失败: {last_exc}") from last_exc


logger = logging.getLogger("wbr.data.update_all")

DATA_DIR = Path(__file__).resolve().parent
TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)



# 16:00 全量更新时，K 线删除并重拉「最近 N 个交易日」，用收盘后的完整 OHLC
# 覆盖开盘抓到的盘中快照（开盘只拉当天，不做覆盖）。
REPULL_TRADING_DAYS = 3
EXTERNAL_CALL_TIMEOUT = 10


class DelistedKlineUnavailableError(RuntimeError):
    """Fail-closed delisted-universe error with machine-readable causes."""

    def __init__(self, diagnostics: list[dict[str, object]]) -> None:
        self.diagnostics = tuple(diagnostics)
        kinds: dict[str, int] = {}
        for item in diagnostics:
            failure = item.get("live_failure") or item.get("restore_failure") or {}
            kind = str(failure.get("kind", "missing_source_data"))
            kinds[kind] = kinds.get(kind, 0) + 1
        summary = ", ".join(f"{kind}={count}" for kind, count in sorted(kinds.items()))
        shown = ", ".join(str(item["code"]) for item in diagnostics[:20])
        suffix = f" ...(+{len(diagnostics) - 20})" if len(diagnostics) > 20 else ""
        super().__init__(
            "[K线-退市] 真实源与本地原始备份均未能形成可验证行情，"
            f"拒绝继续生成不完整 runtime: {len(diagnostics)} 只 "
            f"({summary}): {shown}{suffix}"
        )


def _parse_st_records(
    records: object,
    expected_codes: set[str],
) -> pd.DataFrame:
    """Parse one complete CNINFO ST snapshot and reject partial responses."""
    if not isinstance(records, list) or not records:
        raise ValueError("CNINFO ST records 必须是非空列表")

    required = {"VARYDATE", "F006V", "F002V", "SECCODE"}
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"CNINFO ST records[{index}] 不是对象")
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"CNINFO ST records[{index}] 缺少字段: {sorted(missing)}"
            )
        if any(record[field] is None for field in required):
            raise ValueError(f"CNINFO ST records[{index}] 包含空字段")
        code = str(record["SECCODE"]).strip().zfill(6)
        event = str(record["F006V"]).strip()
        status = str(record["F002V"]).strip()
        try:
            event_date = date.fromisoformat(str(record["VARYDATE"]).strip())
        except ValueError as exc:
            raise ValueError(
                f"CNINFO ST records[{index}] VARYDATE 非法"
            ) from exc
        rows.append({
            "bare_code": code,
            "date": event_date,
            "event": event,
            "status": status,
        })

    frame = pd.DataFrame(rows).sort_values(
        ["bare_code", "date"], kind="stable"
    ).reset_index(drop=True)
    from data.db.stock_name import validate_st_changes

    validate_st_changes(frame)
    missing_codes = expected_codes.difference(frame["bare_code"].astype(str))
    if missing_codes:
        shown = ", ".join(sorted(missing_codes)[:20])
        suffix = f" ...(+{len(missing_codes) - 20})" if len(missing_codes) > 20 else ""
        raise ValueError(
            "CNINFO ST 全量响应未覆盖本地股票全集: "
            f"缺 {len(missing_codes)} 只: {shown}{suffix}"
        )
    return frame


def _save_st_snapshot_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    """Atomically replace ST history without allowing a snapshot to shrink."""
    from data.db.stock_name import invalidate_name_data_cache, validate_st_changes

    validate_st_changes(frame)
    if output_path.exists():
        previous = pd.read_parquet(output_path)
        validate_st_changes(previous)
        key_columns = ["bare_code", "date", "event", "status"]

        def _canonical_rows(value: pd.DataFrame) -> set[tuple[object, ...]]:
            normalized = value[key_columns].copy()
            normalized["bare_code"] = normalized["bare_code"].astype(str)
            normalized["date"] = pd.to_datetime(
                normalized["date"], errors="raise"
            ).dt.date
            return set(normalized.itertuples(index=False, name=None))

        if not _canonical_rows(previous).issubset(_canonical_rows(frame)):
            raise ValueError(
                "CNINFO ST 全量快照发生历史收缩，拒绝覆盖现有文件"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp.parquet"
    )
    try:
        frame.to_parquet(temp_path, index=False)
        validate_st_changes(pd.read_parquet(temp_path))
        temp_path.replace(output_path)
        invalidate_name_data_cache()
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _call_with_timeout(func, timeout=EXTERNAL_CALL_TIMEOUT):
    """Run a potentially blocking third-party call and return after timeout."""
    result = queue.Queue(maxsize=1)

    def _run():
        try:
            result.put((True, func()))
        except BaseException as exc:
            result.put((False, exc))

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"external call exceeded {timeout}s")

    ok, value = result.get_nowait()
    if not ok:
        raise value
    return value

INDEX_INFO = {
    'sh000001': '上证指数',
    'H00300': '沪深300全收益',
    'sh000905': '中证500',
    'sh000852': '中证1000',
}
INDEX_REQUIRED_COLUMNS = ('trade_date', 'open', 'close')


# ============================================================
# 辅助函数
# ============================================================

def _is_valid_index_parquet(path: Path, min_trade_date: date | None = None) -> bool:
    if not path.exists():
        return False
    try:
        frame = _validate_index_snapshot(pd.read_parquet(path))
    except (OSError, ValueError, TypeError):
        return False
    return min_trade_date is None or frame['trade_date'].max().date() >= min_trade_date


def _validate_index_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != list(INDEX_REQUIRED_COLUMNS) or frame.empty:
        raise ValueError("指数快照列不兼容或为空")
    result = frame.copy()
    result['trade_date'] = pd.to_datetime(result['trade_date'], errors='raise')
    if result['trade_date'].duplicated().any():
        raise ValueError("指数快照 trade_date 重复")
    if not result['trade_date'].is_monotonic_increasing:
        raise ValueError("指数快照 trade_date 必须严格递增")
    for field in ('open', 'close'):
        result[field] = pd.to_numeric(result[field], errors='coerce').astype(np.float64)
        values = result[field].to_numpy(dtype=np.float64)
        if np.isinf(values).any() or np.any(values[np.isfinite(values)] <= 0.0):
            raise ValueError(f"指数快照 {field} 包含非法值")
    close_values = result['close'].to_numpy(dtype=np.float64)
    if not np.isfinite(close_values).all():
        raise ValueError("指数快照 close 不得缺失")
    return result.reset_index(drop=True)


def _save_index_snapshot_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    current = _validate_index_snapshot(frame)
    if output_path.exists():
        previous = _validate_index_snapshot(pd.read_parquet(output_path))
        lost = previous['trade_date'][
            ~previous['trade_date'].isin(current['trade_date'])
        ]
        if not lost.empty:
            raise ValueError(
                f"指数历史响应收缩 {len(lost)} 日，拒绝覆盖: "
                f"{lost.dt.strftime('%Y-%m-%d').tolist()[:20]}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp.parquet"
    )
    try:
        current.to_parquet(temp_path, index=False)
        verified = _validate_index_snapshot(pd.read_parquet(temp_path))
        pd.testing.assert_frame_equal(verified, current, check_exact=True)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _indices_ready_today(symbols=None) -> bool:
    from datetime import datetime as _dt
    from utils.stock.time import get_last_trading_day

    target_symbols = tuple(symbols) if symbols is not None else tuple(INDEX_INFO)
    min_trade_date = get_last_trading_day(TODAY - timedelta(days=1))
    for symbol in target_symbols:
        path = DATA_DIR / f"index_{symbol}_daily.parquet"
        if not _is_valid_index_parquet(path, min_trade_date):
            return False
        if _dt.fromtimestamp(path.stat().st_mtime).date() != TODAY:
            return False
    return True


# ============================================================
# 1. K线日线 — mootdx 唯一源（不复权 k-line/ + preClose；复权由 build_runtime 自建）
# ============================================================

def _update_kline(anchor_date: date | None = None):
    """mootdx 刷新：已有股票增量合并最近 REPULL_TRADING_DAYS 个交易日，新股全量补齐。

    退市股缺失时原始 OHLCVA 仍只取 mootdx/本地 mootdx 备份；Baostock
    仅可提供 direct preClose 与逐日生命周期证据。
    """
    from data.db.issue_price import resolve_terminal_active_codes
    from data.db.stock_list import load_current_stock_codes
    from data.kline_mootdx import resolve_recent_range, update_recent

    # Restore the historical delisted universe first.  Passing the delisted
    # union into a strict recent pull would fail before the backup migration,
    # because the live bars endpoint no longer serves many delisted symbols.
    _ensure_delist_kline_mootdx()
    _, _, terminal = resolve_recent_range(REPULL_TRADING_DAYS, anchor_date)
    kline_dir = DATA_DIR / "k-line"
    active_codes = list(
        resolve_terminal_active_codes(
            load_current_stock_codes(),
            terminal,
            (path.stem for path in kline_dir.glob("*.parquet")),
        )
    )
    logger.info("[K线] mootdx 重拉最近 %d 个交易日 + 新股全量", REPULL_TRADING_DAYS)
    update_recent(
        REPULL_TRADING_DAYS,
        anchor_date=anchor_date,
        codes=active_codes,
        strict=True,
    )


def _ensure_delist_kline_mootdx():
    """Restore missing delisted bars, then block any incomplete universe."""
    from data.db.delist import get_delist_stock_info
    from data.kline_mootdx import (
        KlineBatchError,
        KlineSourceError,
        restore_backup_bars,
        update_full,
    )
    from utils.stock.info import is_b_stock

    kline_dir = DATA_DIR / "k-line"
    delist_info = get_delist_stock_info()
    missing = [c for c in delist_info
               if not (kline_dir / f'{c}.parquet').exists() and not is_b_stock(c)]
    if not missing:
        return

    diagnostic_by_code: dict[str, dict[str, object]] = {
        code: {
            "code": code,
            "backup_exists": (
                DATA_DIR / "k-line-mootdx-bak" / f"{code}.parquet"
            ).exists(),
            "restore_failure": None,
            "live_failure": None,
        }
        for code in missing
    }

    def _failure(exc: Exception) -> dict[str, object]:
        if isinstance(exc, KlineSourceError):
            return exc.as_dict()
        return {
            "operation": "local_backup_validation",
            "kind": "local_validation",
            "detail": f"{type(exc).__name__}: {exc}",
            "attempts": 1,
        }

    # Delisted symbols are commonly unavailable from the live bars endpoint.
    # The local mootdx backup contributes only raw OHLCVA.  ``restore_backup_bars``
    # tries current xdxr first; when unavailable it uses Baostock direct
    # preClose/daily lifecycle evidence before atomic per-file publication.
    logger.info("[K线-退市] 从 mootdx 原始备份迁移 %d 只", len(missing))
    try:
        restored = restore_backup_bars(
            missing,
            backup_dir=DATA_DIR / "k-line-mootdx-bak",
            evidence_path=(
                DATA_DIR / "kline_evidence" / "baostock_daily_reference.parquet"
            ),
            strict=True,
        )
    except KlineBatchError as exc:
        restored = {
            code: kline_dir / f"{code}.parquet"
            for code in exc.succeeded
        }
        for code, failure in exc.failures:
            diagnostic_by_code[code]["restore_failure"] = _failure(failure)
    except KlineSourceError as exc:
        restored = {}
        for code in missing:
            if diagnostic_by_code[code]["backup_exists"]:
                diagnostic_by_code[code]["restore_failure"] = _failure(exc)
    remaining = [code for code in missing if code not in restored]
    if remaining:
        logger.info(
            "[K线-退市] 备份未覆盖或未通过校验，mootdx 全量补齐 %d 只",
            len(remaining),
        )
        try:
            update_full(codes=remaining, strict=True)
        except KlineBatchError as exc:
            for code, failure in exc.failures:
                diagnostic_by_code[code]["live_failure"] = _failure(failure)
        except KlineSourceError as exc:
            failure = _failure(exc)
            for code in remaining:
                diagnostic_by_code[code]["live_failure"] = failure
    still_missing = [c for c in missing if not (kline_dir / f'{c}.parquet').exists()]
    if still_missing:
        raise DelistedKlineUnavailableError(
            [diagnostic_by_code[code] for code in still_missing]
        )


# ============================================================
# 2. 股票列表
# ============================================================

def _save_stock_list_snapshot_atomic(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    as_of: date | None = None,
) -> None:
    """Atomically replace a complete current-market snapshot.

    A previously current code may disappear only after the local delist source
    says it has actually delisted.  This rejects empty/partial xtdata responses
    before they can erase the production universe.
    """
    from data.db.delist import get_delist_stock_info
    from data.db.stock_list import validate_stock_list_frame

    current = validate_stock_list_frame(frame)
    cutoff = as_of or TODAY
    if output_path.exists():
        previous = validate_stock_list_frame(pd.read_parquet(output_path))
        delisted = {
            code
            for code, info in get_delist_stock_info().items()
            if info.delist_date <= cutoff
        }
        still_listed = set(previous["stock_code"]).difference(delisted)
        lost = sorted(still_listed.difference(current["stock_code"]))
        if lost:
            shown = ", ".join(lost[:20])
            suffix = f" ...(+{len(lost) - 20})" if len(lost) > 20 else ""
            raise ValueError(
                "stock_list 全量快照丢失仍在市股票，拒绝覆盖: "
                f"{shown}{suffix}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp.parquet"
    )
    try:
        current.to_parquet(temp_path, index=False)
        validate_stock_list_frame(pd.read_parquet(temp_path))
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _update_stock_list():
    from xtquant import xtdata

    response = xtdata.get_stock_list_in_sector('沪深A股')
    if not isinstance(response, (list, tuple, set)) or not response:
        raise ValueError("xtdata 沪深A股股票列表返回空或类型非法")
    codes = sorted(str(code).strip().upper() for code in response)
    df = pd.DataFrame({
        'stock_code': codes,
        'exchange': [code.rsplit('.', 1)[-1] for code in codes],
    })
    path = DATA_DIR / "stock_list" / "stock_list.parquet"
    _save_stock_list_snapshot_atomic(df, path)
    logger.info("[股票列表] 保存 %d 只到 %s", len(codes), path)


# ============================================================
# 3. 股票名称/ST历史 — CNINFO API
# ============================================================

CURRENT_NAMES_REQUEST_TIMEOUT = 10
CURRENT_NAMES_CHUNK_SIZE = 250


def _tencent_quote_symbol(stock_code: str) -> str:
    from utils.stock.info import is_bse_stock

    bare = stock_code.split(".")[0]
    if is_bse_stock(bare):
        prefix = "bj"
    elif bare.startswith(("6", "9")):
        prefix = "sh"
    else:
        prefix = "sz"
    return f"{prefix}{bare}"


def _fetch_current_stock_names(codes: list[str]) -> pd.DataFrame:
    """当前简称全量表：腾讯财经批量行情只取简称，避免 akshare 黑盒调用阻塞。"""
    symbols = [_tencent_quote_symbol(code) for code in sorted(set(codes))]
    rows = []
    total_batches = (len(symbols) + CURRENT_NAMES_CHUNK_SIZE - 1) // CURRENT_NAMES_CHUNK_SIZE
    headers = {"User-Agent": "Mozilla/5.0"}
    for batch_idx, start in enumerate(range(0, len(symbols), CURRENT_NAMES_CHUNK_SIZE), 1):
        chunk = symbols[start:start + CURRENT_NAMES_CHUNK_SIZE]
        url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
        r = requests.get(url, headers=headers, timeout=CURRENT_NAMES_REQUEST_TIMEOUT)
        r.raise_for_status()
        text = r.content.decode("gbk", errors="ignore")
        chunk_rows = []
        for item in text.split(";"):
            if '="' not in item:
                continue
            payload = item.split('"', 1)[1].rsplit('"', 1)[0]
            fields = payload.split("~")
            if len(fields) >= 3 and fields[1].strip() and fields[2].strip():
                chunk_rows.append(
                    {"code": fields[2].strip().zfill(6), "name": fields[1].strip()}
                )
        expected = {symbol[2:] for symbol in chunk}
        actual = {str(row["code"]) for row in chunk_rows}
        if actual != expected:
            missing = sorted(expected.difference(actual))
            extra = sorted(actual.difference(expected))
            raise RuntimeError(
                "[当前简称] 腾讯批次响应未严格覆盖请求: "
                f"missing={missing[:20]}, extra={extra[:20]}"
            )
        rows.extend(chunk_rows)
        if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == total_batches:
            logger.info("[当前简称] 腾讯批量 %d/%d, 已解析 %d 条", batch_idx, total_batches, len(rows))
    result = pd.DataFrame(rows, columns=["code", "name"])
    if result["code"].duplicated().any():
        raise RuntimeError("[当前简称] 腾讯响应包含重复代码")
    return result.sort_values("code", kind="stable").reset_index(drop=True)


def _update_stock_name():
    from datetime import datetime
    import requests
    import py_mini_racer
    from akshare.stock.stock_profile_cninfo import _get_file_content_ths
    from data.db.stock_list import get_all_stock_code_list

    t_step = time.time()
    OUT_DIR = DATA_DIR / "stock_name"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name_path = OUT_DIR / "name_changes.parquet"
    st_path = OUT_DIR / "st_changes.parquet"
    cur_path = OUT_DIR / "current_names.parquet"

    # 当天完整且通过校验才跳过。
    if cur_path.exists() and st_path.exists():
        current_mtime = datetime.fromtimestamp(cur_path.stat().st_mtime).date()
        st_mtime = datetime.fromtimestamp(st_path.stat().st_mtime).date()
        if current_mtime == TODAY and st_mtime == TODAY:
            from data.db.stock_name import validate_st_changes

            validate_st_changes(pd.read_parquet(st_path))
            logger.info("[股票名称] 今日已更新, 跳过")
            return

    codes = get_all_stock_code_list()
    bare_codes = sorted(set(c.split('.')[0] for c in codes))
    logger.info("[股票名称] 开始: 全市场 %d 只 (ST + 更名历史 + 当前简称)", len(codes))

    js = py_mini_racer.MiniRacer()
    js.eval(_get_file_content_ths("cninfo.js"))
    mcode = js.call("getResCode1")
    headers = {
        "Accept": "*/*", "Accept-Enckey": mcode,
        "Origin": "https://webapi.cninfo.com.cn",
        "Referer": "https://webapi.cninfo.com.cn/",
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
    }
    CNINFO_BASE = "https://webapi.cninfo.com.cn/api/stock"

    # ===== ST 变更：全量一把拉（p_stock2117 支持不带 scode 返回全量） =====
    t0 = time.time()
    logger.info("[股票名称] 拉取 ST 变更 (CNINFO 全量)...")
    payload = _post_json_with_retry(
        f"{CNINFO_BASE}/p_stock2117",
        headers=headers,
        timeout=EXTERNAL_CALL_TIMEOUT,
    )
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError("CNINFO ST 响应缺少 records")
    df_st = _parse_st_records(payload["records"], set(bare_codes))
    _save_st_snapshot_atomic(df_st, st_path)
    logger.info("[股票名称] ST变更: %d 条 (%.0fs), %d 只股票", len(df_st), time.time() - t0, df_st['bare_code'].nunique())

    # ===== 名称变更：增量串行拉取（p_stock2109 不支持批量，限流约 2-3 请求后需等待） =====
    name_existing = set()
    if name_path.exists():
        dn = pd.read_parquet(name_path)
        if not dn.empty and 'bare_code' in dn.columns:
            name_existing = set(dn['bare_code'].unique())

    name_pending = [b for b in bare_codes if b not in name_existing]
    t_name = time.time()
    if not name_pending:
        logger.info("[股票名称] 更名历史已齐 %d 只，跳过 CNINFO 增量", len(name_existing))
    else:
        logger.info("[股票名称] 更名历史待拉 %d 只 (已有 %d)，约每 100 只打一条进度",
                    len(name_pending), len(name_existing))
        rows_name = []
        name_date_parse_fail = 0
        for i, bare in enumerate(name_pending, 1):
            try:
                resp = _call_with_timeout(
                    lambda: _post_json_with_retry(
                        f"{CNINFO_BASE}/p_stock2109",
                        max_attempts=1,
                        params={"scode": bare},
                        headers=headers,
                        timeout=EXTERNAL_CALL_TIMEOUT,
                    )
                )
            except TimeoutError:
                logger.warning("[股票名称] %s 请求超过 %ds，跳过", bare, EXTERNAL_CALL_TIMEOUT)
                continue
            except Exception as e:
                logger.warning("[股票名称] %s 请求失败，跳过: %s", bare, e)
                continue
            records = resp.get("records", [])
            for rec in records:
                start_str = rec.get("STARTDATE")
                old_name = rec.get("F002V", "")
                if not start_str or not old_name:
                    continue
                try:
                    start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
                except ValueError:
                    name_date_parse_fail += 1
                    continue
                rows_name.append({"bare_code": bare, "start_date": start_date, "old_name": old_name.strip()})

            if i == 1 or i % 100 == 0 or i == len(name_pending):
                logger.info("[股票名称] 更名历史 %d/%d (%.0fs)",
                            i, len(name_pending), time.time() - t_name)

        if name_date_parse_fail:
            logger.warning("[股票名称] 更名历史日期解析失败: %d 条", name_date_parse_fail)
        if rows_name:
            df_new = pd.DataFrame(rows_name)
            if name_path.exists() and 'bare_code' in pd.read_parquet(name_path).columns:
                df_old = pd.read_parquet(name_path)
                df_name = pd.concat([df_old, df_new], ignore_index=True)
                df_name = df_name.drop_duplicates(subset=['bare_code', 'start_date'], keep='last')
            else:
                df_name = df_new
            df_name.to_parquet(name_path, index=False)
            logger.info("[股票名称] 名称变更 新增 %d 条, 共 %d 条", len(rows_name), len(df_name))

    # 当前简称：每次 update_all 都必须刷新（与 name_changes 增量是否为空无关）
    _update_current_names(codes)
    logger.info("[股票名称] 全部完成 (%.0fs)", time.time() - t_step)


def _update_current_names(codes: list[str]):
    """写 current_names.parquet — 全市场当前简称（update_all 唯一写入入口）。

    读取侧只认此表 + name_changes（历史时点）；由腾讯财经批量行情对齐 stock_list。
    """
    from data.db.stock_name import invalidate_name_data_cache

    path = DATA_DIR / "stock_name" / "current_names.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 当天已成功落盘 → 跳过（同一交易日名称不变）
    if path.exists():
        from datetime import datetime as _dt
        mtime = _dt.fromtimestamp(path.stat().st_mtime).date()
        if mtime == TODAY:
            logger.info("[当前简称] 今日已更新, 跳过")
            return

    bare_to_code: dict[str, str] = {}
    for code in codes:
        bare_to_code.setdefault(code.split('.')[0], code)

    logger.info("[当前简称] 拉取腾讯财经批量简称 (%d 只待对齐)...", len(bare_to_code))
    df_names = _fetch_current_stock_names(codes)

    if df_names is None or df_names.empty:
        raise RuntimeError("[当前简称] 腾讯财经当前简称返回空，中止更新")

    name_map = dict(zip(
        df_names['code'].astype(str).str.zfill(6),
        df_names['name'].astype(str).str.strip(),
    ))
    rows = []
    for bare, stock_code in bare_to_code.items():
        key = bare.zfill(6) if len(bare) < 6 else bare
        nm = name_map.get(key)
        if nm:
            rows.append({'bare_code': bare, 'stock_code': stock_code, 'name': nm})

    df = pd.DataFrame(rows, columns=['bare_code', 'stock_code', 'name'])
    if len(df) != len(bare_to_code):
        missing = sorted(set(bare_to_code).difference(df['bare_code']))
        raise RuntimeError(
            "[当前简称] 腾讯财经未覆盖完整股票轴，拒绝覆盖: "
            + ", ".join(missing[:20])
        )
    if df['bare_code'].duplicated().any() or df['stock_code'].duplicated().any():
        raise RuntimeError("[当前简称] 快照代码重复，拒绝覆盖")
    if df['name'].astype(str).str.strip().eq('').any():
        raise RuntimeError("[当前简称] 快照包含空简称，拒绝覆盖")
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
    try:
        df.to_parquet(temp_path, index=False)
        verified = pd.read_parquet(temp_path)
        if not verified.equals(df):
            raise RuntimeError("[当前简称] 临时快照写后校验失败")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    invalidate_name_data_cache()
    logger.info(
        "[当前简称] %d 条完整快照已保存 (%.0fs) → %s",
        len(df), time.time() - t0, path.name,
    )


# ============================================================
# 4. 官方资产负债表（总股本）完整性校验
# ============================================================

def _validate_official_balance():
    path = DATA_DIR / "financial" / "balance.parquet"
    if not path.exists():
        raise FileNotFoundError(f"[官方股本] 缺少数据: {path}")
    frame = pd.read_parquet(path)
    required = {"stock_code", "m_anntime", "cap_stk"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"[官方股本] 缺少列: {sorted(missing)}")
    if frame.duplicated(["stock_code", "m_anntime"]).any():
        raise ValueError("[官方股本] 存在重复 (stock_code, m_anntime)")
    logger.info("[官方股本] 校验通过: %d 只、%d 条", frame['stock_code'].nunique(), len(frame))


# ============================================================
# 5. 深历史财务指标 — akshare 同花顺（runtime 财务字段唯一来源）
# ============================================================

def _update_financial_deep():
    """全市场深历史财务（同花顺 stock_financial_abstract_ths，回溯至 1990s）。
    产物 data/financial/deep_indicators.parquet 是 build_runtime 财务字段的唯一来源。"""
    from data.update_financial_deep import main as _deep_main
    _deep_main(refresh=True)


# ============================================================
# 6. 发行价与上市日 — 新浪 IPO 页面
# ============================================================

SINA_IPO_URL = (
    "https://vip.stock.finance.sina.com.cn/corp/go.php/"
    "vISSUE_NewStock/stockid/{code}.phtml"
)
ISSUE_REFERENCE_SOURCE = "sina-vISSUE_NewStock"
ISSUE_REFERENCE_TIMEOUT_SECONDS = 20


def _fetch_sina_issue_record(code: str) -> dict[str, object]:
    """Download one real IPO price/listing-date pair without importing AkShare.

    AkShare's current ``stock_individual_info_em`` no longer exposes the issue
    price.  The canonical source named in CLAUDE.md is the Sina IPO page behind
    ``stock_ipo_info``; parsing it directly also avoids AkShare's MiniRacer
    import conflict.
    """
    bare = str(code).strip().zfill(6)
    if len(bare) != 6 or not bare.isdigit():
        raise ValueError(f"非法股票代码: {code!r}")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(
                SINA_IPO_URL.format(code=bare),
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=ISSUE_REFERENCE_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            html = response.content.decode("gb18030", errors="strict")
            if re.search(rf"(?<!\d){re.escape(bare)}(?!\d)", html) is None:
                raise RuntimeError(f"新浪 IPO 页面股票代码不匹配: {bare}")
            break
        except (requests.RequestException, UnicodeDecodeError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(attempt)
    else:
        raise RuntimeError(
            f"新浪 IPO 页面重试 3 次仍失败: {bare}: {last_error}"
        ) from last_error

    candidates: set[tuple[float, date]] = set()
    for table in pd.read_html(StringIO(html)):
        if table.shape[1] < 2:
            continue
        values: dict[str, object] = {}
        for item, value in table.iloc[:, :2].itertuples(index=False, name=None):
            label = str(item).strip()
            if label.startswith("发行价"):
                values["issue_price"] = value
            elif label == "上市日期":
                values["list_date"] = value
        if set(values) != {"issue_price", "list_date"}:
            continue
        price = float(values["issue_price"])
        listed = date.fromisoformat(str(values["list_date"]).strip()[:10])
        if not np.isfinite(price) or price <= 0.0:
            raise ValueError(f"新浪 IPO 发行价无效: {bare}={price!r}")
        candidates.add((price, listed))
    if len(candidates) != 1:
        raise ValueError(
            f"新浪 IPO 页面未形成唯一发行价/上市日: {bare}, "
            f"candidates={sorted(candidates)}"
        )
    price, listed = next(iter(candidates))
    return {
        "stock_code": bare,
        "issue_price": price,
        "list_date": listed,
        "source": ISSUE_REFERENCE_SOURCE,
        "source_as_of": TODAY,
    }


def _update_issue_price():
    """Incrementally refresh the sole issue-price/listing-date snapshot."""
    from data.db.issue_price import (
        ISSUE_REFERENCE_PATH,
        ISSUE_REFERENCE_MANIFEST_PATH,
        load_issue_reference,
        save_issue_reference_atomic,
    )
    from data.db.stock_list import load_current_stock_codes

    output_path = ISSUE_REFERENCE_PATH
    kline_dir = DATA_DIR / "k-line"
    current_codes = set(load_current_stock_codes())
    bare_current = {code[:6] for code in current_codes}
    bare_kline = {path.stem[:6] for path in kline_dir.glob("*.parquet")}

    if output_path.exists() or ISSUE_REFERENCE_MANIFEST_PATH.exists():
        existing = load_issue_reference(path=output_path)
    else:
        existing = pd.DataFrame(
            columns=(
                "stock_code",
                "issue_price",
                "list_date",
                "source",
                "source_as_of",
            )
        )
    done_codes = set(existing["stock_code"].astype(str))
    axis_missing_current = bare_current.difference(bare_kline)
    remaining = sorted(
        (bare_current | bare_kline).difference(done_codes) | axis_missing_current
    )

    rows: list[dict[str, object]] = []
    failures: dict[str, Exception] = {}
    if remaining:
        logger.info("[发行价] 从新浪 IPO 页面下载 %d 只缺失标的...", len(remaining))
    for index, bare in enumerate(remaining, start=1):
        try:
            rows.append(_fetch_sina_issue_record(bare))
        except Exception as exc:
            failures[bare] = exc
            logger.warning("[发行价] %s 新浪 IPO 失败: %s", bare, exc)
        if index < len(remaining):
            time.sleep(0.05)

    if rows:
        refreshed_codes = {str(row["stock_code"]) for row in rows}
        combined = pd.concat(
            [
                existing.loc[~existing["stock_code"].isin(refreshed_codes)],
                pd.DataFrame(rows),
            ],
            ignore_index=True,
        )
    else:
        combined = existing
    if combined.empty:
        raise RuntimeError("发行价/上市日快照为空，拒绝发布")

    refresh_failures = sorted(axis_missing_current.intersection(failures))
    if refresh_failures:
        details = "; ".join(
            f"{code}: {failures[code]}" for code in refresh_failures[:20]
        )
        raise RuntimeError(
            "当前无 K 标的的上市日刷新失败，保留旧快照并拒绝猜测: "
            + details
        )
    unresolved = sorted(axis_missing_current.difference(set(combined["stock_code"])))
    if unresolved:
        shown = ", ".join(unresolved[:20])
        suffix = f" ...(+{len(unresolved) - 20})" if len(unresolved) > 20 else ""
        raise RuntimeError(
            "当前列表中无 K 线标的缺少真实上市日，不能判定为预上市: "
            f"{shown}{suffix}"
        )
    save_issue_reference_atomic(combined, path=output_path)
    logger.info(
        "[发行价] 新增 %d 条、失败 %d 条、共 %d 条，已原子封存",
        len(rows),
        len(failures),
        len(combined),
    )


# ============================================================
# 7. 大盘指数 — akshare
# ============================================================

def _update_indices(symbols=None):
    import akshare as ak

    target_symbols = tuple(symbols) if symbols is not None else tuple(INDEX_INFO)

    if symbols is None and _indices_ready_today():
        logger.info("[指数] 今日已更新, 跳过")
        return

    for symbol in target_symbols:
        name = INDEX_INFO[symbol]
        path = DATA_DIR / f"index_{symbol}_daily.parquet"
        if symbol == 'H00300':
            df_new = ak.stock_zh_index_hist_csindex(symbol='H00300', start_date='20050101', end_date='20991231')
            dates = df_new['日期'].values
            open_prices = np.full(len(dates), np.nan, dtype=np.float64)
            close_prices = df_new['收盘'].values.astype(np.float64)
        else:
            df_new = ak.stock_zh_index_daily(symbol=symbol)
            dates = df_new['date'].values
            open_prices = df_new['open'].values.astype(np.float64)
            close_prices = df_new['close'].values.astype(np.float64)

        if isinstance(dates[0], str) or isinstance(dates[0], np.str_):
            dates_np = np.array([np.datetime64(d[:10], 'D') for d in dates])
        else:
            dates_np = np.asarray(dates).astype('datetime64[D]')

        sort_idx = np.argsort(dates_np)
        dates_sorted = dates_np[sort_idx]
        open_sorted = open_prices[sort_idx]
        close_sorted = close_prices[sort_idx]

        frame = pd.DataFrame({
            'trade_date': dates_sorted,
            'open': open_sorted,
            'close': close_sorted,
        })
        _save_index_snapshot_atomic(frame, path)
        logger.info("[指数] %s(%s): %d 天, %s ~ %s",
                    name, symbol, len(dates_sorted), dates_sorted[0], dates_sorted[-1])


# ============================================================
# 9. 退市列表 — akshare
# ============================================================


def _validate_delist_snapshot(frame: pd.DataFrame):
    from data.db.delist import _parse_delist_frame

    markets = set(frame.get('exchange', pd.Series(dtype=str)).dropna().astype(str))
    if markets != {'SH', 'SZ'}:
        raise ValueError(f"退市数据必须同时包含 SH/SZ，实际为 {sorted(markets)}")
    return _parse_delist_frame(frame)


def _save_delist_snapshot_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    from data.db.delist import invalidate_delist_cache

    current = _validate_delist_snapshot(frame)
    if output_path.exists():
        previous = _validate_delist_snapshot(pd.read_parquet(output_path))
        lost: list[str] = []
        for code, old in previous.items():
            new = current.get(code)
            if (
                new is None
                or new.name != old.name
                or new.delist_date != old.delist_date
                or not set(old.list_date_candidates).issubset(
                    new.list_date_candidates
                )
            ):
                lost.append(code)
        if lost:
            shown = ", ".join(sorted(lost)[:20])
            suffix = f" ...(+{len(lost) - 20})" if len(lost) > 20 else ""
            raise ValueError(
                "退市全量快照丢失或改写历史记录，拒绝覆盖: "
                f"{shown}{suffix}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f'.{output_path.name}.{os.getpid()}.tmp.parquet'
    )
    try:
        frame.to_parquet(temp_path, index=False)
        _validate_delist_snapshot(pd.read_parquet(temp_path))
        temp_path.replace(output_path)
        invalidate_delist_cache()
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _update_delist():
    import akshare as ak

    # 当天已落盘跳过
    OUT_PATH = DATA_DIR / "delist" / "delist.parquet"
    if OUT_PATH.exists():
        from datetime import datetime as _dt
        mtime = _dt.fromtimestamp(OUT_PATH.stat().st_mtime).date()
        if mtime == TODAY:
            _validate_delist_snapshot(pd.read_parquet(OUT_PATH))
            from data.db.delist import invalidate_delist_cache

            invalidate_delist_cache()
            logger.info("[退市列表] 今日已更新, 跳过")
            return

    rows = []
    for fetch_func, market in [(ak.stock_info_sh_delist, 'SH'), (ak.stock_info_sz_delist, 'SZ')]:
        try:
            df = _call_with_timeout(fetch_func)
        except TimeoutError as exc:
            raise RuntimeError(
                f"[退市列表] {market} 请求超过 {EXTERNAL_CALL_TIMEOUT}s"
            ) from exc
        except Exception as e:
            raise RuntimeError(f"[退市列表] {market} 请求失败: {e}") from e
        if df is None or df.empty:
            raise RuntimeError(f"[退市列表] {market} 返回空数据")
        df = df.copy()
        df['exchange'] = market
        rows.append(df)

    df_all = pd.concat(rows, ignore_index=True)
    _save_delist_snapshot_atomic(df_all, OUT_PATH)
    logger.info("[退市列表] 保存 %d 条", len(df_all))


# ============================================================
# 10. 交易日历 — akshare
# ============================================================

def _update_trading_calendar():
    import akshare as ak

    OUT_PATH = DATA_DIR / "trading_calendar.parquet"
    if OUT_PATH.exists():
        from datetime import datetime as _dt
        mtime = _dt.fromtimestamp(OUT_PATH.stat().st_mtime).date()
        if mtime == TODAY:
            logger.info("[交易日历] 今日已更新, 跳过")
            return

    df = ak.tool_trade_date_hist_sina()
    if 'trade_date' not in df.columns or df.empty:
        raise ValueError("[交易日历] 数据源返回空表或缺少 trade_date")
    dates = pd.to_datetime(df['trade_date'], errors='raise').drop_duplicates().sort_values()
    current = pd.DataFrame({'trade_date': dates.to_numpy(dtype='datetime64[ns]')})
    if current.empty:
        raise ValueError("[交易日历] 数据源没有有效日期")
    if OUT_PATH.exists():
        previous = pd.read_parquet(OUT_PATH)
        if list(previous.columns) != ['trade_date'] or previous.empty:
            raise ValueError("[交易日历] 旧快照非法，拒绝静默覆盖")
        previous_dates = pd.to_datetime(previous['trade_date'], errors='raise')
        lost = previous_dates[~previous_dates.isin(current['trade_date'])]
        if not lost.empty:
            raise ValueError(
                f"[交易日历] 历史响应收缩 {len(lost)} 日，拒绝覆盖"
            )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = OUT_PATH.with_name(
        f".{OUT_PATH.name}.{os.getpid()}.tmp.parquet"
    )
    try:
        current.to_parquet(temp_path, index=False)
        verified = pd.read_parquet(temp_path)
        pd.testing.assert_frame_equal(verified, current, check_exact=True)
        temp_path.replace(OUT_PATH)
    finally:
        temp_path.unlink(missing_ok=True)
    logger.info("[交易日历] 保存 %d 天", len(current))


# ============================================================
# 11. 构建 Runtime NPZ
# ============================================================

def _update_secondary_kline_evidence() -> None:
    """Ensure exact early-calendar evidence exists before any K-line restore."""
    from data.kline_secondary_evidence import (
        DEFAULT_SNAPSHOT_PATH,
        load_verified_daily_evidence,
        refresh_verified_daily_evidence,
    )

    path = DATA_DIR / "kline_evidence" / DEFAULT_SNAPSHOT_PATH.name
    if path.exists():
        load_verified_daily_evidence(path)
        return
    refresh_verified_daily_evidence(path)
    logger.info("[K线证据] 已下载并封存早期逐日次级证据")


def _active_evidence_codes_needing_refresh(
    evidence: pd.DataFrame,
    *,
    kline_dir: Path,
) -> tuple[str, ...]:
    """Return active evidence codes whose local K-line tail moved forward.

    Daily evidence is not a one-off migration artifact: a still-listed code
    that once needed Baostock evidence must keep receiving evidence for every
    newly downloaded local bar.  Otherwise the next runtime build sees the new
    positive-open date without independent daily status and correctly fails.
    """
    required = {"stock_code", "date", "out_date"}
    if not required.issubset(evidence.columns):
        return ()
    stale: list[str] = []
    for code, rows in evidence.groupby("stock_code", sort=False):
        out_dates = pd.to_datetime(rows["out_date"], errors="coerce")
        if out_dates.notna().any():
            continue
        path = kline_dir / f"{code}.parquet"
        if not path.exists():
            continue
        local = pd.read_parquet(path, columns=["time", "open"])
        times = pd.to_datetime(
            pd.to_numeric(local["time"], errors="coerce"),
            unit="ms",
            errors="coerce",
        )
        opens = pd.to_numeric(local["open"], errors="coerce")
        positive_dates = times[opens.gt(0.0) & times.notna()]
        if positive_dates.empty:
            continue
        source_dates = pd.to_datetime(rows["date"], errors="coerce")
        if source_dates.isna().all() or (
            positive_dates.max().date() > source_dates.max().date()
        ):
            stale.append(str(code))
    return tuple(sorted(set(stale)))


def _ensure_baostock_daily_evidence(
    *,
    kline_dir: Path,
    evidence_path: Path,
) -> None:
    """Upgrade/complete independent daily evidence before offline runtime use."""
    from data.db.delist import get_delist_stock_info
    from data.kline_mootdx import (
        BAOSTOCK_EVIDENCE_SCHEMA,
        load_baostock_daily_evidence,
        rebuild_masked_kline_rows_from_backups,
        update_baostock_listing_evidence,
    )
    from utils.stock.info import is_b_stock

    previous_codes: set[str] = set()
    verified_previous = pd.DataFrame()
    requires_rebuild = not evidence_path.exists()
    if evidence_path.exists():
        raw = pd.read_parquet(evidence_path)
        if "stock_code" not in raw.columns or raw.empty:
            raise ValueError("Baostock daily evidence 旧快照为空或缺少 stock_code")
        previous_codes = set(
            raw["stock_code"].astype(str).str.strip().str.upper()
        )
        schema = set(raw.get("schema_version", pd.Series(dtype=str)).astype(str))
        requires_rebuild = schema != {BAOSTOCK_EVIDENCE_SCHEMA}
        if not requires_rebuild:
            verified_previous = load_baostock_daily_evidence(evidence_path)

    required_delisted = {
        code
        for code in get_delist_stock_info()
        if not is_b_stock(code) and (kline_dir / f"{code}.parquet").exists()
    }
    missing = required_delisted.difference(previous_codes)
    stale_active = (
        ()
        if requires_rebuild
        else _active_evidence_codes_needing_refresh(
            verified_previous,
            kline_dir=kline_dir,
        )
    )
    incremental_targets = sorted(set(missing).union(stale_active))
    if not requires_rebuild and not incremental_targets:
        return

    targets = sorted(previous_codes | required_delisted)
    if not targets:
        raise RuntimeError("没有可用于构建 Baostock daily evidence 的 K 线代码")
    if requires_rebuild:
        logger.info(
            "[Runtime] 重建 Baostock %s 逐日证据: %d 只（新增退市 %d 只）",
            BAOSTOCK_EVIDENCE_SCHEMA,
            len(targets),
            len(missing),
        )
        restored = rebuild_masked_kline_rows_from_backups(
            targets,
            kline_dir=kline_dir,
            backup_dir=DATA_DIR / "k-line-mootdx-bak",
            evidence_path=evidence_path,
        )
    else:
        logger.info(
            "[Runtime] 增量封存逐日证据: 新增退市 %d 只，仍在市尾部前滚 %d 只",
            len(missing),
            len(stale_active),
        )
        update_baostock_listing_evidence(
            incremental_targets,
            kline_dir=kline_dir,
            evidence_path=evidence_path,
        )
        restored = {}
    verified = load_baostock_daily_evidence(evidence_path)
    actual_codes = set(verified["stock_code"].astype(str))
    absent = sorted(required_delisted.difference(actual_codes))
    if absent:
        raise RuntimeError(
            "Baostock daily evidence 重建后仍缺退市股票: "
            + ", ".join(absent[:20])
        )
    remaining_stale = _active_evidence_codes_needing_refresh(
        verified,
        kline_dir=kline_dir,
    )
    if remaining_stale:
        raise RuntimeError(
            "Baostock daily evidence 更新后仍未覆盖本地 K 线尾部: "
            + ", ".join(remaining_stale[:20])
        )
    logger.info(
        "[Runtime] Baostock %s 逐日证据完成: %d 只，恢复 %d 行",
        BAOSTOCK_EVIDENCE_SCHEMA,
        len(actual_codes),
        sum(restored.values()),
    )


def _build_runtime():
    from data.build_runtime import ListingDateAlignmentError, build_runtime
    from data.kline_history_repairs import ensure_verified_history_repairs
    from data.kline_mootdx import (
        find_placeholder_candidate_codes,
        reconcile_baostock_placeholder_bars,
        update_baostock_listing_evidence,
    )

    kline_dir = DATA_DIR / "k-line"
    evidence_path = (
        DATA_DIR / "kline_evidence" / "baostock_daily_reference.parquet"
    )

    _update_secondary_kline_evidence()

    # Clean the finite, independently audited source defects before generic
    # daily-evidence reconciliation.  This removes pre-listing pollution and
    # seals known truncated rows, so the generic source-coverage guard never
    # has to weaken itself to accommodate bad local input.
    repaired_codes = ensure_verified_history_repairs(kline_dir=kline_dir)
    if repaired_codes:
        logger.info(
            "[Runtime] 已应用 %d 只可审计早期历史修复: %s",
            len(repaired_codes),
            ", ".join(repaired_codes),
        )

    _ensure_baostock_daily_evidence(
        kline_dir=kline_dir,
        evidence_path=evidence_path,
    )

    placeholder_codes = find_placeholder_candidate_codes(
        kline_dir=kline_dir,
        evidence_path=evidence_path,
    )
    if placeholder_codes:
        logger.info(
            "[Runtime] Baostock 封存并归一化 %d 只正价零成交占位候选",
            len(placeholder_codes),
        )
        reconcile_baostock_placeholder_bars(
            list(placeholder_codes),
            kline_dir=kline_dir,
            evidence_path=evidence_path,
        )

    logger.info("[Runtime] 构建全量 npz...")
    t0 = time.time()
    try:
        path = build_runtime()
    except ListingDateAlignmentError as exc:
        conflict_codes = sorted(
            {str(item["code"]) for item in exc.diagnostics}
        )
        logger.warning(
            "[Runtime] %d 只上市事件冲突，封存 Baostock listing/first-executable 独立证据后重试",
            len(conflict_codes),
        )
        update_baostock_listing_evidence(
            conflict_codes,
            kline_dir=kline_dir,
            evidence_path=evidence_path,
        )
        path = build_runtime()
    elapsed = time.time() - t0
    logger.info("[Runtime] 完成: %s (%.0fs)", path, elapsed)


# ============================================================
# 主入口
# ============================================================

def update_offline_toNow():
    """完整下载并原子发布离线快照，然后构建 runtime npz。"""
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("全量数据更新: %s → %s", YESTERDAY, TODAY)
    logger.info("=" * 60)

    # Phase 1: 全量下载。每一步均在完整校验后原子替换正式快照；
    # 下载失败时上一份数据保持字节不变。
    logger.info("--- Phase 1: 下载更新 ---")
    steps = [
        ("退市列表", _update_delist),
        ("股票列表", _update_stock_list),
        ("股票名称/ST", _update_stock_name),
        ("早期逐日K线证据", _update_secondary_kline_evidence),
        ("发行价/上市日", _update_issue_price),
        ("K线日线", _update_kline),
        ("深历史财务", _update_financial_deep),
        ("官方股本校验", _validate_official_balance),
        ("大盘指数", _update_indices),
        ("交易日历", _update_trading_calendar),
    ]

    for name, func in steps:
        t1 = time.time()
        logger.info(">>> %s <<<", name)
        func()
        logger.info("<<< %s 完成 (%.0fs) >>>", name, time.time() - t1)

    # Phase 2: 构建 Runtime
    logger.info("--- Phase 2: 构建 Runtime ---")
    _build_runtime()

    logger.info("=" * 60)
    logger.info("全量更新完成! 总耗时 %.0fs", time.time() - t0)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    update_offline_toNow()
