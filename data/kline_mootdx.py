"""mootdx 日线下载 —— WBR 唯一原始 OHLCVA 数据源，产出不复权 parquet：

    data/k-line/{code}.parquet  列: time/open/high/low/close/volume/amount/preClose

mootdx 只取 fq=0 的不复权真实价 OHLCVA；preClose 常规由 mootdx xdxr() 除权除息数据
自行计算。退市 live 源失效时，仅从已归档 mootdx OHLCVA 恢复，并以 Baostock direct
preClose、交易状态和生命周期逐日核验；旧备份 preClose 永不复用。
转配股/缩股等非标准事件日期（约 0.6%）preClose=NaN，legality 模块自动跳过不交易。

复权序列由 build_runtime 用 `r = close/preClose - 1` 连乘自建（等比后复权，数学上恒正）。

用法:
    python data/kline_mootdx.py                      # 全量拉取到今天
    python data/kline_mootdx.py --codes 000001.SZ    # 指定代码
    python data/kline_mootdx.py --recent 3           # 只刷新最近 N 个交易日
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import socket
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "k-line"
RAW_BACKUP_DIR = DATA_DIR / "k-line-mootdx-bak"
BAOSTOCK_EVIDENCE_PATH = (
    DATA_DIR / "kline_evidence" / "baostock_daily_reference.parquet"
)
BAOSTOCK_EVIDENCE_SCHEMA = "baostock-daily-reference-v4"
BAOSTOCK_EVIDENCE_MANIFEST_SCHEMA = "baostock-daily-reference-manifest-v2"
BAOSTOCK_EVIDENCE_COLUMNS = (
    "stock_code",
    "baostock_code",
    "instrument_type",
    "instrument_status",
    "date",
    "listing_date",
    "out_date",
    "first_executable_date",
    "last_executable_date",
    "source_tradestatus",
    "tradestatus",
    "status_correction",
    "reference_open",
    "reference_volume",
    "reference_amount",
    "direct_preclose",
    "source_payload_sha256",
    "normalization_applied",
    "applied_k_sha256",
    "source",
    "schema_version",
)

# Baostock incorrectly emits 578 synthetic, unchanged-price daily rows while
# the former 000508 entity was officially suspended continuously after its
# 1997-02-28 final session and before its 1999-07-12 termination.  This is a
# finite source-status correction, not a volume-based suspension heuristic:
# the exact upstream date set and every upstream source value are pinned.
BAOSTOCK_STATUS_CORRECTION_ID = (
    "official-continuous-suspension-000508-19970303-19990709-v1"
)
BAOSTOCK_STATUS_CORRECTIONS = MappingProxyType(
    {
        "000508.SZ": MappingProxyType(
            {
                "start": date(1997, 3, 3),
                "end": date(1999, 7, 9),
                "row_count": 578,
                "date_set_sha256": (
                    "581a4c0596b8c019e041b1140787f6c1"
                    "f5c4105c2d1f6d265eab6f7a7b74f87c"
                ),
                "source_rows_sha256": (
                    "fd68ed182b7473b6cdadff74a7144686"
                    "4b0ddf6bf74ec5a85201081c74194688"
                ),
                "last_executable_boundary": date(1997, 2, 28),
                "termination_boundary": date(1999, 7, 12),
                "source_status": "1",
                "effective_status": "0",
                "correction_id": BAOSTOCK_STATUS_CORRECTION_ID,
            }
        )
    }
)

PER_CODE_RETRIES = 3
RAW_BAR_COLUMNS = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)

# 除权除息类别：1=除权除息 5=股本变化(不调价)。其他(9转配股/15等)标记为不可计算
XD_CAT_STANDARD = 1       # 除权除息 — 可用公式
XDXR_HEALTH_SYMBOL = "600519"
XDXR_REQUIRED_COLUMNS = frozenset(
    {
        "year",
        "month",
        "day",
        "category",
        "fenhong",
        "songzhuangu",
        "peigu",
        "peigujia",
    }
)

START_DEFAULT = '19901219'

PAGE_SIZE = 800            # 单次 API 最大返回条数
MAX_HISTORY_BARS = 10000   # 全量拉取上限（覆盖 1990-至今约 8500 交易日）
RECENT_BARS = 400          # 增量拉取条数上限
TDX_SERVERS = (
    ("119.97.185.59", 7709),
    ("124.70.133.119", 7709),
    ("116.205.183.150", 7709),
    ("123.60.73.44", 7709),
    ("116.205.163.254", 7709),
    ("121.36.225.169", 7709),
    ("123.60.70.228", 7709),
    ("124.71.9.153", 7709),
    ("110.41.147.114", 7709),
    ("124.71.187.122", 7709),
)


class KlineSourceError(RuntimeError):
    """Observable mootdx failure without guessing whether it is permanent.

    ``kind`` describes what the source actually returned.  In particular,
    ``empty_response`` does not mean that a delisted security never traded;
    it means only that the live endpoint no longer supplied the requested
    history at this invocation.
    """

    def __init__(
        self,
        code: str,
        operation: str,
        kind: str,
        detail: str,
        *,
        attempts: int = 1,
    ) -> None:
        self.code = code
        self.operation = operation
        self.kind = kind
        self.detail = detail
        self.attempts = attempts
        super().__init__(
            f"{code} {operation} [{kind}] {detail} (attempts={attempts})"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "operation": self.operation,
            "kind": self.kind,
            "detail": self.detail,
            "attempts": self.attempts,
        }


class KlineBatchError(RuntimeError):
    """A strict batch completed with explicitly classified failures."""

    def __init__(
        self,
        stage: str,
        failures: list[tuple[str, Exception]],
        *,
        succeeded: tuple[str, ...] = (),
    ) -> None:
        self.stage = stage
        self.failures = tuple(failures)
        self.succeeded = tuple(succeeded)
        kinds: dict[str, int] = {}
        for _code, exc in failures:
            kind = exc.kind if isinstance(exc, KlineSourceError) else "local_validation"
            kinds[kind] = kinds.get(kind, 0) + 1
        summary = ", ".join(f"{kind}={count}" for kind, count in sorted(kinds.items()))
        super().__init__(f"{stage}不完整: {len(failures)} 只 ({summary})")


def _log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _warn_failed_codes(stage: str, failures: list[tuple[str, Exception]]):
    if not failures:
        return
    details = '; '.join(
        f"{code}[{exc.kind if isinstance(exc, KlineSourceError) else 'local_validation'}]: {exc}"
        for code, exc in failures[:20]
    )
    suffix = f' ...(+{len(failures) - 20})' if len(failures) > 20 else ''
    message = f'[K线] {stage} 跳过 {len(failures)} 只失败股票: {details}{suffix}'
    _log(f'WARNING {message}')


def _fetch_raw_with_retry(
    mdx,
    code: str,
    xdxr_df=None,
    *,
    start: str | None = None,
    end: str | None = None,
    existing: pd.DataFrame | None = None,
):
    last_exc: KlineSourceError | None = None
    for attempt in range(1, PER_CODE_RETRIES + 1):
        try:
            bars = _fetch_bars_all(mdx, code)
            if bars is None:
                raise KlineSourceError(
                    code,
                    "bars",
                    "empty_response",
                    "mootdx 返回空 K 线；退市后源端不再保留代码历史是常见原因",
                    attempts=attempt,
                )
            raw = _mootdx_bars_to_df(bars, xdxr_df=xdxr_df)
            if raw is None:
                raise KlineSourceError(
                    code,
                    "bars",
                    "unusable_payload",
                    "K 线响应没有有限正价格行",
                    attempts=attempt,
                )
            return _validate_full_download_range(
                code,
                raw,
                start=start,
                end=end,
                existing=existing,
            )
        except KlineSourceError as exc:
            last_exc = exc
        except Exception as exc:
            last_exc = KlineSourceError(
                code,
                "bars",
                "request_error",
                f"{type(exc).__name__}: {exc}",
                attempts=attempt,
            )
    assert last_exc is not None
    raise KlineSourceError(
        code,
        last_exc.operation,
        last_exc.kind,
        last_exc.detail,
        attempts=PER_CODE_RETRIES,
    ) from last_exc


def _fetch_xdxr_with_retry(
    mdx,
    code: str,
    *,
    require_nonempty: bool = False,
):
    last_exc: KlineSourceError | None = None
    for attempt in range(1, PER_CODE_RETRIES + 1):
        try:
            frame = mdx.xdxr(symbol=code[:6])
            if frame is None or not isinstance(frame, pd.DataFrame):
                raise KlineSourceError(
                    code,
                    "xdxr",
                    "invalid_response",
                    "xdxr 未返回 DataFrame",
                    attempts=attempt,
                )
            # Empty is valid for an ordinary live symbol with no corporate
            # actions.  It is not sufficient when restoring a delisted
            # history: after delisting the endpoint commonly drops old xdxr,
            # so empty cannot prove that the security never had an action.
            if require_nonempty and frame.empty:
                raise KlineSourceError(
                    code,
                    "xdxr",
                    "empty_response",
                    "退市历史恢复要求非空 xdxr；空响应无法区分无公司行动与源端已清除历史",
                    attempts=attempt,
                )
            if not frame.empty:
                missing = XDXR_REQUIRED_COLUMNS.difference(frame.columns)
                if missing:
                    raise KlineSourceError(
                        code,
                        "xdxr",
                        "schema_mismatch",
                        f"xdxr missing columns: {sorted(missing)}",
                        attempts=attempt,
                    )
            return frame
        except KlineSourceError as exc:
            last_exc = exc
        except Exception as exc:
            last_exc = KlineSourceError(
                code,
                "xdxr",
                "request_error",
                f"{type(exc).__name__}: {exc}",
                attempts=attempt,
            )
    assert last_exc is not None
    raise KlineSourceError(
        code,
        last_exc.operation,
        last_exc.kind,
        last_exc.detail,
        attempts=PER_CODE_RETRIES,
    ) from last_exc


def _raw_backup_to_production(
    frame: pd.DataFrame,
    xdxr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build production bars from backup OHLCVA and freshly fetched xdxr.

    Any backup ``preClose`` column is intentionally ignored: it is not part of
    the raw-bar contract and may come from a stale or incompatible source.
    """
    missing = set(RAW_BAR_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"mootdx 原始备份缺少列: {sorted(missing)}")
    if frame.empty:
        raise ValueError("mootdx 原始备份不得为空")

    raw = frame.loc[:, list(RAW_BAR_COLUMNS)].copy()
    for column in RAW_BAR_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    values = raw.loc[:, list(RAW_BAR_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("mootdx 原始备份 OHLCVA/time 包含非有限值")
    if raw["time"].duplicated().any():
        raise ValueError("mootdx 原始备份 time 重复")
    if ((raw[["open", "high", "low", "close"]] <= 0.0).any()).any():
        raise ValueError("mootdx 原始备份包含非正价格")
    if ((raw[["volume", "amount"]] < 0.0).any()).any():
        raise ValueError("mootdx 原始备份包含负成交量或成交额")

    raw = raw.sort_values("time", kind="stable").reset_index(drop=True)
    raw["time"] = raw["time"].astype(np.int64)
    raw["preClose"] = _compute_preclose(
        raw["close"].to_numpy(dtype=np.float64),
        raw["time"].to_numpy(dtype=np.int64),
        xdxr_df,
    )
    return raw.loc[:, [*RAW_BAR_COLUMNS, "preClose"]]


def _write_parquet_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp.parquet"
    )
    try:
        frame.to_parquet(temp_path, index=False)
        written = pd.read_parquet(temp_path)
        if list(written.columns) != [*RAW_BAR_COLUMNS, "preClose"] or written.empty:
            raise ValueError("mootdx 迁移产物 schema/内容校验失败")
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _kline_content_sha256(frame: pd.DataFrame) -> str:
    """Hash the canonical K-line values, independent of parquet encoding."""
    columns = [*RAW_BAR_COLUMNS, "preClose"]
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"K线 hash 缺少列: {sorted(missing)}")
    canonical = frame.loc[:, columns].copy()
    canonical["time"] = pd.to_numeric(canonical["time"], errors="raise")
    if canonical["time"].isna().any():
        raise ValueError("K线 hash time 不得为空")
    times = canonical["time"].to_numpy(dtype="<i8")
    if len(np.unique(times)) != len(times):
        raise ValueError("K线 hash time 重复")
    order = np.argsort(times, kind="stable")
    values = canonical.loc[:, columns[1:]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype="<f8")[order]
    if np.isinf(values).any():
        raise ValueError("K线 hash 数值包含无穷")
    # Canonicalize NaN payloads and signed zero so a parquet round-trip keeps
    # the same digest.
    values[np.isnan(values)] = np.nan
    values[values == 0.0] = 0.0
    digest = hashlib.sha256()
    digest.update(times[order].tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


UNAPPLIED_K_SHA256 = hashlib.sha256(b"baostock-reference-not-applied").hexdigest()
SECONDARY_COVERAGE_CODES = frozenset(
    {"000004.SZ", "000012.SZ", "001872.SZ", "600018.SH"}
)
SECONDARY_EXACT_DATE_LIMITS = {
    "000004.SZ": frozenset(
        date.fromisoformat(value)
        for value in (
            "1991-01-19",
            "1991-02-02",
            "1991-02-09",
            "1991-02-23",
            "1991-03-16",
            "1991-03-23",
            "1991-03-30",
            "1991-05-11",
            "1991-05-18",
            "1991-05-25",
            "1991-06-08",
            "1991-06-15",
            "1991-06-29",
            "1991-07-06",
            "1991-07-13",
            "1991-07-20",
            "1991-07-27",
            "1991-08-03",
            "1991-08-10",
            "1991-08-17",
            "1991-08-24",
            "1991-08-31",
            "1991-09-07",
            "1991-09-14",
            "1991-09-21",
            "1991-09-28",
            "1991-09-29",
            "1991-10-05",
            "1991-10-12",
            "1991-10-19",
            "1991-10-26",
            "1991-11-02",
            "1991-11-09",
            "1991-11-16",
            "1991-11-23",
            "1991-11-30",
            "1991-12-07",
            "1991-12-14",
            "1991-12-21",
            "1991-12-28",
            "1992-05-03",
            "1992-10-04",
            "1993-01-03",
        )
    ),
    "000012.SZ": frozenset(
        date.fromisoformat(value)
        for value in (
            "1992-05-03",
            "1992-10-04",
            "1993-01-03",
            "1993-06-05",
            "1993-06-19",
            "1993-07-03",
            "1993-07-17",
            "1993-08-07",
            "1993-08-21",
        )
    ),
    "001872.SZ": frozenset({date(2012, 9, 10)}),
}
SECONDARY_DATE_SET_LIMITS = {
    "600018.SH": {
        "count": 1448,
        "first": date(2000, 7, 19),
        "last": date(2006, 9, 25),
        "sha256": (
            "84e618e5115678c41b208c6766b14bc9"
            "be0424131eb086d256fddac6b5a7941f"
        ),
        "required": frozenset({date(2001, 8, 16)}),
    }
}

BAOSTOCK_SOURCE_PAYLOAD_FIELDS = (
    "stock_code",
    "baostock_code",
    "instrument_type",
    "instrument_status",
    "date",
    "listing_date",
    "out_date",
    "first_executable_date",
    "last_executable_date",
    "source_tradestatus",
    "tradestatus",
    "status_correction",
    "reference_open",
    "reference_volume",
    "reference_amount",
    "direct_preclose",
    "source",
    "schema_version",
)

_BAOSTOCK_DATE_FIELDS = frozenset(
    {
        "date",
        "listing_date",
        "out_date",
        "first_executable_date",
        "last_executable_date",
    }
)
_BAOSTOCK_NUMERIC_FIELDS = frozenset(
    {
        "reference_open",
        "reference_volume",
        "reference_amount",
        "direct_preclose",
    }
)


def _canonical_date_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _canonical_numeric_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Baostock evidence canonical numeric 不得为无穷")
    if number == 0.0:
        number = 0.0
    return format(number, ".17g")


def _canonical_bool_text(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    text = "" if value is None or pd.isna(value) else str(value).strip().lower()
    if text not in {"true", "false"}:
        raise ValueError("Baostock evidence canonical bool 非法")
    return text


def _status_correction_date_set_sha256(values) -> str:
    normalized = sorted({pd.Timestamp(value).strftime("%Y-%m-%d") for value in values})
    return hashlib.sha256("\n".join(normalized).encode("ascii")).hexdigest()


def _status_correction_source_rows_sha256(frame: pd.DataFrame) -> str:
    """Hash the exact raw Baostock payload covered by a status correction."""

    field_map = {
        "date": "date",
        "source_tradestatus": "source_tradestatus",
        "reference_open": (
            "reference_open" if "reference_open" in frame.columns else "open"
        ),
        "direct_preclose": (
            "direct_preclose" if "direct_preclose" in frame.columns else "preclose"
        ),
        "reference_volume": (
            "reference_volume" if "reference_volume" in frame.columns else "volume"
        ),
        "reference_amount": (
            "reference_amount" if "reference_amount" in frame.columns else "amount"
        ),
    }
    missing = set(field_map.values()).difference(frame.columns)
    if missing:
        raise ValueError(f"Baostock status correction 源字段缺失: {sorted(missing)}")
    lines: list[str] = []
    for _index, row in frame.sort_values("date", kind="stable").iterrows():
        payload = {
            "date": _canonical_date_text(row[field_map["date"]]),
            "tradestatus": _canonical_evidence_value(
                "source_tradestatus", row[field_map["source_tradestatus"]]
            ),
            "reference_open": _canonical_numeric_text(
                row[field_map["reference_open"]]
            ),
            "direct_preclose": _canonical_numeric_text(
                row[field_map["direct_preclose"]]
            ),
            "reference_volume": _canonical_numeric_text(
                row[field_map["reference_volume"]]
            ),
            "reference_amount": _canonical_numeric_text(
                row[field_map["reference_amount"]]
            ),
        }
        lines.append(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _apply_baostock_status_corrections(
    code: str,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Return source-preserving effective status for one complete history.

    A correction is accepted only when the entire code-owned raw payload is
    byte-semantically identical to the reviewed source snapshot.  Missing or
    changed rows reject the whole code before evidence or K-line publication.
    """

    result = history.copy()
    if "source_tradestatus" not in result.columns:
        result["source_tradestatus"] = (
            result["tradestatus"].fillna("").astype(str).str.strip()
        )
    else:
        result["source_tradestatus"] = (
            result["source_tradestatus"].fillna("").astype(str).str.strip()
        )
    result["tradestatus"] = (
        result["tradestatus"].fillna("").astype(str).str.strip()
    )
    if "status_correction" not in result.columns:
        result["status_correction"] = ""
    else:
        result["status_correction"] = (
            result["status_correction"].fillna("").astype(str).str.strip()
        )

    contract = BAOSTOCK_STATUS_CORRECTIONS.get(code)
    if contract is None:
        if not result["status_correction"].eq("").all():
            raise ValueError(f"{code} 包含未声明 Baostock status correction")
        if not result["tradestatus"].eq(result["source_tradestatus"]).all():
            raise ValueError(f"{code} effective/source tradestatus 无声明差异")
        return result

    dates = pd.to_datetime(result["date"], errors="raise").dt.date
    target = dates.between(contract["start"], contract["end"])
    rows = result.loc[target].copy()
    actual_dates = tuple(pd.to_datetime(rows["date"], errors="raise").dt.date)
    if (
        len(rows) != contract["row_count"]
        or _status_correction_date_set_sha256(actual_dates)
        != contract["date_set_sha256"]
    ):
        raise ValueError(
            f"{code} Baostock status correction 日期集合不完整或变化"
        )
    if not rows["source_tradestatus"].eq(contract["source_status"]).all():
        raise ValueError(f"{code} Baostock status correction 原始状态发生变化")
    if (
        _status_correction_source_rows_sha256(rows)
        != contract["source_rows_sha256"]
    ):
        raise ValueError(f"{code} Baostock status correction 原始 payload 发生变化")

    previous = dates.eq(contract["last_executable_boundary"])
    termination = dates.eq(contract["termination_boundary"])
    if int(previous.sum()) != 1 or int(termination.sum()) != 1:
        raise ValueError(f"{code} Baostock status correction 生命周期边界缺失")
    if not (
        result.loc[previous, "source_tradestatus"].eq("1").all()
        and result.loc[termination, "source_tradestatus"].eq("0").all()
    ):
        raise ValueError(f"{code} Baostock status correction 生命周期边界冲突")

    # Reset from the immutable source values first, making repeat application
    # deterministic and idempotent.
    result["tradestatus"] = result["source_tradestatus"]
    result["status_correction"] = ""
    result.loc[target, "tradestatus"] = contract["effective_status"]
    result.loc[target, "status_correction"] = contract["correction_id"]
    return result


def _canonical_evidence_value(field: str, value: object) -> str:
    if field in _BAOSTOCK_DATE_FIELDS:
        return _canonical_date_text(value)
    if field in _BAOSTOCK_NUMERIC_FIELDS:
        return _canonical_numeric_text(value)
    if field == "normalization_applied":
        return _canonical_bool_text(value)
    return "" if value is None or pd.isna(value) else str(value).strip()


def _source_payload_sha256(row: pd.Series | dict[str, object]) -> str:
    payload = {
        field: _canonical_evidence_value(field, row[field])
        for field in BAOSTOCK_SOURCE_PAYLOAD_FIELDS
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _baostock_snapshot_sha256(frame: pd.DataFrame) -> str:
    missing = set(BAOSTOCK_EVIDENCE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Baostock evidence snapshot hash 缺少列: {sorted(missing)}")
    ordered = frame.sort_values(["stock_code", "date"], kind="stable")
    digest = hashlib.sha256()
    for _index, row in ordered.iterrows():
        payload = {
            field: _canonical_evidence_value(field, row[field])
            for field in BAOSTOCK_EVIDENCE_COLUMNS
        }
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _baostock_evidence_manifest_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}.manifest.json")


def _baostock_manifest_payload(frame: pd.DataFrame) -> dict[str, object]:
    dates = pd.to_datetime(frame["date"], errors="raise")
    payload: dict[str, object] = {
        "manifest_schema": BAOSTOCK_EVIDENCE_MANIFEST_SCHEMA,
        "evidence_schema": BAOSTOCK_EVIDENCE_SCHEMA,
        "row_count": int(len(frame)),
        "stock_count": int(frame["stock_code"].astype(str).nunique()),
        "first_date": dates.min().strftime("%Y-%m-%d"),
        "last_date": dates.max().strftime("%Y-%m-%d"),
        "snapshot_sha256": _baostock_snapshot_sha256(frame),
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _load_baostock_manifest(path: Path, frame: pd.DataFrame) -> None:
    manifest_path = _baostock_evidence_manifest_path(path)
    if not manifest_path.exists():
        raise ValueError(f"Baostock daily evidence manifest 缺失: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Baostock daily evidence manifest 无法解析") from exc
    expected = _baostock_manifest_payload(frame)
    if manifest != expected:
        raise ValueError(
            "Baostock daily evidence snapshot manifest 校验失败: "
            f"expected={expected} actual={manifest}"
        )


def _canonicalize_baostock_evidence_storage(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(BAOSTOCK_EVIDENCE_COLUMNS).difference(frame.columns)
    if missing or frame.empty:
        raise ValueError(
            f"Baostock daily evidence storage schema/内容不完整: {sorted(missing)}"
        )
    storage = frame.loc[:, BAOSTOCK_EVIDENCE_COLUMNS].copy()
    for field in _BAOSTOCK_DATE_FIELDS:
        storage[field] = storage[field].map(_canonical_date_text)
    for field in _BAOSTOCK_NUMERIC_FIELDS:
        storage[field] = pd.to_numeric(storage[field], errors="coerce").astype(
            np.float64
        )
    storage["stock_code"] = (
        storage["stock_code"].astype(str).str.strip().str.upper()
    )
    storage["baostock_code"] = (
        storage["baostock_code"].astype(str).str.strip().str.lower()
    )
    for field in (
        "instrument_type",
        "instrument_status",
        "source_tradestatus",
        "tradestatus",
        "status_correction",
        "source",
        "schema_version",
    ):
        storage[field] = storage[field].fillna("").astype(str).str.strip()
    storage["normalization_applied"] = storage[
        "normalization_applied"
    ].map(lambda value: _canonical_bool_text(value) == "true")
    for field in ("source_payload_sha256", "applied_k_sha256"):
        storage[field] = (
            storage[field].fillna("").astype(str).str.strip().str.lower()
        )
    expected_source_hashes = storage.apply(_source_payload_sha256, axis=1)
    if not storage["source_payload_sha256"].eq(expected_source_hashes).all():
        raise ValueError(
            "Baostock daily evidence source_payload_sha256 校验失败"
        )
    return storage


def _write_baostock_evidence_stage(frame: pd.DataFrame, staged_path: Path) -> Path:
    """Write and verify one unpublished parquet/manifest pair."""
    staged_path = Path(staged_path)
    manifest_path = _baostock_evidence_manifest_path(staged_path)
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        storage = _canonicalize_baostock_evidence_storage(frame)
        storage.to_parquet(staged_path, index=False)
        written = pd.read_parquet(staged_path)
        if (
            list(written.columns) != list(BAOSTOCK_EVIDENCE_COLUMNS)
            or written.empty
        ):
            raise ValueError("Baostock daily evidence 写后校验失败")
        manifest_path.write_text(
            json.dumps(
                _baostock_manifest_payload(written),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        load_baostock_daily_evidence(staged_path)
    except Exception:
        staged_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return manifest_path


def _replace_staged_files_transactionally(
    pairs: list[tuple[Path, Path]],
    *,
    token: str,
) -> None:
    """Publish several files as one rollback-protected local transaction.

    A normal exception restores every already-replaced target.  If restoring
    any target itself fails, all rollback copies are deliberately retained so
    an operator can recover the exact pre-transaction bytes.
    """
    normalized = [(Path(stage), Path(target)) for stage, target in pairs]
    if not normalized or len({target for _stage, target in normalized}) != len(
        normalized
    ):
        raise ValueError("文件事务目标为空或重复")
    missing = [stage for stage, _target in normalized if not stage.exists()]
    if missing:
        raise FileNotFoundError(f"文件事务缺少 staged 文件: {missing}")

    backups: dict[Path, Path | None] = {}
    published: list[Path] = []
    rollback_complete = False
    committed = False
    try:
        for _stage, target in normalized:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_name(f".{target.name}.{token}.rollback")
                shutil.copy2(target, backup)
                backups[target] = backup
            else:
                backups[target] = None
        try:
            for stage, target in normalized:
                try:
                    stage.replace(target)
                except BaseException:
                    if not stage.exists() and target.exists():
                        published.append(target)
                    raise
                published.append(target)
            committed = True
        except BaseException as publish_exc:
            rollback_errors: list[tuple[Path, BaseException]] = []
            for target in reversed(published):
                backup = backups[target]
                try:
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        shutil.copy2(backup, target)
                except BaseException as exc:
                    rollback_errors.append((target, exc))
            rollback_complete = not rollback_errors
            if rollback_errors:
                retained = [
                    str(path) for path in backups.values() if path is not None
                ]
                details = "; ".join(
                    f"{target}: {type(exc).__name__}: {exc}"
                    for target, exc in rollback_errors
                )
                raise RuntimeError(
                    "文件事务发布失败且回滚不完整；备份已保留: "
                    f"{retained}; rollback_errors={details}"
                ) from publish_exc
            raise
    finally:
        for stage, _target in normalized:
            stage.unlink(missing_ok=True)
        # No target changed, a full commit, or a completed rollback are the
        # only states in which recovery copies are safe to remove.
        if committed or rollback_complete or not published:
            for backup in backups.values():
                if backup is not None:
                    backup.unlink(missing_ok=True)


def replace_staged_files_transactionally(
    pairs: list[tuple[Path, Path]],
    *,
    token: str,
) -> None:
    """Public local-file transaction used by finite offline-data publishers."""

    _replace_staged_files_transactionally(pairs, token=token)


def capture_file_states(paths) -> dict[Path, str | None]:
    """Capture content identities for a later pre-publication CAS check."""

    states: dict[Path, str | None] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if path in states:
            continue
        if not path.exists():
            states[path] = None
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        states[path] = digest.hexdigest()
    return states


def assert_file_states_unchanged(
    expected: dict[Path, str | None],
    *,
    label: str,
) -> None:
    """Reject publication if any protected input changed after it was read."""

    current = capture_file_states(expected)
    changed = sorted(str(path) for path in expected if current[path] != expected[path])
    if changed:
        raise RuntimeError(f"{label} 在构建期间发生并发变化: {changed}")


def _kline_row_state_sha256(day: date, row: pd.Series) -> str:
    """Hash only missing/zero/positive state, never an exact market value."""
    states: list[str] = []
    for field in ("open", "high", "low", "close", "volume", "amount", "preClose"):
        value = float(pd.to_numeric(pd.Series([row[field]]), errors="coerce").iloc[0])
        if np.isnan(value):
            state = "nan"
        elif np.isposinf(value) or np.isneginf(value):
            state = "inf"
        elif value > 0.0:
            state = "positive"
        elif value == 0.0:
            state = "zero"
        else:
            state = "negative"
        states.append(f"{field}={state}")
    payload = f"{day.isoformat()}|" + "|".join(states)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _bind_evidence_to_applied_dates(
    evidence: pd.DataFrame,
    frame: pd.DataFrame,
    applied_dates: set[date],
) -> pd.DataFrame:
    bound = evidence.copy()
    bound["normalization_applied"] = False
    bound["applied_k_sha256"] = UNAPPLIED_K_SHA256
    if not applied_dates:
        return bound
    k_dates = pd.to_datetime(
        pd.to_numeric(frame["time"], errors="raise"), unit="ms"
    ).dt.date
    if k_dates.duplicated().any():
        raise ValueError("K线 application hash 日期重复")
    k_by_date = frame.copy().assign(_date=k_dates).set_index("_date", drop=True)
    evidence_dates = pd.to_datetime(bound["date"], errors="raise").dt.date
    for day in sorted(applied_dates):
        if day not in k_by_date.index:
            raise ValueError(f"application hash 缺少 K 线日期: {day}")
        matches = evidence_dates.eq(day)
        if int(matches.sum()) != 1:
            raise ValueError(f"application hash evidence 日期不唯一: {day}")
        bound.loc[matches, "normalization_applied"] = True
        bound.loc[matches, "applied_k_sha256"] = _kline_row_state_sha256(
            day, k_by_date.loc[day]
        )
    return bound


def _validate_evidence_application_hashes(
    frame: pd.DataFrame,
    evidence: pd.DataFrame,
) -> None:
    applied = evidence[evidence["normalization_applied"].astype(bool)]
    if applied.empty:
        return
    k_dates = pd.to_datetime(
        pd.to_numeric(frame["time"], errors="raise"), unit="ms"
    ).dt.date
    if k_dates.duplicated().any():
        raise ValueError("K线 application hash 日期重复")
    k_by_date = frame.copy().assign(_date=k_dates).set_index("_date", drop=True)
    for row in applied.itertuples(index=False):
        day = pd.Timestamp(row.date).date()
        if day not in k_by_date.index:
            raise ValueError(f"applied evidence 缺少 K 线日期: {day}")
        actual = _kline_row_state_sha256(day, k_by_date.loc[day])
        if actual != row.applied_k_sha256:
            raise ValueError(f"applied evidence/K 状态 hash 不一致: {day}")


def _baostock_symbol(code: str) -> str:
    bare, suffix = code.split(".", maxsplit=1) if "." in code else ("", "")
    if not bare.isdigit() or len(bare) != 6 or suffix not in {"SH", "SZ"}:
        raise ValueError(f"Baostock 仅接受规范 SH/SZ A股代码: {code!r}")
    return f"{suffix.lower()}.{bare}"


def _is_baostock_session_error(error_code: str, error_msg: str) -> bool:
    text = f"{error_code} {error_msg}".lower()
    return error_code in {"10001001", "10001002"} or any(
        token in text for token in ("未登录", "登录失效", "session", "login")
    )


class _BaostockSession:
    """Small fail-closed adapter with one bounded re-login."""

    def __init__(self, module) -> None:
        self.module = module
        self._login()

    def _login(self) -> None:
        result = self.module.login()
        code = str(getattr(result, "error_code", ""))
        message = str(getattr(result, "error_msg", ""))
        if code != "0":
            raise KlineSourceError(
                "*",
                "baostock_login",
                "authentication_error",
                f"{code}: {message}",
            )

    def query(self, code: str, operation: str, factory) -> pd.DataFrame:
        # A long batch may lose the server-side session repeatedly.  The
        # retry budget is deliberately per query, not global for the login,
        # so each failed request gets one (and only one) fresh session.
        relogged = False
        for attempt in range(1, 3):
            try:
                result = factory()
            except Exception as exc:
                raise KlineSourceError(
                    code,
                    operation,
                    "request_error",
                    f"{type(exc).__name__}: {exc}",
                    attempts=attempt,
                ) from exc
            error_code = str(getattr(result, "error_code", ""))
            error_msg = str(getattr(result, "error_msg", ""))
            if error_code != "0":
                if (
                    attempt == 1
                    and not relogged
                    and _is_baostock_session_error(error_code, error_msg)
                ):
                    relogged = True
                    try:
                        self.module.logout()
                    except Exception:
                        pass
                    self._login()
                    continue
                kind = (
                    "session_expired"
                    if _is_baostock_session_error(error_code, error_msg)
                    else "request_error"
                )
                raise KlineSourceError(
                    code,
                    operation,
                    kind,
                    f"{error_code}: {error_msg}",
                    attempts=attempt,
                )
            fields = tuple(str(field) for field in getattr(result, "fields", ()))
            rows: list[list[str]] = []
            while result.next():
                rows.append(result.get_row_data())
            final_code = str(getattr(result, "error_code", ""))
            final_msg = str(getattr(result, "error_msg", ""))
            if final_code != "0":
                if (
                    attempt == 1
                    and not relogged
                    and _is_baostock_session_error(final_code, final_msg)
                ):
                    relogged = True
                    try:
                        self.module.logout()
                    except Exception:
                        pass
                    self._login()
                    continue
                raise KlineSourceError(
                    code,
                    operation,
                    "session_expired" if _is_baostock_session_error(final_code, final_msg) else "request_error",
                    f"{final_code}: {final_msg}",
                    attempts=attempt,
                )
            return pd.DataFrame(rows, columns=fields)
        raise AssertionError("bounded Baostock query loop did not return")

    def close(self) -> None:
        try:
            result = self.module.logout()
        except Exception as exc:
            _log(f"WARNING [Baostock] logout failed: {type(exc).__name__}: {exc}")
            return
        error_code = str(getattr(result, "error_code", "0"))
        if error_code not in {"", "0"}:
            _log(f"WARNING [Baostock] logout failed: {error_code}")


def _baostock_basic_map(
    session: _BaostockSession,
    codes: list[str],
) -> dict[str, pd.Series]:
    frame = session.query("*", "baostock_stock_basic", session.module.query_stock_basic)
    required = {"code", "ipoDate", "outDate", "type", "status"}
    missing = required.difference(frame.columns)
    if missing:
        raise KlineSourceError(
            "*",
            "baostock_stock_basic",
            "schema_mismatch",
            f"missing columns: {sorted(missing)}",
        )
    if frame.empty:
        raise KlineSourceError(
            "*",
            "baostock_stock_basic",
            "empty_response",
            "query_stock_basic 返回空表，不能解释为标的不存在",
        )
    frame = frame.copy()
    frame["code"] = frame["code"].astype(str).str.strip().str.lower()
    if frame["code"].duplicated().any():
        raise KlineSourceError(
            "*", "baostock_stock_basic", "duplicate_rows", "code 重复"
        )
    indexed = frame.set_index("code", drop=False)
    result: dict[str, pd.Series] = {}
    for code in codes:
        symbol = _baostock_symbol(code)
        if symbol not in indexed.index:
            raise KlineSourceError(
                code,
                "baostock_stock_basic",
                "missing_symbol",
                f"全量 basic 未包含 {symbol}",
            )
        row = indexed.loc[symbol]
        if str(row["type"]).strip() != "1":
            raise KlineSourceError(
                code,
                "baostock_stock_basic",
                "instrument_mismatch",
                f"{symbol} type={row['type']!r}，不是股票",
            )
        result[code] = row
    return result


def _baostock_basic_map_isolated(
    session: _BaostockSession,
    codes: list[str],
) -> tuple[dict[str, pd.Series], list[tuple[str, Exception]]]:
    """Resolve one shared basic response while isolating per-symbol defects."""
    frame = session.query("*", "baostock_stock_basic", session.module.query_stock_basic)
    required = {"code", "ipoDate", "outDate", "type", "status"}
    missing = required.difference(frame.columns)
    if missing:
        raise KlineSourceError(
            "*",
            "baostock_stock_basic",
            "schema_mismatch",
            f"missing columns: {sorted(missing)}",
        )
    if frame.empty:
        raise KlineSourceError(
            "*",
            "baostock_stock_basic",
            "empty_response",
            "query_stock_basic 返回空表，不能解释为标的不存在",
        )
    frame = frame.copy()
    frame["code"] = frame["code"].astype(str).str.strip().str.lower()
    if frame["code"].duplicated().any():
        raise KlineSourceError(
            "*", "baostock_stock_basic", "duplicate_rows", "code 重复"
        )
    indexed = frame.set_index("code", drop=False)
    result: dict[str, pd.Series] = {}
    failures: list[tuple[str, Exception]] = []
    for code in codes:
        symbol = _baostock_symbol(code)
        if symbol not in indexed.index:
            failures.append(
                (
                    code,
                    KlineSourceError(
                        code,
                        "baostock_stock_basic",
                        "missing_symbol",
                        f"全量 basic 未包含 {symbol}",
                    ),
                )
            )
            continue
        row = indexed.loc[symbol]
        if str(row["type"]).strip() != "1":
            failures.append(
                (
                    code,
                    KlineSourceError(
                        code,
                        "baostock_stock_basic",
                        "instrument_mismatch",
                        f"{symbol} type={row['type']!r}，不是股票",
                    ),
                )
            )
            continue
        if str(row["status"]).strip() not in {"0", "1"}:
            failures.append(
                (
                    code,
                    KlineSourceError(
                        code,
                        "baostock_stock_basic",
                        "invalid_lifecycle",
                        f"{symbol} status={row['status']!r}，必须是 0/1",
                    ),
                )
            )
            continue
        result[code] = row
    return result, failures


def _baostock_history(
    session: _BaostockSession,
    code: str,
    start_date: date,
    end_date: date,
    *,
    lifecycle_start: date | None = None,
    lifecycle_end: date | None = None,
) -> pd.DataFrame:
    if start_date > end_date:
        raise ValueError(f"Baostock 查询边界倒置: {start_date} > {end_date}")
    active_start = lifecycle_start or start_date
    active_end = lifecycle_end or end_date
    if active_start > active_end:
        raise ValueError(f"Baostock 生命周期边界倒置: {active_start} > {active_end}")
    symbol = _baostock_symbol(code)
    parts: list[pd.DataFrame] = []
    year = start_date.year
    while year <= end_date.year:
        chunk_start = max(start_date, date(year, 1, 1))
        chunk_end = min(end_date, date(year + 4, 12, 31))
        part = session.query(
            code,
            "baostock_history",
            lambda s=chunk_start, e=chunk_end: session.module.query_history_k_data_plus(
                symbol,
                "date,code,open,preclose,volume,amount,tradestatus",
                start_date=s.isoformat(),
                end_date=e.isoformat(),
                frequency="d",
                adjustflag="3",
            ),
        )
        required = {
            "date",
            "code",
            "open",
            "preclose",
            "volume",
            "amount",
            "tradestatus",
        }
        missing = required.difference(part.columns)
        if missing:
            raise KlineSourceError(
                code,
                "baostock_history",
                "schema_mismatch",
                f"{chunk_start}..{chunk_end} missing columns: {sorted(missing)}",
            )
        overlaps_lifecycle = (
            max(chunk_start, active_start) <= min(chunk_end, active_end)
        )
        if part.empty:
            if overlaps_lifecycle:
                raise KlineSourceError(
                    code,
                    "baostock_history",
                    "incomplete_response",
                    f"生命周期内分块 {chunk_start}..{chunk_end} 返回空表",
                )
        else:
            parts.append(part)
        year += 5
    if not parts:
        raise KlineSourceError(
            code,
            "baostock_history",
            "empty_response",
            "完整生命周期查询返回空表，不能解释为无交易历史",
        )
    frame = pd.concat(parts, ignore_index=True)
    required = {
        "date",
        "code",
        "open",
        "preclose",
        "volume",
        "amount",
        "tradestatus",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KlineSourceError(
            code,
            "baostock_history",
            "schema_mismatch",
            f"missing columns: {sorted(missing)}",
        )
    if not frame["code"].astype(str).str.strip().str.lower().eq(symbol).all():
        raise KlineSourceError(
            code,
            "baostock_history",
            "instrument_mismatch",
            f"响应包含非目标代码 {symbol}",
        )
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise").dt.date
    if frame["date"].lt(start_date).any() or frame["date"].gt(end_date).any():
        raise KlineSourceError(
            code,
            "baostock_history",
            "boundary_mismatch",
            f"响应日期越过请求边界 {start_date}..{end_date}",
        )
    if frame["date"].duplicated().any():
        raise KlineSourceError(
            code, "baostock_history", "duplicate_rows", "date 重复"
        )
    for field in ("open", "preclose", "volume", "amount"):
        text = frame[field].fillna("").astype(str).str.strip()
        numeric = pd.to_numeric(text, errors="coerce")
        if (text.ne("") & numeric.isna()).any():
            raise KlineSourceError(
                code,
                "baostock_history",
                "invalid_reference_value",
                f"{field} 包含非空非数值值",
            )
        frame[field] = numeric.astype(np.float64)
    frame["tradestatus"] = frame["tradestatus"].fillna("").astype(str).str.strip()
    if not frame["tradestatus"].isin({"", "0", "1"}).all():
        raise KlineSourceError(
            code,
            "baostock_history",
            "invalid_tradestatus",
            "tradestatus 只能是空串/0/1；空串仅允许由独立开盘价证明可成交",
        )
    numeric = frame[["open", "preclose", "volume", "amount"]].to_numpy(
        dtype=np.float64
    )
    if np.isinf(numeric).any() or (frame[["open", "volume", "amount"]] < 0.0).any().any():
        raise KlineSourceError(
            code,
            "baostock_history",
            "invalid_reference_value",
            "open/volume/amount 存在无穷或负值",
        )
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    return _apply_baostock_status_corrections(code, frame)


def _baostock_executable_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the causal T-open executability observation.

    ``tradestatus=1`` is Baostock's direct daily trading-status evidence;
    ``tradestatus=0`` is a direct suspension observation.  Very old rows can
    have an empty status, in which case only the independent same-day open is
    admissible.  Same-day volume/amount are settlement data and never enter
    this decision.
    """
    status = frame["tradestatus"].fillna("").astype(str).str.strip()
    reference_open = pd.to_numeric(frame["open"], errors="coerce")
    return status.eq("1") | (status.eq("") & reference_open.gt(0.0))


def _build_baostock_evidence(
    code: str,
    basic: pd.Series,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, date, date | None, date, date]:
    """Build source-only lifecycle evidence without requiring a clean local K-line."""
    history = _apply_baostock_status_corrections(code, history)
    ipo_date = pd.Timestamp(str(basic["ipoDate"])).date()
    out_text = str(basic["outDate"]).strip()
    out_date = pd.Timestamp(out_text).date() if out_text else None
    executable = history[_baostock_executable_mask(history)]
    if executable.empty:
        raise KlineSourceError(
            code,
            "baostock_history",
            "empty_response",
            "没有可验证的可成交行",
        )
    first_executable = executable.iloc[0]["date"]
    last_executable = executable.iloc[-1]["date"]

    evidence = history.copy()
    evidence["stock_code"] = code
    evidence["baostock_code"] = _baostock_symbol(code)
    evidence["instrument_type"] = str(basic["type"]).strip()
    evidence["instrument_status"] = str(basic["status"]).strip()
    evidence["listing_date"] = ipo_date.isoformat()
    evidence["out_date"] = out_date.isoformat() if out_date is not None else ""
    evidence["first_executable_date"] = first_executable.isoformat()
    evidence["last_executable_date"] = last_executable.isoformat()
    evidence["source_tradestatus"] = (
        evidence["source_tradestatus"].fillna("").astype(str).str.strip()
    )
    evidence["tradestatus"] = evidence["tradestatus"].fillna("").astype(str).str.strip()
    evidence["status_correction"] = (
        evidence["status_correction"].fillna("").astype(str).str.strip()
    )
    evidence["reference_open"] = evidence["open"].astype(np.float64)
    evidence["reference_volume"] = evidence["volume"].astype(np.float64)
    evidence["reference_amount"] = evidence["amount"].astype(np.float64)
    evidence["direct_preclose"] = evidence["preclose"].astype(np.float64)
    evidence["normalization_applied"] = False
    evidence["applied_k_sha256"] = UNAPPLIED_K_SHA256
    evidence["date"] = evidence["date"].map(date.isoformat)
    evidence["source"] = "baostock.adjustflag3.daily-reference"
    evidence["schema_version"] = BAOSTOCK_EVIDENCE_SCHEMA
    evidence["source_payload_sha256"] = evidence.apply(
        _source_payload_sha256, axis=1
    )
    return (
        evidence.loc[:, BAOSTOCK_EVIDENCE_COLUMNS],
        ipo_date,
        out_date,
        first_executable,
        last_executable,
    )


def _baostock_production_and_evidence(
    code: str,
    backup: pd.DataFrame,
    basic: pd.Series,
    history: pd.DataFrame,
    *,
    require_delisted: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = _apply_baostock_status_corrections(code, history)
    raw_values = backup.loc[:, RAW_BAR_COLUMNS].apply(
        pd.to_numeric, errors="raise"
    )
    if np.isfinite(raw_values.to_numpy(dtype=np.float64)).all():
        production = _raw_backup_to_production(backup, pd.DataFrame())
    else:
        # Targeted reconciliation also accepts an already-normalized local K
        # frame.  Missing OHLC must be all-or-none per row; turnover may be NaN
        # because an earlier normalization deliberately removed a false
        # zero-liquidity signal.  No price is invented on this path.
        production = raw_values.copy()
        if production["time"].isna().any():
            raise ValueError("Baostock reconcile local K time 不得为空")
        prices = production[["open", "high", "low", "close"]]
        missing_prices = prices.isna()
        if (missing_prices.any(axis=1) != missing_prices.all(axis=1)).any():
            raise ValueError("Baostock reconcile local K OHLC 不得部分为空")
        finite_prices = prices[~missing_prices.all(axis=1)].to_numpy(
            dtype=np.float64
        )
        if (
            np.isinf(production.to_numpy(dtype=np.float64)).any()
            or (finite_prices <= 0.0).any()
            or (
                production[["volume", "amount"]]
                .dropna(how="all")
                .lt(0.0)
                .any()
                .any()
            )
        ):
            raise ValueError("Baostock reconcile local K 含非法数值")
        production["time"] = production["time"].astype(np.int64)
        production["preClose"] = np.nan
        production = production.loc[:, [*RAW_BAR_COLUMNS, "preClose"]]
    backup_dates = pd.to_datetime(production["time"], unit="ms").dt.date
    candidate_before = (
        production["open"].gt(0.0)
        & production["volume"].le(0.0)
        & production["amount"].le(0.0)
    ).to_numpy()
    candidate_dates = set(backup_dates[candidate_before])
    reference = history.set_index("date")
    missing_dates = sorted(set(backup_dates).difference(reference.index))
    if missing_dates:
        shown = ", ".join(value.isoformat() for value in missing_dates[:20])
        raise KlineSourceError(
            code,
            "baostock_history",
            "coverage_mismatch",
            f"mootdx backup 有 {len(missing_dates)} 个日期未被 Baostock 覆盖: {shown}",
        )
    aligned = reference.loc[list(backup_dates)]
    direct = aligned["preclose"].to_numpy(dtype=np.float64)
    valid_direct = np.isfinite(direct) & (direct > 0.0)
    executable_aligned = _baostock_executable_mask(aligned).to_numpy()
    require_direct = executable_aligned.copy()
    local_executable_rows = np.flatnonzero(require_direct)
    if local_executable_rows.size:
        # The first executable observation may be an IPO/re-listing row whose
        # direct preclose is legitimately empty.  Later executable rows may not.
        require_direct[local_executable_rows[0]] = False
    if np.any(require_direct & ~valid_direct):
        raise KlineSourceError(
            code,
            "baostock_history",
            "invalid_preclose",
            "除首个可成交日外，backup 可成交日期的 direct preclose 存在空值、非有限值或非正值",
        )

    evidence, ipo_date, out_date, first_executable, last_executable = (
        _build_baostock_evidence(code, basic, history)
    )

    if require_delisted:
        from data.db.delist import get_delist_stock_info

        info = get_delist_stock_info().get(code)
        if info is None:
            raise KlineSourceError(
                code,
                "local_delist_lifecycle",
                "missing_symbol",
                "本地退市快照缺少目标代码",
            )
        # Baostock may expose the exchange exit/effective date while the local
        # snapshot stores the later administrative termination date.  Accept
        # only a narrow, ordered boundary difference for the same delisted
        # symbol; arbitrary lifecycle mismatches still fail closed.
        boundary_gap = (
            (info.delist_date - out_date).days
            if out_date is not None
            else None
        )
        lifecycle_matches = (
            str(basic["status"]).strip() == "0"
            and boundary_gap is not None
            and 0 <= boundary_gap <= 7
            and last_executable <= out_date
        )
        if not lifecycle_matches:
            raise KlineSourceError(
                code,
                "baostock_stock_basic",
                "lifecycle_mismatch",
                "status/outDate/lastExecutable="
                f"{basic['status']!r}/{out_date}/{last_executable} "
                f"与本地退市日 {info.delist_date} 不一致",
            )

    production["preClose"] = direct
    # tradestatus is the primary independent daily suspension observation;
    # old rows with empty status use only reference open.  Runtime legality
    # never infers suspension from same-day volume.  Positive OHLC placeholders
    # on non-executable days must not enter listing_age/scoring/order legality.
    unavailable = ~executable_aligned
    production.loc[unavailable, ["open", "high", "low", "close", "preClose"]] = np.nan
    production.loc[unavailable, ["volume", "amount"]] = 0.0
    # If the independent source confirms execution while the raw archive has
    # zero turnover, retain the verified price row but do not expose a false
    # zero-liquidity signal to amount/volume factors.  Baostock turnover uses a
    # different source/unit contract, so it is evidence only, not a value to
    # copy into the mootdx OHLCVA columns.
    local_zero_turnover = (
        production["volume"].fillna(0.0).le(0.0)
        & production["amount"].fillna(0.0).le(0.0)
    ).to_numpy()
    turnover_mismatch = executable_aligned & local_zero_turnover
    production.loc[turnover_mismatch, ["volume", "amount"]] = np.nan
    production.loc[0, "preClose"] = np.nan

    local_opens = production["open"].to_numpy(dtype=np.float64)
    local_valid = np.flatnonzero(np.isfinite(local_opens) & (local_opens > 0.0))
    if local_valid.size == 0:
        raise KlineSourceError(
            code,
            "baostock_history",
            "empty_response",
            "按独立 tradestatus 清理后 backup 没有可用价格行",
        )
    # Source-wide first/last status remain diagnostics.  Baostock can emit
    # additional primary-only dates that are not real bars, so a missing local
    # row is never inferred from this equality.  Exact secondary evidence is
    # the only authority allowed to require a date absent from local K.

    evidence = _bind_evidence_to_applied_dates(
        evidence,
        production,
        candidate_dates
        | set(
            backup_dates[
                aligned["status_correction"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
                .to_numpy()
            ]
        ),
    )

    return production, evidence


def load_baostock_daily_evidence(
    path: Path = BAOSTOCK_EVIDENCE_PATH,
) -> pd.DataFrame:
    """Load and validate the sealed daily reference; never performs I/O online."""
    if not path.exists():
        return pd.DataFrame(columns=BAOSTOCK_EVIDENCE_COLUMNS)
    raw = pd.read_parquet(path)
    if list(raw.columns) != list(BAOSTOCK_EVIDENCE_COLUMNS):
        raise ValueError("Baostock daily evidence schema/column order 不兼容")
    if raw.empty:
        raise ValueError("Baostock daily evidence 不得为空")
    if not raw["schema_version"].astype(str).eq(BAOSTOCK_EVIDENCE_SCHEMA).all():
        raise ValueError("Baostock daily evidence schema_version 不兼容")
    if not raw["source"].astype(str).eq(
        "baostock.adjustflag3.daily-reference"
    ).all():
        raise ValueError("Baostock daily evidence source 不受信任")

    frame = raw.copy()
    frame["stock_code"] = frame["stock_code"].astype(str).str.strip().str.upper()
    frame["baostock_code"] = frame["baostock_code"].astype(str).str.strip().str.lower()
    frame["instrument_type"] = (
        frame["instrument_type"].fillna("").astype(str).str.strip()
    )
    frame["instrument_status"] = (
        frame["instrument_status"].fillna("").astype(str).str.strip()
    )
    if not frame["instrument_type"].eq("1").all():
        raise ValueError("Baostock daily evidence instrument_type 必须是股票 1")
    if not frame["instrument_status"].isin({"0", "1"}).all():
        raise ValueError("Baostock daily evidence instrument_status 必须是 0/1")
    if frame.duplicated(["stock_code", "date"]).any():
        raise ValueError("Baostock daily evidence stock_code/date 重复")
    for code, symbol in frame[["stock_code", "baostock_code"]].drop_duplicates().itertuples(index=False):
        if symbol != _baostock_symbol(code):
            raise ValueError(f"Baostock daily evidence code/market 不一致: {code}/{symbol}")
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise").values.astype("datetime64[D]")
    for field in (
        "listing_date",
        "first_executable_date",
        "last_executable_date",
    ):
        frame[field] = pd.to_datetime(
            frame[field], format="%Y-%m-%d", errors="raise"
        ).values.astype("datetime64[D]")
    out_text = frame["out_date"].fillna("").astype(str).str.strip()
    parsed_out = pd.to_datetime(
        out_text.replace("", pd.NA),
        format="%Y-%m-%d",
        errors="coerce",
    )
    if (out_text.ne("") & parsed_out.isna()).any():
        raise ValueError("Baostock daily evidence out_date 非法")
    frame["out_date"] = parsed_out.values.astype("datetime64[D]")
    for field in ("source_tradestatus", "tradestatus"):
        frame[field] = frame[field].fillna("").astype(str).str.strip()
        if not frame[field].isin({"", "0", "1"}).all():
            raise ValueError(
                f"Baostock daily evidence {field} 只能是空串/0/1"
            )
    frame["status_correction"] = (
        frame["status_correction"].fillna("").astype(str).str.strip()
    )
    allowed_corrections = {"", *(
        str(contract["correction_id"])
        for contract in BAOSTOCK_STATUS_CORRECTIONS.values()
    )}
    if not frame["status_correction"].isin(allowed_corrections).all():
        raise ValueError("Baostock daily evidence status_correction 未声明")
    for field in (
        "reference_open",
        "reference_volume",
        "reference_amount",
        "direct_preclose",
    ):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    reference_values = frame[
        ["reference_open", "reference_volume", "reference_amount"]
    ].to_numpy(dtype=np.float64)
    if np.isinf(reference_values).any() or (
        frame[["reference_open", "reference_volume", "reference_amount"]] < 0.0
    ).any().any():
        raise ValueError("Baostock daily evidence open/volume/amount 非法")
    direct = frame["direct_preclose"].to_numpy(dtype=np.float64)
    if np.isinf(direct).any() or np.any(direct[np.isfinite(direct)] <= 0.0):
        raise ValueError("Baostock daily evidence direct_preclose 非法")
    applied_text = (
        frame["normalization_applied"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if not applied_text.isin({"true", "false"}).all():
        raise ValueError("Baostock daily evidence normalization_applied 非法")
    frame["normalization_applied"] = applied_text.eq("true")
    hashes = frame["applied_k_sha256"].fillna("").astype(str).str.strip().str.lower()
    if not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Baostock daily evidence applied_k_sha256 非法")
    if not hashes[~frame["normalization_applied"]].eq(UNAPPLIED_K_SHA256).all():
        raise ValueError("未应用 evidence 必须使用固定空绑定 hash")
    if hashes[frame["normalization_applied"]].eq(UNAPPLIED_K_SHA256).any():
        raise ValueError("已应用 evidence 不得使用固定空绑定 hash")
    frame["applied_k_sha256"] = hashes
    source_hashes = (
        frame["source_payload_sha256"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if not source_hashes.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Baostock daily evidence source_payload_sha256 非法")
    frame["source_payload_sha256"] = source_hashes
    expected_source_hashes = frame.apply(_source_payload_sha256, axis=1)
    if not frame["source_payload_sha256"].eq(expected_source_hashes).all():
        raise ValueError(
            "Baostock daily evidence source_payload_sha256 校验失败"
        )

    correction_codes = set(BAOSTOCK_STATUS_CORRECTIONS)
    undeclared = frame[
        frame["status_correction"].ne("")
        & ~frame["stock_code"].isin(correction_codes)
    ]
    if not undeclared.empty:
        raise ValueError("Baostock daily evidence correction 代码未声明")
    unchanged = frame["status_correction"].eq("")
    if not frame.loc[unchanged, "tradestatus"].eq(
        frame.loc[unchanged, "source_tradestatus"]
    ).all():
        raise ValueError("Baostock daily evidence 未纠正行 effective/source 不一致")
    for code, contract in BAOSTOCK_STATUS_CORRECTIONS.items():
        code_rows = frame[frame["stock_code"].eq(code)].copy()
        if code_rows.empty:
            continue
        target_dates = pd.to_datetime(code_rows["date"]).dt.date.between(
            contract["start"], contract["end"]
        )
        corrected = code_rows.loc[target_dates]
        if (
            len(corrected) != contract["row_count"]
            or _status_correction_date_set_sha256(corrected["date"])
            != contract["date_set_sha256"]
            or _status_correction_source_rows_sha256(corrected)
            != contract["source_rows_sha256"]
        ):
            raise ValueError(f"{code} Baostock status correction 固定集合校验失败")
        if not (
            corrected["source_tradestatus"].eq(contract["source_status"]).all()
            and corrected["tradestatus"].eq(contract["effective_status"]).all()
            and corrected["status_correction"].eq(contract["correction_id"]).all()
        ):
            raise ValueError(f"{code} Baostock status correction 状态校验失败")
        outside = code_rows.loc[~target_dates]
        if not outside["status_correction"].eq("").all():
            raise ValueError(f"{code} Baostock status correction 越界")
        by_date = code_rows.set_index("date", drop=False)
        for boundary, expected in (
            (contract["last_executable_boundary"], "1"),
            (contract["termination_boundary"], "0"),
        ):
            key = np.datetime64(boundary, "D")
            if key not in by_date.index or str(
                by_date.loc[key, "tradestatus"]
            ) != expected:
                raise ValueError(f"{code} Baostock status correction 边界校验失败")

    for code, rows in frame.groupby("stock_code", sort=False):
        for field in (
            "baostock_code",
            "instrument_type",
            "instrument_status",
            "listing_date",
            "out_date",
            "first_executable_date",
            "last_executable_date",
        ):
            if rows[field].nunique(dropna=False) != 1:
                raise ValueError(f"Baostock daily evidence {code} {field} 不唯一")
        executable = rows[
            rows["tradestatus"].eq("1")
            | (
                rows["tradestatus"].eq("")
                & rows["reference_open"].gt(0.0)
            )
        ].sort_values("date")
        if executable.empty:
            raise ValueError(f"Baostock daily evidence {code} 无可成交日")
        later_direct = executable.iloc[1:]["direct_preclose"].to_numpy(
            dtype=np.float64
        )
        if later_direct.size and not (
            np.isfinite(later_direct) & (later_direct > 0.0)
        ).all():
            raise ValueError(
                f"Baostock daily evidence {code} 后续可成交日 direct_preclose 非法"
            )
        if (
            executable.iloc[0]["date"] != rows.iloc[0]["first_executable_date"]
            or executable.iloc[-1]["date"] != rows.iloc[0]["last_executable_date"]
        ):
            raise ValueError(f"Baostock daily evidence {code} 首尾可成交日不一致")
    _load_baostock_manifest(path, frame)
    return frame.sort_values(["stock_code", "date"], kind="stable").reset_index(drop=True)


def _merge_baostock_evidence_frames(
    frames: list[pd.DataFrame],
    *,
    previous_path: Path | None,
) -> pd.DataFrame:
    if not frames:
        raise ValueError("Baostock daily evidence 待发布 frames 为空")
    incoming = pd.concat(frames, ignore_index=True).loc[
        :, BAOSTOCK_EVIDENCE_COLUMNS
    ]
    if incoming.empty or incoming.duplicated(["stock_code", "date"]).any():
        raise ValueError("Baostock daily evidence 为空或 stock_code/date 重复")
    incoming_codes = set(
        incoming["stock_code"].astype(str).str.strip().str.upper()
    )
    current = incoming
    if previous_path is not None and Path(previous_path).exists():
        previous = load_baostock_daily_evidence(Path(previous_path))
        incoming_key_to_index = {
            (
                str(row.stock_code).strip().upper(),
                _canonical_date_text(row.date),
            ): index
            for index, row in incoming.iterrows()
        }
        for code in sorted(incoming_codes):
            previous_dates = set(
                pd.to_datetime(
                    previous.loc[previous["stock_code"].eq(code), "date"]
                ).dt.date
            )
            incoming_dates = set(
                pd.to_datetime(
                    incoming.loc[
                        incoming["stock_code"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .eq(code),
                        "date",
                    ]
                ).dt.date
            )
            missing_previous = sorted(previous_dates.difference(incoming_dates))
            if missing_previous:
                raise ValueError(
                    f"Baostock daily evidence {code} 历史日期回退，"
                    f"旧日期必须是新日期子集: {missing_previous[:20]}"
                )
        immutable_daily_fields = (
            "baostock_code",
            "instrument_type",
            "source_tradestatus",
            "tradestatus",
            "status_correction",
            "reference_open",
            "reference_volume",
            "reference_amount",
            "direct_preclose",
            "source",
            "schema_version",
        )
        previous_audited = previous[
            previous["stock_code"].astype(str).isin(incoming_codes)
            & previous["normalization_applied"].astype(bool)
        ]
        for _previous_index, previous_row in previous_audited.iterrows():
            key = (
                str(previous_row["stock_code"]).strip().upper(),
                _canonical_date_text(previous_row["date"]),
            )
            incoming_index = incoming_key_to_index.get(key)
            if incoming_index is None:
                raise ValueError(
                    "Baostock daily evidence 已审计日期在增量源中缺失: "
                    f"{key[0]}/{key[1]}"
                )
            incoming_row = incoming.loc[incoming_index]
            conflicts = [
                field
                for field in immutable_daily_fields
                if _canonical_evidence_value(field, previous_row[field])
                != _canonical_evidence_value(field, incoming_row[field])
            ]
            if conflicts:
                raise ValueError(
                    "Baostock daily evidence 已审计日期源状态/payload 冲突: "
                    f"{key[0]}/{key[1]} fields={conflicts}"
                )
            incoming_applied = (
                _canonical_bool_text(incoming_row["normalization_applied"])
                == "true"
            )
            incoming_hash = str(incoming_row["applied_k_sha256"]).strip()
            if incoming_applied:
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", incoming_hash)
                    or incoming_hash == UNAPPLIED_K_SHA256
                ):
                    raise ValueError(
                        "Baostock daily evidence 新审计 application hash 非法: "
                        f"{key[0]}/{key[1]}"
                    )
                continue
            if incoming_hash != UNAPPLIED_K_SHA256:
                raise ValueError(
                    "Baostock daily evidence 未应用行携带非原子 application hash: "
                    f"{key[0]}/{key[1]}"
                )
            # Evidence-only refreshes legitimately rebuild source/lifecycle
            # rows with an unbound application state.  Preserve the durable
            # audited marker and exact K-state hash from the current snapshot;
            # only a reconcile call may supply a new bound hash.
            incoming.loc[incoming_index, "normalization_applied"] = True
            incoming.loc[incoming_index, "applied_k_sha256"] = str(
                previous_row["applied_k_sha256"]
            )
        retained = previous[~previous["stock_code"].isin(incoming_codes)]
        current = pd.concat([retained, incoming], ignore_index=True)
    current = current.loc[:, BAOSTOCK_EVIDENCE_COLUMNS].sort_values(
        ["stock_code", "date"], kind="stable"
    ).reset_index(drop=True)
    if current.duplicated(["stock_code", "date"]).any():
        raise ValueError("Baostock daily evidence merge 后 stock_code/date 重复")
    return current


def _save_baostock_evidence_atomic(
    frames: list[pd.DataFrame],
    output_path: Path = BAOSTOCK_EVIDENCE_PATH,
) -> None:
    if not frames:
        return
    output_path = Path(output_path)
    output_manifest = _baostock_evidence_manifest_path(output_path)
    protected_state = capture_file_states([output_path, output_manifest])
    current = _merge_baostock_evidence_frames(
        frames,
        previous_path=output_path,
    )
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    staged_path = output_path.with_name(
        f".{output_path.name}.{token}.stage.parquet"
    )
    staged_manifest = _write_baostock_evidence_stage(current, staged_path)
    try:
        assert_file_states_unchanged(
            protected_state,
            label="Baostock evidence/manifest",
        )
        _replace_staged_files_transactionally(
            [
                (staged_path, output_path),
                (
                    staged_manifest,
                    output_manifest,
                ),
            ],
            token=token,
        )
    finally:
        staged_path.unlink(missing_ok=True)
        staged_manifest.unlink(missing_ok=True)


def find_placeholder_candidate_codes(
    *,
    kline_dir: Path = RAW_DIR,
    evidence_path: Path = BAOSTOCK_EVIDENCE_PATH,
) -> tuple[str, ...]:
    """Find audited candidates whose current state is not secondary-exact.

    ``normalization_applied`` is the durable audit marker.  Existing audited
    rows are classified only by that marker plus exact secondary dates; their
    turnover is inspected solely to ensure an old zero placeholder has already
    been normalized.  The legacy positive-price/zero-turnover scan remains
    only for discovering a brand-new, not-yet-audited candidate.
    """
    evidence = load_baostock_daily_evidence(evidence_path)
    secondary_by_code = _secondary_exact_executable_dates_by_code()
    applied = evidence["normalization_applied"].fillna(False).astype(bool)
    audited = evidence.loc[applied].copy()
    audited["audit_date"] = pd.to_datetime(
        audited["date"], errors="raise"
    ).dt.date
    audited_by_code = {
        str(code): rows
        for code, rows in audited.groupby("stock_code", sort=False)
    }
    pending: set[str] = set()
    seen_codes: set[str] = set()
    for path in sorted(kline_dir.glob("*.parquet")):
        code = path.stem
        seen_codes.add(code)
        frame = pd.read_parquet(path)
        required = {*RAW_BAR_COLUMNS, "preClose"}
        missing = required.difference(frame.columns)
        if missing or frame.empty:
            if code in audited_by_code:
                pending.add(code)
            continue
        dates = pd.to_datetime(
            pd.to_numeric(frame["time"], errors="raise"), unit="ms"
        ).dt.date
        if dates.duplicated().any():
            pending.add(code)
            continue
        by_date = frame.assign(_date=dates).set_index("_date", drop=True)

        audited_rows = audited_by_code.get(code)
        if audited_rows is not None:
            exact = secondary_by_code.get(code, frozenset())
            for evidence_row in audited_rows.itertuples(index=False):
                day = evidence_row.audit_date
                if day not in by_date.index:
                    pending.add(code)
                    break
                row = by_date.loc[day]
                prices = pd.to_numeric(
                    row[["open", "high", "low", "close", "preClose"]],
                    errors="coerce",
                ).to_numpy(dtype=np.float64)
                turnover = pd.to_numeric(
                    row[["volume", "amount"]], errors="coerce"
                ).to_numpy(dtype=np.float64)
                if day in exact:
                    turnover_normalized = np.isnan(turnover).all() or (
                        np.isfinite(turnover).all()
                        and (turnover > 0.0).all()
                    )
                    semantic_state = (
                        np.isfinite(prices[:4]).all()
                        and (prices[:4] > 0.0).all()
                        and turnover_normalized
                    )
                else:
                    semantic_state = (
                        np.isnan(prices).all()
                        and np.isfinite(turnover).all()
                        and (turnover == 0.0).all()
                    )
                hash_matches = _kline_row_state_sha256(day, row) == str(
                    evidence_row.applied_k_sha256
                )
                if not semantic_state or not hash_matches:
                    pending.add(code)
                    break

        opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(np.float64)
        volume = pd.to_numeric(frame["volume"], errors="coerce").to_numpy(np.float64)
        amount = pd.to_numeric(frame["amount"], errors="coerce").to_numpy(np.float64)
        candidates = np.flatnonzero(
            np.isfinite(opens)
            & (opens > 0.0)
            & np.isfinite(volume)
            & np.isfinite(amount)
            & (volume <= 0.0)
            & (amount <= 0.0)
        )
        if candidates.size == 0:
            continue
        rows = evidence[evidence["stock_code"].astype(str).eq(code)]
        if rows.empty:
            pending.add(code)
            continue
        evidence_by_date = rows.assign(
            _date=pd.to_datetime(rows["date"]).dt.date
        ).set_index("_date", drop=True)
        for frame_index in candidates:
            day = dates.iloc[int(frame_index)]
            if day not in evidence_by_date.index:
                pending.add(code)
                break
            evidence_row = evidence_by_date.loc[day]
            if not bool(evidence_row["normalization_applied"]):
                pending.add(code)
                break
    pending.update(set(audited_by_code).difference(seen_codes))
    return tuple(sorted(pending))


def _date_set_sha256(values: frozenset[date]) -> str:
    joined = "\n".join(day.isoformat() for day in sorted(values))
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def _secondary_full_date_set(module, code: str) -> frozenset[date]:
    """Load the independently sealed full date set for large remappings."""
    limit = SECONDARY_DATE_SET_LIMITS.get(code)
    if limit is None:
        return frozenset()
    loader = getattr(module, "load_verified_daily_evidence", None)
    if not callable(loader):
        raise TypeError(
            "data.kline_secondary_evidence.load_verified_daily_evidence 必须可调用"
        )
    evidence = loader()
    if not isinstance(evidence, pd.DataFrame) or not {
        "stock_code",
        "date",
    }.issubset(evidence.columns):
        raise TypeError("次级证据 loader 必须返回含 stock_code/date 的 DataFrame")
    rows = evidence[
        evidence["stock_code"].astype(str).str.strip().str.upper().eq(code)
    ]
    dates = frozenset(pd.to_datetime(rows["date"], errors="raise").dt.date)
    if len(dates) != int(limit["count"]):
        raise ValueError(f"{code} 次级证据固定日期数量不一致")
    if not dates or min(dates) != limit["first"] or max(dates) != limit["last"]:
        raise ValueError(f"{code} 次级证据固定日期边界不一致")
    if _date_set_sha256(dates) != limit["sha256"]:
        raise ValueError(f"{code} 次级证据固定日期 digest 不一致")
    required = limit["required"]
    if not required.issubset(dates):
        raise ValueError(f"{code} 次级证据缺少强制日期: {sorted(required)}")
    return dates


def _secondary_exact_executable_dates_by_code() -> dict[str, frozenset[date]]:
    """Load the hash-sealed secondary snapshot as exact executable dates."""

    module = importlib.import_module("data.kline_secondary_evidence")
    loader = getattr(module, "load_verified_daily_evidence", None)
    if not callable(loader):
        raise TypeError(
            "data.kline_secondary_evidence.load_verified_daily_evidence 必须可调用"
        )
    frame = loader()
    required = {"stock_code", "date", "tradestatus", "reference_open"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise TypeError(
            "次级证据 loader 必须返回含 "
            "stock_code/date/tradestatus/reference_open 的 DataFrame"
        )
    evidence = frame.copy()
    evidence["stock_code"] = (
        evidence["stock_code"].astype(str).str.strip().str.upper()
    )
    evidence["date"] = pd.to_datetime(evidence["date"], errors="raise").dt.date
    evidence["tradestatus"] = (
        evidence["tradestatus"].fillna("").astype(str).str.strip()
    )
    open_values = pd.to_numeric(
        evidence["reference_open"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if (
        evidence.duplicated(["stock_code", "date"]).any()
        or not evidence["tradestatus"].eq("1").all()
        or not np.isfinite(open_values).all()
    ):
        raise ValueError("次级证据必须是唯一的明确可成交证据日")
    return {
        str(code): frozenset(rows["date"])
        for code, rows in evidence.groupby("stock_code", sort=False)
    }


def _secondary_coverage_dates(
    code: str,
    missing_dates: frozenset[date],
) -> frozenset[date]:
    """Resolve exact offline/hash-sealed secondary-evidence dates.

    The optional module owns manifest/schema/source/hash validation.  This
    caller still enforces a narrow code whitelist, exact ``date`` values and
    subset semantics, so the extension cannot become a generic bypass.
    """
    if code not in SECONDARY_COVERAGE_CODES or not missing_dates:
        return frozenset()
    try:
        module = importlib.import_module("data.kline_secondary_evidence")
    except ModuleNotFoundError as exc:
        if exc.name == "data.kline_secondary_evidence":
            return frozenset()
        raise
    resolver = getattr(module, "covered_local_dates", None)
    if not callable(resolver):
        raise TypeError(
            "data.kline_secondary_evidence.covered_local_dates 必须可调用"
        )
    covered = resolver(code, missing_dates)
    if not isinstance(covered, frozenset) or any(
        type(day) is not date for day in covered
    ):
        raise TypeError(
            "covered_local_dates 必须返回 frozenset[datetime.date]"
        )
    if not covered.issubset(missing_dates):
        unexpected = sorted(covered.difference(missing_dates))
        raise ValueError(
            f"{code} 次级证据返回非请求日期: {unexpected[:20]}"
        )
    exact_limit = SECONDARY_EXACT_DATE_LIMITS.get(code)
    if exact_limit is not None and not covered.issubset(exact_limit):
        unexpected = sorted(covered.difference(exact_limit))
        raise ValueError(
            f"{code} 次级证据越过本层精确日期白名单: {unexpected[:20]}"
        )
    full_limit = _secondary_full_date_set(module, code)
    if full_limit and covered != missing_dates.intersection(full_limit):
        raise ValueError(f"{code} 次级证据返回集合与固定日期集合不一致")
    return covered


def _validate_baostock_local_coverage(
    code: str,
    local: pd.DataFrame,
    history: pd.DataFrame,
    *,
    query_start: date,
    query_end: date,
    ipo_date: date,
    out_date: date | None,
) -> None:
    times = pd.to_datetime(
        pd.to_numeric(local["time"], errors="raise"),
        unit="ms",
    )
    if times.isna().any() or times.duplicated().any():
        raise KlineSourceError(
            code,
            "local_kline",
            "duplicate_rows",
            "本地 K 线 time 为空或重复",
        )
    local_dates = set(times.dt.date)
    source_dates = set(history["date"])
    missing = frozenset(local_dates.difference(source_dates))
    secondary_covered = _secondary_coverage_dates(
        code,
        missing,
    )
    uncovered = sorted(missing.difference(secondary_covered))
    if uncovered:
        shown = ", ".join(value.isoformat() for value in uncovered[:20])
        raise KlineSourceError(
            code,
            "baostock_history",
            "coverage_mismatch",
            f"本地 K 线有 {len(uncovered)} 个日期未被源或显式离线证据覆盖: {shown}",
        )
    if code in SECONDARY_DATE_SET_LIMITS:
        module = importlib.import_module("data.kline_secondary_evidence")
        required_local_dates = _secondary_full_date_set(module, code)
        missing_local = sorted(required_local_dates.difference(local_dates))
        if missing_local:
            shown = ", ".join(value.isoformat() for value in missing_local[:20])
            raise KlineSourceError(
                code,
                "local_kline",
                "coverage_mismatch",
                "本地 K 线缺少次级源确认的真实可成交日 "
                f"{len(missing_local)} 个: {shown}",
            )
    if query_start > ipo_date:
        raise KlineSourceError(
            code,
            "baostock_history",
            "boundary_mismatch",
            f"查询起点 {query_start} 晚于 IPO {ipo_date}",
        )
    lifecycle_tail = out_date or max(local_dates)
    if query_end < lifecycle_tail:
        raise KlineSourceError(
            code,
            "baostock_history",
            "boundary_mismatch",
            f"查询终点 {query_end} 早于生命周期尾 {lifecycle_tail}",
        )


def _collect_baostock_reference_batch(
    backups: dict[str, pd.DataFrame],
    *,
    require_delisted: bool,
    baostock_module=None,
    evidence_only: bool = False,
) -> tuple[
    dict[str, pd.DataFrame],
    list[pd.DataFrame],
    list[tuple[str, Exception]],
]:
    if not backups:
        return {}, [], []
    if baostock_module is None:
        import baostock as baostock_module

    session = _BaostockSession(baostock_module)
    production: dict[str, pd.DataFrame] = {}
    evidence_frames: list[pd.DataFrame] = []
    failures: list[tuple[str, Exception]] = []
    try:
        try:
            basics, basic_failures = _baostock_basic_map_isolated(
                session, sorted(backups)
            )
            failures.extend(basic_failures)
        except Exception as exc:
            return {}, [], [(code, exc) for code in sorted(backups)]
        for code in sorted(backups):
            if code not in basics:
                continue
            try:
                backup = backups[code]
                times = pd.to_datetime(
                    pd.to_numeric(backup["time"], errors="raise"),
                    unit="ms",
                )
                backup_start = times.min().date()
                backup_end = times.max().date()
                ipo_date = pd.Timestamp(str(basics[code]["ipoDate"])).date()
                out_text = str(basics[code]["outDate"]).strip()
                out_date = pd.Timestamp(out_text).date() if out_text else None
                query_start = min(backup_start, ipo_date)
                query_end = max(
                    backup_end,
                    out_date or (date.today() if evidence_only else backup_end),
                )
                history = _baostock_history(
                    session,
                    code,
                    query_start,
                    query_end,
                    lifecycle_start=ipo_date,
                    lifecycle_end=out_date or query_end,
                )
                if evidence_only:
                    _validate_baostock_local_coverage(
                        code,
                        backup,
                        history,
                        query_start=query_start,
                        query_end=query_end,
                        ipo_date=ipo_date,
                        out_date=out_date,
                    )
                    evidence, *_lifecycle = _build_baostock_evidence(
                        code,
                        basics[code],
                        history,
                    )
                    frame = backup
                else:
                    frame, evidence = _baostock_production_and_evidence(
                        code,
                        backup,
                        basics[code],
                        history,
                        require_delisted=require_delisted,
                    )
                production[code] = frame
                evidence_frames.append(evidence)
            except Exception as exc:
                failures.append((code, exc))
    finally:
        session.close()
    return production, evidence_frames, failures


def _baostock_reference_batch(
    backups: dict[str, pd.DataFrame],
    *,
    require_delisted: bool,
    evidence_path: Path,
    baostock_module=None,
    evidence_only: bool = False,
) -> tuple[
    dict[str, pd.DataFrame],
    list[tuple[str, Exception]],
]:
    production, evidence_frames, failures = _collect_baostock_reference_batch(
        backups,
        require_delisted=require_delisted,
        baostock_module=baostock_module,
        evidence_only=evidence_only,
    )
    # A batch is one evidence snapshot transaction.  A single failed symbol
    # makes the requested set incomplete, so no successful subset is sealed.
    if not failures:
        _save_baostock_evidence_atomic(evidence_frames, evidence_path)
    return production, failures


def update_baostock_listing_evidence(
    codes: list[str],
    *,
    kline_dir: Path = RAW_DIR,
    evidence_path: Path = BAOSTOCK_EVIDENCE_PATH,
    baostock_module=None,
) -> tuple[str, ...]:
    """Persist daily lifecycle/executable evidence without changing K-line files."""
    normalized = sorted(set(codes))
    backups: dict[str, pd.DataFrame] = {}
    missing = []
    for code in normalized:
        path = kline_dir / f"{code}.parquet"
        if not path.exists():
            missing.append(code)
            continue
        backups[code] = pd.read_parquet(path)
    if missing:
        raise KlineBatchError(
            "Baostock 上市证据更新",
            [
                (
                    code,
                    KlineSourceError(
                        code,
                        "local_kline",
                        "missing_file",
                        f"{kline_dir / f'{code}.parquet'} 不存在",
                    ),
                )
                for code in missing
            ],
        )
    production, failures = _baostock_reference_batch(
        backups,
        require_delisted=False,
        evidence_path=evidence_path,
        baostock_module=baostock_module,
        evidence_only=True,
    )
    if failures:
        raise KlineBatchError(
            "Baostock 上市证据更新",
            failures,
            succeeded=(),
        ) from failures[0][1]
    return tuple(production)


def reconcile_baostock_placeholder_bars(
    codes: list[str],
    *,
    kline_dir: Path = RAW_DIR,
    evidence_path: Path = BAOSTOCK_EVIDENCE_PATH,
    baostock_module=None,
) -> dict[str, Path]:
    """Seal evidence and atomically normalize existing placeholder candidates.

    Raw OHLCVA remains exclusively the existing local mootdx frame.  Baostock
    contributes direct daily preClose, lifecycle and status validation only.
    An audited placeholder candidate is retained only by exact hash-sealed
    secondary evidence; primary-only missing dates are never inferred.
    """
    normalized = sorted(set(codes))
    if not normalized:
        raise ValueError("Baostock candidate reconcile 目标代码不得为空")
    kline_dir = Path(kline_dir)
    evidence_path = Path(evidence_path)
    kline_paths = {
        code: kline_dir / f"{code}.parquet" for code in normalized
    }
    evidence_manifest_path = _baostock_evidence_manifest_path(evidence_path)
    protected_state = capture_file_states(
        [*kline_paths.values(), evidence_path, evidence_manifest_path]
    )
    local: dict[str, pd.DataFrame] = {}
    missing: list[tuple[str, Exception]] = []
    for code in normalized:
        path = kline_paths[code]
        if not path.exists():
            missing.append(
                (
                    code,
                    KlineSourceError(
                        code,
                        "local_kline",
                        "missing_file",
                        f"{path} 不存在",
                    ),
                )
            )
            continue
        local[code] = pd.read_parquet(path)
    if missing:
        raise KlineBatchError("Baostock 占位行情归一化", missing)

    previous_audited: dict[str, frozenset[date]] = {}
    if evidence_path.exists():
        previous = load_baostock_daily_evidence(evidence_path)
        applied = previous["normalization_applied"].fillna(False).astype(bool)
        previous_audited = {
            str(code): frozenset(pd.to_datetime(rows["date"]).dt.date)
            for code, rows in previous.loc[applied].groupby(
                "stock_code", sort=False
            )
            if str(code) in normalized
        }

    production, evidence_frames, failures = _collect_baostock_reference_batch(
        local,
        require_delisted=False,
        baostock_module=baostock_module,
    )
    if failures:
        raise KlineBatchError(
            "Baostock 占位行情归一化",
            failures,
            succeeded=(),
        ) from failures[0][1]

    secondary_by_code = _secondary_exact_executable_dates_by_code()
    rebound_frames: list[pd.DataFrame] = []
    for evidence in evidence_frames:
        code = str(evidence.iloc[0]["stock_code"])
        fresh_applied = frozenset(
            pd.to_datetime(
                evidence.loc[evidence["normalization_applied"], "date"]
            ).dt.date
        )
        audited = previous_audited.get(code, frozenset()) | fresh_applied
        secondary_dates = secondary_by_code.get(code, frozenset())
        production[code], _unsupported = _mask_unconfirmed_audited_candidates(
            code,
            production[code],
            audited,
            secondary_dates,
        )
        production_dates = pd.to_datetime(
            pd.to_numeric(production[code]["time"], errors="raise"), unit="ms"
        ).dt.date
        production_by_date = production[code].assign(
            _date=production_dates
        ).set_index("_date", drop=True)
        unavailable_confirmed = sorted(
            day
            for day in audited.intersection(secondary_dates)
            if day not in production_by_date.index
            or not (
                np.isfinite(float(production_by_date.loc[day, "open"]))
                and float(production_by_date.loc[day, "open"]) > 0.0
            )
        )
        if unavailable_confirmed:
            raise KlineSourceError(
                code,
                "secondary_evidence",
                "coverage_mismatch",
                "次级源确认的 candidate 本地价格仍为空，"
                f"必须从已归档 mootdx 重建: {unavailable_confirmed[:20]}",
            )
        rebound_frames.append(
            _bind_evidence_to_applied_dates(
                evidence,
                production[code],
                set(audited),
            )
        )
    evidence_frames = rebound_frames

    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    staged_evidence = evidence_path.with_name(
        f".{evidence_path.name}.{token}.stage.parquet"
    )
    staged_manifest = _baostock_evidence_manifest_path(staged_evidence)
    staged_k: dict[str, Path] = {}
    try:
        merged_evidence = _merge_baostock_evidence_frames(
            evidence_frames,
            previous_path=evidence_path,
        )
        staged_manifest = _write_baostock_evidence_stage(
            merged_evidence,
            staged_evidence,
        )
        evidence_by_code = {
            str(frame.iloc[0]["stock_code"]): frame
            for frame in evidence_frames
        }
        for code, frame in production.items():
            stage_path = kline_dir / f".{code}.{token}.stage.parquet"
            _write_parquet_atomic(frame, stage_path)
            staged = pd.read_parquet(stage_path)
            if _kline_content_sha256(staged) != _kline_content_sha256(frame):
                raise ValueError(f"{code} staged K 内容 hash 不一致")
            _validate_evidence_application_hashes(
                staged,
                evidence_by_code[code],
            )
            staged_k[code] = stage_path
        assert_file_states_unchanged(
            protected_state,
            label="Baostock candidate K/evidence",
        )
        _replace_staged_files_transactionally(
            [
                *[
                    (staged_k[code], kline_dir / f"{code}.parquet")
                    for code in sorted(staged_k)
                ],
                (staged_evidence, evidence_path),
                (
                    staged_manifest,
                    _baostock_evidence_manifest_path(evidence_path),
                ),
            ],
            token=token,
        )
    except Exception as exc:
        raise KlineBatchError(
            "Baostock 占位行情归一化",
            [("*", exc)],
            succeeded=(),
        ) from exc
    finally:
        for path in [*staged_k.values(), staged_evidence, staged_manifest]:
            path.unlink(missing_ok=True)
    return {
        code: kline_dir / f"{code}.parquet" for code in sorted(production)
    }


def reconcile_baostock_status_corrections(
    codes: list[str] | None = None,
    *,
    kline_dir: Path = RAW_DIR,
    evidence_path: Path = BAOSTOCK_EVIDENCE_PATH,
    baostock_module=None,
) -> dict[str, Path]:
    """Atomically apply the finite reviewed status corrections and evidence.

    This is a permanent targeted updater, not a schema migration.  It queries
    the complete real primary history, verifies every fixed correction row,
    stages the corrected K files plus the v4 evidence/sidecar, and delegates
    publication to the same rollback-protected transaction as ordinary
    placeholder reconciliation.
    """

    requested = sorted(
        set(BAOSTOCK_STATUS_CORRECTIONS if codes is None else codes)
    )
    unknown = sorted(set(requested).difference(BAOSTOCK_STATUS_CORRECTIONS))
    if unknown or not requested:
        raise ValueError(
            "Baostock status correction 只能更新固定目标: "
            f"unknown={unknown}"
        )
    published = reconcile_baostock_placeholder_bars(
        requested,
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=baostock_module,
    )
    verified = load_baostock_daily_evidence(evidence_path)
    for code in requested:
        contract = BAOSTOCK_STATUS_CORRECTIONS[code]
        rows = verified[
            verified["stock_code"].eq(code)
            & verified["status_correction"].eq(contract["correction_id"])
        ]
        if len(rows) != contract["row_count"]:
            raise RuntimeError(f"{code} status correction 发布后覆盖不完整")
    return published


def _restore_masked_backup_intersections(
    code: str,
    current: pd.DataFrame,
    backup: pd.DataFrame,
    history: pd.DataFrame,
    *,
    first_executable: date,
    secondary_executable_dates: frozenset[date],
) -> tuple[pd.DataFrame, set[date]]:
    """Restore audited candidates only with exact secondary confirmation."""
    required = [*RAW_BAR_COLUMNS, "preClose"]
    missing = set(required).difference(current.columns)
    if missing or current.empty:
        raise KlineSourceError(
            code,
            "local_kline",
            "schema_mismatch",
            f"main K 线为空或缺列: {sorted(missing)}",
        )
    result = current.loc[:, required].copy().reset_index(drop=True)
    result["time"] = pd.to_numeric(result["time"], errors="raise").astype(np.int64)
    if result["time"].duplicated().any() or not result["time"].is_monotonic_increasing:
        raise KlineSourceError(
            code,
            "local_kline",
            "duplicate_rows",
            "main K 线 time 必须严格升序且不重复",
        )
    numeric = result.loc[:, required[1:]].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(dtype=np.float64)).any():
        raise KlineSourceError(
            code, "local_kline", "invalid_value", "main K 线包含无穷"
        )
    result.loc[:, required[1:]] = numeric

    archived = _raw_backup_to_production(backup, pd.DataFrame())
    archived_dates = pd.to_datetime(archived["time"], unit="ms").dt.date
    archived = archived.assign(_date=archived_dates).set_index("_date", drop=True)
    if archived.index.duplicated().any():
        raise KlineSourceError(
            code, "backup_kline", "duplicate_rows", "backup 日期重复"
        )

    current_dates = pd.to_datetime(result["time"], unit="ms").dt.date
    reference = history.set_index("date", drop=False)
    restored_dates: set[date] = set()
    for row_index, day in enumerate(current_dates):
        if day not in archived.index:
            continue
        if day not in secondary_executable_dates:
            continue
        current_prices = result.loc[
            row_index, ["open", "high", "low", "close"]
        ].to_numpy(dtype=np.float64)
        missing_prices = ~np.isfinite(current_prices)
        if missing_prices.any() and not missing_prices.all():
            raise KlineSourceError(
                code,
                "local_kline",
                "partial_mask",
                f"{day} main OHLC 仅部分为空，拒绝猜测修复",
            )
        if day not in reference.index:
            # Coverage exceptions prove that retaining this local date is
            # legitimate, but they do not fabricate Baostock daily status.
            # Therefore an exception date is never restored from backup here.
            continue
        source_row = reference.loc[day]
        if not missing_prices.all():
            continue
        archived_row = archived.loc[day]
        archived_prices = archived_row[["open", "high", "low", "close"]].to_numpy(
            dtype=np.float64
        )
        if not (np.isfinite(archived_prices) & (archived_prices > 0.0)).all():
            raise KlineSourceError(
                code,
                "backup_kline",
                "invalid_value",
                f"{day} backup OHLC 不能证明原始价格",
            )
        result.loc[row_index, ["open", "high", "low", "close"]] = archived_prices
        archived_turnover = archived_row[["volume", "amount"]].to_numpy(
            dtype=np.float64
        )
        if (archived_turnover <= 0.0).all():
            result.loc[row_index, ["volume", "amount"]] = np.nan
        else:
            result.loc[row_index, ["volume", "amount"]] = archived_turnover
        direct = float(source_row["preclose"])
        if day == first_executable:
            result.loc[row_index, "preClose"] = np.nan
        elif np.isfinite(direct) and direct > 0.0:
            result.loc[row_index, "preClose"] = direct
        else:
            raise KlineSourceError(
                code,
                "baostock_history",
                "invalid_preclose",
                f"{day} 恢复行缺少有效 direct preclose",
            )
        restored_dates.add(day)
    return result, restored_dates


def _mask_unconfirmed_audited_candidates(
    code: str,
    current: pd.DataFrame,
    audited_dates: frozenset[date],
    secondary_executable_dates: frozenset[date],
) -> tuple[pd.DataFrame, set[date]]:
    """Undo former primary-only candidate restores without turnover inference."""

    if not audited_dates:
        return current.copy(), set()
    required = [*RAW_BAR_COLUMNS, "preClose"]
    missing = set(required).difference(current.columns)
    if missing or current.empty:
        raise KlineSourceError(
            code,
            "local_kline",
            "schema_mismatch",
            f"main K 线为空或缺列: {sorted(missing)}",
        )
    result = current.loc[:, required].copy().reset_index(drop=True)
    current_dates = pd.to_datetime(
        pd.to_numeric(result["time"], errors="raise"), unit="ms"
    ).dt.date
    if current_dates.duplicated().any():
        raise KlineSourceError(
            code, "local_kline", "duplicate_rows", "main K 线日期重复"
        )
    local_dates = set(current_dates)
    missing_audited = sorted(audited_dates.difference(local_dates))
    if missing_audited:
        raise KlineSourceError(
            code,
            "local_kline",
            "coverage_mismatch",
            "旧 evidence 已审计 candidate 日期在本地 K 线中缺失: "
            f"{missing_audited[:20]}",
        )
    unsupported = set(audited_dates).difference(secondary_executable_dates)
    mask = current_dates.isin(unsupported)
    result.loc[mask, ["open", "high", "low", "close", "preClose"]] = np.nan
    result.loc[mask, ["volume", "amount"]] = 0.0
    return result, unsupported


def rebuild_masked_kline_rows_from_backups(
    codes: list[str],
    *,
    kline_dir: Path = RAW_DIR,
    backup_dir: Path = RAW_BACKUP_DIR,
    evidence_path: Path = BAOSTOCK_EVIDENCE_PATH,
    baostock_module=None,
) -> dict[str, int]:
    """Atomically rebuild v4 evidence and reconcile audited candidates.

    The main file is authoritative.  Backup rows are never appended and never
    replace an already finite main OHLC row; they can only restore an existing
    same-date row whose complete OHLC was hidden and whose exact hash-sealed
    secondary evidence says it was executable.  Previously normalized
    candidates without that exact override are re-masked by their stored audit
    marker, never by volume/amount.  Every code is downloaded, validated and
    staged before any production file is replaced.
    """
    normalized = sorted(set(codes))
    if not normalized:
        return {}
    previous_codes: set[str] = set()
    backup_required_codes: set[str] = set()
    audited_dates_by_code: dict[str, frozenset[date]] = {}
    if evidence_path.exists():
        previous_evidence = load_baostock_daily_evidence(evidence_path)
        previous_codes = set(
            previous_evidence["stock_code"].astype(str).str.strip().str.upper()
        )
        if "normalization_applied" not in previous_evidence.columns:
            backup_required_codes = previous_codes.copy()
        else:
            applied = (
                previous_evidence["normalization_applied"]
                .fillna(False)
                .astype(bool)
            )
            backup_required_codes = set(
                previous_evidence.loc[applied, "stock_code"]
                .astype(str)
                .str.strip()
                .str.upper()
            )
            audited_dates_by_code = {
                str(code): frozenset(pd.to_datetime(rows["date"]).dt.date)
                for code, rows in previous_evidence.loc[applied].groupby(
                    "stock_code", sort=False
                )
            }
        omitted = sorted(previous_codes.difference(normalized))
        if omitted:
            raise ValueError(
                "重建代码未覆盖旧 evidence 全集，拒绝丢弃未重建代码: "
                + ", ".join(omitted[:20])
            )
    failures: list[tuple[str, Exception]] = []
    current_by_code: dict[str, pd.DataFrame] = {}
    backup_by_code: dict[str, pd.DataFrame] = {}
    for code in normalized:
        current_path = kline_dir / f"{code}.parquet"
        backup_path = backup_dir / f"{code}.parquet"
        try:
            if not current_path.exists() or (
                code in backup_required_codes and not backup_path.exists()
            ):
                raise KlineSourceError(
                    code,
                    "local_kline",
                    "missing_file",
                    f"main/backup 缺失: {current_path.exists()}/{backup_path.exists()}",
                )
            current_by_code[code] = pd.read_parquet(current_path)
            if backup_path.exists():
                backup_by_code[code] = pd.read_parquet(backup_path)
        except Exception as exc:
            failures.append((code, exc))
    if failures:
        raise KlineBatchError("Baostock v4 candidate 重建", failures)

    secondary_dates_by_code = _secondary_exact_executable_dates_by_code()

    if baostock_module is None:
        import baostock as baostock_module

    session = _BaostockSession(baostock_module)
    rebuilt: dict[str, pd.DataFrame] = {}
    evidence_frames: list[pd.DataFrame] = []
    restored_counts: dict[str, int] = {}
    try:
        try:
            basics, basic_failures = _baostock_basic_map_isolated(
                session, normalized
            )
            failures.extend(basic_failures)
        except Exception as exc:
            failures.extend((code, exc) for code in normalized)
            basics = {}
        for code in normalized:
            if code not in basics:
                continue
            try:
                current = current_by_code[code]
                times = pd.to_datetime(
                    pd.to_numeric(current["time"], errors="raise"), unit="ms"
                )
                current_start = times.min().date()
                current_end = times.max().date()
                ipo_date = pd.Timestamp(str(basics[code]["ipoDate"])).date()
                out_text = str(basics[code]["outDate"]).strip()
                out_date = pd.Timestamp(out_text).date() if out_text else None
                query_start = min(current_start, ipo_date)
                query_end = max(current_end, out_date or date.today())
                history = _baostock_history(
                    session,
                    code,
                    query_start,
                    query_end,
                    lifecycle_start=ipo_date,
                    lifecycle_end=out_date or query_end,
                )
                _validate_baostock_local_coverage(
                    code,
                    current,
                    history,
                    query_start=query_start,
                    query_end=query_end,
                    ipo_date=ipo_date,
                    out_date=out_date,
                )
                evidence, _ipo, _out, _source_first, _source_last = (
                    _build_baostock_evidence(code, basics[code], history)
                )
                audited_dates = audited_dates_by_code.get(code, frozenset())
                secondary_dates = secondary_dates_by_code.get(code, frozenset())
                remasked, _unsupported_dates = (
                    _mask_unconfirmed_audited_candidates(
                        code,
                        current,
                        audited_dates,
                        secondary_dates,
                    )
                )
                remasked_dates = pd.to_datetime(
                    pd.to_numeric(remasked["time"], errors="raise"), unit="ms"
                ).dt.date
                finite_local = remasked.loc[
                    pd.to_numeric(remasked["open"], errors="coerce").gt(0.0)
                ]
                finite_dates = set(
                    pd.to_datetime(
                        pd.to_numeric(finite_local["time"], errors="raise"),
                        unit="ms",
                    ).dt.date
                )
                restorable_secondary = secondary_dates.intersection(
                    set(remasked_dates)
                )
                exact_first = min(finite_dates | restorable_secondary)
                if code in backup_by_code:
                    frame, restored_dates = _restore_masked_backup_intersections(
                        code,
                        remasked,
                        backup_by_code[code],
                        history,
                        first_executable=exact_first,
                        secondary_executable_dates=secondary_dates,
                    )
                else:
                    frame, restored_dates = remasked.copy(), set()
                correction_dates = frozenset(
                    pd.to_datetime(
                        evidence.loc[
                            evidence["status_correction"].ne(""), "date"
                        ]
                    ).dt.date
                ).intersection(set(remasked_dates))
                evidence = _bind_evidence_to_applied_dates(
                    evidence,
                    frame,
                    set(audited_dates)
                    | set(correction_dates)
                    | set(restored_dates),
                )
                rebuilt[code] = frame
                evidence_frames.append(evidence)
                restored_counts[code] = len(restored_dates)
            except Exception as exc:
                failures.append((code, exc))
    finally:
        session.close()
    if failures:
        raise KlineBatchError(
            "Baostock v4 candidate 重建",
            failures,
            succeeded=(),
        ) from failures[0][1]

    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    staged_evidence = evidence_path.with_name(
        f".{evidence_path.name}.{token}.stage.parquet"
    )
    staged_manifest = _baostock_evidence_manifest_path(staged_evidence)
    staged_k: dict[str, Path] = {}
    try:
        merged_evidence = _merge_baostock_evidence_frames(
            evidence_frames,
            previous_path=evidence_path,
        )
        staged_manifest = _write_baostock_evidence_stage(
            merged_evidence,
            staged_evidence,
        )
        evidence_by_code = {
            str(frame.iloc[0]["stock_code"]): frame for frame in evidence_frames
        }
        for code, frame in rebuilt.items():
            stage_path = kline_dir / f".{code}.{token}.stage.parquet"
            _write_parquet_atomic(frame, stage_path)
            staged = pd.read_parquet(stage_path)
            if _kline_content_sha256(staged) != _kline_content_sha256(frame):
                raise ValueError(f"{code} staged K 内容 hash 不一致")
            _validate_evidence_application_hashes(
                staged, evidence_by_code[code]
            )
            staged_k[code] = stage_path
        _replace_staged_files_transactionally(
            [
                *[
                    (staged_k[code], kline_dir / f"{code}.parquet")
                    for code in normalized
                ],
                (staged_evidence, evidence_path),
                (
                    staged_manifest,
                    _baostock_evidence_manifest_path(evidence_path),
                ),
            ],
            token=token,
        )
    finally:
        for path in [*staged_k.values(), staged_evidence, staged_manifest]:
            path.unlink(missing_ok=True)
    return restored_counts


def restore_backup_bars(
    codes: list[str],
    *,
    backup_dir: Path = RAW_BACKUP_DIR,
    evidence_path: Path = BAOSTOCK_EVIDENCE_PATH,
    strict: bool = True,
    mdx=None,
    baostock_module=None,
    use_baostock_fallback: bool = True,
) -> dict[str, Path]:
    """Atomically restore delisted bars from local raw backups.

    The live mootdx corporate-action endpoint is tried first.  When it is empty
    or unavailable, Baostock supplies direct preClose plus sealed daily
    executable evidence; only local mootdx backup OHLCVA is retained.  Files
    without a backup are left untouched for the caller's live-bars retry.
    """
    available = [
        code for code in sorted(set(codes))
        if (backup_dir / f"{code}.parquet").exists()
    ]
    if not available:
        return {}
    backups: dict[str, pd.DataFrame] = {}
    failures: list[tuple[str, Exception]] = []
    for code in available:
        try:
            backups[code] = pd.read_parquet(backup_dir / f"{code}.parquet")
        except Exception as exc:
            failures.append((code, exc))

    client = mdx
    connection_failure: KlineSourceError | None = None
    if client is None:
        try:
            client = _connect_mootdx()
        except KlineSourceError as exc:
            connection_failure = exc

    production_by_code: dict[str, pd.DataFrame] = {}
    baostock_backups: dict[str, pd.DataFrame] = {}
    for code, backup in backups.items():
        if client is None:
            if use_baostock_fallback:
                baostock_backups[code] = backup
            else:
                assert connection_failure is not None
                failures.append((code, connection_failure))
            continue
        try:
            xdxr = _fetch_xdxr_with_retry(
                client,
                code,
                require_nonempty=True,
            )
            production_by_code[code] = _raw_backup_to_production(backup, xdxr)
        except KlineSourceError as exc:
            if use_baostock_fallback:
                baostock_backups[code] = backup
            else:
                failures.append((code, exc))
        except Exception as exc:
            failures.append((code, exc))

    evidence_frames: list[pd.DataFrame] = []
    if baostock_backups:
        try:
            fallback, fallback_evidence, fallback_failures = (
                _collect_baostock_reference_batch(
                    baostock_backups,
                    require_delisted=True,
                    baostock_module=baostock_module,
                )
            )
            failures.extend(fallback_failures)
            if not fallback_failures:
                production_by_code.update(fallback)
                evidence_frames.extend(fallback_evidence)
        except Exception as exc:
            failures.extend((code, exc) for code in baostock_backups)

    # strict means the requested restore set is one transaction.  Source
    # failures therefore leave every formal K/evidence file unchanged.
    if strict and failures:
        _warn_failed_codes("退市原始备份迁移", failures)
        raise KlineBatchError(
            "退市原始备份迁移",
            failures,
            succeeded=(),
        ) from failures[0][1]

    restored: dict[str, Path] = {}
    if production_by_code:
        token = f"{os.getpid()}.{uuid.uuid4().hex}"
        staged_k: dict[str, Path] = {}
        staged_evidence = evidence_path.with_name(
            f".{evidence_path.name}.{token}.stage.parquet"
        )
        staged_manifest = _baostock_evidence_manifest_path(staged_evidence)
        try:
            pairs: list[tuple[Path, Path]] = []
            if evidence_frames:
                merged_evidence = _merge_baostock_evidence_frames(
                    evidence_frames,
                    previous_path=evidence_path,
                )
                staged_manifest = _write_baostock_evidence_stage(
                    merged_evidence,
                    staged_evidence,
                )
            evidence_by_code = {
                str(frame.iloc[0]["stock_code"]): frame
                for frame in evidence_frames
            }
            for code, production in sorted(production_by_code.items()):
                stage_path = RAW_DIR / f".{code}.{token}.stage.parquet"
                _write_parquet_atomic(production, stage_path)
                staged = pd.read_parquet(stage_path)
                if _kline_content_sha256(staged) != _kline_content_sha256(
                    production
                ):
                    raise ValueError(f"{code} staged K 内容 hash 不一致")
                if code in evidence_by_code:
                    _validate_evidence_application_hashes(
                        staged,
                        evidence_by_code[code],
                    )
                staged_k[code] = stage_path
                pairs.append((stage_path, RAW_DIR / f"{code}.parquet"))
            if evidence_frames:
                pairs.extend(
                    [
                        (staged_evidence, evidence_path),
                        (
                            staged_manifest,
                            _baostock_evidence_manifest_path(evidence_path),
                        ),
                    ]
                )
            _replace_staged_files_transactionally(pairs, token=token)
            restored = {
                code: RAW_DIR / f"{code}.parquet"
                for code in sorted(production_by_code)
            }
        except Exception as exc:
            failures.append(("*", exc))
        finally:
            for path in [*staged_k.values(), staged_evidence, staged_manifest]:
                path.unlink(missing_ok=True)
    _warn_failed_codes("退市原始备份迁移", failures)
    if strict and failures:
        raise KlineBatchError(
            "退市原始备份迁移",
            failures,
            succeeded=(),
        ) from failures[0][1]
    return restored


def _fetch_recent_raw_with_retry(mdx, code: str, days: int):
    xdxr_df = _fetch_xdxr_with_retry(mdx, code)
    last_exc: KlineSourceError | None = None
    offset = min(days * 5, RECENT_BARS)
    for attempt in range(1, PER_CODE_RETRIES + 1):
        try:
            bars = mdx.bars(symbol=code[:6], frequency=9, start=0, offset=offset, fq=0)
            if bars is None or bars.empty:
                raise KlineSourceError(
                    code,
                    "bars_recent",
                    "empty_response",
                    "mootdx 返回空增量 K 线",
                    attempts=attempt,
                )
            raw = _mootdx_bars_to_df(bars, xdxr_df=xdxr_df)
            if raw is None:
                raise KlineSourceError(
                    code,
                    "bars_recent",
                    "unusable_payload",
                    "增量 K 线响应没有有限正价格行",
                    attempts=attempt,
                )
            return raw
        except KlineSourceError as exc:
            last_exc = exc
        except Exception as exc:
            last_exc = KlineSourceError(
                code,
                "bars_recent",
                "request_error",
                f"{type(exc).__name__}: {exc}",
                attempts=attempt,
            )
    assert last_exc is not None
    raise KlineSourceError(
        code,
        last_exc.operation,
        last_exc.kind,
        last_exc.detail,
        attempts=PER_CODE_RETRIES,
    ) from last_exc


def _probe_tdx_server(ip: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def _validate_tdx_client(client) -> bool:
    """Require real bar and positive corporate-action endpoint probes."""
    try:
        bars = client.bars(symbol="000001", frequency=9, offset=1)
        xdxr = client.xdxr(symbol=XDXR_HEALTH_SYMBOL)
    except Exception:
        return False
    return bool(
        bars is not None
        and isinstance(bars, pd.DataFrame)
        and not bars.empty
        and isinstance(xdxr, pd.DataFrame)
        and not xdxr.empty
        and not XDXR_REQUIRED_COLUMNS.difference(xdxr.columns)
    )


def _connect_mootdx():
    """Return the first server passing bar and corporate-action health checks."""
    from mootdx.quotes import Quotes

    for server in TDX_SERVERS:
        if not _probe_tdx_server(*server):
            continue
        try:
            client = Quotes.factory(market="std", server=server)
        except Exception:
            continue
        if _validate_tdx_client(client):
            return client
    for kwargs in ({"bestip": True}, {}):
        try:
            client = Quotes.factory(market="std", **kwargs)
        except Exception:
            continue
        if _validate_tdx_client(client):
            return client
    raise KlineSourceError(
        "*",
        "connect",
        "server_unavailable",
        "所有 mootdx 服务器均无法同时返回真实 A 股 K 线与除权数据；拒绝使用空响应继续更新",
        attempts=len(TDX_SERVERS) + 2,
    )


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
    # The first downloaded bar has no prior close in this source.  Using its
    # own close would leak close[T] into a T-open observation and legality.
    preclose[0] = np.nan

    # 构建 xdxr 除权日 → record 映射: YYYYMMDD int → row
    xd_map = {}
    unsupported_dates = set()
    if xdxr_df is not None and not xdxr_df.empty:
        xd = xdxr_df.copy()
        xd['ymd'] = (xd['year'].astype(int) * 10000
                     + xd['month'].astype(int) * 100
                     + xd['day'].astype(int))
        for _, row in xd.iterrows():
            d = int(row['ymd'])
            category = int(row['category'])
            if category == XD_CAT_STANDARD:
                xd_map[d] = row
            elif category != 5:
                unsupported_dates.add(d)

    dates_ymd = _ms_to_date_ymd(time_ms)

    for i in range(1, n):
        d = int(dates_ymd[i])
        if d in unsupported_dates:
            preclose[i] = np.nan
        elif d in xd_map:
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


def _validate_mootdx_page(
    code: str,
    bars: pd.DataFrame,
    *,
    start_pos: int,
) -> np.ndarray:
    required = {"open", "high", "low", "close", "volume", "amount"}
    missing = required.difference(bars.columns)
    has_datetime = (
        "datetime" in bars.columns
        or _datetime_in_index(bars)
        or isinstance(bars.index, pd.DatetimeIndex)
    )
    if missing or not has_datetime:
        raise KlineSourceError(
            code,
            "bars",
            "schema_mismatch",
            f"分页 start={start_pos} missing columns: {sorted(missing)}",
        )
    dates = _bars_to_date_ymd(bars)
    if len(dates) != len(bars) or len(np.unique(dates)) != len(dates):
        raise KlineSourceError(
            code,
            "bars",
            "pagination_incomplete",
            f"分页 start={start_pos} 日期为空或重复",
        )
    return dates


def _fetch_bars_all(mdx, code: str) -> pd.DataFrame | None:
    """Fail-closed full-history pagination for the mootdx 800-row endpoint."""
    all_parts: list[pd.DataFrame] = []
    previous_oldest: int | None = None
    terminal_seen = False
    for start_pos in range(0, MAX_HISTORY_BARS, PAGE_SIZE):
        bars = mdx.bars(symbol=code[:6], frequency=9, start=start_pos, offset=PAGE_SIZE, fq=0)
        if bars is None or bars.empty:
            if start_pos == 0:
                return None
            raise KlineSourceError(
                code,
                "bars",
                "pagination_incomplete",
                f"完整页后 start={start_pos} 突然返回空，无法证明已到历史起点",
            )
        if not isinstance(bars, pd.DataFrame) or len(bars) > PAGE_SIZE:
            raise KlineSourceError(
                code,
                "bars",
                "schema_mismatch",
                f"分页 start={start_pos} 未返回合法 DataFrame/行数",
            )
        if "datetime" not in bars.columns:
            if _datetime_in_index(bars):
                bars = bars.reset_index()
            elif isinstance(bars.index, pd.DatetimeIndex):
                bars = bars.reset_index().rename(
                    columns={bars.index.name or "index": "datetime"}
                )
        dates = _validate_mootdx_page(code, bars, start_pos=start_pos)
        newest = int(dates.max())
        oldest = int(dates.min())
        if previous_oldest is not None and newest >= previous_oldest:
            raise KlineSourceError(
                code,
                "bars",
                "pagination_incomplete",
                f"分页 start={start_pos} 未严格向更早日期推进",
            )
        all_parts.append(bars)
        previous_oldest = oldest
        if len(bars) < PAGE_SIZE:
            probe_start = start_pos + len(bars)
            probe = mdx.bars(
                symbol=code[:6],
                frequency=9,
                start=probe_start,
                offset=1,
                fq=0,
            )
            if probe is not None and not isinstance(probe, pd.DataFrame):
                raise KlineSourceError(
                    code,
                    "bars",
                    "schema_mismatch",
                    f"终点探针 start={probe_start} 未返回 DataFrame",
                )
            if probe is not None and not probe.empty:
                raise KlineSourceError(
                    code,
                    "bars",
                    "pagination_incomplete",
                    f"短页 start={start_pos} 后仍存在更早数据",
                )
            terminal_seen = True
            break
    if not all_parts:
        return None
    if not terminal_seen:
        raise KlineSourceError(
            code,
            "bars",
            "pagination_incomplete",
            f"在 MAX_HISTORY_BARS={MAX_HISTORY_BARS} 内未证明历史终点",
        )
    df = pd.concat(all_parts, ignore_index=True)
    if df.index.name == 'datetime':
        df = df.reset_index(drop=True)
    if "datetime" in df.columns and df["datetime"].duplicated().any():
        raise KlineSourceError(
            code,
            "bars",
            "pagination_incomplete",
            "跨页 datetime 重复",
        )
    return df


def _validate_full_download_range(
    code: str,
    raw: pd.DataFrame,
    *,
    start: str | None,
    end: str | None,
    existing: pd.DataFrame | None,
) -> pd.DataFrame:
    requested_start = (
        pd.to_datetime(start, format="%Y%m%d", errors="raise").date()
        if start is not None
        else None
    )
    requested_end = (
        pd.to_datetime(end, format="%Y%m%d", errors="raise").date()
        if end is not None
        else None
    )
    if (
        requested_start is not None
        and requested_end is not None
        and requested_start > requested_end
    ):
        raise KlineSourceError(
            code,
            "bars",
            "boundary_mismatch",
            f"请求起点 {requested_start} 晚于终点 {requested_end}",
        )
    dates = pd.to_datetime(
        pd.to_numeric(raw["time"], errors="raise"), unit="ms"
    ).dt.date
    keep = pd.Series(True, index=raw.index)
    if requested_start is not None:
        keep &= dates.ge(requested_start)
    if requested_end is not None:
        keep &= dates.le(requested_end)
    selected = raw.loc[keep].reset_index(drop=True)
    if selected.empty:
        raise KlineSourceError(
            code,
            "bars",
            "boundary_mismatch",
            f"源历史与请求区间 {requested_start}..{requested_end} 无交集",
        )
    selected_dates = set(dates[keep])
    if existing is not None:
        if "time" not in existing.columns:
            raise KlineSourceError(
                code,
                "local_kline",
                "schema_mismatch",
                "旧全量 K 线缺少 time",
            )
        old_dates = pd.to_datetime(
            pd.to_numeric(existing["time"], errors="raise"), unit="ms"
        ).dt.date
        old_keep = pd.Series(True, index=existing.index)
        if requested_start is not None:
            old_keep &= old_dates.ge(requested_start)
        if requested_end is not None:
            old_keep &= old_dates.le(requested_end)
        missing_old = sorted(set(old_dates[old_keep]).difference(selected_dates))
        if missing_old:
            shown = ", ".join(day.isoformat() for day in missing_old[:20])
            raise KlineSourceError(
                code,
                "bars",
                "history_regression",
                f"新全量缺少旧文件区间内 {len(missing_old)} 个日期: {shown}",
            )
    return selected


def download(
    mdx,
    codes: list[str],
    start: str,
    end: str,
    *,
    strict: bool = False,
) -> dict[str, pd.DataFrame]:
    """全量下载（分页拉取全历史）并写 parquet。返回 {code: combined_bar_dict}。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total = len(codes)
    t0 = time.time()
    written = 0
    empty = 0
    out: dict = {}

    # 预拉取 xdxr 数据
    xdxr_cache: dict[str, pd.DataFrame | None] = {}
    xdxr_failures = []
    for code in codes:
        try:
            xdxr_cache[code] = _fetch_xdxr_with_retry(mdx, code)
        except Exception as exc:
            xdxr_failures.append((code, exc))
    _warn_failed_codes('xdxr 预取', xdxr_failures)

    failures = []
    for i, code in enumerate(codes):
        if code not in xdxr_cache:
            empty += 1
            continue
        try:
            output_path = RAW_DIR / f'{code}.parquet'
            existing = pd.read_parquet(output_path) if output_path.exists() else None
            raw = _fetch_raw_with_retry(
                mdx,
                code,
                xdxr_cache.get(code),
                start=start,
                end=end,
                existing=existing,
            )
        except Exception as exc:
            failures.append((code, exc))
            empty += 1
            continue
        _write_parquet_atomic(raw, output_path)
        out[code] = _combined_bar_dict(raw)
        written += 1

        if (i + 1) % 500 == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            _log(f"进度 {i+1}/{total} 写入 {written} 无数据 {empty} | {elapsed:.0f}s ETA {eta:.0f}s")

    _log(f"完成: 写入 {written} 无数据 {empty} 用时 {time.time() - t0:.0f}s")
    _warn_failed_codes('全量下载', failures)
    if strict and (xdxr_failures or failures):
        raise KlineBatchError(
            "全量下载",
            [*xdxr_failures, *failures],
            succeeded=tuple(out),
        )
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
    _write_parquet_atomic(df, path)
    return df


def update_full(start: str = START_DEFAULT, end: str | None = None,
                codes: list[str] | None = None, *, strict: bool = False) -> dict:
    """全量拉取到 end（默认今天）。"""
    mdx = _connect_mootdx()
    end = end or datetime.now().strftime('%Y%m%d')
    codes = _all_codes() if codes is None else sorted(codes)
    _log(f"全量 {len(codes)} 只 → {RAW_DIR} (start={start} end={end})")
    return download(mdx, codes, start, end, strict=strict)


def update_recent(days: int, *, anchor_date: date | None = None, collect: bool = False,
                  codes: list[str] | None = None, strict: bool = False) -> dict:
    """刷新最近 days 个交易日并合并进 parquet；新股全量补齐。
    codes 非空时只处理指定股票子集（用于实盘 prefilter 加速）。
    """
    mdx = _connect_mootdx()
    start, end, end_d = resolve_recent_range(days, anchor_date)

    all_codes = sorted(codes) if codes is not None else _all_codes()
    existing = {f.stem for f in RAW_DIR.glob('*.parquet')}
    new_codes = [c for c in all_codes if c not in existing]
    upd_codes = [c for c in all_codes if c in existing]

    _log(f"增量刷新最近 {days} 日 [{start}~{end}] 锚定={end_d.isoformat()}: "
         f"更新 {len(upd_codes)} 只, 新股全量 {len(new_codes)} 只")

    out = {}
    failures = []
    if new_codes:
        _log(f"  处理新股 {len(new_codes)} 只...")
        out.update(download(mdx, new_codes, START_DEFAULT, end, strict=strict))

    # 增量：只拉最近 bars 覆盖 + 合并
    if upd_codes:
        _log(f"  增量合并最近 {len(upd_codes)} 只...")
        t0 = time.time()
        written = empty = 0
        for i, code in enumerate(upd_codes):
            try:
                raw = _fetch_recent_raw_with_retry(mdx, code, days)
            except Exception as exc:
                failures.append((code, exc))
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

    _warn_failed_codes('增量更新', failures)
    if strict and failures:
        raise KlineBatchError(
            "增量更新",
            failures,
            succeeded=tuple(out),
        )
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
