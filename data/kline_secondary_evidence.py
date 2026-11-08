"""Hash-sealed, multi-source evidence for exact executable dates.

The snapshot is an offline T-open calendar reference, not a replacement K-line
feed. Refresh downloads the complete finite Tencent/Baidu evidence contract,
adds three locally cross-confirmed rows, validates one fixed manifest, and only
then atomically replaces the snapshot. Runtime loading never performs network
I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from types import MappingProxyType
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_PATH = (
    ROOT / "data" / "kline_evidence" / "secondary_executable_dates.parquet"
)
DEFAULT_KLINE_DIR = ROOT / "data" / "k-line"
DEFAULT_PRIMARY_EVIDENCE_PATH = (
    ROOT / "data" / "kline_evidence" / "baostock_daily_reference.parquet"
)

SNAPSHOT_SCHEMA = "secondary-executable-dates-v3"
TENCENT_SOURCE = "tencent.none.day.daily-executable-reference"
BAIDU_SOURCE = "baidu.full.daily-executable-reference"
CHAIN_SOURCE = "mootdx.raw+baostock.next-preclose-chain"
SNAPSHOT_COLUMNS = (
    "source",
    "schema_version",
    "stock_code",
    "source_symbol",
    "confirmation_symbol",
    "date",
    "tradestatus",
    "reference_open",
    "source_row_sha256",
    "confirmation_row_sha256",
    "payload_sha256",
)

# Replaced after a complete real-source refresh. These are deliberately total
# seals: recomputing every row hash after tampering still fails here.
EXPECTED_CODE_DATE_SHA256 = (
    "bb47992675decfdcc6a28fa6971c9ee2e41f9455cf3546e3cb189d4df19287b9"
)
EXPECTED_SNAPSHOT_SHA256 = (
    "9d55f5c3999724fd24ef8b36725017d404a585219b09580296d1bd6beb8df10e"
)


def _days(*values: str) -> tuple[str, ...]:
    return tuple(values)


# Tencent rows already present in v2 plus the 42 newly cross-confirmed rows.
TENCENT_EXPLICIT_DATES = MappingProxyType(
    {
        "000004.SZ": _days(
            "1991-01-19", "1991-01-29", "1991-02-01", "1991-02-02",
            "1991-02-09", "1991-02-23", "1991-03-16", "1991-03-23",
            "1991-03-30", "1991-04-01", "1991-05-02", "1991-05-11",
            "1991-05-18", "1991-05-25", "1991-06-06", "1991-06-08",
            "1991-06-15", "1991-06-29", "1991-07-06", "1991-07-13",
            "1991-07-20", "1991-07-27", "1991-08-03", "1991-08-10",
            "1991-08-17", "1991-08-24", "1991-08-31", "1991-09-07",
            "1991-09-14", "1991-09-21", "1991-09-28", "1991-09-29",
            "1991-10-05", "1991-10-09", "1991-10-12", "1991-10-19",
            "1991-10-26", "1991-11-02", "1991-11-09", "1991-11-16",
            "1991-11-23", "1991-11-30", "1991-12-07", "1991-12-14",
            "1991-12-21", "1991-12-28", "1992-02-25", "1992-05-03",
            "1992-05-05", "1992-10-04", "1993-01-03", "1993-08-09",
        ),
        "000012.SZ": _days(
            "1992-05-03", "1992-10-04", "1993-01-03", "1993-06-05",
            "1993-06-19", "1993-07-03", "1993-07-17", "1993-08-07",
            "1993-08-21",
        ),
        "001872.SZ": _days("2012-09-10"),
        "300029.SZ": _days("2026-07-09"),
        "600604.SH": _days("1993-01-04"),
        "600605.SH": _days("1993-01-04"),
        "600606.SH": _days("1993-01-04"),
        "600608.SH": _days("1993-01-04"),
        "600618.SH": _days("1992-11-17", "1993-01-04"),
        "600619.SH": _days("1992-11-18", "1993-01-04"),
        "600620.SH": _days("1992-11-19", "1993-01-04"),
        "600636.SH": _days("1994-04-07"),
        "600651.SH": _days(
            "1990-12-25", "1991-01-15", "1991-01-16", "1991-01-17",
            "1991-01-21", "1991-01-25", "1991-06-06", "1991-07-03",
            "1991-12-25", "1992-01-14", "1992-01-17", "1992-03-05",
            "1992-03-11",
        ),
        "600653.SH": _days(
            "1991-03-28", "1991-06-10", "1991-06-24", "1993-12-23",
        ),
        "600654.SH": _days("1991-01-02", "1992-01-02", "1993-01-04"),
        "600665.SH": _days("1993-12-23"),
    }
)
TENCENT_SOURCE_SYMBOLS = MappingProxyType(
    {
        **{
            code: ("sh" if code.endswith(".SH") else "sz") + code[:6]
            for code in TENCENT_EXPLICIT_DATES
        },
        "001872.SZ": "sz000022",
    }
)

# The reused 600018 code is too large to embed as a literal. Its complete
# predecessor date set is pinned by count, boundaries and date-set hash.
TENCENT_DATE_SET_SUMMARIES = MappingProxyType(
    {
        "600018.SH": MappingProxyType(
            {
                "count": 1448,
                "first": "2000-07-19",
                "last": "2006-09-25",
                "sha256": (
                    "84e618e5115678c41b208c6766b14bc9"
                    "be0424131eb086d256fddac6b5a7941f"
                ),
            }
        )
    }
)
TENCENT_SUMMARY_SYMBOLS = MappingProxyType({"600018.SH": "sh600018"})
TENCENT_QUERY_WINDOWS = MappingProxyType(
    {
        "600018.SH": (
            ("2000-07-19", "2000-12-31"),
            ("2001-01-01", "2001-12-31"),
            ("2002-01-01", "2002-12-31"),
            ("2003-01-01", "2003-12-31"),
            ("2004-01-01", "2004-12-31"),
            ("2005-01-01", "2005-12-31"),
            ("2006-01-01", "2006-10-25"),
        )
    }
)
REQUIRED_EXECUTABLE_DATES = MappingProxyType(
    {"600018.SH": _days("2001-08-16")}
)

# Baidu full-history stable core rows. Request-sensitive MA columns are never
# hashed. Finite fields plus positive volume and amount prove exact executable
# existence (tradestatus="1"). Baidu's adjusted prices are diagnostic hash
# material only: entity-replacement history can make reference_open negative,
# and no Baidu price from this module may be copied into canonical K.
BAIDU_RESTORED_6 = MappingProxyType(
    {
        "000005.SZ": _days("1991-01-14", "1992-02-03", "1992-02-07"),
        "000023.SZ": _days("1995-04-28"),
        "000540.SZ": _days("1995-04-28"),
        "600093.SH": _days("1999-01-06"),
    }
)
BAIDU_EXPLICIT_DATES = MappingProxyType(
    {**BAIDU_RESTORED_6, "001872.SZ": _days("1993-07-15")}
)
BAIDU_SOURCE_SYMBOLS = MappingProxyType(
    {code: code[:6] for code in BAIDU_EXPLICIT_DATES}
)

# Local rows that primary evidence marked normalization_applied=True. The next
# direct Baostock row's pre-close must equal the local target close exactly.
LOCAL_PRECLOSE_CHAINS = MappingProxyType(
    {
        "600656.SH": MappingProxyType(
            {"date": "1996-07-29", "next_date": "1996-07-30"}
        ),
        "600799.SH": MappingProxyType(
            {"date": "1999-11-11", "next_date": "1999-11-12"}
        ),
        "600832.SH": MappingProxyType(
            {"date": "1998-03-04", "next_date": "1998-03-05"}
        ),
    }
)

# First expansion batch: 42 Tencent rows plus 001872/1993-07-15 from Baidu.
CROSSCONFIRMED_MISSING_43 = MappingProxyType(
    {
        "000004.SZ": _days(
            "1991-01-29", "1991-02-01", "1991-04-01", "1991-05-02",
            "1991-06-06", "1991-10-09", "1992-02-25", "1992-05-05",
            "1993-08-09",
        ),
        "001872.SZ": _days("1993-07-15"),
        "300029.SZ": _days("2026-07-09"),
        "600604.SH": _days("1993-01-04"),
        "600605.SH": _days("1993-01-04"),
        "600606.SH": _days("1993-01-04"),
        "600608.SH": _days("1993-01-04"),
        "600618.SH": _days("1992-11-17", "1993-01-04"),
        "600619.SH": _days("1992-11-18", "1993-01-04"),
        "600620.SH": _days("1992-11-19", "1993-01-04"),
        "600636.SH": _days("1994-04-07"),
        "600651.SH": _days(
            "1990-12-25", "1991-01-15", "1991-01-16", "1991-01-17",
            "1991-01-21", "1991-01-25", "1991-06-06", "1991-07-03",
            "1991-12-25", "1992-01-14", "1992-01-17", "1992-03-05",
            "1992-03-11",
        ),
        "600653.SH": _days(
            "1991-03-28", "1991-06-10", "1991-06-24", "1993-12-23",
        ),
        "600654.SH": _days("1991-01-02", "1992-01-02", "1993-01-04"),
        "600665.SH": _days("1993-12-23"),
    }
)
CROSSCONFIRMED_MISSING_43_SHA256 = (
    "88b39b7580c1bdd264510b76dc43d981890000a2f126ed23a945b0a29094eeec"
)

# Exact dates whose canonical K history is repaired by the separate finite
# repair snapshot. This contract intentionally contains dates only: Tencent
# has no amount/pre-close here and Baidu prices are adjusted (sometimes even
# negative after entity replacement), so neither is a safe canonical OHLCVA
# write source.
HISTORY_REPAIR_49 = MappingProxyType(
    {
        code: tuple(
            sorted(
                set(CROSSCONFIRMED_MISSING_43.get(code, ()))
                | set(BAIDU_RESTORED_6.get(code, ()))
            )
        )
        for code in sorted(
            set(CROSSCONFIRMED_MISSING_43) | set(BAIDU_RESTORED_6)
        )
    }
)
if sum(map(len, HISTORY_REPAIR_49.values())) != 49:
    raise RuntimeError("49-row history-repair date contract mismatch")

KNOWN_ABSENT_DATES = MappingProxyType(
    {
        "000004.SZ": _days("1991-11-17"),
        "000012.SZ": _days("1993-05-01", "1993-05-02"),
    }
)
BAIDU_CORE_KEYS = (
    "timestamp", "time", "open", "close", "volume", "high", "low",
    "amount", "range", "ratio", "turnoverratio", "preClose",
)


def _date_set_sha256(values) -> str:
    normalized = sorted(
        {pd.Timestamp(value).strftime("%Y-%m-%d") for value in values}
    )
    return hashlib.sha256("\n".join(normalized).encode()).hexdigest()


def _date_set_summary(values) -> dict[str, object]:
    normalized = sorted(
        {pd.Timestamp(value).strftime("%Y-%m-%d") for value in values}
    )
    return {
        "count": len(normalized),
        "first": normalized[0] if normalized else None,
        "last": normalized[-1] if normalized else None,
        "sha256": _date_set_sha256(normalized),
    }


def _code_date_sha256(items) -> str:
    """Hash sorted uppercase ``CODE|YYYY-MM-DD`` rows joined by LF.

    There is deliberately no trailing LF; this is the canonical form used by
    both the 43-row audit seal and the complete snapshot date-set seal.
    """
    normalized = sorted(
        {
            f"{str(code).strip().upper()}|"
            f"{pd.Timestamp(day).strftime('%Y-%m-%d')}"
            for code, day in items
        }
    )
    return hashlib.sha256("\n".join(normalized).encode()).hexdigest()


def _mapping_code_dates(mapping) -> tuple[tuple[str, str], ...]:
    return tuple(
        (code, day) for code, dates in mapping.items() for day in dates
    )


_actual_crossconfirmed_43_sha256 = _code_date_sha256(
    _mapping_code_dates(CROSSCONFIRMED_MISSING_43)
)
if _actual_crossconfirmed_43_sha256 != CROSSCONFIRMED_MISSING_43_SHA256:
    raise RuntimeError("43-row cross-confirmation contract hash mismatch")


def _canonical_source_row_hash(row) -> str:
    raw = [str(value).strip() for value in row]
    return hashlib.sha256(
        json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_numeric(value) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(numeric) else format(float(numeric), ".17g")


def _payload_sha256(row: pd.Series | dict[str, object]) -> str:
    values: dict[str, str] = {}
    for field in SNAPSHOT_COLUMNS:
        if field == "payload_sha256":
            continue
        value = row[field]
        values[field] = (
            _canonical_numeric(value)
            if field == "reference_open"
            else ("" if pd.isna(value) else str(value).strip())
        )
    return hashlib.sha256(
        json.dumps(
            values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _manifest_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["stock_code", "date"], kind="stable")
    return hashlib.sha256(
        "\n".join(ordered["payload_sha256"].astype(str)).encode()
    ).hexdigest()


def _explicit_expected_dates() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for mapping in (TENCENT_EXPLICIT_DATES, BAIDU_EXPLICIT_DATES):
        for code, dates in mapping.items():
            result.setdefault(code, set()).update(dates)
    for code, chain in LOCAL_PRECLOSE_CHAINS.items():
        result.setdefault(code, set()).add(str(chain["date"]))
    return result


def _expected_source_contract(code: str, day: str) -> tuple[str, str, str]:
    if code in TENCENT_DATE_SET_SUMMARIES:
        return TENCENT_SOURCE, TENCENT_SUMMARY_SYMBOLS[code], ""
    if day in TENCENT_EXPLICIT_DATES.get(code, ()):
        return TENCENT_SOURCE, TENCENT_SOURCE_SYMBOLS[code], ""
    if day in BAIDU_EXPLICIT_DATES.get(code, ()):
        return BAIDU_SOURCE, BAIDU_SOURCE_SYMBOLS[code], ""
    chain = LOCAL_PRECLOSE_CHAINS.get(code)
    if chain is not None and day == chain["date"]:
        return CHAIN_SOURCE, code, "sh." + code[:6]
    raise ValueError(f"次级逐日证据未声明 source contract: {code}/{day}")


def _validate_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
    if list(raw.columns) != list(SNAPSHOT_COLUMNS) or raw.empty:
        raise ValueError("次级逐日证据 snapshot schema/内容不完整")
    frame = raw.copy()
    for field in (
        "source", "schema_version", "stock_code", "source_symbol",
        "confirmation_symbol", "date", "tradestatus", "source_row_sha256",
        "confirmation_row_sha256", "payload_sha256",
    ):
        frame[field] = frame[field].fillna("").astype(str).str.strip()
    frame["stock_code"] = frame["stock_code"].str.upper()
    frame["date"] = pd.to_datetime(
        frame["date"], format="%Y-%m-%d", errors="raise"
    ).dt.strftime("%Y-%m-%d")
    frame["reference_open"] = pd.to_numeric(
        frame["reference_open"], errors="raise"
    ).astype(np.float64)
    if not frame["schema_version"].eq(SNAPSHOT_SCHEMA).all():
        raise ValueError("次级逐日证据 schema_version 不兼容")
    if not frame["tradestatus"].eq("1").all():
        raise ValueError("次级逐日证据仅允许明确可成交行")
    if not np.isfinite(frame["reference_open"].to_numpy()).all():
        raise ValueError("次级逐日证据 reference_open 非有限值")
    if frame.duplicated(["stock_code", "date"]).any():
        raise ValueError("次级逐日证据 stock_code/date 重复")
    for field in ("source_row_sha256", "payload_sha256"):
        frame[field] = frame[field].str.lower()
        if not frame[field].str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"次级逐日证据 {field} 非法")
    frame["confirmation_row_sha256"] = frame[
        "confirmation_row_sha256"
    ].str.lower()
    valid_confirmation = frame["confirmation_row_sha256"].eq("") | frame[
        "confirmation_row_sha256"
    ].str.fullmatch(r"[0-9a-f]{64}")
    if not valid_confirmation.all():
        raise ValueError("次级逐日证据 confirmation_row_sha256 非法")

    explicit = _explicit_expected_dates()
    expected_codes = set(explicit) | set(TENCENT_DATE_SET_SUMMARIES)
    actual_codes = set(frame["stock_code"])
    if actual_codes != expected_codes:
        raise ValueError(
            "次级逐日证据代码集合不完整: "
            f"missing={sorted(expected_codes - actual_codes)} "
            f"extra={sorted(actual_codes - expected_codes)}"
        )
    for code, rows in frame.groupby("stock_code", sort=False):
        actual_dates = set(rows["date"])
        if code in TENCENT_DATE_SET_SUMMARIES:
            expected_summary = dict(TENCENT_DATE_SET_SUMMARIES[code])
            actual_summary = _date_set_summary(actual_dates)
            if actual_summary != expected_summary:
                raise ValueError(
                    f"次级逐日证据日期集合摘要不一致: {code} "
                    f"expected={expected_summary} actual={actual_summary}"
                )
            missing_required = sorted(
                set(REQUIRED_EXECUTABLE_DATES.get(code, ())) - actual_dates
            )
            if missing_required:
                raise ValueError(
                    f"次级逐日证据缺少独立源确认日期: {code}/{missing_required}"
                )
        elif actual_dates != explicit[code]:
            raise ValueError(
                f"次级逐日证据精确日期集合不一致: {code} "
                f"missing={sorted(explicit[code] - actual_dates)} "
                f"extra={sorted(actual_dates - explicit[code])}"
            )

    for _, row in frame.iterrows():
        expected = _expected_source_contract(row["stock_code"], row["date"])
        actual = (
            row["source"], row["source_symbol"], row["confirmation_symbol"]
        )
        if actual != expected:
            raise ValueError(
                "次级逐日证据 source/source_symbol 不受信任: "
                f"{row['stock_code']}/{row['date']} expected={expected} actual={actual}"
            )
        requires_confirmation = row["source"] == CHAIN_SOURCE
        if requires_confirmation != bool(row["confirmation_row_sha256"]):
            raise ValueError(
                f"次级逐日证据 confirmation hash 契约不一致: "
                f"{row['stock_code']}/{row['date']}"
            )

    if _code_date_sha256(zip(frame["stock_code"], frame["date"])) != (
        EXPECTED_CODE_DATE_SHA256
    ):
        raise ValueError("次级逐日证据完整 code/date SHA-256 不一致")
    expected_hashes = pd.Series(
        [_payload_sha256(row) for _, row in frame.iterrows()], index=frame.index
    )
    if not frame["payload_sha256"].eq(expected_hashes).all():
        raise ValueError("次级逐日证据 payload_sha256 校验失败")
    if _manifest_sha256(frame) != EXPECTED_SNAPSHOT_SHA256:
        raise ValueError("次级逐日证据 manifest SHA-256 与代码封存值不一致")
    return frame.loc[:, SNAPSHOT_COLUMNS].sort_values(
        ["stock_code", "date"], kind="stable"
    ).reset_index(drop=True)


def load_verified_daily_evidence(
    path: Path = DEFAULT_SNAPSHOT_PATH,
) -> pd.DataFrame:
    """Load and strictly verify v3 without network access."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"次级逐日证据不存在: {path}")
    frame = _validate_snapshot(pd.read_parquet(path))
    frame["date"] = pd.to_datetime(frame["date"]).values.astype("datetime64[D]")
    return frame


def verified_history_repair_evidence(
    path: Path = DEFAULT_SNAPSHOT_PATH,
) -> pd.DataFrame:
    """Return exactly the 49 secondary rows used to audit K-history repairs.

    The returned rows retain every v3 provenance/hash column. They prove the
    dates were executable; they are not a canonical OHLCVA write source.
    """
    frame = load_verified_daily_evidence(path)
    expected = {
        (code, pd.Timestamp(day).strftime("%Y-%m-%d"))
        for code, day in _mapping_code_dates(HISTORY_REPAIR_49)
    }
    normalized_dates = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    selected = frame[
        [
            (code, day) in expected
            for code, day in zip(frame["stock_code"], normalized_dates)
        ]
    ].copy()
    actual = {
        (code, pd.Timestamp(day).strftime("%Y-%m-%d"))
        for code, day in zip(selected["stock_code"], selected["date"])
    }
    if actual != expected or len(selected) != 49:
        raise ValueError(
            "次级逐日证据 history-repair 49-row contract 不完整: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    return selected.loc[:, SNAPSHOT_COLUMNS].sort_values(
        ["stock_code", "date"], kind="stable"
    ).reset_index(drop=True)


def verified_tencent_history_repair_ohlcv(
    code: str,
    day: str,
    *,
    evidence_path: Path = DEFAULT_SNAPSHOT_PATH,
    timeout: float = 30.0,
) -> dict[str, object]:
    """Re-download one sealed Tencent repair row without publishing it.

    The requested pair must belong to ``HISTORY_REPAIR_49`` and be assigned to
    Tencent by the fixed source contract. The live raw D/O/C/H/L/V row must
    exactly reproduce the source-row hash already sealed in the verified v3
    snapshot. Returned prices are audit material, not a canonical K write.
    """
    normalized_code = str(code).strip().upper()
    normalized_day = pd.Timestamp(day).strftime("%Y-%m-%d")
    if normalized_day not in HISTORY_REPAIR_49.get(normalized_code, ()):
        raise ValueError(
            "Tencent history repair 不在固定 49-row contract: "
            f"{normalized_code}/{normalized_day}"
        )
    expected_source = _expected_source_contract(
        normalized_code, normalized_day
    )
    if expected_source[0] != TENCENT_SOURCE:
        raise ValueError(
            "Tencent history repair 日期不属于 Tencent source: "
            f"{normalized_code}/{normalized_day}"
        )
    symbol = expected_source[1]
    sealed = verified_history_repair_evidence(Path(evidence_path))
    sealed_row = sealed[
        sealed["stock_code"].eq(normalized_code)
        & sealed["date"].eq(normalized_day)
    ]
    if len(sealed_row) != 1 or sealed_row.iloc[0]["source"] != TENCENT_SOURCE:
        raise ValueError(
            "Tencent history repair 封存行不唯一: "
            f"{normalized_code}/{normalized_day}"
        )
    downloaded = _download_tencent_rows(
        symbol, normalized_day, normalized_day, timeout=timeout
    )
    by_date = _rows_by_date(downloaded, code=normalized_code)
    if set(by_date) != {normalized_day}:
        raise ValueError(
            "Tencent history repair 下载日期不精确: "
            f"{normalized_code}/{normalized_day} actual={sorted(by_date)}"
        )
    raw_row = by_date[normalized_day]
    _validate_tencent_executable_row(
        raw_row, code=normalized_code, day=normalized_day
    )
    source_hash = _canonical_source_row_hash(raw_row)
    if source_hash != sealed_row.iloc[0]["source_row_sha256"]:
        raise ValueError(
            "Tencent history repair 原始行 hash 与 v3 封存不一致: "
            f"{normalized_code}/{normalized_day}"
        )
    return {
        "date": str(raw_row[0]),
        "open": raw_row[1],
        "close": raw_row[2],
        "high": raw_row[3],
        "low": raw_row[4],
        "volume": raw_row[5],
        "source_symbol": symbol,
        "source_row_sha256": source_hash,
    }


def covered_local_dates(
    code: str, local_dates, *, path: Path = DEFAULT_SNAPSHOT_PATH
) -> frozenset[date]:
    """Return only exact requested dates covered by the sealed snapshot."""
    normalized = str(code).strip().upper()
    expected_codes = set(_explicit_expected_dates()) | set(
        TENCENT_DATE_SET_SUMMARIES
    )
    if normalized not in expected_codes:
        return frozenset()
    requested = frozenset(local_dates)
    evidence = load_verified_daily_evidence(path)
    rows = evidence[evidence["stock_code"].eq(normalized)]
    covered = frozenset(pd.Timestamp(value).date() for value in rows["date"])
    return frozenset(value for value in requested if value in covered)


def _download_tencent_rows(
    symbol: str, start: str, end: str, *, timeout: float
) -> list[list[object]]:
    param = quote(f"{symbol},day,{start},{end},640,none", safe=",")
    url = f"https://ifzq.gtimg.cn/appstock/app/fqkline/get?param={param}"
    request = Request(url, headers={"User-Agent": "WBR-offline-data-updater/1"})
    with urlopen(request, timeout=timeout) as response:
        if getattr(response, "status", 200) != 200:
            raise RuntimeError(f"Tencent K 线 HTTP {response.status}")
        payload = json.loads(response.read().decode("utf-8"))
    data = (payload.get("data") or {}).get(symbol)
    if not isinstance(data, dict):
        raise ValueError(f"Tencent K 线缺少目标 symbol: {symbol}")
    rows = data.get("day")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Tencent K 线目标区间返回空表: {symbol}")
    normalized: list[list[object]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            raise ValueError(f"Tencent K 线行 schema 非法: {symbol}")
        day = pd.Timestamp(str(row[0])).strftime("%Y-%m-%d")
        if not start <= day <= end:
            raise ValueError(f"Tencent K 线返回查询区间外日期: {symbol}/{day}")
        if day in seen:
            raise ValueError(f"Tencent K 线日期重复: {symbol}/{day}")
        seen.add(day)
        normalized.append(row)
    return normalized


def _rows_by_date(rows, *, code: str, date_index: int = 0):
    by_date: dict[str, list[object]] = {}
    for row in rows:
        day = pd.Timestamp(str(row[date_index])).strftime("%Y-%m-%d")
        if day in by_date:
            raise ValueError(f"逐日证据日期重复: {code}/{day}")
        by_date[day] = row
    return by_date


def _annual_query_windows(dates) -> tuple[tuple[str, str], ...]:
    by_year: dict[str, list[str]] = {}
    for value in dates:
        day = pd.Timestamp(value).strftime("%Y-%m-%d")
        by_year.setdefault(day[:4], []).append(day)
    return tuple(
        (min(values), max(values)) for _, values in sorted(by_year.items())
    )


def _record(
    *, source: str, code: str, symbol: str, day: str,
    reference_open: float, source_hash: str,
    confirmation_symbol: str = "", confirmation_hash: str = "",
) -> dict[str, object]:
    record: dict[str, object] = {
        "source": source,
        "schema_version": SNAPSHOT_SCHEMA,
        "stock_code": code,
        "source_symbol": symbol,
        "confirmation_symbol": confirmation_symbol,
        "date": day,
        "tradestatus": "1",
        "reference_open": reference_open,
        "source_row_sha256": source_hash,
        "confirmation_row_sha256": confirmation_hash,
    }
    record["payload_sha256"] = _payload_sha256(record)
    return record


def _validate_tencent_executable_row(
    row: list[object], *, code: str, day: str
) -> None:
    """Require a real positive unadjusted OHLCV row before sealing it."""
    ohlcv = pd.to_numeric(pd.Series(row[1:6]), errors="coerce").to_numpy(
        dtype=np.float64
    )
    if len(ohlcv) != 5 or not np.isfinite(ohlcv).all() or not (ohlcv > 0).all():
        raise ValueError(f"Tencent K 线非正 OHLCV 行: {code}/{day}")


def _download_tencent_contract(*, timeout: float) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for code, expected_dates in TENCENT_EXPLICIT_DATES.items():
        symbol = TENCENT_SOURCE_SYMBOLS[code]
        audit_dates = KNOWN_ABSENT_DATES.get(code, ())
        rows: list[list[object]] = []
        for start, end in _annual_query_windows((*expected_dates, *audit_dates)):
            rows.extend(_download_tencent_rows(symbol, start, end, timeout=timeout))
        by_date = _rows_by_date(rows, code=code)
        missing = sorted(set(expected_dates) - set(by_date))
        if missing:
            raise ValueError(f"Tencent K 线未覆盖封存日期: {code}/{missing}")
        unexpected = sorted(set(audit_dates) & set(by_date))
        if unexpected:
            raise ValueError(f"Tencent K 线污染日期不再缺失: {code}/{unexpected}")
        for day in expected_dates:
            source_row = by_date[day]
            _validate_tencent_executable_row(source_row, code=code, day=day)
            records.append(
                _record(
                    source=TENCENT_SOURCE, code=code, symbol=symbol, day=day,
                    reference_open=float(source_row[1]),
                    source_hash=_canonical_source_row_hash(source_row),
                )
            )

    for code, expected_summary in TENCENT_DATE_SET_SUMMARIES.items():
        symbol = TENCENT_SUMMARY_SYMBOLS[code]
        rows = []
        for start, end in TENCENT_QUERY_WINDOWS[code]:
            rows.extend(_download_tencent_rows(symbol, start, end, timeout=timeout))
        by_date = _rows_by_date(rows, code=code)
        actual_summary = _date_set_summary(by_date)
        if actual_summary != dict(expected_summary):
            raise ValueError(
                f"Tencent K 线封存日期集合摘要不一致: {code} "
                f"expected={dict(expected_summary)} actual={actual_summary}"
            )
        missing_required = sorted(
            set(REQUIRED_EXECUTABLE_DATES.get(code, ())) - set(by_date)
        )
        if missing_required:
            raise ValueError(
                f"Tencent K 线缺少独立源确认日期: {code}/{missing_required}"
            )
        for day in sorted(by_date):
            source_row = by_date[day]
            _validate_tencent_executable_row(source_row, code=code, day=day)
            records.append(
                _record(
                    source=TENCENT_SOURCE, code=code, symbol=symbol, day=day,
                    reference_open=float(source_row[1]),
                    source_hash=_canonical_source_row_hash(source_row),
                )
            )
    return records


def _download_baidu_rows(
    symbol: str, expected_dates, *, timeout: float
) -> dict[str, list[object]]:
    earliest = min(pd.Timestamp(day) for day in expected_dates)
    params = {
        "all": "1", "isIndex": "false", "isBk": "false",
        "isBlock": "false", "isFutures": "false", "isStock": "true",
        "newFormat": "1", "group": "quotation_kline_ab",
        "finClientType": "pc", "code": symbol,
        "start_time": str(int(earliest.timestamp())), "ktype": "1",
    }
    response = requests.get(
        "https://finance.pae.baidu.com/selfselect/getstockquotation",
        params=params,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/vnd.finance-web.v1+json",
            "Origin": "https://gushitong.baidu.com",
            "Referer": "https://gushitong.baidu.com/",
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Baidu K 线 HTTP {response.status_code}")
    payload = response.json()
    if str(payload.get("ResultCode", "")) != "0":
        raise ValueError(f"Baidu K 线拒绝 symbol: {symbol}")
    market = ((payload.get("Result") or {}).get("newMarketData") or {})
    keys = tuple(market.get("keys") or ())
    if keys[: len(BAIDU_CORE_KEYS)] != BAIDU_CORE_KEYS:
        raise ValueError(
            f"Baidu K 线 stable core schema 变化: {symbol}/{keys[:12]}"
        )
    raw_rows = str(market.get("marketData") or "")
    if not raw_rows:
        raise ValueError(f"Baidu K 线目标区间返回空表: {symbol}")
    rows = [item.split(",") for item in raw_rows.split(";") if item]
    if any(len(row) < len(BAIDU_CORE_KEYS) for row in rows):
        raise ValueError(f"Baidu K 线行 schema 非法: {symbol}")
    by_date = _rows_by_date(rows, code=symbol, date_index=1)
    missing = sorted(set(expected_dates) - set(by_date))
    if missing:
        raise ValueError(f"Baidu K 线未覆盖封存日期: {symbol}/{missing}")
    return by_date


def _download_baidu_contract(*, timeout: float) -> list[dict[str, object]]:
    """Download exact-date existence evidence, never canonical K prices.

    ``reference_open`` deliberately preserves Baidu's adjusted diagnostic even
    when it is negative. Executability is established by a finite stable core
    with strictly positive volume and amount; the emitted ``tradestatus='1'``
    only carries that existence meaning.
    """
    records: list[dict[str, object]] = []
    for code, expected_dates in BAIDU_EXPLICIT_DATES.items():
        symbol = BAIDU_SOURCE_SYMBOLS[code]
        by_date = _download_baidu_rows(symbol, expected_dates, timeout=timeout)
        for day in expected_dates:
            core = by_date[day][: len(BAIDU_CORE_KEYS)]
            numeric = [float(core[index]) for index in (2, 3, 4, 5, 6, 7)]
            if (
                not np.isfinite(numeric).all()
                or float(core[4]) <= 0
                or float(core[7]) <= 0
            ):
                raise ValueError(f"Baidu K 线非正成交行: {code}/{day}")
            records.append(
                _record(
                    source=BAIDU_SOURCE, code=code, symbol=symbol, day=day,
                    reference_open=float(core[2]),
                    source_hash=_canonical_source_row_hash(core),
                )
            )
    return records


def _local_day_frame(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    required = {
        "time", "open", "high", "low", "close", "volume", "amount", "preClose"
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"本地 K 缺列: {path.name}/{sorted(missing)}")
    frame = raw.copy()
    frame["date"] = pd.to_datetime(
        pd.to_numeric(frame["time"], errors="raise"), unit="ms"
    ).dt.strftime("%Y-%m-%d")
    if frame.duplicated("date").any():
        raise ValueError(f"本地 K 日期重复: {path.name}")
    return frame.set_index("date", drop=False)


def _download_chain_contract(
    *, kline_dir: Path, primary_evidence_path: Path
) -> list[dict[str, object]]:
    from data.kline_mootdx import (
        _kline_row_state_sha256,
        load_baostock_daily_evidence,
    )

    primary = load_baostock_daily_evidence(Path(primary_evidence_path)).copy()
    primary["date"] = pd.to_datetime(primary["date"]).dt.strftime("%Y-%m-%d")
    records: list[dict[str, object]] = []
    local_fields = (
        "time", "open", "high", "low", "close", "volume", "amount", "preClose"
    )
    confirmation_fields = (
        # Hash only the stable direct next-row core. Primary snapshot schema,
        # lineage hashes and correction-audit columns may evolve without
        # changing this independent pre-close fact.
        "baostock_code", "date", "tradestatus", "reference_open",
        "reference_volume", "reference_amount", "direct_preclose",
    )
    for code, contract in LOCAL_PRECLOSE_CHAINS.items():
        day, next_day = str(contract["date"]), str(contract["next_date"])
        local = _local_day_frame(Path(kline_dir) / f"{code}.parquet")
        if day not in local.index:
            raise ValueError(f"本地 K 缺少 chain target: {code}/{day}")
        target = local.loc[day]
        if isinstance(target, pd.DataFrame):
            raise ValueError(f"本地 K chain target 重复: {code}/{day}")
        positive = pd.to_numeric(
            target[["open", "high", "low", "close", "volume", "amount"]],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(positive).all() or not (positive > 0).all():
            raise ValueError(f"本地 K chain target 非正 OHLCVA: {code}/{day}")
        target_primary = primary[
            primary["stock_code"].eq(code) & primary["date"].eq(day)
        ]
        next_primary = primary[
            primary["stock_code"].eq(code) & primary["date"].eq(next_day)
        ]
        if len(target_primary) != 1 or len(next_primary) != 1:
            raise ValueError(f"Baostock chain 行不唯一: {code}/{day}->{next_day}")
        target_primary_row, next_primary_row = (
            target_primary.iloc[0], next_primary.iloc[0]
        )
        required_primary_fields = {
            "source_tradestatus", "tradestatus", "status_correction",
            "normalization_applied", "applied_k_sha256",
        }
        missing_primary_fields = required_primary_fields - set(primary.columns)
        if missing_primary_fields:
            raise ValueError(
                "Baostock chain evidence 缺少绑定字段: "
                f"{sorted(missing_primary_fields)}"
            )
        if not bool(target_primary_row["normalization_applied"]):
            raise ValueError(f"Baostock target 不再是 normalized candidate: {code}/{day}")
        if (
            str(target_primary_row["source_tradestatus"]) != "1"
            or str(target_primary_row["tradestatus"]) != "1"
            or str(target_primary_row["status_correction"]).strip()
        ):
            raise ValueError(
                f"Baostock target 非原始 direct executable: {code}/{day}"
            )
        actual_target_state_hash = _kline_row_state_sha256(
            pd.Timestamp(day).date(), target
        )
        if str(target_primary_row["applied_k_sha256"]).lower() != (
            actual_target_state_hash
        ):
            raise ValueError(
                f"Baostock target applied K 状态 hash 不一致: {code}/{day}"
            )
        if (
            bool(next_primary_row["normalization_applied"])
            or str(next_primary_row["source_tradestatus"]) != "1"
            or str(next_primary_row["tradestatus"]) != "1"
            or str(next_primary_row["status_correction"]).strip()
        ):
            raise ValueError(f"Baostock next row 非 direct executable: {code}/{next_day}")
        close = float(target["close"])
        next_preclose = float(next_primary_row["direct_preclose"])
        if close != next_preclose:
            raise ValueError(
                f"Baostock next direct_preclose chain 断裂: {code}/{day} "
                f"local_close={close} next_preclose={next_preclose}"
            )
        records.append(
            _record(
                source=CHAIN_SOURCE, code=code, symbol=code, day=day,
                confirmation_symbol=str(next_primary_row["baostock_code"]),
                reference_open=float(target["open"]),
                source_hash=_canonical_source_row_hash(
                    [target[field] for field in local_fields]
                ),
                confirmation_hash=_canonical_source_row_hash(
                    [next_primary_row[field] for field in confirmation_fields]
                ),
            )
        )
    return records


def _download_snapshot_frame(
    *, timeout: float = 30.0, kline_dir: Path = DEFAULT_KLINE_DIR,
    primary_evidence_path: Path = DEFAULT_PRIMARY_EVIDENCE_PATH,
) -> pd.DataFrame:
    # No staging until all independent batches finish: one failure means zero
    # publication.
    records = _download_tencent_contract(timeout=timeout)
    records.extend(_download_baidu_contract(timeout=timeout))
    records.extend(
        _download_chain_contract(
            kline_dir=Path(kline_dir),
            primary_evidence_path=Path(primary_evidence_path),
        )
    )
    return pd.DataFrame(records, columns=SNAPSHOT_COLUMNS)


def _download_snapshot(
    *, timeout: float = 30.0, kline_dir: Path = DEFAULT_KLINE_DIR,
    primary_evidence_path: Path = DEFAULT_PRIMARY_EVIDENCE_PATH,
) -> pd.DataFrame:
    return _validate_snapshot(
        _download_snapshot_frame(
            timeout=timeout, kline_dir=kline_dir,
            primary_evidence_path=primary_evidence_path,
        )
    )


def refresh_verified_daily_evidence(
    path: Path = DEFAULT_SNAPSHOT_PATH,
    *, timeout: float = 30.0, kline_dir: Path = DEFAULT_KLINE_DIR,
    primary_evidence_path: Path = DEFAULT_PRIMARY_EVIDENCE_PATH,
) -> pd.DataFrame:
    """Refresh the complete v3 contract and atomically publish one parquet."""
    path = Path(path)
    frame = _download_snapshot(
        timeout=timeout, kline_dir=kline_dir,
        primary_evidence_path=primary_evidence_path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp.parquet"
    )
    try:
        frame.to_parquet(staged, index=False)
        verified = _validate_snapshot(pd.read_parquet(staged))
        pd.testing.assert_frame_equal(verified, frame, check_exact=True)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)
    return frame


__all__ = [
    "BAIDU_EXPLICIT_DATES", "BAIDU_RESTORED_6", "CHAIN_SOURCE",
    "CROSSCONFIRMED_MISSING_43",
    "DEFAULT_SNAPSHOT_PATH", "EXPECTED_CODE_DATE_SHA256",
    "EXPECTED_SNAPSHOT_SHA256", "HISTORY_REPAIR_49",
    "LOCAL_PRECLOSE_CHAINS",
    "REQUIRED_EXECUTABLE_DATES", "SNAPSHOT_COLUMNS", "SNAPSHOT_SCHEMA",
    "TENCENT_DATE_SET_SUMMARIES", "TENCENT_EXPLICIT_DATES",
    "covered_local_dates", "load_verified_daily_evidence",
    "refresh_verified_daily_evidence", "verified_history_repair_evidence",
    "verified_tencent_history_repair_ohlcv",
]
