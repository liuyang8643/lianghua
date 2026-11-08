"""Fail-closed, evidence-backed repairs for a finite K-line defect set.

A refresh binds 6 archived mootdx rows, 42 archived QMT rows and one targeted
Baostock row to exact secondary executable-date evidence.  The remaining
legacy repairs stay Baostock- or code-owned.  Every source row, unit conversion,
turnover decision and required successor ``preClose`` is hash sealed; normal
replay is fully offline and publishes the snapshot plus all changed K files in
one rollback-protected transaction.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

import numpy as np
import pandas as pd

from data.kline_mootdx import (
    RAW_DIR,
    _BaostockSession,
    _baostock_basic_map,
    _baostock_symbol,
    assert_file_states_unchanged,
    capture_file_states,
    replace_staged_files_transactionally,
)
from data.kline_secondary_evidence import (
    DEFAULT_SNAPSHOT_PATH as DEFAULT_SECONDARY_SNAPSHOT_PATH,
    HISTORY_REPAIR_49,
    verified_history_repair_evidence,
    verified_tencent_history_repair_ohlcv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_PATH = (
    ROOT
    / "data"
    / "kline_evidence"
    / "baostock_verified_history_patches.parquet"
)
DEFAULT_MOOTDX_ARCHIVE_DIR = ROOT / "data" / "k-line-mootdx-bak"
DEFAULT_QMT_ARCHIVE_DIR = ROOT / "data" / "k-line-qmt-bak"
SNAPSHOT_SCHEMA = "verified-history-patches-v5"
SNAPSHOT_SOURCE = "baostock.adjustflag3.daily.raw"
MULTISOURCE_GAP_SOURCE = (
    "qmt.none.daily.crosschecked.tencent-none+baidu+10jqka"
)
ARCHIVED_MOOTDX_SOURCE = "mootdx.fq0.daily.archived.raw.finite-6-v1"
ARCHIVED_QMT_SOURCE = "qmt.none.daily.archived.raw.finite-42-v1"
# Fixed aggregate over the sorted per-row payload hashes.  Per-row hashes
# detect accidental corruption; this code-owned manifest also prevents a
# modified row from being accepted merely because its row hash was recomputed.
EXPECTED_SNAPSHOT_MANIFEST_SHA256 = (
    "36f31221bdbcd3ec3a736837815691a8e990284edf1351e27483877073202983"
)
_USE_CODE_MANIFEST = object()

HISTORY_REPAIR_49_CODE_DATE_SHA256 = (
    "da340c0947604f7c855d64ccf55013d7514183da9617f5f4fdb74bf331091545"
)
EXPECTED_SOURCE_COUNTS = MappingProxyType(
    {
        SNAPSHOT_SOURCE: 248,
        MULTISOURCE_GAP_SOURCE: 2,
        ARCHIVED_MOOTDX_SOURCE: 6,
        ARCHIVED_QMT_SOURCE: 42,
    }
)
EXPECTED_OPERATION_COUNTS = MappingProxyType(
    {
        "upsert": 226,
        "delete_nontrading": 31,
        "delete": 27,
        "patch_preclose": 13,
        "delete_before": 1,
    }
)

UNIT_CONTRACT_NONE = "no-market-row-v1"
UNIT_CONTRACT_SHARE_VOLUME = "cny-unadjusted-share-volume-to-lot-v1"
UNIT_CONTRACT_LOT_VOLUME = "cny-unadjusted-lot-volume-identity-v1"
UNIT_CONTRACTS = MappingProxyType(
    {
        UNIT_CONTRACT_NONE: MappingProxyType(
            {
                "price_unit": "none",
                "volume_unit": "none",
                "amount_unit": "none",
                "volume_divisor": 1.0,
                "adjustment": "none",
            }
        ),
        UNIT_CONTRACT_SHARE_VOLUME: MappingProxyType(
            {
                "price_unit": "CNY/share",
                "volume_unit": "share",
                "amount_unit": "CNY",
                "volume_divisor": 100.0,
                "adjustment": "none",
            }
        ),
        UNIT_CONTRACT_LOT_VOLUME: MappingProxyType(
            {
                "price_unit": "CNY/share",
                "volume_unit": "lot(100-share)",
                "amount_unit": "CNY",
                "volume_divisor": 1.0,
                "adjustment": "none",
            }
        ),
    }
)
TURNOVER_PRESERVE = "preserve"
TURNOVER_NAN = "set_nan"
TURNOVER_NOT_WRITTEN = "not_written"

PRODUCTION_COLUMNS = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "preClose",
)
RAW_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
)
SNAPSHOT_COLUMNS = (
    "source",
    "schema_version",
    "code",
    "baostock_code",
    "source_symbol",
    "source_artifact",
    "source_time_ms",
    "date",
    "ipo_date",
    *RAW_FIELDS,
    "source_tradestatus",
    "tradestatus",
    "operation",
    "unit_contract",
    "turnover_policy",
    "executable_evidence_payload_sha256",
    "source_row_sha256",
    "successor_date",
    "successor_preclose",
    "successor_source_row_sha256",
    "payload_sha256",
)
QUERY_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,tradestatus"

# A separate offline listing validator consumes this explicit same-code
# predecessor fact.  MappingProxyType prevents accidental runtime mutation.
SAME_CODE_PREDECESSOR_FIRST_EXECUTABLE = MappingProxyType(
    {"600018.SH": np.datetime64("2000-07-19", "D")}
)


def _days(*values: str) -> tuple[str, ...]:
    return tuple(values)


EXPECTED_IPO_DATES = MappingProxyType(
    {
        "000001.SZ": "1991-04-03",
        "000004.SZ": "1991-01-14",
        "000005.SZ": "1990-12-10",
        "000006.SZ": "1992-04-27",
        "000007.SZ": "1992-04-13",
        "000008.SZ": "1992-05-07",
        "000011.SZ": "1992-03-30",
        "000012.SZ": "1992-02-28",
        "000014.SZ": "1992-06-02",
        "000016.SZ": "1992-03-27",
        "000019.SZ": "1992-10-12",
        "000020.SZ": "1992-04-28",
        "000023.SZ": "1993-04-29",
        "000028.SZ": "1993-08-09",
        "000514.SZ": "1993-07-12",
        "000501.SZ": "1992-11-20",
        "000503.SZ": "1992-11-30",
        "000504.SZ": "1992-12-08",
        "000506.SZ": "1993-03-12",
        "000507.SZ": "1993-03-26",
        "000540.SZ": "1994-02-02",
        "001872.SZ": "1993-05-05",
        "300029.SZ": "2009-12-25",
        # The code belonged to the independently verified predecessor entity
        # from this first executable date until the 2006 re-listing.
        "600018.SH": "2000-07-19",
        "600093.SH": "1997-06-26",
        "600604.SH": "1992-03-27",
        "600605.SH": "1992-03-27",
        "600606.SH": "1992-03-27",
        "600608.SH": "1992-03-27",
        "600607.SH": "1992-03-27",
        "600618.SH": "1992-11-13",
        "600619.SH": "1992-11-16",
        "600620.SH": "1992-11-17",
        "600636.SH": "1993-03-16",
        "600651.SH": "1990-12-19",
        "600652.SH": "1990-12-19",
        "600653.SH": "1990-12-19",
        "600654.SH": "1990-12-19",
        "600656.SH": "1990-12-19",
        "600665.SH": "1993-07-09",
        "600744.SH": "1996-09-05",
    }
)

# These rows occur after IPO but are independently disproved by Baostock,
# Tencent and QMT calendars.  The May rows are a mootdx date-shift defect:
# their OHLC values duplicate the following real sessions.  Keeping this set
# separate from pre-IPO deletion makes the evidence semantics reviewable.
MAY_1993_CALENDAR_DEFECT_CODES = (
    "000001.SZ",
    "000006.SZ",
    "000007.SZ",
    "000008.SZ",
    "000011.SZ",
    "000012.SZ",
    "000014.SZ",
    "000016.SZ",
    "000019.SZ",
    "000020.SZ",
    "000501.SZ",
    "000503.SZ",
    "000504.SZ",
    "000506.SZ",
    "000507.SZ",
)
DELETE_NONTRADING_DATES = MappingProxyType(
    {
        "000004.SZ": _days("1991-11-17"),
        **{
            code: _days("1993-05-01", "1993-05-02")
            for code in MAY_1993_CALENDAR_DEFECT_CODES
        },
    }
)

# Exact pre-listing rows independently disproved by Baostock stock_basic.
DELETE_DATES = MappingProxyType(
    {
        "000012.SZ": _days("1992-01-07"),
        "000514.SZ": _days(
            "1993-07-05",
            "1993-07-06",
            "1993-07-07",
            "1993-07-08",
            "1993-07-09",
        ),
        "001872.SZ": _days(
            "1992-05-05",
            "1992-05-06",
            "1992-05-07",
            "1992-05-11",
            "1992-05-12",
            "1992-05-13",
            "1992-05-14",
            "1992-05-18",
            "1992-05-19",
            "1992-05-20",
            "1992-05-21",
            "1992-05-25",
            "1992-05-26",
            "1992-05-27",
            "1992-05-28",
            "1992-06-01",
            "1992-06-02",
        ),
        "600618.SH": _days("1992-11-12"),
        "600619.SH": _days("1992-11-13"),
        "600620.SH": _days("1992-11-16"),
        "600744.SH": _days("1996-09-04"),
    }
)

# This symbol has a whole pre-IPO prefix, rather than one isolated row.
DELETE_BEFORE = MappingProxyType({"000028.SZ": "1993-08-09"})

# Full raw replacement/insertion dates.  Exact calendars make a truncated
# source response observable instead of silently accepting a partial patch.
_BASE_UPSERT_DATES = MappingProxyType(
    {
        "000514.SZ": _days(
            "1993-07-12",
            "1993-07-13",
            "1993-07-14",
            "1993-07-15",
            "1993-07-16",
            "1993-07-19",
            "1993-07-20",
            "1993-07-21",
            "1993-07-22",
            "1993-07-23",
            "1993-07-26",
            "1993-07-27",
            "1993-07-28",
            "1993-07-29",
            "1993-07-30",
            "1993-08-02",
            "1993-08-03",
            "1993-08-04",
            "1993-08-05",
        ),
        "001872.SZ": _days(
            "1993-05-05",
            "1993-05-06",
            "1993-05-07",
            "1993-05-10",
            "1993-05-11",
            "1993-05-12",
            "1993-05-13",
            "1993-05-14",
            "1993-05-17",
            "1993-05-18",
            "1993-05-19",
            "1993-05-20",
            "1993-05-21",
            "1993-05-24",
            "1993-05-25",
            "1993-05-26",
            "1993-05-27",
            "1993-05-28",
            "1993-05-31",
            "1993-06-01",
            "1993-06-02",
        ),
        "600608.SH": _days(
            "1992-03-27",
            "1992-03-30",
            "1992-03-31",
            "1992-04-01",
            "1992-04-02",
            "1992-04-03",
            "1992-04-06",
            "1992-04-07",
            "1992-04-08",
            "1992-04-09",
        ),
        "600607.SH": _days(
            "1992-03-27",
            "1992-03-30",
            "1992-03-31",
            "1992-04-01",
            "1992-04-02",
            "1992-04-08",
            "1992-04-09",
        ),
        "600651.SH": _days("1990-12-19"),
        "600652.SH": _days("1990-12-19"),
        "600653.SH": _days(
            "1990-12-19",
            "1990-12-20",
            "1990-12-21",
            "1990-12-24",
            "1990-12-25",
            "1990-12-26",
            "1990-12-27",
            "1990-12-28",
            "1990-12-31",
        ),
        "600665.SH": _days("1993-07-09"),
        "600656.SH": _days(
            "1990-12-26",
            "1990-12-27",
            "1990-12-28",
            "1990-12-31",
            "1991-01-07",
            "1991-01-08",
            "1991-01-14",
            "1991-01-15",
            "1991-01-28",
            "1991-01-29",
            "1991-01-30",
            "1991-01-31",
            "1991-02-01",
            "1991-02-04",
            "1991-02-05",
            "1991-02-06",
            "1991-02-08",
            "1991-02-11",
            "1991-02-12",
            "1991-02-13",
            "1991-02-14",
            "1991-02-19",
            "1991-02-20",
            "1991-02-21",
            "1991-02-22",
            "1991-02-25",
            "1991-02-26",
            "1991-02-27",
            "1991-02-28",
            "1991-03-01",
            "1991-03-11",
            "1991-03-12",
            "1991-03-13",
            "1991-03-14",
            "1991-03-15",
            "1991-03-18",
            "1991-03-19",
            "1991-03-20",
            "1991-03-21",
            "1991-03-22",
            "1991-03-25",
            "1991-03-26",
            "1991-03-27",
            "1991-03-28",
            "1991-04-11",
            "1991-06-05",
            "2016-05-12",
        ),
    }
)

MAY_1993_REAL_SESSION_DATES = _days(
    "1993-05-03",
    "1993-05-04",
    "1993-05-05",
    "1993-05-06",
)
_LEGACY_UPSERT_DATES = MappingProxyType(
    {
        **_BASE_UPSERT_DATES,
        "600018.SH": _days("2001-08-16"),
        **{
            code: tuple(
                sorted(
                    set(_BASE_UPSERT_DATES.get(code, ()))
                    | set(MAY_1993_REAL_SESSION_DATES)
                )
            )
            for code in MAY_1993_CALENDAR_DEFECT_CODES
        },
    }
)

ARCHIVED_MOOTDX_REPAIR_DATES = MappingProxyType(
    {
        "000005.SZ": _days("1991-01-14", "1992-02-03", "1992-02-07"),
        "000023.SZ": _days("1995-04-28"),
        "000540.SZ": _days("1995-04-28"),
        "600093.SH": _days("1999-01-06"),
    }
)
TARGETED_BAOSTOCK_REPAIR_DATES = MappingProxyType(
    {"300029.SZ": _days("2026-07-09")}
)
_HISTORY_REPAIR_49_SET = frozenset(
    (code, day)
    for code, days in HISTORY_REPAIR_49.items()
    for day in days
)
_ARCHIVED_MOOTDX_REPAIR_SET = frozenset(
    (code, day)
    for code, days in ARCHIVED_MOOTDX_REPAIR_DATES.items()
    for day in days
)
_TARGETED_BAOSTOCK_REPAIR_SET = frozenset(
    (code, day)
    for code, days in TARGETED_BAOSTOCK_REPAIR_DATES.items()
    for day in days
)
_ARCHIVED_QMT_REPAIR_SET = frozenset(
    _HISTORY_REPAIR_49_SET
    - _ARCHIVED_MOOTDX_REPAIR_SET
    - _TARGETED_BAOSTOCK_REPAIR_SET
)
if not (
    len(_HISTORY_REPAIR_49_SET) == 49
    and len(_ARCHIVED_MOOTDX_REPAIR_SET) == 6
    and len(_ARCHIVED_QMT_REPAIR_SET) == 42
    and len(_TARGETED_BAOSTOCK_REPAIR_SET) == 1
):
    raise RuntimeError("history-repair 49-row source split contract mismatch")
_history_contract_payload = "\n".join(
    f"{code}|{day}" for code, day in sorted(_HISTORY_REPAIR_49_SET)
)
if hashlib.sha256(_history_contract_payload.encode()).hexdigest() != (
    HISTORY_REPAIR_49_CODE_DATE_SHA256
):
    raise RuntimeError("history-repair 49-row code/date contract hash mismatch")

# Only these archived rows have internally coherent OHLC/lot/amount units.
# All other 49-row archive inserts retain raw OHLC/preClose but deliberately
# write turnover as NaN; source volume/amount remain sealed in the snapshot.
RELIABLE_HISTORY_REPAIR_TURNOVER = frozenset(
    {
        ("000004.SZ", "1992-05-05"),
        ("000004.SZ", "1993-08-09"),
        ("001872.SZ", "1993-07-15"),
        ("300029.SZ", "2026-07-09"),
        ("600093.SH", "1999-01-06"),
        ("600604.SH", "1993-01-04"),
        ("600605.SH", "1993-01-04"),
        ("600606.SH", "1993-01-04"),
        ("600608.SH", "1993-01-04"),
        ("600618.SH", "1992-11-17"),
        ("600618.SH", "1993-01-04"),
        ("600619.SH", "1992-11-18"),
        ("600620.SH", "1993-01-04"),
        ("600636.SH", "1994-04-07"),
        ("600654.SH", "1993-01-04"),
    }
)

# Each target points to the first later executable local row whose direct
# Baostock preClose is currently stale.  The successor source row and value
# are sealed into the same 49-row upsert record, avoiding duplicate operations.
HISTORY_REPAIR_SUCCESSORS = MappingProxyType(
    {
        ("000004.SZ", "1991-01-29"): "1991-01-30",
        ("000004.SZ", "1991-04-01"): "1991-04-02",
        ("000004.SZ", "1991-05-02"): "1991-05-09",
        ("000004.SZ", "1991-06-06"): "1991-06-07",
        ("000004.SZ", "1991-10-09"): "1991-10-10",
        ("000004.SZ", "1992-02-25"): "1992-02-26",
        ("000004.SZ", "1992-05-05"): "1992-05-06",
        ("000004.SZ", "1993-08-09"): "1993-08-10",
        ("001872.SZ", "1993-07-15"): "1993-07-16",
        ("600604.SH", "1993-01-04"): "1993-01-05",
        ("600605.SH", "1993-01-04"): "1993-01-05",
        ("600606.SH", "1993-01-04"): "1993-01-05",
        ("600608.SH", "1993-01-04"): "1993-01-05",
        ("600618.SH", "1992-11-17"): "1992-11-18",
        ("600618.SH", "1993-01-04"): "1993-01-05",
        ("600619.SH", "1992-11-18"): "1992-11-19",
        ("600619.SH", "1993-01-04"): "1993-01-05",
        ("600620.SH", "1992-11-19"): "1992-11-20",
        ("600620.SH", "1993-01-04"): "1993-01-05",
        ("600636.SH", "1994-04-07"): "1994-04-08",
        ("600651.SH", "1991-01-25"): "1991-01-29",
        ("600653.SH", "1991-03-28"): "1991-03-29",
        ("600653.SH", "1993-12-23"): "1993-12-24",
        ("600654.SH", "1991-01-02"): "1991-01-03",
        ("600654.SH", "1992-01-02"): "1992-01-03",
        ("600654.SH", "1993-01-04"): "1993-01-05",
        ("600665.SH", "1993-12-23"): "1993-12-24",
    }
)

UPSERT_DATES = MappingProxyType(
    {
        code: tuple(
            sorted(
                set(_LEGACY_UPSERT_DATES.get(code, ()))
                | set(HISTORY_REPAIR_49.get(code, ()))
            )
        )
        for code in sorted(set(_LEGACY_UPSERT_DATES) | set(HISTORY_REPAIR_49))
    }
)

# Deleting/replacing a prefix also invalidates the surviving boundary row's
# old predecessor.  Only preClose is patched there so mootdx remains the raw
# OHLCVA authority outside the explicitly requested replacement segments.
PATCH_PRECLOSE_DATES = MappingProxyType(
    {
        "000004.SZ": _days("1991-11-18"),
        "000012.SZ": _days("1992-02-28"),
        "000028.SZ": _days("1993-08-09"),
        "000514.SZ": _days("1993-08-06"),
        "001872.SZ": _days("1993-06-03"),
        "600608.SH": _days("1992-04-10"),
        "600018.SH": _days("2001-08-17"),
        "600618.SH": _days("1992-11-13"),
        "600619.SH": _days("1992-11-16"),
        "600620.SH": _days("1992-11-17"),
        "600651.SH": _days("1990-12-21"),
        "600653.SH": _days("1991-01-08"),
        "600744.SH": _days("1996-09-05"),
    }
)

EARLY_UNRELIABLE_TURNOVER = frozenset(
    {
        "600607.SH",
        "600608.SH",
        "600651.SH",
        "600652.SH",
        "600653.SH",
        "600656.SH",
    }
)

# mootdx/TDX omits this real predecessor session on every currently reachable
# server.  QMT supplies the unadjusted row written below; Tencent none/day,
# Baidu volume/amount and the 10jqka calendar independently corroborate it.
# Values remain finite and code-owned because this is a two-row historical
# repair, never a fallback daily K-line source.
SEALED_MULTISOURCE_RECORDS = MappingProxyType(
    {
        ("600018.SH", "2001-08-16", "upsert"): MappingProxyType(
            {
                "open": 19.40,
                "high": 19.76,
                "low": 19.31,
                "close": 19.60,
                "preclose": 19.40,
                "volume": 913_000.0,
                "amount": 17_768_754.0,
                "tradestatus": "1",
            }
        ),
        ("600018.SH", "2001-08-17", "patch_preclose"): MappingProxyType(
            {
                "open": 19.59,
                "high": 19.59,
                "low": 19.20,
                "close": 19.30,
                "preclose": 19.60,
                "volume": 571_700.0,
                "amount": 11_059_520.0,
                "tradestatus": "1",
            }
        ),
    }
)


def _expected_operations() -> set[tuple[str, str, str]]:
    expected: set[tuple[str, str, str]] = set()
    for code, dates in DELETE_DATES.items():
        expected.update((code, day, "delete") for day in dates)
    for code, dates in DELETE_NONTRADING_DATES.items():
        expected.update((code, day, "delete_nontrading") for day in dates)
    expected.update(
        (code, day, "delete_before") for code, day in DELETE_BEFORE.items()
    )
    for code, dates in UPSERT_DATES.items():
        expected.update((code, day, "upsert") for day in dates)
    for code, dates in PATCH_PRECLOSE_DATES.items():
        expected.update((code, day, "patch_preclose") for day in dates)
    return expected


EXPECTED_OPERATIONS = frozenset(_expected_operations())
TARGET_CODES = tuple(sorted(EXPECTED_IPO_DATES))
BAOSTOCK_TARGET_CODES = tuple(
    code for code in TARGET_CODES if code != "600018.SH"
)


def _expected_source(code: str, day: str, operation: str) -> str:
    if operation == "upsert" and (code, day) in _ARCHIVED_MOOTDX_REPAIR_SET:
        return ARCHIVED_MOOTDX_SOURCE
    if operation == "upsert" and (code, day) in _ARCHIVED_QMT_REPAIR_SET:
        return ARCHIVED_QMT_SOURCE
    if (code, day, operation) in SEALED_MULTISOURCE_RECORDS:
        return MULTISOURCE_GAP_SOURCE
    return SNAPSHOT_SOURCE


def _expected_source_artifact(
    code: str, day: str, operation: str
) -> str:
    source = _expected_source(code, day, operation)
    if source == ARCHIVED_MOOTDX_SOURCE:
        return f"data/k-line-mootdx-bak/{code}.parquet"
    if source == ARCHIVED_QMT_SOURCE:
        return f"data/k-line-qmt-bak/{code}.parquet"
    if source == MULTISOURCE_GAP_SOURCE:
        return "code-owned:SEALED_MULTISOURCE_RECORDS-v1"
    return "baostock.query_history_k_data_plus(adjustflag=3)"


def _expected_source_symbol(code: str, source: str) -> str:
    return _baostock_symbol(code) if source == SNAPSHOT_SOURCE else code


def _expected_unit_contract(
    code: str, day: str, operation: str
) -> str:
    if operation not in {"upsert", "patch_preclose"}:
        return UNIT_CONTRACT_NONE
    if (code, day) in _ARCHIVED_QMT_REPAIR_SET:
        return UNIT_CONTRACT_LOT_VOLUME
    return UNIT_CONTRACT_SHARE_VOLUME


def _expected_turnover_policy(
    code: str, day: str, operation: str
) -> str:
    if operation != "upsert":
        return TURNOVER_NOT_WRITTEN
    if (code, day) in _HISTORY_REPAIR_49_SET:
        return (
            TURNOVER_PRESERVE
            if (code, day) in RELIABLE_HISTORY_REPAIR_TURNOVER
            else TURNOVER_NAN
        )
    return TURNOVER_NAN if code in EARLY_UNRELIABLE_TURNOVER else TURNOVER_PRESERVE


def _canonical_hash_value(value: object, *, numeric: bool = False) -> str:
    if numeric:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "" if pd.isna(parsed) else format(float(parsed), ".17g")
    return "" if pd.isna(value) else str(value).strip()


def _source_row_sha256(row: pd.Series | dict[str, object]) -> str:
    payload = {
        "source": _canonical_hash_value(row["source"]),
        "source_symbol": _canonical_hash_value(row["source_symbol"]),
        "source_artifact": _canonical_hash_value(row["source_artifact"]),
        "source_time_ms": _canonical_hash_value(row["source_time_ms"], numeric=True),
        "code": _canonical_hash_value(row["code"]),
        "date": _canonical_hash_value(row["date"]),
        **{
            field: _canonical_hash_value(row[field], numeric=True)
            for field in RAW_FIELDS
        },
        "source_tradestatus": _canonical_hash_value(
            row["source_tradestatus"]
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _baostock_raw_row_sha256(
    *, code: str, day: str, row: pd.Series
) -> str:
    payload = {
        "source": SNAPSHOT_SOURCE,
        "source_symbol": _baostock_symbol(code),
        "date": day,
        **{
            field: _canonical_hash_value(row[field], numeric=True)
            for field in RAW_FIELDS
        },
        "tradestatus": _canonical_hash_value(row["tradestatus"]),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_sha256(row: pd.Series | dict[str, object]) -> str:
    """Hash one canonical source/operation record for offline replay audits."""

    values: dict[str, str] = {}
    for field in SNAPSHOT_COLUMNS:
        if field == "payload_sha256":
            continue
        value = row[field]
        if field in {*RAW_FIELDS, "source_time_ms", "successor_preclose"}:
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            values[field] = "" if pd.isna(numeric) else format(float(numeric), ".17g")
        else:
            values[field] = "" if pd.isna(value) else str(value).strip()
    payload = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _attach_payload_hash(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["source_row_sha256"] = [
        _source_row_sha256(row) for _, row in result.iterrows()
    ]
    result["payload_sha256"] = [
        _payload_sha256(row) for _, row in result.iterrows()
    ]
    return result


def _snapshot_manifest_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(
        ["code", "date", "operation"], kind="stable"
    )
    payload = "\n".join(
        ordered["payload_sha256"].fillna("").astype(str).str.strip().str.lower()
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _validate_snapshot(
    raw: pd.DataFrame,
    *,
    expected_manifest_sha256: str | None | object = _USE_CODE_MANIFEST,
) -> pd.DataFrame:
    if list(raw.columns) != list(SNAPSHOT_COLUMNS) or raw.empty:
        raise ValueError("历史修复证据 snapshot schema/内容不完整")
    frame = raw.copy()
    for field in (
        "source",
        "schema_version",
        "code",
        "baostock_code",
        "source_symbol",
        "source_artifact",
        "date",
        "ipo_date",
        "source_tradestatus",
        "tradestatus",
        "operation",
        "unit_contract",
        "turnover_policy",
        "executable_evidence_payload_sha256",
        "source_row_sha256",
        "successor_date",
        "successor_source_row_sha256",
    ):
        frame[field] = frame[field].fillna("").astype(str).str.strip()
    frame["payload_sha256"] = (
        frame["payload_sha256"].fillna("").astype(str).str.strip().str.lower()
    )
    if not frame["schema_version"].eq(SNAPSHOT_SCHEMA).all():
        raise ValueError("历史修复证据 schema_version 不兼容")
    frame["code"] = frame["code"].str.upper()
    frame["baostock_code"] = frame["baostock_code"].str.lower()
    frame["date"] = pd.to_datetime(
        frame["date"], format="%Y-%m-%d", errors="raise"
    ).dt.strftime("%Y-%m-%d")
    frame["ipo_date"] = pd.to_datetime(
        frame["ipo_date"], format="%Y-%m-%d", errors="raise"
    ).dt.strftime("%Y-%m-%d")
    if frame.duplicated(["code", "date", "operation"]).any():
        raise ValueError("历史修复证据 code/date/operation 重复")
    actual = set(frame[["code", "date", "operation"]].itertuples(index=False, name=None))
    if actual != set(EXPECTED_OPERATIONS):
        missing = sorted(set(EXPECTED_OPERATIONS).difference(actual))
        extra = sorted(actual.difference(EXPECTED_OPERATIONS))
        raise ValueError(f"历史修复证据目标覆盖不完整: missing={missing[:5]} extra={extra[:5]}")
    if frame["source"].value_counts().to_dict() != dict(EXPECTED_SOURCE_COUNTS):
        raise ValueError("历史修复证据 source 计数不符合固定契约")
    if frame["operation"].value_counts().to_dict() != dict(
        EXPECTED_OPERATION_COUNTS
    ):
        raise ValueError("历史修复证据 operation 计数不符合固定契约")
    expected_sources = pd.Series(
        [
            _expected_source(code, day, operation)
            for code, day, operation in frame[
                ["code", "date", "operation"]
            ].itertuples(index=False, name=None)
        ],
        index=frame.index,
    )
    if not frame["source"].eq(expected_sources).all():
        raise ValueError("历史修复证据 source 不受信任或与目标不匹配")
    expected_artifacts = pd.Series(
        [
            _expected_source_artifact(code, day, operation)
            for code, day, operation in frame[
                ["code", "date", "operation"]
            ].itertuples(index=False, name=None)
        ],
        index=frame.index,
    )
    if not frame["source_artifact"].eq(expected_artifacts).all():
        raise ValueError("历史修复证据 source_artifact 与固定来源不一致")
    expected_symbols = pd.Series(
        [
            _expected_source_symbol(code, source)
            for code, source in frame[["code", "source"]].itertuples(
                index=False, name=None
            )
        ],
        index=frame.index,
    )
    if not frame["source_symbol"].eq(expected_symbols).all():
        raise ValueError("历史修复证据 source_symbol 与固定来源不一致")

    for code, rows in frame.groupby("code", sort=False):
        if code not in EXPECTED_IPO_DATES:
            raise ValueError(f"历史修复证据包含未声明代码: {code}")
        if not rows["baostock_code"].eq(_baostock_symbol(code)).all():
            raise ValueError(f"历史修复证据代码/市场不一致: {code}")
        if not rows["ipo_date"].eq(EXPECTED_IPO_DATES[code]).all():
            raise ValueError(f"历史修复证据 IPO 日期不一致: {code}")

    numeric = frame.loc[:, RAW_FIELDS].apply(pd.to_numeric, errors="coerce")
    for field in RAW_FIELDS:
        frame[field] = numeric[field].astype(np.float64)
    frame["source_time_ms"] = pd.to_numeric(
        frame["source_time_ms"], errors="coerce"
    ).astype(np.float64)
    frame["successor_preclose"] = pd.to_numeric(
        frame["successor_preclose"], errors="coerce"
    ).astype(np.float64)
    expected_units = pd.Series(
        [
            _expected_unit_contract(code, day, operation)
            for code, day, operation in frame[
                ["code", "date", "operation"]
            ].itertuples(index=False, name=None)
        ],
        index=frame.index,
    )
    if not frame["unit_contract"].eq(expected_units).all():
        raise ValueError("历史修复证据 unit_contract 与固定来源不一致")
    expected_turnover = pd.Series(
        [
            _expected_turnover_policy(code, day, operation)
            for code, day, operation in frame[
                ["code", "date", "operation"]
            ].itertuples(index=False, name=None)
        ],
        index=frame.index,
    )
    if not frame["turnover_policy"].eq(expected_turnover).all():
        raise ValueError("历史修复证据 turnover_policy 与逐行固定契约不一致")
    archive_rows = frame["source"].isin(
        {ARCHIVED_MOOTDX_SOURCE, ARCHIVED_QMT_SOURCE}
    )
    archive_times = frame.loc[archive_rows, "source_time_ms"]
    if (
        archive_times.isna().any()
        or not np.isfinite(archive_times.to_numpy(dtype=np.float64)).all()
        or (archive_times % 1.0).ne(0.0).any()
    ):
        raise ValueError("历史修复 archive source_time_ms 必须为有限整数")
    if frame.loc[~archive_rows, "source_time_ms"].notna().any():
        raise ValueError("非 archive 历史证据不得伪造 source_time_ms")
    archive_dates = pd.to_datetime(
        frame.loc[archive_rows, "source_time_ms"].astype(np.int64),
        unit="ms",
    ).dt.strftime("%Y-%m-%d")
    if not np.array_equal(
        archive_dates.to_numpy(),
        frame.loc[archive_rows, "date"].to_numpy(),
    ):
        raise ValueError("历史修复 archive source_time_ms/date 不一致")
    source_rows = frame["operation"].isin({"upsert", "patch_preclose"})
    source_numeric = numeric.loc[source_rows]
    prices = source_numeric.loc[:, ["open", "high", "low", "close", "preclose"]]
    if prices.isna().any().any() or not np.isfinite(
        prices.to_numpy(dtype=np.float64)
    ).all() or (prices <= 0.0).any().any():
        raise ValueError("历史修复真实行情价格/preclose 必须为正")
    turnover = source_numeric.loc[:, ["volume", "amount"]]
    preserve = frame.loc[source_rows, "turnover_policy"].eq(TURNOVER_PRESERVE)
    reliable_turnover = turnover.loc[preserve]
    if (
        reliable_turnover.isna().any().any()
        or not np.isfinite(reliable_turnover.to_numpy(dtype=np.float64)).all()
        or (reliable_turnover <= 0.0).any().any()
    ):
        raise ValueError("历史修复可靠量额目标必须有正成交量和成交额")
    unreliable_turnover = turnover.loc[~preserve]
    finite_unreliable = unreliable_turnover.to_numpy(dtype=np.float64)
    if np.isinf(finite_unreliable).any() or np.any(
        finite_unreliable[np.isfinite(finite_unreliable)] < 0.0
    ):
        raise ValueError("历史修复不可靠量额只能为非负值或空值")
    if not frame.loc[source_rows, "tradestatus"].eq("1").all():
        raise ValueError("历史修复目标日期缺少明确可成交证据")
    status_bearing = source_rows & ~archive_rows
    if not frame.loc[status_bearing, "source_tradestatus"].eq("1").all():
        raise ValueError("带状态的历史真实源必须返回 tradestatus=1")
    if not frame.loc[archive_rows, "source_tradestatus"].eq("").all():
        raise ValueError("archive OHLCVA 不得伪造源内 tradestatus")
    if (
        (prices["high"] < prices[["open", "close"]].max(axis=1)).any()
        or (prices["low"] > prices[["open", "close"]].min(axis=1)).any()
        or (prices["high"] < prices["low"]).any()
    ):
        raise ValueError("历史修复真实行情 OHLC 关系非法")

    deletion_rows = ~source_rows
    if not numeric.loc[deletion_rows].isna().all().all():
        raise ValueError("删除证据不得伪装为真实行情")
    if not frame.loc[
        deletion_rows, ["source_tradestatus", "tradestatus"]
    ].eq("").all().all():
        raise ValueError("删除证据 tradestatus 必须为空")
    parsed_date = pd.to_datetime(frame["date"])
    parsed_ipo = pd.to_datetime(frame["ipo_date"])
    exact_delete = frame["operation"].eq("delete")
    nontrading_delete = frame["operation"].eq("delete_nontrading")
    before_delete = frame["operation"].eq("delete_before")
    if not (parsed_date[exact_delete] < parsed_ipo[exact_delete]).all():
        raise ValueError("delete 证据日期必须早于独立 IPO 日期")
    if not (parsed_date[before_delete] == parsed_ipo[before_delete]).all():
        raise ValueError("delete_before 截止日必须等于独立 IPO 日期")
    if not (parsed_date[nontrading_delete] >= parsed_ipo[nontrading_delete]).all():
        raise ValueError("delete_nontrading 日期不得早于独立 IPO 日期")
    if not (parsed_date[source_rows] >= parsed_ipo[source_rows]).all():
        raise ValueError("真实修复行情不得早于独立 IPO 日期")

    history_mask = pd.Series(
        [
            operation == "upsert" and (code, day) in _HISTORY_REPAIR_49_SET
            for code, day, operation in frame[
                ["code", "date", "operation"]
            ].itertuples(index=False, name=None)
        ],
        index=frame.index,
    )
    if int(history_mask.sum()) != 49:
        raise ValueError("history-repair 49-row snapshot 覆盖不完整")
    evidence_hashes = frame["executable_evidence_payload_sha256"]
    if not evidence_hashes.loc[history_mask].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("history-repair 49-row 缺少 secondary payload hash")
    if not evidence_hashes.loc[~history_mask].eq("").all():
        raise ValueError("非 history-repair 行不得绑定 secondary payload hash")

    expected_successors = pd.Series(
        [
            HISTORY_REPAIR_SUCCESSORS.get((code, day), "")
            if operation == "upsert"
            else ""
            for code, day, operation in frame[
                ["code", "date", "operation"]
            ].itertuples(index=False, name=None)
        ],
        index=frame.index,
    )
    if not frame["successor_date"].eq(expected_successors).all():
        raise ValueError("历史修复 successor_date 与固定链不一致")
    has_successor = expected_successors.ne("")
    if (
        frame.loc[has_successor, "successor_preclose"].isna().any()
        or not np.isfinite(
            frame.loc[has_successor, "successor_preclose"].to_numpy(
                dtype=np.float64
            )
        ).all()
        or (frame.loc[has_successor, "successor_preclose"] <= 0.0).any()
        or not frame.loc[
            has_successor, "successor_source_row_sha256"
        ].str.fullmatch(r"[0-9a-f]{64}").all()
    ):
        raise ValueError("历史修复 successor preClose/source hash 不完整")
    if (
        frame.loc[~has_successor, "successor_preclose"].notna().any()
        or not frame.loc[~has_successor, "successor_source_row_sha256"].eq("").all()
    ):
        raise ValueError("无 successor 的历史修复行不得携带 successor payload")
    successor_dates = pd.to_datetime(
        frame.loc[has_successor, "successor_date"],
        format="%Y-%m-%d",
        errors="raise",
    )
    if not (
        successor_dates.to_numpy()
        > parsed_date.loc[has_successor].to_numpy()
    ).all():
        raise ValueError("历史修复 successor_date 必须晚于 upsert 日期")

    expected_source_hashes = pd.Series(
        [_source_row_sha256(row) for _, row in frame.iterrows()],
        index=frame.index,
    )
    if not frame["source_row_sha256"].eq(expected_source_hashes).all():
        raise ValueError("历史修复证据 source_row_sha256 校验失败")

    expected_hashes = pd.Series(
        [_payload_sha256(row) for _, row in frame.iterrows()],
        index=frame.index,
    )
    if not frame["payload_sha256"].eq(expected_hashes).all():
        raise ValueError("历史修复证据 payload_sha256 校验失败")
    if expected_manifest_sha256 is _USE_CODE_MANIFEST:
        expected_manifest_sha256 = EXPECTED_SNAPSHOT_MANIFEST_SHA256
    actual_manifest = _snapshot_manifest_sha256(frame)
    if (
        expected_manifest_sha256 is not None
        and actual_manifest != expected_manifest_sha256
    ):
        raise ValueError(
            "历史修复证据 manifest_sha256 校验失败: "
            f"{actual_manifest}"
        )

    return frame.loc[:, SNAPSHOT_COLUMNS].sort_values(
        ["code", "date", "operation"], kind="stable"
    ).reset_index(drop=True)


def _load_snapshot(path: Path) -> pd.DataFrame:
    return _validate_snapshot(pd.read_parquet(path))


def _source_record(
    *,
    code: str,
    day: str,
    operation: str,
    source_values: dict[str, object] | pd.Series,
    source_time_ms: object = np.nan,
    effective_tradestatus: str = "1",
    source_tradestatus: str = "1",
    executable_evidence_payload_sha256: str = "",
    successor_values: pd.Series | None = None,
) -> dict[str, object]:
    source = _expected_source(code, day, operation)
    successor_day = HISTORY_REPAIR_SUCCESSORS.get((code, day), "")
    if successor_day:
        if successor_values is None:
            raise ValueError(f"{code}/{day} 缺少固定 successor 源行")
        if str(successor_values["tradestatus"]).strip() != "1":
            raise ValueError(f"{code}/{successor_day} successor 非可成交状态")
        successor_preclose = float(successor_values["preclose"])
        if not np.isfinite(successor_preclose) or successor_preclose <= 0.0:
            raise ValueError(f"{code}/{successor_day} successor preClose 非法")
        successor_hash = _baostock_raw_row_sha256(
            code=code,
            day=successor_day,
            row=successor_values,
        )
    else:
        if successor_values is not None:
            raise ValueError(f"{code}/{day} 未声明 successor 却携带源行")
        successor_preclose = np.nan
        successor_hash = ""
    record: dict[str, object] = {
        "source": source,
        "schema_version": SNAPSHOT_SCHEMA,
        "code": code,
        "baostock_code": _baostock_symbol(code),
        "source_symbol": _expected_source_symbol(code, source),
        "source_artifact": _expected_source_artifact(code, day, operation),
        "source_time_ms": source_time_ms,
        "date": day,
        "ipo_date": EXPECTED_IPO_DATES[code],
        "source_tradestatus": source_tradestatus,
        "tradestatus": effective_tradestatus,
        "operation": operation,
        "unit_contract": _expected_unit_contract(code, day, operation),
        "turnover_policy": _expected_turnover_policy(code, day, operation),
        "executable_evidence_payload_sha256": (
            executable_evidence_payload_sha256
        ),
        "successor_date": successor_day,
        "successor_preclose": successor_preclose,
        "successor_source_row_sha256": successor_hash,
    }
    record.update({field: source_values[field] for field in RAW_FIELDS})
    return record


def _validated_archive_rows(
    archive_dir: Path,
    expected: frozenset[tuple[str, str]],
) -> dict[tuple[str, str], pd.Series]:
    selected: dict[tuple[str, str], pd.Series] = {}
    by_code: dict[str, set[str]] = {}
    for code, day in expected:
        by_code.setdefault(code, set()).add(day)
    for code, days in sorted(by_code.items()):
        path = Path(archive_dir) / f"{code}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"history-repair archive 缺失: {path}")
        frame = pd.read_parquet(path)
        missing_columns = set(PRODUCTION_COLUMNS).difference(frame.columns)
        if missing_columns or frame.empty:
            raise ValueError(f"history-repair archive schema/内容非法: {path}")
        frame = frame.loc[:, PRODUCTION_COLUMNS].copy()
        times = pd.to_numeric(frame["time"], errors="raise")
        if times.isna().any() or times.duplicated().any():
            raise ValueError(f"history-repair archive time 非法/重复: {path}")
        frame = frame.copy()
        frame["time"] = times.astype(np.int64)
        frame["date"] = pd.to_datetime(frame["time"], unit="ms").dt.strftime(
            "%Y-%m-%d"
        )
        if frame["date"].duplicated().any():
            raise ValueError(f"history-repair archive 日期重复: {path}")
        indexed = frame.set_index("date", drop=False)
        missing = sorted(days.difference(indexed.index))
        if missing:
            raise ValueError(f"history-repair archive 目标未覆盖: {code}/{missing}")
        for day in sorted(days):
            row = indexed.loc[day]
            values = pd.to_numeric(
                row[["open", "high", "low", "close", "preClose", "volume", "amount"]],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"history-repair archive 行含非有限值: {code}/{day}")
            prices = values[:5]
            if (
                (prices <= 0.0).any()
                or prices[1] < max(prices[0], prices[3])
                or prices[2] > min(prices[0], prices[3])
                or prices[1] < prices[2]
            ):
                raise ValueError(f"history-repair archive OHLC/preClose 非法: {code}/{day}")
            if (values[5:] < 0.0).any():
                raise ValueError(f"history-repair archive 量额为负: {code}/{day}")
            selected[(code, day)] = row
    if set(selected) != set(expected):
        raise ValueError("history-repair archive 批次覆盖不完整")
    return selected


def _validate_targeted_baostock_tencent_crosscheck(
    baostock_row: pd.Series,
    tencent_row: dict[str, object],
    sealed_secondary_row: pd.Series,
) -> None:
    """Cross-check the one live Baostock K row without copying Tencent prices."""

    if (
        str(tencent_row.get("date", "")) != "2026-07-09"
        or str(tencent_row.get("source_row_sha256", ""))
        != str(sealed_secondary_row["source_row_sha256"])
    ):
        raise ValueError("300029 Tencent 同日证据/hash 与封存 secondary 不一致")
    baostock_prices = np.asarray(
        [
            float(baostock_row["open"]),
            float(baostock_row["close"]),
            float(baostock_row["high"]),
            float(baostock_row["low"]),
        ],
        dtype=np.float64,
    )
    tencent_prices = np.asarray(
        [
            float(tencent_row["open"]),
            float(tencent_row["close"]),
            float(tencent_row["high"]),
            float(tencent_row["low"]),
        ],
        dtype=np.float64,
    )
    if (
        not np.isfinite(baostock_prices).all()
        or not np.isfinite(tencent_prices).all()
        or not np.allclose(
            baostock_prices,
            tencent_prices,
            rtol=0.0,
            atol=0.0005,
        )
    ):
        raise ValueError("300029 Baostock/Tencent unadjusted OHLC 交叉不一致")
    baostock_volume_shares = float(baostock_row["volume"])
    tencent_volume_lots = float(tencent_row["volume"])
    if (
        not np.isfinite(baostock_volume_shares)
        or not np.isfinite(tencent_volume_lots)
        or baostock_volume_shares <= 0.0
        or tencent_volume_lots <= 0.0
        or not np.isclose(
            baostock_volume_shares / 100.0,
            tencent_volume_lots,
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise ValueError("300029 Baostock股/Tencent手 volume 单位交叉不一致")
    amount = float(baostock_row["amount"])
    low_notional = baostock_volume_shares * float(baostock_row["low"])
    high_notional = baostock_volume_shares * float(baostock_row["high"])
    if (
        not np.isfinite(amount)
        or amount <= 0.0
        or amount < low_notional * 0.999
        or amount > high_notional * 1.001
    ):
        raise ValueError("300029 Baostock amount 与股数/OHLC 单位不自洽")
    if (
        str(baostock_row["tradestatus"]).strip() != "1"
        or not np.isfinite(float(baostock_row["preclose"]))
        or float(baostock_row["preclose"]) <= 0.0
    ):
        raise ValueError("300029 Baostock preClose/status 不可用")


def _download_snapshot(
    baostock_module=None,
    *,
    mootdx_archive_dir: Path = DEFAULT_MOOTDX_ARCHIVE_DIR,
    qmt_archive_dir: Path = DEFAULT_QMT_ARCHIVE_DIR,
    secondary_snapshot_path: Path = DEFAULT_SECONDARY_SNAPSHOT_PATH,
    tencent_ohlcv: dict[str, object] | None = None,
    tencent_timeout: float = 30.0,
) -> pd.DataFrame:
    secondary = verified_history_repair_evidence(
        path=Path(secondary_snapshot_path)
    ).copy()
    secondary["date"] = pd.to_datetime(secondary["date"]).dt.strftime("%Y-%m-%d")
    secondary_by_key = {
        (str(row["stock_code"]), str(row["date"])): row
        for _, row in secondary.iterrows()
    }
    if set(secondary_by_key) != set(_HISTORY_REPAIR_49_SET):
        raise ValueError("secondary history-repair 49-row 覆盖不完整")
    if tencent_ohlcv is None:
        tencent_ohlcv = verified_tencent_history_repair_ohlcv(
            "300029.SZ",
            "2026-07-09",
            evidence_path=Path(secondary_snapshot_path),
            timeout=tencent_timeout,
        )

    archive_rows = {
        **_validated_archive_rows(
            Path(mootdx_archive_dir), _ARCHIVED_MOOTDX_REPAIR_SET
        ),
        **_validated_archive_rows(
            Path(qmt_archive_dir), _ARCHIVED_QMT_REPAIR_SET
        ),
    }
    if set(archive_rows) != (
        set(_ARCHIVED_MOOTDX_REPAIR_SET) | set(_ARCHIVED_QMT_REPAIR_SET)
    ):
        raise ValueError("history-repair 48-row archive source split 不完整")

    module = baostock_module or importlib.import_module("baostock")
    session = _BaostockSession(module)
    records: list[dict[str, object]] = []
    baostock_rows_by_code: dict[str, pd.DataFrame] = {}
    try:
        basic = _baostock_basic_map(session, list(BAOSTOCK_TARGET_CODES))
        for code in BAOSTOCK_TARGET_CODES:
            expected = EXPECTED_IPO_DATES[code]
            actual = pd.Timestamp(str(basic[code]["ipoDate"])).strftime("%Y-%m-%d")
            if actual != expected:
                raise ValueError(f"{code} Baostock IPO {actual} != 已核验 {expected}")

        for code, dates in DELETE_DATES.items():
            for day in dates:
                records.append(_deletion_record(code, day, "delete"))
        for code, day in DELETE_BEFORE.items():
            records.append(_deletion_record(code, day, "delete_before"))

        source_operations: dict[str, dict[str, str]] = {}
        for operation, mapping in (
            ("upsert", UPSERT_DATES),
            ("patch_preclose", PATCH_PRECLOSE_DATES),
        ):
            for code, dates in mapping.items():
                for day in dates:
                    if (code, day, operation) in SEALED_MULTISOURCE_RECORDS:
                        continue
                    if (code, day) in (
                        _ARCHIVED_MOOTDX_REPAIR_SET | _ARCHIVED_QMT_REPAIR_SET
                    ):
                        continue
                    by_date = source_operations.setdefault(code, {})
                    if day in by_date:
                        raise AssertionError(f"重复历史修复源日期: {code}/{day}")
                    by_date[day] = operation

        query_dates_by_code: dict[str, set[str]] = {
            code: set(operations) for code, operations in source_operations.items()
        }
        for (code, _day), successor in HISTORY_REPAIR_SUCCESSORS.items():
            query_dates_by_code.setdefault(code, set()).add(successor)

        for code, query_dates in sorted(query_dates_by_code.items()):
            dates = sorted(query_dates)
            symbol = _baostock_symbol(code)
            result = session.query(
                code,
                "baostock_verified_history_patch",
                lambda c=symbol, start=dates[0], end=dates[-1]: module.query_history_k_data_plus(
                    c,
                    QUERY_FIELDS,
                    start_date=start,
                    end_date=end,
                    frequency="d",
                    adjustflag="3",
                ),
            )
            missing_fields = set(QUERY_FIELDS.split(",")).difference(result.columns)
            if missing_fields:
                raise ValueError(
                    f"{code} Baostock 历史缺字段: {sorted(missing_fields)}"
                )
            result = result.copy()
            result["date"] = pd.to_datetime(
                result["date"], format="%Y-%m-%d", errors="raise"
            ).dt.strftime("%Y-%m-%d")
            result["code"] = result["code"].astype(str).str.strip().str.lower()
            if not result["code"].eq(symbol).all():
                raise ValueError(f"{code} Baostock 历史返回了错误市场/代码")
            if result["date"].duplicated().any():
                raise ValueError(f"{code} Baostock 历史日期重复")
            indexed = result.set_index("date", drop=False)
            uncovered = sorted(set(dates).difference(indexed.index))
            if uncovered:
                raise ValueError(f"{code} Baostock 目标日期未覆盖: {uncovered}")
            baostock_rows_by_code[code] = indexed
            for day, operation in sorted(source_operations.get(code, {}).items()):
                source = indexed.loc[day]
                evidence_hash = ""
                if (code, day) in _HISTORY_REPAIR_49_SET:
                    evidence_hash = str(
                        secondary_by_key[(code, day)]["payload_sha256"]
                    )
                successor_day = HISTORY_REPAIR_SUCCESSORS.get((code, day), "")
                successor = indexed.loc[successor_day] if successor_day else None
                records.append(
                    _source_record(
                        code=code,
                        day=day,
                        operation=operation,
                        source_values=source,
                        source_tradestatus=str(source["tradestatus"]).strip(),
                        effective_tradestatus=str(source["tradestatus"]).strip(),
                        executable_evidence_payload_sha256=evidence_hash,
                        successor_values=successor,
                    )
                )

                if (code, day) == ("300029.SZ", "2026-07-09"):
                    _validate_targeted_baostock_tencent_crosscheck(
                        source,
                        tencent_ohlcv,
                        secondary_by_key[(code, day)],
                    )

        # Post-IPO deletions require an independent calendar absence check;
        # an IPO-date assertion alone cannot justify removing those rows.
        for code, dates in sorted(DELETE_NONTRADING_DATES.items()):
            symbol = _baostock_symbol(code)
            result = session.query(
                code,
                "baostock_verified_nontrading_date",
                lambda c=symbol, start=min(dates), end=max(dates): module.query_history_k_data_plus(
                    c,
                    QUERY_FIELDS,
                    start_date=start,
                    end_date=end,
                    frequency="d",
                    adjustflag="3",
                ),
            )
            missing_fields = set(QUERY_FIELDS.split(",")).difference(result.columns)
            if missing_fields:
                raise ValueError(
                    f"{code} Baostock 非交易日核验缺字段: {sorted(missing_fields)}"
                )
            returned_dates = set(
                pd.to_datetime(
                    result["date"], format="%Y-%m-%d", errors="raise"
                ).dt.strftime("%Y-%m-%d")
            )
            contradicted = sorted(set(dates).intersection(returned_dates))
            if contradicted:
                raise ValueError(
                    f"{code} 声明污染日期被 Baostock 识别为交易日: {contradicted}"
                )
            for day in dates:
                records.append(_deletion_record(code, day, "delete_nontrading"))

    finally:
        session.close()

    for (code, day), source in sorted(archive_rows.items()):
        operation = "upsert"
        successor_day = HISTORY_REPAIR_SUCCESSORS.get((code, day), "")
        successor = (
            baostock_rows_by_code[code].loc[successor_day]
            if successor_day
            else None
        )
        source_values = {
            "open": source["open"],
            "high": source["high"],
            "low": source["low"],
            "close": source["close"],
            "preclose": source["preClose"],
            "volume": source["volume"],
            "amount": source["amount"],
        }
        records.append(
            _source_record(
                code=code,
                day=day,
                operation=operation,
                source_values=source_values,
                source_time_ms=int(source["time"]),
                source_tradestatus="",
                effective_tradestatus="1",
                executable_evidence_payload_sha256=str(
                    secondary_by_key[(code, day)]["payload_sha256"]
                ),
                successor_values=successor,
            )
        )

    for (code, day, operation), source in sorted(
        SEALED_MULTISOURCE_RECORDS.items()
    ):
        records.append(
            _source_record(
                code=code,
                day=day,
                operation=operation,
                source_values=source,
                source_tradestatus=str(source["tradestatus"]),
                effective_tradestatus=str(source["tradestatus"]),
            )
        )

    downloaded = pd.DataFrame.from_records(
        records,
        columns=[field for field in SNAPSHOT_COLUMNS if field != "payload_sha256"],
    )
    # Structural/source validation happens here.  The code-owned fixed
    # manifest is enforced by _stage_snapshot before any downloaded evidence
    # can be published; keeping the two checks separate also lets unit tests
    # exercise the downloader with deterministic fake source values.
    return _validate_snapshot(
        _attach_payload_hash(downloaded),
        expected_manifest_sha256=None,
    )


def _deletion_record(code: str, day: str, operation: str) -> dict[str, object]:
    source = _expected_source(code, day, operation)
    record: dict[str, object] = {
        "source": source,
        "schema_version": SNAPSHOT_SCHEMA,
        "code": code,
        "baostock_code": _baostock_symbol(code),
        "source_symbol": _expected_source_symbol(code, source),
        "source_artifact": _expected_source_artifact(code, day, operation),
        "source_time_ms": np.nan,
        "date": day,
        "ipo_date": EXPECTED_IPO_DATES[code],
        "source_tradestatus": "",
        "tradestatus": "",
        "operation": operation,
        "unit_contract": UNIT_CONTRACT_NONE,
        "turnover_policy": TURNOVER_NOT_WRITTEN,
        "executable_evidence_payload_sha256": "",
        "successor_date": "",
        "successor_preclose": np.nan,
        "successor_source_row_sha256": "",
    }
    record.update({field: np.nan for field in RAW_FIELDS})
    return record


def _temporary_path(path: Path, label: str) -> Path:
    return path.with_name(
        f".{path.name}.{os.getpid()}.{uuid4().hex}.{label}.tmp.parquet"
    )


def _stage_snapshot(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = _temporary_path(path, "history-evidence")
    try:
        frame.to_parquet(staged, index=False)
        verified = _load_snapshot(staged)
        pd.testing.assert_frame_equal(verified, frame, check_exact=True)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _read_kline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"历史修复目标 K 线不存在: {path}")
    frame = pd.read_parquet(path)
    if list(frame.columns) != list(PRODUCTION_COLUMNS) or frame.empty:
        raise ValueError(f"历史修复目标 K 线 schema/内容非法: {path.name}")
    if not pd.api.types.is_integer_dtype(frame["time"]):
        raise ValueError(f"历史修复目标 time 必须为 int: {path.name}")
    times = frame["time"].to_numpy(dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"历史修复目标 time 必须严格递增且唯一: {path.name}")
    timestamps = pd.to_datetime(times, unit="ms")
    offsets = times % np.int64(86_400_000)
    valid_offsets = {0, 15 * 60 * 60 * 1000}
    unique_offsets = set(int(value) for value in np.unique(offsets))
    if len(unique_offsets) != 1 or not unique_offsets.issubset(valid_offsets):
        raise ValueError(
            f"历史修复目标日线时刻不一致或不受支持: {path.name}/{unique_offsets}"
        )
    if not (
        (timestamps.minute == 0)
        & (timestamps.second == 0)
        & (timestamps.microsecond == 0)
    ).all():
        raise ValueError(f"历史修复目标日线时刻含非整点值: {path.name}")
    return frame


def _date_from_time(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["time"], unit="ms").dt.strftime("%Y-%m-%d")


def _time_ms(day: str, offset_ms: int = 15 * 60 * 60 * 1000) -> np.int64:
    return np.int64(pd.Timestamp(day).value // 10**6 + offset_ms)


def _repair_one(
    code: str,
    original: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    rows = evidence[evidence["code"].eq(code)].set_index(
        ["date", "operation"], drop=False
    )
    time_offset_ms = int(np.int64(original.iloc[0]["time"]) % 86_400_000)
    original_dates = _date_from_time(original)
    impacted_original = pd.Series(False, index=original.index)
    result = original.copy()

    if code in DELETE_BEFORE:
        cutoff = DELETE_BEFORE[code]
        mask = original_dates.lt(cutoff)
        impacted_original |= mask
        result = result.loc[~mask].copy()
    declared_deletes = set(DELETE_DATES.get(code, ()))
    if declared_deletes:
        mask = original_dates.isin(declared_deletes)
        impacted_original |= mask
        result = result.loc[~_date_from_time(result).isin(declared_deletes)].copy()
    invalid_session_dates = set(DELETE_NONTRADING_DATES.get(code, ()))
    if invalid_session_dates:
        mask = original_dates.isin(invalid_session_dates)
        impacted_original |= mask
        result = result.loc[
            ~_date_from_time(result).isin(invalid_session_dates)
        ].copy()

    upsert_dates = UPSERT_DATES.get(code, ())
    if upsert_dates:
        impacted_original |= original_dates.isin(upsert_dates)
        result = result.loc[~_date_from_time(result).isin(upsert_dates)].copy()
        production_rows: list[dict[str, float | np.int64]] = []
        for day in upsert_dates:
            source = rows.loc[(day, "upsert")]
            if isinstance(source, pd.DataFrame):
                raise ValueError(f"{code}/{day} upsert 证据不唯一")
            if source["turnover_policy"] == TURNOVER_PRESERVE:
                unit = UNIT_CONTRACTS[str(source["unit_contract"])]
                volume = float(source["volume"]) / float(
                    unit["volume_divisor"]
                )
                amount = float(source["amount"])
            elif source["turnover_policy"] == TURNOVER_NAN:
                volume = np.nan
                amount = np.nan
            else:
                raise ValueError(f"{code}/{day} upsert turnover_policy 非法")
            production_rows.append(
                {
                    "time": _time_ms(day, time_offset_ms),
                    "open": float(source["open"]),
                    "high": float(source["high"]),
                    "low": float(source["low"]),
                    "close": float(source["close"]),
                    "volume": volume,
                    "amount": amount,
                    "preClose": float(source["preclose"]),
                }
            )
        inserted = pd.DataFrame(production_rows, columns=PRODUCTION_COLUMNS)
        try:
            inserted = inserted.astype(original.dtypes.to_dict())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{code} K 线 dtype 无法容纳真实修复行") from exc
        result = pd.concat([result, inserted], ignore_index=True)

        successor_dates = {
            str(rows.loc[(day, "upsert")]["successor_date"])
            for day in upsert_dates
            if str(rows.loc[(day, "upsert")]["successor_date"])
        }
        if successor_dates:
            impacted_original |= original_dates.isin(successor_dates)
            result_dates = _date_from_time(result)
            for day in upsert_dates:
                source = rows.loc[(day, "upsert")]
                successor_day = str(source["successor_date"])
                if not successor_day:
                    continue
                target = result_dates.eq(successor_day)
                if int(target.sum()) != 1:
                    raise ValueError(
                        f"{code}/{day} successor {successor_day} 不唯一/不存在"
                    )
                result.loc[target, "preClose"] = float(
                    source["successor_preclose"]
                )

    patch_dates = PATCH_PRECLOSE_DATES.get(code, ())
    if patch_dates:
        impacted_original |= original_dates.isin(patch_dates)
        result_dates = _date_from_time(result)
        for day in patch_dates:
            target = result_dates.eq(day)
            if int(target.sum()) != 1:
                raise ValueError(f"{code} preClose 边界目标 {day} 不唯一/不存在")
            source = rows.loc[(day, "patch_preclose")]
            if isinstance(source, pd.DataFrame):
                raise ValueError(f"{code}/{day} preClose 证据不唯一")
            result.loc[target, "preClose"] = float(source["preclose"])

    result = result.sort_values("time", kind="stable").reset_index(drop=True)
    result_dates = _date_from_time(result)
    if result_dates.duplicated().any():
        raise ValueError(f"{code} 修复后日期重复")
    if result_dates.lt(EXPECTED_IPO_DATES[code]).any():
        raise ValueError(f"{code} 修复后仍含独立 IPO 日期之前的行情")

    unaffected_original = original.loc[~impacted_original].reset_index(drop=True)
    impacted_times = set(original.loc[impacted_original, "time"].astype(np.int64))
    impacted_times.update(_time_ms(day, time_offset_ms) for day in upsert_dates)
    impacted_times.update(_time_ms(day, time_offset_ms) for day in patch_dates)
    impacted_times.update(
        _time_ms(day, time_offset_ms)
        for day in (
            str(rows.loc[(upsert_day, "upsert")]["successor_date"])
            for upsert_day in upsert_dates
        )
        if day
    )
    unaffected_result = result.loc[
        ~result["time"].astype(np.int64).isin(impacted_times)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        unaffected_result,
        unaffected_original,
        check_exact=True,
        check_dtype=True,
    )

    for day in upsert_dates:
        source = rows.loc[(day, "upsert")]
        inserted_row = result.loc[result_dates.eq(day), ["volume", "amount"]]
        if source["turnover_policy"] == TURNOVER_NAN:
            if not inserted_row.isna().all().all():
                raise ValueError(f"{code}/{day} 不可靠 volume/amount 必须为 NaN")
        elif inserted_row.isna().any().any():
            raise ValueError(f"{code}/{day} 可靠 volume/amount 不得为空")
    return result


def _stage_kline(frame: pd.DataFrame, path: Path) -> Path:
    staged = _temporary_path(path, "history-kline")
    try:
        frame.to_parquet(staged, index=False)
        written = _read_kline(staged)
        pd.testing.assert_frame_equal(
            written,
            frame,
            check_exact=True,
            check_dtype=True,
        )
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def ensure_verified_history_repairs(
    kline_dir: Path = RAW_DIR,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    baostock_module=None,
    *,
    refresh_snapshot: bool = False,
    mootdx_archive_dir: Path = DEFAULT_MOOTDX_ARCHIVE_DIR,
    qmt_archive_dir: Path = DEFAULT_QMT_ARCHIVE_DIR,
    secondary_snapshot_path: Path = DEFAULT_SECONDARY_SNAPSHOT_PATH,
    tencent_ohlcv: dict[str, object] | None = None,
    tencent_timeout: float = 30.0,
) -> tuple[str, ...]:
    """Apply the sealed, finite history repair set and return changed codes.

    If ``snapshot_path`` exists this function is fully offline unless the
    updater explicitly requests ``refresh_snapshot``.  Otherwise all Baostock
    evidence is downloaded and validated as one batch before any production
    parquet is replaced.  Every changed parquet is staged and read back first;
    snapshot/K files are published by one rollback-protected transaction.
    Content-addressed pre-publication checks reject concurrent source changes.
    """

    kline_dir = Path(kline_dir)
    snapshot_path = Path(snapshot_path)
    mootdx_archive_dir = Path(mootdx_archive_dir)
    qmt_archive_dir = Path(qmt_archive_dir)
    secondary_snapshot_path = Path(secondary_snapshot_path)
    staged_snapshot: Path | None = None
    staged_klines: dict[str, Path] = {}
    kline_paths = {
        code: kline_dir / f"{code}.parquet" for code in TARGET_CODES
    }
    protected_state = capture_file_states(
        [*kline_paths.values(), snapshot_path]
    )
    refresh_sources: dict[Path, str | None] = {}
    if refresh_snapshot or not snapshot_path.exists():
        archive_paths = {
            *(
                mootdx_archive_dir / f"{code}.parquet"
                for code, _day in _ARCHIVED_MOOTDX_REPAIR_SET
            ),
            *(
                qmt_archive_dir / f"{code}.parquet"
                for code, _day in _ARCHIVED_QMT_REPAIR_SET
            ),
            secondary_snapshot_path,
        }
        refresh_sources = capture_file_states(archive_paths)
    try:
        if snapshot_path.exists() and not refresh_snapshot:
            evidence = _load_snapshot(snapshot_path)
        else:
            evidence = _download_snapshot(
                baostock_module,
                mootdx_archive_dir=mootdx_archive_dir,
                qmt_archive_dir=qmt_archive_dir,
                secondary_snapshot_path=secondary_snapshot_path,
                tencent_ohlcv=tencent_ohlcv,
                tencent_timeout=tencent_timeout,
            )
            staged_snapshot = _stage_snapshot(evidence, snapshot_path)

        originals: dict[str, pd.DataFrame] = {}
        repaired: dict[str, pd.DataFrame] = {}
        for code in TARGET_CODES:
            originals[code] = _read_kline(kline_paths[code])
            repaired[code] = _repair_one(code, originals[code], evidence)

        changed = tuple(
            code
            for code in TARGET_CODES
            if not repaired[code].equals(originals[code])
        )
        for code in changed:
            staged_klines[code] = _stage_kline(
                repaired[code], kline_paths[code]
            )

        assert_file_states_unchanged(protected_state, label="history K/snapshot")
        if refresh_sources:
            assert_file_states_unchanged(
                refresh_sources,
                label="history source archive/secondary",
            )
        pairs = [
            (staged_klines[code], kline_paths[code]) for code in changed
        ]
        if staged_snapshot is not None:
            pairs.append((staged_snapshot, snapshot_path))
        if pairs:
            replace_staged_files_transactionally(
                pairs,
                token=f"{os.getpid()}.{uuid4().hex}.history-v5",
            )
            staged_snapshot = None
        return changed
    finally:
        if staged_snapshot is not None:
            staged_snapshot.unlink(missing_ok=True)
        for path in staged_klines.values():
            path.unlink(missing_ok=True)


__all__ = [
    "DEFAULT_SNAPSHOT_PATH",
    "SAME_CODE_PREDECESSOR_FIRST_EXECUTABLE",
    "ensure_verified_history_repairs",
]
