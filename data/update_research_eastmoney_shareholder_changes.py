"""Download a frozen PIT-oriented shareholder-change research snapshot.

This is a research-only pre-download entry.  Strategy and backtest modules
must read the resulting parquet and must never call Eastmoney directly.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = (
    ROOT
    / "results"
    / "strategy_opt_20260730"
    / "agent_research"
)
OUTPUT = RESEARCH_DIR / "shareholder_changes_2008_2018.parquet"
METADATA = (
    RESEARCH_DIR
    / "shareholder_changes_2008_2018_fetch_metadata.json"
)
URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORT_NAME = "RPT_SHARE_HOLDER_INCREASE"
FILTER = (
    "(NOTICE_DATE>='2008-01-01')"
    "(NOTICE_DATE<='2018-12-31')"
)
SORT_COLUMNS = "NOTICE_DATE,SECURITY_CODE,EITIME"
SORT_TYPES = "1,1,1"
PAGE_SIZE = 500
MIN_INTERVAL_SECONDS = 1.0
MAX_JITTER_SECONDS = 0.35
_last_request_time = 0.0


def em_get(
    session: requests.Session,
    *,
    params: dict[str, str],
) -> requests.Response:
    """Serial Eastmoney request with session reuse and jittered throttle."""
    global _last_request_time
    target_interval = (
        MIN_INTERVAL_SECONDS
        + random.uniform(0.0, MAX_JITTER_SECONDS)
    )
    elapsed = time.monotonic() - _last_request_time
    if elapsed < target_interval:
        time.sleep(target_interval - elapsed)
    response = session.get(URL, params=params, timeout=30)
    _last_request_time = time.monotonic()
    return response


def fetch_page(
    session: requests.Session,
    page_number: int,
) -> dict:
    params = {
        "reportName": REPORT_NAME,
        "columns": "ALL",
        "filter": FILTER,
        "pageNumber": str(page_number),
        "pageSize": str(PAGE_SIZE),
        "sortColumns": SORT_COLUMNS,
        "sortTypes": SORT_TYPES,
        "source": "WEB",
        "client": "WEB",
    }
    response = em_get(session, params=params)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(
            f"Eastmoney request failed: {payload.get('message')}"
        )
    result = payload.get("result")
    if not result:
        raise RuntimeError("Eastmoney returned no result")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    fetched_at = datetime.now(timezone.utc).isoformat()
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            ),
            "Referer": "https://data.eastmoney.com/",
        }
    )
    first = fetch_page(session, 1)
    pages = int(first.get("pages") or 1)
    expected_count = int(first.get("count") or 0)
    rows = list(first.get("data") or [])
    print(
        f"page 1/{pages}: {len(rows)} rows; expected={expected_count}",
        flush=True,
    )
    for page in range(2, pages + 1):
        result = fetch_page(session, page)
        page_rows = list(result.get("data") or [])
        rows.extend(page_rows)
        print(
            f"page {page}/{pages}: +{len(page_rows)} "
            f"rows, total={len(rows)}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    required = [
        "SECURITY_CODE",
        "NOTICE_DATE",
        "EITIME",
        "DIRECTION",
        "CHANGE_NUM",
        "CHANGE_NUM_SYMBOL",
        "CHANGE_RATE",
        "CHANGE_FREE_RATIO",
        "TRADE_DATE",
        "START_DATE",
        "END_DATE",
        "HOLDER_NAME",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(
            f"shareholder-change response missing {missing}; "
            f"available={list(frame.columns)}"
        )
    frame = frame[required].copy()
    frame["SECURITY_CODE"] = (
        frame["SECURITY_CODE"].astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )
    for column in (
        "NOTICE_DATE",
        "EITIME",
        "TRADE_DATE",
        "START_DATE",
        "END_DATE",
    ):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in (
        "CHANGE_NUM",
        "CHANGE_RATE",
        "CHANGE_FREE_RATIO",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in (
        "DIRECTION",
        "CHANGE_NUM_SYMBOL",
        "HOLDER_NAME",
    ):
        frame[column] = frame[column].fillna("").astype(str)
    frame = frame[
        frame["NOTICE_DATE"].notna()
        & frame["SECURITY_CODE"].str.match(r"^\d{6}$")
    ].sort_values(
        ["NOTICE_DATE", "SECURITY_CODE", "EITIME", "HOLDER_NAME"],
        kind="stable",
    )
    if expected_count and len(frame) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} rows but got {len(frame)}"
        )
    frame.to_parquet(OUTPUT, index=False)
    metadata = {
        "fetched_at_utc": fetched_at,
        "url": URL,
        "report_name": REPORT_NAME,
        "filter": FILTER,
        "sort_columns": SORT_COLUMNS,
        "sort_types": SORT_TYPES,
        "page_size": PAGE_SIZE,
        "pages": pages,
        "expected_count": expected_count,
        "materialized_rows": int(len(frame)),
        "unique_stocks": int(frame["SECURITY_CODE"].nunique()),
        "unique_stock_notice_dates": int(
            frame[["SECURITY_CODE", "NOTICE_DATE"]]
            .drop_duplicates()
            .shape[0]
        ),
        "notice_date_min": str(frame["NOTICE_DATE"].min().date()),
        "notice_date_max": str(frame["NOTICE_DATE"].max().date()),
        "source_columns": required,
        "output_columns": list(frame.columns),
        "output": OUTPUT.name,
    }
    METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata["output_sha256"] = sha256(OUTPUT)
    METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
