"""Probe/download Eastmoney historical preliminary earnings releases."""

from __future__ import annotations

import argparse
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
OUTPUT = (
    RESEARCH_DIR
    / "performance_express_2008_2018.parquet"
)
METADATA = (
    RESEARCH_DIR
    / "performance_express_2008_2018_fetch_metadata.json"
)
URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORT_NAME = "RPT_FCI_PERFORMANCEE"
FILTER = (
    "(NOTICE_DATE>='2008-01-01')"
    "(NOTICE_DATE<='2018-12-31')"
)
SORT_COLUMNS = "NOTICE_DATE,SECURITY_CODE"
SORT_TYPES = "1,1"
PAGE_SIZE = 500
MIN_INTERVAL_SECONDS = 1.0
MAX_JITTER_SECONDS = 0.35
_last_request_time = 0.0


def em_get(
    session: requests.Session,
    *,
    params: dict[str, str],
) -> requests.Response:
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
    response = em_get(
        session,
        params={
            "reportName": REPORT_NAME,
            "columns": "ALL",
            "filter": FILTER,
            "pageNumber": str(page_number),
            "pageSize": str(PAGE_SIZE),
            "sortColumns": SORT_COLUMNS,
            "sortTypes": SORT_TYPES,
            "source": "WEB",
            "client": "WEB",
        },
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success") or not payload.get("result"):
        raise RuntimeError(
            f"Eastmoney request failed: {payload.get('message')}"
        )
    return payload["result"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        action="store_true",
        help="materialize all pages after source schema inspection",
    )
    args = parser.parse_args()
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
    first_rows = list(first.get("data") or [])
    if not first_rows:
        raise RuntimeError("performance-express report returned no rows")
    schema = list(first_rows[0])
    probe = {
        "report_name": REPORT_NAME,
        "filter": FILTER,
        "count": int(first.get("count") or 0),
        "pages": int(first.get("pages") or 1),
        "schema": schema,
        "first_row": first_rows[0],
    }
    if not args.download:
        print(json.dumps(probe, ensure_ascii=False, indent=2))
        return
    pages = int(first.get("pages") or 1)
    expected_count = int(first.get("count") or 0)
    rows = first_rows
    for page in range(2, pages + 1):
        result = fetch_page(session, page)
        rows.extend(result.get("data") or [])
        print(
            f"page {page}/{pages}; rows={len(rows)}",
            flush=True,
        )
    if len(rows) != expected_count:
        raise RuntimeError(
            f"expected {expected_count}, received {len(rows)}"
        )
    frame = pd.DataFrame(rows)
    required = [
        "SECURITY_CODE",
        "NOTICE_DATE",
        "REPORT_DATE",
        "BASIC_EPS",
        "PARENT_BVPS",
        "WEIGHTAVG_ROE",
        "YSTZ",
        "JLRTBZCL",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(
            f"performance express lacks {missing}; "
            f"available={list(frame.columns)}"
        )
    retained = [
        column
        for column in (
            *required,
            "TOTAL_OPERATE_INCOME",
            "TOTAL_OPERATE_INCOME_SQ",
            "PARENT_NETPROFIT",
            "PARENT_NETPROFIT_SQ",
            "UPDATE_DATE",
            "EITIME",
            "QDATE",
            "DATATYPE",
            "ISNEW",
            "SECUCODE",
            "SECURITY_NAME_ABBR",
            "ORG_CODE",
        )
        if column in frame
    ]
    frame = frame[retained].copy()
    frame["SECURITY_CODE"] = (
        frame["SECURITY_CODE"].astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )
    for column in (
        "NOTICE_DATE",
        "REPORT_DATE",
        "UPDATE_DATE",
        "EITIME",
    ):
        if column in frame:
            frame[column] = pd.to_datetime(
                frame[column],
                errors="coerce",
            )
    numeric = [
        column
        for column in retained
        if column
        not in {
            "SECURITY_CODE",
            "NOTICE_DATE",
            "REPORT_DATE",
            "UPDATE_DATE",
            "EITIME",
            "SECUCODE",
            "SECURITY_NAME_ABBR",
            "ORG_CODE",
            "QDATE",
            "DATATYPE",
            "ISNEW",
        }
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )
    frame = frame[
        frame["NOTICE_DATE"].notna()
        & frame["REPORT_DATE"].notna()
        & frame["SECURITY_CODE"].str.match(r"^\d{6}$")
    ].sort_values(
        ["NOTICE_DATE", "SECURITY_CODE", "REPORT_DATE"],
        kind="stable",
    )
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    metadata = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
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
        "unique_notice_dates": int(frame["NOTICE_DATE"].nunique()),
        "notice_date_min": str(frame["NOTICE_DATE"].min().date()),
        "notice_date_max": str(frame["NOTICE_DATE"].max().date()),
        "source_schema": schema,
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
