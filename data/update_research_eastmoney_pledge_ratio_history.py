"""Download historical CSDC stock-level pledge-ratio snapshots."""

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
OUTPUT = RESEARCH_DIR / "pledge_ratio_history_2017_2018.parquet"
METADATA = (
    RESEARCH_DIR
    / "pledge_ratio_history_2017_2018_fetch_metadata.json"
)
URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORT_NAME = "RPT_CSDC_LIST"
PROFILE_REPORT_NAME = "RPT_CSDC_STATISTICS"
FILTER = (
    "(TRADE_DATE>='2017-01-01')"
    "(TRADE_DATE<='2018-12-31')"
)
SORT_COLUMNS = "TRADE_DATE,SECURITY_CODE"
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
    *,
    report_name: str,
    filter_value: str,
    sort_columns: str,
    sort_types: str,
) -> dict:
    params = {
        "reportName": report_name,
        "columns": "ALL",
        "filter": filter_value,
        "pageNumber": str(page_number),
        "pageSize": str(PAGE_SIZE),
        "sortColumns": sort_columns,
        "sortTypes": sort_types,
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
    profile_first = fetch_page(
        session,
        1,
        report_name=PROFILE_REPORT_NAME,
        filter_value=FILTER,
        sort_columns="TRADE_DATE",
        sort_types="1",
    )
    profile_pages = int(profile_first.get("pages") or 1)
    profile_rows = list(profile_first.get("data") or [])
    for page in range(2, profile_pages + 1):
        profile_result = fetch_page(
            session,
            page,
            report_name=PROFILE_REPORT_NAME,
            filter_value=FILTER,
            sort_columns="TRADE_DATE",
            sort_types="1",
        )
        profile_rows.extend(profile_result.get("data") or [])
    profile = pd.DataFrame(profile_rows)
    if "TRADE_DATE" not in profile:
        raise RuntimeError(
            f"profile response lacks TRADE_DATE: {list(profile.columns)}"
        )
    profile_dates = pd.to_datetime(
        profile["TRADE_DATE"],
        errors="coerce",
    ).dropna()
    monthly_dates = (
        profile_dates.groupby(
            profile_dates.dt.to_period("M"),
            sort=True,
        )
        .max()
        .sort_values()
    )
    if monthly_dates.empty:
        raise RuntimeError("no CSDC profile dates in requested range")

    rows = []
    expected_count = 0
    pages = 0
    per_date_counts = {}
    for date_value in monthly_dates:
        date_text = str(date_value.date())
        date_filter = f"(TRADE_DATE='{date_text}')"
        first = fetch_page(
            session,
            1,
            report_name=REPORT_NAME,
            filter_value=date_filter,
            sort_columns=SORT_COLUMNS,
            sort_types=SORT_TYPES,
        )
        date_pages = int(first.get("pages") or 1)
        date_count = int(first.get("count") or 0)
        date_rows = list(first.get("data") or [])
        for page in range(2, date_pages + 1):
            result = fetch_page(
                session,
                page,
                report_name=REPORT_NAME,
                filter_value=date_filter,
                sort_columns=SORT_COLUMNS,
                sort_types=SORT_TYPES,
            )
            date_rows.extend(result.get("data") or [])
        if len(date_rows) != date_count:
            raise RuntimeError(
                f"{date_text}: expected {date_count}, "
                f"received {len(date_rows)}"
            )
        rows.extend(date_rows)
        expected_count += date_count
        pages += date_pages
        per_date_counts[date_text] = date_count
        print(
            f"{date_text}: {date_count} rows in {date_pages} pages; "
            f"total={len(rows)}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    required = [
        "SECURITY_CODE",
        "TRADE_DATE",
        "PLEDGE_RATIO",
        "REPURCHASE_BALANCE",
        "PLEDGE_DEAL_NUM",
        "REPURCHASE_UNLIMITED_BALANCE",
        "REPURCHASE_LIMITED_BALANCE",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(
            f"pledge-ratio response missing {missing}; "
            f"available={list(frame.columns)}"
        )
    frame = frame[required].copy()
    frame["SECURITY_CODE"] = (
        frame["SECURITY_CODE"].astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )
    frame["TRADE_DATE"] = pd.to_datetime(
        frame["TRADE_DATE"],
        errors="coerce",
    )
    for column in required[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["TRADE_DATE"].notna()
        & frame["SECURITY_CODE"].str.match(r"^\d{6}$")
    ].sort_values(
        ["TRADE_DATE", "SECURITY_CODE"],
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
        "profile_report_name": PROFILE_REPORT_NAME,
        "profile_filter": FILTER,
        "sampling_policy": (
            "last official CSDC TRADE_DATE in each calendar month"
        ),
        "selected_trade_dates": [
            str(value.date()) for value in monthly_dates
        ],
        "per_date_counts": per_date_counts,
        "sort_columns": SORT_COLUMNS,
        "sort_types": SORT_TYPES,
        "page_size": PAGE_SIZE,
        "profile_pages": profile_pages,
        "detail_pages": pages,
        "expected_count": expected_count,
        "materialized_rows": int(len(frame)),
        "unique_stocks": int(frame["SECURITY_CODE"].nunique()),
        "unique_trade_dates": int(frame["TRADE_DATE"].nunique()),
        "date_min": str(frame["TRADE_DATE"].min().date()),
        "date_max": str(frame["TRADE_DATE"].max().date()),
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
