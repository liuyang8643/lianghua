"""Download official CNINFO performance-express correction events."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from calendar import monthrange
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
    / "cninfo_performance_express_corrections_2008_2018.parquet"
)
METADATA = (
    RESEARCH_DIR
    / "cninfo_performance_express_corrections_2008_2018_fetch_metadata.json"
)
URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
START_DATE = "2008-01-01"
END_DATE = "2018-12-31"
SEARCH_KEYS = (
    "业绩快报修正",
)
PAGE_SIZE = 30
MIN_INTERVAL_SECONDS = 0.75
MAX_JITTER_SECONDS = 0.25
CORRECTION_PATTERN = re.compile(
    r"业绩快报.*(?:修正|更正|修订|纠正)"
    r"|(?:修正|更正|修订|纠正).*业绩快报"
)
_last_request_time = 0.0


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            ),
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://www.cninfo.com.cn/new/disclosure",
            "Origin": "https://www.cninfo.com.cn",
        }
    )
    return session


def cninfo_post(
    session: requests.Session,
    *,
    search_key: str,
    page_number: int,
    start_date: str,
    end_date: str,
    sort_type: str = "asc",
) -> dict:
    global _last_request_time
    target_interval = (
        MIN_INTERVAL_SECONDS
        + random.uniform(0.0, MAX_JITTER_SECONDS)
    )
    elapsed = time.monotonic() - _last_request_time
    if elapsed < target_interval:
        time.sleep(target_interval - elapsed)
    response = None
    data = {
        "stock": "",
        "tabName": "fulltext",
        "pageSize": str(PAGE_SIZE),
        "pageNum": str(page_number),
        "column": "",
        "category": "",
        "plate": "",
        "seDate": f"{start_date}~{end_date}",
        "searchkey": search_key,
        "secid": "",
        "sortName": "time",
        "sortType": sort_type,
        "isHLtitle": "true",
    }
    for attempt in range(5):
        response = session.post(URL, data=data, timeout=30)
        _last_request_time = time.monotonic()
        if response.status_code not in {
            429,
            500,
            502,
            503,
            504,
        }:
            break
        time.sleep(min(5.0, 0.75 * (2**attempt)))
    if response is None:
        raise RuntimeError("CNINFO request produced no response")
    response.raise_for_status()
    payload = response.json()
    if (
        payload.get("announcements") is None
        and int(payload.get("totalAnnouncement") or 0) > 0
    ):
        raise RuntimeError(
            f"CNINFO query {search_key!r} returned no announcements"
        )
    return payload


def fetch_query(
    session: requests.Session,
    search_key: str,
    year: int,
    month: int,
) -> tuple[list[dict], dict]:
    start_date = f"{year}-{month:02d}-01"
    end_date = (
        f"{year}-{month:02d}-"
        f"{monthrange(year, month)[1]:02d}"
    )
    first = cninfo_post(
        session,
        search_key=search_key,
        page_number=1,
        start_date=start_date,
        end_date=end_date,
    )
    advertised = int(first.get("totalAnnouncement") or 0)
    pages = max(
        int(first.get("totalpages") or 0),
        int(math.ceil(advertised / PAGE_SIZE)),
    )
    rows = list(first.get("announcements") or [])
    for page in range(2, pages + 1):
        payload = cninfo_post(
            session,
            search_key=search_key,
            page_number=page,
            start_date=start_date,
            end_date=end_date,
        )
        rows.extend(payload.get("announcements") or [])
    unique = {}
    for row in rows:
        announcement_id = str(row.get("announcementId") or "")
        if announcement_id:
            unique[announcement_id] = row
    ascending_unique_ids = len(unique)
    descending_fetched_rows = 0
    if len(unique) < advertised:
        for page in range(1, pages + 1):
            payload = cninfo_post(
                session,
                search_key=search_key,
                page_number=page,
                start_date=start_date,
                end_date=end_date,
                sort_type="desc",
            )
            descending_rows = list(
                payload.get("announcements") or []
            )
            descending_fetched_rows += len(descending_rows)
            for row in descending_rows:
                announcement_id = str(
                    row.get("announcementId") or ""
                )
                if announcement_id:
                    unique[announcement_id] = row
    if len(unique) < advertised:
        raise RuntimeError(
            f"{search_key}: advertised {advertised}, "
            f"only {len(unique)} unique IDs"
        )
    return list(unique.values()), {
        "year": year,
        "month": month,
        "advertised_count": advertised,
        "pages": pages,
        "raw_fetched_rows": int(len(rows)),
        "unique_announcement_ids": int(len(unique)),
        "ascending_duplicate_page_rows": int(
            len(rows) - ascending_unique_ids
        ),
        "descending_fetched_rows_for_recovery": (
            descending_fetched_rows
        ),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    session = make_session()
    rows = []
    query_metadata = {}
    for search_key in SEARCH_KEYS:
        query_metadata[search_key] = {}
        for year in range(2008, 2019):
            query_metadata[search_key][str(year)] = {}
            for month in range(1, 13):
                query_rows, query_audit = fetch_query(
                    session,
                    search_key,
                    year,
                    month,
                )
                rows.extend(query_rows)
                query_metadata[search_key][str(year)][
                    f"{month:02d}"
                ] = query_audit
        print(
            f"{search_key}: yearly partitions complete",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    required = [
        "announcementId",
        "announcementTitle",
        "announcementTime",
        "secCode",
        "secName",
        "adjunctUrl",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(
            f"CNINFO corrections lack {missing}; "
            f"available={list(frame.columns)}"
        )
    retained = [
        column
        for column in (
            *required,
            "orgId",
            "announcementType",
        )
        if column in frame
    ]
    frame = frame[retained].copy()
    frame["announcementTitle"] = (
        frame["announcementTitle"]
        .astype(str)
        .map(lambda value: re.sub(r"<[^>]+>", "", value))
    )
    frame["announcementTime"] = (
        pd.to_datetime(
            frame["announcementTime"],
            unit="ms",
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    frame["secCode"] = (
        frame["secCode"].astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )
    exact_title = frame["announcementTitle"].map(
        lambda value: bool(CORRECTION_PATTERN.search(value))
    )
    frame = frame.loc[
        exact_title
        & frame["announcementId"].notna()
        & frame["announcementTime"].notna()
        & frame["secCode"].str.match(r"^\d{6}$")
    ].drop_duplicates(
        "announcementId",
        keep="first",
    ).sort_values(
        ["announcementTime", "secCode", "announcementId"],
        kind="stable",
    )
    if frame.empty:
        raise RuntimeError(
            "no exact performance-express correction titles"
        )
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    metadata = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "url": URL,
        "date_filter": f"{START_DATE}~{END_DATE}",
        "search_keys": list(SEARCH_KEYS),
        "query_metadata": query_metadata,
        "page_size": PAGE_SIZE,
        "title_filter": CORRECTION_PATTERN.pattern,
        "materialized_rows": int(len(frame)),
        "unique_announcement_ids": int(
            frame["announcementId"].nunique()
        ),
        "unique_stocks": int(frame["secCode"].nunique()),
        "announcement_time_min": str(
            frame["announcementTime"].min()
        ),
        "announcement_time_max": str(
            frame["announcementTime"].max()
        ),
        "timestamp_timezone": "Asia/Shanghai",
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
