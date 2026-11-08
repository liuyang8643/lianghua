"""Download publication-vintage analyst reports through validation end."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "strategy_opt_20260730"
OUTPUT_PATH = RESULT_DIR / "eastmoney_analyst_reports_2010_2022.parquet"
STATUS_PATH = RESULT_DIR / "eastmoney_analyst_reports_2010_2022_status.json"
URL = "https://reportapi.eastmoney.com/report/list"
START = "2010-01-01"
END = "2022-12-31"
PAGE_SIZE = 100
CHECKPOINT_EVERY_PAGES = 50
FIELDS = (
    "stockCode",
    "publishDate",
    "infoCode",
    "orgCode",
    "reportType",
    "emRatingValue",
    "emRatingName",
    "lastEmRatingValue",
    "lastEmRatingName",
    "ratingChange",
    "predictLastYearEps",
    "predictThisYearEps",
    "predictNextYearEps",
    "predictNextTwoYearEps",
    "predictLastYearPe",
    "predictThisYearPe",
    "predictNextYearPe",
    "predictNextTwoYearPe",
    "indvAimPriceL",
    "indvAimPriceT",
    "sRatingName",
)


class EastmoneyReportClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "Chrome/117.0.0.0 Safari/537.36"
                ),
                "Referer": "https://data.eastmoney.com/",
            }
        )
        self.last_call = 0.0

    def get(self, page: int) -> dict:
        wait = 1.0 - (time.time() - self.last_call)
        if wait > 0.0:
            time.sleep(wait + random.uniform(0.1, 0.5))
        try:
            response = self.session.get(
                URL,
                params={
                    "industryCode": "*",
                    "pageSize": str(PAGE_SIZE),
                    "industry": "*",
                    "rating": "*",
                    "ratingChange": "*",
                    "beginTime": START,
                    "endTime": END,
                    "pageNo": str(page),
                    "fields": "",
                    "qType": "0",
                    "orgCode": "",
                    "code": "",
                    "rcode": "",
                    "p": str(page),
                    "pageNum": str(page),
                    "pageNumber": str(page),
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        finally:
            self.last_call = time.time()


def checkpoint(
    rows: list[dict],
    *,
    completed_page: int,
    source_pages: int,
    complete: bool,
) -> None:
    frame = pd.DataFrame(rows, columns=FIELDS).astype("string")
    temporary = OUTPUT_PATH.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, OUTPUT_PATH)
    status = {
        "source": "Eastmoney reportapi report/list",
        "temporal_contract": (
            "report published on date T activates only on the first "
            "trading day strictly after T"
        ),
        "start": START,
        "end": END,
        "page_size": PAGE_SIZE,
        "source_pages": source_pages,
        "completed_page": completed_page,
        "stored_rows": len(frame),
        "complete": complete,
        "test_source_downloaded": False,
    }
    temporary_status = STATUS_PATH.with_suffix(".tmp.json")
    temporary_status.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_status, STATUS_PATH)


def main() -> None:
    rows: list[dict] = []
    start_page = 1
    source_pages = 0
    if OUTPUT_PATH.exists() and STATUS_PATH.exists():
        rows = pd.read_parquet(OUTPUT_PATH).to_dict("records")
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        if status.get("complete"):
            print(f"already complete: {len(rows)} rows", flush=True)
            return
        start_page = int(status.get("completed_page", 0)) + 1
        source_pages = int(status.get("source_pages", 0))

    client = EastmoneyReportClient()
    page = start_page
    while not source_pages or page <= source_pages:
        payload = client.get(page)
        source_pages = int(payload.get("TotalPage") or 0)
        page_rows = payload.get("data") or []
        rows.extend(
            {field: row.get(field) for field in FIELDS}
            for row in page_rows
        )
        complete = page == source_pages
        if page % CHECKPOINT_EVERY_PAGES == 0 or complete:
            checkpoint(
                rows,
                completed_page=page,
                source_pages=source_pages,
                complete=complete,
            )
        print(
            f"page={page}/{source_pages} rows={len(page_rows)} "
            f"stored={len(rows)}",
            flush=True,
        )
        page += 1


if __name__ == "__main__":
    main()
