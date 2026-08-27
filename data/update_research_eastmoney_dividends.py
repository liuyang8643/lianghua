"""Download implemented cash-dividend events through validation end."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "strategy_opt_20260730"

from update_eastmoney_research_common import em_get  # noqa: E402

OUTPUT_PATH = RESULT_DIR / "eastmoney_dividends_2010_2022.parquet"
STATUS_PATH = RESULT_DIR / "eastmoney_dividends_2010_2022_status.json"
START = "2010-01-01"
END = "2022-12-31"
PAGE_SIZE = 500
CHECKPOINT_EVERY_PAGES = 10
FIELDS = (
    "SECUCODE",
    "SECURITY_CODE",
    "PRETAX_BONUS_RMB",
    "EX_DIVIDEND_DATE",
    "EQUITY_RECORD_DATE",
    "PLAN_NOTICE_DATE",
    "NOTICE_DATE",
    "PUBLISH_DATE",
    "REPORT_DATE",
    "ASSIGN_PROGRESS",
    "IMPL_PLAN_PROFILE",
    "TOTAL_SHARES",
)


def checkpoint(
    rows: list[dict],
    *,
    completed_page: int,
    source_pages: int,
    source_count: int,
    complete: bool,
) -> None:
    frame = pd.DataFrame(rows, columns=FIELDS)
    temporary = OUTPUT_PATH.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, OUTPUT_PATH)
    status = {
        "source": "Eastmoney RPT_SHAREBONUS_DET",
        "temporal_contract": (
            "implemented cash dividend with ex-date T activates only on "
            "the first trading day strictly after T"
        ),
        "start": START,
        "end": END,
        "page_size": PAGE_SIZE,
        "source_pages": source_pages,
        "source_count": source_count,
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
    source_count = 0
    if OUTPUT_PATH.exists() and STATUS_PATH.exists():
        rows = pd.read_parquet(OUTPUT_PATH).to_dict("records")
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        if status.get("complete"):
            print(f"already complete: {len(rows)} rows", flush=True)
            return
        start_page = int(status.get("completed_page", 0)) + 1
        source_pages = int(status.get("source_pages", 0))
        source_count = int(status.get("source_count", 0))

    page = start_page
    while not source_pages or page <= source_pages:
        response = em_get(
            {
                "reportName": "RPT_SHAREBONUS_DET",
                "columns": ",".join(FIELDS),
                "filter": (
                    f"(EX_DIVIDEND_DATE>='{START}')"
                    f"(EX_DIVIDEND_DATE<='{END}')"
                ),
                "pageNumber": str(page),
                "pageSize": str(PAGE_SIZE),
                "sortColumns": "EX_DIVIDEND_DATE,SECURITY_CODE",
                "sortTypes": "1,1",
                "source": "WEB",
                "client": "WEB",
            }
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(
                f"page {page}: {payload.get('message') or payload}"
            )
        result = payload.get("result") or {}
        source_pages = int(result.get("pages") or 0)
        source_count = int(result.get("count") or 0)
        page_rows = result.get("data") or []
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
                source_count=source_count,
                complete=complete,
            )
        print(
            f"page={page}/{source_pages} rows={len(page_rows)} "
            f"stored={len(rows)}/{source_count}",
            flush=True,
        )
        page += 1

    if len(rows) != source_count:
        raise RuntimeError(
            f"received {len(rows)} of {source_count} source rows"
        )


if __name__ == "__main__":
    main()
