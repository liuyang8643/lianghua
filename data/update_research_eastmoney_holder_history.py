"""Checkpointed PIT shareholder-count history download from Eastmoney."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from update_eastmoney_research_common import em_get

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "strategy_opt_20260730"
OUTPUT_PATH = RESULT_DIR / "eastmoney_holder_history_2013_2022.parquet"
STATUS_PATH = RESULT_DIR / "eastmoney_holder_history_2013_2022_status.json"
NOTICE_START = "2013-01-01"
NOTICE_END = "2022-12-31"
PAGE_SIZE = 500
CHECKPOINT_EVERY = 25
FIELDS = (
    "SECUCODE",
    "SECURITY_CODE",
    "END_DATE",
    "PRE_END_DATE",
    "HOLD_NOTICE_DATE",
    "HOLDER_NUM",
    "PRE_HOLDER_NUM",
    "HOLDER_NUM_CHANGE",
    "HOLDER_NUM_RATIO",
    "TOTAL_A_SHARES",
    "CHANGE_SHARES",
)


def checkpoint(
    rows: list[dict],
    *,
    completed_page: int,
    pages: int,
    count: int,
    complete: bool,
) -> None:
    frame = pd.DataFrame(rows, columns=FIELDS)
    if not frame.empty:
        frame = frame.drop_duplicates(
            subset=["SECUCODE", "END_DATE", "HOLD_NOTICE_DATE"],
            keep="last",
        ).sort_values(
            ["HOLD_NOTICE_DATE", "SECURITY_CODE", "END_DATE"],
            kind="stable",
        )
    temporary = OUTPUT_PATH.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, OUTPUT_PATH)
    status = {
        "source": "Eastmoney RPT_HOLDERNUM_DET",
        "temporal_contract": (
            "retain HOLD_NOTICE_DATE and activate only from the next trading "
            "day; END_DATE alone is never treated as availability"
        ),
        "notice_start": NOTICE_START,
        "notice_end": NOTICE_END,
        "page_size": PAGE_SIZE,
        "pages": pages,
        "source_count": count,
        "completed_page": completed_page,
        "stored_rows": len(frame),
        "complete": complete,
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
    if OUTPUT_PATH.exists() and STATUS_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        rows = existing.to_dict("records")
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        if status.get("complete"):
            print(f"already complete: {len(rows)} rows", flush=True)
            return
        start_page = int(status.get("completed_page", 0)) + 1

    pages = 0
    count = 0
    for page in range(start_page, 1_000_000):
        response = em_get(
            {
                "reportName": "RPT_HOLDERNUM_DET",
                "columns": ",".join(FIELDS),
                "filter": (
                    f"(HOLD_NOTICE_DATE>='{NOTICE_START}')"
                    f"(HOLD_NOTICE_DATE<='{NOTICE_END}')"
                ),
                "pageNumber": str(page),
                "pageSize": str(PAGE_SIZE),
                "sortColumns": "HOLD_NOTICE_DATE,SECURITY_CODE,END_DATE",
                "sortTypes": "1,1,1",
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
        pages = int(result.get("pages") or 0)
        count = int(result.get("count") or 0)
        page_rows = result.get("data") or []
        rows.extend(
            {field: row.get(field) for field in FIELDS}
            for row in page_rows
        )
        complete = page >= pages
        if page % CHECKPOINT_EVERY == 0 or complete:
            checkpoint(
                rows,
                completed_page=page,
                pages=pages,
                count=count,
                complete=complete,
            )
            print(
                f"page={page}/{pages} raw={len(rows)}/{count}",
                flush=True,
            )
        if complete:
            return


if __name__ == "__main__":
    main()
