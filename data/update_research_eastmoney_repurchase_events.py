"""Download realized share-repurchase updates through validation end."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "strategy_opt_20260730"

from update_eastmoney_research_common import em_get  # noqa: E402

OUTPUT_PATH = RESULT_DIR / "eastmoney_repurchase_events_2010_2022.parquet"
STATUS_PATH = RESULT_DIR / "eastmoney_repurchase_events_2010_2022_status.json"
START = "2010-01-01"
END = "2022-12-31"
PAGE_SIZE = 500
FIELDS = (
    "DIM_SCODE",
    "SECUCODE",
    "REPURCODE",
    "REPURPROGRESS",
    "DIM_DATE",
    "UPDATEDATE",
    "NOTICEDATE",
    "FINISHDATE",
    "REPURAMOUNT",
    "REPURNUM",
    "ZJJE",
    "ZJSL",
    "ZJSZBL",
)


def main() -> None:
    rows = []
    source_pages = 0
    source_count = 0
    page = 1
    while not source_pages or page <= source_pages:
        response = em_get(
            {
                "reportName": "RPTA_WEB_GETHGLIST_NEW",
                "columns": ",".join(FIELDS),
                "filter": (
                    f"(UPDATEDATE>='{START}')(UPDATEDATE<='{END}')"
                ),
                "pageNumber": str(page),
                "pageSize": str(PAGE_SIZE),
                "sortColumns": "UPDATEDATE,DIM_SCODE,REPURCODE",
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
        source_pages = int(result.get("pages") or 0)
        source_count = int(result.get("count") or 0)
        page_rows = result.get("data") or []
        rows.extend(
            {field: row.get(field) for field in FIELDS}
            for row in page_rows
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
    frame = pd.DataFrame(rows, columns=FIELDS)
    metadata_dates = frame[
        ["UPDATEDATE", "NOTICEDATE", "FINISHDATE"]
    ].apply(lambda values: pd.to_datetime(values, errors="coerce"))
    post_validation = metadata_dates.max(axis=1) > pd.Timestamp(END)
    dropped_post_validation = int(post_validation.sum())
    frame = frame.loc[~post_validation].reset_index(drop=True)
    temporary = OUTPUT_PATH.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, OUTPUT_PATH)
    status = {
        "source": "Eastmoney RPTA_WEB_GETHGLIST_NEW",
        "temporal_contract": (
            "latest realized repurchase update dated T activates only on "
            "the first trading day strictly after T"
        ),
        "start": START,
        "end": END,
        "page_size": PAGE_SIZE,
        "source_pages": source_pages,
        "source_count": source_count,
        "source_rows_received": len(rows),
        "post_validation_metadata_rows_dropped": dropped_post_validation,
        "stored_rows": len(frame),
        "complete": True,
        "test_source_downloaded": False,
        "test_period_data_used": False,
    }
    temporary_status = STATUS_PATH.with_suffix(".tmp.json")
    temporary_status.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_status, STATUS_PATH)


if __name__ == "__main__":
    main()
