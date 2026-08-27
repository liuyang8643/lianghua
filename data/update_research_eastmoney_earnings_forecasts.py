"""Download PIT earnings-forecast notices through validation end."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "strategy_opt_20260730"

from update_eastmoney_research_common import em_get  # noqa: E402

OUTPUT_PATH = RESULT_DIR / "eastmoney_earnings_forecasts_2010_2022.parquet"
STATUS_PATH = (
    RESULT_DIR / "eastmoney_earnings_forecasts_2010_2022_status.json"
)
START = "2010-01-01"
END = "2022-12-31"
PAGE_SIZE = 5000
MAX_RETRIES = 5
FIELDS = (
    "SECUCODE",
    "SECURITY_CODE",
    "NOTICE_DATE",
    "REPORT_DATE",
    "PREDICT_FINANCE_CODE",
    "PREDICT_FINANCE",
    "PREDICT_AMT_LOWER",
    "PREDICT_AMT_UPPER",
    "ADD_AMP_LOWER",
    "ADD_AMP_UPPER",
    "PREDICT_TYPE",
    "PREYEAR_SAME_PERIOD",
    "INCREASE_JZ",
    "FORECAST_JZ",
    "FORECAST_STATE",
)


def main() -> None:
    rows = []
    source_pages = 0
    source_count = 0
    page = 1
    while not source_pages or page <= source_pages:
        params = {
            "reportName": "RPT_PUBLIC_OP_NEWPREDICT",
            "columns": ",".join(FIELDS),
            "filter": (
                f"(NOTICE_DATE>='{START}')(NOTICE_DATE<='{END}')"
            ),
            "pageNumber": str(page),
            "pageSize": str(PAGE_SIZE),
            "sortColumns": (
                "NOTICE_DATE,SECURITY_CODE,REPORT_DATE,"
                "PREDICT_FINANCE_CODE"
            ),
            "sortTypes": "1,1,1,1",
            "source": "WEB",
            "client": "WEB",
        }
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = em_get(params)
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    raise
                delay = float(2 ** (attempt - 1))
                print(
                    f"page={page} retry={attempt}/{MAX_RETRIES} "
                    f"delay={delay:g}s",
                    flush=True,
                )
                time.sleep(delay)
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
    notice_dates = pd.to_datetime(frame["NOTICE_DATE"], errors="coerce")
    inside = notice_dates.between(START, END, inclusive="both")
    invalid_notice_rows = int((~inside).sum())
    frame = frame.loc[inside].reset_index(drop=True)
    temporary = OUTPUT_PATH.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, OUTPUT_PATH)
    status = {
        "source": "Eastmoney RPT_PUBLIC_OP_NEWPREDICT",
        "temporal_contract": (
            "forecast notice dated T activates only on the first trading "
            "day strictly after T"
        ),
        "start": START,
        "end": END,
        "page_size": PAGE_SIZE,
        "source_pages": source_pages,
        "source_count": source_count,
        "source_rows_received": len(rows),
        "invalid_or_outside_notice_rows_dropped": invalid_notice_rows,
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
