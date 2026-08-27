"""Download completed calendar-month margin snapshots with checkpoints.

Only 2010-2022 train/validation source data is fetched.  Each exchange record
dated T is considered available from the next trading day, never on T itself.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "strategy_opt_20260730"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from update_eastmoney_research_common import em_get  # noqa: E402
from utils.stock.time import get_trading_date_span  # noqa: E402

OUTPUT_PATH = RESULT_DIR / "eastmoney_margin_monthly_2010_2022.parquet"
STATUS_PATH = RESULT_DIR / "eastmoney_margin_monthly_2010_2022_status.json"
START = date(2010, 3, 31)
END = date(2022, 12, 30)
PAGE_SIZE = 500
CHECKPOINT_EVERY_MONTHS = 6
FIELDS = (
    "DATE",
    "SCODE",
    "SECUCODE",
    "RZYE",
    "RQYE",
    "RQYL",
    "RZRQYE",
    "RZMRE",
    "RZCHE",
    "RZJME",
    "SZ",
)


def completed_month_ends() -> list[date]:
    days = get_trading_date_span(START, END)
    result: list[date] = []
    for value in days:
        if not result or (result[-1].year, result[-1].month) != (
            value.year,
            value.month,
        ):
            result.append(value)
        else:
            result[-1] = value
    return result


def checkpoint(
    rows: list[dict],
    *,
    month_ends: list[date],
    completed_month_index: int,
    source_counts: dict[str, int],
    complete: bool,
) -> None:
    frame = pd.DataFrame(rows, columns=FIELDS)
    if not frame.empty:
        frame = (
            frame.drop_duplicates(["DATE", "SCODE"], keep="last")
            .sort_values(["DATE", "SCODE"], kind="stable")
        )
    temporary = OUTPUT_PATH.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, OUTPUT_PATH)
    status = {
        "source": "Eastmoney RPTA_WEB_RZRQ_GGMX",
        "temporal_contract": (
            "monthly snapshot dated T activates only on the first trading "
            "day strictly after T"
        ),
        "start": START.isoformat(),
        "end": END.isoformat(),
        "month_count": len(month_ends),
        "page_size": PAGE_SIZE,
        "completed_month_index": completed_month_index,
        "completed_month_end": (
            month_ends[completed_month_index].isoformat()
            if completed_month_index >= 0
            else None
        ),
        "source_counts": source_counts,
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
    month_ends = completed_month_ends()
    rows: list[dict] = []
    source_counts: dict[str, int] = {}
    start_index = 0
    if OUTPUT_PATH.exists() and STATUS_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        rows = existing.to_dict("records")
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        if status.get("complete"):
            print(f"already complete: {len(rows)} rows", flush=True)
            return
        source_counts = {
            str(key): int(value)
            for key, value in status.get("source_counts", {}).items()
        }
        start_index = int(status.get("completed_month_index", -1)) + 1

    for month_index in range(start_index, len(month_ends)):
        month_end = month_ends[month_index]
        day = month_end.isoformat()
        month_rows: list[dict] = []
        pages = 1
        count = 0
        page = 1
        while page <= pages:
            response = em_get(
                {
                    "reportName": "RPTA_WEB_RZRQ_GGMX",
                    "columns": ",".join(FIELDS),
                    "filter": f"(DATE='{day}')",
                    "pageNumber": str(page),
                    "pageSize": str(PAGE_SIZE),
                    "sortColumns": "SCODE",
                    "sortTypes": "1",
                    "source": "WEB",
                    "client": "WEB",
                }
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success"):
                if payload.get("message") == "返回数据为空":
                    pages = 0
                    count = 0
                    break
                raise RuntimeError(
                    f"{day} page {page}: {payload.get('message') or payload}"
                )
            result = payload.get("result") or {}
            pages = int(result.get("pages") or 0)
            count = int(result.get("count") or 0)
            month_rows.extend(
                {field: row.get(field) for field in FIELDS}
                for row in (result.get("data") or [])
            )
            page += 1
        if len(month_rows) != count:
            raise RuntimeError(
                f"{day}: received {len(month_rows)} of {count} source rows"
            )
        rows.extend(month_rows)
        source_counts[day] = count
        complete = month_index == len(month_ends) - 1
        if (
            (month_index + 1) % CHECKPOINT_EVERY_MONTHS == 0
            or complete
        ):
            checkpoint(
                rows,
                month_ends=month_ends,
                completed_month_index=month_index,
                source_counts=source_counts,
                complete=complete,
            )
        print(
            f"month={month_index + 1}/{len(month_ends)} "
            f"date={day} rows={count} pages={pages}",
            flush=True,
        )


if __name__ == "__main__":
    main()
