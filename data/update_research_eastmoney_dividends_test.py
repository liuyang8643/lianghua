"""Download the post-validation dividend-event segment for style research.

The generic downloader owns pagination, throttling, checkpointing and atomic
writes.  Keeping this as a thin date-range wrapper avoids duplicating that
network logic while preserving the immutable 2010--2022 snapshot.
"""

from __future__ import annotations

import json

from update_research_eastmoney_dividends import (  # noqa: E402
    RESULT_DIR,
)
import update_research_eastmoney_dividends as downloader  # noqa: E402


START = "2023-01-01"
END = "2026-07-31"
OUTPUT_PATH = RESULT_DIR / "eastmoney_dividends_2023_2026.parquet"
STATUS_PATH = RESULT_DIR / "eastmoney_dividends_2023_2026_status.json"


def main() -> None:
    downloader.START = START
    downloader.END = END
    downloader.OUTPUT_PATH = OUTPUT_PATH
    downloader.STATUS_PATH = STATUS_PATH
    downloader.main()
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    status["test_source_downloaded"] = True
    status["segment_role"] = "post_validation_through_current_data_cutoff"
    STATUS_PATH.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
