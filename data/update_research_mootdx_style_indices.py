"""Download immutable mootdx daily histories for style-regime research.

The research runtime must remain offline.  This pre-download entry point stores
the official style proxies used by the research-only style-switch screen: CSI
500 (000905), CSI 1000 (000852), and SSE Dividend (000015).  CSI Dividend
(000922) is not served by this mootdx endpoint, so it is deliberately not
silently substituted.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

from data.kline_mootdx import PAGE_SIZE, _connect_mootdx


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "strategy_opt_20260730"
OUTPUT_PATH = RESULT_DIR / "mootdx_style_indices_daily.parquet"
STATUS_PATH = RESULT_DIR / "mootdx_style_indices_daily_status.json"
MAX_HISTORY_BARS = 10_000
INDEX_CODES = {
    "csi_500": "000905",
    "csi_1000": "000852",
    "sse_dividend": "000015",
}
KEEP_COLUMNS = ("datetime", "open", "high", "low", "close", "volume", "amount")


def _normalize_bars(frame: pd.DataFrame, *, label: str, code: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise RuntimeError(f"{label} ({code}) returned no bars")
    if "datetime" not in frame.columns:
        frame = frame.reset_index()
    if "datetime" not in frame.columns:
        raise RuntimeError(f"{label} ({code}) has no datetime column")
    missing = set(KEEP_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"{label} ({code}) missing columns: {sorted(missing)}")
    normalized = frame.loc[:, KEEP_COLUMNS].copy()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"], errors="coerce")
    normalized = normalized.dropna(subset=["datetime", "open", "close"])
    # The downloader can run before the market close.  A dated row stamped
    # 15:00 is not evidence that its OHLC values are final, so retain only
    # strictly completed calendar days.  A later post-close refresh can add it.
    normalized = normalized[normalized["datetime"].dt.date < date.today()]
    normalized = normalized.drop_duplicates(subset=["datetime"], keep="first")
    normalized = normalized.sort_values("datetime").reset_index(drop=True)
    if normalized.empty:
        raise RuntimeError(f"{label} ({code}) has no usable bars")
    normalized.insert(0, "index", label)
    normalized.insert(1, "code", code)
    return normalized


def _fetch_all_index_bars(mdx, *, label: str, code: str) -> pd.DataFrame:
    pages: list[pd.DataFrame] = []
    for start in range(0, MAX_HISTORY_BARS, PAGE_SIZE):
        page = mdx.index_bars(
            symbol=code,
            frequency=9,
            start=start,
            offset=PAGE_SIZE,
        )
        if page is None or page.empty:
            break
        pages.append(page)
        if len(page) < PAGE_SIZE:
            break
    if not pages:
        raise RuntimeError(f"{label} ({code}) returned no pages")
    return _normalize_bars(pd.concat(pages, ignore_index=True), label=label, code=code)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    mdx = _connect_mootdx()
    frames = [
        _fetch_all_index_bars(mdx, label=label, code=code)
        for label, code in INDEX_CODES.items()
    ]
    combined = pd.concat(frames, ignore_index=True)
    temporary = OUTPUT_PATH.with_suffix(".tmp.parquet")
    combined.to_parquet(temporary, index=False)
    os.replace(temporary, OUTPUT_PATH)
    coverage = {
        label: {
            "code": code,
            "rows": int((combined["index"] == label).sum()),
            "first_date": str(combined.loc[combined["index"] == label, "datetime"].min().date()),
            "last_date": str(combined.loc[combined["index"] == label, "datetime"].max().date()),
        }
        for label, code in INDEX_CODES.items()
    }
    status = {
        "source": "mootdx StdQuotes.index_bars daily frequency=9",
        "indices": coverage,
        "temporal_contract": "signal row T may use index close returns only through T-1",
        "research_availability_gate": {
            "csi_500": "published before the 2010 training start; eligible across the research sample",
            "csi_1000": "backfilled history is stored for audit only; do not use before its 2014-10-17 publication date without a contemporaneous-publication proof",
            "sse_dividend": "price-index proxy; assess total-return mismatch before using relative style performance",
        },
        "data_sha256": _sha256(OUTPUT_PATH),
    }
    temporary_status = STATUS_PATH.with_suffix(".tmp.json")
    temporary_status.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_status, STATUS_PATH)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
