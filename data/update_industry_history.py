"""Download and align the official Shenwan industry membership history.

Network access is restricted to the explicit ``update`` CLI command.  The
normalization, loading, and point-in-time panel helpers are entirely offline.
An industry event dated D becomes usable on the first supplied trading day
strictly after D; ``update_date`` is retained for audit only and is never an
availability date.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "industry_history"
RAW_FILENAME = "StockClassifyUse_stock.xls"
PARQUET_FILENAME = "sw_industry_history.parquet"
METADATA_FILENAME = "sw_industry_history_metadata.json"
SW_URL = (
    "https://www.swsresearch.com/swindex/pdf/SwClass2021/"
    "StockClassifyUse_stock.xls"
)
SOURCE_COLUMNS = ("股票代码", "计入日期", "行业代码", "更新日期")
NORMALIZED_COLUMNS = (
    "stock_code",
    "start_date",
    "industry_code",
    "l1_code",
    "l2_code",
    "l3_code",
    "update_date",
)
OLE2_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
OOXML_MAGIC = b"PK\x03\x04"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_text(value: object, *, label: str) -> str:
    if pd.isna(value):
        raise ValueError(f"{label} contains a null value")
    text = str(value).strip().upper()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", maxsplit=1)[0]
    return text


def _canonical_stock_code(value: object, *, allow_unsupported: bool = False) -> str | None:
    text = _numeric_text(value, label="stock code")
    supplied_exchange = None
    suffix_match = re.fullmatch(r"(.+)\.(SH|SZ)", text)
    if suffix_match:
        text, supplied_exchange = suffix_match.groups()
    if not re.fullmatch(r"\d{1,6}", text):
        raise ValueError(f"invalid stock code: {value!r}")
    digits = text.zfill(6)
    if digits[0] in "569":
        exchange = "SH"
    elif digits[0] in "0123":
        exchange = "SZ"
    else:
        if allow_unsupported and supplied_exchange is None:
            return None
        raise ValueError(f"stock code is not an SH/SZ security: {value!r}")
    if supplied_exchange is not None and supplied_exchange != exchange:
        raise ValueError(f"stock code has a mismatched exchange: {value!r}")
    return f"{digits}.{exchange}"


def _canonical_industry_code(value: object) -> str:
    text = _numeric_text(value, label="industry code")
    suffix_match = re.fullmatch(r"(\d+)\.SI", text)
    if suffix_match:
        text = suffix_match.group(1)
    if not re.fullmatch(r"\d{1,6}", text):
        raise ValueError(f"invalid Shenwan industry code: {value!r}")
    return text.zfill(6)


def _validate_normalized(frame: pd.DataFrame) -> None:
    missing = set(NORMALIZED_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"normalized history is missing columns: {sorted(missing)}")
    if frame.empty:
        raise RuntimeError("normalized industry history is empty")
    if frame["stock_code"].isna().any() or frame["start_date"].isna().any():
        raise RuntimeError("normalized history contains null PIT keys")
    if frame["industry_code"].isna().any():
        raise RuntimeError("normalized history contains null industry codes")
    if not frame["stock_code"].astype(str).str.fullmatch(r"\d{6}\.(SH|SZ)").all():
        raise RuntimeError("normalized history contains non-SH/SZ stock codes")
    for column in ("industry_code", "l1_code", "l2_code", "l3_code"):
        if not frame[column].astype(str).str.fullmatch(r"\d{6}").all():
            raise RuntimeError(f"normalized history contains invalid {column}")
    industry = frame["industry_code"].astype(str)
    if not (frame["l1_code"].astype(str) == industry.str[:2] + "0000").all():
        raise RuntimeError("l1_code is inconsistent with industry_code")
    if not (frame["l2_code"].astype(str) == industry.str[:4] + "00").all():
        raise RuntimeError("l2_code is inconsistent with industry_code")
    if not (frame["l3_code"].astype(str) == industry).all():
        raise RuntimeError("l3_code is inconsistent with industry_code")
    if frame.duplicated(["stock_code", "start_date"]).any():
        raise RuntimeError("multiple industry events share a stock_code/start_date key")


def normalize_industry_history(source: pd.DataFrame) -> pd.DataFrame:
    """Normalize the official worksheet without assigning an availability date."""

    missing = set(SOURCE_COLUMNS).difference(source.columns)
    if missing:
        raise RuntimeError(
            f"Shenwan source schema changed; missing={sorted(missing)}, "
            f"actual={list(source.columns)}"
        )
    frame = source.loc[:, SOURCE_COLUMNS].rename(
        columns={
            "股票代码": "stock_code",
            "计入日期": "start_date",
            "行业代码": "industry_code",
            "更新日期": "update_date",
        }
    )
    try:
        frame["stock_code"] = frame["stock_code"].map(
            lambda value: _canonical_stock_code(value, allow_unsupported=True)
        )
        frame["industry_code"] = frame["industry_code"].map(
            _canonical_industry_code
        )
    except ValueError as exc:
        raise RuntimeError(f"invalid Shenwan source value: {exc}") from exc
    excluded_non_sh_sz_rows = int(frame["stock_code"].isna().sum())
    frame = frame[frame["stock_code"].notna()].copy()
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
    frame["update_date"] = pd.to_datetime(frame["update_date"], errors="coerce")
    if frame["start_date"].isna().any():
        bad_rows = frame.index[frame["start_date"].isna()].tolist()[:10]
        raise RuntimeError(f"unparseable start_date rows: {bad_rows}")
    frame["industry_code"] = frame["industry_code"].astype("string")
    frame["stock_code"] = frame["stock_code"].astype("string")
    frame["l1_code"] = frame["industry_code"].str[:2] + "0000"
    frame["l2_code"] = frame["industry_code"].str[:4] + "00"
    frame["l3_code"] = frame["industry_code"]
    frame = frame.loc[:, NORMALIZED_COLUMNS]
    frame = frame.sort_values(["stock_code", "start_date"], kind="stable")
    frame = frame.reset_index(drop=True)
    _validate_normalized(frame)
    frame.attrs["excluded_non_sh_sz_rows"] = excluded_non_sh_sz_rows
    return frame


def _validate_source_scale(frame: pd.DataFrame) -> None:
    """Reject error pages and truncated workbooks before replacing local data."""

    checks = {
        "rows": (len(frame), 10_000),
        "stocks": (frame["stock_code"].nunique(), 5_000),
        "l1 industries": (frame["l1_code"].nunique(), 30),
        "l2 industries": (frame["l2_code"].nunique(), 100),
        "l3 industries": (frame["l3_code"].nunique(), 400),
    }
    failures = [
        f"{label}={actual} < {minimum}"
        for label, (actual, minimum) in checks.items()
        if actual < minimum
    ]
    if failures:
        raise RuntimeError("Shenwan workbook failed scale sanity: " + ", ".join(failures))


def _download_workbook(*, allow_insecure_fallback: bool) -> tuple[bytes, dict]:
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (WBR industry history updater)"}
    ssl_error = None
    try:
        response = requests.get(SW_URL, headers=headers, timeout=60, verify=True)
        security_status = "verified_tls"
    except requests.exceptions.SSLError as exc:
        ssl_error = str(exc)
        warnings.warn(
            "Shenwan TLS verification failed. Update certifi or inspect the local "
            "proxy/CA chain before permitting an insecure retry.",
            RuntimeWarning,
            stacklevel=2,
        )
        if not allow_insecure_fallback:
            raise RuntimeError(
                "TLS verification failed; no data was written. Re-run with "
                "--allow-insecure-fallback only after validating the network path."
            ) from exc
        response = requests.get(SW_URL, headers=headers, timeout=60, verify=False)
        security_status = "insecure_fallback"
    response.raise_for_status()
    return response.content, {
        "status": security_status,
        "tls_verified": security_status == "verified_tls",
        "insecure_fallback_allowed": allow_insecure_fallback,
        "initial_ssl_error": ssl_error,
        "resolved_url": response.url,
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "content_type": response.headers.get("Content-Type"),
    }


def _validate_payload(payload: bytes, expected_sha256: str | None) -> tuple[str, str]:
    if len(payload) < 100_000:
        raise RuntimeError(f"downloaded workbook is implausibly small: {len(payload)} bytes")
    if payload.startswith(OLE2_MAGIC):
        workbook_format = "xls/ole2"
    elif payload.startswith(OOXML_MAGIC):
        workbook_format = "xlsx/ooxml"
    else:
        raise RuntimeError("downloaded content is not an Excel workbook")
    digest = _sha256_bytes(payload)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise RuntimeError(
            f"workbook SHA-256 mismatch: expected={expected_sha256.lower()} actual={digest}"
        )
    return digest, workbook_format


def _schema_metadata(frame: pd.DataFrame) -> dict:
    return {
        "source_columns": list(SOURCE_COLUMNS),
        "normalized_columns": list(NORMALIZED_COLUMNS),
        "normalized_dtypes": {column: str(frame[column].dtype) for column in frame},
    }


def update_industry_history(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    allow_insecure_fallback: bool = False,
    expected_sha256: str | None = None,
) -> dict:
    """Explicit network update; validate all artifacts before publishing metadata."""

    if expected_sha256 is not None and not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal digits")
    payload, response_metadata = _download_workbook(
        allow_insecure_fallback=allow_insecure_fallback
    )
    raw_sha256, workbook_format = _validate_payload(payload, expected_sha256)
    source = pd.read_excel(io.BytesIO(payload))
    normalized = normalize_industry_history(source)
    excluded_non_sh_sz_rows = int(
        normalized.attrs.get("excluded_non_sh_sz_rows", 0)
    )
    _validate_source_scale(normalized)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_FILENAME
    parquet_path = output_dir / PARQUET_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    raw_temporary: Path | None = None
    parquet_temporary: Path | None = None
    metadata_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir, suffix=".xls", delete=False
        ) as stream:
            stream.write(payload)
            raw_temporary = Path(stream.name)
        if _sha256_file(raw_temporary) != raw_sha256:
            raise RuntimeError("raw workbook hash changed while writing the temporary file")

        with tempfile.NamedTemporaryFile(
            dir=output_dir, suffix=".parquet", delete=False
        ) as stream:
            parquet_temporary = Path(stream.name)
        normalized.to_parquet(parquet_temporary, index=False)
        parquet_check = pd.read_parquet(parquet_temporary)
        _validate_normalized(parquet_check)
        if len(parquet_check) != len(normalized):
            raise RuntimeError("normalized parquet row count changed during round-trip")
        parquet_sha256 = _sha256_file(parquet_temporary)

        metadata = {
            "url": SW_URL,
            "resolved_url": response_metadata["resolved_url"],
            "hash": {
                "algorithm": "sha256",
                "raw": raw_sha256,
                "normalized_parquet": parquet_sha256,
            },
            "etag": response_metadata["etag"],
            "last_modified": response_metadata["last_modified"],
            "download_at": datetime.now(timezone.utc).isoformat(),
            "schema": _schema_metadata(normalized),
            "security_status": {
                "status": response_metadata["status"],
                "tls_verified": response_metadata["tls_verified"],
                "insecure_fallback_allowed": response_metadata[
                    "insecure_fallback_allowed"
                ],
                "initial_ssl_error": response_metadata["initial_ssl_error"],
            },
            "http_content_type": response_metadata["content_type"],
            "workbook_format": workbook_format,
            "raw_file": RAW_FILENAME,
            "normalized_file": PARQUET_FILENAME,
            "rows": len(normalized),
            "stocks": int(normalized["stock_code"].nunique()),
            "excluded_non_sh_sz_rows": excluded_non_sh_sz_rows,
            "coverage": {
                "first_start_date": normalized["start_date"].min().date().isoformat(),
                "last_start_date": normalized["start_date"].max().date().isoformat(),
                "l1_industries": int(normalized["l1_code"].nunique()),
                "l2_industries": int(normalized["l2_code"].nunique()),
                "l3_industries": int(normalized["l3_code"].nunique()),
            },
            "availability_contract": (
                "start_date activates on the first trading day strictly after it; "
                "update_date is audit metadata and is never used as an announcement "
                "or availability date"
            ),
        }
        with tempfile.NamedTemporaryFile(
            dir=output_dir, suffix=".json", mode="w", encoding="utf-8", delete=False
        ) as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            metadata_temporary = Path(stream.name)

        os.replace(raw_temporary, raw_path)
        raw_temporary = None
        os.replace(parquet_temporary, parquet_path)
        parquet_temporary = None
        os.replace(metadata_temporary, metadata_path)
        metadata_temporary = None
        return metadata
    finally:
        for temporary in (raw_temporary, parquet_temporary, metadata_temporary):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def load_industry_history(
    parquet_path: Path = DEFAULT_OUTPUT_DIR / PARQUET_FILENAME,
    metadata_path: Path = DEFAULT_OUTPUT_DIR / METADATA_FILENAME,
) -> pd.DataFrame:
    """Load a normalized artifact offline and verify it against its metadata hash."""

    parquet_path = Path(parquet_path)
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = metadata["hash"]["normalized_parquet"]
    actual = _sha256_file(parquet_path)
    if actual != expected:
        raise RuntimeError(
            f"industry parquet SHA-256 mismatch: expected={expected} actual={actual}"
        )
    frame = pd.read_parquet(parquet_path)
    _validate_normalized(frame)
    if len(frame) != int(metadata["rows"]):
        raise RuntimeError("industry parquet row count does not match metadata")
    return frame


def _parse_level(level: str | int) -> int:
    normalized = str(level).strip().lower()
    aliases = {"1": 1, "l1": 1, "2": 2, "l2": 2, "3": 3, "l3": 3}
    if normalized not in aliases:
        raise ValueError("level must be one of 1/L1, 2/L2, or 3/L3")
    return aliases[normalized]


def build_industry_panel(
    trade_dates: Sequence[object] | np.ndarray,
    stock_codes: Sequence[object] | np.ndarray,
    memberships: pd.DataFrame,
    *,
    level: str | int = 3,
) -> np.ndarray:
    """Build an offline PIT industry-code panel with conservative one-day lag.

    The result has shape ``(len(trade_dates), len(stock_codes))`` and dtype
    ``int32``.  Unknown membership is ``-1``.  L1/L2 values retain Shenwan's
    official six-digit padded form (for example 480000 and 480300).
    """

    level_number = _parse_level(level)
    dates = np.asarray(trade_dates, dtype="datetime64[D]")
    if dates.ndim != 1:
        raise ValueError("trade_dates must be one-dimensional")
    if np.isnat(dates).any():
        raise ValueError("trade_dates contains NaT")
    if len(dates) > 1 and np.any(dates[1:] <= dates[:-1]):
        raise ValueError("trade_dates must be strictly increasing and unique")

    try:
        canonical_stocks = [_canonical_stock_code(code) for code in stock_codes]
    except ValueError as exc:
        raise ValueError(f"invalid stock universe: {exc}") from exc
    if len(set(canonical_stocks)) != len(canonical_stocks):
        raise ValueError("stock_codes must be unique")
    panel = np.full((len(dates), len(canonical_stocks)), -1, dtype=np.int32)
    if panel.size == 0 or memberships.empty:
        return panel

    required = {"stock_code", "start_date", "industry_code"}
    missing = required.difference(memberships.columns)
    if missing:
        raise ValueError(f"memberships is missing columns: {sorted(missing)}")
    events = memberships.loc[:, ["stock_code", "start_date", "industry_code"]].copy()
    try:
        events["stock_code"] = events["stock_code"].map(_canonical_stock_code)
        events["industry_code"] = events["industry_code"].map(
            _canonical_industry_code
        )
    except ValueError as exc:
        raise ValueError(f"invalid membership event: {exc}") from exc
    events["start_date"] = pd.to_datetime(events["start_date"], errors="coerce")
    if events["start_date"].isna().any():
        raise ValueError("memberships contains an unparseable start_date")
    if events.duplicated(["stock_code", "start_date"]).any():
        raise ValueError("memberships contains duplicate stock/date events")
    events = events.sort_values(["stock_code", "start_date"], kind="stable")

    stock_index = {code: index for index, code in enumerate(canonical_stocks)}
    prefix_length = {1: 2, 2: 4, 3: 6}[level_number]
    for event in events.itertuples(index=False):
        column = stock_index.get(event.stock_code)
        if column is None:
            continue
        start_date = np.datetime64(event.start_date.date(), "D")
        first_usable = int(np.searchsorted(dates, start_date, side="right"))
        if first_usable == len(dates):
            continue
        industry = event.industry_code[:prefix_length].ljust(6, "0")
        panel[first_usable:, column] = np.int32(int(industry))
    return panel


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    update = subparsers.add_parser("update", help="download and publish official data")
    update.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    update.add_argument(
        "--allow-insecure-fallback",
        action="store_true",
        help="retry with TLS verification disabled only after a verified TLS failure",
    )
    update.add_argument(
        "--expected-sha256",
        help="optional pinned SHA-256 for the raw official workbook",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "update":
        metadata = update_industry_history(
            output_dir=args.output_dir,
            allow_insecure_fallback=args.allow_insecure_fallback,
            expected_sha256=args.expected_sha256,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
