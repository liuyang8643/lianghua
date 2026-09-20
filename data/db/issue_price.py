"""Validated offline issue-price and listing-date snapshot contract."""

from __future__ import annotations

from utils.atomic_file import file_sha256
from datetime import date
import json
import os
from pathlib import Path
import re
from typing import Collection, Iterable
from uuid import uuid4

from filelock import FileLock
import numpy as np
import pandas as pd


_DATA_DIR = Path(__file__).resolve().parents[1]
ISSUE_REFERENCE_PATH = _DATA_DIR / "issue_price" / "issue_price.parquet"
ISSUE_REFERENCE_MANIFEST_PATH = ISSUE_REFERENCE_PATH.with_name(
    f"{ISSUE_REFERENCE_PATH.name}.manifest.json"
)
ISSUE_REFERENCE_SCHEMA = "issue-price-v2-source-lineage"
_REQUIRED_COLUMNS = (
    "stock_code",
    "issue_price",
    "list_date",
    "source",
    "source_as_of",
)
_STOCK_CODE_PATTERN = re.compile(r"^\d{6}\.(?:SH|SZ|BJ)$")




def validate_issue_reference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise the sole issue-price/listing-date snapshot."""

    actual_columns = set(frame.columns)
    required_columns = set(_REQUIRED_COLUMNS)
    if actual_columns != required_columns:
        missing = sorted(required_columns.difference(actual_columns))
        extra = sorted(actual_columns.difference(required_columns))
        raise ValueError(
            f"issue_price 列集合不匹配: missing={missing}, extra={extra}"
        )
    if frame.empty:
        raise ValueError("issue_price 不得为空")

    result = frame.loc[:, _REQUIRED_COLUMNS].copy()
    result["stock_code"] = (
        result["stock_code"].astype(str).str.strip().str.zfill(6)
    )
    invalid_codes = ~result["stock_code"].str.fullmatch(r"\d{6}")
    if invalid_codes.any():
        raise ValueError(
            "issue_price 包含非法股票代码: "
            + ", ".join(result.loc[invalid_codes, "stock_code"].head(20))
        )
    if result["stock_code"].duplicated().any():
        duplicates = result.loc[
            result["stock_code"].duplicated(), "stock_code"
        ]
        raise ValueError(
            "issue_price 股票代码重复: " + ", ".join(duplicates.head(20))
        )

    result["issue_price"] = pd.to_numeric(
        result["issue_price"], errors="coerce"
    )
    invalid_prices = (
        ~np.isfinite(result["issue_price"].to_numpy(dtype=np.float64))
        | (result["issue_price"] <= 0.0)
    )
    if invalid_prices.any():
        raise ValueError(
            "issue_price 包含无效发行价: "
            + ", ".join(result.loc[invalid_prices, "stock_code"].head(20))
        )

    for column in ("list_date", "source_as_of"):
        parsed = pd.to_datetime(result[column], errors="coerce")
        if parsed.isna().any():
            bad = result.loc[parsed.isna(), "stock_code"]
            raise ValueError(
                f"issue_price {column} 包含无效日期: "
                + ", ".join(bad.head(20))
            )
        result[column] = parsed.dt.date

    missing_source = result["source"].isna()
    result["source"] = result["source"].astype(str).str.strip()
    invalid_source = missing_source | result["source"].str.casefold().isin(
        {"", "nan", "none", "null", "nat"}
    )
    if invalid_source.any():
        bad = result.loc[invalid_source, "stock_code"]
        raise ValueError(
            "issue_price source 为空: " + ", ".join(bad.head(20))
        )

    return result.sort_values("stock_code", kind="stable").reset_index(drop=True)


def save_issue_reference_atomic(
    frame: pd.DataFrame,
    *,
    path: Path = ISSUE_REFERENCE_PATH,
    manifest_path: Path | None = None,
) -> None:
    """Validate, hash-seal and replace the offline snapshot without partial rows."""

    canonical = validate_issue_reference_frame(frame)
    path = Path(path)
    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else path.with_name(f"{path.name}.manifest.json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid4().hex}"
    parquet_temp = path.with_name(f".{path.name}.{token}.tmp.parquet")
    manifest_temp = manifest_path.with_name(
        f".{manifest_path.name}.{token}.tmp"
    )
    lock_path = path.with_name(f".{path.name}.lock")
    with FileLock(str(lock_path), timeout=60):
        try:
            canonical.to_parquet(parquet_temp, index=False)
            roundtrip = validate_issue_reference_frame(pd.read_parquet(parquet_temp))
            if not canonical.equals(roundtrip):
                raise ValueError("issue_price parquet round-trip 后内容不一致")
            payload = {
                "schema_version": ISSUE_REFERENCE_SCHEMA,
                "row_count": len(canonical),
                "snapshot_sha256": file_sha256(parquet_temp),
                "sources": sorted(canonical["source"].unique().tolist()),
                "source_as_of_max": max(canonical["source_as_of"]).isoformat(),
            }
            manifest_temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            from data.kline_mootdx import replace_staged_files_transactionally

            replace_staged_files_transactionally(
                [(parquet_temp, path), (manifest_temp, manifest_path)],
                token=token,
            )
        finally:
            for temporary in (parquet_temp, manifest_temp):
                temporary.unlink(missing_ok=True)


def load_issue_reference(
    *,
    path: Path = ISSUE_REFERENCE_PATH,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    """Load only a schema-valid snapshot whose bytes match its manifest."""

    path = Path(path)
    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else path.with_name(f"{path.name}.manifest.json")
    )
    if not path.exists():
        raise FileNotFoundError(f"issue_price 快照不存在: {path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"issue_price manifest 不存在: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"issue_price manifest 无法解析: {manifest_path}") from exc
    if manifest.get("schema_version") != ISSUE_REFERENCE_SCHEMA:
        raise ValueError(
            "issue_price manifest schema 不匹配: "
            f"{manifest.get('schema_version')!r}"
        )
    expected_hash = manifest.get("snapshot_sha256")
    if not isinstance(expected_hash, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_hash
    ) is None:
        raise ValueError("issue_price manifest snapshot_sha256 格式无效")
    actual_hash = file_sha256(path)
    if expected_hash != actual_hash:
        raise ValueError("issue_price snapshot SHA256 与 manifest 不一致")
    frame = validate_issue_reference_frame(pd.read_parquet(path))
    if manifest.get("row_count") != len(frame):
        raise ValueError("issue_price row_count 与 manifest 不一致")
    if manifest.get("sources") != sorted(frame["source"].unique().tolist()):
        raise ValueError("issue_price sources 与 manifest 不一致")
    source_as_of_max = max(frame["source_as_of"]).isoformat()
    if manifest.get("source_as_of_max") != source_as_of_max:
        raise ValueError("issue_price source_as_of_max 与 manifest 不一致")
    return frame


def issue_listing_dates(frame: pd.DataFrame) -> dict[str, date]:
    """Return the canonical bare-code -> listing-date mapping."""

    canonical = validate_issue_reference_frame(frame)
    return dict(zip(canonical["stock_code"], canonical["list_date"]))


def resolve_terminal_active_codes(
    current_stock_codes: Collection[str],
    terminal_date: date | np.datetime64 | str,
    kline_stock_codes: Iterable[str],
    *,
    issue_reference: pd.DataFrame | None = None,
) -> tuple[str, ...]:
    """Return current-list members whose listing day is not later than T.

    A current code is excluded only when the sealed issue reference positively
    proves ``list_date > T``.  An axis-missing code with unknown listing state
    remains a hard error; an empty market-data response is never lifecycle
    evidence.
    """

    current = tuple(str(code).strip().upper() for code in current_stock_codes)
    if not current or len(set(current)) != len(current):
        raise ValueError("current stock axis must be non-empty and unique")
    invalid_current = [code for code in current if not _STOCK_CODE_PATTERN.fullmatch(code)]
    if invalid_current:
        raise ValueError("current stock axis contains invalid codes: " + ", ".join(invalid_current[:20]))

    if terminal_date is None:
        raise ValueError("terminal_date 无效")
    terminal_timestamp = pd.Timestamp(terminal_date)
    if pd.isna(terminal_timestamp):
        raise ValueError("terminal_date 无效")
    terminal = terminal_timestamp.date()
    kline_codes = {
        str(code).strip().upper() for code in kline_stock_codes
    }
    invalid_kline = sorted(
        code for code in kline_codes if not _STOCK_CODE_PATTERN.fullmatch(code)
    )
    if invalid_kline:
        raise ValueError(
            "K 线股票轴包含非法代码: " + ", ".join(invalid_kline[:20])
        )
    reference = load_issue_reference() if issue_reference is None else issue_reference
    listing_dates = issue_listing_dates(reference)

    active: list[str] = []
    unresolved: list[str] = []
    conflicts: list[str] = []
    for code in current:
        listed = listing_dates.get(code[:6])
        if listed is not None and listed > terminal:
            if code in kline_codes:
                conflicts.append(
                    f"{code}(list_date={listed}, but K file exists)"
                )
            continue
        if code not in kline_codes and listed is None:
            unresolved.append(code)
            continue
        active.append(code)
    if conflicts:
        raise RuntimeError(
            "预上市生命周期与本地 K 轴冲突: " + ", ".join(conflicts[:20])
        )
    if unresolved:
        shown = ", ".join(unresolved[:20])
        suffix = f" ...(+{len(unresolved) - 20})" if len(unresolved) > 20 else ""
        raise RuntimeError(
            "当前列表中的轴外股票缺少封存上市日，不能把缺 K 解释为预上市: "
            f"{shown}{suffix}"
        )
    if not active:
        raise RuntimeError("runtime 末日已上市股票轴为空")
    return tuple(active)


__all__ = [
    "ISSUE_REFERENCE_MANIFEST_PATH",
    "ISSUE_REFERENCE_PATH",
    "ISSUE_REFERENCE_SCHEMA",
    "issue_listing_dates",
    "load_issue_reference",
    "resolve_terminal_active_codes",
    "save_issue_reference_atomic",
    "validate_issue_reference_frame",
]
