"""Audit historical-version semantics of the CSDC pledge-ratio source."""

from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = (
    ROOT
    / "results"
    / "strategy_opt_20260730"
    / "agent_research"
)
SNAPSHOT = RESEARCH_DIR / "pledge_ratio_history_2017_2018.parquet"
FETCH_METADATA = (
    RESEARCH_DIR
    / "pledge_ratio_history_2017_2018_fetch_metadata.json"
)
OUTPUT = (
    RESEARCH_DIR
    / "pledge_ratio_history_2017_2018_source_audit.json"
)
URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
DETAIL_REPORT = "RPT_CSDC_LIST"
PROFILE_REPORT = "RPT_CSDC_STATISTICS"
PAGE_SIZE = 500
MIN_INTERVAL_SECONDS = 1.0
MAX_JITTER_SECONDS = 0.35
SAMPLE_DATE = "2017-01-26"
DETAIL_COLUMNS = [
    "SECURITY_CODE",
    "TRADE_DATE",
    "PLEDGE_RATIO",
    "REPURCHASE_BALANCE",
    "PLEDGE_DEAL_NUM",
    "REPURCHASE_UNLIMITED_BALANCE",
    "REPURCHASE_LIMITED_BALANCE",
]
_last_request_time = 0.0


def em_get(
    session: requests.Session,
    *,
    params: dict[str, str],
) -> requests.Response:
    global _last_request_time
    target_interval = (
        MIN_INTERVAL_SECONDS
        + random.uniform(0.0, MAX_JITTER_SECONDS)
    )
    elapsed = time.monotonic() - _last_request_time
    if elapsed < target_interval:
        time.sleep(target_interval - elapsed)
    response = session.get(URL, params=params, timeout=30)
    _last_request_time = time.monotonic()
    return response


def fetch_page(
    session: requests.Session,
    *,
    report_name: str,
    filter_value: str,
    page_number: int,
    sort_columns: str,
    sort_types: str,
) -> dict:
    response = em_get(
        session,
        params={
            "reportName": report_name,
            "columns": "ALL",
            "filter": filter_value,
            "pageNumber": str(page_number),
            "pageSize": str(PAGE_SIZE),
            "sortColumns": sort_columns,
            "sortTypes": sort_types,
            "source": "WEB",
            "client": "WEB",
        },
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success") or not payload.get("result"):
        raise RuntimeError(
            f"Eastmoney audit request failed: {payload.get('message')}"
        )
    return payload["result"]


def fetch_all(
    session: requests.Session,
    *,
    report_name: str,
    filter_value: str,
    sort_columns: str,
    sort_types: str,
) -> tuple[pd.DataFrame, int, int]:
    first = fetch_page(
        session,
        report_name=report_name,
        filter_value=filter_value,
        page_number=1,
        sort_columns=sort_columns,
        sort_types=sort_types,
    )
    pages = int(first.get("pages") or 1)
    count = int(first.get("count") or 0)
    rows = list(first.get("data") or [])
    for page in range(2, pages + 1):
        result = fetch_page(
            session,
            report_name=report_name,
            filter_value=filter_value,
            page_number=page,
            sort_columns=sort_columns,
            sort_types=sort_types,
        )
        rows.extend(result.get("data") or [])
    if len(rows) != count:
        raise RuntimeError(
            f"{report_name}: expected {count}, received {len(rows)}"
        )
    return pd.DataFrame(rows), pages, count


def normalize_detail(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [
        column for column in DETAIL_COLUMNS if column not in frame
    ]
    if missing:
        raise RuntimeError(f"detail audit lacks columns {missing}")
    result = frame[DETAIL_COLUMNS].copy()
    result["SECURITY_CODE"] = (
        result["SECURITY_CODE"].astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )
    result["TRADE_DATE"] = pd.to_datetime(
        result["TRADE_DATE"],
        errors="coerce",
    )
    for column in DETAIL_COLUMNS[2:]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )
    return result.sort_values(
        ["TRADE_DATE", "SECURITY_CODE"],
        kind="stable",
    ).reset_index(drop=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    metadata = json.loads(
        FETCH_METADATA.read_text(encoding="utf-8")
    )
    if sha256(SNAPSHOT) != metadata["output_sha256"]:
        raise RuntimeError("pledge snapshot hash changed before audit")
    local = normalize_detail(pd.read_parquet(SNAPSHOT))
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            ),
            "Referer": "https://data.eastmoney.com/",
        }
    )
    earliest_profile, earliest_pages, _ = fetch_all(
        session,
        report_name=PROFILE_REPORT,
        filter_value="(TRADE_DATE>='2000-01-01')",
        sort_columns="TRADE_DATE",
        sort_types="1",
    )
    earliest_dates = pd.to_datetime(
        earliest_profile["TRADE_DATE"],
        errors="coerce",
    ).dropna()
    period_profile, period_pages, _ = fetch_all(
        session,
        report_name=PROFILE_REPORT,
        filter_value=(
            "(TRADE_DATE>='2017-01-01')"
            "(TRADE_DATE<='2018-12-31')"
        ),
        sort_columns="TRADE_DATE",
        sort_types="1",
    )
    period_dates = pd.to_datetime(
        period_profile["TRADE_DATE"],
        errors="coerce",
    ).dropna()
    recomputed_month_ends = (
        period_dates.groupby(period_dates.dt.to_period("M"))
        .max()
        .sort_values()
    )
    selected_dates = pd.to_datetime(
        metadata["selected_trade_dates"],
        errors="raise",
    )
    sample_remote_raw, sample_pages, sample_count = fetch_all(
        session,
        report_name=DETAIL_REPORT,
        filter_value=f"(TRADE_DATE='{SAMPLE_DATE}')",
        sort_columns="TRADE_DATE,SECURITY_CODE",
        sort_types="1,1",
    )
    sample_remote = normalize_detail(sample_remote_raw)
    sample_local = local.loc[
        local["TRADE_DATE"] == pd.Timestamp(SAMPLE_DATE)
    ].reset_index(drop=True)
    comparison = sample_local.merge(
        sample_remote,
        on=["SECURITY_CODE", "TRADE_DATE"],
        how="outer",
        suffixes=("_local", "_remote"),
        indicator=True,
    )
    numeric_equal = np.ones(len(comparison), dtype=np.bool_)
    for column in DETAIL_COLUMNS[2:]:
        left = comparison[f"{column}_local"].to_numpy(
            dtype=np.float64
        )
        right = comparison[f"{column}_remote"].to_numpy(
            dtype=np.float64
        )
        numeric_equal &= (
            (np.isnan(left) & np.isnan(right))
            | (left == right)
        )
    local_dates = np.sort(
        np.unique(
            local["TRADE_DATE"].to_numpy(dtype="datetime64[D]")
        )
    )
    value_change_rows = []
    for previous, current in zip(
        local_dates[:-1],
        local_dates[1:],
        strict=True,
    ):
        left = local.loc[
            local["TRADE_DATE"].to_numpy(dtype="datetime64[D]")
            == previous,
            ["SECURITY_CODE", "PLEDGE_RATIO"],
        ]
        right = local.loc[
            local["TRADE_DATE"].to_numpy(dtype="datetime64[D]")
            == current,
            ["SECURITY_CODE", "PLEDGE_RATIO"],
        ]
        paired = left.merge(
            right,
            on="SECURITY_CODE",
            suffixes=("_previous", "_current"),
        )
        finite = (
            np.isfinite(paired["PLEDGE_RATIO_previous"])
            & np.isfinite(paired["PLEDGE_RATIO_current"])
        )
        changes = (
            paired.loc[finite, "PLEDGE_RATIO_previous"]
            != paired.loc[finite, "PLEDGE_RATIO_current"]
        )
        value_change_rows.append(
            {
                "previous_date": str(previous),
                "current_date": str(current),
                "overlapping_finite_stocks": int(
                    np.count_nonzero(finite)
                ),
                "changed_ratio_stocks": int(
                    np.count_nonzero(changes)
                ),
            }
        )
    schema_fields = sorted(sample_remote_raw.columns.astype(str))
    revision_fields = [
        field
        for field in schema_fields
        if any(
            token in field.upper()
            for token in ("UPDATE", "MODIFY", "EITIME", "VERSION")
        )
    ]
    counts = local.groupby("TRADE_DATE").size()
    audit = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_path": SNAPSHOT.name,
        "snapshot_sha256": sha256(SNAPSHOT),
        "reports": {
            "profile": PROFILE_REPORT,
            "stock_detail": DETAIL_REPORT,
        },
        "historical_cross_section_evidence": {
            "exact_date_filter_returns_only_requested_date": bool(
                sample_remote["TRADE_DATE"].nunique() == 1
                and sample_remote["TRADE_DATE"].iloc[0]
                == pd.Timestamp(SAMPLE_DATE)
            ),
            "sample_date": SAMPLE_DATE,
            "sample_pages": sample_pages,
            "sample_count": sample_count,
            "sample_local_count": int(len(sample_local)),
            "sample_code_date_sets_equal": bool(
                np.all(comparison["_merge"] == "both")
            ),
            "sample_all_retained_values_exactly_reproduced": bool(
                np.all(comparison["_merge"] == "both")
                and np.all(numeric_equal)
            ),
            "selected_month_ends_match_profile": bool(
                np.array_equal(
                    recomputed_month_ends.to_numpy(
                        dtype="datetime64[D]"
                    ),
                    selected_dates.to_numpy(dtype="datetime64[D]"),
                )
            ),
            "all_consecutive_snapshots_have_value_changes": bool(
                all(
                    row["changed_ratio_stocks"] > 0
                    for row in value_change_rows
                )
            ),
            "consecutive_snapshot_changes": value_change_rows,
        },
        "coverage": {
            "earliest_profile_trade_date": str(
                earliest_dates.min().date()
            ),
            "latest_profile_trade_date": str(
                earliest_dates.max().date()
            ),
            "earliest_profile_pages": earliest_pages,
            "period_profile_pages": period_pages,
            "local_selected_dates": int(len(local_dates)),
            "local_rows": int(len(local)),
            "local_unique_stocks": int(
                local["SECURITY_CODE"].nunique()
            ),
            "rows_per_snapshot_minimum": int(counts.min()),
            "rows_per_snapshot_maximum": int(counts.max()),
        },
        "duplicates_and_validity": {
            "duplicate_stock_dates": int(
                local.duplicated(
                    ["SECURITY_CODE", "TRADE_DATE"]
                ).sum()
            ),
            "missing_trade_dates": int(
                local["TRADE_DATE"].isna().sum()
            ),
            "missing_pledge_ratios": int(
                local["PLEDGE_RATIO"].isna().sum()
            ),
            "negative_pledge_ratios": int(
                np.count_nonzero(local["PLEDGE_RATIO"] < 0.0)
            ),
            "ratios_above_100": int(
                np.count_nonzero(local["PLEDGE_RATIO"] > 100.0)
            ),
        },
        "revision_semantics": {
            "detail_schema_fields": schema_fields,
            "explicit_revision_or_version_fields": revision_fields,
            "status": (
                "dated historical cross-sections are independently "
                "reproducible, but the endpoint exposes no explicit "
                "original-publication or revision-version timestamp"
            ),
            "activation_policy": (
                "conservatively activate on the first exchange row "
                "strictly after each CSDC TRADE_DATE"
            ),
        },
    }
    OUTPUT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
