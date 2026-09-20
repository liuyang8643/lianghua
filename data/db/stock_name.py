"""Local current-name lookup and validation of the CNINFO ST snapshot."""

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "stock_name"
_CURRENT_NAMES: pd.DataFrame | None = None
_ST_CHANGE_COLUMNS = frozenset({"bare_code", "date", "event", "status"})


def validate_st_changes(frame: pd.DataFrame) -> None:
    """Validate the complete local CNINFO ST/name-status snapshot."""
    missing = _ST_CHANGE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"st_changes 缺少字段: {sorted(missing)}")
    if frame.empty:
        raise ValueError("st_changes 不得为空")

    codes = frame["bare_code"].astype(str).str.strip()
    if not codes.str.fullmatch(r"\d{6}").all():
        raise ValueError("st_changes 包含非法 bare_code")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("st_changes 包含非法 date")
    for column in ("event", "status"):
        values = frame[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"st_changes 包含空 {column}")
    if pd.DataFrame({"code": codes, "date": dates}).duplicated().any():
        raise ValueError("st_changes 同一股票同日事件必须唯一")
    ordered = pd.DataFrame({"code": codes, "date": dates}).groupby(
        "code", sort=False
    )["date"].apply(lambda values: values.is_monotonic_increasing)
    if not ordered.all():
        raise ValueError("st_changes 同一股票的事件日期必须单调递增")


def _load_current_names() -> pd.DataFrame:
    global _CURRENT_NAMES
    if _CURRENT_NAMES is None:
        path = _DATA_DIR / "current_names.parquet"
        _CURRENT_NAMES = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return _CURRENT_NAMES


def invalidate_name_data_cache() -> None:
    """Clear the in-process current-name cache after publishing parquet."""
    global _CURRENT_NAMES
    _CURRENT_NAMES = None


def get_current_stock_name(stock_code: str) -> str | None:
    """当前简称 — 仅读 current_names.parquet（由 update_all 维护）。"""
    bare = stock_code.split('.')[0]
    df = _load_current_names()
    if df.empty or 'bare_code' not in df.columns:
        return None
    mask = df['bare_code'] == bare
    if not mask.any():
        return None
    nm = str(df.loc[mask, 'name'].iloc[-1]).strip()
    return nm or None
