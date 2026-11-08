"""Local current-name lookup and validation of the CNINFO ST snapshot."""

from pathlib import Path

import pandas as pd
import numpy as np

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "stock_name"
_CURRENT_NAMES: pd.DataFrame | None = None
_ST_CHANGE_COLUMNS = frozenset({"bare_code", "date", "event", "status"})


def get_stock_name_at_date(stock_code, at_date=None):
    """Display labels use current verified names; never an input to selection."""
    return get_current_stock_name(stock_code)


def _load_st_changes():
    frame=pd.read_parquet(_DATA_DIR/'st_changes.parquet')
    validate_st_changes(frame)
    return frame


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


def build_st_mask(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
) -> np.ndarray:
    """从 st_changes.parquet 构建 ST 掩码。

    Returns:
        bool ndarray (n_dates, n_stocks), True = ST/*ST/退市状态
    """
    n_dates = len(trade_dates)
    n_stocks = len(stock_codes)
    df_st = _load_st_changes()

    validate_st_changes(df_st)

    trade_date_arr = trade_dates.astype('datetime64[ns]')
    result = np.zeros((n_dates, n_stocks), dtype=bool)

    _KEEP_ST_KEYWORDS = ("披*", "退市整理", "戴帽", "暂停上市", "终止上市")
    _CLEAR_KEYWORDS = ("摘帽", "恢复上市", "新股上市", "重新上市", "转板上市", "摘*摘帽", "发行失败", "拟上市")

    for j, code in enumerate(stock_codes):
        bare = str(code).split('.')[0]
        sc_records = df_st[df_st['bare_code'] == bare]
        if sc_records.empty:
            continue

        dates = sc_records['date'].values.astype('datetime64[ns]')
        events = sc_records['event'].values

        # _KEEP_ST_KEYWORDS 优先：命中则保持 ST
        is_keep = np.array([any(kw in str(e) for kw in _KEEP_ST_KEYWORDS) for e in events])
        is_clear = np.array([any(kw in str(e) for kw in _CLEAR_KEYWORDS) for e in events])
        # keep 优先级最高, clear 次之, 其余保持 ST
        status_changes = np.where(is_keep, True, ~is_clear)

        indices = np.searchsorted(dates, trade_date_arr, side='right') - 1
        valid = indices >= 0
        if valid.any():
            result[valid, j] = status_changes[indices[valid]]

    print(f"ST掩码: {result.sum()} 个 True / {result.size} ({result.sum()/result.size*100:.1f}%)")
    return result

