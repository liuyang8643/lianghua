"""历史股票简称查询

从预下载 parquet 读取 CNINFO 数据，无网络调用。
"""

import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

import logging

data_logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / ".cache" / "stock_name"
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "stock_name"

# 全量数据内存缓存（首次加载后常驻）
_NAME_CHANGES: pd.DataFrame | None = None
_ST_CHANGES: pd.DataFrame | None = None
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


def _load_name_changes() -> pd.DataFrame:
    global _NAME_CHANGES
    if _NAME_CHANGES is None:
        path = _DATA_DIR / "name_changes.parquet"
        _NAME_CHANGES = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return _NAME_CHANGES


def _load_st_changes() -> pd.DataFrame:
    global _ST_CHANGES
    if _ST_CHANGES is None:
        path = _DATA_DIR / "st_changes.parquet"
        if not path.exists():
            raise FileNotFoundError(f"ST 历史快照不存在: {path}")
        frame = pd.read_parquet(path)
        validate_st_changes(frame)
        _ST_CHANGES = frame
    return _ST_CHANGES


def _load_current_names() -> pd.DataFrame:
    global _CURRENT_NAMES
    if _CURRENT_NAMES is None:
        path = _DATA_DIR / "current_names.parquet"
        _CURRENT_NAMES = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return _CURRENT_NAMES


def invalidate_name_data_cache() -> None:
    """update_all 写入 parquet 后调用，清空进程内缓存。"""
    global _NAME_CHANGES, _ST_CHANGES, _CURRENT_NAMES
    _NAME_CHANGES = None
    _ST_CHANGES = None
    _CURRENT_NAMES = None
    _MEM_CACHE.clear()


def get_current_stock_name(stock_code: str) -> Optional[str]:
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


_MEM_CACHE: dict[str, dict] = {}


def _load_cache(stock_code: str) -> Optional[dict]:
    if stock_code in _MEM_CACHE:
        return _MEM_CACHE[stock_code]
    code = stock_code.split(".")[0]
    p = _CACHE_DIR / f"{code}.pkl"
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        _MEM_CACHE[stock_code] = data
        return data
    except Exception:
        return None


def _get_stock_history(stock_code: str, target_date: date | None = None) -> dict:
    """获取（带缓存的）股票名称和 ST 变更历史，从本地 parquet 读取。"""
    cached = _load_cache(stock_code)
    if cached is not None:
        return cached

    bare = stock_code.split(".")[0]

    # 名称变更
    df_name = _load_name_changes()
    name_rows = df_name[df_name['bare_code'] == bare].to_dict('records') if not df_name.empty and 'bare_code' in df_name.columns else []

    # ST变更
    df_st = _load_st_changes()
    st_rows = df_st[df_st['bare_code'] == bare].to_dict('records') if not df_st.empty and 'bare_code' in df_st.columns else []

    current_name = get_current_stock_name(stock_code)

    intervals = []
    if name_rows:
        valid_names = [nc for nc in name_rows
                       if not (nc.get("start_date") is None or pd.isna(nc.get("start_date")))]
        if valid_names:
            intervals.append((date.min, valid_names[0]["old_name"]))
            for i, nc in enumerate(valid_names):
                if i + 1 < len(valid_names):
                    intervals.append((nc["start_date"], valid_names[i + 1]["old_name"]))
                else:
                    intervals.append((nc["start_date"], current_name or ""))
        elif current_name:
            intervals.append((date.min, current_name))
    elif current_name:
        intervals.append((date.min, current_name))

    st_rows = [sc for sc in st_rows
               if not (sc.get("date") is None or pd.isna(sc.get("date")))]

    data = {
        "intervals": intervals,
        "st_changes": st_rows,
        "current_name": current_name,
        "fetched_at": date.today(),
    }
    _MEM_CACHE[stock_code] = data
    return data


def get_stock_name_at_date(stock_code: str, target_date: date | datetime) -> Optional[str]:
    """查询股票在指定日期的简称"""
    if isinstance(target_date, datetime):
        target_date = target_date.date()

    history = _get_stock_history(stock_code, target_date)
    intervals = history.get("intervals", [])
    if not intervals:
        return history.get("current_name") or get_current_stock_name(stock_code)

    name = intervals[0][1]
    for start, n in intervals:
        if start is None or pd.isna(start):
            continue
        if target_date >= start:
            name = n
        else:
            break
    return name or None


# 基于 event 字段的状态机（子串匹配，兼容组合事件如"戴帽披*"）
_KEEP_ST_KEYWORDS = ("披*", "退市整理", "戴帽", "暂停上市", "终止上市")
_CLEAR_ST_KEYWORDS = ("摘帽", "恢复上市", "新股上市", "重新上市", "转板上市", "摘*摘帽", "发行失败", "拟上市")


def _is_keep_event(event: str) -> bool:
    return any(kw in event for kw in _KEEP_ST_KEYWORDS)


def _is_clear_event(event: str) -> bool:
    return any(kw in event for kw in _CLEAR_ST_KEYWORDS)


def is_st_at_date(stock_code: str, target_date: date | datetime) -> bool:
    """判断股票在指定日期是否处于 ST / *ST / 退市整理 / 已终止上市 状态。

    若历史 ST 事件记录缺失（st_changes 为空），一律返回 False，
    避免把"当前 ST 状态"回投到历史日期造成数据泄露。
    """
    if isinstance(target_date, datetime):
        target_date = target_date.date()

    history = _get_stock_history(stock_code, target_date)
    st_changes = history.get("st_changes", [])
    if not st_changes:
        return False

    st = False
    for sc in st_changes:
        d = sc.get("date")
        if d is None or pd.isna(d):
            continue
        if target_date >= d:
            event = sc["event"]
            if _is_keep_event(event):
                st = True
            elif _is_clear_event(event):
                st = False
            else:
                st = True
        else:
            break
    return st


def is_star_st_at_date(stock_code: str, target_date: date | datetime) -> bool:
    """判断股票在指定日期是否处于 *ST / 退市整理 / 已终止上市 状态（不含普通 ST）。

    若历史 ST 事件记录缺失，一律返回 False（避免回投当前状态造成数据泄露）。
    """
    if isinstance(target_date, datetime):
        target_date = target_date.date()

    history = _get_stock_history(stock_code, target_date)
    st_changes = history.get("st_changes", [])
    if not st_changes:
        return False

    star = False
    for sc in st_changes:
        d = sc.get("date")
        if d is None or pd.isna(d):
            continue
        if target_date >= d:
            star = _is_keep_event(sc["event"])
        else:
            break
    return star


def prefetch_stock_histories(stock_codes: list[str], max_workers: int = 8) -> int:
    """批量预加载股票历史名称/ST数据到内存缓存（从 parquet 读取，无网络）。

    Returns:
        成功加载的数量
    """
    df_name = _load_name_changes()
    df_st = _load_st_changes()
    loaded = 0

    for code in stock_codes:
        if code in _MEM_CACHE:
            loaded += 1
            continue
        bare = code.split(".")[0]

        name_rows = df_name[df_name['bare_code'] == bare].to_dict('records') if not df_name.empty and 'bare_code' in df_name.columns else []
        st_rows = df_st[df_st['bare_code'] == bare].to_dict('records') if not df_st.empty and 'bare_code' in df_st.columns else []

        current_name = get_current_stock_name(code)

        intervals = []
        if name_rows:
            valid_names = [nc for nc in name_rows
                           if not (nc.get("start_date") is None or pd.isna(nc.get("start_date")))]
            if valid_names:
                intervals.append((date.min, valid_names[0]["old_name"]))
                for i, nc in enumerate(valid_names):
                    if i + 1 < len(valid_names):
                        intervals.append((nc["start_date"], valid_names[i + 1]["old_name"]))
                    else:
                        intervals.append((nc["start_date"], current_name or ""))
            elif current_name:
                intervals.append((date.min, current_name))
        elif current_name:
            intervals.append((date.min, current_name))

        st_rows = [sc for sc in st_rows
                   if not (sc.get("date") is None or pd.isna(sc.get("date")))]

        _MEM_CACHE[code] = {
            "intervals": intervals,
            "st_changes": st_rows,
            "current_name": current_name,
            "fetched_at": date.today(),
        }
        loaded += 1

    if loaded:
        data_logger.info(f"ST历史数据预加载: {loaded} 只 (从本地 parquet)")
    return loaded


def clear_stock_name_cache():
    """清除全部股票名称缓存"""
    if _CACHE_DIR.exists():
        count = 0
        for f in _CACHE_DIR.glob("*.pkl"):
            f.unlink()
            count += 1
        data_logger.info(f"已清除 {count} 个股票名称缓存文件")


def build_st_mask(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
    df_st: pd.DataFrame | None = None,
) -> np.ndarray:
    """从 st_changes.parquet 构建 ST 掩码。

    Returns:
        bool ndarray (n_dates, n_stocks), True = ST/*ST/退市状态
    """
    n_dates = len(trade_dates)
    n_stocks = len(stock_codes)
    if df_st is None:
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
