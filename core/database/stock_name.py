"""历史股票简称查询

基于巨潮信息网(CNINFO) API 获取沪深两市股票的历史简称变更记录，
支持查询任意 T 日的股票简称（含 ST / *ST 状态）。

数据源:
  - p_stock2109: 证券简称变更记录（含变更日期）
  - p_stock2117: 特别处理(ST/退市)变动记录
"""

import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import py_mini_racer
import requests

from core.logger import core_logger

_CNINFO_BASE = "https://webapi.cninfo.com.cn/api/stock"
_CACHE_DIR = Path(__file__).parent / ".cache" / "stock_name"

_cninfo_headers: Optional[dict] = None


def _get_cninfo_headers() -> dict:
    """获取 CNINFO API 认证头（带 JS 加密 token）"""
    global _cninfo_headers
    if _cninfo_headers is not None:
        return _cninfo_headers

    from akshare.stock.stock_profile_cninfo import _get_file_content_ths

    js = py_mini_racer.MiniRacer()
    js.eval(_get_file_content_ths("cninfo.js"))
    mcode = js.call("getResCode1")
    _cninfo_headers = {
        "Accept": "*/*",
        "Accept-Enckey": mcode,
        "Origin": "https://webapi.cninfo.com.cn",
        "Referer": "https://webapi.cninfo.com.cn/",
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
    }
    return _cninfo_headers


def _ensure_cache_dir():
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(stock_code: str) -> Path:
    code = stock_code.split(".")[0]
    return _CACHE_DIR / f"{code}.pkl"


_MEM_CACHE: dict[str, dict] = {}

def _load_cache(stock_code: str) -> Optional[dict]:
    if stock_code in _MEM_CACHE:
        return _MEM_CACHE[stock_code]
    p = _cache_path(stock_code)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        _MEM_CACHE[stock_code] = data
        return data
    except Exception:
        return None


def _save_cache(stock_code: str, data: dict):
    _ensure_cache_dir()
    with open(_cache_path(stock_code), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _fetch_name_changes(bare_code: str) -> list[dict]:
    """CNINFO p_stock2109: 证券简称变更记录"""
    headers = _get_cninfo_headers()
    url = f"{_CNINFO_BASE}/p_stock2109"
    try:
        r = requests.post(url, params={"scode": bare_code}, headers=headers, timeout=15)
        records = r.json().get("records", [])
    except Exception as e:
        core_logger.warning(f"CNINFO p_stock2109 请求失败 {bare_code}: {e}")
        return []

    result = []
    for rec in records:
        start_str = rec.get("STARTDATE")
        old_name = rec.get("F002V", "")
        if not start_str or not old_name:
            continue
        try:
            start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        result.append({"start_date": start_date, "old_name": old_name.strip()})

    result.sort(key=lambda x: x["start_date"])
    return result


def _fetch_st_changes(bare_code: str) -> list[dict]:
    """CNINFO p_stock2117: 特别处理(ST/退市)变动记录"""
    headers = _get_cninfo_headers()
    url = f"{_CNINFO_BASE}/p_stock2117"
    try:
        r = requests.post(url, params={"scode": bare_code}, headers=headers, timeout=15)
        records = r.json().get("records", [])
    except Exception as e:
        core_logger.warning(f"CNINFO p_stock2117 请求失败 {bare_code}: {e}")
        return []

    result = []
    for rec in records:
        vary_str = rec.get("VARYDATE")
        event = rec.get("F006V", "")
        status = rec.get("F002V", "")
        if not vary_str:
            continue
        try:
            vary_date = datetime.strptime(vary_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        result.append({"date": vary_date, "event": event, "status": status})

    result.sort(key=lambda x: x["date"])
    return result


def _get_stock_history(stock_code: str, target_date: date | None = None) -> dict:
    """获取（带缓存的）股票名称和 ST 变更历史

    缓存策略：缓存记录 fetched_at（拉取日期），
    当 target_date <= fetched_at 时缓存可信（历史数据不变）；
    当 target_date > fetched_at 时可能缺少新事件，需重新拉取。
    """
    cached = _load_cache(stock_code)
    if cached is not None:
        fetched_at = cached.get("fetched_at")
        has_names = cached.get("intervals") or cached.get("current_name")
        if fetched_at is not None and has_names and (target_date is None or target_date <= fetched_at):
            return cached

    bare = stock_code.split(".")[0]
    name_changes = _fetch_name_changes(bare)
    st_changes = _fetch_st_changes(bare)

    # 从 p_stock2109 构建名称时间线:
    # 每条记录 = "在 start_date 这天，股票从 old_name 改为新名"
    # 因此 old_name 在 [上一个 start_date, 本 start_date) 区间有效
    # 当前名称通过 get_stock_detail 获取
    current_name = None
    try:
        from .detail import get_stock_detail
        detail = get_stock_detail(stock_code)
        if detail:
            current_name = detail.get("InstrumentName", "").strip()
    except Exception:
        pass

    # 构建 intervals: [(start_date, name), ...]
    # 表示从 start_date 起到下一个 interval 的 start_date 前一天，股票名为 name
    intervals = []
    if name_changes:
        # 第一条记录的 old_name 是首次变更前的名称
        intervals.append((date.min, name_changes[0]["old_name"]))
        for i, nc in enumerate(name_changes):
            if i + 1 < len(name_changes):
                intervals.append((nc["start_date"], name_changes[i + 1]["old_name"]))
            else:
                intervals.append((nc["start_date"], current_name or ""))
    elif current_name:
        intervals.append((date.min, current_name))

    data = {
        "intervals": intervals,
        "st_changes": st_changes,
        "current_name": current_name,
        "fetched_at": date.today(),
    }
    _save_cache(stock_code, data)
    return data


def get_stock_name_at_date(stock_code: str, target_date: date | datetime) -> Optional[str]:
    """查询股票在指定日期的简称

    Args:
        stock_code: QMT 格式股票代码（如 '600186.SH', '000023.SZ'）
        target_date: 目标日期（date 或 datetime）

    Returns:
        股票简称（如 '莲花味精', '*ST莲花'），查不到返回 None
    """
    if isinstance(target_date, datetime):
        target_date = target_date.date()

    history = _get_stock_history(stock_code, target_date)
    intervals = history.get("intervals", [])
    if not intervals:
        name = history.get("current_name")
    else:
        name = intervals[0][1]
        for start, n in intervals:
            if target_date >= start:
                name = n
            else:
                break

    if not name:
        try:
            from .detail import get_stock_detail
            detail = get_stock_detail(stock_code)
            if detail:
                name = detail.get("InstrumentName", "").strip()
        except Exception:
            pass

    return name or None


# 基于 event 字段的状态机（子串匹配，兼容组合事件如"戴帽披*"）
_KEEP_ST_KEYWORDS = ("披*", "退市整理", "戴帽", "暂停上市", "终止上市")
_CLEAR_ST_KEYWORDS = ("摘帽", "恢复上市", "新股上市", "重新上市", "转板上市", "摘*摘帽", "发行失败", "拟上市")


def _is_keep_event(event: str) -> bool:
    return any(kw in event for kw in _KEEP_ST_KEYWORDS)


def _is_clear_event(event: str) -> bool:
    return any(kw in event for kw in _CLEAR_ST_KEYWORDS)


def is_st_at_date(stock_code: str, target_date: date | datetime) -> bool:
    """判断股票在指定日期是否处于 ST / *ST / 退市整理 / 已终止上市 状态

    基于 p_stock2117 的 event 字段做状态机推演（子串匹配），
    兜底用简称判断（含"ST"或以"退"结尾）。
    """
    if isinstance(target_date, datetime):
        target_date = target_date.date()

    history = _get_stock_history(stock_code, target_date)
    st_changes = history.get("st_changes", [])

    if st_changes:
        st = False
        for sc in st_changes:
            if target_date >= sc["date"]:
                event = sc["event"]
                if _is_keep_event(event):
                    st = True
                elif _is_clear_event(event):
                    st = False
                else:
                    st = True  # 默认戴帽
            else:
                break
        return st

    name = get_stock_name_at_date(stock_code, target_date)
    if not name:
        return False
    return "ST" in name or name.endswith("退")


def is_star_st_at_date(stock_code: str, target_date: date | datetime) -> bool:
    """判断股票在指定日期是否处于 *ST / 退市整理 / 已终止上市 状态（不含普通 ST）"""
    if isinstance(target_date, datetime):
        target_date = target_date.date()

    history = _get_stock_history(stock_code, target_date)
    st_changes = history.get("st_changes", [])

    if st_changes:
        star = False
        for sc in st_changes:
            if target_date >= sc["date"]:
                star = _is_keep_event(sc["event"])
            else:
                break
        return star

    name = get_stock_name_at_date(stock_code, target_date)
    if not name:
        return False
    return "*ST" in name or name.endswith("退")


def prefetch_stock_histories(stock_codes: list[str], max_workers: int = 8) -> int:
    """批量预取股票历史名称/ST数据到本地缓存（多线程并行）

    Returns:
        实际从网络拉取的数量（已缓存的会跳过）
    """
    today = date.today()

    def _needs_fetch(code: str) -> bool:
        cached = _load_cache(code)
        if cached is None:
            return True
        fetched_at = cached.get("fetched_at")
        return fetched_at is None or fetched_at < today

    uncached = [c for c in stock_codes if _needs_fetch(c)]
    if not uncached:
        # 全部命中磁盘缓存，预加载到内存缓存避免热路径磁盘 I/O
        for c in stock_codes:
            _load_cache(c)
        core_logger.info(f"ST历史数据全部命中本地缓存 ({len(stock_codes)} 只)")
        return 0

    core_logger.info(
        f"ST历史数据预取: 共 {len(stock_codes)} 只, "
        f"已缓存 {len(stock_codes) - len(uncached)}, 待下载 {len(uncached)}"
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    done = 0
    failed = 0
    t0 = time.time()
    log_interval = 15

    last_log = t0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_get_stock_history, code, today): code for code in uncached}
        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
            except Exception:
                failed += 1
                done += 1
            now = time.time()
            if now - last_log >= log_interval or done == len(uncached):
                elapsed = now - t0
                speed = done / elapsed if elapsed > 0 else 0
                eta = (len(uncached) - done) / speed if speed > 0 else 0
                core_logger.info(
                    f"ST历史数据预取进度: {done}/{len(uncached)} "
                    f"({done / len(uncached) * 100:.1f}%), "
                    f"耗时 {elapsed:.0f}s, 预计剩余 {eta:.0f}s"
                    + (f", 失败 {failed}" if failed else "")
                )
                last_log = now

    return len(uncached) - failed


def build_st_mask(stock_codes: list[str], trade_dates: list[date]) -> "pd.DataFrame":
    """构建 ST / *ST / 退市 状态掩码面板。

    Args:
        stock_codes: 股票代码列表
        trade_dates: 交易日列表

    Returns:
        DataFrame(index=trade_dates, columns=stock_codes)，
        True 表示该股票在该交易日处于 ST / *ST / 退市整理 / 已终止上市状态。
    """
    import numpy as np
    import pandas as pd

    if not stock_codes or not trade_dates:
        return pd.DataFrame()

    n_dates = len(trade_dates)
    n_stocks = len(stock_codes)
    trade_date_arr = np.array([pd.Timestamp(d) for d in trade_dates], dtype='datetime64[ns]')
    result = np.zeros((n_dates, n_stocks), dtype=bool)

    for j, code in enumerate(stock_codes):
        history = _get_stock_history(code, trade_dates[-1])
        st_changes = history.get("st_changes", [])

        if st_changes:
            change_dates = np.array([pd.Timestamp(sc["date"]) for sc in st_changes], dtype='datetime64[ns]')
            change_states = np.array([
                _is_keep_event(sc["event"]) or not _is_clear_event(sc["event"])
                for sc in st_changes
            ], dtype=bool)

            indices = np.searchsorted(change_dates, trade_date_arr, side='right') - 1
            valid = indices >= 0
            if valid.any():
                result[valid, j] = change_states[indices[valid]]
        else:
            name = get_stock_name_at_date(code, trade_dates[-1])
            if name and ("ST" in name or name.endswith("退")):
                result[:, j] = True

    return pd.DataFrame(result, index=trade_dates, columns=stock_codes)


def clear_stock_name_cache():
    """清除全部股票名称缓存"""
    if _CACHE_DIR.exists():
        count = 0
        for f in _CACHE_DIR.glob("*.pkl"):
            f.unlink()
            count += 1
        core_logger.info(f"已清除 {count} 个股票名称缓存文件")
