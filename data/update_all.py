"""全量数据预下载更新脚本

update_offline_toNow()  — 先删除昨日不完整数据，再全量拉取到今天

用法:
  uv run python data/update_all.py

原则（来自 CLAUDE.md）：
  实盘开盘时触发预下载，此时获取到的日线 close/high/low 是盘中快照而非收盘值。
  因此次日必须先删除昨天的不完整数据，再重新拉取完整日线覆盖到今天。
"""
import time
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 预加载子模块避免多线程 import 死锁 (CNINFO + akshare)
import json as _json_lib
import requests  # noqa: F401
import py_mini_racer  # noqa: F401
from akshare.stock.stock_profile_cninfo import _get_file_content_ths  # noqa: F401
import akshare as ak  # noqa: F401

# 注意：不要在模块层面调用 logging.basicConfig — 它会抢占 Python stdlib 的 root logger，
# 导致 lark_oapi / xtquant 等三方库的日志被重复输出（飞书/QMT 日志会打两遍）。
# 独立运行时（__main__）才配置 root logger。


def _post_json_with_retry(url, max_attempts=3, **kwargs):
    """对 requests.post 做 3 次重试，仅对网络/JSON 解析错误 retry，其他直接 raise。

    网络抖动是预下载里最常见的临时错误；缺字段/缺数据应直接报错让人工排查。
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(url, **kwargs)
            return r.json()
        except (requests.RequestException, _json_lib.JSONDecodeError) as e:
            last_exc = e
            if attempt < max_attempts:
                time.sleep(min(attempt * 2, 5))
                continue
    raise RuntimeError(f"POST {url} 重试 {max_attempts} 次均失败: {last_exc}") from last_exc


class _UpdateAllLog:
    """统一走 trading_logger，盘后 run_update_all 与独立运行均可见进度。"""

    @staticmethod
    def _emit(level: str, msg: str, *args):
        from trading.logger import trading_logger
        text = (msg % args) if args else msg
        getattr(trading_logger, level)(text)

    def info(self, msg: str, *args):
        self._emit('info', msg, *args)

    def warning(self, msg: str, *args):
        self._emit('warning', msg, *args)


logger = _UpdateAllLog()

DATA_DIR = Path(__file__).resolve().parent
TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)


def _run_with_process_timeout(func, timeout: float = 60, label: str = ""):
    """在独立进程中运行 func，超时则 kill 进程（真杀，不残留卡死线程）。"""
    import multiprocessing as _mp

    def _target(conn):
        try:
            conn.send(('ok', func()))
        except Exception as e:
            conn.send(('err', e))

    ctx = _mp.get_context('spawn')
    parent_conn, child_conn = ctx.Pipe()
    p = ctx.Process(target=_target, args=(child_conn,))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        tag = f" ({label})" if label else ""
        raise TimeoutError(f"操作超时 {timeout}s{tag} (进程已 kill)")
    status, result = parent_conn.recv()
    if status == 'err':
        raise result
    return result

# 16:00 全量更新时，K 线删除并重拉「最近 N 个交易日」，用收盘后的完整 OHLC
# 覆盖开盘抓到的盘中快照（开盘只拉当天，不做覆盖）。
REPULL_TRADING_DAYS = 3


# ============================================================
# 辅助函数
# ============================================================

def _clean_parquet_by_date(path: Path, date_col: str, cutoff: date):
    """删除 parquet 中 date_col >= cutoff 的行，原地覆盖。"""
    if not path.exists():
        return
    df = pd.read_parquet(path)
    dates = pd.to_datetime(df[date_col])
    mask = dates.dt.date < cutoff
    if mask.sum() < len(df):
        df_clean = df[mask].reset_index(drop=True)
        df_clean.to_parquet(path, index=False)
        removed = len(df) - len(df_clean)
        logger.info("  清理 %s: 删除 %d 行 (>= %s)", path.name, removed, cutoff)


# ============================================================
# 1. K线日线 — QMT 唯一源（不复权 k-line/ + 官方 preClose；复权由 build_runtime 自建）
# ============================================================

def _update_kline():
    """mootdx 刷新：已有股票增量合并最近 REPULL_TRADING_DAYS 个交易日，新股全量补齐。"""
    # 今天已拉过则跳过（检查一只代表股即可）
    sample = DATA_DIR / "k-line" / "000001.SZ.parquet"
    if sample.exists():
        from datetime import datetime as _dt
        mtime = _dt.fromtimestamp(sample.stat().st_mtime).date()
        if mtime == TODAY:
            logger.info("[K线] 今日已更新, 跳过")
            return
    from data.kline_mootdx import update_recent
    logger.info("[K线] mootdx 重拉最近 %d 个交易日 + 新股全量", REPULL_TRADING_DAYS)
    update_recent(REPULL_TRADING_DAYS)


# ============================================================
# 2. 股票列表
# ============================================================

def _update_stock_list():
    from xtquant import xtdata
    codes = sorted(xtdata.get_stock_list_in_sector('沪深A股'))
    # get_all_stock_code_list already reads from xtdata source, just save
    df = pd.DataFrame({'stock_code': codes})
    for suffix in ['.SH', '.SZ', '.BJ']:
        mask = df['stock_code'].str.endswith(suffix)
        df.loc[mask, 'exchange'] = suffix.replace('.', '')
    path = DATA_DIR / "stock_list" / "stock_list.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("[股票列表] 保存 %d 只到 %s", len(codes), path)


# ============================================================
# 3. 股票名称/ST历史 — CNINFO API
# ============================================================

CURRENT_NAMES_REQUEST_TIMEOUT = 10
CURRENT_NAMES_CHUNK_SIZE = 250


def _tencent_quote_symbol(stock_code: str) -> str:
    from utils.stock.info import is_bse_stock

    bare = stock_code.split(".")[0]
    if is_bse_stock(bare):
        prefix = "bj"
    elif bare.startswith(("6", "9")):
        prefix = "sh"
    else:
        prefix = "sz"
    return f"{prefix}{bare}"


def _fetch_current_stock_names(codes: list[str]) -> pd.DataFrame:
    """当前简称全量表：腾讯财经批量行情只取简称，避免 akshare 黑盒调用阻塞。"""
    symbols = [_tencent_quote_symbol(code) for code in sorted(set(codes))]
    rows = []
    total_batches = (len(symbols) + CURRENT_NAMES_CHUNK_SIZE - 1) // CURRENT_NAMES_CHUNK_SIZE
    headers = {"User-Agent": "Mozilla/5.0"}
    for batch_idx, start in enumerate(range(0, len(symbols), CURRENT_NAMES_CHUNK_SIZE), 1):
        chunk = symbols[start:start + CURRENT_NAMES_CHUNK_SIZE]
        url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
        r = requests.get(url, headers=headers, timeout=CURRENT_NAMES_REQUEST_TIMEOUT)
        text = r.content.decode("gbk", errors="ignore")
        for item in text.split(";"):
            if '="' not in item:
                continue
            payload = item.split('"', 1)[1].rsplit('"', 1)[0]
            fields = payload.split("~")
            if len(fields) >= 3 and fields[1].strip() and fields[2].strip():
                rows.append({"code": fields[2].strip().zfill(6), "name": fields[1].strip()})
        if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == total_batches:
            logger.info("[当前简称] 腾讯批量 %d/%d, 已解析 %d 条", batch_idx, total_batches, len(rows))
    return pd.DataFrame(rows).drop_duplicates(subset=["code"], keep="last")


def _update_stock_name():
    from datetime import datetime
    import requests
    import py_mini_racer
    from akshare.stock.stock_profile_cninfo import _get_file_content_ths
    from data.db.stock_list import get_all_stock_code_list

    t_step = time.time()
    OUT_DIR = DATA_DIR / "stock_name"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name_path = OUT_DIR / "name_changes.parquet"
    st_path = OUT_DIR / "st_changes.parquet"
    cur_path = OUT_DIR / "current_names.parquet"

    # 当天已全量更新过，直接跳过（更名历史+ST+当前简称均为当日产物）
    if cur_path.exists():
        mtime = datetime.fromtimestamp(cur_path.stat().st_mtime).date()
        if mtime == TODAY:
            logger.info("[股票名称] 今日已更新, 跳过")
            return

    codes = get_all_stock_code_list()
    bare_codes = sorted(set(c.split('.')[0] for c in codes))
    logger.info("[股票名称] 开始: 全市场 %d 只 (ST + 更名历史 + 当前简称)", len(codes))

    js = py_mini_racer.MiniRacer()
    js.eval(_get_file_content_ths("cninfo.js"))
    mcode = js.call("getResCode1")
    headers = {
        "Accept": "*/*", "Accept-Enckey": mcode,
        "Origin": "https://webapi.cninfo.com.cn",
        "Referer": "https://webapi.cninfo.com.cn/",
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
    }
    CNINFO_BASE = "https://webapi.cninfo.com.cn/api/stock"

    # ===== ST 变更：全量一把拉（p_stock2117 支持不带 scode 返回全量） =====
    t0 = time.time()
    logger.info("[股票名称] 拉取 ST 变更 (CNINFO 全量)...")
    r = requests.post(f"{CNINFO_BASE}/p_stock2117", headers=headers, timeout=30)
    all_records = r.json().get("records", [])
    rows_st = []
    st_date_parse_fail = 0
    for rec in all_records:
        vary_str = rec.get("VARYDATE")
        event = rec.get("F006V", "")
        status = rec.get("F002V", "")
        sec_code = rec.get("SECCODE", "")
        if not vary_str or not sec_code:
            continue
        try:
            vary_date = datetime.strptime(vary_str, "%Y-%m-%d").date()
        except ValueError:
            st_date_parse_fail += 1
            continue
        rows_st.append({"bare_code": sec_code, "date": vary_date, "event": event, "status": status})

    if st_date_parse_fail:
        logger.warning("[股票名称] ST变更日期解析失败: %d 条", st_date_parse_fail)
    df_st = pd.DataFrame(rows_st)
    df_st = df_st.sort_values(["bare_code", "date"]).reset_index(drop=True)
    df_st.to_parquet(st_path, index=False)
    logger.info("[股票名称] ST变更: %d 条 (%.0fs), %d 只股票", len(df_st), time.time() - t0, df_st['bare_code'].nunique())

    # ===== 名称变更：增量串行拉取（p_stock2109 不支持批量，限流约 2-3 请求后需等待） =====
    name_existing = set()
    if name_path.exists():
        dn = pd.read_parquet(name_path)
        if not dn.empty and 'bare_code' in dn.columns:
            name_existing = set(dn['bare_code'].unique())

    name_pending = [b for b in bare_codes if b not in name_existing]
    t_name = time.time()
    if not name_pending:
        logger.info("[股票名称] 更名历史已齐 %d 只，跳过 CNINFO 增量", len(name_existing))
    else:
        logger.info("[股票名称] 更名历史待拉 %d 只 (已有 %d)，约每 100 只打一条进度",
                    len(name_pending), len(name_existing))
        rows_name = []
        name_date_parse_fail = 0
        for i, bare in enumerate(name_pending, 1):
            resp = _post_json_with_retry(f"{CNINFO_BASE}/p_stock2109",
                                         params={"scode": bare}, headers=headers, timeout=15)
            records = resp.get("records", [])
            for rec in records:
                start_str = rec.get("STARTDATE")
                old_name = rec.get("F002V", "")
                if not start_str or not old_name:
                    continue
                try:
                    start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
                except ValueError:
                    name_date_parse_fail += 1
                    continue
                rows_name.append({"bare_code": bare, "start_date": start_date, "old_name": old_name.strip()})

            if i == 1 or i % 100 == 0 or i == len(name_pending):
                logger.info("[股票名称] 更名历史 %d/%d (%.0fs)",
                            i, len(name_pending), time.time() - t_name)

        if name_date_parse_fail:
            logger.warning("[股票名称] 更名历史日期解析失败: %d 条", name_date_parse_fail)
        if rows_name:
            df_new = pd.DataFrame(rows_name)
            if name_path.exists() and 'bare_code' in pd.read_parquet(name_path).columns:
                df_old = pd.read_parquet(name_path)
                df_name = pd.concat([df_old, df_new], ignore_index=True)
                df_name = df_name.drop_duplicates(subset=['bare_code', 'start_date'], keep='last')
            else:
                df_name = df_new
            df_name.to_parquet(name_path, index=False)
            logger.info("[股票名称] 名称变更 新增 %d 条, 共 %d 条", len(rows_name), len(df_name))

    # 当前简称：每次 update_all 都必须刷新（与 name_changes 增量是否为空无关）
    _update_current_names(codes)
    logger.info("[股票名称] 全部完成 (%.0fs)", time.time() - t_step)


def _update_current_names(codes: list[str]):
    """写 current_names.parquet — 全市场当前简称（update_all 唯一写入入口）。

    读取侧只认此表 + name_changes（历史时点）；由腾讯财经批量行情对齐 stock_list。
    """
    from data.db.stock_name import invalidate_name_data_cache

    path = DATA_DIR / "stock_name" / "current_names.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 当天已成功落盘 → 跳过（同一交易日名称不变）
    if path.exists():
        from datetime import datetime as _dt
        mtime = _dt.fromtimestamp(path.stat().st_mtime).date()
        if mtime == TODAY:
            logger.info("[当前简称] 今日已更新, 跳过")
            return

    bare_to_code: dict[str, str] = {}
    for code in codes:
        bare_to_code.setdefault(code.split('.')[0], code)

    logger.info("[当前简称] 拉取腾讯财经批量简称 (%d 只待对齐)...", len(bare_to_code))
    df_names = _fetch_current_stock_names(codes)

    if df_names is None or df_names.empty:
        raise RuntimeError("[当前简称] 腾讯财经当前简称返回空，中止更新")

    name_map = dict(zip(
        df_names['code'].astype(str).str.zfill(6),
        df_names['name'].astype(str).str.strip(),
    ))
    rows = []
    for bare, stock_code in bare_to_code.items():
        key = bare.zfill(6) if len(bare) < 6 else bare
        nm = name_map.get(key)
        if nm:
            rows.append({'bare_code': bare, 'stock_code': stock_code, 'name': nm})

    df = pd.DataFrame(rows).drop_duplicates(subset=['bare_code'], keep='last')
    df.to_parquet(path, index=False)
    invalidate_name_data_cache()
    miss = len(bare_to_code) - len(df)
    logger.info(
        "[当前简称] %d 条已保存 (缺 %d, %.0fs) → %s",
        len(df), miss, time.time() - t0, path.name,
    )


# ============================================================
# 4. 资产负债表（总股本）— CNINFO API
# ============================================================

def _update_balance():
    import requests
    import py_mini_racer
    from akshare.stock.stock_profile_cninfo import _get_file_content_ths
    from concurrent.futures import ThreadPoolExecutor, as_completed

    OUT_PATH = DATA_DIR / "financial" / "balance.parquet"
    STOCK_LIST_PATH = DATA_DIR / "stock_list" / "stock_list.parquet"

    if STOCK_LIST_PATH.exists():
        codes = pd.read_parquet(STOCK_LIST_PATH)["stock_code"].tolist()
    else:
        from data.db.stock_list import get_all_stock_code_list
        codes = get_all_stock_code_list()

    bare_codes = sorted(set(c.split(".")[0] for c in codes))

    # 跳过已有数据的股票
    if OUT_PATH.exists():
        df_existing = pd.read_parquet(OUT_PATH)
        existing_full_codes = set(df_existing['stock_code'].unique())
        bare_codes = [b for b in bare_codes
                      if not any(b + s in existing_full_codes for s in ('.SH', '.SZ', '.BJ'))]
    if not bare_codes:
        logger.info("[资产负债表] 已是最新, 跳过")
        return
    logger.info("[资产负债表] 待获取: %d 只", len(bare_codes))
    js = py_mini_racer.MiniRacer()
    js.eval(_get_file_content_ths("cninfo.js"))
    mcode = js.call("getResCode1")
    headers = {
        "Accept": "*/*",
        "Accept-Enckey": mcode,
        "Origin": "https://webapi.cninfo.com.cn",
        "Referer": "https://webapi.cninfo.com.cn/",
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
    }

    MIN_DATE = "2009-07-01"

    def _fetch(bare_code):
        resp = _post_json_with_retry(
            "https://webapi.cninfo.com.cn/api/stock/p_stock2300",
            params={"scode": bare_code},
            headers=headers,
            timeout=30,
        )
        records = resp.get("records", [])
        rows = []
        for rec in records:
            dd = rec.get("DECLAREDATE", "")
            cap = rec.get("F062N")
            if not dd or cap is None or dd < MIN_DATE:
                continue
            suffix = ".SH" if bare_code.startswith(("6", "9")) else ".SZ" if bare_code.startswith(("0", "3")) else ".BJ"
            rows.append({"stock_code": bare_code + suffix, "m_anntime": dd, "cap_stk": float(cap)})
        return rows

    logger.info("[资产负债表] 下载 %d 只...", len(bare_codes))
    t0 = time.time()
    all_rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch, c): c for c in bare_codes}
        for fut in as_completed(futures):
            done += 1
            all_rows.extend(fut.result())
            if done % 500 == 0:
                logger.info("[资产负债表] %d/%d", done, len(bare_codes))

    if not all_rows:
        logger.warning("[资产负债表] 未下载到数据")
        return

    df_new = pd.DataFrame(all_rows)
    df_new["m_anntime"] = pd.to_datetime(df_new["m_anntime"])
    df_new["cap_stk"] = df_new["cap_stk"].astype(np.float64)

    if OUT_PATH.exists():
        df_old = pd.read_parquet(OUT_PATH)
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_merged = df_new

    df_merged = df_merged.drop_duplicates(subset=["stock_code", "m_anntime"], keep="last")
    df_merged = df_merged.sort_values(["stock_code", "m_anntime"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(OUT_PATH, index=False)
    logger.info("[资产负债表] 保存 %d 行, 耗时 %.0fs", len(df_merged), time.time() - t0)


# ============================================================
# 5. 每股财务指标 — xtdata PershareIndex
# ============================================================

def _update_pershare_index():
    from xtquant import xtdata
    from data.db.stock_list import get_all_stock_code_list

    OUT_PATH = DATA_DIR / "financial" / "pershare_index.parquet"
    codes = get_all_stock_code_list()

    logger.info("[每股指标] 下载 PershareIndex (xtdata)...")
    xtdata.download_financial_data(codes, '')

    all_rows = []
    error_count=0
    for code in codes:
        try:
            data = xtdata.get_financial_data([code], table_list=['PershareIndex'])
            if code in data.get('PershareIndex', {}):
                df = data['PershareIndex'][code]
                if df is not None and not df.empty:
                    df = df.copy()
                    df['stock_code'] = code
                    all_rows.append(df)
        except Exception as e:
            logger.warning(f"get_financial_data {code} 失败: {e}")
            error_count+=1
    logger.info(f"get_financial_data 失败{error_count} 成功{len(codes)-error_count}")

    if not all_rows:
        logger.warning("[每股指标] 未下载到数据")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=['stock_code', 'm_anntime'], keep='last')
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(OUT_PATH, index=False)
    logger.info("[每股指标] 保存 %d 行, %d 只股票", len(df_all), df_all['stock_code'].nunique())


# ============================================================
# 5b. 深历史财务指标 — akshare 同花顺（runtime 财务字段唯一来源）
# ============================================================

def _update_financial_deep():
    """全市场深历史财务（同花顺 stock_financial_abstract_ths，回溯至 1990s）。
    产物 data/financial/deep_indicators.parquet 是 build_runtime 财务字段的唯一来源。"""
    from data.update_financial_deep import main as _deep_main
    _deep_main()


# ============================================================
# 6. 发行价 — akshare
# ============================================================

def _update_issue_price():
    import akshare as ak

    OUT_PATH = DATA_DIR / "issue_price" / "issue_price.parquet"
    KLINE_DIR = DATA_DIR / "k-line"

    # 当天已落盘跳过
    if OUT_PATH.exists():
        from datetime import datetime as _dt
        mtime = _dt.fromtimestamp(OUT_PATH.stat().st_mtime).date()
        if mtime == TODAY:
            logger.info("[发行价] 今日已更新, 跳过")
            return

    codes_from_kline = sorted({f.stem[:6] for f in KLINE_DIR.glob("*.parquet")})
    done_codes = set()
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        done_codes = set(existing['stock_code'].astype(str).tolist())

    remaining = [c for c in codes_from_kline if c not in done_codes]
    if not remaining:
        logger.info("[发行价] 已是最新")
        return

    logger.info("[发行价] 下载 %d 只新股...", len(remaining))
    rows = []
    for bare in remaining:
        try:
            info = _run_with_process_timeout(
                lambda b=bare: ak.stock_individual_info_em(symbol=b),
                timeout=30, label=f"发行价 {bare}")
            price = None
            list_date = None
            for _, row in info.iterrows():
                if row['item'] == '发行价格':
                    price = float(row['value'])
                elif row['item'] == '上市时间':
                    list_date = str(row['value'])
            if price and price > 0:
                rows.append({"stock_code": bare, "issue_price": price, "list_date": list_date})
        except Exception as e:
            logger.warning("[发行价] %s akshare 失败: %s", bare, e)

    if rows:
        df_new = pd.DataFrame(rows)
        if OUT_PATH.exists():
            df_old = pd.read_parquet(OUT_PATH)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_parquet(OUT_PATH, index=False)
        logger.info("[发行价] 新增 %d 条, 共 %d 条", len(rows), len(df_all))


# ============================================================
# 7. 大盘指数 — akshare
# ============================================================

def _update_indices():
    import akshare as ak
    import pyarrow.parquet as pq
    import pyarrow as pa

    INDEX_INFO = {
        'sh000300': '沪深300',
        'sh000905': '中证500',
        'sh000852': '中证1000',
    }

    # 当天已落盘跳过（检查沪深300代表）
    sample_path = DATA_DIR / "index_sh000300_daily.parquet"
    if sample_path.exists():
        from datetime import datetime as _dt
        mtime = _dt.fromtimestamp(sample_path.stat().st_mtime).date()
        if mtime == TODAY:
            logger.info("[指数] 今日已更新, 跳过")
            return

    for symbol, name in INDEX_INFO.items():
        path = DATA_DIR / f"index_{symbol}_daily.parquet"
        try:
            df_new = _run_with_process_timeout(
                lambda s=symbol: ak.stock_zh_index_daily(symbol=s),
                timeout=60, label=f"指数 {name}")
            dates = df_new['date'].values
            open_prices = df_new['open'].values.astype(np.float64)

            if isinstance(dates[0], str) or isinstance(dates[0], np.str_):
                dates_np = np.array([np.datetime64(d[:10], 'D') for d in dates])
            else:
                dates_np = np.asarray(dates).astype('datetime64[D]')

            sort_idx = np.argsort(dates_np)
            dates_sorted = dates_np[sort_idx]
            open_sorted = open_prices[sort_idx]

            table = pa.table({
                'trade_date': pa.array(dates_sorted),
                'open': pa.array(open_sorted),
            })
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path)
            logger.info("[指数] %s(%s): %d 天, %s ~ %s",
                        name, symbol, len(dates_sorted), dates_sorted[0], dates_sorted[-1])
        except Exception as e:
            logger.warning("[指数] %s 失败: %s", symbol, e)


# ============================================================
# 9. 退市列表 — akshare
# ============================================================

def _update_delist():
    import akshare as ak

    # 当天已落盘跳过
    OUT_PATH = DATA_DIR / "delist" / "delist.parquet"
    if OUT_PATH.exists():
        from datetime import datetime as _dt
        mtime = _dt.fromtimestamp(OUT_PATH.stat().st_mtime).date()
        if mtime == TODAY:
            logger.info("[退市列表] 今日已更新, 跳过")
            return

    rows = []
    for fetch_func, market in [(ak.stock_info_sh_delist, 'SH'), (ak.stock_info_sz_delist, 'SZ')]:
        try:
            df = _run_with_process_timeout(fetch_func, timeout=60, label=f"退市列表 {market}")
            if df is not None and not df.empty:
                df['exchange'] = market
                rows.append(df)
        except Exception as e:
            logger.warning("[退市列表] %s 失败: %s", market, e)

    if rows:
        df_all = pd.concat(rows, ignore_index=True)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_parquet(OUT_PATH, index=False)
        logger.info("[退市列表] 保存 %d 条", len(df_all))
    else:
        logger.warning("[退市列表] 未下载到数据")


# ============================================================
# 10. 交易日历 — akshare
# ============================================================

def _update_trading_calendar():
    import akshare as ak
    import pyarrow as pa

    OUT_PATH = DATA_DIR / "trading_calendar.parquet"
    if OUT_PATH.exists():
        from datetime import datetime as _dt
        mtime = _dt.fromtimestamp(OUT_PATH.stat().st_mtime).date()
        if mtime == TODAY:
            logger.info("[交易日历] 今日已更新, 跳过")
            return

    df = _run_with_process_timeout(ak.tool_trade_date_hist_sina, timeout=60, label="交易日历")
    dates = sorted(set(df['trade_date']))
    table = pa.table({'trade_date': pa.array(dates)})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    pq.write_table(table, OUT_PATH)
    logger.info("[交易日历] 保存 %d 天", len(dates))


# ============================================================
# 11. 构建 Runtime NPZ
# ============================================================

def _build_runtime():
    from data.build_runtime import build_runtime
    logger.info("[Runtime] 构建全量 npz...")
    t0 = time.time()
    path = build_runtime()
    elapsed = time.time() - t0
    logger.info("[Runtime] 完成: %s (%.0fs)", path, elapsed)


# ============================================================
# 主入口
# ============================================================

def update_offline_toNow():
    """删除昨日不完整数据 → 全量预下载 → 构建 runtime npz"""
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("全量数据更新: %s → %s (删除昨日 %s 不完整数据)", YESTERDAY, TODAY, YESTERDAY)
    logger.info("=" * 60)

    # Phase 1: 清理指数最近不完整数据。
    # K 线无需在此清理：_update_kline(QMT update_recent) 会按 time 合并覆盖最近 N 个交易日。
    logger.info("--- Phase 1: 清理指数最近数据 ---")
    for f in DATA_DIR.glob("index_*_daily.parquet"):
        _clean_parquet_by_date(f, 'trade_date', YESTERDAY)

    # Phase 2: 全量下载
    logger.info("--- Phase 2: 下载更新 ---")
    steps = [
        ("股票列表", _update_stock_list),
        ("股票名称/ST", _update_stock_name),
        ("K线日线", _update_kline),
        ("资产负债表", _update_balance),
        ("深历史财务", _update_financial_deep),
        ("发行价", _update_issue_price),
        ("大盘指数", _update_indices),
        ("退市列表", _update_delist),
        ("交易日历", _update_trading_calendar),
        # 每股财务指标(xtdata PershareIndex)已弃用：runtime 财务字段改用「深历史财务」
        # (deep_indicators.parquet, 同花顺深历史) 单一来源，见 build_runtime.build_financial_arrays。
        # ("每股财务指标", _update_pershare_index),
    ]

    for name, func in steps:
        t1 = time.time()
        logger.info(">>> %s <<<", name)
        func()
        logger.info("<<< %s 完成 (%.0fs) >>>", name, time.time() - t1)

    # Phase 3: 构建 Runtime
    logger.info("--- Phase 3: 构建 Runtime ---")
    _build_runtime()

    logger.info("=" * 60)
    logger.info("全量更新完成! 总耗时 %.0fs", time.time() - t0)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    update_offline_toNow()
