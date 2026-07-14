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
import queue
import threading
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 预加载子模块避免运行中重复 import 抖动 (CNINFO + akshare)
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



# 16:00 全量更新时，K 线删除并重拉「最近 N 个交易日」，用收盘后的完整 OHLC
# 覆盖开盘抓到的盘中快照（开盘只拉当天，不做覆盖）。
REPULL_TRADING_DAYS = 3
EXTERNAL_CALL_TIMEOUT = 10


def _call_with_timeout(func, timeout=EXTERNAL_CALL_TIMEOUT):
    """Run a potentially blocking third-party call and return after timeout."""
    result = queue.Queue(maxsize=1)

    def _run():
        try:
            result.put((True, func()))
        except BaseException as exc:
            result.put((False, exc))

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"external call exceeded {timeout}s")

    ok, value = result.get_nowait()
    if not ok:
        raise value
    return value

INDEX_INFO = {
    'sh000001': '上证指数',
    'H00300': '沪深300全收益',
    'sh000905': '中证500',
    'sh000852': '中证1000',
}
INDEX_REQUIRED_COLUMNS = ('trade_date', 'open', 'close')


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


def _is_valid_index_parquet(path: Path, min_trade_date: date | None = None) -> bool:
    import pyarrow.parquet as pq

    if not path.exists():
        return False
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows == 0 or not set(INDEX_REQUIRED_COLUMNS) <= set(parquet.schema.names):
        return False
    trade_dates = pd.to_datetime(pd.read_parquet(path, columns=['trade_date'])['trade_date'], errors='coerce')
    if trade_dates.isna().any():
        return False
    return min_trade_date is None or trade_dates.max().date() >= min_trade_date


def _indices_ready_today(symbols=None) -> bool:
    from datetime import datetime as _dt
    from utils.stock.time import get_last_trading_day

    target_symbols = tuple(symbols) if symbols is not None else tuple(INDEX_INFO)
    min_trade_date = get_last_trading_day(TODAY - timedelta(days=1))
    for symbol in target_symbols:
        path = DATA_DIR / f"index_{symbol}_daily.parquet"
        if not _is_valid_index_parquet(path, min_trade_date):
            return False
        if _dt.fromtimestamp(path.stat().st_mtime).date() != TODAY:
            return False
    return True


# ============================================================
# 1. K线日线 — mootdx 唯一源（不复权 k-line/ + preClose；复权由 build_runtime 自建）
# ============================================================

def _update_kline(anchor_date: date | None = None):
    """mootdx 刷新：已有股票增量合并最近 REPULL_TRADING_DAYS 个交易日，新股全量补齐。

    退市股 parquet 缺失时仍只允许通过 mootdx 全量补齐，禁止混用其他 K 线源。
    """
    from data.kline_mootdx import update_recent
    logger.info("[K线] mootdx 重拉最近 %d 个交易日 + 新股全量", REPULL_TRADING_DAYS)
    update_recent(REPULL_TRADING_DAYS, anchor_date=anchor_date)
    _ensure_delist_kline_mootdx()


def _ensure_delist_kline_mootdx():
    """确保退市股 K 线存在；缺失时用 mootdx 全量补齐，补不齐则中止。"""
    from data.db.delist import get_delist_stock_info
    from data.kline_mootdx import update_full
    from utils.stock.info import is_b_stock

    kline_dir = DATA_DIR / "k-line"
    delist_info = get_delist_stock_info()
    missing = [c for c in delist_info
               if not (kline_dir / f'{c}.parquet').exists() and not is_b_stock(c)]
    if not missing:
        return

    logger.info("[K线-退市] mootdx 补齐缺失退市股 %d 只", len(missing))
    update_full(codes=missing)
    still_missing = [c for c in missing if not (kline_dir / f'{c}.parquet').exists()]
    if still_missing:
        shown = ', '.join(still_missing[:20])
        suffix = f" ...(+{len(still_missing) - 20})" if len(still_missing) > 20 else ''
        raise RuntimeError(f"[K线-退市] mootdx 未补齐 {len(still_missing)} 只退市股: {shown}{suffix}")


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
    r = requests.post(f"{CNINFO_BASE}/p_stock2117", headers=headers, timeout=EXTERNAL_CALL_TIMEOUT)
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
            try:
                resp = _call_with_timeout(
                    lambda: _post_json_with_retry(
                        f"{CNINFO_BASE}/p_stock2109",
                        max_attempts=1,
                        params={"scode": bare},
                        headers=headers,
                        timeout=EXTERNAL_CALL_TIMEOUT,
                    )
                )
            except TimeoutError:
                logger.warning("[股票名称] %s 请求超过 %ds，跳过", bare, EXTERNAL_CALL_TIMEOUT)
                continue
            except Exception as e:
                logger.warning("[股票名称] %s 请求失败，跳过: %s", bare, e)
                continue
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
# 4. 资产负债表（总股本）— akshare CNINFO
# ============================================================

def _derive_missing_total_share(deep: pd.DataFrame, known_codes: set[str]) -> pd.DataFrame:
    deep = deep[~deep['stock_code'].isin(known_codes)].copy()
    deep['cap_stk'] = pd.to_numeric(deep['net_profit'], errors='coerce') / pd.to_numeric(deep['eps'], errors='coerce') / 10_000
    deep = deep[np.isfinite(deep['cap_stk']) & (deep['cap_stk'] > 0)]
    deep['m_anntime'] = pd.to_datetime(deep['report_period'].astype(str), format='%Y%m%d') + pd.Timedelta(days=120)
    return deep[['stock_code', 'm_anntime', 'cap_stk']]


def _update_balance():
    OUT_PATH = DATA_DIR / "financial" / "balance.parquet"
    DERIVED_PATH = DATA_DIR / "financial" / "balance_derived.parquet"
    DEEP_PATH = DATA_DIR / "financial" / "deep_indicators.parquet"
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"[资产负债表] 缺少官方股本数据: {OUT_PATH}")
    if not DEEP_PATH.exists():
        raise FileNotFoundError(f"[资产负债表] 缺少深历史财务数据: {DEEP_PATH}")

    df_old = pd.read_parquet(OUT_PATH)
    deep = pd.read_parquet(DEEP_PATH, columns=['stock_code', 'report_period', 'net_profit', 'eps'])
    df_new = _derive_missing_total_share(deep, set(df_old['stock_code']))
    if df_new.empty:
        logger.info("[资产负债表] 无需补全")
        return

    DERIVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_new.sort_values(['stock_code', 'm_anntime']).to_parquet(DERIVED_PATH, index=False)
    logger.info("[资产负债表] 以深历史财务补全 %d 只、%d 条",
                df_new['stock_code'].nunique(), len(df_new))


# ============================================================
# 5. 深历史财务指标 — akshare 同花顺（runtime 财务字段唯一来源）
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
            info = _call_with_timeout(lambda: ak.stock_individual_info_em(symbol=bare))
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

def _update_indices(symbols=None):
    import akshare as ak
    import pyarrow.parquet as pq
    import pyarrow as pa

    target_symbols = tuple(symbols) if symbols is not None else tuple(INDEX_INFO)

    if symbols is None and _indices_ready_today():
        logger.info("[指数] 今日已更新, 跳过")
        return

    for symbol in target_symbols:
        name = INDEX_INFO[symbol]
        path = DATA_DIR / f"index_{symbol}_daily.parquet"
        if symbol == 'H00300':
            df_new = ak.stock_zh_index_hist_csindex(symbol='H00300', start_date='20050101', end_date='20991231')
            dates = df_new['日期'].values
            open_prices = np.full(len(dates), np.nan, dtype=np.float64)
            close_prices = df_new['收盘'].values.astype(np.float64)
        else:
            df_new = ak.stock_zh_index_daily(symbol=symbol)
            dates = df_new['date'].values
            open_prices = df_new['open'].values.astype(np.float64)
            close_prices = df_new['close'].values.astype(np.float64)

        if isinstance(dates[0], str) or isinstance(dates[0], np.str_):
            dates_np = np.array([np.datetime64(d[:10], 'D') for d in dates])
        else:
            dates_np = np.asarray(dates).astype('datetime64[D]')

        sort_idx = np.argsort(dates_np)
        dates_sorted = dates_np[sort_idx]
        open_sorted = open_prices[sort_idx]
        close_sorted = close_prices[sort_idx]

        table = pa.table({
            'trade_date': pa.array(dates_sorted),
            'open': pa.array(open_sorted),
            'close': pa.array(close_sorted),
        })
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        if not _is_valid_index_parquet(path):
            raise RuntimeError(f"[指数] {symbol} 写出缺列文件: {path}")
        logger.info("[指数] %s(%s): %d 天, %s ~ %s",
                    name, symbol, len(dates_sorted), dates_sorted[0], dates_sorted[-1])


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
            df = _call_with_timeout(fetch_func)
        except TimeoutError:
            logger.warning("[退市列表] %s 请求超过 %ds，跳过本次退市列表更新", market, EXTERNAL_CALL_TIMEOUT)
            return
        except Exception as e:
            logger.warning("[退市列表] %s 请求失败，跳过本次退市列表更新: %s", market, e)
            return
        if df is not None and not df.empty:
            df['exchange'] = market
            rows.append(df)

    if rows:
        df_all = pd.concat(rows, ignore_index=True)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_parquet(OUT_PATH, index=False)
        logger.info("[退市列表] 保存 %d 条", len(df_all))
    else:
        raise RuntimeError("[退市列表] 未下载到数据")


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

    df = ak.tool_trade_date_hist_sina()
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
    # K 线无需在此清理：mootdx update_recent 会按 time 合并覆盖最近 N 个交易日。
    logger.info("--- Phase 1: 清理指数最近数据 ---")
    for f in DATA_DIR.glob("index_*_daily.parquet"):
        _clean_parquet_by_date(f, 'trade_date', YESTERDAY)

    # Phase 2: 全量下载
    logger.info("--- Phase 2: 下载更新 ---")
    steps = [
        ("股票列表", _update_stock_list),
        ("退市列表", _update_delist),
        ("股票名称/ST", _update_stock_name),
        ("K线日线", _update_kline),
        ("资产负债表", _update_balance),
        ("深历史财务", _update_financial_deep),
        ("发行价", _update_issue_price),
        ("大盘指数", _update_indices),
        ("交易日历", _update_trading_calendar),
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
