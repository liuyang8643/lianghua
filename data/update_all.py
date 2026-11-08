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

# 预加载子模块避免运行中重复 import 抖动 (CNINFO + akshare)
import json as _json_lib
import requests  # noqa: F401
import py_mini_racer  # noqa: F401
from akshare.stock.stock_profile_cninfo import _get_file_content_ths  # noqa: F401
import akshare as ak  # noqa: F401
from functools import lru_cache
_LAST_CNINFO_REQUEST = 0.0

# 注意：不要在模块层面调用 logging.basicConfig — 它会抢占 Python stdlib 的 root logger，
# 导致 lark_oapi / xtquant 等三方库的日志被重复输出（飞书/QMT 日志会打两遍）。
# 独立运行时（__main__）才配置 root logger。


@lru_cache(maxsize=1)
def _cninfo_session():
    from data.update_cninfo import CninfoSession
    return CninfoSession()


def _post_json_with_retry(url, max_attempts=3, **kwargs):
    """对 requests.post 做 3 次重试，仅对网络/JSON 解析错误 retry，其他直接 raise。

    网络抖动是预下载里最常见的临时错误；缺字段/缺数据应直接报错让人工排查。
    """
    global _LAST_CNINFO_REQUEST
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            time.sleep(max(0, 1.1 - (time.monotonic() - _LAST_CNINFO_REQUEST)))
            _LAST_CNINFO_REQUEST = time.monotonic()
            r = _cninfo_session().post(url, **kwargs)
            r.raise_for_status()
            payload = r.json()
            if payload['resultcode'] != 200 or not isinstance(payload['records'], list):
                raise ValueError('CNINFO response rejected: %s %s' %
                                 (payload['resultcode'], payload.get('resultmsg', 'malformed records')))
            if payload['count'] != len(payload['records']) or payload['total'] != payload['count']:
                raise ValueError('CNINFO response is incomplete or paginated')
            return payload
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


def current_update_date() -> date:
    """Return the run date dynamically for long-lived trading processes."""
    return date.today()



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
# 1. K线日线 — mootdx 优先、QMT 备用（不复权 k-line/ + preClose；复权由 build_runtime 自建）
# ============================================================

def _update_kline(*, anchor_date=None, days=REPULL_TRADING_DAYS, allow_qmt=True):
    """历史日线刷新：已有股票增量合并，新股全量补齐，mootdx 失效时使用 QMT。

    退市股 parquet 缺失时通过统一日线入口补齐，补不齐则中止。
    """
    from data.update_kline import update_recent, update_full
    from data.kline_mootdx import RECENT_BARS
    logger.info("[K线] mootdx 重拉最近 %d 个交易日 + 新股全量", days)
    options = {} if allow_qmt else {'allow_qmt': False}
    if days >= RECENT_BARS:
        update_full(end=anchor_date.strftime('%Y%m%d') if anchor_date else None, **options)
    else:
        update_recent(days, anchor_date=anchor_date, **options)
    _ensure_delist_kline_mootdx(anchor_date=anchor_date, **options)


def _ensure_delist_kline_mootdx(*, anchor_date=None, allow_qmt=True):
    """确保退市股 K 线存在；缺失时经统一入口补齐，补不齐则中止。"""
    from data.db.delist import get_delist_stock_info
    from data.update_kline import update_full
    from utils.stock.info import is_b_stock

    kline_dir = DATA_DIR / "k-line"
    delist_info = get_delist_stock_info()
    missing = [c for c in delist_info
               if not (kline_dir / f'{c}.parquet').exists() and not is_b_stock(c)]
    if not missing:
        return

    logger.info("[K线-退市] mootdx 补齐缺失退市股 %d 只", len(missing))
    options = {} if allow_qmt else {'allow_qmt': False}
    update_full(codes=missing, end=anchor_date.strftime('%Y%m%d') if anchor_date else None, **options)
    still_missing = [c for c in missing if not (kline_dir / f'{c}.parquet').exists()]
    if still_missing:
        shown = ', '.join(still_missing[:20])
        suffix = f" ...(+{len(still_missing) - 20})" if len(still_missing) > 20 else ''
        raise RuntimeError(f"[K线-退市] mootdx 未补齐 {len(still_missing)} 只退市股: {shown}{suffix}")


# ============================================================
# 2. 股票列表
# ============================================================

def _update_stock_list(*, source='qmt'):
    if source == 'qmt':
        from xtquant import xtdata
        codes = sorted(xtdata.get_stock_list_in_sector('沪深A股'))
    elif source == 'mootdx':
        from data.kline_mootdx import _connect_mootdx, _close_mootdx
        quote = _connect_mootdx(require_daily=False)
        codes = []
        try:
            for market, suffix, prefixes in ((0, '.SZ', ('00', '30')), (1, '.SH', ('60', '68'))):
                stocks = quote.stocks(market=market)
                if stocks is None or stocks.empty:
                    raise RuntimeError('mootdx stock list is incomplete')
                if len(stocks) != quote.stock_count(market=market):
                    raise RuntimeError('mootdx stock-list count mismatch')
                raw_codes = stocks['code'].astype(str).str.zfill(6)
                if raw_codes.duplicated().any() or not raw_codes.str.fullmatch(r'\d{6}').all():
                    raise RuntimeError('mootdx stock-list pagination/codes are inconsistent')
                codes.extend(str(code).zfill(6) + suffix for code in stocks['code']
                             if str(code).zfill(6).startswith(prefixes))
        finally:
            _close_mootdx(quote)
        codes = sorted(set(codes))
    else:
        raise ValueError('Unknown stock-list source: ' + source)
    if not codes:
        raise RuntimeError("[股票列表] QMT 返回空股票池，拒绝覆盖")
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
        r.raise_for_status()
        text = r.content.decode("gbk")
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


def _update_stock_name(*, include_history=True):
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

    codes = get_all_stock_code_list()
    bare_codes = sorted(set(c.split('.')[0] for c in codes))
    logger.info("[股票名称] 开始: 全市场 %d 只 (%s)", len(codes),
                'ST + 更名历史 + 当前简称' if include_history else 'ST + 当前简称，启动所需字段')

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
    payload = _post_json_with_retry(f"{CNINFO_BASE}/p_stock2117", headers=headers, timeout=30)
    all_records = payload['records']
    if not all_records:
        raise ValueError('CNINFO returned no ST history')
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
        raise ValueError('[股票名称] ST变更日期解析失败: %d 条' % st_date_parse_fail)
    df_st = pd.DataFrame(rows_st)
    df_st = df_st.sort_values(["bare_code", "date"]).reset_index(drop=True)
    from data.db.stock_name import validate_st_changes, invalidate_name_data_cache
    validate_st_changes(df_st)
    df_st.to_parquet(st_path, index=False)
    invalidate_name_data_cache()
    logger.info("[股票名称] ST变更: %d 条 (%.0fs), %d 只股票", len(df_st), time.time() - t0, df_st['bare_code'].nunique())

    # ST and current-name coverage are required strategy inputs. Historical
    # display names are best effort and must not block Runtime publication.
    _update_current_names(codes)
    if include_history:
        status = dict(trade_date=current_update_date().isoformat(), required=False,
                      state='completed')
        try:
            _update_name_history(bare_codes, js=js, headers=headers)
        except Exception as error:
            status.update(state='skipped', error=str(error))
            logger.warning('[更名历史] 展示资料暂跳过，保留旧数据，不阻断 Runtime: %s', error)
        status_path = OUT_DIR / ('name_history_status_' + current_update_date().isoformat() + '.json')
        temporary = status_path.with_suffix('.tmp')
        try:
            temporary.write_text(_json_lib.dumps(status, ensure_ascii=False), encoding='utf-8')
            temporary.replace(status_path)
        except OSError as error:
            logger.warning('[更名历史] 可选状态记录失败，不阻断 Runtime: %s', error)
    logger.info('[股票名称] 必需 ST/当前简称已完成 (%.0fs)', time.time() - t_step)


def _update_name_history(bare_codes, *, js, headers, budget_seconds=30):
    """Optional display history: one request attempt, bounded total work."""
    from datetime import datetime
    OUT_DIR = DATA_DIR / 'stock_name'
    name_path = OUT_DIR / 'name_changes.parquet'
    CNINFO_BASE = 'https://webapi.cninfo.com.cn/api/stock'
    # ===== 名称变更：增量串行拉取（p_stock2109 不支持批量，限流约 2-3 请求后需等待） =====
    name_existing = set()
    if name_path.exists():
        dn = pd.read_parquet(name_path)
        if not dn.empty and 'bare_code' in dn.columns:
            name_existing = set(dn['bare_code'].unique())

    name_pending = [b for b in bare_codes if b not in name_existing]
    receipt_path = OUT_DIR / ('name_history_' + current_update_date().isoformat() + '.json')
    receipts = _json_lib.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {}
    t_name = time.time()
    deadline = time.monotonic() + budget_seconds
    if not name_pending:
        logger.info("[股票名称] 更名历史已齐 %d 只，跳过 CNINFO 增量", len(name_existing))
    else:
        logger.info("[股票名称] 更名历史待拉 %d 只 (已有 %d)，约每 100 只打一条进度",
                    len(name_pending), len(name_existing))
        rows_name = []
        name_date_parse_fail = 0
        for i, bare in enumerate(name_pending, 1):
            if time.monotonic() >= deadline:
                raise TimeoutError('Optional name history time budget exhausted')
            if bare not in receipts:
                headers['Accept-Enckey'] = js.call('getResCode1')
                resp = _post_json_with_retry(f"{CNINFO_BASE}/p_stock2109",
                                             params={"scode": bare}, headers=headers,
                                             max_attempts=1, timeout=min(10, max(0.1, deadline - time.monotonic())))
                records = resp['records']
                if any(rec['SECCODE'] != bare for rec in records):
                    raise ValueError('CNINFO returned another stock name history')
                receipts[bare] = records
                temporary = receipt_path.with_suffix('.tmp')
                temporary.write_text(_json_lib.dumps(receipts, ensure_ascii=False), encoding='utf-8')
                for attempt in range(5):
                    try:
                        temporary.replace(receipt_path)
                        break
                    except PermissionError:
                        if attempt == 4:
                            raise
                        time.sleep(.1)  # Windows readers briefly deny rename sharing.
            records = receipts[bare]
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
            raise ValueError('[股票名称] 更名历史日期解析失败: %d 条' % name_date_parse_fail)
        if rows_name:
            df_new = pd.DataFrame(rows_name)
            if name_path.exists() and 'bare_code' in pd.read_parquet(name_path).columns:
                df_old = pd.read_parquet(name_path)
                df_name = pd.concat([df_old, df_new], ignore_index=True)
                df_name = df_name.drop_duplicates(subset=['bare_code', 'start_date'], keep='last')
            else:
                df_name = df_new
            temporary = name_path.with_suffix('.parquet.tmp')
            df_name.to_parquet(temporary, index=False)
            temporary.replace(name_path)
            logger.info("[股票名称] 名称变更 新增 %d 条, 共 %d 条", len(rows_name), len(df_name))



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
        if mtime == current_update_date():
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
    required = {code for code in codes if (DATA_DIR / 'k-line' / (code + '.parquet')).exists()}
    if required:
        from data.db.delist import get_delist_stock_info
        delisted = get_delist_stock_info()
        required = {code for code in required if code not in delisted
                    or delisted[code].delist_date > current_update_date()}
        missing = required - set(df['stock_code'])
        if missing:
            raise RuntimeError('Current names missing active stocks: ' + ','.join(sorted(missing)[:20]))
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
    for done, bare_code in enumerate(bare_codes, start=1):
        all_rows.extend(_fetch(bare_code))
        if done % 500 == 0:
            logger.info("[资产负债表] %d/%d", done, len(bare_codes))

    if not all_rows:
        raise RuntimeError(
            f"[资产负债表] {len(bare_codes)} 只待更新股票均未返回数据")

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
# 5. 深历史财务指标 — akshare 同花顺（runtime 财务字段唯一来源）
# ============================================================

def _update_financial_deep():
    """全市场深历史财务（同花顺 stock_financial_abstract_ths，回溯至 1990s）。
    产物 data/financial/deep_indicators.parquet 是 build_runtime 财务字段的唯一来源。"""
    from data.update_financial_deep import main as _deep_main
    _deep_main()


# ============================================================
# 6. 发行价 — 巨潮上市资料
# ============================================================

def _update_issue_price():

    OUT_PATH = DATA_DIR / "issue_price" / "issue_price.parquet"
    KLINE_DIR = DATA_DIR / "k-line"

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
    failures = []
    js = py_mini_racer.MiniRacer()
    js.eval(_get_file_content_ths('cninfo.js'))
    for bare in remaining:
        try:
            if bare == '688287':
                # 转板不发行新股；不能把 21.88 元开盘参考价冒充发行价。
                # 公司转板上市报告书（2022-05-24）：
                # https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=8243345&stockid=688287
                rows.append({'stock_code': bare, 'issue_price': np.nan,
                             'list_date': '2022-05-25',
                             'source': 'company-transfer-listing-report-20220524',
                             'source_as_of': date.today(),
                             'issue_price_status': 'not_applicable_transfer'})
                continue
            # stock_individual_info_em 是实时行情接口，没有发行价字段。
            # 巨潮上市资料提供真实发行价，包括换代码/重新上市的历史发行信息。
            payload = _post_json_with_retry(
                'https://webapi.cninfo.com.cn/api/sysapi/p_sysapi1134',
                params={'scode': bare},
                headers={'Accept-Enckey': js.call('getResCode1'),
                         'Referer': 'https://webapi.cninfo.com.cn/',
                         'User-Agent': 'Mozilla/5.0'}, timeout=15)
            if payload['resultcode'] != 200 or len(payload['records']) != 1:
                raise ValueError('巨潮发行信息缺失或多义')
            info = payload['records'][0]
            if info['SECCODE'] != bare:
                raise ValueError('巨潮返回错误股票代码')
            price = float(info['F008N'])
            if not np.isfinite(price) or price <= 0:
                raise ValueError(f'发行价非法: {price}')
            list_date = str(pd.Timestamp(info['F006D']).date())
            rows.append({'stock_code': bare, 'issue_price': price,
                         'list_date': list_date, 'source': 'cninfo-p_sysapi1134',
                         'source_as_of': date.today()})
        except Exception as e:
            logger.warning("[发行价] %s 巨潮失败: %s", bare, e)
            failures.append(f"{bare}: {e}")

    if rows:
        df_new = pd.DataFrame(rows)
        if OUT_PATH.exists():
            df_old = pd.read_parquet(OUT_PATH)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        # 旧文件存 Python date，巨潮返回 ISO 字符串；统一后再写 Parquet。
        df_all['list_date'] = pd.to_datetime(
            df_all['list_date'].astype(str), format='mixed', errors='raise').dt.date
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        from data.db.issue_price import save_issue_reference_atomic
        save_issue_reference_atomic(df_all,path=OUT_PATH)
        logger.info("[发行价] 新增 %d 条, 共 %d 条", len(rows), len(df_all))
    if failures:
        raise RuntimeError(
            f"[发行价] {len(failures)}/{len(remaining)} 只下载失败: "
            + '; '.join(failures[:20])
        )


# ============================================================
# 7. 大盘指数 — 新浪历史 + 腾讯补当天
# 新浪 hisdata/klc_kl.js 收盘后仍停在前一交易日，16:00 校验会因此失败。
# 腾讯日线在收盘后已含当日开盘价，只补新浪缺失的日期。
# ============================================================

TENCENT_INDEX_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
TENCENT_INDEX_RECENT_BARS = 120


def _index_max_date(df):
    if df is None or df.empty or 'date' not in df.columns:
        return None
    dates = pd.to_datetime(df['date'])
    if dates.isna().all():
        return None
    return dates.max().date()


def _parse_tencent_index_kline(payload, symbol):
    if payload.get('code') != 0:
        raise RuntimeError(f"腾讯指数 {symbol} 返回失败: {payload.get('msg')}")
    rows = payload['data'][symbol]['day']
    if not rows:
        raise RuntimeError(f"腾讯指数 {symbol} 日线为空")
    frame = pd.DataFrame(rows).iloc[:, :2]
    frame.columns = ['date', 'open']
    frame['date'] = pd.to_datetime(frame['date'])
    frame['open'] = pd.to_numeric(frame['open'])
    valid = frame['date'].notna() & frame['open'].notna() & (frame['open'] > 0)
    frame = frame.loc[valid, ['date', 'open']]
    if frame.empty:
        raise RuntimeError(f"腾讯指数 {symbol} 无有效开盘价")
    return frame.reset_index(drop=True)


def _fetch_tencent_index_daily(symbol):
    response = requests.get(
        TENCENT_INDEX_KLINE_URL,
        params={'param': f'{symbol},day,,,{TENCENT_INDEX_RECENT_BARS},'},
        headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://gu.qq.com/'},
        timeout=15,
    )
    response.raise_for_status()
    return _parse_tencent_index_kline(response.json(), symbol)


def _append_missing_index_days(base, extra):
    """只把新浪没有的交易日补进去，已有开盘价保持新浪原值。"""
    if base is None or base.empty:
        return extra
    have = set(pd.to_datetime(base['date']).dt.date)
    extra_days = pd.to_datetime(extra['date']).dt.date
    missing = extra.loc[~extra_days.isin(have), ['date', 'open']]
    if missing.empty:
        return base
    return pd.concat([base, missing], ignore_index=True)


def _update_indices(*, end_date=None):
    from datetime import datetime as _dt

    import akshare as ak
    import pyarrow.parquet as pq
    import pyarrow as pa

    INDEX_INFO = {
        'sh000300': '沪深300',
        'sh000905': '中证500',
        'sh000852': '中证1000',
    }

    # 只有全部指数均在当天落盘才跳过。若上次运行在中途失败，
    # 不能因第一个文件的 mtime 而把后续缺失项误标成成功。
    run_date = current_update_date()
    from utils.stock.time import get_last_trading_day
    required_end = get_last_trading_day(end_date or run_date)
    index_paths = [DATA_DIR / f"index_{symbol}_daily.parquet" for symbol in INDEX_INFO]

    def is_current(path):
        if not path.exists() or _dt.fromtimestamp(path.stat().st_mtime).date() != run_date:
            return False
        try:
            dates = pd.read_parquet(path, columns=['trade_date'])['trade_date']
            return pd.to_datetime(dates).max().date() >= required_end
        except (OSError, ValueError, KeyError):
            return False

    if end_date is None and all(is_current(path) for path in index_paths):
        logger.info("[指数] 今日已全部更新, 跳过")
        return

    failures = []
    for symbol, name in INDEX_INFO.items():
        path = DATA_DIR / f"index_{symbol}_daily.parquet"
        try:
            df_new = ak.stock_zh_index_daily(symbol=symbol)
            if not df_new.empty and (_index_max_date(df_new) or date.min) < required_end:
                logger.info("[指数] %s 新浪未覆盖 %s，补腾讯日线", symbol, required_end)
                df_new = _append_missing_index_days(df_new, _fetch_tencent_index_daily(symbol))
            if end_date is not None:
                df_new = df_new[pd.to_datetime(df_new['date']).dt.date <= end_date]
            if df_new.empty or pd.to_datetime(df_new['date']).max().date() < required_end:
                raise RuntimeError(f'指数 {symbol} 数据未覆盖 {required_end}')
            open_prices = df_new['open'].values.astype(np.float64)
            dates_np = pd.to_datetime(df_new['date']).to_numpy(dtype='datetime64[ns]').astype('datetime64[D]')

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
            failures.append(f"{symbol}: {e}")
    if failures:
        raise RuntimeError(f"[指数] {len(failures)} 个下载失败: {'; '.join(failures)}")


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
        if mtime == current_update_date():
            logger.info("[退市列表] 今日已更新, 跳过")
            return

    rows = []
    for fetch_func, market in [(ak.stock_info_sh_delist, 'SH'), (ak.stock_info_sz_delist, 'SZ')]:
        df = fetch_func()
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
    from utils.stock import time as stock_time

    OUT_PATH = DATA_DIR / "trading_calendar.parquet"
    if OUT_PATH.exists():
        from datetime import datetime as _dt
        mtime = _dt.fromtimestamp(OUT_PATH.stat().st_mtime).date()
        try:
            calendar_end = pd.to_datetime(pd.read_parquet(OUT_PATH)['trade_date']).max().date()
        except (OSError, ValueError, KeyError):
            calendar_end = date.min
        if mtime == current_update_date() and calendar_end >= current_update_date():
            logger.info("[交易日历] 今日已更新, 跳过")
            stock_time._TRADING_CALENDAR_STATE = None
            return

    df = ak.tool_trade_date_hist_sina()
    dates = sorted(set(df['trade_date']))
    if not dates:
        raise RuntimeError("[交易日历] 数据源返回空数据")
    table = pa.table({'trade_date': pa.array(dates)})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    pq.write_table(table, OUT_PATH)
    stock_time._TRADING_CALENDAR_STATE = None
    logger.info("[交易日历] 保存 %d 天", len(dates))


# ============================================================
# 11. 构建 Runtime NPZ
# ============================================================

def _build_runtime(*, end_date=None):
    from data.build_runtime import build_runtime
    logger.info("[Runtime] 构建全量 npz...")
    t0 = time.time()
    path = build_runtime(end_date=end_date)
    elapsed = time.time() - t0
    logger.info("[Runtime] 完成: %s (%.0fs)", path, elapsed)
    return path


# ============================================================
# 主入口
# ============================================================

def update_offline_toNow():
    """删除昨日不完整数据 → 全量预下载 → 构建 runtime npz"""
    t0 = time.time()
    today = current_update_date()
    yesterday = today - timedelta(days=1)

    logger.info("=" * 60)
    logger.info("全量数据更新: %s → %s (删除昨日 %s 不完整数据)",
                yesterday, today, yesterday)
    logger.info("=" * 60)

    # Phase 1: 清理指数最近不完整数据。
    # K 线无需在此清理：mootdx update_recent 会按 time 合并覆盖最近 N 个交易日。
    logger.info("--- Phase 1: 清理指数最近数据 ---")
    for f in DATA_DIR.glob("index_*_daily.parquet"):
        _clean_parquet_by_date(f, 'trade_date', yesterday)

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
