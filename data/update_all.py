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
logger = logging.getLogger("update_all")

DATA_DIR = Path(__file__).resolve().parent
TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)


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
# 1. K线日线 — mootdx + xtdata 兜底
# ============================================================

def _update_kline():
    """更新全量K线到今日。已有股票移除昨日数据后增量追加。"""
    from data.db.history import get_history_data
    from data.db.stock_list import get_all_stock_code_list
    from datetime import datetime

    KLINE_DIR = DATA_DIR / "k-line"
    KLINE_DIR.mkdir(parents=True, exist_ok=True)

    codes = get_all_stock_code_list()
    logger.info("[K线] 待处理 %d 只股票", len(codes))

    existing = {f.stem for f in KLINE_DIR.glob("*.parquet")}
    new_stocks = [c for c in codes if c not in existing]
    update_stocks = [c for c in codes if c in existing]

    logger.info("[K线] 新下载 %d 只, 增量更新 %d 只", len(new_stocks), len(update_stocks))

    # 先处理已有股票：删除昨日数据，增量下载
    cutoff_ms = int(pd.Timestamp(YESTERDAY).timestamp() * 1000)

    updated = 0
    t0 = time.time()
    for i, code in enumerate(update_stocks, 1):
        path = KLINE_DIR / f"{code}.parquet"
        try:
            df_time = pd.read_parquet(path, columns=['time'])
            last_ts = int(df_time['time'].max())
            removed = int((df_time['time'] >= cutoff_ms).sum())
            last_date = pd.to_datetime(last_ts, unit='ms').date() if last_ts else date(1990, 1, 1)

            need_dl = last_date < YESTERDAY or removed > 0
            if not need_dl:
                continue

            # 需要下载时再读全量
            df_old = pd.read_parquet(path)
            df_old = df_old[df_old['time'] < cutoff_ms]

            # 用 count=20 获取最近K线（足够覆盖 last_date ~ today 的增量）
            result = get_history_data([code], count=max(20, (TODAY - last_date).days + 5),
                                      base_time=datetime.now(), period="1d")
            new_data = result.get(code)
            if new_data is None:
                df_old.to_parquet(path, index=False)
                continue

            df_new = pd.DataFrame(new_data)
            df_new = df_new[df_new['time'] > last_ts]
            if df_new.empty:
                df_old.to_parquet(path, index=False)
                continue

            df_merged = pd.concat([df_old, df_new], ignore_index=True)
            df_merged.sort_values('time', ascending=False, inplace=True)
            df_merged.reset_index(drop=True, inplace=True)
            df_merged.to_parquet(path, index=False)
            updated += 1
        except Exception as e:
            logger.warning("[K线] %s 更新失败: %s", code, e)

        if i % 500 == 0:
            elapsed = time.time() - t0
            speed = i / elapsed if elapsed > 0 else 0
            logger.info("[K线] 增量进度: %d/%d (速度 %.0f只/s, ETA %.0fs)",
                        i, len(update_stocks), speed, (len(update_stocks) - i) / speed if speed > 0 else 0)

    logger.info("[K线] 增量更新: %d/%d 只", updated, len(update_stocks))

    # 处理新股票 — 用 get_history_data(count=None) 下载全量
    new_ok = 0
    for i, code in enumerate(new_stocks, 1):
        try:
            result = get_history_data([code], count=None, base_time=datetime.now(), period="1d")
            new_data = result.get(code)
            if new_data is not None:
                df_out = pd.DataFrame(new_data)
                df_out.to_parquet(KLINE_DIR / f"{code}.parquet", index=False)
                new_ok += 1
        except Exception as e:
            logger.warning("[K线] %s 新股下载失败: %s", code, e)

        if i % 500 == 0:
            logger.info("[K线] 新下载进度: %d/%d, 成功 %d", i, len(new_stocks), new_ok)

    logger.info("[K线] 新下载: %d/%d 只", new_ok, len(new_stocks))


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

def _update_stock_name():
    from datetime import datetime
    import requests
    import py_mini_racer
    from akshare.stock.stock_profile_cninfo import _get_file_content_ths
    from data.db.stock_list import get_all_stock_code_list

    OUT_DIR = DATA_DIR / "stock_name"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name_path = OUT_DIR / "name_changes.parquet"
    st_path = OUT_DIR / "st_changes.parquet"
    codes = get_all_stock_code_list()
    bare_codes = sorted(set(c.split('.')[0] for c in codes))

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
    r = requests.post(f"{CNINFO_BASE}/p_stock2117", headers=headers, timeout=30)
    all_records = r.json().get("records", [])
    rows_st = []
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
            continue
        rows_st.append({"bare_code": sec_code, "date": vary_date, "event": event, "status": status})

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
    if not name_pending:
        logger.info("[股票名称] 名称变更已是最新 (%d 只)", len(name_existing))
        return

    logger.info("[股票名称] 名称变更待获取: %d 只 (已有 %d)", len(name_pending), len(name_existing))

    rows_name = []
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
                continue
            rows_name.append({"bare_code": bare, "start_date": start_date, "old_name": old_name.strip()})

        if i % 100 == 0:
            elapsed = time.time() - t0
            logger.info("[股票名称] 名称 %d/%d (%.0fs)", i, len(name_pending), elapsed)

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

    # ===== 当前简称：用 xtdata 全量拉 InstrumentName，写 current_names.parquet =====
    # 解决「从未更名股票」在 name_changes 中无记录导致名称缺失的问题
    _update_current_names(codes)


def _update_current_names(codes: list[str]):
    """用 xtdata 全量拉每只股票的当前简称，写 data/stock_name/current_names.parquet。

    解决方案：CNINFO p_stock2109 只覆盖更名历史（29.9% 股票），从未更名的
    股票（创业板/科创板新股）在那里返回 0 条。xtdata 本地 instrument_detail
    覆盖全市场，离线可用，无需网络。
    """
    from xtquant import xtdata

    t0 = time.time()
    rows = []
    miss = 0
    for code in codes:
        try:
            detail = xtdata.get_instrument_detail(code)
            if not detail:
                miss += 1
                continue
            name = (detail.get('InstrumentName', '') or '').strip()
            if not name:
                miss += 1
                continue
            rows.append({
                'bare_code': code.split('.')[0],
                'stock_code': code,
                'name': name,
            })
        except Exception:
            miss += 1

    if not rows:
        logger.warning("[当前简称] xtdata 全量拉取失败，跳过 current_names.parquet 更新")
        return

    path = DATA_DIR / "stock_name" / "current_names.parquet"
    df = pd.DataFrame(rows).drop_duplicates(subset=['bare_code'], keep='last')
    df.to_parquet(path, index=False)
    logger.info(
        "[当前简称] %d 条已保存 (miss=%d, %.0fs) → %s",
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
    for code in codes:
        try:
            data = xtdata.get_financial_data([code], table_list=['PershareIndex'])
            if code in data.get('PershareIndex', {}):
                df = data['PershareIndex'][code]
                if df is not None and not df.empty:
                    df = df.copy()
                    df['stock_code'] = code
                    all_rows.append(df)
        except Exception:
            pass

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
            info = ak.stock_individual_info_em(symbol=bare)
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

    for symbol, name in INDEX_INFO.items():
        path = DATA_DIR / f"index_{symbol}_daily.parquet"
        try:
            df_new = ak.stock_zh_index_daily(symbol=symbol)
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

    OUT_PATH = DATA_DIR / "delist" / "delist.parquet"
    rows = []
    for fetch_func, market in [(ak.stock_info_sh_delist, 'SH'), (ak.stock_info_sz_delist, 'SZ')]:
        try:
            df = fetch_func()
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

    # Phase 1: 清理昨日不完整数据
    logger.info("--- Phase 1: 清理昨日数据 ---")
    kline_dir = DATA_DIR / "k-line"
    if kline_dir.exists():
        cutoff_ms = int(pd.Timestamp(YESTERDAY).timestamp() * 1000)
        cleaned = 0
        for f in kline_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(f)
                before = len(df)
                df_clean = df[df['time'] < cutoff_ms]
                if len(df_clean) < before:
                    df_clean.to_parquet(f, index=False)
                    cleaned += 1
            except Exception as e:
                logger.warning("[K线清理] %s 失败: %s", f.name, e)
        logger.info("K线数据清理: %d 只股票移除昨日数据", cleaned)

    # 指数数据清理
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
