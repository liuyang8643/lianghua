"""历史日线入口：mootdx 优先，失效时使用已登录 QMT 的不复权日线。"""
import numpy as np
import pandas as pd
import requests
import json
import time
import random
from tdxpy.exceptions import TdxFunctionCallError, TdxConnectionError

from data import kline_mootdx as mdx


FIELDS = ('time', 'open', 'high', 'low', 'close', 'volume', 'amount', 'preClose')
_LAST_REFERENCE_REQUEST = 0.0


def _reference_rows(report, filter_text, name):
    global _LAST_REFERENCE_REQUEST
    params = dict(reportName=report, columns='ALL', pageSize=500, pageNumber=1, filter=filter_text)
    rows = []
    for page in range(1, 51):
        params['pageNumber'] = page
        time.sleep(max(0, 1.0 + random.random() * .1 - (time.monotonic() - _LAST_REFERENCE_REQUEST)))
        _LAST_REFERENCE_REQUEST = time.monotonic()
        response = requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get',
                                params=params, timeout=15)
        response.raise_for_status()
        result = response.json()['result']
        if result is None:
            raise RuntimeError(f'市场状态查询失败: {name}')
        rows.extend(result['data'])
        if page >= result['pages']:
            break
    else:
        raise RuntimeError('市场状态分页超过上限')
    path = mdx.DATA_DIR / 'market_status' / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    return rows


def _ipo_missing_bars_allowed(row, end):
    """上市日尚未进入本次下载截止日时，空日线是预期结果。"""
    listing = None if row['LISTING_DATE'] is None else pd.Timestamp(row['LISTING_DATE']).date()
    end_day = pd.Timestamp(end).date()
    if listing is None:
        return row['CONTINUOUS_1WORD_NUM'] == '待上市'
    return listing > end_day


def _suspended_dates(code, missing, cache):
    confirmed = []
    verified_path = mdx.ROOT / 'configs' / 'verified_suspensions.json'
    verified = json.loads(verified_path.read_text(encoding='utf-8')) if verified_path.exists() else []
    for day in missing:
        key = day.strftime('%Y-%m-%d')
        if key not in cache:
            cache[key] = _reference_rows('RPT_CUSTOM_SUSPEND_DATA_INTERFACE',
                f'(MARKET="全部")(DATETIME=\'{key}\')', f'suspensions_{key}')
        for row in [*cache[key], *verified]:
            if row['SECUCODE'] != code:
                continue
            first = pd.Timestamp(row['SUSPEND_START_TIME'])
            last = pd.Timestamp(row['SUSPEND_END_TIME']) if row['SUSPEND_END_TIME'] else None
            if first <= day + pd.Timedelta(hours=9, minutes=30) and (last is None or day + pd.Timedelta(hours=15) <= last):
                confirmed.append(day)
                break
    return pd.DatetimeIndex(confirmed)


def _validate_qmt_dates(frame, expected, code):
    actual = pd.to_datetime(frame.index.astype(str), format='%Y%m%d')
    missing = expected.difference(actual)
    if len(missing):
        raise RuntimeError(f'QMT {code} 缺少 {len(missing)} 个交易日: {missing[0]:%Y%m%d}~{missing[-1]:%Y%m%d}')
    if actual.has_duplicates or not actual.is_monotonic_increasing:
        raise RuntimeError(f'QMT {code} 交易日重复或乱序')


def _validate_preserved_history(path):
    frame = pd.read_parquet(path, columns=list(FIELDS))
    if frame.empty or frame.time.duplicated().any() or not frame.time.is_monotonic_increasing:
        raise ValueError(f'退市历史为空/时间异常: {path.name}')
    values = frame[['open', 'high', 'low', 'close', 'volume', 'amount']].to_numpy(float)
    # 旧退市文件包含停牌 NaN、早期未提供的量额 NaN；保留缺失，不伪造行情。
    if (np.isinf(values).any() or (values[:, :4] <= 0).any() or (values[:, 4:] < 0).any()
            or not np.isfinite(values[:, :4]).all(axis=1).any()):
        raise ValueError(f'退市历史价格/量额异常: {path.name}')


def _normalize_qmt(frame, start, end):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=FIELDS)
    result = frame.loc[:, FIELDS].copy()
    # QMT time 为北京时间零点的 Unix 时间；旧 mootdx 文件存无时区 15:00。
    # 使用交易日索引统一至旧文件口径，避免同日重复及前一天错位。
    dates = pd.to_datetime(frame.index.astype(str), format='%Y%m%d')
    result['time'] = (dates + pd.Timedelta(hours=15)).as_unit('ms').astype('int64')
    result = result.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    zero = (result['open'] <= 0) & (result['close'] <= 0)
    if zero.any():
        flags = frame.loc[result.index, 'suspendFlag']
        if (flags[zero] != 1).any():
            raise ValueError('QMT 非停牌日返回零价')
        result = result.loc[~zero]
    if not result.empty:
        prices = result[['open', 'high', 'low', 'close', 'preClose']].to_numpy(float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError('QMT 返回无效价格/前收，拒绝写入')
        activity = result[['volume', 'amount']].to_numpy(float)
        if not np.isfinite(activity).all() or (activity < 0).any():
            raise ValueError('QMT 返回无效成交量/金额，拒绝写入')
    return result.sort_values('time').reset_index(drop=True)


def _qmt_update(start, end, codes=None, collect=False):
    from xtquant import xtdata
    from data.db.delist import get_delist_stock_info
    from utils.stock.time import get_trading_date_span

    if not xtdata.get_client().is_connected():
        raise RuntimeError('QMT 行情未连接')
    selected = sorted(codes) if codes is not None else mdx._all_codes()
    delisted = get_delist_stock_info()
    suspension_cache = {}
    mdx.RAW_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(code, begin):
        xtdata.download_history_data(code, period='1d', start_time=begin,
                                     end_time=end, incrementally=False)
        frames = xtdata.get_market_data_ex(
            [], [code], period='1d', start_time=begin, end_time=end,
            dividend_type='none', fill_data=False)
        detail = xtdata.get_instrument_detail(code)
        open_date = None if not detail else detail['OpenDate']
        open_missing = open_date in (None, '', '0', '19700101')
        if code in delisted:
            listed = delisted[code].list_date
            last = min(pd.Timestamp(end).date(), delisted[code].delist_date)
        elif open_missing:
            rows = _reference_rows('RPTA_APP_IPOAPPLY', f'(SECURITY_CODE="{code[:6]}")',
                                   f'ipo_{code}_{end}')
            matched = [r for r in rows if r['SECURITY_CODE'] == code[:6]]
            if len(matched) != 1:
                raise RuntimeError(f'QMT {code} 无上市日期，无法校验完整性')
            info = matched[0]
            listing = None if info['LISTING_DATE'] is None else pd.Timestamp(info['LISTING_DATE']).date()
            end_day = pd.Timestamp(end).date()
            if frames[code].empty and _ipo_missing_bars_allowed(info, end):
                result = pd.DataFrame(columns=FIELDS)
                result.attrs['not_listed'] = True
                if listing is None:
                    mdx._log(f'{code} 发行资料确认待上市，无历史日线')
                else:
                    mdx._log(f'{code} 上市日 {listing} 晚于 {end_day}，无历史日线')
                return result
            if frames[code].empty or listing is None or listing > end_day:
                raise RuntimeError(f'QMT {code} 无上市日期，无法校验完整性')
            listed = listing
            last = end_day
        else:
            listed = pd.Timestamp(open_date).date()
            last = pd.Timestamp(end).date()
        first = max(pd.Timestamp(begin).date(), listed)
        expected = pd.DatetimeIndex(get_trading_date_span(first, last)) if first <= last else pd.DatetimeIndex([])
        actual = pd.to_datetime(frames[code].index.astype(str), format='%Y%m%d')
        missing = expected.difference(actual)
        if len(missing):
            expected = expected.difference(_suspended_dates(code, missing, suspension_cache))
        _validate_qmt_dates(frames[code], expected, code)
        result = _normalize_qmt(frames[code], begin, end)
        result.attrs['suspended_only'] = bool(
            (len(missing) and not len(expected) and not len(actual))
            or (len(frames[code]) and (frames[code]['suspendFlag'] == 1).all()))
        return result

    # 必须实际取到截至目标交易日的日线，再进入全市场更新。
    for code in ('000001.SZ', '600000.SH', '300750.SZ'):
        sample = fetch(code, start)
        if sample.empty or pd.to_datetime(sample.time.iloc[-1], unit='ms').strftime('%Y%m%d') != end:
            raise RuntimeError(f'QMT 健康探测未覆盖 {end}: {code}')
    mdx._log(f'QMT 历史日线健康探测通过: {start}~{end}, 共 {len(selected)} 只')
    output = {}
    for i, code in enumerate(selected):
        path = mdx.RAW_DIR / f'{code}.parquet'
        exists = path.exists() and path.stat().st_size > 0
        if exists and code in delisted and delisted[code].delist_date <= pd.Timestamp(end).date():
            _validate_preserved_history(path)
            continue  # 已退市历史不会再更新，保留已有历史；QMT 可能已移除合约价格精度。
        begin = start if exists else mdx.START_DEFAULT
        raw = fetch(code, begin)
        if raw.empty:
            if raw.attrs.get('not_listed', False):
                continue
            if not exists or (code not in delisted and not raw.attrs.get('suspended_only', False)):
                raise RuntimeError(f'QMT 日线为空: {code}，中止更新')
        else:
            mdx._merge_recent_into(path, raw)
            if collect:
                output[code] = mdx._combined_bar_dict(raw)
        if (i + 1) % 200 == 0 or i + 1 == len(selected):
            mdx._log(f'QMT 日线进度 {i + 1}/{len(selected)}')
    return output


def update_recent(days, *, anchor_date=None, collect=False, codes=None, allow_qmt=True):
    try:
        return mdx.update_recent(days, anchor_date=anchor_date, collect=collect, codes=codes)
    except (RuntimeError, TdxFunctionCallError, TdxConnectionError) as exc:
        if not allow_qmt:
            if days == 1:
                from data.update_tencent_close import repair_session
                _, end, _ = mdx.resolve_recent_range(days, anchor_date)
                mdx._log(f'mootdx 单日更新失败，校验已收盘腾讯行情: {exc}')
                return repair_session(pd.Timestamp(end).date(), codes, collect=collect)
            raise RuntimeError('Local file planner prohibits QMT fallback; mootdx failed') from exc
        mdx._log(f'mootdx 更新失败，切换 QMT: {exc}')
    start, end, _ = mdx.resolve_recent_range(days, anchor_date)
    return _qmt_update(start, end, codes, collect)


def update_full(start=mdx.START_DEFAULT, end=None, codes=None, allow_qmt=True):
    try:
        return mdx.update_full(start=start, end=end, codes=codes)
    except (RuntimeError, TdxFunctionCallError, TdxConnectionError) as exc:
        if not allow_qmt:
            raise RuntimeError('Local file planner prohibits QMT fallback; mootdx failed') from exc
        mdx._log(f'mootdx 更新失败，切换 QMT: {exc}')
    _, target, _ = mdx.resolve_recent_range(1, pd.Timestamp(end).date() if end else None)
    return _qmt_update(start, target, codes, collect=True)
