"""Repair one completed session from dated Tencent closing quotes, never T-live bars.

This is a one-session fallback, not an historical K-line API. Wrong dates,
ambiguous volume units, missing active stocks or missing older history fail.
"""
from datetime import datetime, time as daytime
import json
import math
import os
from pathlib import Path
import re
import time
import uuid
from decimal import Decimal

import pandas as pd
import requests

from data import kline_mootdx as mdx
from data.update_live import _requested_tencent_symbols

QUOTE = re.compile(r'v_([a-z]{2}\d{6})="([^"]*)"')


def parse_close_packet(text, expected_symbols, day):
    rows = {}
    for statement in text.split(';'):
        statement = statement.strip()
        if not statement or statement == 'v_pv_none_match="1"':
            continue
        match = QUOTE.fullmatch(statement)
        if not match or match.group(1) not in expected_symbols:
            raise ValueError('Unexpected/malformed Tencent closing quote')
        symbol, packed = match.groups()
        if symbol in rows:
            raise ValueError('Duplicate Tencent closing quote: ' + symbol)
        rows[symbol] = None
        fields = packed.split('~')
        if not packed:
            continue
        if len(fields) < 38 or fields[2].zfill(6) != symbol[2:]:
            raise ValueError('Incomplete or mismatched Tencent closing quote: ' + symbol)
        if not fields[30]:
            continue
        stamp = datetime.strptime(fields[30], '%Y%m%d%H%M%S')
        if stamp.date() > day:
            raise ValueError('Tencent quote is newer than requested history; refuse lookahead')
        if stamp.date() < day:
            continue  # Only an explicit suspension/IPO exemption may cover this.
        close, previous, opened, high, low = map(float, (fields[3], fields[4], fields[5], fields[33], fields[34]))
        raw_volume = float(fields[6])
        if not re.fullmatch(r'\d+(?:\.\d+)?', fields[6]):
            raise ValueError('Unsupported closing volume precision: ' + symbol)
        if opened == 0 and raw_volume == 0:
            continue
        if stamp.time() < daytime(15):
            raise ValueError('Tencent quote has not reached market close: %s %s' % (symbol, fields[30]))
        if not all(math.isfinite(v) and v > 0 for v in (close, previous, opened, high, low, raw_volume)):
            raise ValueError('Invalid closing prices/activity: ' + symbol)
        if not low <= min(opened, close) <= max(opened, close) <= high:
            raise ValueError('Closing OHLC range is inconsistent: ' + symbol)
        change, percent = float(fields[31]), float(fields[32])
        if (not math.isfinite(change) or not math.isfinite(percent)
                or abs(close - previous - change) > .011
                or abs(100 * (close / previous - 1) - percent) > .011):
            raise ValueError('Closing reference price disagrees with reported change: ' + symbol)
        summary = fields[35].split('/')
        if len(summary) != 3 or float(summary[0]) != close or float(summary[1]) != raw_volume or float(fields[36]) != raw_volume:
            raise ValueError('Closing quote subfields disagree: ' + symbol)
        amount = float(summary[2])
        if not re.fullmatch(r'\d+(?:\.\d+)?', fields[37]):
            raise ValueError('Invalid displayed turnover: ' + symbol)
        displayed = Decimal(fields[37])
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError('Closing turnover is invalid: ' + symbol)
        quantum = Decimal(10) ** displayed.as_tuple().exponent * 10000
        difference = Decimal(summary[2]) - displayed * 10000
        if not -quantum / 2 <= difference < quantum:
            raise ValueError('Closing turnover units/precision are inconsistent: ' + symbol)
        # Tencent's stock streams use both shares and lots. Accept only the
        # unique unit whose turnover-implied VWAP lies inside reported H/L.
        tolerance = max(.01, high * 1e-6)
        units = [unit for unit in (1, 100)
                 if low - tolerance <= amount / (raw_volume * unit) <= high + tolerance]
        if len(units) != 1:
            raise ValueError('Cannot establish closing volume unit: ' + symbol)
        shares = raw_volume * units[0]
        if abs(shares - round(shares)) > .000001:
            raise ValueError('Reported volume is not an integer number of shares: ' + symbol)
        rows[symbol] = dict(time=int((pd.Timestamp(day) + pd.Timedelta(hours=15)).value // 1000000),
            open=opened, high=high, low=low, close=close, preClose=previous,
            volume=shares / 100, amount=amount,
            _volume_tolerance_lots=(.000001 if units[0] == 1 else
                float(Decimal(10) ** Decimal(fields[6]).as_tuple().exponent) / 2 + .000001))
    return rows


def fetch_closed_quotes(codes, day):
    now = datetime.now()
    if day > now.date() or (day == now.date() and now.time() < daytime(15, 35)):
        raise ValueError('Closing-quote repair is only allowed after 15:35')
    symbols, symbol_to_code = _requested_tencent_symbols(codes)
    path = mdx.DATA_DIR / 'market_status' / ('tencent_close_' + day.isoformat() + '.json')
    packets = []
    if path.exists():
        cache = json.loads(path.read_text(encoding='utf-8'))
        if cache['date'] == day.isoformat() and set(symbols) <= set(cache['symbols']):
            packets = cache['packets']
    if not packets:
        with requests.Session() as session:
            for offset in range(0, len(symbols), 250):
                chunk = symbols[offset:offset + 250]
                response = session.get('https://qt.gtimg.cn/q=' + ','.join(chunk), timeout=15)
                response.raise_for_status()
                text = response.content.decode('gbk')
                parsed = parse_close_packet(text, set(chunk), day)
                if not any(value is not None for value in parsed.values()):
                    raise RuntimeError('Closing quote source returned no valid completed-session bars')
                packets.append(dict(symbols=chunk, text=text))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(date=day.isoformat(), symbols=symbols,
            captured_at=now.isoformat(), packets=packets), ensure_ascii=False), encoding='utf-8')
        temporary.replace(path)
    result = {}
    for packet in packets:
        for symbol, row in parse_close_packet(packet['text'], set(packet['symbols']), day).items():
            if symbol in result:
                raise ValueError('Duplicate symbol across closing packets')
            result[symbol] = row
    return {symbol_to_code[symbol]: result.get(symbol) for symbol in symbols}


def repair_session(day, codes=None, *, collect=False, validate_only=False):
    from data.db.delist import get_delist_stock_info
    from utils.stock.time import get_trading_date_span, _get_trading_calendar_state
    from data.update_kline import FIELDS, _ipo_missing_bars_allowed, _reference_rows, _suspended_dates, _validate_preserved_history

    trading_days, _, _ = _get_trading_calendar_state()
    if day not in trading_days:
        raise ValueError('Closing repair date is not a verified trading day')
    selected = sorted(codes) if codes is not None else mdx._all_codes()
    delisted = get_delist_stock_info()
    active, preserved = [], []
    for code in selected:
        path = mdx.RAW_DIR / (code + '.parquet')
        if code in delisted and delisted[code].delist_date <= day:
            if not path.exists():
                raise RuntimeError('Closing quotes cannot reconstruct delisted history: ' + code)
            _validate_preserved_history(path)
            preserved.append(code)
        else:
            active.append(code)
    quotes = fetch_closed_quotes(active, day)
    pending, exemptions, suspension_cache, backfilled = {}, {}, {}, {}
    calendar = get_trading_date_span(day - pd.Timedelta(days=120).to_pytimedelta(), day)
    checked_days = pd.DatetimeIndex(calendar[-60:])
    for checked_day in checked_days:
        stored = mdx.DATA_DIR / 'market_status' / ('suspensions_' + checked_day.strftime('%Y-%m-%d') + '.json')
        if stored.exists():
            suspension_cache[checked_day.strftime('%Y-%m-%d')] = json.loads(stored.read_text(encoding='utf-8'))
    for code in active:
        path = mdx.RAW_DIR / (code + '.parquet')
        existing = pd.read_parquet(path, columns=list(FIELDS)) if path.exists() else None
        row = quotes[code]
        if existing is None:
            name = 'ipo_' + code + '_' + day.strftime('%Y%m%d')
            reference = mdx.DATA_DIR / 'market_status' / (name + '.json')
            if reference.exists():
                records = json.loads(reference.read_text(encoding='utf-8'))
            else:
                time.sleep(1.05)
                records = _reference_rows('RPTA_APP_IPOAPPLY', '(SECURITY_CODE="' + code[:6] + '")', name)
            matches = [r for r in records if r['SECURITY_CODE'] == code[:6]]
            if len(matches) != 1:
                raise RuntimeError('No unambiguous listing evidence: ' + code)
            info = matches[0]
            if _ipo_missing_bars_allowed(info, day):
                if row is not None:
                    raise RuntimeError('Listing evidence conflicts with actual traded quote: ' + code)
                exemptions[code] = 'not_listed_by_target_date'
                continue
            listed = pd.Timestamp(info['LISTING_DATE']).date()
            if listed < day:
                from data.update_sina_history import fetch_unadjusted_history
                quote = mdx._connect_mootdx(require_daily=False)
                try:
                    history = fetch_unadjusted_history(code, listed, day, mdx._fetch_xdxr(quote, code))
                finally:
                    mdx._close_mootdx(quote)
                actual = pd.DatetimeIndex(pd.to_datetime(history['time'], unit='ms').dt.normalize())
                expected_history = pd.DatetimeIndex(get_trading_date_span(listed, day))
                if len(actual.difference(expected_history)):
                    raise RuntimeError('Sina returned a non-trading-day history row: ' + code)
                missing = expected_history.difference(actual)
                if len(missing):
                    missing = missing.difference(_suspended_dates(code, missing, suspension_cache))
                if len(missing):
                    raise RuntimeError('New stock history still has missing sessions: ' + code)
                if row is not None:
                    last = history.iloc[-1]
                    if (pd.Timestamp(last['time'], unit='ms').date() != day
                            or any(abs(float(last[key]) - row[key]) > .011 for key in ('open', 'high', 'low', 'close', 'preClose'))
                            or abs(last['volume'] - row['volume']) > row['_volume_tolerance_lots']
                            or abs(last['amount'] - row['amount']) > 1):
                        raise RuntimeError('Independent closing sources disagree: ' + code)
                pending[code] = (history, history)
                backfilled[code] = dict(source='Sina unadjusted history plus mootdx corporate actions',
                    first=pd.Timestamp(history.iloc[0]['time'], unit='ms').date().isoformat(),
                    last=pd.Timestamp(history.iloc[-1]['time'], unit='ms').date().isoformat(), bars=len(history))
                continue
        if existing is not None:
            dates = pd.DatetimeIndex(pd.to_datetime(existing['time'], unit='ms').dt.normalize())
            expected = checked_days[checked_days < pd.Timestamp(day)]
            if dates.min() > expected.min():
                name = 'ipo_' + code + '_' + day.strftime('%Y%m%d')
                reference = mdx.DATA_DIR / 'market_status' / (name + '.json')
                if reference.exists():
                    records = json.loads(reference.read_text(encoding='utf-8'))
                else:
                    time.sleep(1.05)
                    records = _reference_rows('RPTA_APP_IPOAPPLY', '(SECURITY_CODE="' + code[:6] + '")', name)
                matched = [r for r in records if r['SECURITY_CODE'] == code[:6]]
                if len(matched) != 1 or not matched[0]['LISTING_DATE']:
                    raise RuntimeError('Recent history lacks verified listing date: ' + code)
                expected = expected[expected >= pd.Timestamp(matched[0]['LISTING_DATE']).normalize()]
            missing = expected.difference(dates)
            if len(missing):
                missing = missing.difference(_suspended_dates(code, missing, suspension_cache))
            if len(missing):
                raise RuntimeError('Older sessions are missing; one-day repair is insufficient: ' + code)
            if existing.time.duplicated().any() or not existing.time.is_monotonic_increasing:
                raise RuntimeError('Existing history is malformed: ' + code)
        if row is None:
            missing = pd.DatetimeIndex([pd.Timestamp(day)])
            if not _suspended_dates(code, missing, suspension_cache).equals(missing):
                raise RuntimeError('Active stock lacks a valid closing quote: ' + code)
            exemptions[code] = 'verified_full_session_suspension'
            continue
        new = pd.DataFrame([row], columns=list(FIELDS))
        merged = (pd.concat([existing, new], ignore_index=True).drop_duplicates('time', keep='last')
                  if existing is not None else new)
        pending[code] = (merged.sort_values('time').reset_index(drop=True), new)
    audit = dict(date=day.isoformat(), source='Tencent completed-session quote',
                 rows=len(pending), preserved_delisted=len(preserved), exemptions=exemptions,
                 backfilled_histories=backfilled,
                 checked_history_start=checked_days[0].date().isoformat(),
                 volume_unit='lots; uniquely validated against turnover and daily price range',
                 volume_precision='Tencent retains reported quote precision (often whole lots); Sina integer shares / 100',
                 validated_at=datetime.now().isoformat(), written=False)
    audit_path = mdx.DATA_DIR / 'market_status' / ('tencent_close_validation_' + day.isoformat() + '.json')
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    if validate_only:
        return audit
    output = {}
    for code, (merged, new) in pending.items():
        path = mdx.RAW_DIR / (code + '.parquet')
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            merged.to_parquet(temporary, index=False)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        if collect:
            output[code] = mdx._combined_bar_dict(new)
    audit['written'] = True
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    mdx._log('Tencent completed-session repair: %d bars, %d explicit exemptions' % (len(pending), len(exemptions)))
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    print(repair_session(datetime.strptime(args.date, '%Y-%m-%d').date(), validate_only=args.validate_only))
