"""Read Sina's raw, unadjusted daily payload, retaining its pre-close field."""
import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from py_mini_racer import MiniRacer
from akshare.stock.stock_zh_a_sina import hk_js_decode, zh_sina_a_stock_hist_url

DATA_DIR = Path(__file__).resolve().parent


def fetch_raw_history(code):
    symbol = ('sh' if code.endswith('.SH') else 'sz') + code[:6]
    response = requests.get(zh_sina_a_stock_hist_url.format(symbol), timeout=15,
                            headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn/'})
    response.raise_for_status()
    match = re.search(r'=\s*"([^"\r\n]+)"\s*;', response.text)
    if not match:
        raise ValueError('Sina daily payload is malformed')
    js = MiniRacer()
    js.eval(hk_js_decode, timeout=5000)
    rows = js.call('d', match.group(1), timeout=5000)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError('Sina daily payload is empty')
    return pd.DataFrame(rows)


def fetch_unadjusted_history(code, listed, end, xdxr):
    from data.kline_mootdx import _compute_preclose
    frame = fetch_raw_history(code)
    provenance = DATA_DIR / 'market_status' / ('sina_history_' + code + '_' + end.isoformat() + '.json')
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(dict(source='Sina raw unadjusted daily payload',
        captured_at=datetime.now().isoformat(), listed=listed.isoformat(), end=end.isoformat(),
        bars=json.loads(frame.to_json(orient='records', date_format='iso')),
        corporate_actions=json.loads(xdxr.to_json(orient='records'))), ensure_ascii=False), encoding='utf-8')
    dates = pd.to_datetime(frame['date'], utc=True).dt.tz_localize(None).dt.normalize()
    selected = (dates.dt.date >= listed) & (dates.dt.date <= end)
    frame, dates = frame.loc[selected].copy(), dates.loc[selected]
    if frame.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Sina history is empty, duplicated or unordered: ' + code)
    result = frame[['open', 'high', 'low', 'close', 'volume', 'amount']].astype(float)
    prices = result[['open', 'high', 'low', 'close']].to_numpy()
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Sina unadjusted prices invalid: ' + code)
    if not ((result.low <= result[['open', 'close']].min(axis=1)) &
            (result.high >= result[['open', 'close']].max(axis=1))).all():
        raise ValueError('Sina OHLC range invalid: ' + code)
    if not np.isfinite(result[['volume', 'amount']].to_numpy()).all() or (result[['volume', 'amount']] < 0).any().any():
        raise ValueError('Sina activity invalid: ' + code)
    if (np.abs(result['volume'] - np.round(result['volume'])) > .000001).any():
        raise ValueError('Sina raw volume is not integer shares: ' + code)
    traded = result['volume'] > 0
    average = result.loc[traded, 'amount'] / result.loc[traded, 'volume']
    if ((result.loc[~traded, 'amount'] != 0).any()
            or (average < result.loc[traded, 'low'] - .01).any()
            or (average > result.loc[traded, 'high'] + .01).any()):
        raise ValueError('Sina turnover/volume units disagree with daily range: ' + code)
    result['time'] = (dates + pd.Timedelta(hours=15)).dt.as_unit('ms').astype('int64')
    result['volume'] /= 100  # Sina raw payload explicitly reports shares.
    result['preClose'] = _compute_preclose(result['close'].to_numpy(), result['time'].to_numpy(), xdxr)
    explicit = pd.to_numeric(frame['prevclose'], errors='raise')
    if np.isinf(explicit).any() or ((explicit.notna()) & (explicit <= 0)).any():
        raise ValueError('Sina explicit pre-close is invalid: ' + code)
    if len(result) and not (pd.notna(explicit.iloc[0]) and explicit.iloc[0] > 0):
        result.iloc[0, result.columns.get_loc('preClose')] = np.nan
    for index in result.index[explicit.notna() & (explicit > 0)]:
        if index != result.index[0] and np.isfinite(result.loc[index, 'preClose']):
            if abs(result.loc[index, 'preClose'] - explicit.loc[index]) > .011:
                raise ValueError('Sina and corporate-action reference disagree: ' + code)
        result.loc[index, 'preClose'] = explicit.loc[index]
    return result[['time', 'open', 'high', 'low', 'close', 'volume', 'amount', 'preClose']].reset_index(drop=True)


if __name__ == '__main__':
    frame = fetch_raw_history('001246.SZ')
    print(frame.columns.tolist())
    print(frame.head(2).to_json(orient='records', date_format='iso'))
    print(frame.tail(3).to_json(orient='records', date_format='iso'))
    from data import kline_mootdx as mdx
    quote = mdx._connect_mootdx(require_daily=False)
    try:
        print(mdx._fetch_xdxr(quote, '001246.SZ').to_json(orient='records'))
    finally:
        mdx._close_mootdx(quote)
