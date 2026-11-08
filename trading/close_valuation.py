"""Date-bound EOD valuation for reports; preserves raw broker snapshots."""
import copy
import hashlib
import math
from io import BytesIO
from pathlib import Path

import pandas as pd


def value_at_close(snapshot, kline_directory):
    day = snapshot['trade_date']
    valued = copy.deepcopy(snapshot)
    evidence, unpriced = [], []
    for position in valued['positions']:
        quantity = int(position['volume'])
        if quantity <= 0:
            continue
        code = position['stock_code']
        path = Path(kline_directory) / (code + '.parquet')
        price, source_hash = None, None
        if path.exists():
            raw = path.read_bytes()
            source_hash = hashlib.sha256(raw).hexdigest()
            frame = pd.read_parquet(BytesIO(raw))
            times = frame['time']
            times = pd.to_datetime(times, unit='ms') if pd.api.types.is_numeric_dtype(times) else pd.to_datetime(times)
            rows = frame.loc[times.dt.strftime('%Y-%m-%d') == day]
            if len(rows) != 1:
                raise ValueError('Missing or duplicate closing valuation: ' + code + ' ' + day)
            price = float(rows.iloc[0]['close'])
            if not math.isfinite(price) or price <= 0:
                raise ValueError('Invalid closing price: ' + code)
        original = float(position['market_value'])
        if price is None:
            # A single unquoted holding can be cross-checked against the
            # independent account total after every quoted holding is valued.
            # Its product NAV is not independently verified.
            unpriced.append(code)
            price = original / quantity
        market_value = price * quantity
        position.update(last_price=price, market_value=market_value)
        evidence.append(dict(code=code, shares=quantity, valuation_date=day,
            raw_market_value=original, market_value=market_value, close=price,
            source='dated_kline_close' if source_hash else 'broker_value_account_total_crosscheck',
            independent_price_verified=source_hash is not None,
            path=str(path.resolve()) if source_hash else None, sha256=source_hash))
    if len(unpriced) > 1:
        raise ValueError('Multiple holdings lack independent closing prices: ' + ', '.join(unpriced))
    asset = valued['asset']
    total_mv = sum(p['market_value'] for p in valued['positions'] if p['volume'] > 0)
    gap = float(asset['total_asset']) - float(asset['cash']) - float(asset.get('frozen_cash', 0)) - total_mv
    if not math.isfinite(gap) or abs(gap) > 1.0:
        raise ValueError('Closing valuations do not reconcile with broker total assets: %.2f' % gap)
    proof = dict(version=1, account=snapshot['account'], trade_date=day,
        captured_at=snapshot['captured_at'], rows=evidence, unpriced_holdings=unpriced,
        raw_market_value=snapshot['asset']['market_value'], market_value=total_mv,
        raw_asset_component_gap=float(asset['total_asset']) - float(asset['cash'])
            - float(asset.get('frozen_cash', 0)) - float(snapshot['asset']['market_value']),
        asset_component_gap=gap)
    asset['market_value'] = total_mv
    return valued, proof
