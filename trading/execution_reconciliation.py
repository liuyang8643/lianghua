"""离线执行归因：固定实际下单计划，以同一收盘估值比较零磨损成交。

收益差 = -滑点 - 手续费 + 成交数量差影响 + 未解释残差。
资金变化不参与该恒等式；持仓转入转出、缺成交等不能冒充交易成功。
"""
from math import isfinite

import pandas as pd


def load_execution_reference(trade_date, directory, kline_directory):
    """只读历史文件；执行日志优先于可能被重启覆盖的 plan。"""
    import json
    from datetime import timedelta
    from utils.stock.time import get_last_trading_day

    previous_date = get_last_trading_day(trade_date - timedelta(days=1))
    previous_path = directory / f'positions_{previous_date.isoformat()}.parquet'
    previous = pd.read_parquet(previous_path) if previous_path.exists() else None
    plan_path = directory / f'plan_{trade_date.isoformat()}.parquet'
    targets = pd.read_parquet(plan_path) if plan_path.exists() else None
    journal_path = directory / f'execution_{trade_date.isoformat()}.json'
    if journal_path.exists():
        journal = json.loads(journal_path.read_text(encoding='utf-8'))
        pending = journal['pending']
        if journal['version'] != 1 or pending['trade_date'] != trade_date.isoformat():
            raise ValueError('执行日志日期或版本不一致')
        rows = []
        for direction, orders in [('buy', pending['buy_allocations'].items()),
                                  ('sell', pending['sell_orders'])]:
            for code, shares in orders:
                rows.append({'code': code, 'direction': direction,
                             'est_volume': shares, 'est_price': pending['prices'][code]})
        targets = pd.DataFrame(rows, columns=['code', 'direction', 'est_volume', 'est_price'])
    current_path = directory / f'positions_{trade_date.isoformat()}.parquet'
    current = pd.read_parquet(current_path) if current_path.exists() else None
    fills_path = directory / f'fills_{trade_date.isoformat()}.parquet'
    fills = pd.read_parquet(fills_path) if fills_path.exists() else None
    codes = set()
    for frame in (previous, current, targets, fills):
        if frame is not None and not frame.empty:
            codes.update(frame['code'])
    closes = {}
    for code in codes:
        path = kline_directory / f'{code}.parquet'
        if path.exists():
            frame = pd.read_parquet(path)
            times = frame['time']
            dates = pd.to_datetime(times, unit='ms') if pd.api.types.is_numeric_dtype(times) else pd.to_datetime(times)
            rows = frame.loc[dates.dt.date == trade_date]
            if not rows.empty:
                closes[code] = float(rows.iloc[-1]['close'])
        # 券商持仓估值支持停牌（无当日日线）；只用于确实持有的股票。
        if code not in closes and current is not None:
            rows = current.loc[(current['code'] == code) & (current['volume'] > 0)]
            if not rows.empty:
                closes[code] = float(rows.iloc[-1]['market_value']) / int(rows.iloc[-1]['volume'])
    return previous, targets, closes


def reconcile_execution(previous, current, fills, targets, closes):
    """DataFrame 输入；None 表示缺证据，空表表示已确认无持仓/成交/目标。"""
    if previous is None or current is None or fills is None or targets is None:
        return {'available': False, 'reason': '缺少昨日/今日持仓、成交或原始执行计划'}
    old = {r['code']: r for r in previous.to_dict('records')}
    now = {r['code']: r for r in current.to_dict('records')}
    planned, actual = {}, {}
    for r in targets.to_dict('records'):
        qty = int(r['est_volume'])
        if qty == 0:
            continue
        price = float(r['est_price'])
        if qty < 0 or not isfinite(price) or price <= 0:
            return {'available': False, 'reason': '原始计划数量或价格无效'}
        key = (r['code'], r['direction'])
        q, amount = planned.get(key, (0, 0.0))
        planned[key] = (q + qty, amount + qty * price)
    for r in fills.to_dict('records'):
        key = (r['code'], r['direction'])
        q, amount, fee = actual.get(key, (0, 0.0, 0.0))
        values = (float(r['amount']), float(r['fee_est']))
        if not all(isfinite(v) for v in values):
            return {'available': False, 'reason': '成交金额或费用缺失'}
        actual[key] = (q + int(r['shares']), amount + values[0], fee + values[1])

    rows = []
    reference_prices = {}
    codes = set(old) | set(now) | {k[0] for k in planned} | {k[0] for k in actual}
    for code in sorted(codes):
        y, t = old.get(code), now.get(code)
        yq, tq = (int(y['volume']) if y else 0), (int(t['volume']) if t else 0)
        ymv = float(y['market_value']) if y else 0.0
        tmv = float(t['market_value']) if t else 0.0
        if yq == tq == 0 and ymv == tmv == 0 and not any(
                (code, direction) in planned or (code, direction) in actual
                for direction in ('buy', 'sell')):
            continue
        close = closes.get(code)
        if close is None or not isfinite(close) or close <= 0:
            return {'available': False, 'reason': f'{code} 缺少当日收盘估值'}
        ideal_q = expected_q = yq
        ideal_cash = actual_cash = slip = fee = quantity_pnl = 0.0
        gaps = {}
        for direction, sign in [('buy', 1), ('sell', -1)]:
            pq, pa = planned.get((code, direction), (0, 0.0))
            aq, aa, af = actual.get((code, direction), (0, 0.0, 0.0))
            # 计划外成交归入数量差；以该股计划价或当日开盘参考价计滑点。
            reference = pa / pq if pq else None
            if aq and reference is None:
                candidates = targets[targets['code'] == code]
                prices = candidates.loc[candidates['est_price'] > 0, 'est_price']
                reference = float(prices.iloc[0]) if len(prices) else None
                if reference is None:
                    if 'est_price' not in fills.columns:
                        return {'available': False, 'reason': f'{code} 计划外成交缺少开盘参考价'}
                    prices = fills.loc[(fills['code'] == code) & (fills['direction'] == direction), 'est_price']
                    prices = prices[prices.notna() & (prices > 0)]
                    reference = float(prices.iloc[0]) if len(prices) else None
                if reference is None:
                    return {'available': False, 'reason': f'{code} 计划外成交缺少开盘参考价'}
            reference = reference or 0.0
            if aq:
                reference_prices.setdefault(code, {})[direction] = reference
            ideal_q += sign * pq
            expected_q += sign * aq
            ideal_cash -= sign * pa
            actual_cash -= sign * aa
            slip += sign * (aa - aq * reference)
            fee += af
            quantity_pnl += sign * (aq - pq) * (close - reference)
            gaps[direction] = pq - aq
        if ideal_q < 0:
            return {'available': False, 'reason': f'{code} 计划卖出超过昨日持仓，需核查仓位变动'}
        if tq != expected_q:
            return {'available': False, 'within_tolerance': False,
                    'reason': f'{code} 持仓变动缺少成交或转入转出证据，盈亏不可验证',
                    'position_gap': tq - expected_q, 'rows': []}
        live = tmv - ymv + actual_cash - fee
        ideal = ideal_q * close - ymv + ideal_cash
        diff = live - ideal
        residual = diff + slip + fee - quantity_pnl
        rows.append({
            'code': code, 'live_daily_pnl': live, 'bt_daily_pnl': ideal,
            'pnl_diff': diff, 'live_volume': tq, 'bt_volume': ideal_q,
            'live_mv': tmv, 'bt_mv': ideal_q * close,
            'slippage_cost': slip, 'fee_cost': fee, 'quantity_pnl': quantity_pnl,
            'residual': residual, 'position_gap': tq - expected_q,
            'buy_shares_gap': gaps['buy'], 'sell_shares_gap': gaps['sell'],
            'live_daily_return_pct': None, 'bt_daily_return_pct': None,
            'ret_diff': None, '_l_eff': live, '_b_eff': ideal,
        })
    totals = {key: sum(r[key] for r in rows) for key in (
        'slippage_cost', 'fee_cost', 'quantity_pnl', 'residual')}
    return {
        'available': True, 'rows': rows, **totals,
        'live_total_pnl': sum(r['live_daily_pnl'] for r in rows),
        'bt_total_pnl': sum(r['bt_daily_pnl'] for r in rows),
        'all_filled': all(r['buy_shares_gap'] == r['sell_shares_gap'] == 0 for r in rows),
        'within_tolerance': all(abs(r['residual']) <= 0.02 and r['position_gap'] == 0 for r in rows),
        'fee_source': 'estimated',
        'reference_prices': reference_prices,
    }
