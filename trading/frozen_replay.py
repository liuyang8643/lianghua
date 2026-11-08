"""Replay an immutable target through the canonical research account engine."""
from datetime import timedelta
import json
from pathlib import Path
import pandas as pd

from env.contracts import AccountState,OrderPlan
from env.fees import FeeSchedule
from env.simulator import DaySimulator
from trading.target_protocol import targets_to_pending
from utils.stock.time import get_last_trading_day


def replay_frozen_targets(directory, trade_date, kline_directory):
    directory=Path(directory)
    journal=json.loads((directory/'journal'/('execution_'+trade_date.isoformat()+'.json')).read_text())
    document=journal['target_document']
    previous=get_last_trading_day(trade_date-timedelta(days=1))
    baseline=json.loads((directory/('account_eod_'+previous.isoformat()+'.json')).read_text())
    if baseline['account']!=journal['account'] or document['trade_date']!=trade_date.isoformat():
        raise ValueError('Frozen replay account/date differs')
    initial=journal['initial_account']
    quantities={p['stock_code']:int(p['volume']) for p in initial['positions'] if p['volume']>0}
    pending=targets_to_pending(document,quantities)
    codes=set(quantities)|set(document['targets'])
    closes={}
    closing=json.loads((directory/('account_eod_'+trade_date.isoformat()+'.json')).read_text())
    if closing['account']!=journal['account'] or closing['trade_date']!=trade_date.isoformat():
        raise ValueError('Closing account/date differs')
    broker_marks={r['stock_code']:float(r['market_value'])/int(r['volume'])
                  for r in closing['positions'] if r['volume']>0}
    for code in codes:
        if code in document['unmanaged_codes']:
            closes[code]=broker_marks[code]
            continue
        frame=pd.read_parquet(Path(kline_directory)/(code+'.parquet'))
        times=frame['time']
        times=pd.to_datetime(times,unit='ms') if pd.api.types.is_numeric_dtype(times) else pd.to_datetime(times)
        rows=frame.loc[times.dt.date==trade_date]
        if len(rows)!=1 or not float(rows.iloc[0]['close'])>0:
            raise ValueError('Unverifiable frozen target closing valuation: '+code)
        closes[code]=float(rows.iloc[0]['close'])
    last={p['stock_code']:float(p['market_value'])/int(p['volume'])
          for p in baseline['positions'] if p['volume']>0}
    opening={c:float(r['open_price']) for c,r in document['targets'].items()}
    account=AccountState(cash=float(initial['asset']['cash']),positions=quantities,
        sellable_positions={p['stock_code']:int(p['can_use_volume']) for p in initial['positions']},last_prices=last)
    plan=OrderPlan(decision_date=trade_date.isoformat(),sell_orders=tuple(pending['sell_orders']),
                   buy_orders=pending['buy_allocations'])
    zero=FeeSchedule(commission_rate=0,minimum_commission=0,stamp_tax_rate=0,
                     transfer_fee_rate=0,slippage_rate=0)
    result=DaySimulator(zero).step(account,plan,opening,closes,
                                   next_decision_date=trade_date.isoformat(),terminated=True)
    def position_rows(volumes,prices):
        return [dict(code=c,volume=q,current_price=prices[c],current_value=q*prices[c])
                for c,q in volumes.items() if q>0]
    old={p['stock_code']:int(p['volume']) for p in baseline['positions'] if p['volume']>0}
    y_nav=float(baseline['asset']['cash'])+sum(q*last[c] for c,q in old.items())
    end=result.account_state
    return dict(daily_snapshots=[
        dict(date=previous.isoformat(),signal_date=previous.isoformat(),total_asset=y_nav,
             cash=baseline['asset']['cash'],positions_eod=position_rows(old,last)),
        dict(date=trade_date.isoformat(),signal_date=trade_date.isoformat(),total_asset=end.nav,
             cash=end.cash,positions_eod=position_rows(end.positions,closes),
             daily_return_pct=(end.nav/y_nav-1)*100)],
        trade_log=[dict(code=f.code,action=f.side,volume=f.quantity,price=f.price,
                        amount=f.quantity*f.price,commission=f.fee,signal_date=trade_date.isoformat())
                   for f in result.fills],
        comparison_basis='frozen target / canonical DaySimulator / zero friction')
