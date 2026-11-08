"""Account-bound acquisition dates; never infer age from broker cost prices."""
from datetime import date,datetime,time
from pathlib import Path
import json
import hashlib


def rebuild_history(directory, account, trade_date, calendar, *, unmanaged_codes=()):
    """Replay verified daily fills; first observed holdings get an explicit lower bound."""
    from trading.target_protocol import broker_fact_date
    from trading.bridge_wire import atomic_json
    from trading.account_snapshot import validate_account_snapshot
    from types import SimpleNamespace
    directory=Path(directory)
    files={}
    for path in directory.glob('account_eod_*.json'):
        day=date.fromisoformat(path.stem.removeprefix('account_eod_'))
        if day in calendar and day<trade_date:
            files[day]=path
    if not files:
        raise ValueError('No account-bound position history')
    expected=[d for d in calendar if min(files)<=d<trade_date]
    if set(expected)!=set(files):
        raise ValueError('Daily account evidence has gaps; holding ages are unverifiable')
    volumes={};state={};evidence=[]
    for index,day in enumerate(sorted(files)):
        path=files[day];raw=path.read_bytes();snapshot=json.loads(raw)
        if snapshot['account']!=account or snapshot['trade_date']!=day.isoformat():
            raise ValueError('Position history account/date mismatch')
        captured=datetime.fromtimestamp(snapshot['captured_at'])
        if captured.date()!=day or captured.time()<time(15,35):
            raise ValueError('Position history is not a confirmed same-day EOD snapshot')
        validate_account_snapshot(SimpleNamespace(**snapshot['asset']),
                                  [SimpleNamespace(**p) for p in snapshot['positions']])
        actual={r['stock_code']:int(r['volume']) for r in snapshot['positions']
                if r['volume']>0 and r['stock_code'] not in unmanaged_codes}
        evidence.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
        if index==0:
            volumes=actual.copy()
            state={c:dict(known_since=day.isoformat(),age_is_lower_bound=True) for c in actual}
            continue
        if not snapshot['trades_available'] or snapshot['trades'] is None:
            raise ValueError('Daily fill evidence missing while reconstructing holding ages')
        unique={}
        for fill in snapshot['trades']:
            if fill['stock_code'] in unmanaged_codes:
                continue
            if broker_fact_date(fill['traded_time'])!=day:
                continue
            key=(fill['stock_code'],fill['traded_id'])
            if not fill['traded_id']:
                raise ValueError('Fill identity missing')
            if key in unique and unique[key]!=fill:
                raise ValueError('Conflicting duplicate fill')
            unique[key]=fill
        for fill in sorted(unique.values(),key=lambda f:(f['traded_time'],f['order_id'],str(f['traded_id']))):
            code=fill['stock_code'];quantity=int(fill['traded_volume'])
            if fill['order_type']==23:
                if volumes.get(code,0)==0:
                    state[code]=dict(known_since=day.isoformat(),age_is_lower_bound=False)
                volumes[code]=volumes.get(code,0)+quantity
            elif fill['order_type']==24:
                volumes[code]=volumes.get(code,0)-quantity
                if volumes[code]<0:
                    raise ValueError('Fill history would oversell '+code)
                if volumes[code]==0:
                    del volumes[code];state.pop(code,None)
            else:
                raise ValueError('Unclassified account movement in holding history')
        if volumes!=actual:
            raise ValueError('Position changes require corporate-action/transfer reconciliation on '+day.isoformat())
    result=dict(account=account,positions={c:dict(**state[c],volume=q,evidence=evidence) for c,q in volumes.items()})
    atomic_json(directory/'position_history.json',result)
    return result


def holding_days(directory, account, trade_date, positions, calendar, *, unmanaged_codes=()):
    if not positions:
        return {}
    history = rebuild_history(directory,account,trade_date,calendar,unmanaged_codes=unmanaged_codes)
    if history['account'] != account:
        raise ValueError('Position history belongs to another account')
    days = {}
    for code,position in positions.items():
        row = history['positions'][code]
        if row['volume'] != int(position.volume) or not row['evidence']:
            raise ValueError('Position history must reconcile to confirmed holdings: '+code)
        acquired = date.fromisoformat(row['known_since'])
        if acquired > trade_date:
            raise ValueError('Acquisition date is in the future')
        days[code] = sum(acquired < day <= trade_date for day in calendar)
    return days
