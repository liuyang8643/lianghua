import sys, json, zipfile, hashlib, socket, subprocess, gc
from pathlib import Path
from datetime import datetime, date, timedelta
from types import SimpleNamespace as NS
import numpy as np
import pandas as pd
import psutil
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
START=date(2026,8,3); END=date(2026,11,3)
def denied(*a,**k): raise RuntimeError('Offline audit: network disabled')
socket.socket.connect=denied; socket.create_connection=denied
from loguru import logger
logger.remove(); logger.add(sys.stderr,level='WARNING')
from core import runtime, backtest
from core.strategy_config import load_strategy_config
from core.strategy import build_rebalance_day
from core.prefilter import select_prefilter_candidates
from core.legality import LegalityChecker
from trading.plan_publisher import document_from_day_plan
from utils.stock.time import get_trading_date_span,get_last_trading_day
from data.db.stock_list import allow_buy_stock_code_list
cfg=load_strategy_config(ROOT/'configs/config.json'); c=cfg['individual_config']
source=runtime.latest_runtime_npz_path(); loads=[]
def bounded(dates,max_lookback=None):
    with zipfile.ZipFile(source) as z:
        td=np.load(z.open('trade_dates.npy'),allow_pickle=False)
        si=max(0,int(np.searchsorted(td,np.datetime64(min(d.date() for d in dates))))-max(65,(max_lookback or 60)+5))
        ei=int(np.searchsorted(td,np.datetime64(max(d.date() for d in dates)),side='right'))
        result={'trade_dates':td[si:ei].copy()}
        for f in ['stock_codes']+runtime._1D_FIELDS+runtime._2D_FIELDS:
            if f+'.npy' not in z.namelist(): continue
            with z.open(f+'.npy') as stream:
                version=np.lib.format.read_magic(stream)
                assert version in ((1,0),(2,0)); shape,fortran,dtype=(np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0)(stream)
                assert not dtype.hasobject and not fortran
                if len(shape)==2:
                    rowbytes=shape[1]*dtype.itemsize
                    stream.seek(si*rowbytes,1)
                    raw=stream.read((ei-si)*rowbytes)
                    result[f]=np.frombuffer(raw,dtype=dtype).reshape(ei-si,shape[1]).copy()
                else:
                    result[f]=np.frombuffer(stream.read(),dtype=dtype).reshape(shape).copy()
        loads.append({'rows':ei-si,'start':str(td[si]),'end':str(td[ei-1]),'bytes':sum(x.nbytes for x in result.values())})
        return result
runtime.load_runtime_npz=bounded; backtest.load_runtime_npz=bounded
calendar=get_trading_date_span(START,END)
dts=[datetime.combine(d,datetime.min.time()) for d in calendar]
data=bounded(dts,60)
pool=[str(s) for s in data['stock_codes'] if str(s).startswith(('00','30','60','68'))]
result=backtest._compute_factor_scores(dts,pool,c['weights'],cfg['factor_classes'],data=data,enable_nan_filter=c['filter'])
data,scores,masks,dates,indices,codes,columns=result
print('loaded',loads,flush=True)
summary=pd.read_parquet(ROOT/'data/live_trades/daily_summary.parquet'); summary['day']=pd.to_datetime(summary['date']).dt.date
files=ROOT/'data/live_trades'
def posmap(frame): return {r['code']:int(r['volume']) for r in frame.to_dict('records') if r['volume']>0}
def diffs(a,b): return {k:{'expected':a.get(k,0),'actual':b.get(k,0)} for k in sorted(set(a)|set(b)) if a.get(k,0)!=b.get(k,0)}
ledger=[]; plans=[]; missing=[]; serialization=[]
for dt,idx in zip(dates,indices):
    day=dt.date(); prev=get_last_trading_day(day-timedelta(days=1))
    pp=files/f'positions_{prev}.parquet'; cp=files/f'positions_{day}.parquet'; fp=files/f'fills_{day}.parquet'; ap=files/f'plan_{day}.parquet'
    if not pp.exists() or not cp.exists():
        missing.append({'date':str(day),'prior_positions':pp.exists(),'current_positions':cp.exists()}); continue
    pf=pd.read_parquet(pp); current=pd.read_parquet(cp); before=posmap(pf); after=posmap(current)
    if fp.exists():
        fills=pd.read_parquet(fp); expected=dict(before)
        for r in fills.to_dict('records'):
            assert r['direction'] in ('buy','sell')
            expected[r['code']]=expected.get(r['code'],0)+int(r['shares'])*(1 if r['direction']=='buy' else -1)
        ledger.append({'date':str(day),'fill_rows':len(fills),'duplicate_trade_ids':int(fills['traded_id'].duplicated().sum()) if 'traded_id' in fills else None,'diff':diffs(expected,after)})
    else: ledger.append({'date':str(day),'status':'missing_fills_not_assumed_zero','position_change':diffs(before,after)})
    cashrows=summary[summary.day==prev]
    if cashrows.empty or not ap.exists(): continue
    cash=float(cashrows.iloc[-1]['cash'])
    allowed=set(allow_buy_stock_code_list(target_date=day)); full=[s for s in codes if s in allowed]
    candidate=select_prefilter_candidates(scores,idx-1,full,columns,c['weights'],c['prefilter_n'],before)
    checker=LegalityChecker(data,columns,limit_up_protection=c['limit_up_protection'])
    plan=build_rebalance_day(data=data,all_scores=scores,date_idx=idx,trade_idx=idx,signal_date=day,
        valid_stocks=candidate,valid_cols=np.array([columns[s] for s in candidate]),stock_indices=columns,
        weights=c['weights'],buy_n=c['buy_n'],sell_m=c['sell_m'],checker=checker,positions=before,sellable_volumes=before,
        cash=cash,rebalance=c['rebalance'],position_multiplier=.85,target_cash=cash,target_positions=before,
        price_codes_extra=set(before),limit_up_protection=c['limit_up_protection'],rank_within_candidates=True)
    objects={s:NS(volume=q,can_use_volume=q) for s,q in before.items()}
    doc=document_from_day_plan(plan,objects,'OFFLINE',day,dt.timestamp(),dt.timestamp())
    delta={s:r['shares']-before.get(s,0) for s,r in doc['targets'].items() if r['shares']!=before.get(s,0)}
    native={s:-(before[s] if q<0 else q) for s,q in plan.sell_orders}
    for s,q in plan.buy_orders.items(): native[s]=native.get(s,0)+q
    native={s:q for s,q in native.items() if q}
    serialization.append({'date':str(day),'diff':diffs(native,delta)})
    archived=pd.read_parquet(ap); old={}
    for r in archived.to_dict('records'):
        q=int(r['est_volume'])
        if q==0: continue
        if r['direction']=='sell': q=-(before.get(r['code'],0) if q<0 else q)
        elif r['direction']!='buy': continue
        old[r['code']]=old.get(r['code'],0)+q
    plans.append({'date':str(day),'diff':diffs(old,delta),'archived_net':old,'current_recomputed_net':delta})
print('ledger/plans',len(ledger),len(plans),flush=True)
# A continuous current-configuration replay is a model account, not actual holdings.
kwargs=dict(weights=c['weights'],buy_n=c['buy_n'],sell_m=c['sell_m'],holding_period=1,
    position_multipliers=backtest._compute_timing_multipliers(c,dates),limit_up_protection=True,
    rebalance=True,filter_masks=masks,prefilter_n=c['prefilter_n'])
bt=backtest._backtest_direct(data,scores,dates,indices,codes,columns,**kwargs)
def safe(v):
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,np.ndarray): return safe(v.tolist())
    if isinstance(v,np.generic): return safe(v.item())
    if isinstance(v,(date,datetime)): return v.isoformat()
    if isinstance(v,float) and not np.isfinite(v): return None
    return v
(OUT/'continuous_backtest.json').write_text(json.dumps(safe(bt),ensure_ascii=False),encoding='utf-8')
report=dict(period=[str(START),str(END)],trade_days=len(dates),loads=loads,config=c,
    ledger=ledger,missing_positions=missing,seed_plan_comparisons=plans,target_conversion=serialization,
    exact_ledger_days=sum('diff'in r and not r['diff'] for r in ledger),
    gap_ledger_days=sum(bool(r.get('diff')) for r in ledger),
    exact_seed_plan_days=sum(not r['diff'] for r in plans),
    exact_target_conversion_days=sum(not r['diff'] for r in serialization),
    continuous=dict(total_return=bt['total_return'],final_asset=bt['final_asset'],init_cash=1000000),
    process_memory=psutil.Process().memory_info()._asdict(),
    limitations=['Current end-of-day historical data, not archived pre-open quote snapshots.',
    'Seed replay assumes T-1 holdings/cash persist pre-open and all holdings are sellable; broker pre-open receipts needed for exact execution equivalence.',
    'Missing fills and missing account snapshots are not zero.',
    'Current config replay is not historical strategy version reconstruction; Git HEAD is 2026-07-06.',
    'No brokerage SDK, production writes, notification, network or model training.'])
(OUT/'audit.json').write_text(json.dumps(safe(report),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:safe(v) for k,v in report.items() if k not in ['ledger','missing_positions','seed_plan_comparisons','target_conversion','config']},ensure_ascii=False),flush=True)

