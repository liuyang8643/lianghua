"""Post-hoc quote, PnL and full-day-volume diagnostics of actual saved fills."""
import gc,gzip,json
from pathlib import Path
import numpy as np
from offline_data import load_runtime_slice
from offline_data.volume_units import canonical_volume
from ai.supervised.periods import observation_splits,data_end
from ai.supervised.explainability import json_value
from scripts.run_research_guarded import require_guarded_job
from utils.atomic_file import atomic_write_json,file_sha256
from delivery_inputs import admit,bind,verify,read

require_guarded_job();root=Path(__file__).resolve().parent;c,bindings,observations=admit(root)
for p in [Path(__file__),Path('offline_data/volume_units.py')]:bind(bindings,p)
active=read(root/'active_data.json');rt=load_runtime_slice(active['runtime_path'],c['train'][0],data_end(c),preload_rows=100001,cache_dir=Path(c['feature_runtime_cache']))
assert rt.manifest.source_sha256==active['runtime_sha256'];lookup={code:i for i,code in enumerate(rt.stock_codes)}
replays={(r['strategy'],r['split']):r for r in read(root/'regression/complete.json')['results']}
out=root/'extreme_audit';out.mkdir(exist_ok=False);reports=[]

def quote(field,i,j):
    v=float(rt.field(field)[i,j]);return v if np.isfinite(v) else None

def change(a,b):return 100*(a/b-1) if a is not None and b is not None and a>0 and b>0 else None

for strategy in ('ml','static'):
    for split in observation_splits(c):
        directory=root/'regression'/f'{strategy}_{split}';journal=directory/'native_steps.jsonl.gz';record=replays[strategy,split]
        expected=read(directory/'trade_audit.json')['native_steps_sha256'] if strategy=='ml' else record['native_steps_sha256'];bind(bindings,journal,expected)
        path=root/('final' if strategy=='ml' else 'static_baseline')/f'{split}.npz'
        with np.load(path) as z:decisions=z['decision_dates'].astype(str);settlements=z['next_dates'].astype(str);nav=z['nav'];returns=z['returns']
        selected=set([int(np.argmax(returns)),int(np.argmin(returns)),*map(int,np.argsort(-np.abs(returns))[:10])])
        extremes=[];capacity={'fills':0,'zero_full_day_volume_fills':0,'unresolved_volume_unit_fills':0,'fills_above_full_day_volume':0,'fills_above_validated_full_day_amount':0,'fills_outside_daily_ohlc_envelope':0};largest=[];unverified=[];count=0
        with gzip.open(journal,'rt',encoding='utf8') as f:
            for index,line in enumerate(f):
                item=json.loads(line);count+=1;before=item['before'];after=item['after'];plan=item['plan'];diag=item['diagnostics'];fills=item['fills'];day=plan['decision_date']
                assert day==decisions[index] and item['info']['next_decision_date']==settlements[index]
                np.testing.assert_allclose(before['nav'],nav[index],rtol=1e-12,atol=1e-6);assert after['nav']==nav[index+1]
                i=rt.index_of(day);next_i=rt.index_of(settlements[index]);indices=np.asarray([lookup[x['code']] for x in fills],dtype=int)
                volumes=canonical_volume({field:rt.field(field)[i,indices] for field in ('volume','amount','low','high')},shares=True)
                for fill,j,volume in zip(fills,indices,volumes):
                    capacity['fills']+=1;assert fill['price']==float(rt.field('open')[i,j])
                    amount=quote('amount',i,j);ratio=fill['quantity']/volume if np.isfinite(volume) and volume>0 else None
                    amount_ratio=fill['quantity']*fill['price']/amount if ratio is not None and amount is not None and amount>0 else None
                    capacity['zero_full_day_volume_fills']+=int(np.isfinite(volume) and volume==0)
                    capacity['unresolved_volume_unit_fills']+=int(not np.isfinite(volume))
                    capacity['fills_above_full_day_volume']+=int(ratio is not None and ratio>1+1e-10)
                    capacity['fills_above_validated_full_day_amount']+=int(amount_ratio is not None and amount_ratio>1+1e-10)
                    low,high=quote('low',i,j),quote('high',i,j)
                    outside=low is not None and high is not None and (fill['price']<low-.0051 or fill['price']>high+.0051);capacity['fills_outside_daily_ohlc_envelope']+=int(outside)
                    candidate={'date':day,'code':fill['code'],'side':fill['side'],'quantity':fill['quantity'],'price':fill['price'],'low':low,'high':high,'source_volume_raw':quote('volume',i,j),'full_day_volume_shares':float(volume) if np.isfinite(volume) else None,'full_day_amount':amount,'order_to_full_day_volume':ratio,'order_to_full_day_amount':amount_ratio}
                    if (ratio is None or outside) and len(unverified)<20:unverified.append(candidate)
                    largest.append(candidate);largest.sort(key=lambda x:max(x['order_to_full_day_volume'] or 0,x['order_to_full_day_amount'] or 0),reverse=True);del largest[20:]
                if index not in selected:continue
                qfill=dict(before['positions']);flow={};fees={};trade_price={}
                for fill in fills:
                    code=fill['code'];signed=fill['quantity'] if fill['side']=='buy' else -fill['quantity'];qfill[code]=qfill.get(code,0)+signed;flow[code]=flow.get(code,0)-signed*fill['price'];fees[code]=fees.get(code,0)+fill['fee'];trade_price[code]=fill['price']
                stocks=[]
                for code in sorted(set(before['positions'])|set(after['positions'])|set(qfill)):
                    j=lookup[code];q0=before['positions'].get(code,0);q1=qfill.get(code,0);q2=after['positions'].get(code,0)
                    mark=float(before['last_prices'][code]) if q0>0 else float(trade_price.get(code) or quote('open',i,j) or 0)
                    next_mark=float(after['last_prices'].get(code,0));action=diag.get('corporate_action_adjustments',{}).get(code,{});residual=float(action.get('rounding_cash_residual',0))
                    pnl=q2*next_mark-q0*mark+flow.get(code,0)+residual-fees.get(code,0)
                    opening=quote('open',i,j);close=quote('close',i,j);next_open=quote('open',next_i,j);ref=quote('preClose',i,j);next_ref=quote('preClose',next_i,j)
                    intraday=change(close,opening);overnight=change(next_open,next_ref)
                    theoretical=(1+intraday/100)*(1+overnight/100)-1 if intraday is not None and overnight is not None else None
                    actual_hold=(q2*next_mark+residual)/(q1*mark)-1 if q1>0 and mark>0 else None
                    ratio=float(action.get('reference_ratio',1));quoted_ratio=close/next_ref if close is not None and next_ref is not None and next_ref>0 else None
                    quote_equivalence=bool(q1>0 and opening==mark and next_open==next_mark and quoted_ratio is not None and np.isclose(quoted_ratio,ratio,rtol=1e-10,atol=1e-10))
                    if quote_equivalence:np.testing.assert_allclose(actual_hold,theoretical,rtol=1e-10,atol=1e-9)
                    stocks.append({'code':code,'net_pnl':pnl,'nav_contribution_pct':100*pnl/before['nav'],'after_fill_weight_on_pretrade_NAV':q1*mark/before['nav'],'before_quantity':q0,'after_fill_quantity':q1,'after_settlement_quantity':q2,'open':opening,'high':quote('high',i,j),'low':quote('low',i,j),'close':close,'reference_preclose':ref,'next_open':next_open,'next_reference_preclose':next_ref,'opening_gap_from_reference_pct':change(opening,ref),'intraday_open_to_close_pct':intraday,'next_open_vs_next_reference_pct':overnight,'native_gross_hold_return_pct':None if actual_hold is None else 100*actual_hold,'quote_component_identity_checked':quote_equivalence,'applied_reference_ratio':ratio,'source_reference_ratio':quoted_ratio,'mark_provenance':after.get('mark_provenance',{}).get(code),'fees':fees.get(code,0),'corporate_cash_residual':residual,'full_day_amount':quote('amount',i,j)})
                error=sum(x['net_pnl'] for x in stocks)-(after['nav']-before['nav']);assert abs(error)<=max(1e-6,abs(before['nav'])*1e-10),(strategy,split,day,error)
                extremes.append({'decision_date':day,'settlement_date':settlements[index],'return_pct':100*returns[index],'pretrade_nav':before['nav'],'ending_nav':after['nav'],'highest_gain_day':index==int(np.argmax(returns)),'worst_day':index==int(np.argmin(returns)),'pnl_reconciliation_error':error,'stocks':sorted(stocks,key=lambda x:abs(x['net_pnl']),reverse=True)})
        assert count==len(decisions)
        result={'strategy':strategy,'split':split,'days':count,'selected_extreme_days':extremes,'capacity':capacity,'largest_full_day_participations':largest,'unverified_volume_or_quote_examples':unverified,'journal_sha256':expected}
        atomic_write_json(out/f'{strategy}_{split}.json',json_value(result));reports.append({'strategy':strategy,'split':split,'capacity':capacity,'highest_return_pct':float(100*returns.max()),'worst_return_pct':float(100*returns.min()),'report_sha256':file_sha256(out/f'{strategy}_{split}.json')});print({'extreme_audit':strategy+'_'+split,'capacity':capacity},flush=True);gc.collect()
verify(bindings)
atomic_write_json(out/'complete.json',{'state':'completed_posthoc_diagnostics_not_execution_certification','results':reports,'inputs':bindings,'training_calls':0,'model_selection_calls':0,'account_replays':0,'scope':'All saved fills compared with unit-validated full-day volume/amount; highest, worst and10largest-absolute days traced to accounts and quotes. Same admitted runtime, not a new independent full-history quote source.','limitations':['Full-day realized volume is used only after the simulation for diagnosis; it is not a T-open signal or fill limit.','Exceeding historical full-day volume/amount flags failure of the unchanged price-taking/capacity assumption; no strategy or returns are changed here.','A small daily participation does not prove auction queue access or same-open executable prices.','Synthetic corporate-action accounting and historical publication-vintage limitations remain.']})
