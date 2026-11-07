"""Native detailed report delivery shared by ML and fixed-strategy observations."""
import gzip,json
from pathlib import Path
import numpy as np
from env.metrics import performance_from_log_rewards
from testback.backtest import trace_to_report_result,core_report_metrics
from testback.reportor import generate_single_report
from .explainability import json_value
from .periods import observation_splits
from utils.atomic_file import atomic_write_json,file_sha256

def write_detailed_observation(root,strategy,split,rt,trace,account,config,ledger):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'));out=root/'reports'/f'{strategy}_{split}';out.mkdir(parents=True,exist_ok=False)
    data=trace_to_report_result(trace,account,canonical_config=config);data['metrics']=core_report_metrics(data);ledger.enrich_report(data,account)
    for name in ['trace','final_account','day_config']:data.pop(name,None)
    years=np.asarray(trace.next_decision_dates,dtype='datetime64[D]').astype('datetime64[Y]').astype(str);rows=[]
    for year in np.unique(years):
        selected=years==year;m=performance_from_log_rewards(np.log1p(trace.portfolio_returns[selected]));rows.append({'year':year,'annualized':m.annualized_return*100,'sharpe':m.sharpe,'max_drawdown':-m.max_drawdown*100,'return':m.total_return*100,'trading_days':int(selected.sum())})
    data.update(init_cash=protocol['initial_cash'],per_year_metrics=rows,per_year_metrics_available=True,period={'start':trace.decision_dates[0],'end':trace.next_decision_dates[-1],'trade_start':trace.decision_dates[0],'trade_end':trace.decision_dates[-1]},report_label=f'{strategy} / {split}',report_metadata={'engine':'env.EpisodeSession','runtime_path':rt.manifest.source_path,'stock_pool_size':rt.n_stocks,'position_accounting':'economic zero-to-zero episode; not brokerFIFO'},rebalance_rule={'signal_timing':'T-open','trade_timing':'T-open','price_field':'open'})
    data['report_navigation']=[{'label':f'{s} / {name}','href':f'../{s}_{name}/single_report.html','current':s==strategy and name==split} for s in ['ml','static'] for name in observation_splits(protocol)]
    active=json.loads((root/'active_data.json').read_text(encoding='utf8'));basic=Path(active['manifest_path']).parent.parent/'tail_source/stock_basic.json'
    if basic.exists():data['stock_name_map']={x['code'].split('.')[1]+'.'+x['code'].split('.')[0].upper():x['code_name'] for x in json.loads(basic.read_text(encoding='utf8'))}
    atomic_write_json(out/'position_ledger.json',{'summary':ledger.summary(account),'closed':ledger.closed,'open':ledger.open,'return_definition':'episode net profit divided by cumulative buy cash includingfees; includes synthetic corporate-action cash residuals'})
    with gzip.open(out/'report_payload.json.gz','wt',encoding='utf8') as f:json.dump(json_value(data),f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    hi=int(np.searchsorted(rt.trade_dates,np.datetime64(trace.next_decision_dates[-1]),side='right'))
    data['_runtime_kline']={'trade_dates':rt.trade_dates[:hi],'stock_codes':np.asarray(rt.stock_codes),**{k:rt.field(k)[:hi] for k in ['open','high','low','close','amount']}}
    path=generate_single_report(data,out)
    atomic_write_json(out/'complete.json',{'state':'complete','report_sha256':file_sha256(path),'payload_sha256':file_sha256(out/'report_payload.json.gz'),'ledger_sha256':file_sha256(out/'position_ledger.json'),'actual_fills':sum(len(x) for x in trace.fills),'closed_episodes':len(ledger.closed),'account_pnl_reconciled':True,'broker_exact':False})
    return path
