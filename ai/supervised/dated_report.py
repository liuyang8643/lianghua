"""A frozen forecast replay, explanations and the existing single-report renderer."""
import argparse,csv,json,shutil,time,traceback
from pathlib import Path
import numpy as np
from offline_data import load_runtime_slice
from factor import factor_coverage
from .portfolio import PreparedModelPortfolio
from .explainability import ReplayAudit,json_value,explain
from testback.backtest import trace_to_report_result,core_report_metrics
from testback.reportor import generate_single_report
from utils.atomic_file import atomic_write_json,file_sha256


def csv_rows(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def replay(root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();begun=time.monotonic();task=json.loads((root/'task.json').read_text());parent=Path(task['parent']);stage=root/'features'
    done=json.loads((stage/'complete.json').read_text());assert done['state']=='complete'
    for name,sha in done['sha256'].items():assert file_sha256(stage/name)==sha,name
    active=json.loads((root/'active_data.json').read_text());assert active['state']=='admitted_market_readback'
    assert file_sha256(parent/'models/final/model.txt')==task['model_sha256'];assert file_sha256(parent/'search/winner.json')==task['winner_sha256']
    rt=load_runtime_slice(active['runtime_path'],task['requested_start'],task['requested_end'],preload_rows=100001,cache_dir=root/'runtime_cache')
    dates=np.load(stage/'dates.npy');np.testing.assert_array_equal(rt.trade_dates[rt.decision_start:rt.decision_stop],dates)
    np.testing.assert_array_equal(np.asarray(rt.stock_codes),np.load(stage/'codes.npy').astype(str))
    scores=np.full((rt.n_dates,rt.n_stocks),np.nan);scores[rt.decision_start:rt.decision_stop]=np.load(stage/'scores.npy')
    out=root/'replay';out.mkdir(exist_ok=False)
    source_paths=[*Path('env').rglob('*.py'),Path(__file__),Path('ai/supervised/portfolio.py'),Path('ai/supervised/explainability.py'),Path('testback/backtest.py')]
    sources={str(p):file_sha256(p) for p in source_paths}
    for p in source_paths:
        target=out/'source'/p.relative_to(Path.cwd()) if p.is_absolute() else out/'source'/p
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    atomic_write_json(out/'identity.json',{'sources':sources,'feature_complete_sha256':file_sha256(stage/'complete.json'),'active_data_sha256':file_sha256(root/'active_data.json'),'model_sha256':task['model_sha256'],'parameters':params if 'params' in locals() else task['parameters']})
    prepared=PreparedModelPortfolio.from_scores(rt,scores,initial_cash=task['initial_cash'],source_hash=file_sha256(stage/'scores.npy'),split='2026_ytd')
    audit=ReplayAudit(out,prepared,task['parameters'])
    params={k:task['parameters'][k] for k in ['buy_n','sell_m','hold_thresh','max_replacements']}
    try:
        trace,metrics,diagnostics=prepared.evaluate('rank_buffer',params,details=True,on_transition=audit)
        audit.finish()
    except BaseException as exc:
        atomic_write_json(out/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'completed_observed_days':len(audit.days),'counts':audit.counts})
        raise
    finally:audit.stream.close()
    result=trace_to_report_result(trace,prepared.final_account,canonical_config={'weights':{'MultiHorizonScore':1.},**params},coverage=factor_coverage(rt,prepared.factors))
    report_metrics=core_report_metrics(result)
    period={'signal_start':result['signal_dates'][0],'signal_end':result['signal_dates'][-1],'trade_start':result['trade_dates'][0],'trade_end':result['trade_dates'][-1],'settlement_start':result['nav_dates'][0],'settlement_end':result['nav_dates'][-1],'start':result['signal_dates'][0],'end':result['nav_dates'][-1]}
    payload={k:v for k,v in result.items() if k not in ('trace','final_account','day_config')}
    payload.update(metrics=report_metrics,init_cash=task['initial_cash'],period=period,report_metadata={'engine':'env.EpisodeSession','runtime_path':active['runtime_path'],'stock_pool_size':rt.n_stocks,'model_sha256':task['model_sha256'],'winner_sha256':task['winner_sha256']},rebalance_rule={'signal_timing':'T-open','trade_timing':'T-open','price_field':'open','mode':'accumulating_rank_buffer','allow_cash':True},per_year_metrics=[],hs300_returns=[])
    basic=json.loads((root/'tail_source/stock_basic.json').read_text(encoding='utf8'));payload['stock_name_map']={x['code'].split('.')[1]+'.'+x['code'].split('.')[0].upper():x['code_name'] for x in basic}
    payload['strategy_explanation']={'horizons':[3,7,15,30,60],'weights':[v/10 for v in task['parameters']['weights']],**params,'model_features':task['feature_count'],'trees':562,'description':'五周期共享模型分别预测开盘累计收益的相对排名；各头再转当日截面百分位并按冻结权重融合。买入候选取可买股票中前40名；持仓与可买股票合并排名，持仓跌出前40且已满3个交易日、能够卖出时才主动退出，每日最多换出1只。新买单笔额度为调仓前净值/40，允许现金，不自动补仓。','timing_note':'仅T日open及此前已完成行情进入决策；T日close等字段只用于下一开盘结算或事后K线展示。看到open后仍按同价成交是本地回测假设，不证明集合竞价可成交。','date_note':f"最后实际交易日{period['trade_end']}，最后估值为{period['end']}开盘；终日模型信号单列，不伪造下一交易日收益。"}
    np.savez_compressed(out/'trace.npz',nav=trace.nav,returns=trace.portfolio_returns,decision_dates=trace.decision_dates,next_dates=trace.next_decision_dates,cash=trace.cash,exposure=trace.exposure)
    atomic_write_json(out/'report_payload.json',json_value(payload));atomic_write_json(out/'strategy_diagnostics.json',json_value(diagnostics))
    csv_rows(out/'fills.csv',result['trade_log'])
    flat=[{k:v for k,v in row.items() if k!='corporate_action'} for row in audit.positions];csv_rows(out/'position_pnl.csv',flat)
    csv_rows(out/'daily_pnl.csv',[{k:v for k,v in day.items() if k!='stocks'} for day in audit.days])
    extremes=sorted(audit.days,key=lambda x:x['return_pct'],reverse=True)[:10]
    absolute=sorted(audit.days,key=lambda x:abs(x['return_pct']),reverse=True)[:10]
    atomic_write_json(out/'extreme_days.json',json_value({'highest_positive_days':extremes,'largest_absolute_days':absolute}))
    traded=sorted({x['code'] for x in result['trade_log']}|set(prepared.final_account.positions));index={c:i for i,c in enumerate(rt.stock_codes)};cols=[index[c] for c in traded];start=max(0,rt.decision_start-252)
    np.savez_compressed(out/'kline.npz',trade_dates=rt.trade_dates[start:rt.decision_stop],stock_codes=np.asarray(traded),**{f:rt.field(f)[start:rt.decision_stop,cols] for f in ['open','high','low','close','amount']})
    last=np.load(stage/'scores.npy',mmap_mode='r')[-1];order=np.argsort(-np.where(np.isfinite(last),last,-np.inf),kind='stable')
    atomic_write_json(out/'final_open_signals.json',{'date':str(dates[-1]),'status':'model_signals_only_no_next_open_settlement','top40':[{'rank':k+1,'code':rt.stock_codes[j],'score':float(last[j]),'held_at_final_mark':rt.stock_codes[j] in prepared.final_account.positions} for k,j in enumerate(order[:40]) if np.isfinite(last[j])]})
    assert file_sha256(parent/'models/final/model.txt')==task['model_sha256'];assert file_sha256(parent/'search/winner.json')==task['winner_sha256']
    for p,sha in sources.items():assert file_sha256(p)==sha,p
    receipt={'state':'complete','period':period,'metrics':metrics,'report_metrics':report_metrics,'seconds':time.monotonic()-begun,'model_sha256':task['model_sha256'],'winner_sha256':task['winner_sha256'],'files':{p.name:file_sha256(p) for p in out.iterdir() if p.is_file()}}
    atomic_write_json(out/'complete.json',receipt);print({'state':'replay_complete','period':period,'metrics':report_metrics},flush=True)


def report(root):
    data=json.loads((root/'replay/report_payload.json').read_text(encoding='utf8'))
    data['explainability']=json.loads((root/'explanations/explanations.json').read_text(encoding='utf8'))
    data['audit_summary']=json.loads((root/'replay/trade_audit.json').read_text(encoding='utf8'))
    audit_notes=root/'audit_notes.json'
    if audit_notes.exists():data['audit_summary']['causality_note']=json.loads(audit_notes.read_text(encoding='utf8'))['causality_note']
    comparison=root/'comparison/report_summary.json'
    if comparison.exists():data['data_comparison']=json.loads(comparison.read_text(encoding='utf8'))
    if (root/'report/explanation_overview.png').exists():data['static_explanation_image']='explanation_overview.png'
    with np.load(root/'replay/kline.npz') as z:data['_runtime_kline']={k:z[k] for k in z.files}
    path=generate_single_report(data,root/'report')
    atomic_write_json(root/'report/receipt.json',{'state':'rendered_pending_review','path':str(path),'html_sha256':file_sha256(path),'native_report_reused':True,'inputs':{str(p.relative_to(root)):file_sha256(p) for p in [root/'replay/complete.json',root/'explanations/complete.json']}})
    print({'report':str(path)},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['replay','explain','report'],required=True);a=p.parse_args()
    if a.stage=='replay':replay(a.root)
    elif a.stage=='explain':explain(a.root)
    else:report(a.root)
