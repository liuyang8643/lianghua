"""Frozen-winner replay, accounting checks, native tabs, and honest comparison."""
import gzip,json,html,time,re
from pathlib import Path
import numpy as np
from run import prepare,read,annual
from mvo_adapter_report import evaluate
from testback.economic_positions import EconomicPositionLedger
from testback.backtest import trace_to_report_result,core_report_metrics
from testback.reportor import generate_single_report
from ai.supervised.explainability import json_value
from env.fees import DEFAULT_FEE_SCHEDULE
from utils.atomic_file import atomic_write_json,file_sha256
from scripts.run_research_guarded import require_guarded_job

R=Path(__file__).resolve().parent;P=R.with_name('multihorizon_pre_live_20261108_volume_fix')
def cell(m):return f"{m['annualized_return']*100:.2f}% / {m['max_drawdown']*100:.2f}% / {m['calmar']:.3f}"
def main():
    require_guarded_job();assert read(R/'complete.json')['state']=='complete';w=read(R/'winner.json')['winner'];params=w['parameters'];begun=time.monotonic()
    prepared,observed,_=prepare('validation');ledger=EconomicPositionLedger();counts={'price':0,'legality':0,'fee':0};fills=[];holdings=[];effective=[];fee_rates=[];max_weights=[];lookup={c:i for i,c in enumerate(prepared.runtime.stock_codes)}
    def audit(before,tr):
        ledger(before,tr);market=prepared.episode.market_at(prepared.runtime.index_of(tr.order_plan.decision_date));legal=market.trade_legality(True)
        for fill in tr.step_result.fills:
            j=lookup[fill.code];buy=fill.side=='buy';notional=fill.quantity*fill.price
            counts['price']+=int(fill.price!=market.open_prices[j]);counts['legality']+=int(not(legal.buy_allowed[j] if buy else legal.sell_allowed[j]));fee=DEFAULT_FEE_SCHEDULE.buy_fee(notional) if buy else DEFAULT_FEE_SCHEDULE.sell_fee(notional);counts['fee']+=int(not np.isclose(fill.fee,fee,rtol=1e-10,atol=1e-8))
            i=prepared.runtime.index_of(tr.order_plan.decision_date);amount=float(prepared.runtime.field('amount')[i,j]);fills.append({'date':tr.order_plan.decision_date,'code':fill.code,'side':fill.side,'notional':notional,'daily_amount':amount,'participation':notional/amount if amount>0 else None})
        holdings.append(len(tr.step_result.account_state.positions))
        ac=tr.step_result.account_state;ws=np.array([q*ac.last_prices[code]/ac.nav for code,q in ac.positions.items()]);effective.append(float(ws.sum()**2/(ws@ws)) if len(ws) else 0.);max_weights.append(float(ws.max()) if len(ws) else 0.);fee_rates.append(sum(x.fee for x in tr.step_result.fills)/tr.step_result.diagnostics['pretrade_nav'])
    trace,m,diagnostics,account=evaluate(prepared,params,observed,details=True,on_transition=audit)
    assert not any(counts.values()),counts
    with np.load(R/'external.npz') as a:
        for key,b in [('nav',trace.nav),('returns',trace.portfolio_returns),('cash',trace.cash),('exposure',trace.exposure),('decision_dates',trace.decision_dates),('next_dates',trace.next_decision_dates)]:np.testing.assert_array_equal(a[key],b)
    data=trace_to_report_result(trace,account,canonical_config={'weights':{'MultiHorizonScore':1.},**params});data['metrics']=core_report_metrics(data);ledger.enrich_report(data,account)
    for name in ['trace','final_account','day_config']:data.pop(name,None)
    data.update(init_cash=prepared.initial_cash,period={'start':trace.decision_dates[0],'end':trace.next_decision_dates[-1],'trade_start':trace.decision_dates[0],'trade_end':trace.decision_dates[-1]},report_label='新模型 + MVO / 外验',report_metadata={'engine':'env.EpisodeSession','stock_pool_size':prepared.runtime.n_stocks},rebalance_rule={'signal_timing':'T-open','trade_timing':'T-open','price_field':'open'},per_year_metrics=[],hs300_returns=[])
    data['audit_summary']={'state':'pass','counts':counts,'causality_note':'MVO仅以2011—2021年度OOF选参。风险窗口截止前一交易日开盘。外验不选参；当日开盘同价成交及权益调整仍是原回测假设，不证明竞价可成交。'}
    out=R/'report';out.mkdir(exist_ok=False)
    with gzip.open(out/'report_payload.json.gz','wt',encoding='utf8') as f:json.dump(json_value(data),f,ensure_ascii=False)
    rt=prepared.runtime;hi=int(np.searchsorted(rt.trade_dates,np.datetime64(trace.next_decision_dates[-1]),side='right'));data['_runtime_kline']={'trade_dates':rt.trade_dates[:hi],'stock_codes':np.asarray(rt.stock_codes),**{k:rt.field(k)[:hi] for k in ['open','high','low','close','amount']}}
    report=generate_single_report(data,out)
    intro='<section style="margin:24px;padding:20px;background:white;border:1px solid #ddd"><h2>新模型 + MVO：冻结外验</h2><p>固定3/7/15/30/60日融合；MVO仅在2011—2021 OOF选参。参数：'+html.escape(json.dumps(params))+'</p><p>配置上限约束新增买入；已有价格漂移超限仓位允许在换手预算内逐步减回。没有另加最短持仓天数；使用原生T+1、费用和开盘成交假设。</p><a href="../comparison.html">查看与TopK、静态策略的对照</a></section>'
    text=report.read_text(encoding='utf8');text,count=re.subn(r'(<body[^>]*>)',lambda m:m.group(1)+intro,text,count=1);assert count==1;report.write_text(text,encoding='utf8')
    atomic_write_json(R/'execution_summary.json',{'fills':len(fills),'average_holdings':float(np.mean(holdings)),'turnover_implied_days_approx':2*m['mean_exposure']/m['mean_gross_turnover'],'effective_holdings':float(np.mean(effective)),'max_realized_weight':max(max_weights),'annual_cost_rate_linear':float(np.mean(fee_rates)*252),'closed_holding_days_mean':float(np.mean([x['holding_days'] for x in ledger.closed])),'mean_daily_amount':float(np.mean([x['daily_amount'] for x in fills])),'median_daily_amount':float(np.median([x['daily_amount'] for x in fills])),'participation_p95':float(np.quantile([x['participation'] for x in fills if x['participation'] is not None],.95)),'ledger':ledger.summary(account),'counts':counts,'regression_all_six_exact':True})
    comp=read(P/'final/delivery_comparison.json');oof=next(x for x in comp['rows'] if x['split']=='oof');external=next(x for x in comp['rows'] if x['split']=='validation');old=read(P.with_name('multihorizon_all_periods_preclose_20261107')/'results/ml_test.json')['metrics']
    rows=[['新模型 + MVO',cell(w['metrics']),cell(m)],['新模型 + 当前TopK',oof['ml_cell'],external['ml_cell']],['旧模型 + 当前TopK','不适用：折训练区间不同',cell(old)],['原静态',oof['static_cell'],external['static_cell']]]
    year_rows=''.join('<tr><td>'+a['year']+'</td><td>'+format(a['return']*100,'.2f')+'%</td><td>'+format(a['drawdown']*100,'.2f')+'%</td></tr>' for a in w['annual'])
    body=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)
    note='每格：扣费年化 / 最大回撤 / Calmar。OOF为2011—2021连续账户；外验为2022—2026-10-30。冻结模型和融合权重，只有MVO参数参与OOF搜索。不是全局最优证明，外验历史已被观察。MVO是Qlib普通均值方差目标的本地约束适配，排名效用并非校准预期收益；保留相关风险，未使用指数跟踪。单股上限限制新增配置，已有价格漂移超限仓位不得继续加仓，可按换手预算逐步减回；不是保证每天实际持仓都低于上限。MVO没有另加最短持仓天数或每日换出只数限制，保留原生T+1和合法性规则。风险窗口120日、收缩50%，风险数据截止前一交易日开盘；首日现金建仓有显式换手豁免。'
    (R/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>MVO比较</title><style>body{font:16px sans-serif;margin:40px;line-height:1.8}td,th{padding:12px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>MVO与现有策略比较</h1><p>'+note+'</p><p>参数：'+html.escape(json.dumps(params))+'</p><table><tr><th>策略</th><th>训练OOF</th><th>外验</th></tr>'+body+'</table><h2>赢家的11折年度表现（连续账户分段统计）</h2><table><tr><th>年度</th><th>年度累计收益</th><th>年内回撤</th></tr>'+year_rows+'</table><p><a href="report/single_report.html">原版详细多标签报告</a></p>',encoding='utf8')
    atomic_write_json(R/'delivery_complete.json',{'state':'complete','seconds':time.monotonic()-begun,'table':rows,'counts':counts,'regression_all_six_exact':True,'html_sha256':file_sha256(report),'winner_sha256':file_sha256(R/'winner.json'),'model_retrained':False,'live_deployed':False})
    print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
