"""Frozen final observations, fill audits and self-contained reproducibility report."""
import argparse,csv,gc,html,json,time
from pathlib import Path
import numpy as np
from env.simulator import DaySimulator
from .multihorizon_search import load_forecasts,evaluate_candidate,candidate_key
from .final_evaluation import write_csv
from .periods import observation_splits,oof_interval
from .reporting import write_detailed_observation
from testback.economic_positions import EconomicPositionLedger
from utils.atomic_file import atomic_write_json,file_sha256

def run(root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();out=root/'final';out.mkdir(exist_ok=False)
    c=json.loads((root/'protocol.json').read_text(encoding='utf8'));winner=json.loads((root/'search/winner.json').read_text());p=winner['winner']['parameters'];wh=file_sha256(root/'search/winner.json')
    rows=[];results=[];original=DaySimulator.step
    for split in observation_splits(c):
        ledger=EconomicPositionLedger()
        rt,cube,provenance=load_forecasts(root,split);lookup={s:i for i,s in enumerate(rt.stock_codes)};di={str(d):i for i,d in enumerate(rt.trade_dates)};born={};fills=[];holder={}
        audit=dict(invalid_open=0,illegal_side=0,premature_sale=0,topup=0,over_budget=0,excess_sells=0,excess_positions=0)
        def observe(self,account,plan,open_prices,next_open_prices,**kw):
            result=original(self,account,plan,open_prices,next_open_prices,**kw);i=di[plan.decision_date]
            ledger.observe(account,result.account_state,plan,result.fills,result.diagnostics)
            legal=holder['prepared'].episode.market_at(i).trade_legality(plan.day_config.limit_up_protection)
            for code,quantity in account.positions.items():
                if quantity>0:born.setdefault(code,i-1)
            qty=dict(account.positions);sells=set();nav=float(result.diagnostics['pretrade_nav'])
            for f in result.fills:
                j=lookup[f.code];price=float(rt.field('open')[i,j]);buy=f.side=='buy';age=i-born.get(f.code,i)
                audit['invalid_open']+=int(not(np.isfinite(price) and price>0))
                audit['illegal_side']+=int(not(legal.buy_allowed[j] if buy else legal.sell_allowed[j]))
                audit['premature_sale']+=int(not buy and age<p['hold_thresh'])
                audit['topup']+=int(buy and account.positions.get(f.code,0)>0)
                audit['over_budget']+=int(buy and f.quantity*f.price>nav/p['sell_m']+max(1e-6,nav*1e-12))
                if buy:born.setdefault(f.code,i)
                else:sells.add(f.code)
                qty[f.code]=qty.get(f.code,0)+(f.quantity if buy else -f.quantity)
                fills.append(dict(date=plan.decision_date,code=f.code,side=f.side,quantity=f.quantity,price=f.price,fee=f.fee,holding_days=age))
            audit['excess_sells']+=int(len(sells)>p['max_replacements']);audit['excess_positions']+=int(sum(q>0 for q in qty.values())>p['sell_m'])
            for code in list(born):
                if result.account_state.positions.get(code,0)==0:born.pop(code)
            return result
        DaySimulator.step=observe
        try:trace,m,_=evaluate_candidate(rt,cube,p,c['initial_cash'],wh,split,details=True,on_prepared=lambda prepared:holder.update(prepared=prepared))
        finally:DaySimulator.step=original
        m['total_fees']=float(sum(f.fee for day in trace.fills for f in day))
        assert not any(audit.values()),audit
        np.testing.assert_allclose(trace.portfolio_returns,trace.nav[1:]/trace.nav[:-1]-1,rtol=1e-10,atol=1e-12)
        path=out/f'{split}.npz';np.savez_compressed(path,nav=trace.nav,returns=trace.portfolio_returns,exposure=trace.exposure,cash=trace.cash,decision_dates=trace.decision_dates,next_dates=trace.next_decision_dates)
        write_csv(out/f'{split}_fills.csv',fills)
        display_config={**p,'weights':{'MultiHorizonScore':1.},'horizon_weights_pct':{str(h):int(w)*10 for h,w in zip(c['model_horizons'],p['weights'])}}
        write_detailed_observation(root,'ml',split,rt,trace,holder['prepared'].final_account,display_config,ledger)
        result={'split':split,'metrics':m,'audit':audit,'trace_sha256':file_sha256(path),'prediction_sources':provenance};results.append(result);rows.append({'split':split,**{k:v for k,v in m.items() if not isinstance(v,dict)}})
        if split=='oof':
            key=__import__('hashlib').sha256(candidate_key(p).encode()).hexdigest()
            with np.load(root/'search/trials'/f'{key}.npz') as old:np.testing.assert_array_equal(trace.nav,old['nav'])
        atomic_write_json(out/f'{split}.json',result);print({'split':split,'metrics':m},flush=True)
        holder.clear();del trace,rt,cube;gc.collect();assert file_sha256(root/'search/winner.json')==wh
    write_csv(out/'metrics.csv',rows)
    atomic_write_json(out/'complete.json',{'state':'complete','results':results,'winner_sha256':wh})

def report(root):
    c=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    out=root/'final';data=json.loads((out/'complete.json').read_text());w=json.loads((root/'search/winner.json').read_text());p=w['winner']['parameters']
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>五周期共享模型与跨折GA</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd;text-align:right}img{width:100%}</style>',
        '<h1>五周期共享LightGBM＋跨折连续账户GA</h1>',
        '<p>周期3/7/15/30/60；每折只在拟合段重筛全部本地因子，禁隔夜跳空；正RankIC与0.80相关性去重；类别型N单模型，按任务/日期等权。训练范围内的年度折外连续账户用于选参，账户跨年不重置。</p>',
        '<p>最终参数：'+html.escape(json.dumps(p,ensure_ascii=False))+'</p>',
        f'<p>GA状态：{w["state"]}；{w["generations"]}代，{w["unique_trials"]}个唯一候选，停止原因{w["stop_reason"]}。权重单位为10%，不是收益率。</p>',
        '<p>最终模型拟合协议中的完整train区间，轮数取各折最佳轮数中位数。因此fit、internal_validation和train均为最终模型的样本内回放；只有OOF使用各年过去模型。协议中存在的外验及测试仅观察。两种模型更新协议分别报告，历史年份已经被研究，不是新盲测。</p>',
        '<p>沿用本地开盘同价执行、费用和公司行动假设；没有验证竞价排队和实际容量。没有接入实盘。</p>',
        '<table><tr><th>区间</th><th>扣费年化</th><th>回撤</th><th>Calmar</th><th>平均仓位</th></tr>']
    for r in data['results']:
        m=r['metrics'];parts.append(f'<tr><td>{r["split"]}</td><td>{m["annualized_return"]:.2%}</td><td>{m["max_drawdown"]:.2%}</td><td>{m["calmar"]:.3f}</td><td>{m["mean_exposure"]:.2%}</td></tr>')
    parts+=['</table><h2>各折输入与早停</h2><table><tr><th>模型</th><th>因子数</th><th>树数</th><th>名单</th></tr>']
    for folder in sorted((root/'models').glob('fold*'))+[root/'models/final']:
        m=json.loads((folder/'complete.json').read_text())
        parts.append(f'<tr><td>{folder.name}</td><td>{m["feature_count"]}</td><td>{m["trees"]}</td><td><a href="../fold_selection/{folder.name}/complete.json">名单与逐周期筛选目录</a></td></tr>')
    parts.append('</table>')
    # Fresh controls are observations, read only after the new winner is frozen.
    static=json.loads((root/'static_baseline/complete.json').read_text())['results']
    parts+=['<h2>数值修复后的静态基准对照</h2><p>本次发现早期极端成交额污染滚动Amihud因子的数值问题，已重建3个基因子与6个窗口变体。旧模型使用过受影响window30输入，因此旧ML/静态成绩不再作为无已知缺陷基准；旧记录保留，当前静态使用相同原配置重新计算。</p><table><tr><th>区间/方案</th><th>扣费年化</th><th>回撤</th><th>Calmar</th></tr>']
    comparison=[]
    for split in observation_splits(c):
        rows=[('当前多周期',next(r for r in data['results'] if r['split']==split)),('本地静态',next(r for r in static if r['split']==split))]
        for name,r in rows:
            m=r['metrics'];comparison.append({'strategy':name,'split':split,**{k:v for k,v in m.items() if not isinstance(v,dict)}})
            parts.append(f'<tr><td>{split}/{name}</td><td>{m["annualized_return"]:.2%}</td><td>{m["max_drawdown"]:.2%}</td><td>{m["calmar"]:.3f}</td></tr>')
    write_csv(out/'comparison.csv',comparison);parts.append('</table>')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chart_splits=[s for s in ['oof','validation','test'] if s in observation_splits(c)]
    fig,axes=plt.subplots(1,len(chart_splits),figsize=(5*len(chart_splits),4),layout='constrained')
    for ax,split in zip(np.atleast_1d(axes),chart_splits):
        with np.load(out/f'{split}.npz') as z:
            dates=np.r_[z['decision_dates'][:1],z['next_dates']].astype('datetime64[D]');ax.plot(dates,z['nav']/z['nav'][0]);ax.set_yscale('log');ax.set_title(split);ax.grid(alpha=.2)
    fig.savefig(out/'equity.png',dpi=150);plt.close(fig)
    parts+=['<img src="equity.png" alt="折外与外部区间净值"><p><a href="metrics.csv">指标</a> · <a href="../search/trials.csv">GA全部结果</a> · <a href="../protocol.json">协议</a> · <a href="../reproduction.json">复现说明</a> · <a href="../independent_review.json">独立审核</a></p>',
        '<p>新增：共享多周期数据适配、逐折筛选、连续OOF GA和复现编排。复用：原因子库、LightGBM、原生账户/费用/执行。替换删除：无。依赖变化：无。领域schema变化：无；新增研究模型与任务字段。冗余检查：没有第二套成交或回测引擎。</p></html>']
    (out/'report.html').write_text('\n'.join(parts),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--report-only',action='store_true');a=p.parse_args()
    if not a.report_only:run(a.root)
    report(a.root)
