"""Saved prediction explanations and accounting diagnostics; never a policy."""
from collections.abc import Mapping
from dataclasses import fields,is_dataclass
import ast,datetime,gzip,json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from env.fees import DEFAULT_FEE_SCHEDULE
from utils.atomic_file import atomic_write_json,file_sha256


def json_value(value):
    if is_dataclass(value):return {f.name:json_value(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,Mapping):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,np.ndarray):return json_value(value.tolist())
    if isinstance(value,np.generic):return json_value(value.item())
    if isinstance(value,(list,tuple)):return [json_value(x) for x in value]
    if isinstance(value,(datetime.date,datetime.datetime)):return value.isoformat()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


class ReplayAudit:
    """Observe the canonical transitions without modifying their decisions."""
    def __init__(self,root,prepared,parameters):
        self.root=Path(root);self.prepared=prepared;self.rt=prepared.runtime;self.parameters=parameters
        self.lookup={c:i for i,c in enumerate(self.rt.stock_codes)};self.day_lookup={str(d):i for i,d in enumerate(self.rt.trade_dates)}
        self.birth={};self.days=[];self.positions=[]
        self.counts={k:0 for k in ['invalid_open','illegal_side','premature_sale','topup','over_budget','excess_sells','excess_positions','fee_error','invalid_quantity']}
        self.stream=gzip.open(self.root/'native_steps.jsonl.gz','wt',encoding='utf8');self.reconciliation_errors=[]

    def __call__(self,before,transition):
        result=transition.step_result;after=result.account_state;plan=transition.order_plan;diag=result.diagnostics
        self.stream.write(json.dumps(json_value({'before':before,'after':after,'plan':plan,'fills':result.fills,'diagnostics':diag,'info':transition.info}),ensure_ascii=False,allow_nan=False)+'\n')
        date=plan.decision_date;i=self.day_lookup[date];nav=float(diag['pretrade_nav']);legal=self.prepared.episode.market_at(i).trade_legality(plan.day_config.limit_up_protection)
        post=dict(before.positions);flow={};cost={};buys=set();sells=set();fees=DEFAULT_FEE_SCHEDULE
        for fill in result.fills:
            j=self.lookup[fill.code];price=float(self.rt.field('open')[i,j]);buy=fill.side=='buy';quantity=fill.quantity
            self.counts['invalid_open']+=int(not np.isfinite(price) or price<=0 or fill.price!=price)
            self.counts['illegal_side']+=int(not(legal.buy_allowed[j] if buy else legal.sell_allowed[j]))
            self.counts['invalid_quantity']+=int(type(quantity) is not int or quantity<=0 or (not buy and quantity>before.sellable_positions.get(fill.code,0)))
            if not buy:self.counts['premature_sale']+=int(fill.code not in self.birth or i-self.birth[fill.code]<self.parameters['hold_thresh'])
            self.counts['topup']+=int(buy and before.positions.get(fill.code,0)>0)
            notional=quantity*fill.price;self.counts['over_budget']+=int(buy and notional>nav/self.parameters['sell_m']+1e-6)
            expected=max(notional*fees.commission_rate,fees.minimum_commission)+notional*(fees.transfer_fee_rate+fees.slippage_rate+(0 if buy else fees.stamp_tax_rate))
            self.counts['fee_error']+=int(not np.isclose(expected,fill.fee,rtol=1e-12,atol=1e-7))
            if buy:buys.add(fill.code);self.birth.setdefault(fill.code,i)
            else:sells.add(fill.code)
            post[fill.code]=post.get(fill.code,0)+(quantity if buy else -quantity)
            flow[fill.code]=flow.get(fill.code,0)+(-notional if buy else notional)
            cost[fill.code]=cost.get(fill.code,0)+fill.fee
        self.counts['excess_sells']+=int(len(sells)>self.parameters['max_replacements'])
        self.counts['excess_positions']+=int(sum(q>0 for q in post.values())>self.parameters['sell_m'])
        adjustments=diag['corporate_action_adjustments'];rows=[]
        for code in sorted(set(before.positions)|set(after.positions)|set(post)|buys|sells):
            j=self.lookup[code];q0=before.positions.get(code,0);q1=post.get(code,0);q2=after.positions.get(code,0)
            raw_open=float(self.rt.field('open')[i,j]);mark=raw_open if np.isfinite(raw_open) and raw_open>0 else float(before.last_prices.get(code,0))
            next_mark=float(after.last_prices.get(code,0));opening=q0*mark;ending=q2*next_mark;adjustment=adjustments.get(code,{})
            residual=float(adjustment.get('rounding_cash_residual',0));net=ending-opening+flow.get(code,0)+residual-cost.get(code,0)
            row={'decision_date':date,'settlement_date':transition.info['next_decision_date'],'code':code,'before_quantity':q0,'after_fill_quantity':q1,'after_settlement_quantity':q2,'open_price':raw_open,'current_mark':mark,'next_mark':next_mark,'gross_trade_cash':flow.get(code,0),'fees':cost.get(code,0),'corporate_cash_residual':residual,'net_pnl':net,'nav_contribution_pct':100*net/nav,'open_weight':q1*mark/nav,'bought':code in buys,'sold':code in sells,'holding_days':i-self.birth.get(code,i),'corporate_action':dict(adjustment),'mark_provenance':after.mark_provenance.get(code),'day_amount':float(self.rt.field('amount')[i,j]),'day_volume_source':float(self.rt.field('volume')[i,j])}
            rows.append(row);self.positions.append(row)
        attributed=sum(x['net_pnl'] for x in rows);actual=float(after.nav)-nav;error=actual-attributed
        self.reconciliation_errors.append(abs(error));assert abs(error)<=max(1e-6,abs(nav)*1e-10),(date,error)
        day={'decision_date':date,'settlement_date':transition.info['next_decision_date'],'pretrade_nav':nav,'ending_nav':float(after.nav),'return_pct':100*result.portfolio_return,'fees':float(diag['total_fees']),'pnl_sum':attributed,'reconciliation_error':error,'cash_before':before.cash,'cash_after':after.cash,'post_fill_cash':float(diag['post_fill_cash']),'post_fill_exposure':float(diag['post_fill_exposure']),'stocks':rows}
        self.days.append(day)
        for code in list(self.birth):
            if after.positions.get(code,0)<=0:self.birth.pop(code)

    def finish(self):
        self.stream.close()
        atomic_write_json(self.root/'daily_attribution.json',json_value(self.days))
        atomic_write_json(self.root/'trade_audit.json',{'state':'failed' if any(self.counts.values()) else 'pass','counts':self.counts,'days':len(self.days),'position_day_rows':len(self.positions),'max_pnl_reconciliation_error':max(self.reconciliation_errors,default=0),'native_steps_sha256':file_sha256(self.root/'native_steps.jsonl.gz'),'scope':'Actual observed native fills, before/after accounts, fees and per-stock NAV attribution. No independent broker execution/auction-capacity claim.'})
        assert not any(self.counts.values()),self.counts


def tree_contributions(model,x):
    """LightGBM TreeSHAP for raw regression scores, including task N and bias."""
    contributions=model.predict(x,pred_contrib=True,num_threads=8)
    raw=model.predict(x,raw_score=True,num_threads=8)
    np.testing.assert_allclose(contributions.sum(axis=1),raw,rtol=1e-10,atol=1e-10)
    return contributions,raw


def feature_descriptions(parent,names):
    records={x['name']:x for x in json.loads((parent/'selection/selected_3d.json').read_text(encoding='utf8'))['selected']}
    explained={
        'AmountBasedSmallCap5':('过去5个已完成交易日平均成交额越低，原始因子分越高；这是低成交额代理，不是直接测量市值。','factor/library/optimized_eight.py'),
        'Optimized__TrueMarketCap':('开盘价乘以滞后一日股本的负值；市值越小，原始因子分越高。模型使用其截面排名。','factor/library/optimized_eight.py'),
        'BiliLowPriceAbove2':('上一交易日不复权收盘价须高于2元；价格越低，原始因子分越高。','factor/library/bilibili_price_patterns.py'),
        'LowResidualVol20':('过去20个已完成交易日，个股收益减当日全市场均值后的波动率取负值；残差波动越低，分越高。','factor/library/optimized_eight.py'),
        'LowCashOutflowProfitGrowthSpread__penalty_0':('净利润同比增速减0倍现金流出同比增速，即利润增速信号；仍要求两项原始财务数据都已可用。','factor/library/local_variants.py'),
        'PBBelowTwoROEAbove10Signal__pb_2__roe_0.05':('PB低于2且ROE高于5%时取1，否则取0；使用当时已知财务数据和开盘价。','factor/library/local_variants.py'),
    }
    result={}
    for name in names:
        record=records[name];meta=record.get('metadata',{});base=name.split('__')[0]
        description='项目注册因子';source=Path('factor_db/factors')/(base+'.py')
        if source.exists():
            tree=ast.parse(source.read_text(encoding='utf-8-sig'))
            for node in tree.body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__thesis__' for t in node.targets) and isinstance(node.value,ast.Constant):description=str(node.value.value)
                elif isinstance(node,ast.ClassDef) and node.name==base and ast.get_docstring(node):description=ast.get_docstring(node).split('\n')[0]
        expression=record.get('expression')
        if expression:description='Qlib量价表达式；外层Ref(...,1)整体滞后至已完成交易日'
        if name in explained:description,source_text=explained[name];source=Path(source_text)
        result[name]={'description':description,'formula':expression or '', 'input_fields':meta.get('required_fields',[]),'version':meta.get('version',''),'source':str(source) if source.exists() else 'factor/library 与注册元数据'}
    return result


def explain(root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job()
    root=Path(root);task=json.loads((root/'task.json').read_text());parent=Path(task['parent']);stage=root/'features';replay=root/'replay';out=root/'explanations';out.mkdir(exist_ok=False)
    done=json.loads((stage/'complete.json').read_text());assert done['state']=='complete'
    assert file_sha256(parent/'models/final/model.txt')==task['model_sha256']
    meta=json.loads((parent/'models/final/model.json').read_text());names=meta['feature_names'];horizons=meta['horizons'];weights=np.asarray(task['parameters']['weights'],float)/10
    inventory=json.loads((stage/'inventory.json').read_text())
    for name in names:
        for key,sha in [('path','sha256'),('raw_path','raw_sha256')]:assert file_sha256(inventory[name][key])==inventory[name][sha],name
    columns=[np.load(inventory[n]['path'],mmap_mode='r') for n in names]
    raw_columns=[np.load(inventory[n]['raw_path'],mmap_mode='r') for n in names]
    dates=np.load(stage/'dates.npy').astype(str);codes=np.load(stage/'codes.npy').astype(str);di={d:i for i,d in enumerate(dates)};ci={c:i for i,c in enumerate(codes)}
    eligible=np.load(stage/'eligible.npy',mmap_mode='r');pred=np.load(stage/'predictions.npy',mmap_mode='r');ranks=np.load(stage/'head_ranks.npy',mmap_mode='r');scores=np.load(stage/'scores.npy',mmap_mode='r')
    days=json.loads((replay/'daily_attribution.json').read_text());samples=[]
    for d,day in enumerate(days):
        for stock in day['stocks']:
            i,j=di[day['decision_date']],ci[stock['code']]
            if eligible[i,j] and np.isfinite(pred[i,j]).all():samples.append((d,i,j,stock))
    model=lgb.Booster(model_file=str(parent/'models/final/model.txt'));contrib=np.zeros((len(samples),len(names)+2));per_head={};prediction_values={}
    x=np.empty((len(samples),len(names)+1),float);ri=np.asarray([s[1] for s in samples]);cj=np.asarray([s[2] for s in samples])
    for k,col in enumerate(columns):x[:,k]=col[ri,cj]
    for k,h in enumerate(horizons):
        if weights[k]==0:continue
        x[:,-1]=k;c,raw=tree_contributions(model,x);np.testing.assert_allclose(raw,pred[ri,cj,k],rtol=1e-12,atol=1e-12)
        np.save(out/f'contributions_{h}d.npy',c);per_head[str(h)]=str(out/f'contributions_{h}d.npy');prediction_values[h]=raw;contrib+=weights[k]*c
    raw_blend=sum(weights[horizons.index(h)]*v for h,v in prediction_values.items());np.testing.assert_allclose(contrib.sum(axis=1),raw_blend,rtol=1e-10,atol=1e-10)
    all_names=[*names,'预测周期N','基准值'];daily=np.zeros((len(days),len(all_names)));coverage=np.zeros(len(days));absolute=np.zeros(len(names))
    stock_details=[[] for _ in days]
    for row,(d,i,j,stock) in enumerate(samples):
        weight=float(stock['open_weight']);daily[d]+=weight*contrib[row];coverage[d]+=weight;absolute+=weight*np.abs(contrib[row,:len(names)])
        top=np.argsort(-np.abs(contrib[row,:len(names)]))[:8]
        stock_details[d].append({'code':str(codes[j]),'open_weight':weight,'final_score':float(scores[i,j]),'head_ranks':ranks[i,j].tolist(),'raw_blended_score':float(raw_blend[row]),'baseline':float(contrib[row,-1]),'horizon_contribution':float(contrib[row,-2]),'other_factor_contribution':float(contrib[row,:len(names)].sum()-contrib[row,top].sum()),'bought':stock['bought'],'sold':stock['sold'],'net_pnl':stock['net_pnl'],'nav_contribution_pct':stock['nav_contribution_pct'],'top_factors':[{'name':names[t],'contribution':float(contrib[row,t]),'normalized_value':float(columns[t][i,j]),'raw_value':float(raw_columns[t][i,j])} for t in top]})
    top=np.argsort(-absolute)[:24];display_names=[names[i] for i in top];heatmap=[]
    for d in range(len(days)):
        for j,k in enumerate(top):heatmap.append([d,j,float(daily[d,k]) if coverage[d]>0 else None])
    result={'method':'LightGBM4.6 native TreeSHAP on raw regression scores; weighted by configured horizon weights then observed post-fill/open portfolio weights','scope_warning':'Explains raw prediction scores, not the nonlinear final percentile-rank fusion and not realized return causation. PnL attribution is a separate exact account reconciliation.','score_units':'raw regression score, not predicted return or probability','zero_weight_horizons':[h for h,w in zip(horizons,weights) if w==0],'horizons':horizons,'weights':weights.tolist(),'dates':[d['decision_date'] for d in days],'settlement_dates':[d['settlement_date'] for d in days],'factor_names':display_names,'heatmap':heatmap,'importance':[{'name':names[k],'mean_abs_weighted_contribution':float(absolute[k]/len(days))} for k in np.argsort(-absolute)],'coverage_open_weight':coverage.tolist(),'daily_raw_score':daily.sum(axis=1).tolist(),'daily_baseline':daily[:,-1].tolist(),'daily_task_contribution':daily[:,-2].tolist(),'daily_stocks':stock_details,'daily_pnl':days}
    result['daily_top_factors']=[{'positive':[{'name':names[k],'contribution':float(daily[d,k])} for k in np.argsort(-daily[d,:len(names)])[:5]],'negative':[{'name':names[k],'contribution':float(daily[d,k])} for k in np.argsort(daily[d,:len(names)])[:5]]} for d in range(len(days))]
    result['feature_info']=feature_descriptions(parent,names)
    np.savez_compressed(out/'all_daily_contributions.npz',contributions=daily,names=np.asarray(all_names),dates=np.asarray(result['dates']),coverage=coverage)
    np.savez_compressed(out/'sample_axes.npz',day=np.asarray([s[0] for s in samples]),row=ri,stock=cj)
    atomic_write_json(out/'explanations.json',json_value(result));assert file_sha256(parent/'models/final/model.txt')==task['model_sha256']
    atomic_write_json(out/'complete.json',{'state':'complete','samples':len(samples),'days':len(days),'factor_count':len(names),'per_head':per_head,'model_sha256':task['model_sha256'],'additivity_checked':True,'head_prediction_parity_checked':True,'outputs':{p.name:file_sha256(p) for p in out.iterdir() if p.is_file()}})
    return result
