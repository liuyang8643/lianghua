"""Re-render native reports from preserved payloads, without replay or selection."""
import gzip,html,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from matplotlib.colors import TwoSlopeNorm
from offline_data import load_runtime_slice
from ai.supervised.periods import observation_splits,interval,data_end
from testback.reportor import generate_single_report
from utils.atomic_file import atomic_write_json,file_sha256
from scripts.run_research_guarded import require_guarded_job
from delivery_inputs import admit,bind,verify,validation_2026,read

root=Path(__file__).resolve().parent
require_guarded_job();c,bindings,_=admit(root);splits=observation_splits(c)
fresh=read(root/'explain_2026/complete.json');assert fresh['state']=='complete';bind(bindings,root/'explain_2026/complete.json')
winner=json.loads((root/'search/winner.json').read_text(encoding='utf8'))['winner']['parameters']
model=json.loads((root/'models/final/complete.json').read_text(encoding='utf8'))
assert fresh['model_sha256']==model['model_sha256'] and fresh['winner_sha256']==file_sha256(root/'search/winner.json')
explain_root=root/'explain_2026/explanations';bind(bindings,explain_root/'complete.json',fresh['explanation_receipt_sha256']);exp_receipt=read(explain_root/'complete.json')
assert exp_receipt['state']=='complete' and exp_receipt['model_sha256']==model['model_sha256'] and exp_receipt['additivity_checked'] and exp_receipt['head_prediction_parity_checked']
for name,sha in exp_receipt['outputs'].items():bind(bindings,explain_root/name,sha)
explanation=read(explain_root/'explanations.json');days=validation_2026(root,bindings)
assert explanation['daily_pnl']==days and explanation['dates']==[d['decision_date'] for d in days] and explanation['settlement_dates']==[d['settlement_date'] for d in days]
assert explanation['weights']==(np.asarray(winner['weights'],float)/10).tolist()
bind(bindings,root/'final/delivery_comparison.json');comparison=read(root/'final/delivery_comparison.json')
assert comparison['state']=='exported_observations_only';verify(comparison['inputs'])
for path,sha in comparison['inputs'].items():bind(bindings,path,sha)
capacity=read(root/'extreme_audit/complete.json');assert capacity['state']=='completed_posthoc_diagnostics_not_execution_certification';bind(bindings,root/'extreme_audit/complete.json')
capacity_by_case={(x['strategy'],x['split']):x['capacity'] for x in capacity['results']}
cap_ml=capacity_by_case['ml','train'];cap_static=capacity_by_case['static','train'];cap_oof=capacity_by_case['static','oof']
capacity_note=f"已观察到容量假设缺口：完整样本内回放中，模型有{cap_ml['fills_above_full_day_volume']}笔、静态有{cap_static['fills_above_full_day_volume']}笔模拟成交量超过历史全日成交量；静态OOF同段另有{cap_oof['fills_above_full_day_volume']}笔。长期复利放大后的样本内高收益不能视为可实盘复制。外验两方案的单笔超全日量/额计数均为0，仍不证明集合竞价可按开盘价成交；原始收益未按后验量额过滤或修改。"
parent_note='因月度成交量单位问题停止的20261107父试验保留8个完整折和未完成的fold2019，已封存，不作为本次交付，也未复用其模型。图中的旧冻结对照是更早的2005—2017训练模型在修正数据上的回放，不是这个停止的父试验。'
denominator_note='SHAP汇总权重＝成交后持股数量×T日开盘价÷T日成交前NAV；未按成交后NAV或可解释仓位重新归一化。它解释原始评分，不是最终排名或收益因果。'
for path in [Path(__file__),Path('testback/reportor/report.py'),Path('testback/reportor/templates/single_report_module.js'),Path('testback/reportor/templates/single_report_styles.css')]:bind(bindings,path)
for strategy in ('ml','static'):
    for split in splits:
        folder=root/'reports'/f'{strategy}_{split}';revision=folder/'presentation_revision001';revision.mkdir(exist_ok=True)
        for name in ('single_report.html','report_payload.json.gz','complete.json'):
            if not (revision/name).exists():shutil.copy2(folder/name,revision/name)
        receipt=read(revision/'complete.json');assert receipt['state']=='complete'
        bind(bindings,revision/'complete.json');bind(bindings,revision/'single_report.html',receipt['report_sha256']);bind(bindings,revision/'report_payload.json.gz',receipt['payload_sha256']);bind(bindings,folder/'position_ledger.json',receipt['ledger_sha256'])
saved=root/'final/presentation_revision001';saved.mkdir(exist_ok=True)
if not (saved/'report.html').exists():shutil.copy2(root/'final/report.html',saved/'report.html')
bind(bindings,saved/'report.html');verify(bindings);atomic_write_json(root/'presentation_inputs.json',{'state':'admitted_before_render','inputs':bindings})
font_path=Path('C:/Windows/Fonts/msyh.ttc');font=FontProperties(fname=str(font_path)) if font_path.exists() else FontProperties()
plt.rcParams['axes.unicode_minus']=False

def label(strategy,split):
    names={'oof':'OOF·已用于选参','fit':'拟合段','internal_validation':'内部诊断段','train':'完整训练段','validation':'外验·原测试段','test':'测试观察段'}
    if strategy=='static' and split=='oof':names['oof']='OOF同区间对照·参数冻结'
    suffix='（最终模型样本内）' if strategy=='ml' and split in ('fit','internal_validation','train') else ''
    dates=interval(c,split);years=dates[0][:4] if dates[0][:4]==dates[1][:4] else dates[0][:4]+'—'+dates[1][:4]
    return ('多周期模型' if strategy=='ml' else '原静态策略')+' · '+names[split]+suffix+' · '+years

fig,axes=plt.subplots(2,2,figsize=(16,10),layout='constrained')
for column,split in enumerate(('oof','validation')):
    paths=[(root/'final'/f'{split}.npz','新多周期方案','#2563eb','-'),(root/'static_baseline'/f'{split}.npz','原静态策略','#d97706','-')]
    if split=='validation':paths.append((Path('artifacts/research/multihorizon_all_periods_preclose_20261107/results/ml_test.npz'),'旧模型（训练至2017，修正数据）','#64748b','--'))
    for path,name,color,style in paths:
        with np.load(path) as z:
            dates=np.r_[z['decision_dates'][:1],z['next_dates']].astype('datetime64[D]');nav=z['nav']
            axes[0,column].plot(dates,nav/nav[0],label=name,color=color,ls=style,lw=1.5)
            axes[1,column].plot(dates,100*(nav/np.maximum.accumulate(nav)-1),label=name,color=color,ls=style,lw=1.2)
    axes[0,column].set_yscale('log');axes[0,column].set_title('模型：连续OOF选参；静态：同区间冻结对照' if split=='oof' else '外部观察（历史已查看，非新盲测）',fontproperties=font)
    axes[0,column].set_ylabel('起点=1；对数净值',fontproperties=font);axes[1,column].set_ylabel('回撤 %',fontproperties=font)
    axes[0,column].legend(prop=font)
    for row in range(2):
        ax=axes[row,column];ax.grid(alpha=.2);loc=mdates.AutoDateLocator(minticks=4,maxticks=7);ax.xaxis.set_major_locator(loc);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
equity=root/'final/equity_comparison.png';fig.savefig(equity,dpi=140);plt.close(fig)

days=explanation['daily_pnl'];nav=np.asarray([days[0]['pretrade_nav'],*[d['ending_nav'] for d in days]])
dates=np.asarray([days[0]['decision_date'],*[d['settlement_date'] for d in days]],dtype='datetime64[D]')
with np.load(root/'final/validation.npz') as z:
    lookup={str(d):i for i,d in enumerate(z['decision_dates'])};rr=np.asarray([lookup[d['decision_date']] for d in days])
    np.testing.assert_array_equal(nav[1:],z['nav'][rr+1]);assert nav[0]==z['nav'][rr[0]]
fig=plt.figure(figsize=(18,14),layout='constrained');grid=fig.add_gridspec(3,1,height_ratios=[2,2,4]);a=fig.add_subplot(grid[0]);b=fig.add_subplot(grid[1]);heat=fig.add_subplot(grid[2])
a.plot(dates,nav/nav[0],color='#2563eb');a.set_title('新模型：2022起始连续账户的2026决策片段（不是2026现金重开）',fontproperties=font);a.set_ylabel('片段起点净值=1',fontproperties=font);a.grid(alpha=.2)
top=explanation['importance'][:8];b.barh([x['name'] for x in top][::-1],[x['mean_abs_weighted_contribution'] for x in top][::-1],color='#2563eb');b.tick_params(axis='y',labelsize=8);b.set_title('平均绝对持仓评分贡献（不是收益贡献）',fontproperties=font);b.grid(axis='x',alpha=.2)
matrix=np.full((len(explanation['factor_names']),len(days)),np.nan)
for x,y,value in explanation['heatmap']:
    if value is not None:matrix[y,x]=value
bound=max(1e-8,float(np.nanmax(np.abs(matrix))));im=heat.imshow(matrix,aspect='auto',cmap='RdBu',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),interpolation='nearest')
heat.set_yticks(range(len(explanation['factor_names'])),explanation['factor_names'],fontsize=8);ticks=np.unique(np.linspace(0,len(days)-1,12,dtype=int));heat.set_xticks(ticks,[explanation['dates'][i] for i in ticks],rotation=35,ha='right',fontsize=9)
heat.set_title('决策日因子贡献：蓝色抬高原始评分，红色压低；收益在下一开盘估值',fontproperties=font);fig.colorbar(im,ax=heat,shrink=.75)
heatmap=root/'explain_2026/explanation_overview.png';fig.savefig(heatmap,dpi=140);plt.close(fig)

active=json.loads((root/'active_data.json').read_text(encoding='utf8'));rt=load_runtime_slice(active['runtime_path'],c['train'][0],data_end(c),preload_rows=100001,cache_dir=Path(c['feature_runtime_cache']))
assert rt.manifest.source_sha256==active['runtime_sha256'] and Path(rt.manifest.source_path).resolve()==Path(active['runtime_path']).resolve()
records=[]
for strategy in ('ml','static'):
    for split in splits:
        out=root/'reports'/f'{strategy}_{split}';revision=out/'presentation_revision001';revision.mkdir(exist_ok=True)
        for name in ('single_report.html','report_payload.json.gz','complete.json'):
            if not (revision/name).exists():shutil.copy2(out/name,revision/name)
        original=json.loads((revision/'complete.json').read_text(encoding='utf8'))
        assert file_sha256(revision/'single_report.html')==original['report_sha256']
        assert file_sha256(revision/'report_payload.json.gz')==original['payload_sha256']
        with gzip.open(revision/'report_payload.json.gz','rt',encoding='utf8') as f:data=json.load(f)
        data['report_label']=label(strategy,split);data['report_navigation']=[{'label':label(s,n),'href':f'../{s}_{n}/single_report.html','current':s==strategy and n==split} for s in ('ml','static') for n in splits]
        if strategy=='ml' and split=='validation':
            data['explainability']=explanation
            data['strategy_explanation']={'description':f"共享LightGBM：{model['feature_count']}个因子加周期标识，{model['trees']}棵树；3/7/15/30/60日百分位融合权重依次为{[10*w for w in winner['weights']]}%。前{winner['buy_n']}名提供买入候选，跌出前{winner['sell_m']}名才进入卖出条件；每次买入预算为当时总资产/{winner['sell_m']}，允许现金，不补仓，最短持仓{winner['hold_thresh']}个交易日，每日最多主动换出{winner['max_replacements']}只。",'timing_note':'沿用项目同开盘成交、费用和权益再投资近似；尚未验证竞价排队和真实容量。2022年之后只供本轮观察，不用于训练或选参。','date_note':f"此解释覆盖{days[0]['decision_date']}至{days[-1]['decision_date']}的决策，最后估值日{days[-1]['settlement_date']}。账户从2022年连续运行。展示真实标准化模型输入；未保存的原始因子值显示为—，不补造原始值。"}
            data['audit_summary']={**json.loads((root/'regression/ml_validation/trade_audit.json').read_text(encoding='utf8')),'causality_note':'本轮训练及搜索只使用训练区间；因子时间契约和抽样检查的范围见独立审核。保留历史数据版本及同开盘成交假设限制。'}
            data['strategy_explanation']['date_note']+=' '+denominator_note
            data['audit_summary']['causality_note']+=' '+capacity_note+' '+parent_note
            data['static_explanation_image']='../../explain_2026/explanation_overview.png'
        with gzip.open(out/'report_payload.json.gz','wt',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        hi=int(np.searchsorted(rt.trade_dates,np.datetime64(data['period']['end']),side='right'));data['_runtime_kline']={'trade_dates':rt.trade_dates[:hi],'stock_codes':np.asarray(rt.stock_codes),**{k:rt.field(k)[:hi] for k in ('open','high','low','close','amount')}}
        prior_timestamped=set(out.glob('backtest_report_*.html'))
        path=generate_single_report(data,out);text=path.read_text(encoding='utf8');text=text.replace('alt="修复前后净值、因子重要性和每日贡献热力图"','alt="新模型2026账户片段、评分贡献与热力图"')
        if strategy=='ml' and split=='validation':text=text.replace('data-tab="explanation">模型解释</button>','data-tab="explanation">2026模型解释</button>')
        path.write_text(text,encoding='utf8')
        for generated in set(out.glob('backtest_report_*.html'))-prior_timestamped:generated.write_text(text,encoding='utf8')
        assert file_sha256(out/'position_ledger.json')==original['ledger_sha256']
        receipt={**original,'report_sha256':file_sha256(path),'payload_sha256':file_sha256(out/'report_payload.json.gz'),'original_report_sha256':original['report_sha256'],'presentation_only_revision':True,'label':data['report_label']}
        atomic_write_json(out/'complete.json',receipt);records.append({'strategy':strategy,'split':split,'report_sha256':receipt['report_sha256'],'ledger_unchanged':True});print({'report_rendered':strategy+'_'+split},flush=True)

summary=root/'final/report.html'
text=(saved/'report.html').read_text(encoding='utf8')
text=text.replace('<th>平均仓位</th>','<th>平均成交后仓位</th>')
old='本次发现早期极端成交额污染滚动Amihud因子的数值问题，已重建3个基因子与6个窗口变体。旧模型使用过受影响window30输入，因此旧ML/静态成绩不再作为无已知缺陷基准；旧记录保留，当前静态使用相同原配置重新计算。'
note='此前的Amihud窗口数值与preClose修复已沿用。本轮另修复月度低换手因子的成交量股/手混用，重建受影响输入并完整重训。旧修正模型未使用该月度因子，仍可在相同外部日期作对照；静态使用原配置和相同数据。'
assert old in text or note in text;text=text.replace(old,note)
table='<h2>最终同区间对照</h2><p>每格：扣费年化 / 最大回撤 / Calmar。外验是原测试段，本轮没有新的独立测试集。原始指标中的平均仓位按成交后计算，详细报告的卡片和图线按下一开盘估值计算，两个口径分别保留。</p><table><tr><th>区间与用途</th><th>新多周期方案</th><th>原静态策略</th></tr>'
for row in comparison['rows']:table+='<tr><td>'+html.escape(row['role'])+'</td><td>'+row['ml_cell']+'</td><td>'+row['static_cell']+'</td></tr>'
table+='</table><p><a href="../reports/ml_validation/single_report.html">打开原版多Tab外验详细报告（含2026模型解释）</a></p><p>'+html.escape(capacity_note)+'</p><p><a href="../extreme_audit/complete.json">查看实际极值与容量诊断记录</a></p><p>'+html.escape(parent_note)+'</p><p>'+html.escape(denominator_note)+'</p><img src="equity_comparison.png" alt="新方案、原静态及旧冻结模型的净值和回撤对照">'
text=text.replace('</h1>','</h1>'+table,1)
for split in splits:
    text=text.replace('<td>'+split+'</td>','<td>'+html.escape(label('ml',split))+'</td>')
    for name,strategy in [('当前多周期','ml'),('本地静态','static')]:text=text.replace('<td>'+split+'/'+name+'</td>','<td>'+html.escape(label(strategy,split))+'</td>')
summary.write_text(text,encoding='utf8')
verify(bindings)
atomic_write_json(root/'presentation_complete.json',{'state':'complete','reports':records,'no_model_training':True,'no_strategy_search':True,'no_financial_replay':True,'primary_report':'reports/ml_validation/single_report.html','model_sha256':model['model_sha256'],'winner_sha256':file_sha256(root/'search/winner.json'),'summary_sha256':file_sha256(summary),'equity_comparison_png_sha256':file_sha256(equity),'explanation_png_sha256':file_sha256(heatmap),'inputs':bindings,'source_sha256':{str(p):file_sha256(p) for p in [Path(__file__),Path('testback/reportor/report.py'),Path('testback/reportor/templates/single_report_module.js'),Path('testback/reportor/templates/single_report_styles.css')]},'browser_rendered':False,'standalone_pngs_require_visual_inspection':True})
print({'state':'complete','reports':len(records)},flush=True)
