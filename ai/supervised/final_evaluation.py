"""Observe frozen train-selected configurations without holdout feedback."""
import argparse
import csv
import gc
import html
import json
import os
from pathlib import Path
import time

if __name__=='__main__':
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from .portfolio import PreparedModelPortfolio
from .ic import summary,daily_ic
from env.forward_returns import label_split_mask
from utils.atomic_file import atomic_write_json,file_sha256
from .search import rank_trial,portfolio_methods


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def validate_frozen_selection(frozen):
    arms=frozen.get('arm_winners',[])
    expected={(h,method) for h in (1,5,10) for method in portfolio_methods(frozen)}
    if frozen.get('state')!='frozen_before_test' or len(arms)!=len(expected) or {(a['horizon'],a['method']) for a in arms}!=expected:
        raise ValueError('all configured distinct portfolio winners must be frozen before test')
    if frozen.get('selection_split')!='train' or any(a.get('selection_split')!='train' for a in arms):raise ValueError('only training-selected configurations are admissible')
    if frozen['winner']!=max(arms,key=lambda a:rank_trial({'metrics':a['train']})):
        raise ValueError('frozen recommendation does not match training selection')


def validate_saved_result(result,path,winner,split,fingerprint):
    if result['selection_sha256']!=fingerprint or result['horizon']!=winner['horizon'] or result['method']!=winner['method'] or result['split']!=split or result['parameters']!=winner['parameters']:
        raise ValueError('saved final result identity differs')
    if Path(result['trace_path']).resolve()!=path.resolve() or file_sha256(path)!=result['trace_sha256']:
        raise ValueError('saved final trace changed')
    with np.load(path,allow_pickle=False) as z:
        nav=z['nav'];returns=z['returns'];count=len(returns)
        if count<1 or nav.shape!=(count+1,) or not np.isfinite(nav).all() or np.any(nav<=0):raise ValueError('saved trace axes or NAV invalid')
        np.testing.assert_allclose(returns,nav[1:]/nav[:-1]-1,rtol=1e-10,atol=1e-12)
        for name in ('decision_dates','next_dates','cash','exposure','full_investment'):
            if len(z[name])!=count:raise ValueError('saved trace sidecar axes differ: '+name)
        cagr=np.expm1(np.log(nav[-1]/nav[0])*252/count)
        drawdown=float(np.max(1-nav/np.maximum.accumulate(nav)))
        deviation=float(np.std(returns,ddof=1)) if count>1 else 0.
        check={'transition_count':count,'total_return':nav[-1]/nav[0]-1,'annualized_return':cagr,'max_drawdown':drawdown,
            'calmar':cagr/max(drawdown,1e-6),'annualized_volatility':deviation*np.sqrt(252),
            'sharpe':np.mean(returns)/deviation*np.sqrt(252) if deviation else 0.,
            'mean_cash':float(np.mean(z['cash'])),'mean_exposure':float(np.mean(z['exposure']))}
        for key,value in check.items():
            if not np.isclose(value,result['metrics'][key],rtol=1e-8,atol=1e-10):raise ValueError('saved final metrics differ from trace: '+key)
    if split=='train':
        for key in ('calmar','annualized_return','max_drawdown'):
            if not np.isclose(result['metrics'][key],winner['train'][key],rtol=1e-10,atol=1e-12):raise ValueError('saved training result differs from frozen winner')


def load_winner_trials(root,arm):
    root=Path(root)
    ledger=Path(arm.get('trial_ledger_path',str(root/'search'/f'n{arm["horizon"]}_{arm["method"]}'/'trials.jsonl')))
    if not ledger.resolve().is_relative_to(root.resolve()):raise ValueError('trial ledger must remain inside this study')
    records=[json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()]
    if arm.get('selection_split')!='train' or any(row.get('selection_split')!='train' for row in records):raise ValueError('nontraining trial cannot enter final selection')
    valid=[row for row in records if row['state']=='complete']
    if len(records)!=arm['trials'] or len(valid)!=arm['valid_trials']:raise ValueError('final trial ledger counts differ')
    best=max(valid,key=rank_trial)
    if best['parameters']!=arm['parameters'] or best['metrics']!=arm['train']:raise ValueError('final winner differs from trial ledger')
    return records


def run(root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();started=time.monotonic()
    frozen_path=root/'search/frozen_selection.json';frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    validate_frozen_selection(frozen)
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    if portfolio_methods(protocol)!=portfolio_methods(frozen):raise ValueError('frozen strategy scope differs from protocol')
    if protocol.get('selection_split')!='train':raise ValueError('holdouts are observation-only under the current policy')
    fingerprint=file_sha256(frozen_path);stage=root/'final';stage.mkdir(exist_ok=True)
    atomic_write_json(stage/'selection_before_test.json',{'sha256':fingerprint,'selection':frozen})
    rows=[];full=[]
    for h in (1,5,10):
        arms=[x for x in frozen['arm_winners'] if x['horizon']==h]
        for split in ('train','validation','test'):
            prepared=PreparedModelPortfolio(root,h,split)
            for winner in arms:
                identifier=f'n{h}_{winner["method"]}_{split}';receipt=stage/(identifier+'.json')
                if receipt.exists():
                    result=json.loads(receipt.read_text(encoding='utf-8'))
                    if result['selection_sha256']!=fingerprint:raise ValueError('final replay selection identity changed')
                else:
                    begin=time.monotonic();trace,metrics,diagnostics=prepared.evaluate(winner['method'],winner['parameters'],details=False)
                    if split=='train':
                        for key in ('calmar','annualized_return','max_drawdown'):
                            if not np.isclose(metrics[key],winner['train'][key],rtol=1e-10,atol=1e-12):raise ValueError('training replay differs from search')
                    path=stage/(identifier+'.npz')
                    np.savez_compressed(path,decision_dates=np.asarray(trace.decision_dates),next_dates=np.asarray(trace.next_decision_dates),
                        nav=trace.nav,returns=trace.portfolio_returns,exposure=trace.exposure,cash=trace.cash,full_investment=trace.full_investment_contract)
                    result={'horizon':h,'method':winner['method'],'split':split,'parameters':winner['parameters'],'metrics':metrics,
                            'elapsed_seconds':time.monotonic()-begin,'selection_sha256':fingerprint,'trace_sha256':file_sha256(path),'trace_path':str(path)}
                    atomic_write_json(receipt,result,allow_nan=False)
                    atomic_write_json(stage/(identifier+'_policy_diagnostics.json'),diagnostics,allow_nan=False)
                validate_saved_result(result,stage/(identifier+'.npz'),winner,split,fingerprint)
                full.append(result);rows.append({'horizon':h,'method':winner['method'],'split':split,**{k:v for k,v in result['metrics'].items() if not isinstance(v,dict)},
                                                'parameters':json.dumps(winner['parameters'],sort_keys=True)})
                print({'completed':len(rows),'horizon':h,'method':winner['method'],'split':split},flush=True)
            del prepared;gc.collect()
    if file_sha256(frozen_path)!=fingerprint:raise ValueError('selection changed while testing')
    write_csv(stage/'portfolio_results.csv',rows)
    trial_rows=[]
    for arm in frozen['arm_winners']:
        for trial in load_winner_trials(root,arm):
            trial_rows.append({'horizon':arm['horizon'],'method':arm['method'],'trial':trial['trial'],'state':trial['state'],
                'phase':trial.get('phase','baseline'),'proposal_method':trial.get('proposal_method','grid' if arm['method']=='topk' else 'GA'),
                **trial['parameters'],'seconds':trial['seconds'],
                **{'train_'+k:v for k,v in trial.get('metrics',{}).items() if not isinstance(v,dict)},'error':trial.get('error','')})
    write_csv(stage/'search_trials.csv',trial_rows)
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'));dates=np.load(root/'features/dates.npy')
    model_rows=[]
    for h in (1,5,10):
        folder=root/'models'/f'n{h}';receipt=json.loads((folder/'complete.json').read_text())
        if file_sha256(folder/'prediction_daily_ic.npy')!=receipt['prediction_ic_sha256']:raise ValueError('model IC diagnostic bytes changed')
        manifest=json.loads((root/'features/data_manifest.json').read_text(encoding='utf-8'))
        for name in ('eligible.npy',f'labels_{h}d.npy'):
            if file_sha256(root/'features'/name)!=manifest[name]:raise ValueError('observation labels or universe changed')
        values=daily_ic(np.load(folder/'predictions.npy',mmap_mode='r'),np.load(root/'features'/f'labels_{h}d.npy',mmap_mode='r'),np.load(root/'features/eligible.npy',mmap_mode='r'))
        np.save(stage/f'model_{h}d_observation_ic.npy',values)
        for split in ('train','validation','test'):
            model_rows.append({'horizon':h,'split':split,'features':receipt['features'],'trees':receipt['trees'],
                               **summary(values,label_split_mask(dates,*protocol[split],h))})
    write_csv(stage/'model_prediction_ic.csv',model_rows)
    factor_rows=[];all_records=[]
    for directory in ('qlib_factors_v2','local_factors_v2','supplement_factors'):
        all_records+=json.loads((root/directory/'inventory.json').read_text(encoding='utf-8'))
    selected={h:json.loads((root/'selection'/f'selected_{h}d.json').read_text(encoding='utf-8')) for h in (1,5,10)}
    selected_names={h:{r['name'] for r in selected[h]['selected']} for h in selected}
    for record in all_records:
        row={'name':record['name'],'coverage':record.get('coverage'),'cadence':record.get('cadence','daily')}
        with np.load(record['daily_path'],allow_pickle=False) as values:
            for h in (1,5,10,20):
                overall=summary(values[str(h)]);row.update({f'full_{h}d_{k}':v for k,v in overall.items()})
                row.update({f'train_{h}d_{k}':v for k,v in record['statistics'][str(h)]['train'].items()})
        for h in (1,5,10):row[f'selected_{h}d']=record['name'] in selected_names[h]
        factor_rows.append(row)
    write_csv(stage/'all_factor_ic.csv',factor_rows)
    scope=json.loads((root/'factor_scope_inventory.json').read_text(encoding='utf-8'))
    write_csv(stage/'factor_exclusions.csv',[*scope['non_score_filters'],
        *[{**r,'reason':'Historical registry entry has no current implementation to evaluate','status':'missing_historical_code'} for r in scope['historical_registry_without_current_implementation']],
        *[{**r,'status':'non_daily_data_unavailable'} for r in scope['qlib']['non_daily_excluded']]])
    for h in (1,5,10):
        chosen=[r for r in factor_rows if r['selected_'+str(h)+'d']]
        chosen.sort(key=lambda r:-r[f'train_{h}d_rank_ic'])
        write_csv(stage/f'selected_factors_{h}d.csv',chosen)
        with np.load(root/'selection'/f'selected_{h}d_correlation.npz') as matrix:
            names=matrix['names'].astype(str);corr=matrix['correlation']
            write_csv(stage/f'correlation_{h}d.csv',[{'factor':name,**dict(zip(names,corr[i]))} for i,name in enumerate(names)])
    summary_result={'state':'results_generated_pending_final_audit','portfolio_methods':list(portfolio_methods(frozen)),'recommendation':frozen['winner'],
        'portfolio_results':full,'model_prediction_ic':model_rows,'factor_count':len(factor_rows),'selected_counts':{str(h):len(selected_names[h]) for h in selected},
        'max_selected_correlations':{str(h):selected[h]['max_absolute_mean_correlation'] for h in selected},
        'unestimated_selected_pairs':{str(h):selected[h]['unestimated_retained_pairs'] for h in selected},
        'elapsed_seconds':time.monotonic()-started,'selection_sha256':fingerprint,
        'pending_user_confirmations':['是否将研究所选方案接入实盘；当前没有启用实盘。',
            '实盘开盘信号与成交时序、资金规模、容量和市场冲击假设。',
            '是否将本地继承的退市/合并零回收和公司行动近似结算，替换为逐笔券商对账规则。'],
        'limitations':['这些历史年份和部分本地因子参数族曾被研究，不能称为全新的盲测。',
            '本轮所有因子筛选、模型训练、策略搜索与最终选型只用2005—2017训练周期；验证和测试仅原样展示，由用户判断过拟合。',
            '此前旧研究按当时要求使用过验证期做搜索；旧适应度、种群、随机数状态及优胜参数未进入本轮搜索，无法把已看过的年份重新称为盲测。',
            '测试截至2026-10-30开盘，不使用次月或当日收盘作为终点。',
            '使用开盘信息并按同一开盘价模拟成交，是研究时序假设；费用和滑点沿用本地回测。',
            '月频来源信号只在规定日期出现，不向后填充，2026年可能缺失。',
            '部分因子对的共同有效样本不足，未知相关性不能当作零相关。',
            '收益标定的预测来自未参与该模型拟合的年份，但因子名单用完整训练期筛选，并非整条流程完全交叉验证。',
            'MVO用N倍日协方差近似N日风险；候选池大小不是最终持仓数的硬上限。',
            '原生回测不设市场容量上限。训练期个别候选的巨大复利触发整数数量溢出和满仓契约失败；失败候选无适应度并排除，未放宽满仓检查或改变账户计算。完整失败原因保留在搜索明细中。',
            '按公告日期回放供应商财务档案，不等于已认证所有历史原始披露和重述版本。']}
    if portfolio_methods(frozen)==('topk',):
        summary_result['limitations']=[x for x in summary_result['limitations'] if not x.startswith(('MVO用','原生回测不设'))]
        summary_result['limitations'] += [
            '应用户补充，本次仅搜索和比较TopK；MVO已停止，已有MVO成绩不参与本次选型。',
            '仅在原生引擎成功完成的候选中按训练期选优。原生回测没有市场容量上限；巨大复利下通过执行检查不代表逐股金额计算精确。此前MVO的整数溢出证据保留在历史研究目录。']
    atomic_write_json(stage/'summary.json',summary_result,allow_nan=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,split in zip(axes,['train','validation','test']):
        for result in full:
            if result['split']!=split:continue
            with np.load(result['trace_path']) as trace:
                calendar=np.r_[trace['decision_dates'][:1],trace['next_dates']].astype('datetime64[D]')
                ax.plot(calendar,trace['nav']/trace['nav'][0],label=f'N{result["horizon"]}-{result["method"]}',linewidth=1)
        ax.set_title(split);ax.set_yscale('log');ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=30)
    axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(stage/'equity_curves.png',dpi=160);plt.close(fig)
    lines=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>开盘排名模型与组合策略</title>',
           '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#182330}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #ccd3db;text-align:right}th:first-child,td:first-child{text-align:left}h1{font-size:26px}code{white-space:pre-wrap}a{color:#1263a8}</style>',
           '<h1>开盘累计收益排名：固定模型与组合策略</h1>',
           '<p>训练2005—2017；验证2018—2021；测试2022—2026-10-30开盘。全部搜索和选型只用训练期扣费后Calmar。验证及测试仅供用户观察，不参与任何选型或过拟合判断。</p>',
           '<p>因子IC对比沿用本地2005—2025口径；正IC筛选与相关性去重只使用2005—2017。每组训练冠军固定后，再展示三段指标。</p>',
           f'<p><b>训练期所选方案：</b>N={frozen["winner"]["horizon"]}，{frozen["winner"]["method"]}；训练期扣费后 Calmar={frozen["winner"]["train"]["calmar"]:.4f}。</p>',
           '<p>参数：<code>'+html.escape(json.dumps(frozen['winner']['parameters'],ensure_ascii=False))+'</code></p>',
           '<table><tr><th>方案</th><th>训练Calmar</th><th>验证Calmar</th><th>测试Calmar</th><th>测试年化</th><th>测试回撤</th></tr>']
    for arm in frozen['arm_winners']:
        matches={r['split']:r['metrics'] for r in full if r['horizon']==arm['horizon'] and r['method']==arm['method']}
        lines.append(f'<tr><td>N={arm["horizon"]} {arm["method"]}</td><td>{matches["train"]["calmar"]:.4f}</td><td>{matches["validation"]["calmar"]:.4f}</td><td>{matches["test"]["calmar"]:.4f}</td><td>{matches["test"]["annualized_return"]:.2%}</td><td>{matches["test"]["max_drawdown"]:.2%}</td></tr>')
    lines+=['</table><p><a href="portfolio_results.csv">完整组合指标</a> · <a href="search_trials.csv">全部搜索明细</a> · <a href="all_factor_ic.csv">全部因子IC表</a> · <a href="model_prediction_ic.csv">模型预测IC</a> · <a href="factor_exclusions.csv">不可定义/不可测项目</a></p>',
            '<img src="equity_curves.png" alt="三段净值曲线" style="width:100%">']
    for h in (1,5,10):lines.append(f'<p>N={h}：{len(selected_names[h])}个因子，已估计的相关性绝对值最大{selected[h]["max_absolute_mean_correlation"]:.6f}，无法估计{selected[h]["unestimated_retained_pairs"]}对；<a href="selected_factors_{h}d.csv">因子与IC</a> · <a href="correlation_{h}d.csv">最终相关矩阵</a></p>')
    lines.append('<table><tr><th>方案</th><th>完成/总搜索次数</th><th>训练期所选参数</th></tr>')
    for arm in frozen['arm_winners']:
        lines.append(f'<tr><td>N={arm["horizon"]} {arm["method"]}</td><td>{arm["valid_trials"]}/{arm["trials"]}</td><td><code>{html.escape(json.dumps(arm["parameters"],ensure_ascii=False))}</code></td></tr>')
    lines.append('</table><p>费用沿用本地配置：双边佣金0.00854%（单笔最低0.1元）、卖出印花税0.05%、过户费0.002%、滑点0.25%；初始资金100万元。</p>')
    lines+=['<p>'+html.escape(x)+'</p>' for x in summary_result['limitations']]
    lines+=['<p>待确认：'+html.escape(x)+'</p>' for x in summary_result['pending_user_confirmations']]
    lines.append('</html>');(stage/'report.html').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
