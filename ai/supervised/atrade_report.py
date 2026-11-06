"""Render the frozen ATrade adaptation results without tuning any parameters."""
from __future__ import annotations
import argparse,html,json,os
from pathlib import Path
import numpy as np
from .final_evaluation import write_csv
from utils.atomic_file import atomic_write_json,file_sha256


def render(root:Path):
    out=root/'evaluation';run=json.loads((out/'complete.json').read_text());assert run['state']=='complete'
    frozen=json.loads((out/'frozen.json').read_text());assert file_sha256(out/'frozen.json')==run['frozen_sha256']
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'));parent=Path(protocol['study_parent'])
    reference=Path(protocol['liquidity_reference_root']) if protocol.get('liquidity_reference_root') else None
    old=json.loads((parent/'final/complete.json').read_text())['results']
    static=json.loads((parent/'static_comparison/complete.json').read_text())['static_results']
    labels={'fit':'2005–2016 拟合期','internal_validation':'2017 内验（早停）','train':'2005–2017 原训练期整体',
            'validation':'2018–2021 外验','test':'2022–2026-10-30 测试'}
    names={'book5':'5日子组合','book20':'20日子组合','book60':'60日子组合','anchor':'三周期合并（不择时）',
           'timed':'三周期合并＋分散度择时','n5_default':'原N5：20/50/7','static':'当前本地静态',
           'top500_timed':'前500基准＋择时','top500_anchor':'前500基准不择时'}
    all_results=list(run['results']);paths={}
    for r in run['results']:paths[(r['strategy'],r['split'])]=out/f'{r["strategy"]}_{r["split"]}.npz'
    if reference:
        previous=json.loads((reference/'evaluation/complete.json').read_text())
        before=json.loads((reference/'evaluation/frozen.json').read_text())
        assert file_sha256(reference/'evaluation/frozen.json')==previous['frozen_sha256']
        assert {k:v for k,v in before['strategy'].items() if k!='liquidity_top_n'}=={k:v for k,v in frozen['strategy'].items() if k!='liquidity_top_n'}
        assert before['model_receipts']==frozen['model_receipts'] and before['fees']==frozen['fees']
        for r in previous['results']:
            if r['strategy'] in ('anchor','timed'):
                name='top500_'+r['strategy'];all_results.append({**r,'strategy':name})
                paths[(name,r['split'])]=reference/'evaluation'/f'{r["strategy"]}_{r["split"]}.npz'
    for split in labels:
        for name,r,path in [('n5_default',next(x for x in old if x['horizon']==5 and x['split']==split),parent/'final'/f'n5_{split}.npz'),
                            ('static',next(x for x in static if x['split']==split),parent/'static_comparison'/f'static_{split}.npz')]:
            all_results.append({**r,'strategy':name});paths[(name,split)]=path
    rows=[]
    for r in all_results:
        path=paths[(r['strategy'],r['split'])];assert file_sha256(path)==r['trace_sha256']
        with np.load(path) as z:
            nav=z['nav'];ret=z['returns'];np.testing.assert_allclose(ret,nav[1:]/nav[:-1]-1,rtol=1e-10,atol=1e-12)
            annual=float(np.expm1(np.log(nav[-1]/nav[0])*252/len(ret)));dd=float(np.max(1-nav/np.maximum.accumulate(nav)))
            for k,v in [('annualized_return',annual),('max_drawdown',dd),('calmar',annual/max(dd,1e-6))]:
                assert np.isclose(r['metrics'][k],v,rtol=1e-10,atol=1e-12)
        rows.append({'strategy':r['strategy'],'split':r['split'],**r['metrics']})
    write_csv(out/'comparison.csv',rows)
    n60_root=Path(frozen['model_sources']['60'])
    model=json.loads((n60_root/'models/n60/complete.json').read_text());audit=json.loads((n60_root/'models/readback_audit.json').read_text())
    audit_link=os.path.relpath(n60_root/'models/readback_audit.json',out).replace('\\','/')
    assert audit['state']=='pass'
    top_n=frozen['strategy']['liquidity_top_n']
    pool_text=(f'采用过去20个已完成交易日均成交额前{top_n}名' if top_n is not None else '取消成交额排名与其20日历史门槛，使用本地PIT预测池')
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ATrade三周期组合本地适配</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1300px;margin:30px auto;padding:0 18px;color:#172333}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.new{background:#eaf4ff}aside{background:#f5f5f5;padding:15px}img{width:100%}a{color:#0866b8}</style>',
        '<h1>ATrade三周期组合：本地模型与执行适配</h1>',
        '<p>参考ATrade提交03bda3a。分支 codex/atrade-three-horizon-20261106。不是上游原始收益复现：保留本地262因子、开盘信号、固定训练方式、费用和账户引擎。</p>',
        '<p>5/20/60日模型 → 各自实际模拟账户 → 目标30只、排名60缓冲 → 逆20日波动配权、单股目标上限5% → 三组目标权重各1/3 → 预测分散度过去250日百分位映射0.5–1仓位缩放 → 每5交易日调仓。</p>',
        f'<p>股票池：{pool_text}，原有上市状态、ST、限价等交易检查仍保留。选股和分散度计算使用同一预测池。模型训练股票池不变；每区间重新从100万元起始。</p>',
        '<p>若包含前500基准，唯一配置差异是流动性股票池开关；三个模型、费用与其他组合参数逐项一致。股票池扩展也会改变预测分散度与相应仓位，但映射公式不变。</p>',
        '<p>历史窗口修订：流动性、波动率和分散度百分位先在完整预测历史计算，再按账户区间切片。内验、外验、测试起点均有完整250个分散度值。初版252日局部预载仅留下233个有效值，旧结果和失败复核保留，当前为修正后结果；模型和组合参数未变。</p>',
        f'<p>模型来源：本地冻结N5/N20以及前序补训N60（{model["trees"]}棵树，拟合样本{model["training_rows"]:,}，内验样本{model["internal_validation_rows"]:,}，原训练{model["fit_seconds"]:.2f}秒）。2005–2016拟合，2017早停，无回填重训，60日标签终点不跨分段。本次取消流动性限制不重训模型、不搜索参数。</p>',
        '<aside>上游每月滚动训练、166因子、5日头部加权与次日开盘信号未移植；本次测试的是其组合规则接入本地冻结模型。当前本地费用/滑点及同开盘成交假设保留，未验证竞价排队或容量。2017既用于模型早停，也已用于历史因子研究；外验/测试曾被查看，不是新盲测。训练期为样本内回放。</aside>',
        '<p>为了匹配本地唯一执行引擎：沿用本地1%调整带、板块整手规则、T+1、停牌/限价与经济结算；不使用上游5000元调整门槛或参与成交量上限。锁仓按当日实际市值保留，只在需要时缩小可调仓目标；实际持仓数、单股权重和总仓位可因锁仓、漂移、余款与原始权重截断偏离目标。0.5–1是缩放系数，不是实际仓位保证。</p>']
    for split,label in labels.items():
        parts.append(f'<h2>{label}</h2><table><tr><th>方案</th><th>扣费年化</th><th>最大回撤</th><th>Calmar</th><th>Sharpe</th><th>平均仓位</th><th>日均双边换手</th></tr>')
        for name in ['timed','anchor']+(['top500_timed','top500_anchor'] if reference else [])+['n5_default','static','book5','book20','book60']:
            r=next(x for x in rows if x['strategy']==name and x['split']==split);style=' class="new"' if name=='timed' else ''
            parts.append(f'<tr{style}><td>{names[name]}</td><td>{r["annualized_return"]:.2%}</td><td>{r["max_drawdown"]:.2%}</td><td>{r["calmar"]:.3f}</td><td>{r["sharpe"]:.3f}</td><td>{r["mean_exposure"]:.2%}</td><td>{r["mean_gross_turnover"]:.2%}</td></tr>')
        parts.append('</table>')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    for ax,split in zip(axes.flat,['fit','internal_validation','validation','test']):
        curves=[('timed','Current timed','#1678c2','-'),('anchor','Current anchor','#25a65b','-'),('n5_default','N5 original','#ed8e29','-'),('static','Local static','black','--')]
        if reference:curves += [('top500_timed','Top500 timed','#925bb5',':'),('top500_anchor','Top500 anchor','#db6060',':')]
        for name,legend,color,style in curves:
            with np.load(paths[(name,split)]) as z:
                dates=np.r_[z['decision_dates'][:1],z['next_dates']].astype('datetime64[D]')
                ax.plot(dates,z['nav']/z['nav'][0],label=legend,color=color,linestyle=style,linewidth=1.2)
        ax.set_title(split);ax.set_yscale('log');ax.set_xlim(dates[0],dates[-1]);ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.set_ylabel('Net NAV / initial NAV (log)');ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=6));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    parts+=['<h2>净值对照</h2><img src="comparison.png" alt="三周期与旧方案净值对照">',
        f'<p><a href="comparison.csv">{len(rows)}组完整指标CSV</a> · <a href="frozen.json">冻结参数与适配说明</a> · <a href="{audit_link}">N60模型核验</a> · <a href="../independent_review.json">独立复核</a></p>',
        '<p>新增：ATrade组合规则、研究装配、N60准备与报告脚本及测试。复用：原监督模型训练器、因子、PIT数据、交易合法性、账户、费用和唯一回测引擎。替换/删除：无。依赖变化：无。领域schema变化：无；仅研究ActionSchema标识新建。冗余检查：无第二套账户或成交引擎。</p>',
        '<p>工程与交付验收见独立复核；当前本地静态配置未更改。新分支只包含本次新增代码，依赖已有工作区监督学习及回测改动，未将原有未提交文件混入提交。</p></html>']
    (out/'report.html').write_text('\n'.join(parts),encoding='utf8')
    atomic_write_json(out/'report_receipt.json',{'state':'rendered','rows':len(rows),'report_sha256':file_sha256(out/'report.html'),
        'image_sha256':file_sha256(out/'comparison.png'),'results_sha256':file_sha256(out/'complete.json'),'html_browser_rendered':False})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);render(ap.parse_args().root)
