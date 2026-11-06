"""Saved-result comparison for a fixed N5 MVO arm, with explicit policy differences."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .final_evaluation import write_csv
from utils.atomic_file import atomic_write_json,file_sha256


def render(root:Path):
    out=root/'evaluation';run=json.loads((out/'complete.json').read_text());assert run['state']=='complete'
    frozen=json.loads((out/'frozen.json').read_text());assert file_sha256(out/'frozen.json')==run['frozen_sha256']
    parent=Path(frozen['parent']);original=json.loads((parent/'final/complete.json').read_text())['results']
    static=json.loads((parent/'static_comparison/complete.json').read_text())['static_results']
    labels={'fit':'2005–2016 拟合期','internal_validation':'2017 内验','train':'2005–2017 原训练期整体',
        'validation':'2018–2021 外验','test':'2022–2026-10-30 测试'}
    names={'score_mvo':'固定N5＋MVO','rank_buffer':'原N5排名缓冲','static':'当前本地静态'}
    rows=[];paths={}
    for split in labels:
        for name,r,path in [('score_mvo',next(x for x in run['results'] if x['split']==split),out/f'{split}.npz'),
            ('rank_buffer',next(x for x in original if x['horizon']==5 and x['split']==split),parent/'final'/f'n5_{split}.npz'),
            ('static',next(x for x in static if x['split']==split),parent/'static_comparison'/f'static_{split}.npz')]:
            assert file_sha256(path)==r['trace_sha256'];m=r['metrics']
            with np.load(path) as z:
                nav=z['nav'];ret=z['returns'];np.testing.assert_allclose(ret,nav[1:]/nav[:-1]-1,rtol=1e-10,atol=1e-12)
                annual=np.expm1(np.log(nav[-1]/nav[0])*252/len(ret));dd=np.max(1-nav/np.maximum.accumulate(nav))
                for k,v in [('annualized_return',annual),('max_drawdown',dd),('calmar',annual/max(dd,1e-6))]:
                    assert np.isclose(m[k],v,rtol=1e-10,atol=1e-12)
            paths[(name,split)]=path;rows.append(dict(strategy=name,split=split,**m))
    write_csv(out/'comparison.csv',rows)
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定N5模型：MVO对照</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1300px;margin:30px auto;padding:0 18px;color:#172333}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.new{background:#eaf4ff}aside{background:#f5f5f5;padding:15px}img{width:100%}a{color:#0866b8}</style>',
        '<h1>同一N=5监督模型：固定MVO vs 原排名缓冲</h1>',
        '<p>复用原模型、预测、修复数据、费用/滑点及100万元起始资金；没有重新训练、标签校准或参数搜索。每个区间独立起始。最早训练期252日仅作性能和正确性检查。</p>',
        '<p>MVO：当前可买预测前50名＋全部实际持仓；120日已知开盘到开盘经济收益估计风险，50%向对角阵收缩，并乘以5近似5日协方差。风险厌恶10，L2正则0.001。</p>',
        '<p>收益信号是候选分数减均值、除标准差，再乘风险矩阵对角均值的平方根。这是风险尺度匹配的相对alpha，不是校准后的预期收益百分比，也没有利用2017或外验/测试未来收益拟合转换。</p>',
        '<p>约束：只做多、优化目标满仓；新增/增加仓位目标上限5%，已有因漂移或锁仓超限的权重允许维持但不再增加。每日目标权重L1变化上限10%；首次建仓及现金补足采用明确的最低资金投入额度。合法投资容量不足时保留实际持仓与现金，不放宽单股上界或求解约束。50是候选池数量，不是持仓硬上限。</p>',
        '<aside>原排名缓冲为buy20/sell50、至少持有7日、每日最多换出2只、新仓上限NAV/50。MVO替换整套组合映射，未保留7日和2只整数规则；这是同模型不同组合方法对照，不是只改变权重的严格单变量试验。MVO参数未调优，不代表最优MVO。</aside>',
        '<p>沿用本地1%调整带、整手/T+1/锁仓/费用及经济结算；目标换手不等于实际成交换手。无流动性前500限制。保留同开盘信号/成交理想化假设，尚未验证竞价排队及成交容量；既有历史反复研究，不是新盲测。拟合期为样本内回放，2017用于模型早停。</p>']
    for split,label in labels.items():
        parts.append(f'<h2>{label}</h2><table><tr><th>方案</th><th>扣费年化</th><th>最大回撤</th><th>Calmar</th><th>Sharpe</th><th>平均仓位</th><th>日均双边换手</th></tr>')
        for r in (r for r in rows if r['split']==split):
            style=' class="new"' if r['strategy']=='score_mvo' else ''
            parts.append(f'<tr{style}><td>{names[r["strategy"]]}</td><td>{r["annualized_return"]:.2%}</td><td>{r["max_drawdown"]:.2%}</td><td>{r["calmar"]:.3f}</td><td>{r["sharpe"]:.3f}</td><td>{r["mean_exposure"]:.2%}</td><td>{r["mean_gross_turnover"]:.2%}</td></tr>')
        r=next(r for r in run['results'] if r['split']==split)
        parts.append(f'</table><p>MVO平均/最多持仓：{r["metrics"]["mean_positions"]:.1f}/{r["metrics"]["max_positions"]}只；最多候选{r["max_candidates"]}只；本段总回放{r["seconds"]:.2f}秒，其中信号/求解{r["solver_seconds"]:.2f}秒；资金投入额度超过10%的日期{r["funding_floor_days"]}个。</p>')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    for ax,split in zip(axes.flat,['fit','internal_validation','validation','test']):
        for name,legend,color,style in [('score_mvo','N5 MVO','#1678c2','-'),('rank_buffer','N5 rank buffer','#ef8c23','-'),('static','Local static','black','--')]:
            with np.load(paths[(name,split)]) as z:
                dates=np.r_[z['decision_dates'][:1],z['next_dates']].astype('datetime64[D]')
                ax.plot(dates,z['nav']/z['nav'][0],label=legend,color=color,linestyle=style,linewidth=1.2)
        ax.set_title(split);ax.set_yscale('log');ax.set_xlim(dates[0],dates[-1]);ax.grid(alpha=.2);ax.legend(fontsize=9)
        ax.set_ylabel('Net NAV / initial NAV (log)');ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=6));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    parts+=['<h2>净值对照</h2><img src="comparison.png" alt="N5 MVO与原配置净值对照">',
        '<p><a href="comparison.csv">完整对照CSV</a> · <a href="frozen.json">固定参数</a> · <a href="../independent_review_v2.json">最终独立复核</a> · <a href="../independent_review.json">首次边界问题记录</a></p>',
        '<p>独审发现无合法候选且全现金时应持现金而不是报错，现已补齐该分支及无可观察持仓信号的保留逻辑。原运行8453个交易步全部进入正常Solved分支，未触发这些边界；原执行源码已归档。修复仅影响未触发的持现金/持仓分支，未重训、未搜索，也未修改已存回测结果。</p>',
        '<p>新增：风险尺度分数MVO策略、研究/报告脚本及测试。复用：原N5预测、风险估计器、稠密与因子化等价QP、唯一账户/成交链。替换/删除：无。依赖变化：无。领域schema变化：无。冗余检查：无第二套求解器或回测引擎。原有未提交依赖保留，当前静态配置不变。</p></html>']
    (out/'report.html').write_text('\n'.join(parts),encoding='utf8')
    atomic_write_json(out/'report_receipt.json',{'state':'rendered','rows':len(rows),'report_sha256':file_sha256(out/'report.html'),
        'image_sha256':file_sha256(out/'comparison.png'),'results_sha256':file_sha256(out/'complete.json'),'html_browser_rendered':False})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);render(ap.parse_args().root)
