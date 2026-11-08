"""Render an honest, self-contained overview of a completed release replay."""
import argparse
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--release-archive', type=Path, required=True)
    args = parser.parse_args()
    out = args.replay.resolve()
    repo = Path(__file__).resolve().parents[1]
    run, prediction = read(out/'complete.json'), read(out/'prediction_verification.json')
    if run['state'] != 'complete':
        raise ValueError('Replay is incomplete')
    bindings = {}
    with zipfile.ZipFile(args.release_archive) as archive:
        commit = archive.comment.decode('ascii')
        if not commit.startswith('98075c2'):
            raise ValueError('Wrong git archive commit')
        for item in archive.infolist():
            if item.is_dir():
                continue
            path = repo/item.filename
            digest = hashlib.sha256(archive.read(item)).hexdigest()
            if sha(path) != digest:
                raise ValueError('Imported release differs from Git: '+item.filename)
            bindings[item.filename] = digest
    labels = {'train':'训练期（最终模型样本内）', 'validation':'外部观察期'}
    scope = ('本报告使用冻结研究行情和已校验的284个归一化特征，不重新训练模型，也未从实盘仓库重新计算原始因子。'
             '外部观察期逐日重新预测；训练期使用摘要核验后的研究预测，并抽样17个完整截面重新预测。'
             '两个区间均重新运行交付包 targets()、冻结研究账户引擎及费用结算。'
             '这是交付复现验收，不是实盘数据适配或真实成交权限验收；生产策略尚未切换。')
    banner = '<aside style="margin:16px;padding:16px;border:1px solid #a7c3df;background:#edf5ff;color:#18354f;font:14px/1.7 sans-serif">'+html.escape(scope)+' <a href="../../report.html">返回复现核验总览</a></aside>'
    reports = {}
    for result in run['results']:
        split = result['split']
        report = out/'reports'/('ml_'+split)/'single_report.html'
        text = report.read_text(encoding='utf8')
        text = re.sub(r'<a\b[^>]*href="[^"]*\.\./static_[^"]*"[^>]*>.*?</a>', '', text, flags=re.S)
        text = text.replace('<body>', '<body>'+banner, 1)
        report.write_text(text, encoding='utf8')
        payload = report.parent/'report_payload.json.gz'
        with gzip.open(payload,'rt',encoding='utf8') as stream:
            data = json.load(stream)
        data['report_navigation'] = [x for x in data['report_navigation'] if '../ml_' in x['href']]
        data['release_verification_scope'] = scope
        with gzip.open(payload,'wt',encoding='utf8') as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        receipt = read(report.parent/'complete.json')
        receipt.update(report_sha256=sha(report), payload_sha256=sha(payload), scope_banner_added=True, missing_static_navigation_removed=True)
        (report.parent/'complete.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
        reports[split] = report.relative_to(out).as_posix()
    fig = make_subplots(rows=2, cols=1, subplot_titles=('外部观察期：净值与研究版重合', '外部观察期：回撤'), shared_xaxes=True, vertical_spacing=.12)
    with np.load(out/'validation.npz') as data:
        dates = np.r_[data['decision_dates'][:1],data['next_dates']].astype(str)
        nav = data['nav']/data['nav'][0]
        fig.add_trace(go.Scatter(x=dates,y=nav,name='交付包离线重跑',line=dict(color='#176d99')),row=1,col=1)
        fig.add_trace(go.Scatter(x=dates,y=nav,name='研究版（已逐点核验相等）',line=dict(color='#e29a3b',dash='dot')),row=1,col=1)
        fig.add_trace(go.Scatter(x=dates,y=(nav/np.maximum.accumulate(nav)-1)*100,name='回撤 %',fill='tozeroy',line=dict(color='#bf5262')),row=2,col=1)
    fig.update_layout(height=650,template='plotly_white',margin=dict(l=55,r=25,t=55,b=35),legend=dict(orientation='h',y=1.12))
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>98075c2 策略离线复现核验</title>',
           '<style>body{font:16px/1.7 system-ui,sans-serif;color:#173047;background:#f5f7fa;margin:0}main{max-width:1180px;margin:32px auto;padding:28px;background:white;border-radius:14px}h1{font-size:29px}h2{font-size:21px;margin-top:30px}.tag{color:#176d66;font-weight:700}.note{padding:18px;background:#edf5ff;border-left:4px solid #4683aa}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:12px;text-align:right;border-bottom:1px solid #dde4ea}td:first-child,th:first-child{text-align:left}a{color:#176d99}.small{font-size:14px;color:#546779}</style><main>',
           '<div class="tag">离线交付复现 · 模型未重训 · 实盘尚未切换</div><h1>新模型 + TopK 40/40/3/1</h1>',
           '<p>固定提交 <code>'+commit+'</code>；284 因子、523 棵树；3/7/15/30/60 日周期按 20/30/10/30/10% 融合；MVO 关闭。</p>',
           '<p class="note">'+scope+'</p><h2>回测结果与研究版对照</h2><p>各区间独立以 100 万元空仓起步，保留研究版费用、滑点、整手、T+1 和公司行动处理。</p>',
           '<table><tr><th>区间</th><th>年化收益</th><th>最大回撤</th><th>Sharpe</th><th>Calmar</th><th>一致性</th></tr>']
    for split in ('validation','train'):
        result = next(r for r in run['results'] if r['split']==split)
        m = result['metrics']
        with np.load(out/(split+'.npz')) as data:
            interval = str(data['decision_dates'][0])+' — '+str(data['next_dates'][-1])
        parts.append(f'<tr><td><a href="{reports[split]}">{labels[split]}</a><br><small>{interval}</small></td><td>{m["annualized_return"]:.2%}</td><td>{m["max_drawdown"]:.2%}</td><td>{m["sharpe"]:.4f}</td><td>{m["calmar"]:.4f}</td><td>逐日数组完全一致<br>{result["fills"]:,} 笔成交一致</td></tr>')
    parts += ['</table>',fig.to_html(full_html=False,include_plotlyjs=True),'<h2>核验范围</h2><ul>',
              f'<li>外部观察期重新预测 {prediction["validation_days"]:,} 个完整截面，五周期预测最大绝对差 {prediction["max_abs_prediction_difference"]:.3g}；训练期抽查17日。</li>',
              '<li>两段账户净值、收益、现金、仓位比例与日期数组逐点完全一致；成交日期、证券、方向、数量一致，价格及费用按严格浮点容差核验。</li>',
              '<li>每日交付 targets() 输出与冻结研究分配器完全一致；实际回测消费交付接口输出。原始因子和 normalize() 尚未由实盘数据重新验证。</li>',
              '<li>训练期为样本内结果；外部历史已被研究，不是新的盲测。收益按开盘到开盘计，使用研究版经济除权处理，不代表真实券商收益。</li></ul>',
              '<h2>详细报告与证据</h2><p>'+ ' · '.join(f'<a href="{reports[s]}">{labels[s]}详细报告</a>' for s in ('validation','train'))+'</p>',
              '<p><a href="validation_fills.csv">外部期逐笔成交 CSV</a> · <a href="train_fills.csv">训练期逐笔成交 CSV</a> · <a href="complete.json">回测核验结果</a> · <a href="prediction_verification.json">预测核验</a> · <a href="input_hashes.json">输入摘要</a></p>',
              '<p class="small">费用：佣金 0.00854%，最低 0.1 元；滑点 0.25%；印花税 0.05%；过户费 0.002%。均沿用冻结研究引擎的计算口径。实盘原始因子、最新数据及账户迁移仍待后续接入验收。</p></main></html>']
    (out/'report.html').write_text('\n'.join(parts),encoding='utf8')
    (out/'delivery_receipt.json').write_text(json.dumps(dict(commit=commit, release_files=bindings, overview_sha256=sha(out/'report.html'), detailed_reports={s:sha(out/p) for s,p in reports.items()}),ensure_ascii=False,indent=2),encoding='utf8')
    print(out/'report.html')


if __name__ == '__main__':
    main()
