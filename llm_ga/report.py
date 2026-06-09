"""因子进化结果报告：生成可读 HTML 榜单 + 谱系，并打印控制台榜单。

默认只用库中已存的训练期指标（fitness）。可选 --oos-start/--oos-end：对每个 passed 因子在样本外
区间重测，附加 OOS 夏普列，用于人工判断过拟合（IS 高 / OOS 塌 = 过拟合，对照论文 p-hacking 分析）。

用法:
  uv run python -m llm_ga.report
  uv run python -m llm_ga.report --oos-start 20190101 --oos-end 20211231
"""
import argparse
import html
import json
import webbrowser
from collections import defaultdict
from pathlib import Path

from factor_db import db, records, similarity
from llm_ga.config import REPO_ROOT

_OUT = REPO_ROOT / 'factor_db' / 'report.html'


def _fmt(v, nd=3):
    return '' if v is None else f'{v:.{nd}f}'


def _sharpe_color(v):
    if v is None:
        return '#888'
    return '#1a7f37' if v > 0 else '#cf222e'


def _compute_oos(factors, oos_start, oos_end):
    from llm_ga import evaluator
    dates, stocks = evaluator.build_universe(oos_start, oos_end)
    oos = {}
    for f in factors:
        if f['status'] != 'passed':
            continue
        path = REPO_ROOT / f['file_path']
        if not path.exists():
            continue
        try:
            cls = evaluator.load_factor_class(path, f['name'])
            m = evaluator.evaluate(cls, f['name'], dates, stocks)
            oos[f['name']] = m['sharpe']
        except Exception:
            oos[f['name']] = None
    return oos


def _gen_summary(factors):
    by_gen = defaultdict(list)
    for f in factors:
        by_gen[f['generation']].append(f)
    rows = []
    for gen in sorted(by_gen):
        fs = by_gen[gen]
        sharpes = [f['train_sharpe'] for f in fs if f['train_sharpe'] is not None]
        best = max(sharpes) if sharpes else None
        passed = sum(1 for f in fs if f['status'] == 'passed')
        rows.append((gen, len(fs), passed, best))
    return rows


def _merge_run_metrics(factors, runs_by_name):
    """榜单展示指标统一取 factor_runs 中训练区间的最新回测（当前正确结果）。

    GA 因子 db 表里的 train_sharpe 是创建时冻结的 fitness（GA 选择仍读 db，不受影响），
    但榜单展示一律以最新重测为准，保证看到的都是当前合法性口径下的正确指标。
    无 run 的因子回退 db 原值。返回拷贝，不改 db。
    """
    merged = []
    for f in factors:
        f = dict(f)
        run = runs_by_name.get(f['name'])
        f['has_run'] = run is not None
        if run is not None:
            f['train_sharpe'] = run.get('sharpe')
            f['annualized'] = run.get('annualized')
            f['max_dd'] = run.get('max_dd')
            f['n_trades'] = run.get('n_trades')
        merged.append(f)
    return merged


def _matrix_color(v):
    """相关/重叠值 → 背景色：高(红)→低(绿)，None 灰。"""
    if v is None:
        return '#f0f0f0'
    v = max(-1.0, min(1.0, float(v)))
    if v >= 0:
        r, g, b = 255, int(255 - 120 * v), int(255 - 160 * v)
    else:
        r, g, b = int(255 + 120 * v), int(255 + 60 * v), 255
    return f'rgb({r},{g},{b})'


def _render_matrix(title, desc, names, mat):
    if not names:
        return ''
    short = [html.escape(n if len(n) <= 16 else n[:14] + '…') for n in names]
    header = ''.join(f'<th class="mh" title="{html.escape(n)}">{s}</th>' for n, s in zip(names, short))
    rows = ''
    for i, n in enumerate(names):
        cells = ''
        for j in range(len(names)):
            v = mat[i][j]
            txt = '' if v is None else f'{v:.2f}'
            bg = _matrix_color(v)
            cells += f'<td class="mc" style="background:{bg}" title="{html.escape(n)} × {html.escape(names[j])}: {txt}">{txt}</td>'
        rows += f'<tr><th class="mh" title="{html.escape(n)}">{short[i]}</th>{cells}</tr>'
    return (
        f'<h2>{title}</h2><p style="color:#57606a;font-size:13px">{desc}</p>'
        f'<div style="overflow:auto"><table class="matrix"><tr><th></th>{header}</tr>{rows}</table></div>'
    )


def _similarity_section(names, corr):
    if len(names) < 2:
        return ('<h2>全截面 rank 相关性 &amp; 多样性</h2>'
                '<p style="color:#57606a">指纹缓存不足（需 ≥2 个因子）。'
                '运行 <code>uv run python -m factor_db.build_signatures</code> 构建。</p>')
    return _render_matrix(
        '全截面 rank 相关性（平均每日 Spearman，指纹近似）',
        '两因子每日全市场截面打分排名的相关；越高(越红)=选股逻辑越同质，越低(越绿)=越互补。'
        '多样性 = 1 − 与最近邻的相关。', names, corr.tolist())


def _diversity_by_name(names, sigs) -> dict:
    if len(names) < 2:
        return {}
    div = similarity.diversity_from_signatures(sigs)
    return {n: float(d) for n, d in zip(names, div)}


def _by_name(factors):
    return {f['name']: f for f in factors}


def _parents_of(f):
    raw = f.get('parent_ids')
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def _lineage_path(name, fmap):
    """沿“最优父代”回溯到种子，返回 root→...→name 的标签列表。"""
    chain, seen, cur = [], set(), name
    while cur and cur in fmap and cur not in seen:
        seen.add(cur)
        f = fmap[cur]
        chain.append(cur if f['op'] == 'seed' else f"g{f['generation']}·{cur}")
        best_parent, best_sh = None, None
        for pn in _parents_of(f):
            pf = fmap.get(pn)
            if pf is None:
                continue
            sh = pf['train_sharpe']
            if best_parent is None or (sh is not None and (best_sh is None or sh > best_sh)):
                best_parent, best_sh = pn, sh
        cur = best_parent
    chain.reverse()
    return chain


def _per_gen_best(factors):
    """返回 (run_id, [(gen, best_factor), ...])，取最近一次运行的每代 passed 最优。"""
    run_ids = [f['run_id'] for f in factors if f.get('run_id')]
    latest = max(run_ids) if run_ids else None
    pool = [f for f in factors if f.get('run_id') == latest] if latest else \
        [f for f in factors if f['status'] == 'passed' and f['generation'] > 0]
    by_gen = defaultdict(list)
    for f in pool:
        if f['status'] == 'passed':
            by_gen[f['generation']].append(f)
    out = [(g, max(by_gen[g], key=lambda x: x['train_sharpe'] if x['train_sharpe'] is not None else -1e9))
           for g in sorted(by_gen)]
    return latest, out


def render_html(factors, oos=None, diversity_html='') -> str:
    factors = sorted(
        factors,
        key=lambda f: (f['train_sharpe'] is not None, f['train_sharpe'] or 0.0),
        reverse=True,
    )
    oos = oos or {}
    has_oos = bool(oos)

    head = """<meta charset="utf-8"><title>LLM-GA 因子进化报告</title>
<style>
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#1f2328;background:#fff}
h1{font-size:22px}h2{font-size:16px;margin-top:28px;border-bottom:1px solid #d0d7de;padding-bottom:6px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
th,td{border:1px solid #d0d7de;padding:6px 8px;text-align:left}
th{background:#f6f8fa;position:sticky;top:0}
tr:nth-child(even){background:#f6f8fa}
.num{text-align:right;font-variant-numeric:tabular-nums}
.tag{display:inline-block;padding:1px 6px;border-radius:6px;font-size:11px;background:#ddf4ff;color:#0969da}
.seed{background:#eee;color:#555}.rej{background:#ffebe9;color:#cf222e}
code{font-family:Consolas,monospace}details{margin:4px 0}summary{cursor:pointer;color:#0969da}
pre{background:#f6f8fa;padding:10px;border-radius:6px;overflow:auto;font-size:12px}
.kpi{display:inline-block;margin-right:24px}.kpi b{font-size:20px}
table.matrix{border-collapse:collapse;font-size:11px}
table.matrix td.mc,table.matrix th.mh{border:1px solid #d0d7de;padding:3px 5px;text-align:center;min-width:34px}
table.matrix th.mh{background:#f6f8fa;white-space:nowrap;position:static}
</style>"""

    passed = [f for f in factors if f['status'] == 'passed']
    sharpes = [f['train_sharpe'] for f in passed if f['train_sharpe'] is not None]
    gens = max((f['generation'] for f in factors), default=0)
    kpis = (
        f'<div class="kpi">因子总数<br><b>{len(factors)}</b></div>'
        f'<div class="kpi">进化产出(passed)<br><b>{len(passed)}</b></div>'
        f'<div class="kpi">代数<br><b>{gens}</b></div>'
        f'<div class="kpi">最佳训练夏普<br><b>{_fmt(max(sharpes)) if sharpes else "-"}</b></div>'
    )

    gen_rows = ''.join(
        f'<tr><td class="num">{g}</td><td class="num">{n}</td>'
        f'<td class="num">{p}</td><td class="num">{_fmt(b)}</td></tr>'
        for g, n, p, b in _gen_summary(factors)
    )

    fmap = _by_name(factors)
    latest_run, gen_best = _per_gen_best(factors)
    best_rows = ''
    for g, bf in gen_best:
        path = ' → '.join(html.escape(x) for x in _lineage_path(bf['name'], fmap)) or '—'
        best_rows += (
            f'<tr><td class="num">{g}</td>'
            f'<td><span class="tag">{bf["op"]}</span> {html.escape(bf["name"])}</td>'
            f'<td class="num" style="color:{_sharpe_color(bf["train_sharpe"])}"><b>{_fmt(bf["train_sharpe"])}</b></td>'
            f'<td>{html.escape(bf.get("thesis") or "(未提供)")}</td>'
            f'<td><code>{path}</code></td></tr>'
        )
    best_section = (
        f'<h2>每代最优因子 &amp; 进化路径{f"（run {html.escape(latest_run)}）" if latest_run else ""}</h2>'
        f'<table><tr><th class="num">代</th><th>最优因子</th><th class="num">训练夏普</th>'
        f'<th>思路</th><th>进化路径（种子→…→当代）</th></tr>{best_rows or "<tr><td colspan=5>暂无进化产出</td></tr>"}</table>'
    )

    oos_th = '<th class="num">OOS夏普</th><th class="num">IS-OOS差</th>' if has_oos else ''
    rows = []
    for f in factors:
        name = html.escape(f['name'])
        op_cls = {'seed': 'seed', 'rejected': 'rej'}.get(f['op'], '') or (
            'rej' if f['status'] == 'rejected' else '')
        is_sh = f['train_sharpe']
        oos_cell = ''
        if has_oos:
            ov = oos.get(f['name'])
            diff = (is_sh - ov) if (is_sh is not None and ov is not None) else None
            oos_cell = (f'<td class="num" style="color:{_sharpe_color(ov)}">{_fmt(ov)}</td>'
                        f'<td class="num">{_fmt(diff)}</td>')
        parents = html.escape(f['parent_ids'] or '')
        thesis = html.escape(f.get('thesis') or '')
        code_path = REPO_ROOT / f['file_path']
        code = html.escape(code_path.read_text(encoding='utf-8')) if code_path.exists() else '(文件缺失)'
        div = f.get('diversity')
        div_cell = f'<td class="num">{_fmt(div)}</td>'
        rows.append(
            f'<tr><td class="num">{f["factor_id"]}</td>'
            f'<td><span class="tag {op_cls}">{f["op"]}</span> {name}'
            f'<details><summary>代码</summary><pre>{code}</pre></details></td>'
            f'<td class="num">{f["generation"]}</td>'
            f'<td class="num" style="color:{_sharpe_color(is_sh)}"><b>{_fmt(is_sh)}</b></td>'
            f'{div_cell}'
            f'<td class="num">{_fmt(f.get("annualized"), 1)}</td>'
            f'<td class="num">{_fmt(f["max_dd"], 1)}</td>'
            f'{oos_cell}'
            f'<td class="num">{f["params_count"]}</td>'
            f'<td>{thesis}</td>'
            f'<td><code>{parents}</code></td>'
            f'<td>{html.escape((f["created_at"] or "")[:19])}</td></tr>'
        )

    return f"""<!doctype html><html><head>{head}</head><body>
<h1>LLM-GA 因子进化报告</h1>
<p>{kpis}</p>
<h2>分代统计</h2>
<table><tr><th class="num">代</th><th class="num">个体数</th><th class="num">通过</th><th class="num">最佳训练夏普</th></tr>{gen_rows}</table>
{best_section}
{diversity_html}
<h2>因子榜单（按训练期夏普降序）</h2>
<table><tr><th class="num">ID</th><th>因子 / 代码</th><th class="num">代</th>
<th class="num">训练夏普(IS)</th><th class="num">多样性</th><th class="num">年化%</th><th class="num">最大回撤%</th>{oos_th}
<th class="num">参数</th><th>思路</th><th>父代</th><th>创建时间</th></tr>
{''.join(rows)}</table>
</body></html>"""


def generate(oos_start=None, oos_end=None, open_browser=True) -> Path:
    factors = db.list_factors()
    # 展示指标取每个因子最新一次回测（backfill 落库），保证榜单是当前结果
    runs_summary = records.latest_runs_by_factor()
    factors = _merge_run_metrics(factors, runs_summary)
    oos = None
    if oos_start and oos_end:
        print(f'计算 OOS（{oos_start}~{oos_end}）...')
        oos = _compute_oos(factors, oos_start, oos_end)

    sig_names, sig_sigs, _ = similarity.load_cache()
    sig_corr = similarity.correlation_matrix(sig_sigs)
    diversity_html = _similarity_section(sig_names, sig_corr)
    div_map = _diversity_by_name(sig_names, sig_sigs)
    for f in factors:
        f['diversity'] = div_map.get(f['name'])
    _OUT.write_text(render_html(factors, oos, diversity_html), encoding='utf-8')

    fmap = _by_name(factors)
    latest_run, gen_best = _per_gen_best(factors)
    if gen_best:
        print(f'\n=== 每代最优因子 & 进化路径{f"（run {latest_run}）" if latest_run else ""} ===')
        for g, bf in gen_best:
            path = ' -> '.join(_lineage_path(bf['name'], fmap)) or '-'
            print(f'[gen {g}] {bf["name"]} sharpe={_fmt(bf["train_sharpe"])}'
                  f' | 思路: {bf.get("thesis") or "(未提供)"}')
            print(f'         路径: {path}')

    print(f'\n{"ID":>4}  {"因子":<26}{"代":>3}{"训练夏普":>10}  {"op"}')
    for f in sorted(factors, key=lambda x: (x['train_sharpe'] is not None, x['train_sharpe'] or 0), reverse=True):
        print(f'{f["factor_id"]:>4}  {f["name"]:<26}{f["generation"]:>3}{_fmt(f["train_sharpe"]):>10}  {f["op"]}')
    print(f'\nHTML 报告: {_OUT}')

    if open_browser:
        webbrowser.open(_OUT.as_uri())
    return _OUT


def main():
    p = argparse.ArgumentParser(description='LLM-GA 因子进化报告')
    p.add_argument('--oos-start', type=str, default=None)
    p.add_argument('--oos-end', type=str, default=None)
    p.add_argument('--no-open', action='store_true')
    args = p.parse_args()
    generate(args.oos_start, args.oos_end, open_browser=not args.no_open)


if __name__ == '__main__':
    main()
