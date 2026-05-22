"""GA 进度可视化 — 提取每10代最优个体，生成 HTML 报告（含图表+统计+权重分析）"""
import pickle, json, sys
from pathlib import Path
from datetime import datetime
import numpy as np

FACTOR_NAMES = [
    "AmountBasedSmallCap","TrueMarketCap",
    "ShortTermReversal10D","CashFlowQuality",
    "TMC_MarginExpansion","MarginExpansion","TMC_ReversalProfitGrowth10D",
    "TMC_AdditiveBlend","TMC_GARP_Mult","TMC_ProfitYoy_25_LowVol",
    "LowTurnover20D",
    "ADX14Trend","CloseMom21D","CCI14","BB_PercentB",
    "OvernightGap1D","OBVRateOfChange10D","LowATR14","PricePosition256D","EWMADeviation",
    "AroonOscillator14","ProfitYoy","ROE","EarningsYield",
    "MACDHistogram","TRIX","KDJ_Spread","VolumeRatio5D","SAR_Distance",
]

POOL_LABELS = {
    '60': '沪主', '0': '深主', '30': '创业板', '688': '科创板',
}


def _pool_label(pool):
    if not pool:
        return "全市场"
    parts = [POOL_LABELS.get(p, p) for p in pool]
    return '+'.join(parts)


def _pct(v):
    if v is None or np.isnan(v):
        return "-"
    return f"{v:+.1f}%"


def _num(v, d=3):
    if v is None or np.isnan(v):
        return "-"
    return f"{v:.{d}f}"


def _color_sharpe(v):
    """Sharpe 值着色"""
    if v is None or np.isnan(v) or v == 0:
        return "#8b949e"
    if v >= 1.5: return "#7ee787"
    if v >= 1.0: return "#a5d6ff"
    if v >= 0.5: return "#d2a8ff"
    return "#f85149"


def load_data(ckpt_dir: Path):
    ckpt_path = ckpt_dir / 'checkpoint.pkl'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'checkpoint.pkl not found: {ckpt_path}')

    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    all_results = ckpt['all_results']
    generation_results = ckpt['generation_results']
    ga_cache = ckpt.get('ga_cache', {})

    test_ckpt_path = ckpt_dir / 'test_final_results.json'
    test_results = {}
    if test_ckpt_path.exists():
        with open(test_ckpt_path, 'r', encoding='utf-8') as f:
            test_results = json.load(f)

    best_config_path = ckpt_dir / 'best_individual_config.json'
    best_config = None
    if best_config_path.exists():
        with open(best_config_path, 'r', encoding='utf-8') as f:
            best_config = json.load(f)

    return all_results, generation_results, ga_cache, test_results, best_config


def build_snapshot_table(all_results, generation_results, ga_cache=None, step=10):
    """每 step 代取验证集最优个体，收集 train/val 指标 + config"""
    snapshots = []

    # all_results 按代分组
    by_gen = {}
    for r in all_results:
        g = r.get('generation', 0)
        by_gen.setdefault(g, []).append(r)

    # ga_cache 按代分组（补充 all_results 截断丢失的早期代）
    cache_by_gen = {}
    if ga_cache:
        for v in ga_cache.values():
            g = v.get('generation', 0)
            cache_by_gen.setdefault(g, []).append(v)

    for per_gen_stats in generation_results:
        g = per_gen_stats['generation']
        has_val = per_gen_stats.get('val_best_sharpe') is not None
        if not has_val and g != len(generation_results) - 1:
            continue

        vs = per_gen_stats.get('val_best_sharpe')
        va = per_gen_stats.get('val_best_annualized')
        vm = per_gen_stats.get('val_best_mdd')
        ts = per_gen_stats.get('test_train_best_sharpe')
        ta = per_gen_stats.get('test_train_best_annualized')
        tm = per_gen_stats.get('test_train_best_mdd')

        # 优先 all_results，fallback ga_cache
        gen_results = by_gen.get(g, []) or cache_by_gen.get(g, [])
        valid = [r for r in gen_results if r.get('_error') is None and r.get('sharpe') is not None]
        train_best = max(valid, key=lambda r: r.get('sharpe', -999)) if valid else None

        snapshots.append({
            'generation': g,
            'population_size': per_gen_stats['population_size'],
            'gen_time': per_gen_stats['generation_time'],
            'train_sharpe': train_best['sharpe'] if train_best else None,
            'train_annual': train_best['annualized'] if train_best else None,
            'train_mdd': train_best['max_drawdown'] if train_best else None,
            'val_sharpe': vs or 0,
            'val_annual': va or 0,
            'val_mdd': vm or 0,
            'test_sharpe': ts or 0,
            'test_annual': ta or 0,
            'test_mdd': tm or 0,
            'config': train_best['individual_config'] if train_best else None,
            'mean_fitness': per_gen_stats.get('mean_fitness'),
            'val_mean_score': per_gen_stats.get('val_mean_score'),
        })

    return snapshots


def build_summary(snapshots):
    """摘要统计"""
    val_sharpes = [s['val_sharpe'] for s in snapshots if s['val_sharpe']]
    test_sharpes = [s['test_sharpe'] for s in snapshots if s['test_sharpe']]
    train_sharpes = [s['train_sharpe'] for s in snapshots if s['train_sharpe']]

    best_val = max(snapshots, key=lambda s: s['val_sharpe'])
    best_test = max(snapshots, key=lambda s: s['test_sharpe'])
    best_train = max(snapshots, key=lambda s: s['train_sharpe'] or -999)

    return {
        'total_gens': snapshots[-1]['generation'],
        'total_snapshots': len(snapshots),
        'best_val_sharpe': best_val['val_sharpe'],
        'best_val_gen': best_val['generation'],
        'best_test_sharpe': best_test['test_sharpe'],
        'best_test_gen': best_test['generation'],
        'best_train_sharpe': best_train['train_sharpe'],
        'best_train_gen': best_train['generation'],
        'latest_val': snapshots[-1]['val_sharpe'],
        'latest_test': snapshots[-1]['test_sharpe'],
        'latest_train': snapshots[-1]['train_sharpe'],
        'val_mean': np.mean(val_sharpes) if val_sharpes else 0,
        'test_mean': np.mean(test_sharpes) if test_sharpes else 0,
        'val_std': np.std(val_sharpes) if val_sharpes else 0,
        'test_std': np.std(test_sharpes) if test_sharpes else 0,
    }


def build_weight_analysis(snapshots, ga_cache=None, top_n=15):
    """最近 N 个有完整 config 的快照的因子权重统计 + ga_cache 全局探索范围"""
    with_cfg = [s for s in snapshots if s['config'] is not None]
    if len(with_cfg) == 0:
        return [], [], {}

    recent = with_cfg[-top_n:]
    factor_weights = {fn: [] for fn in FACTOR_NAMES}

    for s in recent:
        w = s['config'].get('weights', {})
        for fn in FACTOR_NAMES:
            factor_weights[fn].append(w.get(fn, 0.0))

    # 全局探索范围（来自 ga_cache）
    global_range = {}
    if ga_cache:
        gr = {fn: {'min': 999, 'max': -999} for fn in FACTOR_NAMES}
        for v in ga_cache.values():
            if v.get('_error') is not None:
                continue
            w = v.get('individual_config', {}).get('weights', {})
            for fn in FACTOR_NAMES:
                val = w.get(fn, 0.0)
                if val < gr[fn]['min']: gr[fn]['min'] = val
                if val > gr[fn]['max']: gr[fn]['max'] = val
        global_range = {fn: (gr[fn]['min'], gr[fn]['max']) for fn in FACTOR_NAMES
                        if gr[fn]['min'] < gr[fn]['max']}

    rows = []
    for fn in FACTOR_NAMES:
        vals = factor_weights[fn]
        mean_w = np.mean(vals)
        std_w = np.std(vals)
        nonzero = sum(1 for v in vals if v != 0) / len(vals) * 100
        max_w = max(vals)
        min_w = min(vals)
        if mean_w == 0 and std_w == 0:
            continue
        rows.append({
            'name': fn,
            'mean': mean_w,
            'std': std_w,
            'nonzero_pct': nonzero,
            'max': max_w,
            'min': min_w,
            'global_min': global_range.get(fn, (None, None))[0],
            'global_max': global_range.get(fn, (None, None))[1],
        })

    rows.sort(key=lambda r: abs(r['mean']), reverse=True)
    return rows, recent, global_range


def _summary_card(label, value, sub="", color="#8b949e"):
    return f'''<div class="card">
      <div class="card-label">{label}</div>
      <div class="card-value" style="color:{color}">{value}</div>
      <div class="card-sub">{sub}</div>
    </div>'''


def render_html(snapshots, test_results, best_config, ga_cache, output_dir_name):
    # ---- 摘要统计 ----
    sm = build_summary(snapshots)
    summary_cards = f'''
    <div class="summary-grid">
      {_summary_card("最佳验证Sharpe", f"{sm['best_val_sharpe']:.3f}", f"Gen {sm['best_val_gen']}", _color_sharpe(sm['best_val_sharpe']))}
      {_summary_card("最佳测试Sharpe", f"{sm['best_test_sharpe']:.3f}", f"Gen {sm['best_test_gen']}", _color_sharpe(sm['best_test_sharpe']))}
      {_summary_card("最佳训练Sharpe", f"{sm['best_train_sharpe']:.3f}" if sm['best_train_sharpe'] else "-", f"Gen {sm['best_train_gen']}", _color_sharpe(sm['best_train_sharpe']))}
      {_summary_card("最新验证Sharpe", f"{sm['latest_val']:.3f}", f"Gen {sm['total_gens']}", _color_sharpe(sm['latest_val']))}
      {_summary_card("最新测试Sharpe", f"{sm['latest_test']:.3f}", f"Gen {sm['total_gens']}", _color_sharpe(sm['latest_test']))}
      {_summary_card("验证均值±std", f"{sm['val_mean']:.3f}±{sm['val_std']:.3f}", f"{sm['total_snapshots']}个快照", "#a5d6ff")}
    </div>'''

    # ---- 趋势图数据 ----
    chart_data = json.dumps([{
        'g': s['generation'],
        'tr': s['train_sharpe'],
        'va': s['val_sharpe'] if s['val_sharpe'] else None,
        'te': s['test_sharpe'] if s['test_sharpe'] else None,
    } for s in snapshots])

    # ---- 权重分析 ----
    weight_rows, weight_source, global_range = build_weight_analysis(snapshots, ga_cache)
    weight_section = ''
    if weight_rows:
        wrows_html = ''
        for r in weight_rows:
            bar_w = min(abs(r['mean']) * 40, 120)
            bar_color = '#7ee787' if r['mean'] > 0 else '#f85149'
            gmin = f"{r['global_min']:+.0f}" if r['global_min'] is not None else '-'
            gmax = f"{r['global_max']:+.0f}" if r['global_max'] is not None else '-'
            wrows_html += f'''
            <tr>
              <td class="wname">{r['name']}</td>
              <td><span style="color:{bar_color}">{r['mean']:+.2f}</span></td>
              <td>{r['std']:.2f}</td>
              <td>{r['nonzero_pct']:.0f}%</td>
              <td><span class="w-pos">{r['max']:+.1f}</span></td>
              <td><span class="w-neg">{r['min']:+.1f}</span></td>
              <td><span class="w-zero">{gmin}~{gmax}</span></td>
              <td><div class="bar-bg"><div class="bar-fill" style="width:{bar_w}px;background:{bar_color}"></div></div></td>
            </tr>'''
        weight_section = f'''
    <div class="section">
      <h2>因子权重分析（最近{len(weight_source)}个可配快照，Gen {weight_source[0]["generation"]}~{weight_source[-1]["generation"]}）</h2>
      <table>
        <thead><tr>
          <th>因子</th><th>平均权重</th><th>标准差</th><th>非零占比</th><th>最优个体+</th><th>最优个体-</th><th>全量探索</th><th>分布</th>
        </tr></thead>
        <tbody>{wrows_html}</tbody>
      </table>
    </div>'''

    # ---- 历史表格 ----
    rows = []
    for s in snapshots:
        cfg = s['config']
        if cfg:
            w = cfg.get('weights', {})
            top_w = sorted(w.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
            w_str = ' '.join(f"{n}={v:+.1f}" for n, v in top_w)
            hp = cfg.get('holding_period') or 1
            buy_n = cfg.get('buy_n', 0)
            sell_m = cfg.get('sell_m', 0)
            buy_sell = f"{buy_n}/{sell_m}"
            pool = _pool_label(cfg.get('stock_pool'))
            turnover = buy_n / hp if hp else 0
            turnover_str = f"{turnover:.1f}只/天"
        else:
            w_str = '-'
            hp = '-'
            buy_sell = '-'
            pool = '-'
            turnover_str = '-'

        rows.append(f'''
        <tr>
          <td class="gen">{s['generation']}</td>
          <td style="color:{_color_sharpe(s['train_sharpe'])}">{_num(s['train_sharpe'])}</td>
          <td style="color:{_color_sharpe(s['val_sharpe'])}">{_num(s['val_sharpe'])}</td>
          <td style="color:{_color_sharpe(s['test_sharpe'])}">{_num(s['test_sharpe'])}</td>
          <td>{buy_sell}</td>
          <td>{hp}</td>
          <td>{turnover_str}</td>
          <td>{pool}</td>
          <td class="weights">{w_str}</td>
          <td>{s['gen_time']:.0f}s</td>
        </tr>''')

    test_section = ''
    if test_results:
        ts = test_results
        test_section = f'''
    <div class="section">
      <h2>最终测试集评估</h2>
      <table>
        <tr><th>训练最优-测试Sharpe</th><td>{_num(ts.get('test_train_best_score'))}</td></tr>
        <tr><th>验证最优-测试Sharpe</th><td>{_num(ts.get('test_val_best_score'))}</td></tr>
        <tr><th>训练最优-测试年化</th><td>{_pct(ts.get('test_train_best_annual'))}</td></tr>
        <tr><th>验证最优-测试年化</th><td>{_pct(ts.get('test_val_best_annual'))}</td></tr>
      </table>
    </div>'''

    best_section = ''
    if best_config:
        bc = best_config
        best_section = f'''
    <div class="section">
      <h2>当前最优个体 (gen {bc.get('generation', '?')})</h2>
      <pre>{json.dumps(bc.get('individual_config', {}), indent=2, ensure_ascii=False)}</pre>
    </div>'''

    return f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>GA 进度 — {output_dir_name}</title>
<style>
  body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; margin: 20px; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; }}
  h2 {{ color: #8b949e; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ padding: 6px 8px; border: 1px solid #21262d; text-align: right; }}
  th {{ background: #161b22; color: #8b949e; position: sticky; top: 0; }}
  tr:hover {{ background: #161b22; }}
  td.gen {{ font-weight: bold; color: #58a6ff; }}
  td.weights {{ text-align: left; font-size: 11px; max-width: 350px; word-break: break-all; }}
  td.wname {{ text-align: left; font-weight: bold; white-space: nowrap; }}
  .w-pos {{ color: #7ee787; }}
  .w-neg {{ color: #f85149; }}
  .w-zero {{ color: #8b949e; }}
  .section {{ margin-bottom: 30px; }}
  pre {{ background: #161b22; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 12px; }}
  .updated {{ color: #8b949e; font-size: 12px; }}

  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 20px; }}
  .card {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 14px 16px; text-align: center; }}
  .card-label {{ font-size: 12px; color: #8b949e; margin-bottom: 4px; }}
  .card-value {{ font-size: 24px; font-weight: bold; }}
  .card-sub {{ font-size: 11px; color: #484f58; margin-top: 2px; }}

  .chart-wrap {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .chart-box {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px; }}
  .chart-box h3 {{ color: #8b949e; margin: 0 0 10px 0; font-size: 14px; }}
  canvas {{ width: 100%; max-height: 350px; }}

  .bar-bg {{ display: inline-block; width: 120px; height: 8px; background: #21262d; border-radius: 4px; vertical-align: middle; }}
  .bar-fill {{ height: 8px; border-radius: 4px; }}

  @media (max-width: 900px) {{ .chart-wrap {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>GA 进度报告</h1>
<p class="updated">更新于 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 目录: {output_dir_name}</p>

<div class="section">
  <h2>摘要</h2>
  {summary_cards}
</div>

<div class="section">
  <h2>Sharpe 趋势</h2>
  <div class="chart-wrap">
    <div class="chart-box">
      <h3>验证 / 测试 Sharpe</h3>
      <canvas id="chartValTest"></canvas>
    </div>
    <div class="chart-box">
      <h3>训练 Sharpe</h3>
      <canvas id="chartTrain"></canvas>
    </div>
  </div>
</div>

{weight_section}

<div class="section">
  <h2>每10代最优个体</h2>
  <table>
    <thead>
      <tr>
        <th>代</th>
        <th>训练Sharpe</th><th>验证Sharpe</th><th>测试Sharpe</th>
        <th>buy/sell</th><th>持仓周期</th><th>理论调仓</th><th>股票池</th>
        <th>Top权重</th>
        <th>耗时</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</div>
{test_section}
{best_section}

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
<script>
const data = {chart_data};

const gens = data.map(d => d.g);
const commonOptions = {{
  responsive: true,
  maintainAspectRatio: false,
  plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
  scales: {{
    x: {{ title: {{ display: true, text: 'Generation', color: '#8b949e' }}, ticks: {{ color: '#484f58', maxTicksLimit: 20 }} }},
    y: {{ title: {{ display: true, text: 'Sharpe', color: '#8b949e' }}, ticks: {{ color: '#484f58' }}, grid: {{ color: '#21262d' }} }}
  }},
  interaction: {{ mode: 'nearest', intersect: false }}
}};

// 验证/测试
new Chart(document.getElementById('chartValTest'), {{
  type: 'line',
  data: {{
    labels: gens,
    datasets: [
      {{
        label: 'Val Sharpe',
        data: data.map(d => d.va),
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.1)',
        borderWidth: 2, pointRadius: 2, tension: 0.1,
        spanGaps: false
      }},
      {{
        label: 'Test Sharpe',
        data: data.map(d => d.te),
        borderColor: '#f0883e',
        backgroundColor: 'rgba(240,136,62,0.1)',
        borderWidth: 2, pointRadius: 2, tension: 0.1,
        borderDash: [5,3],
        spanGaps: false
      }}
    ]
  }},
  options: commonOptions
}});

// 训练
new Chart(document.getElementById('chartTrain'), {{
  type: 'line',
  data: {{
    labels: gens,
    datasets: [{{
      label: 'Train Sharpe',
      data: data.map(d => d.tr),
      borderColor: '#7ee787',
      backgroundColor: 'rgba(126,231,135,0.1)',
      borderWidth: 2, pointRadius: 2, tension: 0.1,
      spanGaps: false
    }}]
  }},
  options: commonOptions
}});
</script>
</body>
</html>'''


def find_latest_ga_dir():
    results = Path(__file__).resolve().parent.parent / 'results'
    ga_dirs = sorted(
        [d for d in results.iterdir() if d.is_dir() and d.name.startswith('ga_2026')],
        key=lambda d: d.name, reverse=True
    )
    return ga_dirs[0] if ga_dirs else None


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('--open', '-o'):
        ckpt_dir = find_latest_ga_dir()
        if not ckpt_dir:
            print("未找到 GA 结果目录")
            sys.exit(1)
        report = ckpt_dir / 'ga_progress.html'
        if not report.exists():
            print(f"报告文件不存在，正在生成: {report}")
            all_results, generation_results, ga_cache, test_results, best_config = load_data(ckpt_dir)
            snapshots = build_snapshot_table(all_results, generation_results, ga_cache)
            html = render_html(snapshots, test_results, best_config, ga_cache, ckpt_dir.name)
            with open(report, 'w', encoding='utf-8') as f:
                f.write(html)
        import webbrowser
        webbrowser.open(str(report))
        print(f"已打开: {report}")
        return

    ckpt_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_ga_dir()
    if ckpt_dir is None:
        print("未找到 GA 结果目录")
        sys.exit(1)

    print(f"读取: {ckpt_dir}")
    all_results, generation_results, ga_cache, test_results, best_config = load_data(ckpt_dir)
    print(f"  总个体数: {len(all_results)} (all_results), {len(ga_cache)} (ga_cache), 总代数: {len(generation_results)}")

    snapshots = build_snapshot_table(all_results, generation_results, ga_cache)
    print(f"  快照数: {len(snapshots)} (每10代 + 末代)")

    html = render_html(snapshots, test_results, best_config, ga_cache, ckpt_dir.name)
    out_path = ckpt_dir / 'ga_progress.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"报告已生成: {out_path}")


if __name__ == '__main__':
    main()
