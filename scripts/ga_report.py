"""GA 进度可视化 — 提取每10代最优个体，生成 HTML 报告"""
import pickle, json, sys
from pathlib import Path
from datetime import datetime
import numpy as np

FACTOR_NAMES = [
    "SmallCap","TrueMarketCap",
    "ShortTermReversal","CashFlowQuality",
    "SmallCapMarginExpansion","PureProfitYoyAccel","QualityReversal10D",
    "TMC_GARP_Broad","TMC_GARP_Mult","TMC_ProfitYoy_25_LowVol",
    "LowTurnover20D",
    "ADX14Trend","CloseMom21D","CCI14","BB20Position",
    "OvernightGap1D","OBVSlope","ATR14","PricePosition256D","EWMADivergence",
    "Aroon14","ProfitYoy","ROE","EPValuation",
    "MACD","TRIX","KDJ","Turnover","SAR",
]

POOL_LABELS = {
    '60': '沪主', '0': '深主', '30': '创业板', '688': '科创板',
}


def _pool_label(pool):
    if not pool:
        return "全市场"
    parts = [POOL_LABELS.get(p, p) for p in pool]
    return '+'.join(parts)


def _weight_badge(w):
    if w > 0:
        return f'<span class="w-pos">{w:+.1f}</span>'
    elif w < 0:
        return f'<span class="w-neg">{w:+.1f}</span>'
    return f'<span class="w-zero">{w:+.0f}</span>'


def _pct(v):
    if v is None or np.isnan(v):
        return "-"
    return f"{v:+.1f}%"


def _num(v, d=3):
    if v is None or np.isnan(v):
        return "-"
    return f"{v:.{d}f}"


def load_data(ckpt_dir: Path):
    ckpt_path = ckpt_dir / 'checkpoint.pkl'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'checkpoint.pkl not found: {ckpt_path}')

    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    all_results = ckpt['all_results']
    generation_results = ckpt['generation_results']

    # 测试集结果（如果存在）
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

    return all_results, generation_results, test_results, best_config


def build_snapshot_table(all_results, generation_results, step=10):
    """每 step 代取验证集最优个体，收集 train/val 指标 + config"""
    snapshots = []

    # 按代分组
    by_gen = {}
    for r in all_results:
        g = r.get('generation', 0)
        by_gen.setdefault(g, []).append(r)

    for per_gen_stats in generation_results:
        g = per_gen_stats['generation']
        if g % step != 0 and g != len(generation_results) - 1:
            continue
        gen_results = by_gen.get(g, [])
        valid = [r for r in gen_results if r.get('_error') is None and r.get('sharpe') is not None]
        if not valid:
            continue

        # 验证集最优
        val_best = max(valid, key=lambda r: r.get('val_sharpe', -999))
        train_best = max(valid, key=lambda r: r.get('sharpe', -999))

        snapshots.append({
            'generation': g,
            'population_size': per_gen_stats['population_size'],
            'gen_time': per_gen_stats['generation_time'],
            # 训练最优个体
            'train_sharpe': train_best['sharpe'],
            'train_annual': train_best['annualized'],
            'train_mdd': train_best['max_drawdown'],
            # 训练最优在验证集
            'train_best_val': next((r.get('val_sharpe') for r in valid
                                    if r['individual_config'] == train_best['individual_config']), None),
            # 验证集最优个体
            'val_sharpe': val_best['val_sharpe'],
            'val_annual': val_best['val_annualized'],
            'val_mdd': val_best['val_max_drawdown'],
            # 验证最优在训练集
            'val_best_train': val_best['sharpe'],
            # 最优个体 config
            'config': val_best['individual_config'],
            # 代均值
            'mean_fitness': per_gen_stats['mean_fitness'],
            'val_mean_score': per_gen_stats.get('val_mean_score'),
        })

    return snapshots


def render_html(snapshots, test_results, best_config, output_dir_name):
    rows = []
    for s in snapshots:
        cfg = s['config']
        w = cfg.get('weights', {})
        top_w = sorted(w.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        w_str = ' '.join(f"{n}={v:+.1f}" for n, v in top_w)
        hp = cfg.get('holding_period') or 1

        rows.append(f'''
        <tr>
          <td class="gen">{s['generation']}</td>
          <td>{_num(s['train_sharpe'])}</td>
          <td>{_pct(s['train_annual'])}</td>
          <td>{_pct(s['train_mdd'])}</td>
          <td class="val">{_num(s['val_sharpe'])}</td>
          <td>{_pct(s['val_annual'])}</td>
          <td>{_pct(s['val_mdd'])}</td>
          <td>{cfg.get('buy_n')}/{cfg.get('sell_m')}</td>
          <td>{hp}天</td>
          <td>{_pool_label(cfg.get('stock_pool'))}</td>
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
  td.val {{ color: #7ee787; }}
  td.weights {{ text-align: left; font-size: 11px; max-width: 400px; word-break: break-all; }}
  .w-pos {{ color: #7ee787; }}
  .w-neg {{ color: #f85149; }}
  .w-zero {{ color: #8b949e; }}
  .section {{ margin-bottom: 30px; }}
  pre {{ background: #161b22; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 12px; }}
  .updated {{ color: #8b949e; font-size: 12px; }}
</style>
</head>
<body>
<h1>GA 进度报告</h1>
<p class="updated">更新于 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 目录: {output_dir_name}</p>

<div class="section">
  <h2>每10代最优个体（按验证集Sharpe排序）</h2>
  <table>
    <thead>
      <tr>
        <th>代</th>
        <th>训练Sharpe</th><th>训练年化</th><th>训练回撤</th>
        <th>验证Sharpe</th><th>验证年化</th><th>验证回撤</th>
        <th>buy/sell</th><th>持仓周期</th><th>股票池</th>
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
            all_results, generation_results, test_results, best_config = load_data(ckpt_dir)
            snapshots = build_snapshot_table(all_results, generation_results)
            html = render_html(snapshots, test_results, best_config, ckpt_dir.name)
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
    all_results, generation_results, test_results, best_config = load_data(ckpt_dir)
    print(f"  总个体数: {len(all_results)}, 总代数: {len(generation_results)}")

    snapshots = build_snapshot_table(all_results, generation_results)
    print(f"  快照数: {len(snapshots)} (每10代 + 末代)")

    html = render_html(snapshots, test_results, best_config, ckpt_dir.name)
    out_path = ckpt_dir / 'ga_progress.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"报告已生成: {out_path}")


if __name__ == '__main__':
    main()
