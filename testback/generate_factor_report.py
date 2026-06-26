"""从 factor_scan.json 生成交互式 HTML 可视化报告。

用法:
    uv run python testback/generate_factor_report.py
    uv run python testback/generate_factor_report.py --input results/factor_scan.json --output results/factor_scan.html
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

HOLDING_PERIODS = [1, 3, 7, 15, 30]


def _build_table_rows(results: list[dict], best_sharpe: dict) -> str:
    """生成主表格 HTML 行"""
    rows = []
    for r in results:
        sharpe_cls = "positive" if r["sharpe"] > 0 else "negative"
        ann_cls = "positive" if r["annualized"] > 0 else "negative"
        dd_cls = "negative" if r["max_drawdown"] < -20 else (
            "warning" if r["max_drawdown"] < -10 else ""
        )
        is_best = best_sharpe.get(r["factor"]) == r["holding_period"]
        best_mark = " ★" if is_best else ""
        rows.append(
            f"<tr>"
            f"<td>{r['factor']}</td>"
            f"<td>{r['holding_period']}{best_mark}</td>"
            f'<td class="{ann_cls}">{r["annualized"]:+.2f}%</td>'
            f'<td class="{dd_cls}">{r["max_drawdown"]:.2f}%</td>'
            f'<td class="{sharpe_cls}">{r["sharpe"]:.2f}</td>'
            f'<td>{r["calmar"]:.2f}</td>'
            f'<td class="{ann_cls}">{r["total_return"]:+.2f}%</td>'
            f"<td>{r['total_trades']}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _build_heatmap(factor_names: list[str], results: list[dict], metric: str) -> str:
    """生成因子×持仓周期 热力图"""
    # 构建二维数据
    f_to_row = {n: i for i, n in enumerate(factor_names)}
    hp_to_col = {hp: i for i, hp in enumerate(HOLDING_PERIODS)}
    grid = np.full((len(factor_names), len(HOLDING_PERIODS)), np.nan)
    for r in results:
        fi = f_to_row.get(r["factor"])
        hi = hp_to_col.get(r["holding_period"])
        if fi is not None and hi is not None:
            grid[fi, hi] = r[metric]

    # 双色渐变色阶：红=好，蓝=差
    vals = grid[~np.isnan(grid)]
    if len(vals) == 0:
        return ""
    vmin, vmax = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    if vmax <= vmin:
        vmax = vmin + 1

    cells = []
    for fi, name in enumerate(factor_names):
        row_cells = []
        for hi, hp in enumerate(HOLDING_PERIODS):
            v = grid[fi, hi]
            if np.isnan(v):
                row_cells.append('<td class="na">—</td>')
            else:
                t = max(0, min(1, (v - vmin) / (vmax - vmin)))
                r = int(255 * (1 - t))
                g = int(255 * t)
                b = 80
                row_cells.append(
                    f'<td style="background:rgb({r},{g},{b});color:{"#fff" if t<0.5 else "#000"}">'
                    f"{v:.2f}</td>"
                )
        cells.append(f"<tr><td class='factor-label'>{name}</td>{''.join(row_cells)}</tr>")
    return "\n".join(cells)


def _build_best_summary(factor_names: list[str], best_sharpe: dict, results: list[dict]) -> str:
    """按夏普最优汇总"""
    factor_data = {}
    for r in results:
        d = factor_data.setdefault(r["factor"], {})
        d[r["holding_period"]] = r
        if r["holding_period"] == best_sharpe.get(r["factor"]):
            d["best"] = r

    rows = []
    for name in factor_names:
        d = factor_data.get(name, {})
        r = d.get("best")
        if r is None:
            continue
        s_cls = "positive" if r["sharpe"] > 0 else "negative"
        rows.append(
            f"<tr>"
            f"<td>{name}</td>"
            f"<td>{r['holding_period']}</td>"
            f'<td class="{s_cls}">{r["annualized"]:+.2f}%</td>'
            f'<td>{r["max_drawdown"]:.2f}%</td>'
            f'<td class="{s_cls}">{r["sharpe"]:.2f}</td>'
            f'<td>{r["calmar"]:.2f}</td>'
            f"</tr>"
        )
    return "\n".join(rows)


def generate_report(results: list[dict], config: dict, output_path: Path):
    factor_names = sorted(set(r["factor"] for r in results))

    # 每个因子的夏普最优持仓周期
    best_sharpe: dict[str, int] = {}
    for r in results:
        prev = best_sharpe.get(r["factor"])
        if prev is None:
            best_sharpe[r["factor"]] = r["holding_period"]
        else:
            prev_r = next(
                x for x in results
                if x["factor"] == r["factor"] and x["holding_period"] == prev
            )
            if r["sharpe"] > prev_r["sharpe"]:
                best_sharpe[r["factor"]] = r["holding_period"]

    # 统计
    best_vals = []
    for name in factor_names:
        hp = best_sharpe.get(name)
        if hp:
            r = next(x for x in results if x["factor"] == name and x["holding_period"] == hp)
            best_vals.append(r)
    sharpes = [r["sharpe"] for r in best_vals]
    pos_count = sum(1 for s in sharpes if s > 0)

    table_rows = _build_table_rows(results, best_sharpe)
    heatmap_sharpe = _build_heatmap(factor_names, results, "sharpe")
    heatmap_ann = _build_heatmap(factor_names, results, "annualized")
    best_rows = _build_best_summary(factor_names, best_sharpe, results)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WBR 单因子扫描报告</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
h1 {{ color: #58a6ff; margin-bottom: 8px; }}
h2 {{ color: #f0883e; margin: 24px 0 12px; border-bottom: 1px solid #30363d; padding-bottom: 6px; }}

.config-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 16px 0; }}
.config-box h3 {{ color: #7ee787; margin-bottom: 8px; }}
.config-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px; }}
.config-item {{ display:flex; justify-content:space-between; padding: 4px 8px; background:#0d1117; border-radius:4px; }}
.config-key {{ color: #8b949e; }}
.config-val {{ color: #e6edf3; font-weight:600; }}

.stats {{ display:flex; gap:16px; flex-wrap:wrap; margin:16px 0; }}
.stat-card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px 20px; text-align:center; min-width:120px; }}
.stat-card .val {{ font-size:24px; font-weight:700; }}
.stat-card .lbl {{ font-size:12px; color:#8b949e; margin-top:4px; }}
.val.green {{ color:#3fb950; }}
.val.red {{ color:#f85149; }}
.val.yellow {{ color:#d2991d; }}

.table-wrapper {{ overflow-x:auto; max-height:70vh; overflow-y:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
thead {{ position:sticky; top:0; z-index:2; }}
th {{ background:#21262d; color:#8b949e; padding:8px 10px; text-align:left; cursor:pointer; white-space:nowrap; user-select:none; }}
th:hover {{ color:#e6edf3; }}
th .sort-icon {{ margin-left:4px; }}
td {{ padding:5px 10px; border-top:1px solid #21262d; white-space:nowrap; }}
tr:hover td {{ background:#1c2129; }}

.positive {{ color:#3fb950; }}
.negative {{ color:#f85149; }}
.warning {{ color:#d2991d; }}
.na {{ color:#484f58; text-align:center; }}

.heatmap-table td {{ padding:3px 6px; text-align:center; font-size:11px; min-width:42px; }}
.heatmap-table .factor-label {{ text-align:left; font-size:12px; background:#161b22; position:sticky; left:0; }}

.search-box {{ margin:12px 0; }}
.search-box input {{ width:100%; padding:8px 12px; background:#161b22; border:1px solid #30363d; border-radius:6px; color:#c9d1d9; font-size:14px; }}
.search-box input:focus {{ outline:none; border-color:#58a6ff; }}

.tab-bar {{ display:flex; gap:4px; margin:16px 0; }}
.tab {{ padding:6px 16px; background:#21262d; border:1px solid #30363d; border-radius:6px 6px 0 0; cursor:pointer; color:#8b949e; }}
.tab.active {{ background:#161b22; color:#e6edf3; border-bottom-color:#161b22; }}
.tab-content {{ display:none; }}
.tab-content.active {{ display:block; }}

footer {{ margin-top:40px; padding-top:16px; border-top:1px solid #30363d; color:#8b949e; font-size:12px; }}
</style>
</head>
<body>

<h1>📊 WBR 单因子扫描报告</h1>

<div class="config-box">
<h3>⚙️ 回测配置</h3>
<div class="config-grid">
<div class="config-item"><span class="config-key">回测区间</span><span class="config-val">{config['start_date']} ~ {config['end_date']}</span></div>
<div class="config-item"><span class="config-key">初始资金</span><span class="config-val">{config['init_cash']}</span></div>
<div class="config-item"><span class="config-key">买入数量 buy_n</span><span class="config-val">{config['buy_n']}</span></div>
<div class="config-item"><span class="config-key">卖出数量 sell_m</span><span class="config-val">{config['sell_m']}</span></div>
<div class="config-item"><span class="config-key">权重</span><span class="config-val">单因子 weight=1.0</span></div>
<div class="config-item"><span class="config-key">持仓周期</span><span class="config-val">{', '.join(map(str, HOLDING_PERIODS))} 日</span></div>
<div class="config-item"><span class="config-key">调仓方式</span><span class="config-val">均仓多退少补 (rebalance)</span></div>
<div class="config-item"><span class="config-key">涨停保护</span><span class="config-val">禁用 (limit_up_protection=false)</span></div>
<div class="config-item"><span class="config-key">滑点</span><span class="config-val">默认 (SIM_SLIPPAGE_RATE)</span></div>
<div class="config-item"><span class="config-key">费率</span><span class="config-val">默认 (佣金+印花税+过户费)</span></div>
<div class="config-item"><span class="config-key">生成时间</span><span class="config-val">{config['generated_at']}</span></div>
</div>
</div>

<div class="stats">
<div class="stat-card"><div class="val">{len(factor_names)}</div><div class="lbl">因子总数</div></div>
<div class="stat-card"><div class="val">{len(results)}</div><div class="lbl">回测总数 (×{len(HOLDING_PERIODS)})</div></div>
<div class="stat-card"><div class="val green">{pos_count}</div><div class="lbl">夏普>0 因子数</div></div>
<div class="stat-card"><div class="val">{pos_count/len(factor_names)*100:.0f}%</div><div class="lbl">夏普>0 占比</div></div>
<div class="stat-card"><div class="val {'green' if np.mean(sharpes)>0 else 'red'}">{np.mean(sharpes):.2f}</div><div class="lbl">夏普均值</div></div>
<div class="stat-card"><div class="val">{np.median(sharpes):.2f}</div><div class="lbl">夏普中位数</div></div>
</div>

<div class="search-box">
<input type="text" id="search" placeholder="🔍 搜索因子名称..." oninput="filterTable()">
</div>

<div class="tab-bar">
<div class="tab active" onclick="switchTab('detail')">📋 全部明细</div>
<div class="tab" onclick="switchTab('best')">⭐ 最优持仓周期（按夏普）</div>
<div class="tab" onclick="switchTab('heatmap_sharpe')">🔥 夏普热力图</div>
<div class="tab" onclick="switchTab('heatmap_ann')">📈 年化收益热力图</div>
</div>

<!-- 全部明细 -->
<div id="tab-detail" class="tab-content active">
<div class="table-wrapper">
<table id="detail-table">
<thead>
<tr>
<th onclick="sortTable(0,'detail-table')">因子名称 <span class="sort-icon">⇅</span></th>
<th onclick="sortTable(1,'detail-table')">持仓周期 <span class="sort-icon">⇅</span></th>
<th onclick="sortTable(2,'detail-table')">年化收益 <span class="sort-icon">⇅</span></th>
<th onclick="sortTable(3,'detail-table')">最大回撤 <span class="sort-icon">⇅</span></th>
<th onclick="sortTable(4,'detail-table')">夏普比率 <span class="sort-icon">⇅</span></th>
<th onclick="sortTable(5,'detail-table')">卡玛比率 <span class="sort-icon">⇅</span></th>
<th onclick="sortTable(6,'detail-table')">总收益 <span class="sort-icon">⇅</span></th>
<th onclick="sortTable(7,'detail-table')">总成交 <span class="sort-icon">⇅</span></th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>
</div>

<!-- 最优持仓周期 -->
<div id="tab-best" class="tab-content">
<div class="table-wrapper">
<table id="best-table">
<thead>
<tr>
<th onclick="sortTable(0,'best-table')">因子名称 <span class="sort-icon">⇅</span></th>
<th onclick="sortTable(1,'best-table')">最优HP <span class="sort-icon">⇅</span></th>
<th onclick="sortTable(2,'best-table')">年化收益 <span class="sort-icon">⇅</span></th>
<th onclick="sortTable(3,'best-table')">最大回撤 <span class="sort-icon">⇅</span></th>
<th onclick="sortTable(4,'best-table')">夏普比率 <span class="sort-icon">⇅</span></th>
<th onclick="sortTable(5,'best-table')">卡玛比率 <span class="sort-icon">⇅</span></th>
</tr>
</thead>
<tbody>
{best_rows}
</tbody>
</table>
</div>
</div>

<!-- 夏普热力图 -->
<div id="tab-heatmap_sharpe" class="tab-content">
<div class="table-wrapper">
<table class="heatmap-table">
<thead>
<tr><th>因子 \\ 持仓周期</th>
{"".join(f'<th>{hp}日</th>' for hp in HOLDING_PERIODS)}
</tr>
</thead>
<tbody>
{heatmap_sharpe}
</tbody>
</table>
</div>
<p style="color:#8b949e;font-size:11px;margin-top:8px;">颜色: 绿=高夏普 ←→ 红=低夏普（5%~95%分位映射）</p>
</div>

<!-- 年化热力图 -->
<div id="tab-heatmap_ann" class="tab-content">
<div class="table-wrapper">
<table class="heatmap-table">
<thead>
<tr><th>因子 \\ 持仓周期</th>
{"".join(f'<th>{hp}日</th>' for hp in HOLDING_PERIODS)}
</tr>
</thead>
<tbody>
{heatmap_ann}
</tbody>
</table>
</div>
<p style="color:#8b949e;font-size:11px;margin-top:8px;">颜色: 绿=高年化 ←→ 红=低年化（5%~95%分位映射）</p>
</div>

<footer>
生成时间: {config['generated_at']} | 数据来源: {config.get('data_source', 'factor_scan.json')}<br>
回测引擎: WBR core.backtest | 成交价: T日 open | 收益基准: preClose
</footer>

<script>
function switchTab(tabName) {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById('tab-' + tabName).classList.add('active');
}}

function filterTable() {{
    const q = document.getElementById('search').value.toLowerCase();
    document.querySelectorAll('#detail-table tbody tr').forEach(tr => {{
        tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
    }});
    document.querySelectorAll('#best-table tbody tr').forEach(tr => {{
        tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
    }});
}}

function sortTable(colIdx, tableId) {{
    const table = document.getElementById(tableId);
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const asc = table.dataset.sortCol == colIdx ? !(table.dataset.sortAsc == 'true') : true;
    rows.sort((a, b) => {{
        let va = a.cells[colIdx].textContent.replace(/[★%+¥,]/g,'').trim();
        let vb = b.cells[colIdx].textContent.replace(/[★%+¥,]/g,'').trim();
        let na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
        return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }});
    rows.forEach(r => tbody.appendChild(r));
    table.dataset.sortCol = colIdx;
    table.dataset.sortAsc = asc;
    // update sort icons
    table.querySelectorAll('th .sort-icon').forEach(s => s.textContent = '⇅');
    const icon = table.querySelectorAll('th')[colIdx].querySelector('.sort-icon');
    icon.textContent = asc ? '▲' : '▼';
}}
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"报告已生成: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="生成因子扫描 HTML 报告")
    parser.add_argument("--input", type=str, default="results/factor_scan.json")
    parser.add_argument("--output", type=str, default="results/factor_scan.html")
    parser.add_argument("--buy-n", type=int, default=20)
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: 未找到 {input_path}，请先运行 scan_factors.py")
        return

    results = json.loads(input_path.read_text(encoding="utf-8"))

    config = {
        "start_date": args.start,
        "end_date": args.end,
        "init_cash": "100万 (¥1,000,000)",
        "buy_n": args.buy_n,
        "sell_m": args.buy_n,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": str(input_path),
    }

    generate_report(results, config, Path(args.output))


if __name__ == "__main__":
    main()
