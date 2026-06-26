"""将单因子扫描结果写入因子库（factor_scans 表），并重新生成 factor_db/report.html。

用法:
    uv run python factor_db/update_scan_results.py
    uv run python factor_db/update_scan_results.py --input results/factor_scan.json
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_DB_PATH = Path(__file__).resolve().parent / "registry.db"
HOLDING_PERIODS = [1, 3, 7, 15, 30]

_SCHEMA_SCANS = """
CREATE TABLE IF NOT EXISTS factor_scans (
    scan_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name  TEXT    NOT NULL,
    holding_period INTEGER NOT NULL,
    annualized   REAL,
    max_drawdown REAL,
    sharpe       REAL,
    calmar       REAL,
    total_return REAL,
    total_trades INTEGER,
    buy_n        INTEGER NOT NULL,
    start_date   TEXT    NOT NULL,
    end_date     TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    UNIQUE(factor_name, holding_period, start_date, end_date)
);
"""


def init_scans_table():
    conn = sqlite3.connect(_DB_PATH)
    conn.executescript(_SCHEMA_SCANS)
    conn.commit()
    conn.close()


def upsert_scan(conn, record: dict):
    conn.execute(
        """
        INSERT OR REPLACE INTO factor_scans
            (factor_name, holding_period, annualized, max_drawdown, sharpe,
             calmar, total_return, total_trades, buy_n, start_date, end_date, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["factor"], record["holding_period"],
            record["annualized"], record["max_drawdown"], record["sharpe"],
            record["calmar"], record["total_return"], record["total_trades"],
            record["buy_n"], record["start_date"], record["end_date"],
            record["created_at"],
        ),
    )


def load_scan_results(input_path: Path) -> list[dict]:
    return json.loads(input_path.read_text(encoding="utf-8"))


def write_scans_to_db(results: list[dict], buy_n: int, start_date: str,
                      end_date: str):
    init_scans_table()
    conn = sqlite3.connect(_DB_PATH)
    created_at = datetime.now(timezone.utc).isoformat()

    count = 0
    for r in results:
        record = {
            **r,
            "buy_n": buy_n,
            "start_date": start_date,
            "end_date": end_date,
            "created_at": created_at,
        }
        upsert_scan(conn, record)
        count += 1

    conn.commit()
    conn.close()
    print(f"因子库已更新: {count} 条扫描记录 → factor_db/registry.db (factor_scans 表)")


def query_scans_from_db() -> list[dict]:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM factor_scans ORDER BY factor_name, holding_period"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def generate_report(results: list[dict], config: dict):
    """生成 factor_db/report.html — 覆盖因子库报告"""
    factor_names = sorted(set(r["factor"] for r in results))

    # 最优夏普
    # 每个因子的最佳持仓周期记录
    best_map = {}
    for r in results:
        prev = best_map.get(r["factor"])
        if prev is None or r["sharpe"] > prev["sharpe"]:
            best_map[r["factor"]] = r

    best_list = list(best_map.values())
    # 按因子名排序保持一致性
    best_list.sort(key=lambda x: x["factor"])

    sharpes = [r["sharpe"] for r in best_list]
    pos_count = sum(1 for s in sharpes if s > 0)
    anns = [r["annualized"] for r in best_list]
    dds = [r["max_drawdown"] for r in best_list]

    # 热力图数据
    hp_to_col = {hp: i for i, hp in enumerate(HOLDING_PERIODS)}
    f_to_row = {n: i for i, n in enumerate(factor_names)}

    def _build_heatmap(metric):
        grid = np.full((len(factor_names), len(HOLDING_PERIODS)), np.nan)
        for r in results:
            fi = f_to_row.get(r["factor"])
            hi = hp_to_col.get(r["holding_period"])
            if fi is not None and hi is not None:
                grid[fi, hi] = r[metric]
        vals = grid[~np.isnan(grid)]
        vmin, vmax = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
        if vmax <= vmin:
            vmax = vmin + 1

        cells = []
        for fi, name in enumerate(factor_names):
            row = f"<tr><td class='mc'>{name}</td>"
            for hi in range(len(HOLDING_PERIODS)):
                v = grid[fi, hi]
                if np.isnan(v):
                    row += '<td class="mc na">—</td>'
                else:
                    t = max(0, min(1, (v - vmin) / (vmax - vmin)))
                    red = int(255 * (1 - t))
                    green = int(180 * t + 30)
                    row += (
                        f'<td class="mc" style="background:rgb({red},{green},60);'
                        f'color:{"#fff" if t < 0.45 else "#000"}">{v:.2f}</td>'
                    )
            row += "</tr>"
            cells.append(row)
        return "\n".join(cells)

    heatmap_sharpe = _build_heatmap("sharpe")
    heatmap_ann = _build_heatmap("annualized")

    # 全部明细行
    detail_rows = []
    for r in results:
        best_r = best_map.get(r["factor"])
        is_best = best_r is not None and best_r["holding_period"] == r["holding_period"]
        best_mark = " ★" if is_best else ""
        ann_cls = "pos" if r["annualized"] > 0 else "neg"
        s_cls = "pos" if r["sharpe"] > 0 else "neg"
        detail_rows.append(
            f"<tr>"
            f"<td>{r['factor']}</td>"
            f"<td class='num'>{r['holding_period']}{best_mark}</td>"
            f'<td class="num {ann_cls}">{r["annualized"]:+.2f}%</td>'
            f'<td class="num">{r["max_drawdown"]:.2f}%</td>'
            f'<td class="num {s_cls}">{r["sharpe"]:.2f}</td>'
            f'<td class="num">{r["calmar"]:.2f}</td>'
            f'<td class="num {ann_cls}">{r["total_return"]:+.2f}%</td>'
            f'<td class="num">{r["total_trades"]}</td>'
            f"</tr>"
        )

    # 最优汇总行
    best_rows = []
    for r in best_list:
        s_cls = "pos" if r["sharpe"] > 0 else "neg"
        best_rows.append(
            f"<tr>"
            f"<td>{r['factor']}</td>"
            f"<td class='num'>{r['holding_period']}</td>"
            f'<td class="num {s_cls}">{r["annualized"]:+.2f}%</td>'
            f'<td class="num">{r["max_drawdown"]:.2f}%</td>'
            f'<td class="num {s_cls}">{r["sharpe"]:.2f}</td>'
            f'<td class="num">{r["calmar"]:.2f}</td>'
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>WBR 因子库 — 单因子扫描报告</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:20px;color:#1f2328;background:#fff}}
h1{{font-size:22px;margin-bottom:6px}}h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #d0d7de;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
th,td{{border:1px solid #d0d7de;padding:5px 8px}}
th{{background:#f6f8fa;position:sticky;top:0;cursor:pointer;user-select:none;white-space:nowrap}}
th:hover{{background:#ddf4ff}}
tr:nth-child(even){{background:#f6f8fa}}
tr:hover td{{background:#ddf4ff}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.pos{{color:#1a7f37;font-weight:600}}
.neg{{color:#cf222e}}
.na{{color:#8c959f;text-align:center}}
.config-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px;margin:12px 0;padding:12px;background:#f6f8fa;border-radius:8px}}
.config-item{{display:flex;justify-content:space-between;font-size:13px}}
.config-key{{color:#656d76}}
.config-val{{font-weight:600}}
.kpi{{display:inline-block;margin-right:24px}}
.kpi b{{font-size:20px}}
.kpi .pos{{color:#1a7f37}}.kpi .neg{{color:#cf222e}}
.tab{{display:inline-block;padding:6px 14px;cursor:pointer;border:1px solid #d0d7de;border-radius:6px 6px 0 0;background:#f6f8fa;margin-right:4px;font-size:13px}}
.tab.active{{background:#fff;border-bottom-color:#fff;font-weight:600}}
.tab-content{{display:none}}
.tab-content.active{{display:block}}
.matrix{{font-size:11px}}
.matrix td.mc,.matrix th.mh{{border:1px solid #d0d7de;padding:2px 5px;text-align:center;min-width:38px}}
.matrix th.mh{{background:#f6f8fa;white-space:nowrap;font-size:11px}}
.search-box{{margin:10px 0}}
.search-box input{{width:100%;padding:6px 10px;border:1px solid #d0d7de;border-radius:6px;font-size:13px}}
footer{{margin-top:30px;padding-top:12px;border-top:1px solid #d0d7de;color:#656d76;font-size:12px}}
</style>
</head>
<body>

<h1>📊 WBR 因子库 — 单因子扫描报告</h1>

<div class="config-grid">
<div class="config-item"><span class="config-key">回测区间</span><span class="config-val">{config['start_date']} ~ {config['end_date']}</span></div>
<div class="config-item"><span class="config-key">初始资金</span><span class="config-val">100万 (¥1,000,000)</span></div>
<div class="config-item"><span class="config-key">买入数 buy_n</span><span class="config-val">{config['buy_n']}</span></div>
<div class="config-item"><span class="config-key">卖出数 sell_m</span><span class="config-val">{config['buy_n']} (=buy_n)</span></div>
<div class="config-item"><span class="config-key">权重</span><span class="config-val">单因子 weight=1.0</span></div>
<div class="config-item"><span class="config-key">持仓周期</span><span class="config-val">{', '.join(map(str, HOLDING_PERIODS))} 日</span></div>
<div class="config-item"><span class="config-key">调仓方式</span><span class="config-val">均仓多退少补 (rebalance=True)</span></div>
<div class="config-item"><span class="config-key">涨停保护</span><span class="config-val">关闭 (limit_up_protection=False)</span></div>
<div class="config-item"><span class="config-key">滑点/费率</span><span class="config-val">默认配置</span></div>
<div class="config-item"><span class="config-key">生成时间</span><span class="config-val">{config['generated_at']}</span></div>
</div>

<div>
<div class="kpi">因子总数<br><b>{len(factor_names)}</b></div>
<div class="kpi">回测总数<br><b>{len(results)}</b></div>
<div class="kpi">夏普&gt;0<br><b class="pos">{pos_count}</b></div>
<div class="kpi">夏普&gt;0 占比<br><b>{pos_count/len(factor_names)*100:.0f}%</b></div>
<div class="kpi">夏普均值<br><b class="{'pos' if np.mean(sharpes)>0 else 'neg'}">{np.mean(sharpes):.2f}</b></div>
<div class="kpi">夏普中位数<br><b>{np.median(sharpes):.2f}</b></div>
<div class="kpi">年化均值<br><b class="{'pos' if np.mean(anns)>0 else 'neg'}">{np.mean(anns):.2f}%</b></div>
<div class="kpi">回撤均值<br><b>{np.mean(dds):.2f}%</b></div>
</div>

<div class="search-box">
<input type="text" id="search" placeholder="🔍 搜索因子..." oninput="filterAll()">
</div>

<div class="tab-bar">
<div class="tab active" onclick="switchTab(event, 'best')">⭐ 最优持仓周期（夏普）</div>
<div class="tab" onclick="switchTab(event, 'detail')">📋 全部明细</div>
<div class="tab" onclick="switchTab(event, 'heatmap_s')">🔥 夏普热力图</div>
<div class="tab" onclick="switchTab(event, 'heatmap_a')">📈 年化热力图</div>
</div>

<div id="tab-best" class="tab-content active">
<div style="overflow-x:auto;max-height:80vh;overflow-y:auto">
<table id="best-table">
<thead><tr>
<th onclick="sortTable(0,'best-table')">因子名称</th>
<th onclick="sortTable(1,'best-table')" class="num">最优HP</th>
<th onclick="sortTable(2,'best-table')" class="num">年化收益</th>
<th onclick="sortTable(3,'best-table')" class="num">最大回撤</th>
<th onclick="sortTable(4,'best-table')" class="num">夏普</th>
<th onclick="sortTable(5,'best-table')" class="num">卡玛</th>
</tr></thead>
<tbody>{"".join(best_rows)}</tbody>
</table>
</div>
</div>

<div id="tab-detail" class="tab-content">
<div style="overflow-x:auto;max-height:80vh;overflow-y:auto">
<table id="detail-table">
<thead><tr>
<th onclick="sortTable(0,'detail-table')">因子名称</th>
<th onclick="sortTable(1,'detail-table')" class="num">持仓周期</th>
<th onclick="sortTable(2,'detail-table')" class="num">年化收益</th>
<th onclick="sortTable(3,'detail-table')" class="num">最大回撤</th>
<th onclick="sortTable(4,'detail-table')" class="num">夏普</th>
<th onclick="sortTable(5,'detail-table')" class="num">卡玛</th>
<th onclick="sortTable(6,'detail-table')" class="num">总收益</th>
<th onclick="sortTable(7,'detail-table')" class="num">总成交</th>
</tr></thead>
<tbody>{"".join(detail_rows)}</tbody>
</table>
</div>
</div>

<div id="tab-heatmap_s" class="tab-content">
<p style="color:#656d76;font-size:12px;margin-bottom:4px">绿=高夏普 ←→ 红=低夏普（5%~95%分位映射）</p>
<div style="overflow-x:auto;max-height:80vh;overflow-y:auto">
<table class="matrix">
<thead><tr><th class="mh">因子 \\ HP</th>
{"".join(f'<th class="mh">{hp}日</th>' for hp in HOLDING_PERIODS)}
</tr></thead>
<tbody>{heatmap_sharpe}</tbody>
</table>
</div>
</div>

<div id="tab-heatmap_a" class="tab-content">
<p style="color:#656d76;font-size:12px;margin-bottom:4px">绿=高年化 ←→ 红=低年化（5%~95%分位映射）</p>
<div style="overflow-x:auto;max-height:80vh;overflow-y:auto">
<table class="matrix">
<thead><tr><th class="mh">因子 \\ HP</th>
{"".join(f'<th class="mh">{hp}日</th>' for hp in HOLDING_PERIODS)}
</tr></thead>
<tbody>{heatmap_ann}</tbody>
</table>
</div>
</div>

<footer>
WBR 因子库报告 | 回测引擎: core.backtest | 成交价: T日 open | 收益计算: preClose 基准 | 生成: {config['generated_at']}
</footer>

<script>
function switchTab(ev, id) {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    ev.target.classList.add('active');
    document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    document.getElementById('tab-' + id).classList.add('active');
}}
function filterAll() {{
    const q = document.getElementById('search').value.toLowerCase();
    ['best-table','detail-table'].forEach(tid => {{
        const t = document.getElementById(tid);
        if (!t) return;
        t.querySelectorAll('tbody tr').forEach(tr => {{
            tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
        }});
    }});
}}
function sortTable(colIdx, tableId) {{
    const table = document.getElementById(tableId);
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const asc = table.dataset.sortCol == colIdx ? !(table.dataset.sortAsc == 'true') : true;
    rows.sort((a, b) => {{
        let va = a.cells[colIdx].textContent.replace(/[★%+]/g,'').trim();
        let vb = b.cells[colIdx].textContent.replace(/[★%+]/g,'').trim();
        let na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
        return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }});
    rows.forEach(r => tbody.appendChild(r));
    table.dataset.sortCol = colIdx;
    table.dataset.sortAsc = asc;
}}
</script>
</body>
</html>"""

    output_path = Path(__file__).resolve().parent / "report.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"因子库报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="更新因子库扫描结果")
    parser.add_argument("--input", type=str, default="results/factor_scan.json")
    parser.add_argument("--buy-n", type=int, default=20)
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: 未找到 {input_path}，请先运行 scan_factors.py")
        return

    results = load_scan_results(input_path)
    print(f"加载 {len(results)} 条扫描结果")

    # 写入因子库
    write_scans_to_db(results, args.buy_n, args.start, args.end)

    # 生成报告
    config = {
        "start_date": args.start,
        "end_date": args.end,
        "buy_n": args.buy_n,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    generate_report(results, config)


if __name__ == "__main__":
    main()
