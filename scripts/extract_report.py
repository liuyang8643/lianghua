"""从单回测 HTML 报告提取核心指标 + 分年表现。

用法: python scripts/extract_report.py <report_html_or_dir>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np


def _find_html(p: Path) -> Path:
    if p.is_dir():
        cands = list(p.glob('single_report.html')) or list(p.glob('backtest_report_*.html'))
        return sorted(cands)[-1]
    return p


def main():
    p = _find_html(Path(sys.argv[1]))
    raw = p.read_bytes()
    m = re.search(rb'<script id="report-data" type="application/json">(.*?)</script>', raw, re.S)
    d = json.loads(m.group(1))
    s = d['summary']
    meta = d['meta']
    print(f'== {p} ==')
    print(f"区间   {meta.get('period_start')} ~ {meta.get('period_end')}  ({s.get('total_days')}日)")
    print(f"总收益 {s.get('total_return_pct')}%   年化 {s.get('annualized_return_pct')}%")
    print(f"夏普   {s.get('sharpe_ratio')}   最大回撤 {s.get('max_drawdown_pct')}%   卡玛 {s.get('calmar_ratio')}")
    print(f"胜率   {s.get('win_rate_pct')}%   round_trips={s.get('round_trips')}  手续费={s.get('total_commission')}")

    eq = d['charts']['equity']
    dates = eq['trade_dates']
    nav = np.array(eq['strategy_nav'], dtype=float)
    years = sorted(set(dt[:4] for dt in dates))
    print('\n分年: year  return%   sharpe  maxDD%   days')
    sharpes = []
    for y in years:
        idx = [i for i, dt in enumerate(dates) if dt[:4] == y]
        if len(idx) < 5:
            continue
        f, l = idx[0], idx[-1]
        start_nav = nav[f - 1] if f > 0 else 1.0
        yn = np.concatenate([[start_nav], nav[f:l + 1]])
        ret = (yn[-1] / yn[0] - 1) * 100
        dr = yn[1:] / yn[:-1] - 1
        sh = float(np.mean(dr) / np.std(dr, ddof=1) * np.sqrt(252)) if len(dr) > 1 and np.std(dr, ddof=1) > 0 else 0.0
        dd = float(np.min(yn / np.maximum.accumulate(yn) - 1) * 100)
        sharpes.append(sh)
        print(f"  {y}  {ret:8.2f}  {sh:6.2f}  {dd:7.2f}   {len(idx)}")
    if sharpes:
        print(f"  分年夏普均值={np.mean(sharpes):.2f}  正夏普年数={sum(1 for x in sharpes if x>0)}/{len(sharpes)}")


if __name__ == '__main__':
    main()
