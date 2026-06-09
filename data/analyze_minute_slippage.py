"""分析开盘价 vs 09:31-09:35 各分钟收盘价的价差分布。

数据: data/minute/{code}.parquet (每只股票91交易日×5分钟)
产物: data/minute/slippage_stats.parquet

用法: uv run python -m data.analyze_minute_slippage
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MINUTE_DIR = ROOT / "data" / "minute"

MINUTE_LABELS = ["09:31", "09:32", "09:33", "09:34", "09:35"]


def _log(msg: str):
    print(f"[{pd.Timestamp.now():%H:%M:%S}] {msg}", flush=True)


def compute_slippage():
    files = sorted(MINUTE_DIR.glob("*.parquet"))
    if not files:
        _log("没有分钟数据，请先运行 download_minute")
        return

    records = []
    total = len(files)

    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        df["date"] = df["time"].dt.strftime("%Y%m%d").astype(int)
        df["minute"] = df["time"].dt.strftime("%H:%M")

        # 每日开盘价 = 当日第一根bar (09:31) 的 open
        open_ref = df[df["minute"] == "09:31"].set_index("date")["open"]

        for m in MINUTE_LABELS:
            mdf = df[df["minute"] == m].set_index("date")
            common = mdf.index.intersection(open_ref.index)
            if len(common) < 10:
                continue
            close_vals = mdf.loc[common, "close"]
            open_vals = open_ref.loc[common]
            slip = (close_vals - open_vals) / open_vals  # 小数
            records.append({
                "code": f.stem,
                "minute": m,
                "mean": float(slip.mean()),
                "std": float(slip.std()),
                "p5": float(slip.quantile(0.05)),
                "p25": float(slip.quantile(0.25)),
                "p50": float(slip.quantile(0.50)),
                "p75": float(slip.quantile(0.75)),
                "p95": float(slip.quantile(0.95)),
                "n": len(slip),
            })

        if (i + 1) % 1000 == 0:
            _log(f"进度 {i+1}/{total}")

    stats = pd.DataFrame(records)
    out_path = MINUTE_DIR / "slippage_stats.parquet"
    stats.to_parquet(out_path, index=False)
    _log(f"完成: {len(stats)} 条记录 → {out_path}")

    # === 汇总报告 ===
    print("\n全市场 open → 分钟close 价差 (%):")
    print(f"{'minute':<8} {'mean':>8} {'median':>8} {'p25':>8} {'p75':>8} {'n(avg)':>8}")
    print("-" * 48)
    for m in MINUTE_LABELS:
        sub = stats[stats["minute"] == m]
        print(f"{m:<8} {sub['mean'].mean()*100:>+7.3f}% {sub['p50'].median()*100:>+7.3f}% "
              f"{sub['p25'].median()*100:>+7.3f}% {sub['p75'].median()*100:>+7.3f}% "
              f"{sub['n'].mean():>7.0f}")

    print("\n累积效应 (每日1笔, 91交易日, 中位数价差):")
    for m in MINUTE_LABELS:
        sub = stats[stats["minute"] == m]
        slip = sub["p50"].median()
        cum = (1 + slip) ** 91 - 1
        print(f"  {m}: 每笔 {slip*100:+.4f}% → 91日累积 {cum*100:+.3f}%")

    return stats


if __name__ == "__main__":
    compute_slippage()
