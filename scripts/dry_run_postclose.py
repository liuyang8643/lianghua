"""离线 dry-run：用已有的 parquet 数据 + 当日单日回测，生成 5 维 Diff 报告。

不依赖 QMT、不联网。可以随时跑历史日期的 diff。

用法:
    python scripts/dry_run_postclose.py 2026-05-27 [--config configs/single_tmc_pure.json] [--src-dir data/live_trades_rebuilt_2026-05-27]

参数:
    DATE          目标日期 YYYY-MM-DD
    --config      个体配置 JSON
    --src-dir     parquet 源目录（默认 data/live_trades_rebuilt_{DATE}/）
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ga import get_profile_factor_classes, resolve_profile_name
from trading.post_close import _run_single_day_backtest
from trading.report import PostCloseReport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('target_date', type=str, help='目标日期 YYYY-MM-DD')
    parser.add_argument('--config', type=str,
                        default='configs/single_tmc_pure.json',
                        help='个体配置 JSON 路径')
    parser.add_argument('--src-dir', type=str, default=None,
                        help='parquet 源目录（默认 data/live_trades_rebuilt_{DATE}/）')
    parser.add_argument('--send-lark', action='store_true',
                        help='发送飞书卡片+HTML 附件（默认只写本地 HTML）')
    args = parser.parse_args()

    target = datetime.strptime(args.target_date, '%Y-%m-%d').date()
    src_dir = Path(args.src_dir) if args.src_dir else (
        ROOT / "data" / f"live_trades_rebuilt_{args.target_date}"
    )
    if not src_dir.exists():
        print(f"❌ {src_dir} 不存在。请先跑:")
        print(f"   python scripts/rebuild_live_trades_for_diff.py {args.target_date}")
        sys.exit(1)

    print(f"=== 离线 Dry-Run Diff @ {target} ===")
    print(f"源目录: {src_dir}")

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    profile = resolve_profile_name(cfg)
    factor_classes = get_profile_factor_classes(profile)
    individual_config = cfg['individual_config']
    print(f"配置: profile={profile} buy_n={individual_config['buy_n']} factors={sorted(individual_config['weights'].keys())}")

    # 1. 单日回测
    print(f"\n[1/3] 单日回测 {target} ...")
    bt_result = _run_single_day_backtest(target, individual_config, factor_classes)
    if bt_result is None:
        print("❌ 单日回测失败（NPZ 中可能无当日数据）")
        sys.exit(2)
    bt_snap = (bt_result.get('daily_snapshots') or [{}])[0]
    print(f"   回测: {len(bt_snap.get('raw_buy_n_list', []))} 候选 → "
          f"{len(bt_snap.get('buy_n_list', []))} 可交易 → "
          f"{len(bt_snap.get('executed_buy_list', []))} 已执行")

    # 2. 加载实盘 parquet
    print(f"\n[2/3] 加载实盘 parquet ...")
    plan_df = pd.read_parquet(src_dir / f"plan_{args.target_date}.parquet")
    fills_df = pd.read_parquet(src_dir / f"fills_{args.target_date}.parquet")
    positions_df = pd.read_parquet(src_dir / f"positions_{args.target_date}.parquet")
    print(f"   plan: {len(plan_df)} 行 | fills: {len(fills_df)} 行 | positions: {len(positions_df)} 行")

    # 3. 资产层（从 daily_summary 反查）
    daily_sum_path = ROOT / "data" / "live_trades" / "daily_summary.parquet"
    live_asset = prev_asset = net_cf = None
    if daily_sum_path.exists():
        ds = pd.read_parquet(daily_sum_path)
        today_rows = ds[ds['date'].astype(str) == args.target_date]
        prev_rows = ds[ds['date'].astype(str) < args.target_date]
        if not today_rows.empty:
            live_asset = float(today_rows['total_asset'].iloc[-1])
            net_cf = float(today_rows['net_cash_flow'].iloc[-1]) if 'net_cash_flow' in today_rows.columns else 0.0
        if not prev_rows.empty:
            prev_asset = float(prev_rows['total_asset'].iloc[-1])
    print(f"   资产: live={live_asset} prev={prev_asset} net_cf={net_cf}")

    # 4. 组装报告
    print(f"\n[3/3] 组装 PostCloseReport ...")
    report = PostCloseReport(target)
    report.feed_plan_df(plan_df)
    report.feed_fills_df(fills_df)
    report.feed_positions_df(positions_df)
    report.feed_backtest(bt_result)
    report.feed_asset(live_asset, prev_asset, net_cash_flow=net_cf or 0.0)

    data = report.build()

    # 终端摘要
    print()
    print("━" * 60)
    s = data['summary']
    print(f"📊 账户:   实盘 ¥{s['live_total_asset']:,.0f} | 回测 ¥{s['bt_total_asset']:,.0f}")
    if s['live_daily_pnl'] is not None:
        print(f"💰 日P&L:  实盘 {s['live_daily_pnl']:+,.0f} ({s['live_daily_return_pct']:+.2f}%) "
              f"| 回测 {s['bt_daily_pnl']:+,.0f} ({s['bt_daily_return_pct']:+.2f}%)")
    d1, d2 = data['dim1_candidates'], data['dim2_tradable']
    print(f"🔍 维度1 候选股: 实盘 {d1['live_count']} | 回测 {d1['bt_count']} | 匹配率 {d1['match_rate']*100:.0f}%")
    print(f"🔍 维度2 可交易: 实盘 {d2['live_count']} | 回测 {d2['bt_count']} | 匹配率 {d2['match_rate']*100:.0f}%")
    d3, d4, d5 = data['dim3_orders'], data['dim4_slippage'], data['dim5_pnl']
    print(f"📦 维度3 订单:   实盘买入 ¥{d3['live_buy_total']:,.0f} | 回测买入 ¥{d3['bt_buy_total']:,.0f}")
    print(f"📉 维度4 滑点:   平均 {d4['avg_slippage']:+.3f}% | 总成本 ¥{d4['total_slippage_cost']:+,.0f} ({len(d4['rows'])} 笔)")
    print(f"💹 维度5 日P&L: 实盘合计 ¥{d5['live_total_pnl']:+,.0f} | 回测合计 ¥{d5['bt_total_pnl']:+,.0f} ({len(d5['rows'])} 只)")
    print("━" * 60)

    # HTML 报告
    out = ROOT / "data" / "live_trades" / "reports" / f"diff_{args.target_date}_dryrun.html"
    report.to_html(out)
    print(f"\n📄 HTML 报告: {out}")

    if args.send_lark:
        print("\n📱 发送飞书卡片 + HTML 附件 ...")
        report.send(html_path=out, attach_html=True)
        print("   完成。")


if __name__ == '__main__':
    main()
