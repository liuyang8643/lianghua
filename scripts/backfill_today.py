"""一次性数据回填 —— 从 QMT 拉当日全量成交补齐 fills/events，重算 snapshot 与日终摘要，
重生成 diff 报告。修复 watcher 漏接 callback 导致的历史数据缺口。

精准修复条件：QMT 客户端在线 + 目标日是「QMT 当日交易日」（QMT 内部账本只保留当日）。
跨日回填理论上不可行（QMT 不返历史成交明细）；如果今天已开新交易日，本脚本只能修今天。

用法：
  python scripts/backfill_today.py                       # 修复今天
  python scripts/backfill_today.py --date 2026-05-28     # 修复指定日（仍需 QMT 当日为该日）
  python scripts/backfill_today.py --no-report           # 仅修 parquet，不重生成 HTML/飞书
"""
from __future__ import annotations
import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from configs import TRADE_ACCOUNT
from trading.logger import trading_logger
from trading.persistence import live_trade_recorder
from trading.trader import Trader

_TRADE_DIR = ROOT / "data" / "live_trades"
_REPORT_DIR = _TRADE_DIR / "reports"


def _backup(path: Path) -> Path | None:
    """备份 parquet 到 .bak.{ts}，便于回滚。"""
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + f".bak.{int(pd.Timestamp.now().timestamp())}")
    bak.write_bytes(path.read_bytes())
    trading_logger.info(f"  备份: {path.name} → {bak.name}")
    return bak


def backfill_one_day(target: date, *, gen_report: bool = True,
                     send_lark: bool = False) -> int:
    """完整回填一天的数据，返回总修复条数。"""
    trading_logger.info(f"========== 回填 {target} ==========")

    # 1. 备份现有 parquet（先备份再修，留退路）
    for fname in [
        f"fills_{target.isoformat()}.parquet",
        f"events_{target.isoformat()}.parquet",
        f"positions_{target.isoformat()}.parquet",
        "daily_summary.parquet",
    ]:
        _backup(_TRADE_DIR / fname)

    # 2. 连 QMT
    trading_logger.info("[1/5] 连接 QMT…")
    try:
        td = Trader(TRADE_ACCOUNT)
    except Exception as e:
        trading_logger.error(f"QMT 连接失败：{e}。请确认 XtMiniQmt 已启动")
        return 0

    # 3. 从 QMT query_stock_trades 全量成交补 fills/events
    trading_logger.info("[2/5] QMT 全量成交回填…")
    n_new = live_trade_recorder.backfill_fills_from_qmt(td, trade_date=target)
    trading_logger.info(f"  → 新增 {n_new} 条 fills/events")

    # 4. 重新查持仓 + 重算 snapshot（含 daily_pnl）
    trading_logger.info("[3/5] 重算 positions snapshot + daily_pnl…")
    positions = td.query_positions() or []
    fills_df = live_trade_recorder.get_today_fills_df(trade_date=target)
    live_trade_recorder.snapshot_positions(
        positions=positions, fills_df=fills_df, trade_date=target)

    # 5. 重写 daily_summary
    trading_logger.info("[4/5] 重写 daily_summary…")
    asset = td.query_asset()
    if asset:
        live_trade_recorder.write_daily_summary(
            total_asset=float(asset.total_asset),
            cash=float(asset.cash),
            market_value=float(asset.market_value),
            trade_date=target,
        )

    # 6. 重生成 diff 报告（可选）
    if gen_report:
        trading_logger.info("[5/5] 重生成 diff 报告（HTML + 飞书）…")
        _regen_report(target, send_lark=send_lark)
    else:
        trading_logger.info("[5/5] 跳过报告重生成（--no-report）")

    return n_new


def _regen_report(target: date, *, send_lark: bool):
    """基于修复后的 parquet 重建 PostCloseReport。"""
    from trading.report import PostCloseReport
    from trading.persistence import live_trade_recorder

    report = PostCloseReport(target)

    pos_path = _TRADE_DIR / f"positions_{target.isoformat()}.parquet"
    if pos_path.exists():
        report.feed_positions_df(pd.read_parquet(pos_path))

    plan_path = _TRADE_DIR / f"plan_{target.isoformat()}.parquet"
    if plan_path.exists():
        report.feed_plan_df(pd.read_parquet(plan_path))

    fills_df = live_trade_recorder.get_today_fills_df(trade_date=target)
    if not fills_df.empty:
        report.feed_fills_df(fills_df)

    summary_path = _TRADE_DIR / "daily_summary.parquet"
    if summary_path.exists():
        s = pd.read_parquet(summary_path)
        cur = s[s['date'] == target]
        prev_rows = s[s['date'] < target]
        live_asset = float(cur['total_asset'].iloc[-1]) if not cur.empty else None
        prev_asset = float(prev_rows['total_asset'].iloc[-1]) if not prev_rows.empty else None
        net_cf = float(cur['net_cash_flow'].iloc[-1]) if not cur.empty else 0.0
        report.feed_asset(live_asset, prev_asset, net_cash_flow=net_cf)

    # 回测结果：单日回测可用 testback.run_ga single 模式，此处先 stub 为空 dict
    # 用户已有 bt_result 的话可以扩展 --bt-result-pickle 参数加载
    report.feed_backtest({'daily_snapshots': [{}]})

    out_html = _REPORT_DIR / f"diff_{target.isoformat()}_backfilled.html"
    report.to_html(out_html)
    trading_logger.info(f"  → HTML: {out_html}")

    if send_lark:
        report.send(html_path=out_html)
        trading_logger.info("  → 飞书卡片已发送")


def main():
    parser = argparse.ArgumentParser(description="QMT 当日成交回填 + 报告重生成")
    parser.add_argument('--date', type=str, default=None,
                        help='目标日 YYYY-MM-DD，默认今天。仅当 QMT 当日为该日时可成功。')
    parser.add_argument('--no-report', action='store_true', help='只修 parquet，不重生成报告')
    parser.add_argument('--send-lark', action='store_true', help='重生成报告后推送飞书卡片')
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    n = backfill_one_day(target, gen_report=not args.no_report, send_lark=args.send_lark)

    trading_logger.info(f"\n========== 回填完成（新增 {n} 条）==========")
    trading_logger.info("建议跑 `python scripts/verify_reports.py` 验证恒等式")


if __name__ == "__main__":
    main()
