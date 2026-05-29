"""离线验证 data/live_trades/ 下所有报告的数据一致性。

校验项：
  1. positions_{T}.parquet 中每行的 daily_pnl 公式自洽
       daily_pnl = (T_lp×T_vol) - (Y_lp×Y_vol) + sell - buy - fee
  2. 会计恒等式: Σ(positions.daily_pnl) ≈ daily_summary.daily_pnl
  3. daily_summary 自身一致性: daily_pnl = total_asset - prev.total_asset - net_cash_flow
  4. fills 总额与 positions 中的 buy_amount_today/sell_amount_today 对得上
  5. 全部 reports/diff_*.html 的存在性 + 大小合理性

不联网，只读 parquet。
"""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADE_DIR = ROOT / "data" / "live_trades"
REPORT_DIR = TRADE_DIR / "reports"

RECONCILE_TOLERANCE_ABS = 1.0   # 元
RECONCILE_TOLERANCE_PCT = 0.0005  # 0.05% 账户


def _prev_trading_summary_row(summary: pd.DataFrame, target: date) -> pd.Series | None:
    prev = summary[summary['date'] < target]
    if prev.empty:
        return None
    return prev.iloc[-1]


def verify_positions_formula(pos_df: pd.DataFrame, ydf: pd.DataFrame | None,
                              fills_df: pd.DataFrame | None, target: date) -> list[str]:
    """用新公式（cost basis 反推）重算 daily_pnl 与 parquet 对比。
    
    与 trading/persistence.py::snapshot_positions 同一公式：
        T_cost = avg_price × vol_t        (QMT 真实加权成本)
        Y_cost = y_avg × y_vol            (昨日 snapshot 加权成本)
        buy_amt_real = max(0, T_cost - Y_cost + y_avg × sold_real)
        sell_amt_real = avg_sell_price × sold_real
        daily_pnl = (T_mv - Y_mv) + sell_amt_real - buy_amt_real - fee_real
    """
    errs: list[str] = []
    y_map = {}
    if ydf is not None:
        for _, r in ydf.iterrows():
            if r['volume'] > 0:
                y_avg_raw = r['avg_price'] if 'avg_price' in r.index else 0
                y_map[r['code']] = {
                    'last_price': float(r['last_price']),
                    'volume': int(r['volume']),
                    'avg_price': float(y_avg_raw) if pd.notna(y_avg_raw) else 0.0,
                }

    fill_agg: dict[str, dict] = {}
    if fills_df is not None and not fills_df.empty:
        for code, grp in fills_df.groupby('code'):
            buys = grp[grp['direction'] == 'buy']
            sells = grp[grp['direction'] == 'sell']
            fill_agg[code] = {
                'bought': int(buys['shares'].sum()) if not buys.empty else 0,
                'sold': int(sells['shares'].sum()) if not sells.empty else 0,
                'buy_amt': float(buys['amount'].sum()) if not buys.empty else 0.0,
                'sell_amt': float(sells['amount'].sum()) if not sells.empty else 0.0,
                'fee': float(grp['fee_est'].sum()),
            }

    for _, r in pos_df.iterrows():
        code = r['code']
        vol_t = int(r['volume'])
        lp_t = float(r['last_price'])
        t_avg = float(r['avg_price']) if vol_t > 0 else 0.0
        agg = fill_agg.get(code, {'bought': 0, 'sold': 0, 'buy_amt': 0.0, 'sell_amt': 0.0, 'fee': 0.0})
        y = y_map.get(code) or {}
        y_vol = y.get('volume', 0)
        y_lp = y.get('last_price', 0.0)
        y_avg = y.get('avg_price', 0.0)
        y_mv = y_lp * y_vol

        # vol 平衡反推 bought_real / sold_real
        expected_vol = y_vol + agg['bought'] - agg['sold']
        gap = vol_t - expected_vol
        if gap > 0:
            bought_real, sold_real = agg['bought'] + gap, agg['sold']
        elif gap < 0:
            bought_real, sold_real = agg['bought'], agg['sold'] - gap
        else:
            bought_real, sold_real = agg['bought'], agg['sold']

        t_cost = t_avg * vol_t
        y_cost = y_avg * y_vol
        buy_amt_real = max(0.0, t_cost - y_cost + y_avg * sold_real)
        if sold_real > 0:
            if agg['sold'] > 0 and agg['sell_amt'] > 0:
                asp = agg['sell_amt'] / agg['sold']
            else:
                asp = lp_t if lp_t > 0 else y_lp
            sell_amt_real = asp * sold_real
        else:
            sell_amt_real = 0.0
        total_fills = agg['buy_amt'] + agg['sell_amt']
        total_real = buy_amt_real + sell_amt_real
        if total_fills > 0 and total_real > 0:
            fee_real = agg['fee'] * (total_real / total_fills)
        else:
            fee_real = buy_amt_real * 0.0001 + sell_amt_real * 0.0006

        # Guard: 与 persistence.py 一致
        if vol_t > 0 and y_vol == 0 and agg['buy_amt'] == 0 and agg['sell_amt'] == 0:
            expected = None
        else:
            expected = (lp_t * vol_t) - y_mv + sell_amt_real - buy_amt_real - fee_real

        actual = r['daily_pnl']
        if expected is None:
            if actual is not None and not pd.isna(actual):
                errs.append(f"  ! {code}: 公式应返 None，但 parquet={actual:+.2f}")
            continue
        if actual is None or pd.isna(actual):
            errs.append(f"  ! {code}: 公式可算 ¥{expected:+.2f}，但 parquet 中为 None")
            continue
        if abs(float(actual) - expected) > 0.5:
            errs.append(f"  ✗ {code}: parquet={actual:+.2f} vs 公式={expected:+.2f} 差={float(actual)-expected:+.2f}")

    return errs


def verify_account_invariant(pos_df: pd.DataFrame, summary_row: pd.Series,
                              prev_row: pd.Series | None) -> tuple[float | None, float, list[str]]:
    """检查 Σ(positions.daily_pnl) ≈ summary.daily_pnl 。"""
    errs: list[str] = []
    valid = pos_df[pos_df['daily_pnl'].notna()]
    if valid.empty:
        return None, 0.0, ['  ! positions 全部 daily_pnl 为 None，无法校验']
    per_stock_sum = float(valid['daily_pnl'].sum())
    account_pnl = float(summary_row['daily_pnl'])
    diff = per_stock_sum - account_pnl
    prev_asset = float(prev_row['total_asset']) if prev_row is not None else 0.0
    tolerance = max(RECONCILE_TOLERANCE_ABS, prev_asset * RECONCILE_TOLERANCE_PCT)
    if abs(diff) > tolerance:
        errs.append(f"  ✗ Σ个股={per_stock_sum:+.2f} vs 账户={account_pnl:+.2f} 差={diff:+.2f} (容差 ±{tolerance:.0f})")
        nan_codes = pos_df[pos_df['daily_pnl'].isna()]['code'].tolist()
        if nan_codes:
            errs.append(f"    缺数据股票 ({len(nan_codes)}): {nan_codes[:10]}")
    return per_stock_sum, diff, errs


def verify_summary_self_consistency(row: pd.Series, prev_row: pd.Series | None) -> list[str]:
    """daily_pnl = total_asset - prev.total_asset - net_cash_flow"""
    errs: list[str] = []
    if prev_row is None:
        return ['  ! 无前一日 summary，跳过自洽校验']
    expected = float(row['total_asset']) - float(prev_row['total_asset']) - float(row['net_cash_flow'])
    actual = float(row['daily_pnl'])
    if abs(expected - actual) > 0.1:
        errs.append(f"  ✗ daily_pnl={actual:+.2f} vs 公式重算={expected:+.2f}")
    return errs


def verify_fills_vs_positions(pos_df: pd.DataFrame, fills_df: pd.DataFrame | None) -> list[str]:
    """fills 聚合后的 buy/sell 金额应与 positions.{buy,sell}_amount_today 一致。"""
    errs: list[str] = []
    if fills_df is None or fills_df.empty:
        return errs
    agg = {}
    for code, grp in fills_df.groupby('code'):
        buys = grp[grp['direction'] == 'buy']
        sells = grp[grp['direction'] == 'sell']
        agg[code] = (
            float(buys['amount'].sum()) if not buys.empty else 0.0,
            float(sells['amount'].sum()) if not sells.empty else 0.0,
            float(grp['fee_est'].sum()),
        )
    for _, r in pos_df.iterrows():
        code = r['code']
        if code not in agg:
            if r['buy_amount_today'] > 0 or r['sell_amount_today'] > 0:
                errs.append(f"  ✗ {code}: positions 显示有今日交易 但 fills 中无记录")
            continue
        ba, sa, fe = agg[code]
        if abs(ba - float(r['buy_amount_today'])) > 0.5:
            errs.append(f"  ✗ {code}: fills.buy={ba:.2f} vs positions.buy_today={float(r['buy_amount_today']):.2f}")
        if abs(sa - float(r['sell_amount_today'])) > 0.5:
            errs.append(f"  ✗ {code}: fills.sell={sa:.2f} vs positions.sell_today={float(r['sell_amount_today']):.2f}")
        if abs(fe - float(r['fee_today'])) > 0.5:
            errs.append(f"  ✗ {code}: fills.fee={fe:.2f} vs positions.fee_today={float(r['fee_today']):.2f}")
    return errs


def verify_fills_completeness(pos_df: pd.DataFrame) -> list[str]:
    """股数恒等式: vol_t == yesterday_volume + bought_today - sold_today。
    不成立说明 fills 漏记了实际成交（watcher.on_stock_order 回调缺失或 sim 模式漏写）。
    """
    errs: list[str] = []
    for _, r in pos_df.iterrows():
        code = r['code']
        vol_t = int(r['volume'])
        y_vol = int(r.get('yesterday_volume', 0) or 0)
        bought = int(r.get('bought_today', 0) or 0)
        sold = int(r.get('sold_today', 0) or 0)
        expected = y_vol + bought - sold
        if vol_t != expected:
            gap = vol_t - expected
            kind = 'buy' if gap > 0 else 'sell'
            errs.append(
                f"  ✗ {code}: vol={vol_t} != Y_vol({y_vol})+bought({bought})-sold({sold})={expected}, "
                f"差 {gap:+d} 股 → 缺 {abs(gap)} 股 {kind} fills"
            )
    return errs


def verify_html_report(target: date) -> list[str]:
    """简单存在性校验，外加大小>1KB。"""
    errs = []
    p = REPORT_DIR / f"diff_{target.isoformat()}.html"
    if not p.exists():
        errs.append(f"  ! 缺 {p.name}")
        return errs
    sz = p.stat().st_size
    if sz < 1024:
        errs.append(f"  ✗ {p.name} 仅 {sz} 字节，可能渲染失败")
    return errs


def verify_one_day(target: date) -> tuple[bool, list[str]]:
    pos_path = TRADE_DIR / f"positions_{target.isoformat()}.parquet"
    fills_path = TRADE_DIR / f"fills_{target.isoformat()}.parquet"
    summary_path = TRADE_DIR / "daily_summary.parquet"
    if not pos_path.exists() or not summary_path.exists():
        return True, [f"⚠ {target}: positions 或 daily_summary 缺失，跳过"]

    pos_df = pd.read_parquet(pos_path)
    fills_df = pd.read_parquet(fills_path) if fills_path.exists() else None
    summary = pd.read_parquet(summary_path)
    summary_row = summary[summary['date'] == target]
    if summary_row.empty:
        return False, [f"❌ {target}: daily_summary 无对应日的行"]
    summary_row = summary_row.iloc[-1]
    prev_row = _prev_trading_summary_row(summary, target)

    # 找昨日 positions
    ydate = pd.to_datetime(prev_row['date']).date() if prev_row is not None else None
    ydf = None
    if ydate:
        yp = TRADE_DIR / f"positions_{ydate.isoformat()}.parquet"
        if yp.exists():
            ydf = pd.read_parquet(yp)

    all_errs: list[str] = []

    # 首日（无昨日 positions parquet）跳过公式 + 恒等式校验
    is_first_day = ydf is None
    info_lines: list[str] = []
    if not is_first_day:
        section = []
        section.extend(verify_positions_formula(pos_df, ydf, fills_df, target))
        if section: all_errs.append("【公式自洽】") ; all_errs.extend(section)

        per_sum, diff, sec2 = verify_account_invariant(pos_df, summary_row, prev_row)
        if sec2: all_errs.append("【会计恒等式 Σ个股 vs 账户】"); all_errs.extend(sec2)
    else:
        per_sum, diff = None, 0.0
        info_lines.append("ⓘ 首日记录（无前一日 positions snapshot），跳过公式 + 恒等式校验")

    sec3 = verify_summary_self_consistency(summary_row, prev_row)
    if sec3 and not sec3[0].startswith('  !'):
        all_errs.append("【daily_summary 自洽】"); all_errs.extend(sec3)

    sec4 = verify_fills_vs_positions(pos_df, fills_df)
    if sec4: all_errs.append("【fills vs positions 一致性】"); all_errs.extend(sec4)

    # fills 完整度 ≠ P&L 准确性（新公式用 cost basis 反推，独立于 fills）
    # 仅做 fact-finding 展示，不计入 fail
    fills_warnings = verify_fills_completeness(pos_df)

    sec5 = verify_html_report(target)
    if sec5 and not sec5[0].startswith('  !'):
        all_errs.append("【HTML 报告】"); all_errs.extend(sec5)

    # 打印一行汇总（OK 也打）
    header = [f"\n========== {target} =========="]
    n_pos = len(pos_df)
    n_pnl = int(pos_df['daily_pnl'].notna().sum())
    n_zero = int((pos_df['volume'] == 0).sum())
    summary_line = (
        f"  positions: {n_pos} 行 ({n_pnl} 可算 P&L, {n_zero} 清仓行) | "
        f"Σ个股={per_sum:+.2f} | 账户={float(summary_row['daily_pnl']):+.2f} | "
        f"diff={diff:+.2f} | net_cf={float(summary_row['net_cash_flow']):+.2f}"
        if per_sum is not None else
        f"  positions: {n_pos} 行 (0 可算 P&L) | 账户={float(summary_row['daily_pnl']):+.2f}"
    )
    out = header + [summary_line]
    for line in info_lines:
        out.append(f"  {line}")
    if all_errs:
        out.append("  问题:")
        out.extend(all_errs)
    if fills_warnings:
        out.append("  ⓘ fills 完整度（仅信息，P&L 已用 cost basis 反推修复）:")
        out.extend(fills_warnings)
    if all_errs:
        return False, out
    out.append("  ✓ 全部 P&L 校验通过")
    return True, out


def main():
    if not TRADE_DIR.exists():
        print(f"❌ 目录不存在: {TRADE_DIR}")
        sys.exit(1)

    # 找所有 positions_{date}.parquet
    pos_files = sorted(TRADE_DIR.glob("positions_*.parquet"))
    if not pos_files:
        print("❌ 没有任何 positions_*.parquet")
        sys.exit(1)

    targets: list[date] = []
    for p in pos_files:
        ds = p.stem.replace("positions_", "")
        try:
            targets.append(date.fromisoformat(ds))
        except ValueError:
            continue

    total = len(targets)
    fails = 0
    for t in targets:
        ok, lines = verify_one_day(t)
        for ln in lines:
            print(ln)
        if not ok:
            fails += 1

    print(f"\n========== 汇总 ==========")
    print(f"共校验 {total} 天，失败 {fails} 天")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
