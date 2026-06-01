"""从 events_{date}.parquet 重建 fills_{date}.parquet。

背景：旧版 fills 去重键是 (order_id, price, shares)，会把「同一委托拆成多笔等量同价
成交」（如 800 股市价单成交为 4×200@同价）误判为重复而吞掉，导致 fills 少记成交量，
dim3/飞书的成交金额随之偏小。events 流以微秒级 ts 区分，是完整的，可据此重建 fills。

本脚本：
  1. 读 events_{date}.parquet，取 event_type=='trade' 的全部成交；
  2. 逐笔还原成 fills 行（est_price 取自 plan_{date}，slippage/fee 重算）；
  3. 备份原 fills_{date}.parquet → .bak，再覆盖写入完整版。

用法:
    python scripts/rebuild_fills_from_events.py 2026-06-01 [--dir data/live_trades]
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xtquant import xtconstant
from trading.persistence import FILL_COLS, EVT_TRADE


def _fee_est(amount: float, direction: str) -> float:
    """与 persistence.record_event 派生 fill 的费用口径一致。"""
    fee = max(amount * 0.0000854, 0.1) + amount * 0.00002
    if direction == 'sell':
        fee += amount * 0.0005
    return round(fee, 4)


def _load_plan_est_prices(live_dir: Path, date_str: str) -> dict[str, float]:
    plan_path = live_dir / f"plan_{date_str}.parquet"
    if not plan_path.exists():
        return {}
    df = pd.read_parquet(plan_path)
    est = {}
    for _, r in df.iterrows():
        code = r['code']
        ep = float(r.get('est_price', 0) or 0)
        if code not in est and ep > 0:
            est[code] = ep
    return est


def rebuild(date_str: str, live_dir: Path) -> pd.DataFrame:
    events_path = live_dir / f"events_{date_str}.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"找不到 {events_path}")

    ev = pd.read_parquet(events_path)
    trades = ev[ev['event_type'] == EVT_TRADE].copy()
    if trades.empty:
        print(f"⚠️  {events_path.name} 中无 trade 事件")
        return pd.DataFrame(columns=FILL_COLS)

    est_prices = _load_plan_est_prices(live_dir, date_str)
    target = datetime.strptime(date_str, '%Y-%m-%d').date()

    rows = []
    for _, r in trades.iterrows():
        direction = r.get('direction')
        if not isinstance(direction, str) or direction not in ('buy', 'sell'):
            ot = r.get('order_type')
            direction = 'buy' if (ot is not None and int(ot) == xtconstant.STOCK_BUY) else 'sell'
        price = round(float(r.get('traded_price', 0) or 0), 4)
        shares = int(r.get('traded_volume', 0) or 0)
        if shares <= 0 or price <= 0:
            continue
        amount = float(r.get('amount', 0) or 0) or price * shares
        code = r['code']
        est_price = est_prices.get(code)
        slippage_pct = None
        if est_price and est_price > 0 and price > 0:
            slippage_pct = round((price - est_price) / est_price * 100, 4)
        tid = str(r['traded_id']) if 'traded_id' in r.index and pd.notna(r.get('traded_id')) else ''
        rows.append({
            'date': target, 'code': code,
            'name': (r.get('name') or '').strip() if isinstance(r.get('name'), str) else '',
            'direction': direction,
            'price': price, 'shares': shares,
            'amount': round(amount, 2),
            'fee_est': _fee_est(amount, direction),
            'order_id': int(r.get('order_id', 0) or 0),
            'traded_id': tid,
            'fill_time': r['ts'],
            'est_price': round(float(est_price), 4) if est_price else None,
            'slippage_pct': slippage_pct,
        })
    return pd.DataFrame(rows, columns=FILL_COLS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('date', type=str, help='目标日期 YYYY-MM-DD')
    parser.add_argument('--dir', type=str, default='data/live_trades', help='live_trades 目录')
    parser.add_argument('--dry-run', action='store_true', help='只打印结果，不写盘')
    args = parser.parse_args()

    live_dir = (ROOT / args.dir) if not Path(args.dir).is_absolute() else Path(args.dir)
    new_fills = rebuild(args.date, live_dir)

    old_path = live_dir / f"fills_{args.date}.parquet"
    old_n = len(pd.read_parquet(old_path)) if old_path.exists() else 0
    buy_amt = float(new_fills[new_fills['direction'] == 'buy']['amount'].sum())
    sell_amt = float(new_fills[new_fills['direction'] == 'sell']['amount'].sum())
    print(f"重建 fills: {old_n} → {len(new_fills)} 行 | 买入 ¥{buy_amt:,.0f} | 卖出 ¥{sell_amt:,.0f}")

    if args.dry_run:
        print("[dry-run] 不写盘。按股票卖出金额:")
        s = new_fills[new_fills['direction'] == 'sell'].groupby('code')['amount'].sum()
        print(s.to_string())
        return

    if old_path.exists():
        bak = old_path.with_suffix('.parquet.bak')
        old_path.replace(bak)
        print(f"原文件备份 → {bak.name}")
    new_fills.to_parquet(old_path, index=False)
    print(f"✅ 已写入 {old_path.name}")


if __name__ == '__main__':
    main()
