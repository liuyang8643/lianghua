"""逐股 dump daily_pnl 来源，定位 reconcile 差额。"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADE_DIR = ROOT / "data" / "live_trades"

t = '2026-05-28'
pos = pd.read_parquet(TRADE_DIR / f'positions_{t}.parquet')
ydf = pd.read_parquet(TRADE_DIR / f'positions_2026-05-27.parquet')
y_map = {r['code']: r for _, r in ydf.iterrows() if r['volume'] > 0}

print(f"\n=== {t} 逐股 daily_pnl 拆解 ===\n")
print(f"{'code':10s} {'vol':>5s} {'y_vol':>5s} "
      f"{'T_mv':>9s} {'Y_mv':>9s} {'T_cost':>9s} {'Y_cost':>9s} "
      f"{'buy_real':>9s} {'sell_real':>10s} {'fee':>6s} {'daily_pnl':>10s}")
total = 0
for _, r in pos.iterrows():
    code = r['code']
    vol_t = int(r['volume'])
    lp_t = float(r['last_price'])
    t_avg = float(r['avg_price'])
    t_mv = lp_t * vol_t
    t_cost = t_avg * vol_t
    y = y_map.get(code)
    if y is not None:
        y_vol = int(y['volume'])
        y_lp = float(y['last_price'])
        y_avg = float(y['avg_price'])
        y_mv = y_lp * y_vol
        y_cost = y_avg * y_vol
    else:
        y_vol = y_lp = y_avg = y_mv = y_cost = 0
    bought = int(r['bought_today'])
    sold = int(r['sold_today'])
    gap = vol_t - (y_vol + bought - sold)
    if gap > 0:
        bought_real, sold_real = bought + gap, sold
    elif gap < 0:
        bought_real, sold_real = bought, sold - gap
    else:
        bought_real, sold_real = bought, sold
    buy_amt_real = max(0.0, t_cost - y_cost + y_avg * sold_real)
    sell_amt = float(r['sell_amount_today'])
    if sold_real > 0:
        if sold > 0 and sell_amt > 0:
            asp = sell_amt / sold
        else:
            asp = lp_t if lp_t > 0 else y_lp
        sell_amt_real = asp * sold_real
    else:
        sell_amt_real = 0.0
    fee = float(r['fee_today'])
    bf = float(r['buy_amount_today'])
    sf = sell_amt
    tot_f = bf + sf
    tot_r = buy_amt_real + sell_amt_real
    if tot_f > 0 and tot_r > 0:
        fee_real = fee * (tot_r / tot_f)
    else:
        fee_real = buy_amt_real * 0.0001 + sell_amt_real * 0.0006
    dpnl = t_mv - y_mv + sell_amt_real - buy_amt_real - fee_real
    total += dpnl
    print(f"{code:10s} {vol_t:>5d} {y_vol:>5d} "
          f"{t_mv:>9.0f} {y_mv:>9.0f} {t_cost:>9.0f} {y_cost:>9.0f} "
          f"{buy_amt_real:>9.0f} {sell_amt_real:>10.0f} {fee_real:>6.1f} {dpnl:>10.2f}")
print(f"\nΣ = {total:+.2f}")
