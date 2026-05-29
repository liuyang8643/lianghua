"""快速 dump positions / fills / summary 关键字段，定位为什么 daily_pnl=None。"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADE_DIR = ROOT / "data" / "live_trades"

for d in ['2026-05-27', '2026-05-28']:
    pp = TRADE_DIR / f'positions_{d}.parquet'
    if not pp.exists():
        continue
    df = pd.read_parquet(pp)
    print(f"\n========== positions_{d}.parquet ({len(df)} 行) ==========")
    cols = ['code', 'name', 'volume', 'yesterday_volume', 'bought_today',
            'sold_today', 'buy_amount_today', 'sell_amount_today',
            'fee_today', 'last_price', 'daily_pnl']
    print(df[cols].to_string())

# daily_summary 内容
sp = TRADE_DIR / 'daily_summary.parquet'
if sp.exists():
    print("\n========== daily_summary.parquet ==========")
    print(pd.read_parquet(sp).to_string())

# cash_flows 内容
cf = TRADE_DIR / 'cash_flows.parquet'
if cf.exists():
    print("\n========== cash_flows.parquet ==========")
    print(pd.read_parquet(cf).to_string())
