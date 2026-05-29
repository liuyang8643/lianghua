"""一次性迁移：把老 schema 的 27 号实盘 parquet 复制到 ai-rebuilt 目录并升级成新 schema。

操作：
    源:  data/live_trades/{fills,positions}_2026-05-27.parquet           (老 schema)
    目标: data/live_trades_rebuilt_2026-05-27/{plan,fills,positions}_2026-05-27.parquet (新 schema)

老 → 新 字段补全规则：
  fills (10 → 12 列):
    + est_price:    从 NPZ 取当日 open[code]（即 plan 在盘前会预估的开盘价）
    + slippage_pct: (traded_price - est_price) / est_price × 100

  positions (8 → 18 列):
    用 fills 聚合 + 昨日 positions（不存在则视为 0）重建：
      + yesterday_volume   (无昨日快照 → 0)
      + avg_price          (从 cost_price 或反算)
      + last_price         (老 schema 中 current_price 即 close)
      + open_cost, float_profit (默认 0)
      + bought_today, sold_today, buy/sell_amount_today, fee_today  (fills 聚合)
      + daily_pnl          (T_mv - Y_mv + S - B - fee)
      + daily_return_pct

  plan (从 fills 反推):
    fills 中实际成交的所有 buy/sell 都视为 plan 中 limit_status='ok' 的行；
    无法反推被涨停过滤掉的候选股（dim1/dim2 因此会显示 100% 匹配率，这是已知失真）。

用法:
    python scripts/rebuild_live_trades_for_diff.py 2026-05-27
"""
from __future__ import annotations
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.persistence import PLAN_COLS, FILL_COLS, POSITION_COLS

LIVE_DIR = ROOT / "data" / "live_trades"


def load_npz_for_date(target: date) -> tuple[dict, int]:
    """从最新 NPZ 加载，返回 (data_dict, date_idx_for_target)。"""
    npzs = sorted((ROOT / "data" / "runtime").glob("runtime_*.npz"))
    if not npzs:
        raise FileNotFoundError("找不到 runtime_*.npz")
    arr = np.load(npzs[-1])
    trade_dates = arr['trade_dates']
    tgt_dt64 = np.datetime64(target, 'D')
    idx_arr = np.where(trade_dates == tgt_dt64)[0]
    if len(idx_arr) == 0:
        raise ValueError(f"NPZ 中无 {target} 数据")
    idx = int(idx_arr[0])
    return arr, idx


def build_new_fills(old_fills: pd.DataFrame, npz: dict, date_idx: int) -> pd.DataFrame:
    """老 fills 加 est_price/slippage_pct，重排列序。"""
    stock_codes = list(npz['stock_codes'])
    code_to_idx = {c: i for i, c in enumerate(stock_codes)}
    open_row = npz['open'][date_idx]

    df = old_fills.copy()

    def _est(code):
        si = code_to_idx.get(code)
        if si is None:
            return None
        v = open_row[si]
        return float(v) if not np.isnan(v) and v > 0 else None

    df['est_price'] = df['code'].apply(_est)
    df['slippage_pct'] = df.apply(
        lambda r: round((r['price'] - r['est_price']) / r['est_price'] * 100, 4)
                  if r['est_price'] and r['est_price'] > 0 else None,
        axis=1,
    )
    # 重排列到 FILL_COLS
    return df[FILL_COLS]


def build_new_positions(old_positions: pd.DataFrame, new_fills: pd.DataFrame,
                        npz: dict, date_idx: int, target: date,
                        yesterday_positions: pd.DataFrame | None) -> pd.DataFrame:
    """老 positions 用 fills 聚合 + 昨日快照（无则填 0）重算成新 schema。"""
    stock_codes = list(npz['stock_codes'])
    code_to_idx = {c: i for i, c in enumerate(stock_codes)}
    close_row = npz['close'][date_idx]

    # fills 聚合
    fill_agg = {}
    if not new_fills.empty:
        for code, grp in new_fills.groupby('code'):
            buys = grp[grp['direction'] == 'buy']
            sells = grp[grp['direction'] == 'sell']
            fill_agg[code] = {
                'bought': int(buys['shares'].sum()) if not buys.empty else 0,
                'sold': int(sells['shares'].sum()) if not sells.empty else 0,
                'buy_amount': float(buys['amount'].sum()) if not buys.empty else 0.0,
                'sell_amount': float(sells['amount'].sum()) if not sells.empty else 0.0,
                'fee': float(grp['fee_est'].sum()),
            }

    y_map = {}
    if yesterday_positions is not None and not yesterday_positions.empty:
        for _, r in yesterday_positions.iterrows():
            if r['volume'] > 0:
                y_map[r['code']] = {
                    'volume': int(r['volume']),
                    'last_price': float(r['last_price'])
                                  if 'last_price' in r.index and not pd.isna(r['last_price']) and r['last_price'] > 0
                                  else float(r.get('current_price', 0) or 0),
                }

    rows = []
    for _, p in old_positions.iterrows():
        code = p['code']
        vol = int(p['volume'])
        if vol <= 0:
            continue

        si = code_to_idx.get(code)
        # 老 schema 的 current_price 在 27 号文件里全是 0，需要从 NPZ close 反取
        npz_close = float(close_row[si]) if si is not None and not np.isnan(close_row[si]) else 0.0
        last_price = npz_close if npz_close > 0 else (
            float(p['market_value']) / vol if vol > 0 else 0.0
        )
        mv = last_price * vol

        agg = fill_agg.get(code, {'bought': 0, 'sold': 0,
                                  'buy_amount': 0.0, 'sell_amount': 0.0, 'fee': 0.0})

        y = y_map.get(code)
        if y:
            y_mv = y['last_price'] * y['volume']
            daily_pnl = (last_price * vol) - y_mv + agg['sell_amount'] - agg['buy_amount'] - agg['fee']
            daily_ret = (daily_pnl / y_mv * 100) if y_mv > 0 else None
        elif agg['buy_amount'] > 0 and agg['bought'] == vol:
            # 完全新开仓（今日买入量 = 当前持仓量），公式可信
            daily_pnl = (last_price * vol) - agg['buy_amount'] - agg['fee']
            daily_ret = (daily_pnl / agg['buy_amount'] * 100)
        else:
            # 昨日有持仓但无快照，或加仓但缺昨日数据 → 无法可信算
            daily_pnl = None
            daily_ret = None

        # 老 schema 的 cost_price 在 27 号文件里全是 0，用反算填充
        avg_price = float(p.get('cost_price', 0) or 0)
        if avg_price <= 0:
            avg_price = agg['buy_amount'] / agg['bought'] if agg['bought'] > 0 else last_price

        rows.append({
            'date': target, 'code': code, 'name': p.get('name', ''),
            'volume': vol,
            'can_use_volume': int(p.get('can_use_volume', vol) or 0),
            'yesterday_volume': y['volume'] if y else 0,
            'market_value': round(mv, 2),
            'avg_price': round(avg_price, 4),
            'last_price': round(last_price, 4),
            'open_cost': 0.0,
            'float_profit': 0.0,
            'bought_today': agg['bought'],
            'sold_today': agg['sold'],
            'buy_amount_today': round(agg['buy_amount'], 2),
            'sell_amount_today': round(agg['sell_amount'], 2),
            'fee_today': round(agg['fee'], 4),
            'daily_pnl': round(daily_pnl, 2) if daily_pnl is not None else None,
            'daily_return_pct': round(daily_ret, 4) if daily_ret is not None else None,
        })

    return pd.DataFrame(rows, columns=POSITION_COLS)


def build_plan_from_fills(new_fills: pd.DataFrame, target: date) -> pd.DataFrame:
    """从真实 fills 反推 plan。
    所有实际成交了的 code 都视为 plan limit_status='ok'。
    无法重建：被涨停/资金不足过滤掉的候选股 → dim1/dim2 会显示 100% 匹配，是已知失真。
    """
    rows = []
    # 先按 code 聚合 fills（如果一个 code 既有 buy 又有 sell，分别建 2 行）
    seq = 1
    for direction in ['buy', 'sell']:
        grp = new_fills[new_fills['direction'] == direction].groupby('code')
        for code, sub in grp:
            est = sub['est_price'].dropna()
            est_price = float(est.iloc[0]) if not est.empty else float(sub['price'].mean())
            shares = int(sub['shares'].sum())
            amount = float(sub['amount'].sum())
            rows.append({
                'date': target, 'code': code,
                'name': sub['name'].iloc[0] if not sub['name'].empty else '',
                'direction': direction,
                'est_price': round(est_price, 4),
                'est_volume': shares,
                'est_amount': round(amount, 2),
                'factor_score': None,
                'limit_status': 'ok',
                'reason': 'rebuilt_from_fills',
                'plan_seq': seq,
            })
            seq += 1
    return pd.DataFrame(rows, columns=PLAN_COLS)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target_str = sys.argv[1]
    target = datetime.strptime(target_str, '%Y-%m-%d').date()

    src_fills = LIVE_DIR / f"fills_{target_str}.parquet"
    src_pos = LIVE_DIR / f"positions_{target_str}.parquet"
    if not src_fills.exists() or not src_pos.exists():
        print(f"❌ 找不到源数据 {src_fills} / {src_pos}")
        sys.exit(2)

    out_dir = ROOT / "data" / f"live_trades_rebuilt_{target_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    npz, date_idx = load_npz_for_date(target)
    print(f"NPZ date_idx={date_idx}")

    # 1. fills
    old_fills = pd.read_parquet(src_fills)
    new_fills = build_new_fills(old_fills, npz, date_idx)
    fill_path = out_dir / f"fills_{target_str}.parquet"
    new_fills.to_parquet(fill_path, index=False)
    print(f"✅ fills: {len(new_fills)} 行 ({list(new_fills.columns)}) → {fill_path.name}")

    # 2. positions (无昨日 positions 文件，y_map 视为空)
    yesterday = target - timedelta(days=1)
    yesterday_path = LIVE_DIR / f"positions_{yesterday.isoformat()}.parquet"
    yesterday_pos = pd.read_parquet(yesterday_path) if yesterday_path.exists() else None
    if yesterday_pos is None:
        print(f"⚠️  无昨日快照 positions_{yesterday.isoformat()}.parquet → daily_pnl 仅对当日新开仓可算")

    old_pos = pd.read_parquet(src_pos)
    new_pos = build_new_positions(old_pos, new_fills, npz, date_idx, target, yesterday_pos)
    pos_path = out_dir / f"positions_{target_str}.parquet"
    new_pos.to_parquet(pos_path, index=False)
    print(f"✅ positions: {len(new_pos)} 行 ({list(new_pos.columns)}) → {pos_path.name}")
    n_pnl = new_pos['daily_pnl'].notna().sum()
    print(f"   daily_pnl 可算 {n_pnl}/{len(new_pos)} 只")

    # 3. plan (从 fills 反推)
    new_plan = build_plan_from_fills(new_fills, target)
    plan_path = out_dir / f"plan_{target_str}.parquet"
    new_plan.to_parquet(plan_path, index=False)
    print(f"✅ plan: {len(new_plan)} 行 (反推, dim1/dim2 会失真) → {plan_path.name}")

    print()
    print("迁移完成。")
    print("跑离线 dry-run diff:")
    print(f"  python scripts/dry_run_postclose.py {target_str}")


if __name__ == '__main__':
    main()
