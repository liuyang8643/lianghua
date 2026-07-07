"""低开策略开盘成交率分析（离线，用 data/minute 分钟数据，不碰网络/QMT）。

问题：09:26 挂 limit=open 的买单，9:30 连续竞价成交。能不能吃到开盘价？
核心风险：逆向选择——秒回归的赢家瞬间冲离开盘价（买不到），阴跌的输家全成交。

用 09:31 bar（覆盖 09:30:00~09:31:00 的真实撮合）度量：
  - open O = NPZ 官方开盘价（集合竞价，9:25 已定）
  - low1  = 第一分钟最低价；low1 <= O → 开盘后价格回到过 ≤O，limit=O 能成交
  - low1  > O  → 开盘后瞬间冲高、再没回到 O → limit=O 基本买不到（这些往往是赢家）
  - amt1  = 第一分钟成交额；对比 4 万单量看量能约束
  - ret_1m = close1/O-1 用作"即时反弹"代理：>0 偏赢家、<0 偏输家
"""
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.backtest import _compute_factor_scores
from core.legality import LegalityChecker
from core.scoring import select_topn
from core.strategy_config import load_strategy_config

ORDER_AMOUNT = 40000.0          # 每只 4 万
BUY_N = 20
LIVE_START = date(2026, 1, 19)  # 分钟数据覆盖起点
LIVE_END = date(2026, 6, 8)
TICK_EPS = 0.011                # ≈1 分钱容差（触及开盘价即算可成交）


def build_minute_first_bar():
    """{code: {date_int: (open1, close1, high1, low1, vol1, amt1)}}，只取 09:31 bar。"""
    minute_dir = ROOT / 'data' / 'minute'
    lookup = {}
    for f in minute_dir.glob('*.parquet'):
        if f.stem == 'slippage_stats':
            continue
        df = pd.read_parquet(f)
        df = df[df['time'].dt.strftime('%H:%M') == '09:31']
        if df.empty:
            continue
        cm = {}
        for _, r in df.iterrows():
            d_int = int(r['time'].strftime('%Y%m%d'))
            cm[d_int] = (float(r['open']), float(r['close']), float(r['high']),
                         float(r['low']), float(r['volume']), float(r['amount']))
        if cm:
            lookup[f.stem] = cm
    return lookup


def main():
    cfg = load_strategy_config(str(ROOT / 'configs' / 'config.json'))
    factor_classes = cfg['factor_classes']
    ic = cfg['individual_config']
    weights = ic['weights']
    limit_up_protection = ic.get('limit_up_protection', False)
    print(f"因子权重: {weights}")

    # 分钟窗口所有交易日
    minute = build_minute_first_bar()
    print(f"分钟数据: {len(minute)} 只")

    # 覆盖窗口的所有交易日（从任一只票的日期并集）
    all_days = sorted({d for cm in minute.values() for d in cm})
    day_dts = [datetime.strptime(str(d), '%Y%m%d') for d in all_days]

    # 全量股票作为候选池
    import numpy as _np
    tmp = _np.load(next((ROOT / 'data' / 'runtime').glob('runtime_*.npz')), allow_pickle=False)
    all_stocks = [str(s) for s in tmp['stock_codes']]
    tmp.close()

    result = _compute_factor_scores(day_dts, all_stocks, weights, factor_classes,
                                    enable_nan_filter=False)
    data, all_scores, _, valid_dates, date_indices, valid_stocks, stock_indices = result
    valid_cols = np.array([stock_indices[s] for s in valid_stocks], dtype=np.intp)
    checker = LegalityChecker(data, stock_indices, limit_up_protection=limit_up_protection)

    rows = []
    for i, dt in enumerate(valid_dates):
        di = date_indices[i]
        d_int = int(dt.strftime('%Y%m%d'))
        topn, fscore = select_topn(all_scores, di, valid_stocks, valid_cols, weights, BUY_N)
        # 买入合法性闸门（涨停禁买等）
        idxs = [stock_indices[c] for c in topn]
        ok, _r = checker.check(idxs, di, dt.date(), is_buy=True)
        picks = [c for c, o in zip(topn, ok) if o]

        for code in picks:
            si = stock_indices[code]
            O = float(data['open'][di, si])
            pc = float(data['preClose'][di, si])
            if not (O > 0):
                continue
            bar = minute.get(code, {}).get(d_int)
            if bar is None:
                rows.append({'date': d_int, 'code': code, 'open': O,
                             'gap': (O / pc - 1) if pc > 0 else np.nan,
                             'has_min': False})
                continue
            o1, c1, h1, l1, v1, a1 = bar
            order_shares = ORDER_AMOUNT / O
            rows.append({
                'date': d_int, 'code': code, 'open': O,
                'gap': (O / pc - 1) if pc > 0 else np.nan,
                'has_min': True,
                'low1': l1, 'high1': h1, 'close1': c1, 'amt1': a1,
                'touched_open': l1 <= O + TICK_EPS,     # 能否以 open 成交
                'ran_away': l1 > O + TICK_EPS,           # 冲离开盘价、买不到
                'ret_1m': c1 / O - 1.0,                  # 即时反弹代理
                'amt_cover': a1 / ORDER_AMOUNT,          # 第一分钟成交额是 4 万的几倍
                'order_shares': order_shares,
            })

    df = pd.DataFrame(rows)
    tot = len(df)
    have = df[df['has_min']].copy()
    print("\n" + "=" * 64)
    print(f"选股-成交率分析  窗口 {LIVE_START}~{LIVE_END}  买入 {BUY_N} 只/日  单量 {ORDER_AMOUNT:.0f} 元")
    print("=" * 64)
    print(f"总选股样本(过合法性闸门后): {tot}   有分钟数据: {len(have)} ({len(have)/tot*100:.1f}%)")
    if have.empty:
        return

    touched = have['touched_open'].mean()
    ran = have['ran_away'].mean()
    print(f"\n【价格可成交性 · limit=开盘价】")
    print(f"  第一分钟回到过 ≤开盘价(可成交): {touched*100:.1f}%")
    print(f"  开盘即冲离、第一分钟未回到开盘价(买不到): {ran*100:.1f}%")

    print(f"\n【量能约束】第一分钟成交额 / 4万")
    print(f"  中位数覆盖倍数: {have['amt_cover'].median():.0f}x   "
          f"<1倍(量能不足4万)占比: {(have['amt_cover']<1).mean()*100:.1f}%")

    # 逆向选择：即时反弹(赢家) vs 阴跌(输家) 的可成交性
    win = have[have['ret_1m'] > 0]
    los = have[have['ret_1m'] <= 0]
    print(f"\n【逆向选择】以 close(09:31)/open-1 判即时方向")
    print(f"  即时反弹(赢家) {len(win)} 只 ({len(win)/len(have)*100:.1f}%): "
          f"其中开盘价可成交 {win['touched_open'].mean()*100:.1f}%, "
          f"平均首分钟涨 {win['ret_1m'].mean()*100:.2f}%")
    print(f"  即时阴跌(输家) {len(los)} 只 ({len(los)/len(have)*100:.1f}%): "
          f"其中开盘价可成交 {los['touched_open'].mean()*100:.1f}%, "
          f"平均首分钟跌 {los['ret_1m'].mean()*100:.2f}%")

    # 综合"有效成交"估计：赢家里买不到的部分是纯损失
    win_fillable = win['touched_open'].mean() if len(win) else 0.0
    print(f"\n【结论量化】")
    print(f"  你想要的'赢家'里，只有 ~{win_fillable*100:.0f}% 能在开盘价成交；"
          f"另外 ~{(1-win_fillable)*100:.0f}% 秒冲离、你买不到。")
    print(f"  而输家 {los['touched_open'].mean()*100:.0f}% 都能成交 → 系统性买到差票。")


if __name__ == '__main__':
    main()
