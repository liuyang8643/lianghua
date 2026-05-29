"""盘后对比报告：QMT实盘 vs 单日回测 → 飞书卡片。

实盘数据全部来自 QMT 接口（持仓/资产）+ 回调存储（成交/快照/出入金），
严禁使用 K 线数据。
"""
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

TRADE_DIR = Path(__file__).resolve().parents[1] / "data" / "live_trades"


def get_qmt_data():
    from trading.trader import Trader
    from configs import TRADE_ACCOUNT
    td = Trader(TRADE_ACCOUNT)
    time.sleep(0.5)
    asset = td.query_asset()
    positions = td.query_positions() or []
    return {
        'asset': float(asset.total_asset) if asset else 0,
        'cash': float(asset.cash) if asset else 0,
        'market_value': float(asset.market_value) if asset else 0,
        'positions': [{
            'code': p.stock_code, 'volume': int(p.volume),
            'can_use_volume': int(p.can_use_volume),
            'market_value': float(p.market_value),
            'last_price': float(p.last_price),
            'open_cost': float(getattr(p, 'open_cost', 0) or 0),
            'float_profit': float(getattr(p, 'float_profit', 0) or 0),
        } for p in positions],
    }


def get_prev_asset_and_cf(trade_date: date) -> tuple[float, float]:
    """返回 (昨日收盘资产, 今日净入金)。"""
    summary_path = TRADE_DIR / "daily_summary.parquet"
    prev_asset = 0.0
    if summary_path.exists():
        df = pd.read_parquet(summary_path)
        prev_rows = df[df['date'] < trade_date]
        if not prev_rows.empty:
            prev_asset = float(prev_rows['total_asset'].iloc[-1])

    cf_path = TRADE_DIR / "cash_flows.parquet"
    net_cf = 0.0
    if cf_path.exists():
        cf = pd.read_parquet(cf_path)
        today_cf = cf[cf['date'] == trade_date]
        if len(today_cf) > 0:
            net_cf = float(today_cf['amount'].sum())

    return prev_asset, net_cf


def run_backtest(trade_date: date, config_path: str):
    from core.backtest import _compute_factor_scores, _backtest_direct
    from data.db.stock_list import allow_buy_stock_code_list
    from core.factors.registry import get_factor_class as _get_factor_class

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    ic = config_data['individual_config']
    signal_dt = datetime.combine(trade_date, datetime.min.time())
    all_stocks = allow_buy_stock_code_list(target_date=trade_date)
    factor_classes = [_get_factor_class(f) for f in ic['weights']]

    t0 = time.time()
    sr = _compute_factor_scores([signal_dt], all_stocks, weights=ic['weights'],
                                factor_classes=factor_classes)
    if sr is None:
        return None
    data, scores, dates, d_idx, stocks, s_idx = sr
    if not dates:
        return None
    result = _backtest_direct(data, scores, dates, d_idx, stocks, s_idx,
                              weights=ic['weights'], buy_n=ic['buy_n'],
                              sell_m=ic.get('sell_m', ic['buy_n']),
                              temperatures=ic['temperatures'], lightweight=False)
    print(f"[Backtest] 完成 ({time.time() - t0:.1f}s): 收益={result.get('total_return', 0):.2f}%")
    return result


class FakePos:
    def __init__(self, d):
        self.stock_code = d['code']
        self.volume = d['volume']
        self.market_value = d['market_value']
        self.can_use_volume = d.get('can_use_volume', d['volume'])


def main():
    trade_date = date.today()
    print(f"=== 盘后对比报告 @ {trade_date.isoformat()} ===\n")

    # 1. QMT 实盘
    qmt = get_qmt_data()
    active = [p for p in qmt['positions'] if p['volume'] > 0]
    print(f"[QMT] 资产=¥{qmt['asset']:,.0f} 持仓={len(active)}只")

    # 2. 昨日资产 + 出入金（回调存储，非 K 线）
    prev_asset, net_cf = get_prev_asset_and_cf(trade_date)
    print(f"[存储] 昨日收盘资产=¥{prev_asset:,.0f} 今日净入金=¥{net_cf:+,.0f}")

    # 3. 账户日收益
    if prev_asset > 0:
        live_pnl = qmt['asset'] - prev_asset - net_cf
        live_return = live_pnl / prev_asset * 100
        print(f"[账户] 日收益={live_return:+.2f}% 盈亏=¥{live_pnl:+,.0f}")
    else:
        live_return = 0.0
        live_pnl = None

    # 4. 回测
    config_path = Path(__file__).resolve().parents[1] / "configs" / "20260526.json"
    bt_result = run_backtest(trade_date, str(config_path))
    if bt_result is None:
        print("[错误] 回测失败")
        sys.exit(1)

    # 5. 今日成交
    fills_path = TRADE_DIR / f"fills_{trade_date.isoformat()}.parquet"
    fills_df = pd.read_parquet(fills_path) if fills_path.exists() else pd.DataFrame()

    # 6. 个股日收益 — 检查是否有昨日快照
    from utils.stock.time import get_last_trading_day
    prev_day = get_last_trading_day(trade_date - timedelta(days=1))
    snap_path = TRADE_DIR / f"positions_{prev_day.isoformat()}.parquet"
    daily_prices = {}
    if snap_path.exists():
        prev_pos = pd.read_parquet(snap_path)
        price_col = 'last_price' if 'last_price' in prev_pos.columns else 'current_price'
        prev_map = {}
        for _, r in prev_pos.iterrows():
            price = float(r.get(price_col, 0) or 0)
            if r['volume'] > 0 and price > 0:
                prev_map[r['code']] = {'volume': int(r['volume']), 'last_price': price}
        fill_prices = {}
        if not fills_df.empty:
            buys = fills_df[fills_df['direction'] == 'buy']
            for code, grp in buys.groupby('code'):
                total_amt = (grp['price'] * grp['shares']).sum()
                total_shares = grp['shares'].sum()
                if total_shares > 0:
                    fill_prices[code] = total_amt / total_shares
        for p in active:
            code = p['code']
            lp = p['last_price']
            if lp <= 0:
                continue
            yp = prev_map.get(code)
            fp = fill_prices.get(code)
            if yp and yp['volume'] > 0 and yp['last_price'] > 0:
                old_ret = (lp - yp['last_price']) / yp['last_price'] * 100
                if fp and p['volume'] > yp['volume']:
                    new_vol = p['volume'] - yp['volume']
                    new_ret = (lp - fp) / fp * 100 if fp > 0 else 0.0
                    daily_prices[code] = (old_ret * yp['volume'] + new_ret * new_vol) / p['volume']
                else:
                    daily_prices[code] = old_ret
            elif fp:
                daily_prices[code] = (lp - fp) / fp * 100 if fp > 0 else 0.0
        print(f"[个股] 日收益可算: {len(daily_prices)}/{len(active)}只 (有昨日快照)")
    else:
        print(f"[个股] 无昨日快照({prev_day.isoformat()}), 全部不可算 → 显示\"-\"")

    # 7. 构建报告
    from trading.report import PostCloseReport

    report = PostCloseReport(trade_date)
    report.feed_positions([FakePos(p) for p in active])
    report.feed_asset(qmt['asset'], prev_asset, net_cash_flow=net_cf)
    report.feed_backtest(bt_result)
    report.feed_live_fills(fills_df)
    report.feed_daily_prices(daily_prices)

    report_data = report.build()

    # 8. 打印
    print(f"\n=== 报告摘要 ===")
    s = report_data['summary']
    print(f"持仓: 实盘{s['n_pos']}只")
    print(f"总市值: 实盘¥{s['total_live_mv']:,.0f} | 回测¥{s['total_bt_mv']:,.0f}")
    print(f"手续费: 实盘¥{s['live_fee']:.1f} | 回测¥{s['bt_fee']:.1f}")
    print(f"日收益: 实盘{s['live_return']:+.2f}% | 回测{s['bt_return']:+.2f}%")
    if s['live_pnl'] is not None:
        print(f"日盈亏: 实盘¥{s['live_pnl']:+,.0f} | 回测{s['bt_pnl']:+,.0f}")
    print(f"出入金: ¥{net_cf:+,.0f}")

    print(f"\n=== 逐股对比 ===")
    for r in report_data['rows']:
        if r['live_mv'] == 0 and r['bt_mv'] == 0:
            continue
        lr = f"{r['live_ret']:+.2f}%" if r['live_ret'] is not None else '-'
        br = f"{r['bt_ret']:+.2f}%" if r['bt_ret'] is not None else '-'
        rd = f"{r['ret_diff']:+.2f}%" if r['ret_diff'] is not None else '-'
        tag = "" if r.get('live_ret') is not None else " [不可算]"
        print(f"  {r['code']} {r['name']:<8s}{tag}"
              f"  MV:¥{r['live_mv']:,.0f}|¥{r['bt_mv']:,.0f}|{r['mv_diff']:+,.0f}"
              f"  Ret:{lr}|{br}|{rd}")

    # 9. 发送飞书
    print(f"\n=== 发送飞书 ===")
    report.send()
    print("发送完成!")


if __name__ == '__main__':
    main()
