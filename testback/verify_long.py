"""长周期验证：16年回测，对比 lightweight vs normal 所有指标"""
import numpy as np
from datetime import date, datetime
from pathlib import Path
from data.db.delist import get_delist_stock_info
from core.strategies.runtime import load_runtime_npz
from utils.stock.time import get_trading_date_span
from testback.main import _backtest_direct, _compute_factor_scores, _compute_list_dates
import core.factors as _all_factors

def metrics_from_returns(daily_returns: list):
    daily = np.array(daily_returns, dtype=float)
    daily = daily[np.isfinite(daily)]
    n = len(daily)
    if n < 2:
        return {'annualized': 0.0, 'max_drawdown': 0.0, 'sharpe': 0.0, 'total_return': 0.0}
    mean_ret = float(np.mean(daily))
    std_ret = float(np.std(daily, ddof=1))
    sharpe = float(mean_ret / std_ret * np.sqrt(252.0)) if std_ret > 0 else 0.0
    cum_ret = np.cumprod(1.0 + daily / 100.0)
    years = n / 252.0
    annualized = float((cum_ret[-1] ** (1.0 / years) - 1) * 100) if years > 0 and cum_ret[-1] > 0 else 0.0
    peaks = np.maximum.accumulate(cum_ret)
    drawdowns = cum_ret / peaks - 1.0
    max_dd = float(np.min(drawdowns) * 100)
    total_return = float((cum_ret[-1] - 1.0) * 100)
    return {'annualized': annualized, 'max_drawdown': max_dd, 'sharpe': sharpe, 'total_return': total_return, 'n_days': n}

SCENARIOS = [
    # (start, end, weights, buy_n, sell_m, hp, label)
    (date(2010, 1, 1), date(2026, 5, 15), {'TrueMarketCap': 1.0}, 10, 10, 1, '小市值_top10_16年'),
    (date(2010, 1, 1), date(2026, 5, 15), {'TrueMarketCap': 1.0}, 25, 25, 20, '小市值_top25_hp20_16年'),
    (date(2010, 1, 1), date(2026, 5, 15), {'ROE': 0.5, 'TrueMarketCap': 0.5}, 15, 15, 1, '小市值+ROE_top15_16年'),
]

if __name__ == '__main__':
    for start, end, weights, buy_n, sell_m, hp, label in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"{label}: {start} ~ {end}")
        print(f"{'='*60}")

        bt_list = [datetime.combine(d, datetime.min.time()) for d in get_trading_date_span(start, end)]
        npz_dir = Path('data/runtime')
        npz_files = sorted(npz_dir.glob('runtime_*.npz'))
        all_stocks = [str(s) for s in np.load(npz_files[0], allow_pickle=False)['stock_codes']]

        factor_classes = [getattr(_all_factors, fname) for fname in weights]
        temps = {k: 1.0 for k in weights}
        t0 = __import__('time').time()
        scores_result = _compute_factor_scores(bt_list, all_stocks, weights=weights, factor_classes=factor_classes)
        data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices = scores_result
        list_dates_map = _compute_list_dates(data['stock_codes'], data['open'], data['trade_dates'])
        print(f"  数据+因子: {__import__('time').time()-t0:.1f}s, {len(valid_dates)}调仓日, {len(valid_stocks)}股")

        for mode_name, lightweight in [('normal', False), ('lightweight', True)]:
            t1 = __import__('time').time()
            r = _backtest_direct(data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
                weights=weights, buy_n=buy_n, sell_m=sell_m, temperatures=temps, holding_period=hp,
                list_dates_map=list_dates_map, lightweight=lightweight)
            elapsed = __import__('time').time() - t1
            m = metrics_from_returns(r['daily_returns'])
            print(f"  [{mode_name}] {elapsed:.2f}s | 总收益={m['total_return']:.2f}% | 年化={m['annualized']:.2f}% | 夏普={m['sharpe']:.4f} | 最大回撤={m['max_drawdown']:.2f}% | 清仓={r['cleared_positions_count']}")
