"""
基线测试脚本：运行回测并保存关键结果。
优化前运行一次保存基线，优化后再运行对比。
"""
import json
import sys
import numpy as np
from datetime import date, datetime
from pathlib import Path

from data.db.delist import get_delist_stock_info
from core.strategies.runtime import load_runtime_npz
from utils.stock.time import get_trading_date_span
from testback.main import (
    _backtest_direct, _compute_factor_scores, _compute_list_dates,
    _scores_to_ranks,
)
import core.factors as _all_factors

def run_baseline(start_date: date, end_date: date, config: dict, label: str):
    """运行回测并保存结果作为基线"""
    backtest_datetime_list = [
        datetime.combine(d, datetime.min.time())
        for d in get_trading_date_span(start_date, end_date)
    ]

    # 从 npz 取全量股票
    npz_dir = Path(__file__).resolve().parent.parent / 'data' / 'runtime'
    npz_files = sorted(npz_dir.glob('runtime_*.npz'))
    all_stocks = [str(s) for s in np.load(npz_files[0], allow_pickle=False)['stock_codes']]

    individual_config = config['individual_config']

    # 因子计算
    factor_classes = []
    for fname in individual_config['weights']:
        cls = getattr(_all_factors, fname, None)
        if cls is None:
            raise ValueError(f"因子类 {fname} 不存在")
        factor_classes.append(cls)

    scores_result = _compute_factor_scores(
        backtest_datetime_list, all_stocks,
        weights=individual_config['weights'], factor_classes=factor_classes,
    )
    data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices = scores_result

    list_dates_map = _compute_list_dates(data['stock_codes'], data['open'], data['trade_dates'])

    # 回测
    result = _backtest_direct(
        data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
        weights=individual_config['weights'],
        buy_n=individual_config['buy_n'], sell_m=individual_config['sell_m'],
        temperatures=individual_config['temperatures'],
        holding_period=individual_config.get('holding_period'),
        list_dates_map=list_dates_map,
    )

    # 提取可对比的关键字段
    baseline = {
        'label': label,
        'start_date': str(start_date),
        'end_date': str(end_date),
        'config': config,
        'total_return': result['total_return'],
        'final_asset': result['final_asset'],
        'daily_returns': result['daily_returns'],
        'cumulative_returns': result['cumulative_returns'],
        'cleared_positions_count': result['cleared_positions_count'],
        'current_positions_count': result['current_positions_count'],
        'round_trip_count': result['round_trip_count'],
        'executed_buy_count': result['executed_buy_count'],
        'executed_sell_count': result['executed_sell_count'],
        'delist_count': result['delist_count'],
        'cleared_codes': sorted([p['code'] for p in result['cleared_positions']]),
        'position_codes': sorted([p['code'] for p in result['positions']]),
        'daily_snapshot_dates': [s['date'] for s in result['daily_snapshots']],
        'daily_snapshot_assets': [s['total_asset'] for s in result['daily_snapshots']],
    }

    output_path = Path(__file__).resolve().parent.parent / 'results' / f'baseline_{label}.json'
    output_path.parent.mkdir(exist_ok=True)

    # numpy 数组转 list 以便 JSON 序列化
    def convert(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert(baseline), f, ensure_ascii=False)

    print(f"基线已保存: {output_path}")
    print(f"  总收益: {result['total_return']:.2f}%")
    print(f"  调仓日: {len(valid_dates)}")
    print(f"  清仓次数: {result['cleared_positions_count']}")
    print(f"  持仓数: {result['current_positions_count']}")
    return baseline


if __name__ == '__main__':
    # 测试用例1: 短周期 1年，单因子 TrueMarketCap
    config1 = {
        "individual_config": {
            "weights": {"TrueMarketCap": 1.0},
            "buy_n": 10, "sell_m": 10,
            "temperatures": {"TrueMarketCap": 1.0},
            "holding_period": 1,
        }
    }
    run_baseline(date(2024, 1, 1), date(2024, 12, 31), config1, "test1_truemarketcap_2024")

    # 测试用例2: 短周期 1年，带 holding_period
    config2 = {
        "individual_config": {
            "weights": {"TrueMarketCap": 1.0},
            "buy_n": 25, "sell_m": 25,
            "temperatures": {"TrueMarketCap": 1.0},
            "holding_period": 20,
        }
    }
    run_baseline(date(2023, 1, 1), date(2024, 12, 31), config2, "test2_hold20_2023_2024")
