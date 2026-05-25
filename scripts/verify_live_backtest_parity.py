"""验收：同一日期回测 vs 实盘选股一致性。

用法:
    uv run python scripts/verify_live_backtest_parity.py --date 2026-05-22
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testback.ga_config import get_profile_factor_classes, resolve_profile_name
from core.strategies.runtime import load_runtime_npz
from core.strategies.scoring import scores_to_ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True, help='信号日期 YYYY-MM-DD')
    parser.add_argument('--config', type=str,
                        default='configs/single_smallcap_g2a_config.json',
                        help='individual config JSON 路径')
    args = parser.parse_args()

    signal_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    signal_datetime = datetime.combine(signal_date, datetime.min.time())

    with open(args.config, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    profile_name = resolve_profile_name(config_data)
    factor_classes = get_profile_factor_classes(profile_name)
    individual_config = config_data['individual_config']
    weights = individual_config['weights']
    temperatures = individual_config['temperatures']
    buy_n = individual_config['buy_n']

    # ====== 回测路径 ======
    from testback.main import _compute_factor_scores
    from data.db import allow_buy_stock_code_list

    all_stocks = list(allow_buy_stock_code_list(target_date=signal_date))
    result = _compute_factor_scores(
        [signal_datetime], all_stocks,
        weights=weights, factor_classes=factor_classes,
    )
    if result is None:
        print("FAIL: 回测 _compute_factor_scores 返回 None")
        sys.exit(1)
    data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices = result

    date_idx = date_indices[0]
    valid_cols = np.array([stock_indices[s] for s in valid_stocks], dtype=np.intp)

    bt_final = np.zeros(len(valid_stocks))
    for name, ranks_mat in all_scores.items():
        w = weights.get(name, 0.0)
        if w == 0: continue
        row = ranks_mat[date_idx, valid_cols]
        temp = temperatures.get(name, 1.0)
        if temp != 1.0:
            row = np.power(row, 1.0 / temp)
        bt_final += row * w
    bt_top = [valid_stocks[i] for i in np.argsort(-bt_final)[:buy_n]]
    print(f"回测 Top{buy_n}: {bt_top}")

    # ====== 实盘路径 ======
    from trading.main import _compute_live_scores

    live_data, live_scores, live_date_idx, live_valid_stocks, live_si = _compute_live_scores(
        signal_datetime, all_stocks, weights, factor_classes)

    live_valid_cols = np.array([live_si[s] for s in live_valid_stocks], dtype=np.intp)

    live_final = np.zeros(len(live_valid_stocks))
    for name, ranks_mat in live_scores.items():
        w = weights.get(name, 0.0)
        if w == 0: continue
        row = ranks_mat[live_date_idx, live_valid_cols]
        temp = temperatures.get(name, 1.0)
        if temp != 1.0:
            row = np.power(row, 1.0 / temp)
        live_final += row * w
    live_top = [live_valid_stocks[i] for i in np.argsort(-live_final)[:buy_n]]
    print(f"实盘 Top{buy_n}: {live_top}")

    # ====== 对比 ======
    if bt_top == live_top:
        print("PASS: 回测与实盘选股完全一致")
        return 0

    bt_set = set(bt_top)
    live_set = set(live_top)
    print(f"交集: {bt_set & live_set}")
    print(f"仅回测: {bt_set - live_set}")
    print(f"仅实盘: {live_set - live_set}")
    print(f"差异: 回测={len(bt_set - live_set)}, 实盘={len(live_set - bt_set)}")
    return 1


if __name__ == '__main__':
    sys.exit(main())
