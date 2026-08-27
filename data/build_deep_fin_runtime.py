"""把深历史财务指标 (deep_indicators.parquet) 对齐成 PIT 矩阵辅助 NPZ。

产物：data/runtime/deep_fin_pit.npz
  trade_dates / stock_codes 与主 runtime 完全一致；
  每个指标 -> (n_dates, n_stocks) float32 矩阵，已做 point-in-time 处理：
    - 使用法定最晚披露日后的首个交易日生效；
    - 生效后向前填充，直到下一期报告覆盖；首份报告前为 NaN。

用法：
  uv run python data/build_deep_fin_runtime.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.financial_pit import (
    build_pit_source_indices,
    materialize_pit_field,
)

DATA_DIR = Path(__file__).resolve().parent
DEEP_PATH = DATA_DIR / 'financial' / 'deep_indicators.parquet'
OUT_PATH = DATA_DIR / 'runtime' / 'deep_fin_pit.npz'

_INDICATORS = ['bps', 'eps', 'ocfps', 'roe', 'net_profit', 'revenue',
               'profit_yoy', 'revenue_yoy', 'net_margin', 'gross_margin',
               'debt_ratio']


def main():
    rt = sorted((DATA_DIR / 'runtime').glob('runtime_*.npz'))[-1]
    d = np.load(rt, allow_pickle=False)
    trade_dates = d['trade_dates'].astype('datetime64[D]')
    stock_codes = d['stock_codes']
    n_dates = len(trade_dates)
    n_stocks = len(stock_codes)

    df = pd.read_parquet(DEEP_PATH)
    print(f'deep_indicators: {len(df)} 行, {df["stock_code"].nunique()} 股票')

    source_indices = build_pit_source_indices(
        df,
        stock_codes,
        trade_dates,
    )

    out = {'trade_dates': d['trade_dates'], 'stock_codes': stock_codes}
    for name in _INDICATORS:
        mat = materialize_pit_field(
            df,
            source_indices,
            name,
        ).astype(np.float32)
        out[name] = mat
        cov = np.isfinite(mat).any(axis=0).sum()
        print(f'  {name:14s} 覆盖股票={int(cov):4d}  非空单元={int(np.isfinite(mat).sum()):,}')

    np.savez_compressed(OUT_PATH, **out)
    print(f'保存 -> {OUT_PATH}  ({n_dates}d x {n_stocks}s, 法定披露日后生效)')


if __name__ == '__main__':
    sys.exit(main())
