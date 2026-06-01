"""把深历史财务指标 (deep_indicators.parquet) 对齐成 PIT 矩阵辅助 NPZ。

产物：data/runtime/deep_fin_pit.npz
  trade_dates / stock_codes 与主 runtime 完全一致；
  每个指标 -> (n_dates, n_stocks) float32 矩阵，已做 point-in-time 处理：
    - 报告期 period_end 后滞后 LAG_DAYS 天才生效（避免前视野泄露：年报 12-31
      通常次年 4-30 才披露，统一 +120 天确保信息已公开）；
    - 生效后向前填充，直到下一期报告覆盖；首份报告前为 NaN。

用法：
  uv run python data/build_deep_fin_runtime.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
DEEP_PATH = DATA_DIR / 'financial' / 'deep_indicators.parquet'
OUT_PATH = DATA_DIR / 'runtime' / 'deep_fin_pit.npz'

LAG_DAYS = 120  # 报告期 -> 生效日的滞后天数（防数据泄露）

_INDICATORS = ['bps', 'eps', 'ocfps', 'roe', 'net_profit', 'revenue',
               'profit_yoy', 'revenue_yoy', 'net_margin', 'debt_ratio']


def _ffill_axis0(mat: np.ndarray) -> np.ndarray:
    n_dates = mat.shape[0]
    mask = ~np.isnan(mat)
    idx = np.where(mask, np.arange(n_dates)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = np.take_along_axis(mat, idx, axis=0)
    # 首个有效值之前（mask 在该列累计前全 False）保持 NaN
    has_any = np.cumsum(mask, axis=0) > 0
    out[~has_any] = np.nan
    return out


def main():
    rt = sorted((DATA_DIR / 'runtime').glob('runtime_*.npz'))[-1]
    d = np.load(rt, allow_pickle=False)
    trade_dates = d['trade_dates'].astype('datetime64[D]')
    stock_codes = d['stock_codes']
    n_dates = len(trade_dates)
    n_stocks = len(stock_codes)
    code_to_col = {str(c): i for i, c in enumerate(stock_codes)}

    df = pd.read_parquet(DEEP_PATH)
    print(f'deep_indicators: {len(df)} 行, {df["stock_code"].nunique()} 股票')

    # 报告期 -> 生效日 -> 生效行索引
    period_end = pd.to_datetime(df['report_period'].astype(int).astype(str), format='%Y%m%d')
    eff_date = (period_end + pd.Timedelta(days=LAG_DAYS)).values.astype('datetime64[D]')
    eff_row = np.searchsorted(trade_dates, eff_date, side='left')
    col = df['stock_code'].map(code_to_col).to_numpy()

    valid = (col >= 0) & np.isfinite(col.astype(np.float64)) & (eff_row < n_dates)
    # 按报告期升序，保证同一 (col,row) 时后期覆盖前期
    order = np.argsort(df['report_period'].to_numpy()[valid], kind='stable')
    eff_row_v = eff_row[valid][order]
    col_v = col[valid][order].astype(np.intp)

    out = {'trade_dates': d['trade_dates'], 'stock_codes': stock_codes}
    for name in _INDICATORS:
        vals = pd.to_numeric(df[name], errors='coerce').to_numpy()[valid][order]
        mat = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
        place = np.isfinite(vals)
        mat[eff_row_v[place], col_v[place]] = vals[place]
        mat = _ffill_axis0(mat).astype(np.float32)
        out[name] = mat
        cov = np.isfinite(mat).any(axis=0).sum()
        print(f'  {name:14s} 覆盖股票={int(cov):4d}  非空单元={int(np.isfinite(mat).sum()):,}')

    np.savez_compressed(OUT_PATH, **out)
    print(f'保存 -> {OUT_PATH}  ({n_dates}d x {n_stocks}s, LAG={LAG_DAYS}d)')


if __name__ == '__main__':
    sys.exit(main())
