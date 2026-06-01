"""快速验证：财务全空(2015) vs 财务齐全(2024)，g38_4 分数是否退化成 size。"""
import numpy as np
from factor_db.factors.Factor_20260531_005210_g38_4 import Factor_20260531_005210_g38_4
from factor_db.factors.TrueMarketCap import TrueMarketCap

d = np.load('data/runtime/runtime_1990-12-19_2026-05-29.npz', allow_pickle=False)
panel = {k: d[k] for k in d.files}
dts = d['trade_dates'].astype('datetime64[D]')

g38 = Factor_20260531_005210_g38_4().calc_batch(panel)
tmc = TrueMarketCap().calc_batch(panel)


def analyze(label, t):
    row, sz, eps = g38[t], tmc[t], panel['eps'][t]
    v = np.isfinite(row) & np.isfinite(sz)
    n = int(v.sum())
    uniq = len(np.unique(np.round(row[v], 6)))
    # 截面 rank 相关 g38 vs 市值(size)
    a = np.argsort(np.argsort(row[v])); b = np.argsort(np.argsort(sz[v]))
    corr = np.corrcoef(a, b)[0, 1] if n > 2 else float('nan')
    print(f'{label} ({dts[t]}):')
    print(f'   eps有效股: {np.isfinite(eps).sum():>4} 只 | g38有效股: {n:>4} 只 | g38不同分数值: {uniq} 个')
    print(f'   g38 与 TrueMarketCap 截面rank相关: {corr:+.3f}')


# 找 2015 和 2024 各一个交易日
t2015 = int(np.where(dts.astype("datetime64[Y]").astype(int) + 1970 == 2015)[0][100])
t2024 = int(np.where(dts.astype("datetime64[Y]").astype(int) + 1970 == 2024)[0][100])
analyze('财务全空 2015', t2015)
analyze('财务齐全 2024', t2024)
