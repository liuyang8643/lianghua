"""为 factor_db 中所有因子构建全截面 rank 指纹缓存（signatures.npz）。

单次加载 npz，逐因子算每日截面 rank → 指纹（similarity.signature），写入缓存。
指纹用于多样性 / 全截面相关矩阵（GA NSGA 目标、报告矩阵）。

用法:
  uv run python -m factor_db.build_signatures
  uv run python -m factor_db.build_signatures --start 19930101 --end 20181231 --dim 16384
"""
import argparse
from datetime import datetime

import numpy as np

from core.factors.registry import get_all_factor_classes
from core.runtime import load_runtime_npz
from core.scoring import scores_to_ranks
from factor_db import db, similarity
from llm_ga.evaluator import _parse_date
from llm_ga.config import TRAIN_END, TRAIN_START
from utils.stock.time import get_trading_date_span


def build(start: str, end: str, pool: str, dim: int, seed: int) -> None:
    all_classes = get_all_factor_classes()
    names = [f['name'] for f in db.list_factors() if f['name'] in all_classes]
    print(f'因子 {len(names)} 个，区间 {start}~{end}，dim={dim}')

    dts = [datetime.combine(d, datetime.min.time())
           for d in get_trading_date_span(_parse_date(start), _parse_date(end))]
    max_lb = max((all_classes[n].hist_days for n in names), default=0) or None
    data = load_runtime_npz(dts, max_lookback=max_lb)
    if data is None:
        raise FileNotFoundError('runtime npz 未覆盖该区间')

    npz_stocks = [str(s) for s in data['stock_codes']]
    py_dates = [d.astype('datetime64[D]').item() for d in data['trade_dates']]
    date_to_idx = {d: i for i, d in enumerate(py_dates)}
    di = np.array([date_to_idx[dt.date()] for dt in dts if dt.date() in date_to_idx])
    factor_data = {**data, 'stock_codes': npz_stocks, 'trade_dates': py_dates}
    print(f'交易日 {len(di)} 个，股票 {len(npz_stocks)} 只\n')

    sigs = []
    for name in names:
        raw = all_classes[name]().calc_batch(factor_data)
        ranks = scores_to_ranks(raw.astype(np.float32, copy=False))
        sig = similarity.signature(ranks[di], dim, seed)
        sigs.append(sig)
        print(f'  + {name}: |sig|={np.linalg.norm(sig):.3f}')

    sigs = np.array(sigs, dtype=np.float32)
    meta = {'dim': dim, 'seed': seed, 'start': start, 'end': end,
            'n_days': int(len(di)), 'n_stocks': len(npz_stocks), 'pool': pool}
    similarity.save_cache(names, sigs, meta)

    corr = similarity.correlation_matrix(sigs)
    div = similarity.diversity_scores(corr)
    order = np.argsort(div)
    print(f'\n指纹缓存已写入: {similarity._CACHE}')
    print('多样性最低（最像别人）TOP5:')
    for i in order[:5]:
        j = int(np.argmax(np.where(np.arange(len(corr)) == i, -9, corr[i])))
        print(f'  {names[i]:<22} div={div[i]:.3f}  最像 {names[j]}(corr={corr[i][j]:.3f})')


def main():
    p = argparse.ArgumentParser(description='构建因子全截面 rank 指纹缓存')
    p.add_argument('--start', type=str, default=TRAIN_START)
    p.add_argument('--end', type=str, default=TRAIN_END)
    p.add_argument('--pool', type=str, default='all_A')
    p.add_argument('--dim', type=int, default=similarity.DEFAULT_DIM)
    p.add_argument('--seed', type=int, default=similarity.DEFAULT_SEED)
    args = p.parse_args()
    build(args.start, args.end, args.pool, args.dim, args.seed)


if __name__ == '__main__':
    main()
