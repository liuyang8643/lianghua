"""把 factor_db 中所有现存因子按训练区间回测一遍，补全 factor_runs 明细记录（幂等）。

复用 GA 评测器（evaluator.evaluate_detailed），与 GA fitness 同口径：
  - 区间默认 1993-2018（TRAIN）、股票池 all_A、top20
  - 跳过连续性闸门（现存库里既有连续也有离散因子，都要落明细）
  - 已有 run 的因子跳过（append-only，不重复）

用法:
  uv run python -m factor_db.backfill_records
  uv run python -m factor_db.backfill_records --start 19930101 --end 20181231 --buy-n 20
"""
import argparse

from core.factors.registry import get_all_factor_classes
from factor_db import db, records
from llm_ga import evaluator
from llm_ga.config import TRAIN_END, TRAIN_START

_POOL_PRESETS = {
    'all_A': ('60', '00', '30', '688'),
    'main': ('60', '00'),
    'gem': ('30',),
    'star': ('688',),
}


def backfill(start: str, end: str, buy_n: int, pool: str, *, force: bool = False) -> None:
    db.init_db()
    records.init_records()
    classes = get_all_factor_classes()
    factors = db.list_factors()
    pool_prefixes = _POOL_PRESETS[pool]

    print(f'回测口径: {start}~{end}, 股票池={pool}, top{buy_n}')
    print(f'因子总数 {len(factors)}，构建股票池/日期...')
    dates, stocks = evaluator.build_universe(start, end, pool_prefixes)
    print(f'交易日 {len(dates)} 个，股票 {len(stocks)} 只\n')

    done, skipped, failed = 0, 0, 0
    for f in factors:
        name = f['name']
        if not force and records.has_run(name):
            skipped += 1
            print(f'  - {name}: 已有 run，跳过')
            continue
        cls = classes.get(name)
        if cls is None:
            failed += 1
            print(f'  ! {name}: 注册表中找不到因子类，跳过')
            continue
        try:
            m = evaluator.evaluate_detailed(cls, name, dates, stocks, buy_n, check=False)
        except Exception as e:
            failed += 1
            print(f'  ! {name}: 回测失败 ({type(e).__name__}: {e})')
            continue
        records.add_run(
            name, bt_start=start, bt_end=end, buy_n=buy_n, stock_pool=pool,
            dates=m['dates'], daily_returns=m['daily_returns'], topn=m['topn'],
            sharpe=m['sharpe'], annualized=m['annualized'],
            max_dd=m['max_dd'], n_trades=m['n_trades'],
        )
        done += 1
        print(f'  + {name}: sharpe={m["sharpe"]:.3f} 年化={m["annualized"]:.1f}% '
              f'maxdd={m["max_dd"]:.1f}% trades={m["n_trades"]} days={len(m["dates"])}')

    print(f'\n完成: 新增 {done}，跳过 {skipped}，失败 {failed}')


def main():
    p = argparse.ArgumentParser(description='回填 factor_db 明细记录')
    p.add_argument('--start', type=str, default=TRAIN_START)
    p.add_argument('--end', type=str, default=TRAIN_END)
    p.add_argument('--buy-n', type=int, default=20)
    p.add_argument('--pool', type=str, default='all_A', choices=list(_POOL_PRESETS))
    p.add_argument('--force', action='store_true', help='已有 run 也重跑（append 一条新记录）')
    args = p.parse_args()
    backfill(args.start, args.end, args.buy_n, args.pool, force=args.force)


if __name__ == '__main__':
    main()
