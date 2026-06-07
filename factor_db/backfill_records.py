"""把 factor_db 中所有现存因子按训练区间回测一遍，补全 factor_runs 明细记录（幂等）。

复用 GA 评测器（evaluator.evaluate_detailed），与 GA fitness 同口径：
  - 区间默认见 llm_ga.config TRAIN_START/TRAIN_END、股票池 all_A、top20
  - 跳过连续性闸门（现存库里既有连续也有离散因子，都要落明细）
  - 已有 run 的因子跳过（append-only，不重复）

用法:
  uv run python -m factor_db.backfill_records
  uv run python -m factor_db.backfill_records --start 20050101 --end 20260605 --force
  uv run python -m factor_db.backfill_records --force --workers 30
"""
import argparse
from multiprocessing import get_context

from core.factors.registry import get_all_factor_classes
from core.runtime import load_runtime_npz
from factor_db import db, records
from llm_ga import evaluator
from llm_ga.config import TRAIN_END, TRAIN_START

_POOL_PRESETS = {
    'all_A': ('60', '00', '30', '688'),
    'main': ('60', '00'),
    'gem': ('30',),
    'star': ('688',),
}

_g_dates = None
_g_stocks = None
_g_panel = None
_g_classes = None


def _worker_init(dates, stocks, max_lookback):
    global _g_dates, _g_stocks, _g_panel, _g_classes
    _g_dates = dates
    _g_stocks = stocks
    _g_panel = load_runtime_npz(dates, max_lookback=max_lookback)
    _g_classes = get_all_factor_classes()


def _worker_eval(task):
    name, buy_n = task
    cls = _g_classes.get(name)
    if cls is None:
        return name, None, '注册表中找不到因子类'
    try:
        m = evaluator.evaluate_detailed(
            cls, name, _g_dates, _g_stocks, buy_n, check=False, data=_g_panel,
        )
        return name, m, None
    except Exception as e:
        return name, None, f'{type(e).__name__}: {e}'


def _max_lookback(classes, names):
    lbs = [classes[n].hist_days for n in names if n in classes and classes[n].hist_days]
    return max(lbs) if lbs else None


def _save_run(name, start, end, buy_n, stock_pool, m):
    records.add_run(
        name, bt_start=start, bt_end=end, buy_n=buy_n, stock_pool=stock_pool,
        dates=m['dates'], daily_returns=m['daily_returns'], topn=m['topn'],
        sharpe=m['sharpe'], annualized=m['annualized'],
        max_dd=m['max_dd'], n_trades=m['n_trades'],
    )


def _print_ok(name, m):
    print(f'  + {name}: sharpe={m["sharpe"]:.3f} 年化={m["annualized"]:.1f}% '
          f'maxdd={m["max_dd"]:.1f}% trades={m["n_trades"]} days={len(m["dates"])}')


def backfill(start: str, end: str, buy_n: int, stock_pool: str, *,
             force: bool = False, workers: int = 1) -> None:
    db.init_db()
    records.init_records()
    classes = get_all_factor_classes()
    factors = db.list_factors()
    pool_prefixes = _POOL_PRESETS[stock_pool]

    print(f'回测口径: {start}~{end}, 股票池={stock_pool}, top{buy_n}, workers={workers}')
    print(f'因子总数 {len(factors)}，构建股票池/日期...')
    dates, stocks = evaluator.build_universe(start, end, pool_prefixes)
    print(f'交易日 {len(dates)} 个，股票 {len(stocks)} 只\n')

    pending = [f['name'] for f in factors if force or not records.has_run(f['name'])]
    skipped = len(factors) - len(pending)
    for f in factors:
        if f['name'] not in pending:
            print(f'  - {f["name"]}: 已有 run，跳过')

    if not pending:
        print(f'\n完成: 新增 0，跳过 {skipped}，失败 0')
        return

    max_lb = _max_lookback(classes, pending)
    done, failed = 0, 0

    if workers <= 1:
        panel = load_runtime_npz(dates, max_lookback=max_lb)
        for name in pending:
            cls = classes.get(name)
            if cls is None:
                failed += 1
                print(f'  ! {name}: 注册表中找不到因子类，跳过')
                continue
            try:
                m = evaluator.evaluate_detailed(
                    cls, name, dates, stocks, buy_n, check=False, data=panel,
                )
            except Exception as e:
                failed += 1
                print(f'  ! {name}: 回测失败 ({type(e).__name__}: {e})')
                continue
            _save_run(name, start, end, buy_n, stock_pool, m)
            done += 1
            _print_ok(name, m)
    else:
        ctx = get_context('spawn')
        tasks = [(n, buy_n) for n in pending]
        with ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(dates, stocks, max_lb),
        ) as proc_pool:
            for name, m, err in proc_pool.imap_unordered(_worker_eval, tasks):
                if err is not None:
                    failed += 1
                    print(f'  ! {name}: 回测失败 ({err})')
                    continue
                _save_run(name, start, end, buy_n, stock_pool, m)
                done += 1
                _print_ok(name, m)

    print(f'\n完成: 新增 {done}，跳过 {skipped}，失败 {failed}')


def main():
    p = argparse.ArgumentParser(description='回填 factor_db 明细记录')
    p.add_argument('--start', type=str, default=TRAIN_START)
    p.add_argument('--end', type=str, default=TRAIN_END)
    p.add_argument('--buy-n', type=int, default=20)
    p.add_argument('--pool', type=str, default='all_A', choices=list(_POOL_PRESETS))
    p.add_argument('--force', action='store_true', help='已有 run 也重跑（append 一条新记录）')
    p.add_argument('--workers', type=int, default=1, help='并行 worker 数（每 worker 预加载一份 NPZ）')
    args = p.parse_args()
    backfill(args.start, args.end, args.buy_n, args.pool,
             force=args.force, workers=args.workers)


if __name__ == '__main__':
    main()
