import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)


from core.scoring import scores_to_ranks
from core.runtime import load_runtime_npz
from utils.stock.time import get_trading_date_span
from utils.windows_awake import keep_windows_awake
from core.ga import (
    DEFAULT_GA_PROFILE,
    build_individual_config,
    generate_initial_configs,
    get_intrinsic_params,
    get_mode_configs,
    get_profile,
    get_profile_factor_classes,
    get_profile_preload_range,
    get_profile_search_spaces,
    get_profile_weight_search_spaces,
    resolve_profile_name,
    sample_factor_choice,
    get_profile_filter_factor_classes,
)

from datetime import date, datetime
from testback.logger import testback_logger
from core.backtest import (
  _backtest_direct,
  _compute_list_dates, _compute_timing_multipliers,
  _load_all_index_data,
  _format_pool, _format_timing,
  _resolve_output_dir,
)
from core.metrics import compute_core_metrics
from core.runtime import load_runtime_stock_codes
from core.strategy_config import strategy_config_payload


# ========== GA 核心函数 ==========

def _config_key(config: dict) -> tuple:
    def freeze(value):
        if isinstance(value, dict):
            return tuple(sorted((k, freeze(v)) for k, v in value.items()))
        if isinstance(value, list):
            return tuple(freeze(v) for v in value)
        return value

    parts = []
    for pdef in get_intrinsic_params():
        val = config.get(pdef['config_key'])
        parts.append(freeze(val))
    parts.append(tuple(sorted(config['weights'].items())))
    return tuple(parts)


def _format_config_params(config: dict, profile_name: str) -> str:
    """动态格式化所有搜索参数，驱动自 intrinsic_params + search_spaces。"""
    spaces = get_profile_search_spaces(profile_name)
    parts = []
    for pdef in get_intrinsic_params():
        key = pdef['key']
        if key not in spaces:
            continue
        val = config.get(pdef['config_key'])
        if val is None or (pdef['type'] == 'int' and val == 0):
            continue
        display = pdef['display']
        if isinstance(val, bool):
            parts.append(f"{display}={'ON' if val else 'OFF'}")
        elif isinstance(val, float):
            parts.append(f"{display}={val:.2f}")
        elif isinstance(val, list):
            parts.append(f"{display}={_format_pool(tuple(val))}")
        else:
            parts.append(f"{display}={val}")
    return ', '.join(parts)

def _format_best_log(best_cfg: dict, profile_name: str, prefix: str, train_fitness: float, avg_fitness: float,
                     sharpe=None, val_sharpe=None, test_sharpe=None, live_total=None, live_test=None,
                     benchmark_calmar=None):
    sorted_w = sorted(best_cfg['weights'].items(), key=lambda x: -abs(x[1]))
    w_str = ', '.join(f'{k}={v:.1f}' for k, v in sorted_w)
    params_str = _format_config_params(best_cfg, profile_name)
    benchmark_str = f'(上证指数={benchmark_calmar:.3f})' if benchmark_calmar is not None else ''
    msg = f"{prefix}: 训练Calmar={train_fitness:.3f}{benchmark_str}/{avg_fitness:.3f}"
    if sharpe is not None:
        msg += f" S={sharpe:.2f}"
    if val_sharpe is not None:
        msg += f" | 验证Calmar={val_sharpe:.3f}"
    if test_sharpe is not None:
        msg += f" | 测试Calmar={test_sharpe:.3f}"
    msg += f" | {params_str} | {w_str}"
    if live_total is not None:
        msg += f" | 实盘: total={live_total:.3f}"
        if live_test is not None:
            msg += f" test={live_test:.3f}"
    return msg


def ga_optimizer(results, state, population_size=24, hall_of_fame_size=24, profile_name=DEFAULT_GA_PROFILE, ga_cache=None, gen=0):
  import random

  results_list = list(results) if not isinstance(results, list) else results

  for r in results_list:
    key = _config_key(r['individual_config'])
    state['fitness_cache'][key] = r.get('calmar', r.get('sharpe', -1000.0))

  if not state['population']:
    results_list.sort(key=lambda r: r.get('calmar', r.get('sharpe', -1000.0)), reverse=True)
    state['population'] = [r['individual_config'] for r in results_list[:population_size]]

  def get_fitness(config):
    key = _config_key(config)
    return state.get('fitness_cache', {}).get(key)

  # GA 多样性参数
  elite_frac = 0.10
  tournament_k = 5          # 中等选择压力 (3太弱/7过强在欺骗区)
  immigrant_frac = 0.15     # 中等移民率：每代保留足够随机移民，避免早熟收敛

  # 父代选择：少量真精英 + 锦标赛选择（从历史全局池抽，打破当代近亲繁殖、降低选择压力）
  if ga_cache and len(ga_cache) >= population_size:
    unique_cfgs = {}
    for key, val in ga_cache.items():
      cfg = val.get('individual_config')
      fit = val.get('calmar', val.get('sharpe', -1000.0))
      if cfg is not None and fit > -900:
        unique_cfgs[key] = (cfg, fit)
    pool = sorted(unique_cfgs.values(), key=lambda x: x[1], reverse=True)
    total = len(pool)
    n_elite = min(max(2, int(round(elite_frac * population_size))), total)
    elite_cfgs = [cfg for cfg, _ in pool[:n_elite]]
    # 剩余父代用锦标赛选择：每次随机取 k 个，留适应度最高者，给次优个体繁殖机会
    n_tournament = population_size - n_elite
    selected = []
    for _ in range(n_tournament):
      aspirants = random.sample(pool, min(tournament_k, total))
      selected.append(max(aspirants, key=lambda x: x[1])[0])
    parents = elite_cfgs + selected
  else:
    population_with_fitness = [(ind, fit) for ind in state['population']
                               if (fit := get_fitness(ind)) is not None]
    if not population_with_fitness:
      population_with_fitness = [
        (r['individual_config'], r.get('calmar', r.get('sharpe', -1000.0)))
        for r in results_list
      ]
    population_with_fitness.sort(key=lambda x: x[1], reverse=True)
    parents = [ind for ind, _ in population_with_fitness[:population_size]]

  all_individuals = state['hall_of_fame'] + parents
  unique_dict = {}
  for ind in all_individuals:
    key = _config_key(ind)
    fitness = get_fitness(ind)
    if fitness is None:
      continue
    if key not in unique_dict or fitness > unique_dict[key][1]:
      unique_dict[key] = (ind, fitness)
  sorted_hof = sorted(unique_dict.values(), key=lambda x: x[1], reverse=True)
  state['hall_of_fame'] = [ind for ind, _ in sorted_hof[:hall_of_fame_size]]

  has_weight_search = get_profile_weight_search_spaces(profile_name) is not None
  profile = get_profile(profile_name)
  has_factor_choice = profile.get('factor_choice_space') is not None
  search_spaces = get_profile_search_spaces(profile_name)

  def _crossover_field(p1, p2, key, default=None):
    return (p1.get(key, default) if random.random() < 0.5 else p2.get(key, default))

  def crossover_config(p1, p2):
    kwargs = {}
    for pdef in get_intrinsic_params():
        key = pdef['key']
        if key in search_spaces:
            kwargs[key] = _crossover_field(p1, p2, pdef['config_key'])
    if has_factor_choice:
        kwargs['factor_choice'] = _crossover_field(p1, p2, 'factor_choice')
    crossed_weights = None
    if has_weight_search:
      crossed_weights = {}
      all_keys = set(p1.get('weights', {})) | set(p2.get('weights', {}))
      for k in all_keys:
        w1 = p1['weights'].get(k, 0.0)
        w2 = p2['weights'].get(k, 0.0)
        crossed_weights[k] = w1 if random.random() < 0.5 else w2
    kwargs['weights'] = crossed_weights
    kwargs['profile_name'] = profile_name
    return build_individual_config(**kwargs)

  weight_spaces = get_profile_weight_search_spaces(profile_name) if has_weight_search else {}

  def _collect_dims():
    dims = [p['key'] for p in get_intrinsic_params() if p['key'] in search_spaces]
    if has_factor_choice: dims.append('factor_choice')
    for k in weight_spaces: dims.append(f'weight_{k}')
    return dims

  import math
  _all_dims = _collect_dims()

  def mutate_config(config):
    if not _all_dims:
      return build_individual_config(config['buy_n'],
                                      weights=config.get('weights'), profile_name=profile_name)
    # 变异率退火：前期 25-35% 广泛探索，后期 8-15% 精细搜索
    t = gen / max(gen + 20, 1)
    lo = 0.25 - 0.17 * t
    hi = 0.35 - 0.20 * t
    n_mutate = max(1, min(math.ceil(random.uniform(lo, hi) * len(_all_dims)), len(_all_dims)))
    mutate_dims = set(random.sample(_all_dims, n_mutate))
    kwargs = {}
    for pdef in get_intrinsic_params():
        key = pdef['key']
        if key in mutate_dims:
            space = search_spaces.get(key)
            kwargs[key] = random.choice(space) if space else config.get(pdef['config_key'])
        else:
            kwargs[key] = config.get(pdef['config_key'])
    if has_factor_choice:
        kwargs['factor_choice'] = sample_factor_choice(profile_name=profile_name) if 'factor_choice' in mutate_dims else config.get('factor_choice')
    mutated_weights = dict(config.get('weights', {}))
    for k in weight_spaces:
      if f'weight_{k}' in mutate_dims:
        mutated_weights[k] = random.choice(weight_spaces[k])
      elif k not in mutated_weights:
        mutated_weights[k] = 0.0
    kwargs['weights'] = mutated_weights
    kwargs['profile_name'] = profile_name
    return build_individual_config(**kwargs)

  # 随机移民：每代注入全新随机个体，维持探索、逃离局部盆地（不依赖现有父代）
  n_immigrants = min(max(1, int(round(immigrant_frac * population_size))), population_size)
  n_crossover_children = population_size - n_immigrants

  children = []
  while len(children) < n_crossover_children:
    if len(parents) == 1:
      child = mutate_config(parents[0])
    else:
      p1, p2 = random.sample(parents, 2)
      child = mutate_config(crossover_config(p1, p2))
    children.append(child)
  children.extend(generate_initial_configs(n_immigrants, profile_name=profile_name))
  state['population'] = parents + children

  next_configs = parents + children
  return next_configs


# ========== 工具函数 ==========


def _find_latest_ga_dir() -> Path | None:
  results_dir = Path('results')
  if not results_dir.is_dir():
    return None
  candidates = []
  for d in results_dir.iterdir():
    if not d.is_dir():
      continue
    if not (d.name.startswith('ga_') or d.name.startswith('debug_')):
      continue
    if (d / 'all_results.jsonl').exists():
      candidates.append((d.name, d))
  if not candidates:
    return None
  candidates.sort(reverse=True)
  return candidates[0][1]


def _rebuild_from_jsonl(output_dir: Path) -> tuple[dict, int]:
  import json as _json
  ga_cache = {}
  last_gen = -1
  jsonl_path = output_dir / 'all_results.jsonl'
  if jsonl_path.exists():
    with open(jsonl_path, 'r', encoding='utf-8') as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        r = _json.loads(line)
        cfg = r['config']
        key = _config_key(cfg)
        fitness = r.get('calmar', r['sharpe'])  # 兼容旧 JSONL（只有 sharpe）
        if key not in ga_cache or fitness > ga_cache[key]['calmar']:
          ga_cache[key] = {'individual_config': cfg, 'calmar': fitness, 'sharpe': r['sharpe']}
        if r['generation'] > last_gen:
          last_gen = r['generation']
  return ga_cache, last_gen


def _rebuild_ga_state(ga_cache: dict) -> dict:
  """从 ga_cache 重建 ga_state（population + hall_of_fame + fitness_cache）。"""
  sorted_entries = sorted(ga_cache.values(), key=lambda v: v['calmar'], reverse=True)
  hall_of_fame = [v['individual_config'] for v in sorted_entries[:100]]
  fitness_cache = {_config_key(v['individual_config']): v['calmar'] for v in sorted_entries}
  return {
    'population': [],
    'hall_of_fame': hall_of_fame,
    'fitness_cache': fitness_cache,
  }



# === SharedMemory 数据共享（Windows spawn，替代 memmap 文件 I/O） ===
_MEMMAP_DIRS: list = []
_SHM_BLOCKS: list = []  # (SharedMemory, np.ndarray) 引用，防止被 GC 回收

# Per-worker cache: pool initializer 打开一次，所有 eval 复用
_worker_shm_cache: dict = {}

def _worker_initializer(shm_entries):
  """Pool initializer：每个 worker 进程启动时打开 SharedMemory，缓存在模块全局。
  shm_entries: [(name, shm_name, shape, dtype_str), ...]"""
  from multiprocessing.shared_memory import SharedMemory
  global _worker_shm_cache
  _worker_shm_cache = {}
  for name, shm_name, shape, dtype_str in shm_entries:
    shm = SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    _worker_shm_cache[name] = (shm, arr)

def _arrays_from_memmap(info: dict) -> dict:
  return {name: np.memmap(filepath, dtype=np.dtype(dtype_str), mode='r', shape=shape)
          for name, (filepath, shape, dtype_str) in info.items()}

def _cleanup_memmap():
  import shutil
  for d in _MEMMAP_DIRS:
    shutil.rmtree(d, ignore_errors=True)
  _MEMMAP_DIRS.clear()

def _cleanup_shm():
  """清理所有 SharedMemory 块（主进程调用）"""
  for shm, _ in _SHM_BLOCKS:
    try:
      shm.close()
      shm.unlink()
    except Exception as e:
      testback_logger.warning(f"SharedMemory 清理失败 {shm.name}: {e}")
  _SHM_BLOCKS.clear()


def _factor_worker(args):
  """Compute single factor → rank + NaN mask → memmap. Module-level for spawn pickle."""
  factor_cls, base_info, stock_codes, trade_dates, tmpdir, row_slice = args
  data = _arrays_from_memmap(base_info)
  n_full = len(trade_dates)
  if row_slice is not None:
    r0, r1 = row_slice
    data = {k: (v[r0:r1] if v.ndim >= 2 else v) for k, v in data.items()}
    trade_dates = trade_dates[r0:r1]
  data['stock_codes'] = stock_codes
  data['trade_dates'] = trade_dates
  f = factor_cls()
  name = f.__class__.__name__
  with np.errstate(all='ignore'):
      raw = f.calc_batch(data)
  raw_nan = np.isnan(raw)
  scores = scores_to_ranks(raw.astype(np.float32, copy=False))
  if row_slice is not None:
    full = np.full((n_full, scores.shape[1]), np.nan, dtype=scores.dtype)
    full[r0:r1] = scores
    scores = full
    full_nan = np.ones((n_full, raw_nan.shape[1]), dtype=bool)
    full_nan[r0:r1] = raw_nan
    raw_nan = full_nan
  filepath = Path(tmpdir) / f'factor_{name}.bin'
  scores.tofile(str(filepath))
  return name, (str(filepath), scores.shape, str(scores.dtype)), raw_nan


def _worker_evaluate(args):
  global _worker_shm_cache
  train_info, score_keys, valid_dates, date_indices, stock_indices, \
      all_stocks_list, config, index_data, list_dates_map = args

  needed = score_keys | {'open', 'close', 'high', 'low', 'preClose', 'volume', 'amount', 'total_share', 'st_mask', 'issue_price', 'stock_codes', 'trade_dates', '_factor_intersection'}
  all_arrays = {k: arr for k, (shm, arr) in _worker_shm_cache.items() if k in needed}
  data = {k: v for k, v in all_arrays.items() if k not in score_keys and k != '_factor_intersection'}
  all_scores = {k: v for k, v in all_arrays.items() if k in score_keys}
  # 加权因子缺失集合固定取交集
  intersection = all_arrays.get('_factor_intersection')
  filter_masks = {'_factor_intersection': intersection} if intersection is not None else {}
  for name, enabled in config.get('filter_factors', {}).items():
    if enabled and name in all_arrays:
      filter_masks[name] = np.isfinite(all_arrays[name])

  stock_pool = config.get('stock_pool') or ('60', '00', '30', '688')
  if isinstance(stock_pool, list):
      stock_pool = tuple(stock_pool)
  pool_stocks = [s for s in all_stocks_list if s.startswith(stock_pool)]

  timing_multipliers = _compute_timing_multipliers(config, valid_dates, index_data)

  hp = config.get('holding_period', 1)
  n_starts = hp if hp > 1 else 1

  calmar_list, sharpe_list, ann_list, dd_list, tr_list = [], [], [], [], []
  for offset in range(n_starts):
      od = valid_dates[offset:]
      oi = date_indices[offset:]
      om = _compute_timing_multipliers(config, od, index_data) if index_data else None
      r = _backtest_direct(
          data, all_scores, od, oi, pool_stocks, stock_indices,
          weights=config['weights'], buy_n=config['buy_n'], sell_m=config['sell_m'],
          holding_period=hp,
          position_multipliers=om if om is not None and len(om) == len(od) else None,
          list_dates_map=list_dates_map,
          lightweight=True, limit_up_protection=config.get('limit_up_protection', False),
          rebalance=config.get('rebalance', True),
          prefilter_n=config.get('prefilter_n'),
          filter_masks=filter_masks)
      m = compute_core_metrics(r['daily_returns'])
      calmar_list.append(abs(m['annualized']) / abs(m['max_drawdown']) if m['max_drawdown'] != 0 else 0.0)
      sharpe_list.append(m['sharpe'])
      ann_list.append(m['annualized'])
      dd_list.append(m['max_drawdown'])
      tr_list.append(r['total_return'])

  return {
    'individual_config': config,
    'total_return': float(np.mean(tr_list)),
    'calmar': float(np.mean(calmar_list)),
    'sharpe': float(np.mean(sharpe_list)),
    'annualized': float(np.mean(ann_list)),
    'max_drawdown': float(np.mean(dd_list)),
    'cleared_positions_count': r['cleared_positions_count'],
  }


def _eval_parallel(worker_args, results_list, ga_cache, logger, pool):
  import gc
  import json as json_mod

  for result in pool.imap_unordered(_worker_evaluate, worker_args, chunksize=1):
    results_list.append(result)
    key = json_mod.dumps(_config_key(result['individual_config']), ensure_ascii=False)
    ga_cache[key] = result
  gc.collect()


def _run_ga(args, mode_config, backtest_datetime_list, all_stocks, profile_name=DEFAULT_GA_PROFILE, resume_dir=None):
  """GA/调试模式：多进程并行回测。预计算共享内存复用。"""
  import json as json_mod
  import os
  import pickle
  import time

  is_debug = args.mode == 'debug'
  generations = mode_config['generations']

  if is_debug:
    population_size = mode_config['population_size']
  else:
    population_size = mode_config['population_size'] or 50

  mode_name = 'debug' if is_debug else 'ga'
  output_dir = resume_dir or _resolve_output_dir(args.output_dir, mode_name)
  factor_classes = get_profile_factor_classes(profile_name)
  filter_factor_classes = get_profile_filter_factor_classes(profile_name)

  # 添加文件日志 sink，将所有控制台日志同步写入 log 文件
  log_path = output_dir / 'ga.log'
  testback_logger.add_file_sink(str(log_path))
  testback_logger.info(f"日志文件: {log_path}")

  from multiprocessing import get_context

  ga_state = {'population': [], 'hall_of_fame': [], 'fitness_cache': {}}
  start_generation = 0
  ga_cache = {}

  val_cache_path = output_dir / 'val_cache.json'
  val_eval_cache = {}
  if val_cache_path.exists():
    with open(val_cache_path, 'r', encoding='utf-8') as f:
      val_eval_cache = json_mod.load(f)
    testback_logger.info(f"加载验证集缓存: {len(val_eval_cache)} 条")

  test_cache_path = output_dir / 'test_cache.json'
  test_eval_cache = {}
  if test_cache_path.exists():
    with open(test_cache_path, 'r', encoding='utf-8') as f:
      test_eval_cache = json_mod.load(f)
    testback_logger.info(f"加载测试集缓存: {len(test_eval_cache)} 条")


  t0 = time.time()

  def _build_date_subset(date_list, npz_dates, date_to_idx):
    """从全量 npz_dates 中提取目标日期段的 indices。纯索引，无 NPZ 加载无因子计算。"""
    date_indices = []
    valid_dates = []
    for dt in date_list:
      d = dt.date()
      di = date_to_idx.get(d)
      if di is None:
        continue
      date_indices.append(di)
      valid_dates.append(dt)
    return valid_dates, date_indices

  # === 全量 NPZ 加载 + 因子计算（一次性）===
  data = load_runtime_npz(backtest_datetime_list)
  if data is None:
    first_d = backtest_datetime_list[0].strftime('%Y%m%d')
    last_d = backtest_datetime_list[-1].strftime('%Y%m%d')
    raise FileNotFoundError(f"未找到覆盖 {first_d}~{last_d} 的 runtime npz 文件")

  npz_stocks = [str(s) for s in data['stock_codes']]
  stock_indices = {c: i for i, c in enumerate(npz_stocks)}
  all_valid_stocks = [s for s in all_stocks if s in stock_indices]
  npz_dates = data['trade_dates']
  py_dates = [d.astype('datetime64[D]').item() for d in npz_dates]

  date_to_idx = {}
  for i, d in enumerate(npz_dates):
    date_to_idx[d.astype('datetime64[D]').item()] = i

  import tempfile
  tmpdir = tempfile.mkdtemp(prefix='memmap_', dir=str(output_dir))
  _MEMMAP_DIRS.append(tmpdir)
  scores_dir = Path(tmpdir) / "scores"
  scores_dir.mkdir(exist_ok=True)

  info = {}
  for k, v in data.items():
    if k == 'stock_names':
      continue
    arr = np.ascontiguousarray(v)
    filepath = Path(tmpdir) / f'{k}.bin'
    arr.tofile(str(filepath))
    info[k] = (str(filepath), arr.shape, str(arr.dtype))

  # 提前计算三段时间索引，确定因子计算所需的最小天数范围
  valid_dates, date_indices = _build_date_subset(backtest_datetime_list, npz_dates, date_to_idx)

  val_start = date(2019, 1, 1)
  val_end = date(2022, 12, 31)
  val_datetime_list = [datetime.combine(d, datetime.min.time()) for d in get_trading_date_span(val_start, val_end)]
  val_valid_dates, val_date_indices = _build_date_subset(val_datetime_list, npz_dates, date_to_idx)

  test_start = date(2023, 1, 1)
  # 测试集末日 = runtime npz 实际数据末日（train/val/test 同切自一份全量 npz），避免硬编码过期；
  # 不可用训练集末日（= 预加载区间末日，可能早于 test_start），否则 test_start>test_end。
  test_end = py_dates[-1]
  test_datetime_list = [datetime.combine(d, datetime.min.time()) for d in get_trading_date_span(test_start, test_end)]
  test_valid_dates, test_date_indices = _build_date_subset(test_datetime_list, npz_dates, date_to_idx)

  max_hist = max(f.hist_days for f in factor_classes)
  all_idx = date_indices + val_date_indices + test_date_indices
  row_start = max(0, min(all_idx) - max_hist)
  row_end = max(all_idx) + 1
  n_needed = row_end - row_start
  testback_logger.info(f"因子计算范围: [{row_start}:{row_end}] = {n_needed} 天 (全量 {len(npz_dates)} 天, 截断 {len(npz_dates)-n_needed} 天)")

  # 统一打印当前 profile 搜索空间
  profile = get_profile(profile_name)
  spaces = get_profile_search_spaces(profile_name)
  weight_spaces = get_profile_weight_search_spaces(profile_name)
  search_info = []
  for pdef in get_intrinsic_params():
      space = spaces.get(pdef['key'])
      if space:
          vals = [str(v) for v in space]
          search_info.append(f"{pdef['display']}=[{','.join(vals[:8])}{'...' if len(vals) > 8 else ''}]" if len(vals) <= 10 else f"{pdef['display']}={len(vals)}值")
  if profile.get('factor_choice_space'):
      search_info.append(f"fc={len(profile['factor_choice_space'])}值")
  if weight_spaces:
      for k, vals in weight_spaces.items():
          search_info.append(f"w_{k}=[{min(vals):.1f}-{max(vals):.1f}]")
  testback_logger.info(f"搜索空间: " + " | ".join(search_info))

  score_keys = set()
  t_f_all = time.time()
  base_info = {k: v for k, v in info.items() if k not in ('stock_codes', 'trade_dates')}
  row_slice = (row_start, row_end)
  all_worker_args = [
    (f_cls, base_info, npz_stocks, py_dates, str(scores_dir), row_slice)
    for f_cls in [*factor_classes, *filter_factor_classes]
  ]
  _factor_intersection = None
  factor_names = {f.__name__ for f in factor_classes}
  for wargs in all_worker_args:
    name, entry, raw_nan = _factor_worker(wargs)
    info[name] = entry
    if name in factor_names:
      score_keys.add(name)
      if _factor_intersection is None:
        _factor_intersection = ~raw_nan
      else:
        _factor_intersection = _factor_intersection & ~raw_nan
  if _factor_intersection is not None:
    intersection_path = scores_dir / '_factor_intersection.bin'
    _factor_intersection.astype(bool).tofile(str(intersection_path))
    info['_factor_intersection'] = (str(intersection_path), _factor_intersection.shape, str(_factor_intersection.dtype))
  testback_logger.info(f"{len(factor_classes)} 因子计算完成 ({time.time() - t_f_all:.1f}s)")

  # 共用 list_dates
  list_dates_full = _compute_list_dates(npz_stocks, data['open'], npz_dates)
  del data

  # === 构建 SharedMemory 供 worker 高速访问（替代 memmap 磁盘 I/O）===
  from multiprocessing.shared_memory import SharedMemory
  filter_mask_keys = {'_factor_intersection', *(f.__name__ for f in filter_factor_classes)}
  needed_shm = score_keys | filter_mask_keys | {'open', 'close', 'high', 'low', 'preClose', 'volume', 'amount', 'total_share', 'st_mask', 'issue_price', 'stock_codes', 'trade_dates'}
  shm_entries = []
  for name in needed_shm:
    if name not in info:
      continue
    filepath, shape, dtype_str = info[name]
    arr = np.memmap(filepath, dtype=np.dtype(dtype_str), mode='r', shape=shape)
    contig = np.ascontiguousarray(arr)
    shm = SharedMemory(create=True, size=contig.nbytes)
    shared_arr = np.ndarray(contig.shape, dtype=contig.dtype, buffer=shm.buf)
    shared_arr[:] = contig
    _SHM_BLOCKS.append((shm, shared_arr))
    shm_entries.append((name, shm.name, shape, dtype_str))
  shm_gb = sum(int(np.prod(e[2])) * np.dtype(e[3]).itemsize for e in shm_entries) / 1024**3
  testback_logger.info(f"SharedMemory 就绪: {len(shm_entries)} 数组 ({shm_gb:.1f} GB)")

  testback_logger.info(f"训练集就绪 ({time.time() - t0:.1f}s), {len(all_valid_stocks)} 只, {len(valid_dates)} 天")

  index_data = _load_all_index_data(valid_dates)

  import gc
  gc.collect()

  n_workers = 20
  ctx = get_context('spawn')
  ga_pool = ctx.Pool(processes=n_workers, initializer=_worker_initializer, initargs=(shm_entries,))
  testback_logger.info(f"多进程池已创建: {n_workers} workers")

  val_index_data = _load_all_index_data(val_valid_dates)
  testback_logger.info(f"验证集就绪: {val_start} - {val_end}, {len(val_valid_dates)} 天")

  test_index_data = _load_all_index_data(test_valid_dates)
  testback_logger.info(f"测试集就绪: {test_start} - {test_end}, {len(test_valid_dates)} 天")

  generation_results = []
  if resume_dir:
    ga_cache, last_gen = _rebuild_from_jsonl(resume_dir)
    gr_path = resume_dir / 'generation_results.pkl'
    if gr_path.exists():
      generation_results = pickle.loads(gr_path.read_bytes())
    ga_state = _rebuild_ga_state(ga_cache)
    next_configs = ga_optimizer([], state=ga_state, population_size=population_size,
                                hall_of_fame_size=population_size, profile_name=profile_name,
                                ga_cache=ga_cache, gen=last_gen)
    start_generation = last_gen + 1
    testback_logger.info(f"从 JSONL 恢复: {len(ga_cache)} 个唯一配置, 第 {start_generation} 代开始")
  elif args.warm_start:
    with open(args.warm_start, 'r', encoding='utf-8') as f:
      cache_data = json_mod.load(f)
    if isinstance(cache_data, list):
      cache_data = {json_mod.dumps(_config_key(v['individual_config']), ensure_ascii=False): v for v in cache_data}
    for v in cache_data.values():
      v.setdefault('calmar', v.get('fitness', v['sharpe']))
    sorted_results = sorted(cache_data.values(), key=lambda v: v['calmar'], reverse=True)
    next_configs = [v['individual_config'] for v in sorted_results[:2 * population_size]]
    testback_logger.info(f"热启动: 从 {args.warm_start} 加载 top {len(next_configs)} 个种子配置 (共 {len(cache_data)} 条缓存)")
  else:
    next_configs = generate_initial_configs(2 * population_size, profile_name=profile_name)

  try:
    for generation in range(start_generation, generations):
      generation_start_ts = time.time()
      n_configs = len(next_configs)
      # 分离缓存命中与待执行配置
      cached_results = []
      uncached_configs = []
      for cfg in next_configs:
        key = json_mod.dumps(_config_key(cfg), ensure_ascii=False)
        if key in ga_cache:
          cached_results.append(ga_cache[key])
        else:
          uncached_configs.append(cfg)

      worker_args = [
        (info, score_keys, valid_dates, date_indices, stock_indices,
         all_valid_stocks, config, index_data, list_dates_full)
        for config in uncached_configs
      ]

      results_list = list(cached_results)
      if worker_args:
        _eval_parallel(worker_args, results_list, ga_cache, testback_logger, pool=ga_pool)

      if not is_debug:
        # 训练集统计
        calmars = [r['calmar'] for r in results_list]
        best_idx = max(range(len(results_list)), key=lambda i: calmars[i])
        best = results_list[best_idx]
        best_cfg = best['individual_config']
        best_m = {'calmar': best['calmar'],
                  'sharpe': best['sharpe'],
                  'annualized': best['annualized'], 'max_drawdown': best['max_drawdown']}
        avg_calmar = sum(calmars) / len(calmars)
        avg_ann = sum(r['annualized'] for r in results_list) / len(results_list)
        avg_dd = sum(r['max_drawdown'] for r in results_list) / len(results_list)

        # 上证指数训练基线
        benchmark_vals = index_data['sh000001'].astype(float)
        benchmark_daily = np.diff(benchmark_vals) / benchmark_vals[:-1] * 100.0
        benchmark_daily = benchmark_daily[np.isfinite(benchmark_daily)]
        benchmark_m = compute_core_metrics(list(benchmark_daily))
        benchmark_calmar = abs(benchmark_m['annualized']) / abs(benchmark_m['max_drawdown']) if benchmark_m['max_drawdown'] != 0 else 0.0

        gen_time = time.time() - generation_start_ts
        sorted_w = sorted(best_cfg['weights'].items(), key=lambda x: -abs(x[1]))
        w_str = ', '.join(f'{k}={v:.2f}' for k, v in sorted_w)
        timing_str = _format_timing(best_cfg)

        # 验证集+测试集评估：第1代 + 每10代，仅评估训练最优个体
        report_gen = (generation == 0 or (generation + 1) % 10 == 0)
        if report_gen:
          # 验证集评估训练最优个体
          val_best_key = json_mod.dumps(_config_key(best_cfg), ensure_ascii=False)
          if val_best_key in val_eval_cache:
            train_best_val_m = dict(val_eval_cache[val_best_key])
          else:
            val_worker_args = [
              (info, score_keys, val_valid_dates, val_date_indices, stock_indices,
               all_valid_stocks, best_cfg, val_index_data, list_dates_full)
            ]
            val_res = []
            _eval_parallel(val_worker_args, val_res, {}, testback_logger, pool=ga_pool)
            train_best_val_m = {'calmar': val_res[0]['calmar'], 'sharpe': val_res[0]['sharpe'], 'annualized': val_res[0]['annualized'], 'max_drawdown': val_res[0]['max_drawdown']} if val_res else {'calmar': 0, 'sharpe': 0, 'annualized': 0, 'max_drawdown': 0}
            val_eval_cache[val_best_key] = dict(train_best_val_m)
            with open(val_cache_path, 'w', encoding='utf-8') as f:
              json_mod.dump(val_eval_cache, f, ensure_ascii=False)
          # 训练最优个体的 val metrics 写入 ga_cache
          ga_cache[val_best_key]['val_calmar'] = train_best_val_m['calmar']
          ga_cache[val_best_key]['val_sharpe'] = train_best_val_m['sharpe']
          ga_cache[val_best_key]['val_annualized'] = train_best_val_m['annualized']
          ga_cache[val_best_key]['val_max_drawdown'] = train_best_val_m['max_drawdown']

          # 测试集评估训练最优个体
          best_test_key = json_mod.dumps(_config_key(best_cfg), ensure_ascii=False)
          if best_test_key in test_eval_cache:
            _test_train_best_m = dict(test_eval_cache[best_test_key])
          else:
            test_worker_args = [
              (info, score_keys, test_valid_dates, test_date_indices, stock_indices,
               all_valid_stocks, best_cfg, test_index_data, list_dates_full)
            ]
            test_res = []
            _eval_parallel(test_worker_args, test_res, {}, testback_logger, pool=ga_pool)
            _test_train_best_m = {'calmar': test_res[0]['calmar'], 'sharpe': test_res[0]['sharpe'], 'annualized': test_res[0]['annualized'], 'max_drawdown': test_res[0]['max_drawdown']} if test_res else {'calmar': 0, 'sharpe': 0, 'annualized': 0, 'max_drawdown': 0}
            test_eval_cache[best_test_key] = dict(_test_train_best_m)
            with open(test_cache_path, 'w', encoding='utf-8') as f:
              json_mod.dump(test_eval_cache, f, ensure_ascii=False)

          # 实盘个体：全部历史中 (train_calmar+val_calmar)/2 最高者
          best_live_total = -999.0
          best_live_cfg = None
          best_live_train_calmar = 0.0
          best_live_val_calmar = 0.0
          for entry in ga_cache.values():
            vs = entry.get('val_calmar')
            if vs is not None:
              total = (entry['calmar'] + vs) / 2.0
              if total > best_live_total:
                best_live_total = total
                best_live_cfg = entry['individual_config']
                best_live_train_calmar = entry['calmar']
                best_live_val_calmar = vs
          live_test_calmar = 0.0
          if best_live_cfg is not None:
            live_test_key = json_mod.dumps(_config_key(best_live_cfg), ensure_ascii=False)
            if live_test_key in test_eval_cache:
              live_test_calmar = test_eval_cache[live_test_key]['calmar']
            else:
              live_test_args = [
                (info, score_keys, test_valid_dates, test_date_indices, stock_indices,
                 all_valid_stocks, best_live_cfg, test_index_data, list_dates_full)
              ]
              live_test_res = []
              _eval_parallel(live_test_args, live_test_res, {}, testback_logger, pool=ga_pool)
              if live_test_res:
                lt = live_test_res[0]
                live_test_calmar = lt['calmar']
                test_eval_cache[live_test_key] = {'calmar': lt['calmar'], 'sharpe': lt['sharpe'], 'annualized': lt['annualized'], 'max_drawdown': lt['max_drawdown']}
              with open(test_cache_path, 'w', encoding='utf-8') as f:
                json_mod.dump(test_eval_cache, f, ensure_ascii=False)

          sorted_w = sorted(best_cfg['weights'].items(), key=lambda x: -abs(x[1]))
          testback_logger.info(
            f"GA gen{generation + 1}: 训练Calmar={best_m['calmar']:.3f}(上证指数={benchmark_calmar:.3f})/{avg_calmar:.3f} S={best_m['sharpe']:.2f} | "
            f"验证Calmar={train_best_val_m['calmar']:.3f} | "
            f"测试Calmar={_test_train_best_m['calmar']:.3f} | "
            f"{_format_config_params(best_cfg, profile_name)} | "
            f"{', '.join(f'{k}={v:.1f}' for k, v in sorted_w)}")
          if best_live_cfg is not None:
            testback_logger.info(
              f"实盘: total={best_live_total:.3f}(train={best_live_train_calmar:.3f}/val={best_live_val_calmar:.3f}) test={live_test_calmar:.3f} | "
              f"{_format_config_params(best_live_cfg, profile_name)} | "
              f"{', '.join(f'{k}={v:.1f}' for k, v in sorted(best_live_cfg['weights'].items(), key=lambda x: -abs(x[1])))}")
            best_result = strategy_config_payload(profile_name, best_live_cfg)
            with open(output_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
              json_mod.dump(best_result, f, indent=2, ensure_ascii=False)


      if not results_list:
        raise RuntimeError(f"{'调试模式' if is_debug else '第 ' + str(generation + 1) + ' 代'}未获得任何有效回测结果")

      generation_time = time.time() - generation_start_ts
      for result in results_list:
        result['generation'] = generation
        result['fitness'] = result['calmar']


      best_in_gen = max(results_list, key=lambda x: x['fitness'])

      if not is_debug:
        fitnesses = [ind['fitness'] for ind in results_list]
        generation_stats = {
          'generation': generation, 'generation_time': generation_time,
          'population_size': len(results_list),
          'max_fitness': max(fitnesses), 'mean_fitness': sum(fitnesses) / len(fitnesses),
          'min_fitness': min(fitnesses),
        }
        generation_stats['best_weights'] = dict(best_in_gen['individual_config']['weights'])
        generation_stats['best_buy_n'] = best_in_gen['individual_config']['buy_n']
        generation_stats['best_stock_pool'] = best_in_gen['individual_config'].get('stock_pool')
        generation_stats['best_timing_enabled'] = best_in_gen['individual_config'].get('timing_enabled')
        generation_stats['best_timing_base'] = best_in_gen['individual_config'].get('timing_base')
        generation_stats['best_timing_leverage'] = best_in_gen['individual_config'].get('timing_leverage')
        if report_gen:
          generation_stats['val_best_calmar'] = float(train_best_val_m['calmar'])
          generation_stats['val_best_sharpe'] = float(train_best_val_m['sharpe'])
          generation_stats['val_best_annualized'] = float(train_best_val_m['annualized'])
          generation_stats['val_best_mdd'] = float(train_best_val_m['max_drawdown'])
          generation_stats['test_train_best_calmar'] = float(_test_train_best_m['calmar'])
          generation_stats['test_train_best_sharpe'] = float(_test_train_best_m['sharpe'])
          generation_stats['test_train_best_annualized'] = float(_test_train_best_m['annualized'])
          generation_stats['test_train_best_mdd'] = float(_test_train_best_m['max_drawdown'])
        generation_results.append(generation_stats)

      import json as _json
      best_key = _config_key(best_in_gen['individual_config'])
      with open(output_dir / 'all_results.jsonl', 'a', encoding='utf-8') as _f:
        for _r in results_list:
          entry = {
            'generation': _r['generation'],
            'calmar': _r['calmar'], 'sharpe': _r['sharpe'],
            'annualized': _r['annualized'],
            'max_drawdown': _r['max_drawdown'], 'total_return': _r['total_return'],
            'val_calmar': _r.get('val_calmar'), 'val_sharpe': _r.get('val_sharpe'), 'config': _r['individual_config'],
          }
          if _config_key(_r['individual_config']) == best_key and not is_debug:
            if report_gen:
              entry['val_calmar'] = train_best_val_m['calmar']
              entry['val_sharpe'] = train_best_val_m['sharpe']
              entry['val_annualized'] = train_best_val_m['annualized']
              entry['val_max_drawdown'] = train_best_val_m['max_drawdown']
              entry['test_calmar'] = _test_train_best_m['calmar']
              entry['test_sharpe'] = _test_train_best_m['sharpe']
              entry['test_annualized'] = _test_train_best_m['annualized']
              entry['test_max_drawdown'] = _test_train_best_m['max_drawdown']
          _f.write(_json.dumps(entry, ensure_ascii=False) + '\n')

      if is_debug:
        best_result = strategy_config_payload(profile_name, best_in_gen['individual_config'])
        with open(output_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
          json_mod.dump(best_result, f, indent=2, ensure_ascii=False)

      if not is_debug:
        with open(output_dir / 'generation_results.pkl', 'wb') as f:
          pickle.dump(generation_results, f)

      next_configs = ga_optimizer(results_list, state=ga_state, population_size=population_size,
                                  hall_of_fame_size=population_size, profile_name=profile_name,
                                  ga_cache=ga_cache, gen=generation)

    # 最终测试集评估必须在 Pool terminate 前完成（共用 ga_pool）
    if not is_debug:
      testback_logger.info(f"\n{'=' * 60}")
      testback_logger.info(f"最终测试集评估 ({test_start} - {test_end})")
      train_best_config = ga_state['hall_of_fame'][0]
      train_test_args = [(
        info, score_keys, test_valid_dates, test_date_indices, stock_indices,
        all_valid_stocks, train_best_config, test_index_data, list_dates_full
      )]
      train_test_results = []
      _eval_parallel(train_test_args, train_test_results, {}, testback_logger, pool=ga_pool)
      if train_test_results:
        tr = train_test_results[0]
        testback_logger.info(f"  [训练最优] 测试夏普={tr['sharpe']:.3f}, 年化={tr['annualized']:.1f}%, 回撤={tr['max_drawdown']:.1f}%")
      testback_logger.info(f"{'=' * 60}")

  finally:
    ga_pool.terminate()
    ga_pool.join()
    _cleanup_shm()
    _cleanup_memmap()

  if not is_debug:
    # 实盘个体：全部历史中 (train_calmar+val_calmar)/2 最高者
    best_live_total = -999.0
    best_live_cfg = None
    for entry in ga_cache.values():
      vs = entry.get('val_calmar')
      if vs is not None:
        total = (entry['calmar'] + vs) / 2.0
        if total > best_live_total:
          best_live_total = total
          best_live_cfg = entry['individual_config']
    if best_live_cfg is not None:
      best_result = strategy_config_payload(profile_name, best_live_cfg)
      with open(output_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
        json_mod.dump(best_result, f, indent=2, ensure_ascii=False)
      best_cfg = best_live_cfg
      w_str = ', '.join(f'{k}={v:.2f}' for k, v in best_cfg['weights'].items())
      timing_str = _format_timing(best_cfg)
      testback_logger.info(f"\n最优参数已保存:")
      testback_logger.info(f"  - {output_dir / 'best_individual_config.json'}")
      testback_logger.info(f"实盘最优 (train_calmar+val_calmar)/2={best_live_total:.3f}: [{w_str}], pool={_format_pool(best_cfg.get('stock_pool'))}, buy={best_cfg['buy_n']},sell={best_cfg['sell_m']}{timing_str}, rebal={'ON' if best_cfg.get('rebalance') else 'OFF'}")
    else:
      testback_logger.warning("无实盘最优个体（缺少验证集评估结果），回退到训练最优")
      best_config = ga_state['hall_of_fame'][0]
      best_result = strategy_config_payload(profile_name, best_config)
      with open(output_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
        json_mod.dump(best_result, f, indent=2, ensure_ascii=False)
      testback_logger.info(f"\n最优参数已保存（训练最优）: {output_dir / 'best_individual_config.json'}")

    calmars = [v['calmar'] for v in ga_cache.values()]
    testback_logger.info(f"\n{'=' * 60}")
    testback_logger.info("回测执行完成")
    testback_logger.info(f"  总回测次数: {sum(g.get('population_size', len(g.get('best_weights', {}))) for g in generation_results)}")
    testback_logger.info(f"  唯一配置数: {len(ga_cache)}")
    testback_logger.info(f"  平均Calmar: {sum(calmars) / len(calmars):.3f}")
    testback_logger.info(f"  最大Calmar: {max(calmars):.3f}")
    testback_logger.info(f"{'=' * 60}")

  testback_logger.info(f"\n{'调试' if is_debug else 'GA'}模式执行完成，结果目录: {output_dir}")
  testback_logger.remove_file_sink()
  return None


# ========== 主入口 ==========

def main():
  import argparse

  from loguru import logger as loguru_logger

  ts = datetime.now()

  parser = argparse.ArgumentParser(description='WBR GA 参数搜索')
  parser.add_argument('--mode', type=str, default='ga', choices=['debug', 'ga'])
  parser.add_argument('--output-dir', type=str, default=None)
  parser.add_argument('--warm-start', type=str, default=None, help='热启动种群 JSON 文件路径')
  parser.add_argument('--resume', type=str, nargs='?', const='auto', default=None,
                      help='从 JSONL 恢复: 不传则自动找最新, 或指定目录路径')
  parser.add_argument('--profile', type=str, default=None)
  parser.add_argument('--live-sim', action='store_true', default=True, help='启用实盘模拟 (默认开启)')
  args = parser.parse_args()

  profile_name = args.profile or DEFAULT_GA_PROFILE
  mode_configs = get_mode_configs(profile_name)
  mode_config = mode_configs[args.mode].copy()

  loguru_logger.remove()
  loguru_logger.add(sys.stderr, level=mode_config['log_level'])

  testback_logger.info(f"运行模式: {args.mode} - {mode_config['desc']}")

  start_date, end_date = get_profile_preload_range(profile_name)
  testback_logger.info(f"GA 预加载区间: {start_date.strftime('%Y%m%d')} - {end_date.strftime('%Y%m%d')}")

  backtest_datetime_list = [
    datetime.combine(d, datetime.min.time())
    for d in get_trading_date_span(start_date, end_date)
  ]

  factor_classes = get_profile_factor_classes(profile_name)
  factor_histories = {factor_cls.__name__: factor_cls().hist_days for factor_cls in factor_classes}
  max_hist_days = max(factor_histories.values(), default=0)
  hist_detail = ', '.join(f'{name}={days}天' for name, days in factor_histories.items())
  testback_logger.info(f"因子历史需求: {hist_detail}，最大需求={max_hist_days}天")

  # GA/debug: 从最新 runtime NPZ 取历史全集，避免当前可买池带来幸存者偏差
  filtered_stocks = load_runtime_stock_codes()
  testback_logger.info(f"股票池(runtime历史全集): {len(filtered_stocks)} 只")

  resume_dir = None
  if args.resume:
    if args.resume == 'auto':
      resume_dir = _find_latest_ga_dir()
      if resume_dir is None:
        testback_logger.error('--resume 未找到可恢复的 GA 目录（需含 all_results.jsonl）')
        sys.exit(1)
    else:
      resume_dir = Path(args.resume)
      if not resume_dir.is_dir() or not (resume_dir / 'all_results.jsonl').exists():
        testback_logger.error(f'--resume 指定目录无效或缺少 all_results.jsonl: {resume_dir}')
        sys.exit(1)
    testback_logger.info(f'从 JSONL 恢复: {resume_dir}')

  with keep_windows_awake() as keep_awake_enabled:
    if keep_awake_enabled:
      testback_logger.info('已启用 Windows 防休眠，任务结束后自动恢复')

    result = _run_ga(args, mode_config, backtest_datetime_list, filtered_stocks, profile_name=profile_name, resume_dir=resume_dir)

  te = datetime.now()
  testback_logger.info(f"总耗时: {(te - ts).total_seconds():.2f} 秒")
  return result


if __name__ == "__main__":
  main()
