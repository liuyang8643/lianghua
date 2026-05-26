import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


from core.scoring import scores_to_ranks
from core.runtime import load_runtime_npz
from utils.stock.time import get_trading_date_span
from utils.windows_awake import keep_windows_awake
from core.ga import (
    DEFAULT_GA_PROFILE,
    build_individual_config, repair_config,
    generate_initial_configs,
    get_mode_configs,
    get_profile,
    get_profile_factor_classes,
    get_profile_metadata,
    get_profile_preload_range,
    get_profile_search_spaces,
    get_profile_weight_search_spaces,
    resolve_profile_name,
    sample_factor_choice,
    sample_position_count,
    sample_holding_period,
    sample_stock_pool,
    sample_timing_base,
    sample_timing_leverage,
    sample_timing_direction,
    sample_timing_index,
    sample_timing_window,
)

from datetime import date, date as date_type, datetime
from testback.logger import testback_logger
from core.backtest import (
  _count_holding_trading_days, _get_stock_name_map, _calc_holding_stats,
  _parse_single_verify_config, _resolve_single_stock_pool,
  _extend_verify_stock_pool_with_historical_codes,
  _compute_factor_scores, _backtest_direct,
  _compute_list_dates, _compute_timing_multipliers, _compute_metrics_simple,
  _load_all_index_data,
  _format_pool, _format_timing, _try_send_feishu,
  _resolve_output_dir,
  run_single_mode,
)


# ========== GA 核心函数 ==========

def _config_key(config: dict) -> tuple:
  w = config['weights']
  sp = config.get('stock_pool')
  if isinstance(sp, list):
    sp = tuple(sp)
  return (config['buy_n'], config['sell_m'],
          sp, config.get('holding_period'),
          config.get('timing_base'), config.get('timing_leverage'),
          config.get('timing_direction'),
          config.get('timing_window'), config.get('timing_index'),
          tuple(sorted(w.items())))


def ga_optimizer(results, state, population_size=24, hall_of_fame_size=24, profile_name=DEFAULT_GA_PROFILE, ga_cache=None):
  import random

  results_list = list(results) if not isinstance(results, list) else results

  for r in results_list:
    key = _config_key(r['individual_config'])
    state['fitness_cache'][key] = r['sharpe']

  if not state['population']:
    results_list.sort(key=lambda r: r['sharpe'], reverse=True)
    state['population'] = [r['individual_config'] for r in results_list[:population_size]]

  def get_fitness(config):
    key = _config_key(config)
    return state.get('fitness_cache', {}).get(key)

  # 父代选择：从历史全局池挑选，打破当代近亲繁殖
  if ga_cache and len(ga_cache) >= population_size:
    unique_cfgs = {}
    for key, val in ga_cache.items():
      cfg = val.get('individual_config')
      fit = val.get('sharpe', -999)
      if cfg is not None and fit > -900:
        unique_cfgs[key] = (cfg, fit)
    sorted_global = sorted(unique_cfgs.values(), key=lambda x: x[1], reverse=True)
    total = len(sorted_global)
    n_elite = min(20, total)
    n_mid = population_size - n_elite
    elite_cfgs = [cfg for cfg, _ in sorted_global[:n_elite]]
    mid_start = 45
    mid_end = max(mid_start + 1, total // 2)
    if mid_start < total:
      mid_pool = sorted_global[mid_start:mid_end]
      mid_cfgs = [cfg for cfg, _ in random.sample(mid_pool, min(n_mid, len(mid_pool)))]
    else:
      mid_cfgs = []
    while len(mid_cfgs) < n_mid:
      remaining = [cfg for cfg, _ in sorted_global if cfg not in elite_cfgs and cfg not in mid_cfgs]
      if remaining:
        mid_cfgs.append(random.choice(remaining))
      else:
        break
    parents = elite_cfgs + mid_cfgs
  else:
    population_with_fitness = [(ind, fit) for ind in state['population']
                               if (fit := get_fitness(ind)) is not None]
    if not population_with_fitness:
      population_with_fitness = [(r['individual_config'], r['sharpe']) for r in results_list]
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
    position_count = _crossover_field(p1, p2, 'buy_n')
    fc = _crossover_field(p1, p2, 'factor_choice') if has_factor_choice else None
    stock_pool = _crossover_field(p1, p2, 'stock_pool') if 'stock_pool' in search_spaces else None
    holding_period = _crossover_field(p1, p2, 'holding_period') if 'holding_period' in search_spaces else None
    timing_base = _crossover_field(p1, p2, 'timing_base') if 'timing_base' in search_spaces else None
    timing_leverage = _crossover_field(p1, p2, 'timing_leverage') if 'timing_leverage' in search_spaces else None
    timing_dir = _crossover_field(p1, p2, 'timing_direction') if 'timing_direction' in search_spaces else None
    timing_window = _crossover_field(p1, p2, 'timing_window') if 'timing_window' in search_spaces else None
    timing_index = _crossover_field(p1, p2, 'timing_index') if 'timing_index' in search_spaces else None
    crossed_weights = None
    if has_weight_search:
      crossed_weights = {}
      all_keys = set(p1.get('weights', {})) | set(p2.get('weights', {}))
      for k in all_keys:
        w1 = p1['weights'].get(k, 0.0)
        w2 = p2['weights'].get(k, 0.0)
        crossed_weights[k] = w1 if random.random() < 0.5 else w2
    return build_individual_config(position_count, weights=crossed_weights,
                                   factor_choice=fc,
                                   stock_pool=stock_pool, holding_period=holding_period,
                                   timing_base=timing_base, timing_leverage=timing_leverage,
                                   timing_direction=timing_dir,
                                   timing_window=timing_window,
                                   timing_index=timing_index, profile_name=profile_name)

  weight_spaces = get_profile_weight_search_spaces(profile_name) if has_weight_search else {}

  def _collect_dims():
    dims = ['position_count']
    if has_factor_choice: dims.append('factor_choice')
    if 'stock_pool' in search_spaces: dims.append('stock_pool')
    if 'holding_period' in search_spaces: dims.append('holding_period')
    if 'timing_base' in search_spaces: dims.append('timing_base')
    if 'timing_leverage' in search_spaces: dims.append('timing_leverage')
    if 'timing_direction' in search_spaces: dims.append('timing_direction')
    if 'timing_window' in search_spaces: dims.append('timing_window')
    if 'timing_index' in search_spaces: dims.append('timing_index')
    for k in weight_spaces: dims.append(f'weight_{k}')
    return dims

  import math
  _all_dims = _collect_dims()

  def mutate_config(config):
    if not _all_dims:
      return build_individual_config(config['buy_n'],
                                      weights=config.get('weights'), profile_name=profile_name)
    n_mutate = max(1, min(math.ceil(random.uniform(0.25, 0.50) * len(_all_dims)), len(_all_dims)))
    mutate_dims = set(random.sample(_all_dims, n_mutate))
    position_count = sample_position_count(profile_name=profile_name) if 'position_count' in mutate_dims else config['buy_n']
    fc = sample_factor_choice(profile_name=profile_name) if 'factor_choice' in mutate_dims else config.get('factor_choice')
    stock_pool = sample_stock_pool(profile_name=profile_name) if 'stock_pool' in mutate_dims else config.get('stock_pool')
    holding_period = sample_holding_period(profile_name=profile_name) if 'holding_period' in mutate_dims else config.get('holding_period')
    timing_base = sample_timing_base(profile_name=profile_name) if 'timing_base' in mutate_dims else config.get('timing_base')
    timing_leverage = sample_timing_leverage(profile_name=profile_name) if 'timing_leverage' in mutate_dims else config.get('timing_leverage')
    timing_dir = sample_timing_direction(profile_name=profile_name) if 'timing_direction' in mutate_dims else config.get('timing_direction')
    timing_window = sample_timing_window(profile_name=profile_name) if 'timing_window' in mutate_dims else config.get('timing_window')
    timing_index = sample_timing_index(profile_name=profile_name) if 'timing_index' in mutate_dims else config.get('timing_index')
    mutated_weights = dict(config.get('weights', {}))
    for k in weight_spaces:
      if f'weight_{k}' in mutate_dims:
        mutated_weights[k] = random.choice(weight_spaces[k])
      elif k not in mutated_weights:
        mutated_weights[k] = 0.0
    return build_individual_config(position_count, weights=mutated_weights,
                                   factor_choice=fc,
                                   stock_pool=stock_pool, holding_period=holding_period,
                                   timing_base=timing_base, timing_leverage=timing_leverage,
                                   timing_direction=timing_dir,
                                   timing_window=timing_window,
                                   timing_index=timing_index, profile_name=profile_name)

  children = []
  while len(children) < population_size:
    if len(parents) == 1:
      child = mutate_config(parents[0])
    else:
      p1, p2 = random.sample(parents, 2)
      child = mutate_config(crossover_config(p1, p2))
    children.append(child)
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
        sharpe = r['sharpe']
        if key not in ga_cache or sharpe > ga_cache[key].get('sharpe', -999):
          ga_cache[key] = {'individual_config': cfg, 'sharpe': sharpe}
        if r['generation'] > last_gen:
          last_gen = r['generation']
  return ga_cache, last_gen


def _rebuild_ga_state(ga_cache: dict) -> dict:
  """从 ga_cache 重建 ga_state（population + hall_of_fame + fitness_cache）。"""
  sorted_entries = sorted(ga_cache.values(), key=lambda v: v.get('sharpe', -999), reverse=True)
  hall_of_fame = [v['individual_config'] for v in sorted_entries[:100]]
  fitness_cache = {_config_key(v['individual_config']): v['sharpe'] for v in sorted_entries}
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
    except Exception:
      pass
  _SHM_BLOCKS.clear()


def _factor_worker(args):
  """Compute single factor → rank → memmap. Module-level for spawn pickle."""
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
  raw = f.calc_batch(data)
  scores = scores_to_ranks(raw.astype(np.float32, copy=False))
  if row_slice is not None:
    full = np.full((n_full, scores.shape[1]), np.nan, dtype=scores.dtype)
    full[r0:r1] = scores
    scores = full
  filepath = Path(tmpdir) / f'factor_{name}.bin'
  scores.tofile(str(filepath))
  return name, (str(filepath), scores.shape, str(scores.dtype))


def _worker_evaluate(args):
  global _worker_shm_cache
  train_info, score_keys, valid_dates, date_indices, stock_indices, \
      all_stocks_list, config, index_data, list_dates_map = args

  needed = score_keys | {'open', 'close', 'high', 'low', 'st_mask', 'issue_price', 'stock_codes', 'trade_dates'}
  all_arrays = {k: arr for k, (shm, arr) in _worker_shm_cache.items() if k in needed}
  data = {k: v for k, v in all_arrays.items() if k not in score_keys}
  all_scores = {k: v for k, v in all_arrays.items() if k in score_keys}

  stock_pool = config.get('stock_pool') or ('60', '00', '30', '688')
  pool_stocks = [s for s in all_stocks_list if s.startswith(stock_pool)]

  timing_multipliers = _compute_timing_multipliers(config, valid_dates, index_data)

  r = _backtest_direct(
    data, all_scores, valid_dates, date_indices, pool_stocks, stock_indices,
    weights=config['weights'], buy_n=config['buy_n'], sell_m=config['sell_m'],
    temperatures=config['temperatures'],
    holding_period=config.get('holding_period'),
    position_multipliers=timing_multipliers, list_dates_map=list_dates_map,
    lightweight=True)

  metrics = _compute_metrics_simple(r['daily_returns'])
  return {
    'individual_config': config,
    'total_return': r['total_return'],
    'sharpe': metrics['sharpe'],
    'annualized': metrics['annualized'],
    'max_drawdown': metrics['max_drawdown'],
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
      d = dt.date() if hasattr(dt, 'date') else dt
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
  test_end = date(2026, 5, 22)
  test_datetime_list = [datetime.combine(d, datetime.min.time()) for d in get_trading_date_span(test_start, test_end)]
  test_valid_dates, test_date_indices = _build_date_subset(test_datetime_list, npz_dates, date_to_idx)

  max_hist = max(f.hist_days for f in factor_classes)
  all_idx = date_indices + val_date_indices + test_date_indices
  row_start = max(0, min(all_idx) - max_hist)
  row_end = max(all_idx) + 1
  n_needed = row_end - row_start
  testback_logger.info(f"因子计算范围: [{row_start}:{row_end}] = {n_needed} 天 (全量 {len(npz_dates)} 天, 截断 {len(npz_dates)-n_needed} 天)")

  score_keys = set()
  t_f_all = time.time()
  base_info = {k: v for k, v in info.items() if k not in ('stock_codes', 'trade_dates')}
  row_slice = (row_start, row_end)
  all_worker_args = [
    (f_cls, base_info, npz_stocks, py_dates, str(scores_dir), row_slice)
    for f_cls in factor_classes
  ]
  for wargs in all_worker_args:
    name, entry = _factor_worker(wargs)
    info[name] = entry
    score_keys.add(name)
  testback_logger.info(f"{len(factor_classes)} 因子计算完成 ({time.time() - t_f_all:.1f}s)")

  # 共用 list_dates
  list_dates_full = _compute_list_dates(npz_stocks, data['open'], npz_dates)
  del data

  # === 构建 SharedMemory 供 worker 高速访问（替代 memmap 磁盘 I/O）===
  from multiprocessing.shared_memory import SharedMemory
  needed_shm = score_keys | {'open', 'close', 'high', 'low', 'st_mask', 'issue_price', 'stock_codes', 'trade_dates'}
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

  _train_info = info
  _score_keys = score_keys
  train_list_dates = list_dates_full
  testback_logger.info(f"训练集就绪 ({time.time() - t0:.1f}s), {len(all_valid_stocks)} 只, {len(valid_dates)} 天")

  index_data = _load_all_index_data(valid_dates)

  import gc, ctypes
  gc.collect()

  n_workers = 28
  ctx = get_context('spawn')
  ga_pool = ctx.Pool(processes=n_workers, initializer=_worker_initializer, initargs=(shm_entries,))
  testback_logger.info(f"多进程池已创建: {n_workers} workers")

  _val_info = info
  _score_keys_val = score_keys
  val_stock_indices = stock_indices
  val_valid_stocks = all_valid_stocks
  val_list_dates = list_dates_full
  val_index_data = _load_all_index_data(val_valid_dates)
  testback_logger.info(f"验证集就绪: {val_start} - {val_end}, {len(val_valid_dates)} 天")

  _test_info = info
  _score_keys_test = score_keys
  test_stock_indices = stock_indices
  test_valid_stocks = all_valid_stocks
  test_list_dates = list_dates_full
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
                                ga_cache=ga_cache)
    start_generation = last_gen + 1
    testback_logger.info(f"从 JSONL 恢复: {len(ga_cache)} 个唯一配置, 第 {start_generation} 代开始")
  elif args.warm_start:
    with open(args.warm_start, 'r', encoding='utf-8') as f:
      cache_data = json_mod.load(f)
    if isinstance(cache_data, list):
      cache_data = {json_mod.dumps(_config_key(v['individual_config']), ensure_ascii=False): v for v in cache_data}
    for v in cache_data.values():
      v.setdefault('sharpe', v.get('fitness', -999))
    sorted_results = sorted(cache_data.values(), key=lambda v: v.get('sharpe', -999), reverse=True)
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
        (_train_info, _score_keys, valid_dates, date_indices, stock_indices,
         all_valid_stocks, config, index_data, train_list_dates)
        for config in uncached_configs
      ]

      results_list = list(cached_results)
      if worker_args:
        _eval_parallel(worker_args, results_list, ga_cache, testback_logger, pool=ga_pool)

      if not is_debug:
        # 训练集统计
        sharpes = [r['sharpe'] for r in results_list]
        best_idx = max(range(len(results_list)), key=lambda i: sharpes[i])
        best = results_list[best_idx]
        best_cfg = best['individual_config']
        best_m = {'sharpe': best['sharpe'],
                  'annualized': best['annualized'], 'max_drawdown': best['max_drawdown']}
        avg_sharpe = sum(sharpes) / len(sharpes)
        avg_ann = sum(r['annualized'] for r in results_list) / len(results_list)
        avg_dd = sum(r['max_drawdown'] for r in results_list) / len(results_list)

        # HS300 训练基线
        hs300_vals = index_data['sh000300'].astype(float)
        hs300_daily = np.diff(hs300_vals) / hs300_vals[:-1] * 100.0
        hs300_daily = hs300_daily[np.isfinite(hs300_daily)]
        hs300_m = _compute_metrics_simple(list(hs300_daily))

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
              (_val_info, _score_keys, val_valid_dates, val_date_indices, val_stock_indices,
               val_valid_stocks, best_cfg, val_index_data, val_list_dates)
            ]
            val_res = []
            _eval_parallel(val_worker_args, val_res, {}, testback_logger, pool=ga_pool)
            train_best_val_m = {'sharpe': val_res[0]['sharpe'], 'annualized': val_res[0]['annualized'], 'max_drawdown': val_res[0]['max_drawdown']} if val_res else {'sharpe': 0, 'annualized': 0, 'max_drawdown': 0}
            val_eval_cache[val_best_key] = dict(train_best_val_m)
            with open(val_cache_path, 'w', encoding='utf-8') as f:
              json_mod.dump(val_eval_cache, f, ensure_ascii=False)
          # 训练最优个体的 val metrics 写入 ga_cache
          ga_cache[val_best_key]['val_sharpe'] = train_best_val_m['sharpe']
          ga_cache[val_best_key]['val_annualized'] = train_best_val_m['annualized']
          ga_cache[val_best_key]['val_max_drawdown'] = train_best_val_m['max_drawdown']

          # 测试集评估训练最优个体
          best_test_key = json_mod.dumps(_config_key(best_cfg), ensure_ascii=False)
          if best_test_key in test_eval_cache:
            _test_train_best_m = dict(test_eval_cache[best_test_key])
          else:
            test_worker_args = [
              (_test_info, _score_keys_test, test_valid_dates, test_date_indices, test_stock_indices,
               test_valid_stocks, best_cfg, test_index_data, test_list_dates)
            ]
            test_res = []
            _eval_parallel(test_worker_args, test_res, {}, testback_logger, pool=ga_pool)
            _test_train_best_m = {'sharpe': test_res[0]['sharpe'], 'annualized': test_res[0]['annualized'], 'max_drawdown': test_res[0]['max_drawdown']} if test_res else {'sharpe': 0, 'annualized': 0, 'max_drawdown': 0}
            test_eval_cache[best_test_key] = dict(_test_train_best_m)
            with open(test_cache_path, 'w', encoding='utf-8') as f:
              json_mod.dump(test_eval_cache, f, ensure_ascii=False)

          # 实盘个体：全部历史中 (train+val)/2 最高者
          best_live_total = -999.0
          best_live_cfg = None
          best_live_train_sharpe = 0.0
          best_live_val_sharpe = 0.0
          for entry in ga_cache.values():
            vs = entry.get('val_sharpe')
            if vs is not None:
              total = (entry['sharpe'] + vs) / 2.0
              if total > best_live_total:
                best_live_total = total
                best_live_cfg = entry['individual_config']
                best_live_train_sharpe = entry['sharpe']
                best_live_val_sharpe = vs
          live_test_sharpe = 0.0
          if best_live_cfg is not None:
            live_test_key = json_mod.dumps(_config_key(best_live_cfg), ensure_ascii=False)
            if live_test_key in test_eval_cache:
              live_test_sharpe = test_eval_cache[live_test_key]['sharpe']
            else:
              live_test_args = [
                (_test_info, _score_keys_test, test_valid_dates, test_date_indices, test_stock_indices,
                 test_valid_stocks, best_live_cfg, test_index_data, test_list_dates)
              ]
              live_test_res = []
              _eval_parallel(live_test_args, live_test_res, {}, testback_logger, pool=ga_pool)
              live_test_sharpe = live_test_res[0]['sharpe'] if live_test_res else 0.0
              test_eval_cache[live_test_key] = {'sharpe': live_test_sharpe, 'annualized': live_test_res[0].get('annualized', 0) if live_test_res else 0, 'max_drawdown': live_test_res[0].get('max_drawdown', 0) if live_test_res else 0}
              with open(test_cache_path, 'w', encoding='utf-8') as f:
                json_mod.dump(test_eval_cache, f, ensure_ascii=False)

          testback_logger.info(
            f"GA gen{generation + 1}: 训练Sharpe={best_m['sharpe']:.3f}/{avg_sharpe:.3f} | "
            f"验证Sharpe={train_best_val_m['sharpe']:.3f} | "
            f"测试Sharpe={_test_train_best_m['sharpe']:.3f} | "
            f"{_format_pool(best_cfg.get('stock_pool'))}, hp={best_cfg.get('holding_period')}, pos={best_cfg['buy_n']}{_format_timing(best_cfg)}, rebal={'ON' if best_cfg.get('rebalance') else 'OFF'} | "
            f"{', '.join(f'{k}={v:.1f}' for k, v in sorted_w)}")
          if best_live_cfg is not None:
            live_sorted_w = sorted(best_live_cfg['weights'].items(), key=lambda x: -abs(x[1]))
            testback_logger.info(
              f"实盘: total={best_live_total:.3f}(train={best_live_train_sharpe:.3f}/val={best_live_val_sharpe:.3f}) test={live_test_sharpe:.3f} | "
              f"{_format_pool(best_live_cfg.get('stock_pool'))}, hp={best_live_cfg.get('holding_period')}, pos={best_live_cfg['buy_n']}{_format_timing(best_live_cfg)}, rebal={'ON' if best_live_cfg.get('rebalance') else 'OFF'} | "
              f"{', '.join(f'{k}={v:.1f}' for k, v in live_sorted_w)}")
      if not results_list:
        raise RuntimeError(f"{'调试模式' if is_debug else '第 ' + str(generation + 1) + ' 代'}未获得任何有效回测结果")

      generation_time = time.time() - generation_start_ts
      for result in results_list:
        result['generation'] = generation
        result['fitness'] = result['sharpe']


      if not is_debug:
        fitnesses = [ind['fitness'] for ind in results_list]
        generation_stats = {
          'generation': generation, 'generation_time': generation_time,
          'population_size': len(results_list),
          'max_fitness': max(fitnesses), 'mean_fitness': sum(fitnesses) / len(fitnesses),
          'min_fitness': min(fitnesses),
        }
        best_in_gen = max(results_list, key=lambda x: x['fitness'])
        generation_stats['best_weights'] = dict(best_in_gen['individual_config']['weights'])
        generation_stats['best_buy_n'] = best_in_gen['individual_config']['buy_n']
        generation_stats['best_stock_pool'] = best_in_gen['individual_config'].get('stock_pool')
        generation_stats['best_timing_enabled'] = best_in_gen['individual_config'].get('timing_enabled')
        generation_stats['best_timing_base'] = best_in_gen['individual_config'].get('timing_base')
        generation_stats['best_timing_leverage'] = best_in_gen['individual_config'].get('timing_leverage')
        if report_gen:
          generation_stats['val_best_sharpe'] = float(train_best_val_m['sharpe'])
          generation_stats['val_best_annualized'] = float(train_best_val_m['annualized'])
          generation_stats['val_best_mdd'] = float(train_best_val_m['max_drawdown'])
          generation_stats['test_train_best_sharpe'] = float(_test_train_best_m['sharpe'])
          generation_stats['test_train_best_annualized'] = float(_test_train_best_m['annualized'])
          generation_stats['test_train_best_mdd'] = float(_test_train_best_m['max_drawdown'])
        generation_results.append(generation_stats)

        import json as _json
        best_key = _config_key(best_in_gen['individual_config'])
        best_val = train_best_val_m if report_gen else None
        best_test = _test_train_best_m if report_gen else None
        with open(output_dir / 'all_results.jsonl', 'a', encoding='utf-8') as _f:
          for _r in results_list:
            entry = {
              'generation': _r['generation'],
              'sharpe': _r['sharpe'], 'annualized': _r['annualized'],
              'max_drawdown': _r['max_drawdown'], 'total_return': _r['total_return'],
              'val_sharpe': _r.get('val_sharpe'), 'config': _r['individual_config'],
            }
            if _config_key(_r['individual_config']) == best_key:
              if best_val:
                entry['val_sharpe'] = best_val['sharpe']
                entry['val_annualized'] = best_val['annualized']
                entry['val_max_drawdown'] = best_val['max_drawdown']
              if best_test:
                entry['test_sharpe'] = best_test['sharpe']
                entry['test_annualized'] = best_test['annualized']
                entry['test_max_drawdown'] = best_test['max_drawdown']
            _f.write(_json.dumps(entry, ensure_ascii=False) + '\n')

        with open(output_dir / 'generation_results.pkl', 'wb') as f:
          pickle.dump(generation_results, f)

        profile_metadata = get_profile_metadata(profile_name)
        best_result = {
          **profile_metadata,
          'individual_config': best_in_gen['individual_config'],
          'fitness': best_in_gen['fitness'], 'generation': generation + 1,
          'generation_time': generation_time, 'population_size': len(results_list),
        }
        with open(output_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
          json_mod.dump(best_result, f, indent=2, ensure_ascii=False)

      next_configs = ga_optimizer(results_list, state=ga_state, population_size=population_size,
                                  hall_of_fame_size=population_size, profile_name=profile_name,
                                  ga_cache=ga_cache)

  finally:
    ga_pool.terminate()
    ga_pool.join()
    _cleanup_shm()
    _cleanup_memmap()

  if not is_debug:
    best_config = ga_state['hall_of_fame'][0]
    key = _config_key(best_config)
    best_fitness = ga_state['fitness_cache'].get(key, 0.0)

    profile_metadata = get_profile_metadata(profile_name)
    best_result = {
      **profile_metadata,
      'individual_config': best_config, 'fitness': best_fitness,
      'generation': generations, 'population_size': population_size,
    }
    with open(output_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
      json_mod.dump(best_result, f, indent=2, ensure_ascii=False)

    best_cfg = best_config
    w_str = ', '.join(f'{k}={v:.2f}' for k, v in best_cfg['weights'].items())
    timing_str = _format_timing(best_cfg)
    testback_logger.info(f"\n最优参数已保存:")
    testback_logger.info(f"  - {output_dir / 'best_individual_config.json'}")
    testback_logger.info(f"最优夏普率: {best_fitness:.3f}, 参数: [{w_str}], pool={_format_pool(best_cfg.get('stock_pool'))}, pos={best_cfg['buy_n']}{timing_str}, rebal={'ON' if best_cfg.get('rebalance') else 'OFF'}")
    testback_logger.info(f"最优buy_n: {best_config['buy_n']}, sell_m: {best_config['sell_m']}")

    sharpes = [v['sharpe'] for v in ga_cache.values() if v.get('sharpe') is not None]
    testback_logger.info(f"\n{'=' * 60}")
    testback_logger.info("回测执行完成")
    testback_logger.info(f"  总回测次数: {sum(g.get('population_size', len(g.get('best_weights', {}))) for g in generation_results)}")
    testback_logger.info(f"  唯一配置数: {len(ga_cache)}")
    testback_logger.info(f"  平均夏普率: {sum(sharpes) / len(sharpes):.3f}")
    testback_logger.info(f"  最大夏普率: {max(sharpes):.3f}")
    testback_logger.info(f"{'=' * 60}")

  # 最终测试集评估（2022-2026，不参与优化）
  if not is_debug:
    testback_logger.info(f"\n{'=' * 60}")
    testback_logger.info("最终测试集评估 (2022-2026)")

    # 评估训练最优个体
    train_best_config = ga_state['hall_of_fame'][0]
    train_test_args = [(
      _test_info, _score_keys, test_valid_dates, test_date_indices, test_stock_indices,
      test_valid_stocks, train_best_config, test_index_data, test_list_dates
    )]
    train_test_results = []
    _eval_parallel(train_test_args, train_test_results, {}, testback_logger, pool=ga_pool)
    if train_test_results:
      tr = train_test_results[0]
      testback_logger.info(f"  [训练最优] 测试夏普={tr['sharpe']:.3f}, 年化={tr['annualized']:.1f}%, 回撤={tr['max_drawdown']:.1f}%")


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

  # GA/debug: 从 npz 取股票列表
  npz_dir = Path(__file__).resolve().parent.parent / 'data' / 'runtime'
  npz_files = sorted(npz_dir.glob('runtime_*.npz'))
  if not npz_files:
    raise FileNotFoundError(f"未找到 runtime npz 文件: {npz_dir}")
  filtered_stocks = [str(s) for s in np.load(npz_files[0], allow_pickle=False)['stock_codes']]
  testback_logger.info(f"股票池 (npz): {len(filtered_stocks)} 只")

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
