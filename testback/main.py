import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from data.db.delist import get_delist_stock_info
from core.strategies.runtime import load_runtime_npz
from utils.stock.time import get_trading_date_span
from utils.stock.legality import LegalityValidator

from utils.windows_awake import keep_windows_awake
from testback.account import StockAccountMocker
from testback.ga_config import (
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
  sample_timing_enabled,
  sample_timing_index,
  sample_timing_window,
)
from testback.metrics import compute_hs300_cumulative_returns, compute_strategy_metrics, compute_per_year_metrics

from datetime import date, date as date_type, datetime
from testback.logger import testback_logger


# ========== GA 核心函数 ==========

_BT_KEYS = ['open','high','low','close','volume','amount','st_mask','stock_codes','trade_dates','issue_price','stock_names']

def _config_key(config: dict) -> tuple:
  w = config['weights']
  sp = config.get('stock_pool')
  if isinstance(sp, list):
    sp = tuple(sp)
  return (config['buy_n'], config['sell_m'],
          sp, config.get('holding_period'),
          config.get('timing_base'), config.get('timing_leverage'),
          config.get('timing_direction'), config.get('timing_enabled'),
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
    n_elite = int(population_size * 0.9)
    n_mid = population_size - n_elite
    n_elite = min(n_elite, total)
    elite_cfgs = [cfg for cfg, _ in sorted_global[:n_elite]]
    mid_start = n_elite
    mid_end = max(mid_start + 1, total // 2)
    mid_pool = sorted_global[mid_start:mid_end]
    mid_cfgs = [cfg for cfg, _ in random.sample(mid_pool, min(n_mid, len(mid_pool)))]
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
    timing_enabled = _crossover_field(p1, p2, 'timing_enabled') if 'timing_enabled' in search_spaces else None
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
                                   timing_enabled=timing_enabled, timing_window=timing_window,
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
    if 'timing_enabled' in search_spaces: dims.append('timing_enabled')
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
    timing_enabled = sample_timing_enabled(profile_name=profile_name) if 'timing_enabled' in mutate_dims else config.get('timing_enabled')
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
                                   timing_enabled=timing_enabled, timing_window=timing_window,
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

def _count_holding_trading_days(start_date: date_type, end_date: date_type) -> int:
  if start_date is None or end_date is None:
    return 0
  if end_date < start_date:
    return 0
  return len(get_trading_date_span(start_date, end_date))


def _get_stock_name_map(traded_codes: set[str], stock_names=None, stock_codes=None) -> Dict[str, str]:
  """优先从 npz stock_names 取名称，无数据则返回空字典。"""
  if stock_names is None or stock_codes is None:
    return {}
  name_map: Dict[str, str] = {}
  for i, code in enumerate(stock_codes):
    if code in traded_codes:
      name_map[str(code)] = str(stock_names[i]) if stock_names[i] else ''
  return name_map


def _calc_holding_stats(current_positions: List[Dict], cleared_positions: List[Dict]) -> Dict:
  current_days = [p.get('holding_days', 0) for p in current_positions]
  cleared_days = [p.get('holding_days', 0) for p in cleared_positions]
  all_days = current_days + cleared_days

  def _avg(values: List[int]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0

  return {
    'average_holding_days': _avg(all_days),
    'average_current_holding_days': _avg(current_days),
    'average_cleared_holding_days': _avg(cleared_days),
    'max_holding_days': max(all_days) if all_days else 0,
    'min_holding_days': min(all_days) if all_days else 0,
    'current_positions_count': len(current_positions),
    'cleared_positions_count': len(cleared_positions),
  }


def _parse_single_verify_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
  verify_config = config_data.get('verify_delist_case') or {}
  if not verify_config:
    return {}

  force_stock_code = (verify_config.get('force_stock_code') or '').strip()
  if not force_stock_code:
    raise ValueError('verify_delist_case.force_stock_code 不能为空')

  candidate_stock_codes = verify_config.get('candidate_stock_codes') or []
  normalized_candidates = []
  seen = set()
  for code in [force_stock_code, *candidate_stock_codes]:
    normalized = (code or '').strip()
    if not normalized or normalized in seen:
      continue
    seen.add(normalized)
    normalized_candidates.append(normalized)

  sample_pool = verify_config.get('sample_pool')
  if sample_pool is not None and not isinstance(sample_pool, list):
    raise ValueError('verify_delist_case.sample_pool 必须是股票代码列表')

  normalized_sample_pool = []
  if sample_pool:
    seen_sample = set()
    for code in [*sample_pool, force_stock_code]:
      normalized = (code or '').strip()
      if not normalized or normalized in seen_sample:
        continue
      seen_sample.add(normalized)
      normalized_sample_pool.append(normalized)

  return {
    'enabled': True,
    'label': verify_config.get('label') or '退市归零验证',
    'force_stock_code': force_stock_code,
    'candidate_stock_codes': normalized_candidates,
    'sample_pool': normalized_sample_pool,
    'notes': verify_config.get('notes') or '',
  }


def _resolve_single_stock_pool(all_stocks: List[str], verify_config: Dict[str, Any]) -> List[str]:
  sample_pool = verify_config.get('sample_pool') or []
  if not sample_pool:
    return all_stocks

  available = set(all_stocks)
  resolved = [code for code in sample_pool if code in available]
  if verify_config['force_stock_code'] not in resolved and verify_config['force_stock_code'] in available:
    resolved.append(verify_config['force_stock_code'])

  if not resolved:
    raise ValueError('verify_delist_case.sample_pool 中没有可用股票')

  return resolved


def _extend_verify_stock_pool_with_historical_codes(
    all_stocks: List[str],
    backtest_datetime_list: List[datetime],
    verify_config: Dict[str, Any],
) -> List[str]:
  if not verify_config:
    return all_stocks

  requested_codes = []
  seen = set()
  for code in [
    verify_config.get('force_stock_code'),
    *(verify_config.get('candidate_stock_codes') or []),
    *(verify_config.get('sample_pool') or []),
  ]:
    normalized = (code or '').strip()
    if not normalized or normalized in seen:
      continue
    seen.add(normalized)
    requested_codes.append(normalized)

  missing_codes = [code for code in requested_codes if code not in all_stocks]
  if not missing_codes or not backtest_datetime_list:
    return all_stocks

  from data.db import get_all_stock_code_list

  historical_date = backtest_datetime_list[0].date()
  historical_stocks = get_all_stock_code_list(historical_date)
  historical_stocks = set(historical_stocks)
  recovered_codes = [code for code in missing_codes if code in historical_stocks]
  if not recovered_codes:
    return all_stocks

  testback_logger.info(
    f"single 退市验证补充历史股票池: {', '.join(recovered_codes)} @ {historical_date}"
  )
  return list(dict.fromkeys([*all_stocks, *recovered_codes]))


# ========== 直接回测路径（无TopN对象） ==========

def _scores_to_ranks(scores: np.ndarray) -> np.ndarray:
  """将原始得分转为每日截面排名 (0~1, 1=最优), NaN 给 0。原地修改。"""
  n_days = scores.shape[0]
  ranks = np.empty_like(scores, dtype=np.float32)
  for d in range(n_days):
    row = scores[d]
    nans = np.isnan(row)
    valid_mask = ~nans
    n_valid = valid_mask.sum()
    if n_valid == 0:
      ranks[d] = 0.0
      continue
    order = np.argsort(row[valid_mask])[::-1]
    ranks[d, nans] = 0.0
    col_idx = np.where(valid_mask)[0]
    ranks[d, col_idx[order]] = 1.0 - np.arange(n_valid, dtype=np.float32) / n_valid
  return ranks


def _compute_factor_scores(backtest_datetime_list, all_stocks, weights, factor_classes):
  """加载 NPZ 并批量计算因子分数，返回 (data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices)。"""
  data = load_runtime_npz(backtest_datetime_list)
  if data is None:
    first_d = backtest_datetime_list[0].strftime('%Y%m%d')
    last_d = backtest_datetime_list[-1].strftime('%Y%m%d')
    raise FileNotFoundError(f"未找到覆盖 {first_d}~{last_d} 的 runtime npz 文件")

  npz_stocks = [str(s) for s in data['stock_codes']]
  stock_indices = {c: i for i, c in enumerate(npz_stocks)}
  valid_stocks = [s for s in all_stocks if s in stock_indices]

  npz_dates = data['trade_dates']
  date_to_idx = {}
  for i, d in enumerate(npz_dates):
    date_to_idx[d.astype('datetime64[D]').item()] = i

  date_indices = []
  valid_dates = []
  for dt in backtest_datetime_list:
    d = dt.date() if hasattr(dt, 'date') else dt
    di = date_to_idx.get(d)
    if di is None:
      continue
    date_indices.append(di)
    valid_dates.append(dt)

  if not valid_dates:
    testback_logger.warning("没有交易日落在 runtime npz 日期范围内")
    return None

  import time
  t0 = time.time()

  factor_meta = []
  for f_cls in factor_classes:
    f = f_cls()
    name = f.__class__.__name__
    if weights is not None and weights.get(name, 0.0) == 0:
      continue
    factor_meta.append((name, f))

  py_dates = [d.astype('datetime64[D]').item() for d in npz_dates]
  factor_data = {**data, 'stock_codes': npz_stocks, 'trade_dates': py_dates}
  all_scores: dict[str, np.ndarray] = {}
  for name, f in factor_meta:
    raw = f.calc_batch(factor_data)
    all_scores[name] = _scores_to_ranks(raw.astype(np.float32, copy=False))

  testback_logger.info(f"因子批量+预排名完成 ({time.time() - t0:.1f}s), {len(valid_dates)} 个调仓日")
  return data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices


# ═══════════ numpy 批量合法性检查 ═══════════

_EPS = 0.001

def _round_half_up_np(values):
  return np.floor(values * 100.0 + 0.5 + 1e-9) / 100.0

def _precompute_limit_helpers(data, stock_indices, list_dates_map):
  codes = [str(s) for s in data['stock_codes']]
  n = len(codes)
  bt = np.zeros(n, dtype=np.int8); br = np.full(n, 0.10, dtype=np.float64)
  for i, c in enumerate(codes):
    if c.startswith('300') or c.startswith('301'): bt[i] = 1; br[i] = 0.20
    elif c.startswith('688'): bt[i] = 2; br[i] = 0.20
    elif c.startswith('83') or c.startswith('87') or c.startswith('43') or c.startswith('92'): bt[i] = 3; br[i] = 0.30
  tdp = [d.astype('datetime64[D]').item() for d in data['trade_dates']]
  d2t = {d: i for i, d in enumerate(tdp)}
  lt = np.full(n, -1, dtype=np.int32)
  if list_dates_map:
    for code, ld in list_dates_map.items():
      si = stock_indices.get(code)
      if si is None: continue
      ldi = d2t.get(ld)
      if ldi is None:
        for d in tdp:
          if d >= ld: ldi = d2t[d]; break
      if ldi is not None: lt[si] = ldi
  return bt, br, lt

def _batch_limit_check(candidates, candidates_idx, trade_idx, signal_date,
                       board_type, base_ratio, list_tidx,
                       open_all, close_all, high_all, low_all, st_all, issue_price_all, is_buy):
  if len(candidates) == 0:
    return np.array([], dtype=bool), {}
  idx = np.asarray(candidates_idx, dtype=np.intp); n = len(idx)
  opens = open_all[trade_idx, idx].astype(np.float64)
  valid_open = ~np.isnan(opens) & (opens > 0)
  if not np.any(valid_open):
    return np.zeros(n, dtype=bool), {'suspended': n}

  precloses = close_all[trade_idx - 1, idx].astype(np.float64) if trade_idx > 0 else np.full(n, np.nan)
  valid_preclose = (trade_idx > 0) & ~np.isnan(precloses) & (precloses > 0)

  st_arr = st_all[trade_idx, idx] if st_all is not None else np.zeros(n, dtype=bool)
  ratios = np.where(st_arr, 0.05, base_ratio[idx])
  boards = board_type[idx]
  cyb_pre = (boards == 1) & (signal_date < date(2020, 8, 24))
  ratios[cyb_pre] = 0.10
  lti = list_tidx[idx]

  is_ipo_first = np.zeros(n, dtype=bool)
  exempt = np.zeros(n, dtype=bool)
  for i in range(n):
    lt = lti[i]
    if lt < 0 or trade_idx < lt: continue
    ds = trade_idx - lt; b = boards[i]
    if b == 3: exempt[i] = (ds == 0)
    elif b == 2: exempt[i] = (signal_date >= date(2019, 7, 22) and ds <= 4)
    elif b == 1:
      if signal_date >= date(2020, 8, 24) and ds <= 4: exempt[i] = True
      elif signal_date < date(2014, 1, 1) and ds == 0: exempt[i] = True
    else:
      if signal_date >= date(2023, 4, 10) and ds <= 4: exempt[i] = True
      elif signal_date < date(2014, 1, 1) and ds == 0: exempt[i] = True
    if not exempt[i] and lt >= 0 and trade_idx == lt and signal_date >= date(2014, 1, 1):
      is_ipo_first[i] = True

  # issuePrice fallback 仅 IPO 首日
  if issue_price_all is not None and np.any(is_ipo_first):
    need_fb = is_ipo_first & ~valid_preclose
    if np.any(need_fb):
      ips = issue_price_all[idx].astype(np.float64)
      vip = need_fb & ~np.isnan(ips) & (ips > 0)
      precloses[vip] = ips[vip]; valid_preclose[vip] = True

  ratios = np.where(is_ipo_first & ~exempt, 0.44, ratios)
  has_limit = valid_preclose & ~exempt

  if is_buy:
    up_limits = np.where(has_limit, _round_half_up_np(precloses * (1.0 + ratios)), np.nan)
    limit_up = valid_open & has_limit & (opens >= up_limits - _EPS)
    blocked = limit_up.copy()
    for i in range(n):
      if not is_ipo_first[i] or blocked[i]: continue
      if abs(float(opens[i]) - float(low_all[trade_idx, idx[i]])) < _EPS and float(high_all[trade_idx, idx[i]]) >= up_limits[i] - _EPS:
        blocked[i] = True
    tradable = valid_open & ~blocked
  else:
    down_limits = np.where(has_limit, _round_half_up_np(precloses * (1.0 - ratios)), np.nan)
    limit_down = valid_open & has_limit & (opens <= down_limits + _EPS)
    tradable = valid_open & ~limit_down

  suspended = np.sum(~valid_open)
  reasons = {'suspended': int(suspended)} if suspended > 0 else {}
  return tradable, reasons


def _backtest_direct(data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
                     weights, buy_n, sell_m, temperatures, holding_period=None,
                     verify_config=None,
                     position_multipliers=None, list_dates_map=None,
                     lightweight=False, rebalance=True):
  """直接 numpy 回测，不创建 TopN 对象。lightweight=True 跳过明细组装，仅返回收益序列。"""
  from core.strategies.sizers.sizer import Sizer

  account = StockAccountMocker(cash=500_000.0)
  delist_stock_info = get_delist_stock_info()

  daily_snapshots: List[Dict] = []
  prices: dict[str, float] = {}
  skipped_buy_reasons: Dict[str, int] = {}
  skipped_sell_reasons: Dict[str, int] = {}
  delist_events: List[Dict] = []

  n_stocks = len(valid_stocks)
  valid_cols = np.array([stock_indices[s] for s in valid_stocks], dtype=np.intp)

  # 批量合法性检查预计算
  board_type, base_ratio, list_tidx_arr = _precompute_limit_helpers(data, stock_indices, list_dates_map)
  # 原版 LegalityValidator 保留备用（与 _batch_limit_check 等价，已验证 467万次零差异）
  _validator_ref = LegalityValidator(
    st_mask=data.get('st_mask'), stock_codes=data.get('stock_codes'),
    trade_dates=data.get('trade_dates'), list_dates=list_dates_map,
  )

  open_all = data['open']
  close_all = data['close']
  high_all = data['high']
  low_all = data['low']
  st_all = data.get('st_mask')
  issue_price_all = data.get('issue_price')

  force_codes = []
  if verify_config:
    force_codes = [verify_config['force_stock_code']]
    force_codes += [c for c in verify_config.get('candidate_stock_codes', []) if c != verify_config['force_stock_code']]

  def _write_off_delisted_positions(signal_date: date_type, trade_date: date_type):
    if not account.positions:
      return
    for stock in list(account.positions.keys()):
      delist_info = delist_stock_info.get(stock)
      if delist_info is None or trade_date <= delist_info.delist_date:
        continue
      position = dict(account.positions[stock])
      buy_trade_date = position.get('buy_trade_date') or position.get('buy_date')
      cost = position.get('cost', 0.0)
      account.write_off_stock(
        code=stock, write_off_date=trade_date, write_off_reason='退市归零',
        signal_date=signal_date, price_field='delist_zero',
        signal_dividend_type='back', execution_dividend_type='none',
      )
      delist_events.append({
        'code': stock, 'delist_date': delist_info.delist_date,
        'clear_signal_date': signal_date, 'clear_trade_date': trade_date,
        'buy_trade_date': buy_trade_date,
        'holding_days': _count_holding_trading_days(buy_trade_date, trade_date),
        'volume': position.get('volume', 0), 'cost': cost, 'income': -cost,
        'income_pct': -100.0 if cost else 0.0, 'clear_reason': '退市归零',
      })

  if force_codes:
    def _prepend_forced(lst, n):
      ordered, seen = [], set()
      for code in force_codes:
        if code and code not in seen:
          seen.add(code)
          ordered.append(code)
      for code in lst:
        if code not in seen:
          seen.add(code)
          ordered.append(code)
      return ordered[:n]
  else:
    _prepend_forced = None

  if lightweight:
    _lw_daily_returns = []
    _lw_last_asset = account.init_cash
    _daily_return_time = 0.0

  for i, dt in enumerate(valid_dates):
    signal_date = dt.date() if hasattr(dt, 'date') else dt
    date_idx = date_indices[i]
    trade_idx = date_idx
    trade_date = signal_date

    _write_off_delisted_positions(signal_date, trade_date)

    is_rebalance_day = True
    if holding_period and holding_period > 1:
      is_rebalance_day = (i % holding_period == 0)

    if is_rebalance_day:
      final_score = np.zeros(n_stocks)
      for name, ranks_mat in all_scores.items():
        w = weights.get(name, 0.0)
        if w == 0:
          continue
        ranks = ranks_mat[date_idx][valid_cols]
        temp = temperatures.get(name, 1.0)
        if temp != 1.0:
          np.power(ranks, 1.0 / temp, out=ranks)
        final_score += ranks * w

      top_idx = np.argsort(-final_score)
      buy_n_stocks = [valid_stocks[i] for i in top_idx[:buy_n]]
      sell_m_stocks = [valid_stocks[i] for i in top_idx[:sell_m]]

      if force_codes:
        buy_n_stocks = _prepend_forced(buy_n_stocks, buy_n)
        sell_m_stocks = _prepend_forced(sell_m_stocks, sell_m)
    else:
      buy_n_stocks = []
      sell_m_stocks = []

    # 预提取当日数据行，避免逐股 2D 索引
    day_open = open_all[trade_idx]
    day_close = close_all[trade_idx]

    current_position_codes = set(account.positions.keys())
    price_universe = current_position_codes | set(sell_m_stocks) | set(buy_n_stocks)

    prices = {}
    for stock in price_universe:
      si = stock_indices.get(stock)
      if si is None:
        continue
      open_val = day_open[si]
      if np.isnan(open_val) or open_val <= 0:
        if stock in current_position_codes:
          close_val = day_close[si]
          if not np.isnan(close_val) and close_val > 0:
            prices[stock] = float(close_val)
          else:
            delist_info = delist_stock_info.get(stock)
            is_delisted = delist_info is not None and trade_date > delist_info.delist_date
            if not is_delisted:
              for t in range(trade_idx - 1, -1, -1):
                pc = close_all[t, si]
                if not np.isnan(pc) and pc > 0:
                  prices[stock] = float(pc)
                  break
        continue
      prices[stock] = float(open_val)

    executed_sell_list: List[str] = []
    if is_rebalance_day:
      sell_check = [s for s in current_position_codes if s not in sell_m_stocks]
      if sell_check:
        sell_idx = [stock_indices[s] for s in sell_check if s in stock_indices]
        sell_ok, _ = _batch_limit_check(
          sell_check, sell_idx, trade_idx, signal_date,
          board_type, base_ratio, list_tidx_arr,
          open_all, close_all, high_all, low_all, st_all, issue_price_all, is_buy=False)
        for j, stock in enumerate(sell_check):
          if not sell_ok[j] or stock not in prices:
            continue
          account.clear_stock(
            code=stock, price=prices[stock], clear_date=trade_date, clear_reason='调仓换出',
            signal_date=signal_date, price_field='open',
            signal_dividend_type='back', execution_dividend_type='none',
          )
          executed_sell_list.append(stock)

    tradable_buy_stocks = []
    blocked_buy_details: List[Dict] = []
    if is_rebalance_day and buy_n_stocks:
      buy_idx = [stock_indices[s] for s in buy_n_stocks if s in stock_indices]
      # 先过滤无价格的股票
      valid_buy, valid_buy_idx = [], []
      for s, si in zip(buy_n_stocks, buy_idx):
        if s in prices:
          valid_buy.append(s); valid_buy_idx.append(si)
      if valid_buy:
        buy_ok, _ = _batch_limit_check(
          valid_buy, valid_buy_idx, trade_idx, signal_date,
          board_type, base_ratio, list_tidx_arr,
          open_all, close_all, high_all, low_all, st_all, issue_price_all, is_buy=True)
        for j, stock in enumerate(valid_buy):
          if not buy_ok[j]:
            continue
          tradable_buy_stocks.append(stock)

    executed_buy_records: List[Dict] = []
    if is_rebalance_day and tradable_buy_stocks:
      stock_infos = [(s, prices[s]) for s in tradable_buy_stocks]
      effective_cash = account.current_cash
      if position_multipliers is not None and not np.isnan(position_multipliers[i]):
        effective_cash = account.current_cash * position_multipliers[i]
      allocations = Sizer.allocate(stock_infos, total_capital=effective_cash)
      for stock, volume in allocations.items():
        if stock in account.positions or volume <= 0:
          continue
        price = prices[stock]
        buy_fee_rate = account.commission + account.transfer_fee + account.slippage
        max_vol = int(account.current_cash / (price * (1 + buy_fee_rate)))
        max_vol = max_vol // 100 * 100
        volume = min(volume, max_vol)
        if volume <= 0:
          skipped_buy_reasons['insufficient_cash'] = skipped_buy_reasons.get('insufficient_cash', 0) + 1
          blocked_buy_details.append({
            'code': stock, 'reason': 'insufficient_cash',
            'signal_date': signal_date.isoformat(), 'trade_date': trade_date.isoformat(),
          })
          continue
        if not account.buy_stock(
          code=stock, volume=volume, price=price, buy_date=trade_date,
          signal_date=signal_date, price_field='open',
          signal_dividend_type='back', execution_dividend_type='none',
        ):
          skipped_buy_reasons['insufficient_cash'] = skipped_buy_reasons.get('insufficient_cash', 0) + 1
          blocked_buy_details.append({
            'code': stock, 'reason': 'insufficient_cash',
            'signal_date': signal_date.isoformat(), 'trade_date': trade_date.isoformat(),
          })
          continue
        executed_buy_records.append({
          'code': stock, 'signal_date': signal_date.isoformat(), 'trade_date': trade_date.isoformat(),
          'price': price, 'price_field': 'open', 'volume': volume,
          'signal_dividend_type': 'back', 'execution_dividend_type': 'none',
        })

    if rebalance:
      pos_snapshot = [(code, dict(pos)) for code, pos in account.positions.items() if code in prices]
      if len(pos_snapshot) > 1:
        pos_vals = {code: pos['volume'] * prices[code] for code, pos in pos_snapshot}
        total_eq = account.current_cash + sum(pos_vals.values())
        target = total_eq / len(pos_snapshot)

        for code, pos in pos_snapshot:
          cv = pos_vals[code]
          if cv > target * 1.01:
            sell_vol = int((cv - target) / prices[code] / 100) * 100
            if 0 < sell_vol < pos['volume']:
              si = stock_indices.get(code)
              if si is not None:
                ok, _ = _batch_limit_check(
                  [code], [si], trade_idx, signal_date,
                  board_type, base_ratio, list_tidx_arr,
                  open_all, close_all, high_all, low_all, st_all, issue_price_all, is_buy=False)
                if ok[0]:
                  account.sell_stock(code, sell_vol, prices[code], trade_date,
                                    clear_reason='rebalance', signal_date=signal_date)

        for code, pos in pos_snapshot:
          if code not in account.positions:
            continue
          cv = account.positions[code]['volume'] * prices[code]
          if cv < target * 0.99:
            buy_vol = int((target - cv) / prices[code] / 100) * 100
            if buy_vol > 0:
              cost = buy_vol * prices[code]
              fee = cost * (account.commission + account.transfer_fee + account.slippage)
              if account.current_cash >= cost + fee:
                si = stock_indices.get(code)
                if si is not None:
                  ok, _ = _batch_limit_check(
                    [code], [si], trade_idx, signal_date,
                    board_type, base_ratio, list_tidx_arr,
                    open_all, close_all, high_all, low_all, st_all, issue_price_all, is_buy=True)
                  if ok[0]:
                    account.buy_stock(code, buy_vol, prices[code], trade_date,
                                     signal_date=signal_date)

    if lightweight:
      import time as _time
      _t0 = _time.perf_counter()
      mkt_val = sum(prices.get(c, 0.0) * p['volume'] for c, p in account.positions.items())
      total_asset = account.current_cash + mkt_val
      if _lw_last_asset > 0:
        daily_ret = (total_asset - _lw_last_asset) / _lw_last_asset * 100
      else:
        daily_ret = 0.0
      _lw_daily_returns.append(daily_ret)
      _lw_last_asset = total_asset
      _daily_return_time += _time.perf_counter() - _t0
    else:
      assets = account.calc_assets(trade_date, prices)
      prev_total_asset = daily_snapshots[-1]['total_asset'] if daily_snapshots else account.init_cash
      daily_ret = (assets['total_asset'] - prev_total_asset) / prev_total_asset * 100 if prev_total_asset else 0.0
      daily_snapshots.append({
        'date': trade_date.strftime('%Y-%m-%d'),
        'signal_date': signal_date.strftime('%Y-%m-%d'),
        'trade_date': trade_date.strftime('%Y-%m-%d'),
        'signal_dividend_type': 'back', 'execution_dividend_type': 'none', 'price_field': 'open',
        'cash': assets['cash'], 'market_value': assets['market_value'],
        'total_asset': assets['total_asset'], 'daily_return_pct': daily_ret, 'cumulative_return_pct': 0.0,
        'sell_m_list': sell_m_stocks, 'buy_n_list': buy_n_stocks,
        'buy_n_diff_list': [s for s in buy_n_stocks if s not in sell_m_stocks],
        'executed_sell_list': executed_sell_list,
        'executed_buy_list': [r['code'] for r in executed_buy_records],
        'executed_buy_details': executed_buy_records,
        'blocked_buy_details': blocked_buy_details,
      })

  if lightweight:
    mkt_val = sum(prices.get(c, 0.0) * p['volume'] for c, p in account.positions.items())
    total_asset = account.current_cash + mkt_val
    total_return = (total_asset - account.init_cash) / account.init_cash * 100
    return {
      'total_return': total_return,
      'cleared_positions_count': len(account.cleared_positions),
      'daily_returns': _lw_daily_returns,
      'daily_snapshots': [], 'cumulative_returns': [], 'trade_log': [],
      'positions': [], 'cleared_positions': [], 'delist_events': [],
      'stock_name_map': {}, 'holding_stats': {},
      'executed_buy_count': 0, 'executed_sell_count': 0,
      'delist_count': 0, 'round_trip_count': 0, 'current_positions_count': 0,
      'skipped_buy_reasons': {}, 'skipped_sell_reasons': {},
      'final_asset': total_asset,
      '_daily_return_time_sec': _daily_return_time,
    }

  final_signal_date = valid_dates[-1].date() if hasattr(valid_dates[-1], 'date') else valid_dates[-1]
  final_trade_date = final_signal_date
  final_assets = account.calc_assets(final_trade_date, prices)
  total_return = (final_assets['total_asset'] - account.init_cash) / account.init_cash * 100

  cumulative_returns = []
  if daily_snapshots:
    for snap in daily_snapshots:
      cum = (snap['total_asset'] - account.init_cash) / account.init_cash * 100
      cumulative_returns.append(cum)
    for i, snap in enumerate(daily_snapshots):
      snap['cumulative_return_pct'] = cumulative_returns[i]

  daily_returns = [snap.get('daily_return_pct', 0.0) for snap in daily_snapshots]

  positions = account.calc_position_values(prices)
  for position in positions:
    position['holding_days'] = _count_holding_trading_days(
      position.get('buy_trade_date') or position.get('buy_date'), final_trade_date)

  cleared_positions = []
  for cleared in account.cleared_positions:
    buy_trade_date = cleared['pos'].get('buy_trade_date') or cleared['pos'].get('buy_date')
    clear_trade_date = cleared.get('clear_trade_date') or cleared.get('clear_date')
    holding_days = _count_holding_trading_days(buy_trade_date, clear_trade_date)
    cost = cleared['pos'].get('cost', 0)
    income = cleared.get('income', 0)
    income_pct = (income / cost * 100) if cost else 0.0
    cleared_positions.append({
      'code': cleared['code'],
      'buy_date': cleared['pos'].get('buy_date'),
      'buy_signal_date': cleared['pos'].get('buy_signal_date'),
      'buy_trade_date': buy_trade_date,
      'clear_date': cleared.get('clear_date'),
      'clear_signal_date': cleared.get('clear_signal_date'),
      'clear_trade_date': clear_trade_date,
      'holding_days': holding_days,
      'volume': cleared['pos'].get('volume', 0),
      'avg_price': cleared['pos'].get('avg_price', 0),
      'cost': cost,
      'clear_price': cleared.get('clear_price', 0),
      'income': income,
      'income_pct': income_pct,
      'clear_reason': cleared.get('clear_reason'),
      'price_field': cleared.get('price_field'),
      'signal_dividend_type': cleared.get('signal_dividend_type'),
      'execution_dividend_type': cleared.get('execution_dividend_type'),
    })

  trade_log = account.get_trade_log()
  executed_buy_count = len([t for t in trade_log if t.get('action') == 'buy'])
  executed_sell_count = len([t for t in trade_log if t.get('action') == 'sell'])

  all_stock_codes = set()
  for trade in trade_log:
    if trade.get('code'):
      all_stock_codes.add(trade['code'])
  for snapshot in daily_snapshots:
    all_stock_codes.update(snapshot.get('buy_n_list', []))
    all_stock_codes.update(snapshot.get('executed_buy_list', []))
  for position in positions:
    if position.get('code'):
      all_stock_codes.add(position['code'])
  for cleared in cleared_positions:
    if cleared.get('code'):
      all_stock_codes.add(cleared['code'])

  stock_name_map = _get_stock_name_map(all_stock_codes, data.get('stock_names'), data.get('stock_codes'))
  holding_stats = _calc_holding_stats(positions, cleared_positions)

  return {
    'total_return': total_return,
    'cleared_positions_count': len(cleared_positions),
    'round_trip_count': len(cleared_positions),
    'current_positions_count': len(account.positions),
    'daily_returns': daily_returns,
    'cumulative_returns': cumulative_returns,
    'trade_log': trade_log,
    'daily_snapshots': daily_snapshots,
    'positions': positions,
    'cleared_positions': cleared_positions,
    'delist_events': delist_events,
    'stock_name_map': stock_name_map,
    'holding_stats': holding_stats,
    'executed_buy_count': executed_buy_count,
    'executed_sell_count': executed_sell_count,
    'delist_count': len(delist_events),
    'skipped_buy_reasons': skipped_buy_reasons,
    'skipped_sell_reasons': skipped_sell_reasons,
    'final_asset': final_assets['total_asset'],
  }


def _find_latest_checkpoint() -> Path | None:
  """自动找到 results/ 下最新的 GA checkpoint 目录。"""
  results_dir = Path('results')
  if not results_dir.is_dir():
    return None
  candidates = []
  for d in results_dir.iterdir():
    if not d.is_dir():
      continue
    if not (d.name.startswith('ga_') or d.name.startswith('debug_')):
      continue
    ckpt = d / 'checkpoint.pkl'
    if ckpt.exists():
      candidates.append((d.name, d))
  if not candidates:
    return None
  candidates.sort(reverse=True)
  return candidates[0][1]


def _save_checkpoint(output_dir: Path, generation: int, ga_state: dict, ga_cache: dict,
           all_results: list, generation_results: list, next_configs: list,
           val_metrics_cache: dict):
  import pickle
  ckpt = {
    'generation': generation,
    'ga_state': ga_state,
    'ga_cache': ga_cache,
    'all_results': all_results,
    'generation_results': generation_results,
    'next_configs': next_configs,
    'val_metrics_cache': val_metrics_cache,
  }
  tmp = output_dir / 'checkpoint.pkl.tmp'
  with open(tmp, 'wb') as f:
    pickle.dump(ckpt, f)
  tmp.replace(output_dir / 'checkpoint.pkl')


def _load_checkpoint(checkpoint_dir: Path) -> dict:
  import pickle
  ckpt_path = checkpoint_dir / 'checkpoint.pkl'
  if not ckpt_path.exists():
    raise FileNotFoundError(f'checkpoint 文件不存在: {ckpt_path}')
  with open(ckpt_path, 'rb') as f:
    return pickle.load(f)


def _resolve_output_dir(output_dir_arg: str | None, mode: str) -> Path:
  if output_dir_arg:
    output_dir = Path(output_dir_arg)
  else:
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = results_dir / f'{mode}_{timestamp}'
  output_dir.mkdir(parents=True, exist_ok=True)
  return output_dir


# ========== 模式执行函数 ==========

def run_single_mode(args, mode_config, backtest_datetime_list, all_stocks):
  """单次回测模式：直接 numpy 回测，无 TopN 对象中间层"""
  import json

  if not args.individual_config:
    testback_logger.error('--individual-config 参数在 single 模式下必须指定')
    sys.exit(1)

  with open(args.individual_config, 'r', encoding='utf-8') as f:
    config_data = json.load(f)
  profile_name = resolve_profile_name(config_data)
  individual_config = dict(config_data['individual_config'])
  verify_config = _parse_single_verify_config(config_data)

  output_dir = _resolve_output_dir(args.output_dir, 'single')
  testback_logger.add_file_sink(str(output_dir / 'single.log'))
  testback_logger.info(f"日志文件: {output_dir / 'single.log'}")
  testback_logger.info(f"从文件加载 Individual_config: {args.individual_config}")

  stock_pool = individual_config.get('stock_pool')
  if stock_pool:
    pool_tuple = tuple(stock_pool) if isinstance(stock_pool, list) else stock_pool
    all_stocks = [s for s in all_stocks if s.startswith(pool_tuple)]
    testback_logger.info(f"stock_pool={_format_pool(stock_pool)}: {len(all_stocks)} 只")

  candidate_stock_pool = _extend_verify_stock_pool_with_historical_codes(
    all_stocks, backtest_datetime_list, verify_config)
  single_stock_pool = _resolve_single_stock_pool(candidate_stock_pool, verify_config)
  if verify_config:
    testback_logger.info(
      f"启用 single 退市验证: {verify_config['force_stock_code']}, "
      f"候选池={len(single_stock_pool)} 只"
    )

  testback_logger.info(f"使用配置进行单次回测: buy_n={individual_config['buy_n']}, sell_m={individual_config['sell_m']}")

  import core.factors as _all_factors
  config_factor_classes = []
  for fname in individual_config['weights']:
    cls = getattr(_all_factors, fname, None)
    if cls is None:
      testback_logger.error(f"因子类 {fname} 不存在")
      sys.exit(1)
    config_factor_classes.append(cls)

  scores_result = _compute_factor_scores(
    backtest_datetime_list, single_stock_pool,
    weights=individual_config['weights'], factor_classes=config_factor_classes,
  )
  if scores_result is None:
    testback_logger.error("因子计算失败，无有效交易日")
    sys.exit(1)
  data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices = scores_result

  list_dates_map = _compute_list_dates(data['stock_codes'], data['open'], data['trade_dates'])

  signal_dates = [d.date() if hasattr(d, 'date') else d for d in valid_dates]
  trade_dates = list(signal_dates)
  testback_logger.info(
    f"回测信号范围: {signal_dates[0]} ~ {signal_dates[-1]}，"
    f"执行范围: {trade_dates[0]} ~ {trade_dates[-1]}，共 {len(valid_dates)} 个调仓日"
  )

  timing_multipliers = _compute_timing_multipliers(individual_config, valid_dates)
  if timing_multipliers is not None:
    from testback.market_timing import INDEX_INFO
    timing_index = individual_config.get('timing_index', 'sh000852')
    direction = individual_config.get('timing_direction', 1)
    d_name = '顺势' if direction == 1 else '逆势'
    idx_name = INDEX_INFO.get(timing_index, timing_index)
    testback_logger.info(f"大盘择时启用: {idx_name} {d_name}, base={individual_config.get('timing_base', 0.5)}, "
                         f"leverage={individual_config.get('timing_leverage', 10)}, "
                         f"window={individual_config.get('timing_window', 20)}, "
                         f"multiplier范围=[{np.nanmin(timing_multipliers):.2f}, {np.nanmax(timing_multipliers):.2f}]")

  result = _backtest_direct(
    data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
    weights=individual_config['weights'],
    buy_n=individual_config['buy_n'], sell_m=individual_config['sell_m'],
    temperatures=individual_config['temperatures'],
    holding_period=individual_config.get('holding_period'),
    verify_config=verify_config,
    position_multipliers=timing_multipliers,
    list_dates_map=list_dates_map,
    rebalance=True,
  )

  testback_logger.info(f"回测完成: 总收益={result['total_return']:.2f}%, 日收益计算耗时={result.get('_daily_return_time_sec', 0):.3f}s")

  signal_date_strs = [d.strftime('%Y-%m-%d') for d in signal_dates]
  trade_date_strs = [d.strftime('%Y-%m-%d') for d in trade_dates]

  metrics = compute_strategy_metrics(
    cumulative_returns_pct=result.get('cumulative_returns', []),
    trade_dates=trade_date_strs, init_cash=500_000.0,
    final_asset=result.get('final_asset', 500_000.0),
    trade_log=result.get('trade_log', []),
  )
  hs300_returns = compute_hs300_cumulative_returns(trade_date_strs)
  per_year_metrics = compute_per_year_metrics(result.get('cumulative_returns', []), trade_date_strs)

  report_data = {
    'individual_config': individual_config,
    'total_return': result['total_return'],
    'daily_returns': result.get('daily_returns', []),
    'cumulative_returns': result.get('cumulative_returns', []),
    'signal_dates': signal_date_strs,
    'trade_dates': trade_date_strs,
    'trade_log': result.get('trade_log', []),
    'daily_snapshots': result.get('daily_snapshots', []),
    'positions': result.get('positions', []),
    'cleared_positions': result.get('cleared_positions', []),
    'delist_events': result.get('delist_events', []),
    'stock_name_map': result.get('stock_name_map', {}),
    'holding_stats': result.get('holding_stats', {}),
    'executed_buy_count': result.get('executed_buy_count', 0),
    'executed_sell_count': result.get('executed_sell_count', 0),
    'delist_count': result.get('delist_count', 0),
    'round_trip_count': result.get('round_trip_count', result.get('cleared_positions_count', 0)),
    'final_asset': result.get('final_asset', 500_000.0),
    'metrics': metrics,
    'per_year_metrics': per_year_metrics,
    'hs300_returns': hs300_returns,
    'cleared_positions_count': result['cleared_positions_count'],
    'current_positions_count': result['current_positions_count'],
    'init_cash': 500_000.0,
    'verify_config': verify_config,
    'report_metadata': {
      'config_path': str(Path(args.individual_config).resolve()),
      'stock_pool_size': len(single_stock_pool),
    },
    'rebalance_rule': {
      'signal_timing': 'T-1', 'trade_timing': 'T open',
      'signal_dividend_type': 'back', 'execution_dividend_type': 'none', 'price_field': 'open',
    },
    'period': {
      'signal_start': signal_date_strs[0], 'signal_end': signal_date_strs[-1],
      'trade_start': trade_date_strs[0], 'trade_end': trade_date_strs[-1],
      'start': trade_date_strs[0], 'end': trade_date_strs[-1],
    },
  }

  if mode_config['save_charts']:
    try:
      from testback.reportor import generate_single_report
      html_path = generate_single_report(report_data, output_dir)
      testback_logger.info(f"可视化报告已保存至: {html_path}")
    except ImportError:
      testback_logger.warning("testback.report 模块未找到，跳过可视化报告生成")
    except Exception as e:
      testback_logger.warning(f"可视化报告生成失败: {e}")

  testback_logger.remove_file_sink()
  return result


# === 文件后备 memmap 数据共享（Windows spawn，每数组独立文件 → offset=0 天然对齐） ===
_MEMMAP_DIRS: list = []

def _arrays_from_memmap(info: dict) -> dict:
  return {name: np.memmap(filepath, dtype=np.dtype(dtype_str), mode='r', shape=shape)
          for name, (filepath, shape, dtype_str) in info.items()}

def _cleanup_memmap():
  import shutil
  for d in _MEMMAP_DIRS:
    shutil.rmtree(d, ignore_errors=True)
  _MEMMAP_DIRS.clear()


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
  scores = _scores_to_ranks(raw.astype(np.float32, copy=False))
  if row_slice is not None:
    full = np.full((n_full, scores.shape[1]), np.nan, dtype=scores.dtype)
    full[r0:r1] = scores
    scores = full
  filepath = Path(tmpdir) / f'factor_{name}.bin'
  scores.tofile(str(filepath))
  return name, (str(filepath), scores.shape, str(scores.dtype))

def _compute_list_dates(stock_codes_arr, open_arr, trade_dates_arr) -> dict:
  """从 open 数据推算上市日期，返回 {code: date}。在主进程调用一次，避免多进程内存压力。"""
  result = {}
  valid = ~np.isnan(open_arr) & (open_arr > 0)
  first_idx = np.argmax(valid, axis=0)
  has_valid = np.any(valid, axis=0)
  for i, code in enumerate(stock_codes_arr):
    if has_valid[i]:
      result[str(code)] = trade_dates_arr[first_idx[i]].astype('datetime64[D]').astype(date)
  return result


def _compute_timing_multipliers(config, valid_dates, index_data=None):
    """提取择时计算逻辑，避免 run_single / _worker / ga_runner 三处重复"""
    timing_enabled = config.get('timing_enabled', True)
    timing_base = config.get('timing_base')
    if not (timing_enabled and timing_base is not None):
        return None
    from testback.market_timing import load_index_open, compute_position_multiplier, INDEX_INFO
    idx_symbol = config.get('timing_index', 'sh000852')
    if index_data is not None:
        idx_open = index_data.get(idx_symbol)
    else:
        _, idx_open = load_index_open(idx_symbol, valid_dates)
    if idx_open is None:
        return None
    window = config.get('timing_window', 20)
    leverage = config.get('timing_leverage', 10)
    direction = config.get('timing_direction', 1)
    return compute_position_multiplier(idx_open, window=window, base=timing_base, leverage=leverage, direction=direction)


def _load_all_index_data(valid_dates):
    from testback.market_timing import load_index_open, INDEX_INFO
    index_data = {}
    for sym in INDEX_INFO:
        _, index_data[sym] = load_index_open(sym, valid_dates)
    return index_data


def _worker_evaluate(args):
  try:
    train_info, score_keys, valid_dates, date_indices, stock_indices, \
        all_stocks_list, config, index_data, list_dates_map = args

    all_arrays = _arrays_from_memmap(train_info)
    data = {k: v for k, v in all_arrays.items() if k not in score_keys}
    all_scores = {k: v for k, v in all_arrays.items() if k in score_keys}

    stock_pool = config.get('stock_pool')
    pool_stocks = [s for s in all_stocks_list if s.startswith(tuple(stock_pool))] if stock_pool else list(all_stocks_list)

    timing_multipliers = _compute_timing_multipliers(config, valid_dates, index_data)

    r = _backtest_direct(
      data, all_scores, valid_dates, date_indices, pool_stocks, stock_indices,
      weights=config['weights'], buy_n=config['buy_n'], sell_m=config['sell_m'],
      temperatures=config['temperatures'],
      holding_period=config.get('holding_period'),
      position_multipliers=timing_multipliers, list_dates_map=list_dates_map,
      lightweight=True, rebalance=True)

    # 释放大内存，worker 常驻复用
    del all_arrays, all_scores, data
    import gc; gc.collect()

    metrics = _compute_metrics_simple(r['daily_returns'])
    sharpe = metrics['sharpe']
    annualized = metrics['annualized']
    dd = metrics['max_drawdown']
    total_return = r['total_return']
    cleared_count = r['cleared_positions_count']
    daily_ret_time = r.get('_daily_return_time_sec', 0.0)
  except Exception:
    import traceback
    _err = traceback.format_exc()
    sys.stderr.write(_err)
    sys.stderr.flush()
    sharpe = -999.0
    total_return = -999.0
    cleared_count = 0
    dd = 0.0
    annualized = 0.0
    daily_ret_time = 0.0

  return {
    'individual_config': config,
    'total_return': total_return,
    'sharpe': sharpe,
    'annualized': annualized,
    'max_drawdown': dd,
    'cleared_positions_count': cleared_count,
    'current_positions_count': 0,
    '_error': locals().get('_err', None),
    '_daily_return_time_sec': daily_ret_time,
  }


def _compute_metrics_simple(daily_returns: list) -> dict:
  daily = np.array(daily_returns, dtype=float)
  daily = daily[np.isfinite(daily)]
  n = len(daily)
  if n < 2:
    return {'annualized': 0.0, 'max_drawdown': 0.0, 'sharpe': 0.0}

  mean_ret = float(np.mean(daily))
  std_ret = float(np.std(daily, ddof=1))
  sharpe = float(mean_ret / std_ret * np.sqrt(252.0)) if std_ret > 0 else 0.0

  cum_ret = np.cumprod(1.0 + daily / 100.0)
  years = n / 252.0
  annualized = float((cum_ret[-1] ** (1.0 / years) - 1) * 100) if years > 0 and cum_ret[-1] > 0 else 0.0

  peaks = np.maximum.accumulate(cum_ret)
  drawdowns = cum_ret / peaks - 1.0
  max_dd = float(np.min(drawdowns) * 100)

  return {'annualized': annualized, 'max_drawdown': max_dd, 'sharpe': sharpe}


def _format_pool(pool: tuple) -> str:
  names = {'60': '沪主', '00': '深主', '0': '深主', '30': '创业板', '688': '科创板'}
  return '+'.join(names.get(str(p), str(p)) for p in pool) if pool else 'all'

def _format_timing(config: dict) -> str:
  if not config.get('timing_enabled', True):
    return ', timing=OFF'
  base = config.get('timing_base')
  if base is None:
    return ''
  d = config.get('timing_direction', 1)
  d_str = '顺势' if d == 1 else '逆势'
  leverage = config.get('timing_leverage', 10)
  window = config.get('timing_window', 20)
  idx_val = config.get('timing_index', 'sh000852')
  from testback.market_timing import INDEX_INFO as _IDX
  idx_name = _IDX.get(idx_val, idx_val)
  return f', timing={base}/{leverage}x({d_str}, win={window}, {idx_name})'


def _try_send_feishu(msg: str):
  try:
    from trading.lark.sender import lark_sender
    lark_sender.send_msg(msg)
  except Exception:
    pass


def _eval_parallel(worker_args, results_list, ga_cache, logger, pool):
  import gc
  import json as json_mod

  for result in pool.imap_unordered(_worker_evaluate, worker_args, chunksize=1):
    results_list.append(result)
    key = json_mod.dumps(_config_key(result['individual_config']), ensure_ascii=False)
    ga_cache[key] = result
    if result.get('sharpe', -999) <= -900:
      logger.warning(f"  失败: {result.get('_error', '未知')[:200]}")
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
  val_metrics_cache = {}
  start_generation = 0

  cache_path = output_dir / f'state.{os.getpid()}.jsonl'
  ga_cache = {}
  if cache_path.exists():
    try:
      with open(cache_path, 'r', encoding='utf-8') as f:
        for line in f:
          line = line.strip()
          if not line:
            continue
          try:
            entry = json_mod.loads(line)
            ga_cache.update(entry)
          except Exception:
            pass
    except Exception:
      pass
  for v in ga_cache.values():
    v.setdefault('annualized', 0.0)
    v.setdefault('max_drawdown', 0.0)
  if ga_cache:
    testback_logger.info(f"加载 GA 缓存: {len(ga_cache)} 条")

  val_cache_path = output_dir / 'val_cache.json'
  val_eval_cache = {}
  if val_cache_path.exists():
    try:
      with open(val_cache_path, 'r', encoding='utf-8') as f:
        val_eval_cache = json_mod.load(f)
    except Exception:
      pass
  if val_eval_cache:
    testback_logger.info(f"加载验证集缓存: {len(val_eval_cache)} 条")

  test_cache_path = output_dir / 'test_cache.json'
  test_eval_cache = {}
  if test_cache_path.exists():
    try:
      with open(test_cache_path, 'r', encoding='utf-8') as f:
        test_eval_cache = json_mod.load(f)
    except Exception:
      pass
  if test_eval_cache:
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
  test_end = date(2026, 5, 15)
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

  _train_info = info
  _score_keys = score_keys
  train_list_dates = list_dates_full
  testback_logger.info(f"训练集就绪 ({time.time() - t0:.1f}s), {len(all_valid_stocks)} 只, {len(valid_dates)} 天")

  index_data = _load_all_index_data(valid_dates)

  import gc, ctypes
  gc.collect()

  n_workers = 30
  ctx = get_context('spawn')
  ga_pool = ctx.Pool(processes=n_workers)
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

  if resume_dir:
    ckpt = _load_checkpoint(resume_dir)
    ga_state = ckpt['ga_state']
    ga_cache = ckpt['ga_cache']
    all_results = ckpt['all_results']
    generation_results = ckpt['generation_results']
    next_configs = ckpt['next_configs']
    val_metrics_cache = ckpt.get('val_metrics_cache', ckpt.get('test_metrics_cache', {}))
    start_generation = ckpt['generation'] + 1

    # 修复 checkpoint 中不在当前搜索空间内的配置
    repaired = 0
    for cfg in next_configs:
        if repair_config(cfg, profile_name):
            repaired += 1
    for ind in ga_state.get('population', []):
        if isinstance(ind, dict) and repair_config(ind, profile_name):
            repaired += 1
    for ind in ga_state.get('hall_of_fame', []):
        if isinstance(ind, dict) and repair_config(ind, profile_name):
            repaired += 1
    for r in all_results:
        if isinstance(r.get('individual_config'), dict) and repair_config(r['individual_config'], profile_name):
            repaired += 1
    # 清理与旧配置关联的 fitness_cache（key 已变）
    ga_state['fitness_cache'] = {}
    ga_cache.clear()
    if repaired:
        testback_logger.warning(f"从 checkpoint 修复了 {repaired} 个不在当前搜索空间的配置参数")

    testback_logger.info(f"从 checkpoint 恢复: 第 {start_generation} 代开始 (已完成 {ckpt['generation'] + 1}/{generations} 代)")
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

  if not resume_dir:
    all_results = []
    generation_results = []

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
        try:
          with open(cache_path, 'a', encoding='utf-8') as f:
            for cfg in uncached_configs:
              key = json_mod.dumps(_config_key(cfg), ensure_ascii=False)
              entry = ga_cache.get(key)
              if entry is not None:
                f.write(json_mod.dumps({key: entry}, ensure_ascii=False) + '\n')
        except (PermissionError, OSError) as e:
          testback_logger.warning(f"保存缓存失败（非致命）: {e}")

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
            try:
              with open(val_cache_path, 'w', encoding='utf-8') as f:
                json_mod.dump(val_eval_cache, f, ensure_ascii=False)
            except (PermissionError, OSError) as e:
              testback_logger.warning(f"保存验证集缓存失败: {e}")
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
            try:
              with open(test_cache_path, 'w', encoding='utf-8') as f:
                json_mod.dump(test_eval_cache, f, ensure_ascii=False)
            except (PermissionError, OSError) as e:
              testback_logger.warning(f"保存测试集缓存失败: {e}")

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
              try:
                with open(test_cache_path, 'w', encoding='utf-8') as f:
                  json_mod.dump(test_eval_cache, f, ensure_ascii=False)
              except (PermissionError, OSError) as e:
                testback_logger.warning(f"保存测试集缓存失败: {e}")

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

      all_results.extend(results_list)

      if not is_debug:
        fitnesses = [ind['fitness'] for ind in results_list]
        generation_stats = {
          'generation': generation, 'generation_time': generation_time,
          'population_size': len(results_list),
          'max_fitness': max(fitnesses), 'mean_fitness': sum(fitnesses) / len(fitnesses),
          'min_fitness': min(fitnesses),
        }
        if report_gen:
          generation_stats['val_best_sharpe'] = float(train_best_val_m['sharpe'])
          generation_stats['val_best_annualized'] = float(train_best_val_m['annualized'])
          generation_stats['val_best_mdd'] = float(train_best_val_m['max_drawdown'])
          generation_stats['test_train_best_sharpe'] = float(_test_train_best_m['sharpe'])
          generation_stats['test_train_best_annualized'] = float(_test_train_best_m['annualized'])
          generation_stats['test_train_best_mdd'] = float(_test_train_best_m['max_drawdown'])
        generation_results.append(generation_stats)

        try:
          with open(output_dir / 'generation_results.pkl', 'wb') as f:
            pickle.dump(generation_results, f)

          best_in_gen = max(results_list, key=lambda x: x['fitness'])
          profile_metadata = get_profile_metadata(profile_name)
          best_result = {
            **profile_metadata,
            'individual_config': best_in_gen['individual_config'],
            'fitness': best_in_gen['fitness'], 'generation': generation + 1,
            'generation_time': generation_time, 'population_size': len(results_list),
          }
          with open(output_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
            json_mod.dump(best_result, f, indent=2, ensure_ascii=False)
        except (PermissionError, OSError) as e:
          testback_logger.warning(f"保存generation_results失败（非致命）: {e}")

      next_configs = ga_optimizer(results_list, state=ga_state, population_size=population_size,
                                  hall_of_fame_size=population_size, profile_name=profile_name,
                                  ga_cache=ga_cache)

      if not is_debug:
        _save_checkpoint(output_dir, generation, ga_state, ga_cache,
                         all_results, generation_results, next_configs, val_metrics_cache)
        # 限制 all_results 大小，防止内存无界增长
        max_results = 5000
        if len(all_results) > max_results:
          all_results = all_results[-max_results:]

  finally:
    ga_pool.terminate()
    ga_pool.join()
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

    sharpes = [r['sharpe'] for r in all_results if r.get('sharpe') is not None]
    returns = [r['total_return'] for r in all_results]
    testback_logger.info(f"\n{'=' * 60}")
    testback_logger.info("回测执行完成")
    testback_logger.info(f"  总回测次数: {len(all_results)}")
    testback_logger.info(f"  平均夏普率: {sum(sharpes) / len(sharpes):.3f}")
    testback_logger.info(f"  最大夏普率: {max(sharpes):.3f}")
    testback_logger.info(f"  平均收益率: {sum(returns) / len(returns):.2f}%")
    testback_logger.info(f"  最大收益率: {max(returns):.2f}%")
    testback_logger.info(f"  正收益策略: {len([r for r in returns if r > 0])} 个")
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

def _main_impl():
  import argparse

  from loguru import logger as loguru_logger

  ts = datetime.now()

  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default='ga', choices=['single', 'debug', 'ga'])
  parser.add_argument('--individual-config', type=str, default=None)
  parser.add_argument('--output-dir', type=str, default=None)
  parser.add_argument('--start-date', type=str, default=None)
  parser.add_argument('--end-date', type=str, default=None)
  parser.add_argument('--warm-start', type=str, default=None, help='热启动种群 JSON 文件路径')
  parser.add_argument('--resume', type=str, nargs='?', const='auto', default=None,
                      help='从 checkpoint 恢复: 不传则自动找最新, 或指定目录路径')
  parser.add_argument('--profile', type=str, default=None)
  args = parser.parse_args()

  profile_name = args.profile or DEFAULT_GA_PROFILE
  mode_configs = get_mode_configs(profile_name)
  mode_config = mode_configs[args.mode].copy()

  loguru_logger.remove()
  loguru_logger.add(sys.stderr, level=mode_config['log_level'])

  testback_logger.info(f"运行模式: {args.mode} - {mode_config['desc']}")

  def parse_date(s):
    if s is None:
      return None
    s = s.replace('-', '')
    if len(s) == 8:
      return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    raise ValueError(f'日期格式错误: {s}')

  start_date = parse_date(args.start_date) or date(2020, 6, 30)
  end_date = parse_date(args.end_date) or date(2024, 12, 31)

  if args.mode == 'ga':
    start_date, end_date = get_profile_preload_range(profile_name)
    testback_logger.info(f"GA 模式固定预加载区间: {start_date.strftime('%Y%m%d')} - {end_date.strftime('%Y%m%d')}")

  backtest_datetime_list = [
    datetime.combine(d, datetime.min.time())
    for d in get_trading_date_span(start_date, end_date)
  ]

  factor_classes = get_profile_factor_classes(profile_name)
  factor_histories = {factor_cls.__name__: factor_cls().hist_days for factor_cls in factor_classes}
  max_hist_days = max(factor_histories.values(), default=0)
  hist_detail = ', '.join(f'{name}={days}天' for name, days in factor_histories.items())
  testback_logger.info(f"因子历史需求: {hist_detail}，最大需求={max_hist_days}天")

  if args.mode == 'single':
    from data.db import allow_buy_stock_code_list
    filtered_stocks = list(allow_buy_stock_code_list())
    testback_logger.info(f"股票池 (allow_buy): {len(filtered_stocks)} 只")
    result = run_single_mode(args, mode_config, backtest_datetime_list, filtered_stocks)
  else:
    # GA/debug: 从 npz 取股票列表，不触发 xtdata 连接
    npz_dir = Path(__file__).resolve().parent.parent / 'data' / 'runtime'
    npz_files = sorted(npz_dir.glob('runtime_*.npz'))
    if not npz_files:
      raise FileNotFoundError(f"未找到 runtime npz 文件: {npz_dir}")
    filtered_stocks = [str(s) for s in np.load(npz_files[0], allow_pickle=False)['stock_codes']]
    testback_logger.info(f"股票池 (npz): {len(filtered_stocks)} 只")
    resume_dir = None
    if args.resume:
      if args.resume == 'auto':
        resume_dir = _find_latest_checkpoint()
        if resume_dir is None:
          testback_logger.error('--resume 未找到可恢复的 checkpoint')
          sys.exit(1)
      else:
        resume_dir = Path(args.resume)
        if not resume_dir.is_dir() or not (resume_dir / 'checkpoint.pkl').exists():
          testback_logger.error(f'--resume 指定目录无效或缺少 checkpoint.pkl: {resume_dir}')
          sys.exit(1)
      testback_logger.info(f'从 checkpoint 恢复: {resume_dir}')
    result = _run_ga(args, mode_config, backtest_datetime_list, filtered_stocks, profile_name=profile_name, resume_dir=resume_dir)

  te = datetime.now()
  testback_logger.info(f"总耗时: {(te - ts).total_seconds():.2f} 秒")
  return result


def main():
  with keep_windows_awake() as keep_awake_enabled:
    if keep_awake_enabled:
      testback_logger.info('已启用 Windows 防休眠，任务结束后自动恢复')
    else:
      testback_logger.warning('未能启用 Windows 防休眠，系统可能仍按当前电源策略休眠')
    return _main_impl()


if __name__ == "__main__":
  main()
