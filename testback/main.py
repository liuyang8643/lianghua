import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core import get_stock_detail, init_stock_detail_cache
from core.database import allow_buy_stock_code_list
from core.database.delist import get_delist_stock_info
from core.strategies.top_n import _load_runtime_npz
from utils.stock.time import AFTERNOON_END, get_trading_date_span
from utils.stock.legality import LegalityValidator
from utils.windows_awake import keep_windows_awake
from testback.account import StockAccountMocker
from testback.ga_config import (
  DEFAULT_GA_PROFILE,
  build_individual_config,
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
  sample_stock_pool,
  sample_timing_sensitivity,
  sample_timing_direction,
  sample_timing_enabled,
  sample_timing_index,
  sample_ma_period,
)
from testback.metrics import compute_hs300_cumulative_returns, compute_strategy_metrics, compute_per_year_metrics

from datetime import date, date as date_type, datetime
from testback.logger import testback_logger


# ========== GA 核心函数 ==========

def _config_key(config: dict) -> tuple:
  w = config['weights']
  sp = config.get('stock_pool')
  if isinstance(sp, list):
    sp = tuple(sp)
  return (config['buy_n'], config['sell_m'], config.get('freeze_days', 0),
          sp, config.get('timing_sensitivity'),
          config.get('timing_direction'), config.get('timing_enabled'),
          config.get('ma_period'), config.get('timing_index'),
          tuple(sorted(w.items())))


def ga_optimizer(results, state, population_size=24, hall_of_fame_size=24, profile_name=DEFAULT_GA_PROFILE):
  import random

  results_list = list(results) if not isinstance(results, list) else results

  for r in results_list:
    key = _config_key(r['individual_config'])
    state['fitness_cache'][key] = r['sharpe']

  testback_logger.info(f"GA 本轮有效结果: {len(results_list)} 个")

  if not state['population']:
    results_list.sort(key=lambda r: r['sharpe'], reverse=True)
    state['population'] = [r['individual_config'] for r in results_list[:population_size]]
    testback_logger.info(f"GA 初始化种群: {len(state['population'])} 个个体")

  def get_fitness(config):
    key = _config_key(config)
    return state['fitness_cache'][key]

  population_with_fitness = [(ind, get_fitness(ind)) for ind in state['population']]
  population_with_fitness.sort(key=lambda x: x[1], reverse=True)
  parents = [ind for ind, _ in population_with_fitness[:population_size]]

  all_individuals = state['hall_of_fame'] + parents
  unique_dict = {}
  for ind in all_individuals:
    key = _config_key(ind)
    fitness = get_fitness(ind)
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
    freeze_days = p1.get('freeze_days', 0)
    fc = _crossover_field(p1, p2, 'factor_choice') if has_factor_choice else None
    stock_pool = _crossover_field(p1, p2, 'stock_pool') if 'stock_pool' in search_spaces else None
    timing_sens = _crossover_field(p1, p2, 'timing_sensitivity') if 'timing_sensitivity' in search_spaces else None
    timing_dir = _crossover_field(p1, p2, 'timing_direction') if 'timing_direction' in search_spaces else None
    timing_enabled = _crossover_field(p1, p2, 'timing_enabled') if 'timing_enabled' in search_spaces else None
    ma_period = _crossover_field(p1, p2, 'ma_period') if 'ma_period' in search_spaces else None
    timing_index = _crossover_field(p1, p2, 'timing_index') if 'timing_index' in search_spaces else None
    crossed_weights = None
    if has_weight_search:
      crossed_weights = {}
      for k in p1['weights']:
        crossed_weights[k] = p1['weights'][k] if random.random() < 0.5 else p2['weights'][k]
    return build_individual_config(position_count, freeze_days=freeze_days, weights=crossed_weights,
                                   factor_choice=fc,
                                   stock_pool=stock_pool, timing_sensitivity=timing_sens,
                                   timing_direction=timing_dir,
                                   timing_enabled=timing_enabled, ma_period=ma_period,
                                   timing_index=timing_index, profile_name=profile_name)

  weight_spaces = get_profile_weight_search_spaces(profile_name) if has_weight_search else {}

  def _collect_dims():
    dims = ['position_count']
    if has_factor_choice: dims.append('factor_choice')
    if 'stock_pool' in search_spaces: dims.append('stock_pool')
    if 'timing_sensitivity' in search_spaces: dims.append('timing_sensitivity')
    if 'timing_direction' in search_spaces: dims.append('timing_direction')
    if 'timing_enabled' in search_spaces: dims.append('timing_enabled')
    if 'ma_period' in search_spaces: dims.append('ma_period')
    if 'timing_index' in search_spaces: dims.append('timing_index')
    for k in weight_spaces: dims.append(f'weight_{k}')
    return dims

  import math
  _all_dims = _collect_dims()

  def mutate_config(config):
    if not _all_dims:
      return build_individual_config(config['buy_n'], freeze_days=config.get('freeze_days', 0),
                                      weights=config.get('weights'), profile_name=profile_name)
    n_mutate = max(1, min(math.ceil(random.uniform(0.25, 0.50) * len(_all_dims)), len(_all_dims)))
    mutate_dims = set(random.sample(_all_dims, n_mutate))
    position_count = sample_position_count(profile_name=profile_name) if 'position_count' in mutate_dims else config['buy_n']
    fc = sample_factor_choice(profile_name=profile_name) if 'factor_choice' in mutate_dims else config.get('factor_choice')
    stock_pool = sample_stock_pool(profile_name=profile_name) if 'stock_pool' in mutate_dims else config.get('stock_pool')
    timing_sens = sample_timing_sensitivity(profile_name=profile_name) if 'timing_sensitivity' in mutate_dims else config.get('timing_sensitivity')
    timing_dir = sample_timing_direction(profile_name=profile_name) if 'timing_direction' in mutate_dims else config.get('timing_direction')
    timing_enabled = sample_timing_enabled(profile_name=profile_name) if 'timing_enabled' in mutate_dims else config.get('timing_enabled')
    ma_period = sample_ma_period(profile_name=profile_name) if 'ma_period' in mutate_dims else config.get('ma_period')
    timing_index = sample_timing_index(profile_name=profile_name) if 'timing_index' in mutate_dims else config.get('timing_index')
    mutated_weights = dict(config['weights'])
    for k in weight_spaces:
      if f'weight_{k}' in mutate_dims:
        mutated_weights[k] = random.choice(weight_spaces[k])
    return build_individual_config(position_count, freeze_days=config.get('freeze_days', 0), weights=mutated_weights,
                                   factor_choice=fc,
                                   stock_pool=stock_pool, timing_sensitivity=timing_sens,
                                   timing_direction=timing_dir,
                                   timing_enabled=timing_enabled, ma_period=ma_period,
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

  best = state['hall_of_fame'][0]
  best_fitness = get_fitness(best)
  weights_str = ', '.join(f'{k}={v:.2f}' for k, v in best['weights'].items())
  pool_str = _format_pool(best.get('stock_pool'))
  timing_str = _format_timing(best)
  testback_logger.info(f"GA 当前最优: 夏普={best_fitness:.3f}, buy_n={best['buy_n']}, weights=[{weights_str}], pool={pool_str}, pos={best['buy_n']}{timing_str}")

  next_configs = parents + children
  return next_configs


# ========== 工具函数 ==========

def _count_holding_trading_days(start_date: date_type, end_date: date_type) -> int:
  if start_date is None or end_date is None:
    return 0
  if end_date < start_date:
    return 0
  return len(get_trading_date_span(start_date, end_date))


def _get_stock_name_map(stock_codes: set[str]) -> Dict[str, str]:
  stock_name_map: Dict[str, str] = {}
  for code in sorted(stock_codes):
    if not code:
      continue
    try:
      detail = get_stock_detail(code)
      stock_name_map[code] = detail.get('InstrumentName', '') if detail else ''
    except Exception as e:
      testback_logger.warning(f'股票名称获取失败: {code}, {e}')
      stock_name_map[code] = ''
  return stock_name_map


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
    'freeze_days_override': verify_config.get('freeze_days_override'),
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

  from core.database import get_all_stock_code_list

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

def _compute_factor_scores(backtest_datetime_list, all_stocks, weights, factor_classes):
  """加载 NPZ 并批量计算因子分数，返回 (data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices)。"""
  data = _load_runtime_npz(backtest_datetime_list)
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
  all_scores: dict[str, pd.DataFrame] = {}
  for name, f in factor_meta:
    all_scores[name] = f.calc_batch(factor_data)

  testback_logger.info(f"因子批量计算完成 ({time.time() - t0:.1f}s), {len(valid_dates)} 个调仓日")
  return data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices


def _backtest_direct(data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
                     weights, buy_n, sell_m, temperatures, freeze_days=0, verify_config=None,
                     position_multipliers=None, list_dates_map=None):
  """直接 numpy 回测，不创建 TopN 对象。"""
  from core.strategies.sizers.sizer import Sizer

  account = StockAccountMocker(cash=500_000.0, commission=2 / 1000, min_commission=5.0)
  delist_stock_info = get_delist_stock_info()

  daily_snapshots: List[Dict] = []
  prices: dict[str, float] = {}
  skipped_buy_reasons: Dict[str, int] = {}
  skipped_sell_reasons: Dict[str, int] = {}
  delist_events: List[Dict] = []

  n_stocks = len(valid_stocks)
  valid_cols = np.array([stock_indices[s] for s in valid_stocks], dtype=np.intp)

  validator = LegalityValidator(
    st_mask=data.get('st_mask'),
    stock_codes=data.get('stock_codes'),
    trade_dates=data.get('trade_dates'),
    list_dates=list_dates_map,
  )

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
      testback_logger.info(f'{stock} 已于 {delist_info.delist_date} 退市，{trade_date} 按零价值核销持仓')

  for i, dt in enumerate(valid_dates):
    if i % 500 == 0 and i > 0:
      testback_logger.info(f"回测进度: {i}/{len(valid_dates)} ({i/len(valid_dates)*100:.1f}%)")
    signal_date = dt.date() if hasattr(dt, 'date') else dt
    date_idx = date_indices[i]
    trade_idx = date_idx
    trade_date = signal_date
    trade_datetime = datetime.combine(trade_date, AFTERNOON_END)

    _write_off_delisted_positions(signal_date, trade_date)

    # 1. Rank 归一化 + 加权求和
    final_score = np.zeros(n_stocks)
    for name, scores_df in all_scores.items():
      w = weights.get(name, 0.0)
      if w == 0:
        continue
      full_row = scores_df[date_idx] if isinstance(scores_df, np.ndarray) else scores_df.iloc[date_idx].values
      row = full_row[valid_cols]
      nan_mask = np.isnan(row)
      ranks = np.zeros(n_stocks)
      valid = np.where(~nan_mask)[0]
      if len(valid) > 0:
        order = np.argsort(row[valid])[::-1]
        ranks[valid[order]] = 1.0 - np.arange(len(valid)) / len(valid)
      temp = temperatures.get(name, 1.0)
      if temp != 1.0:
        ranks = ranks ** (1.0 / temp)
      final_score += ranks * w

    # 2. 排序取 top N
    top_idx = np.argsort(-final_score)
    buy_n_stocks = [valid_stocks[i] for i in top_idx[:buy_n]]
    sell_m_stocks = [valid_stocks[i] for i in top_idx[:sell_m]]

    # verify 强制股票置顶
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
      buy_n_stocks = _prepend_forced(buy_n_stocks, buy_n)
      sell_m_stocks = _prepend_forced(sell_m_stocks, sell_m)

    # 3. 价格查询
    current_position_codes = set(account.positions.keys())
    price_universe = current_position_codes | set(sell_m_stocks) | set(buy_n_stocks)
    trade_bars = {}
    prices = {}
    for stock in price_universe:
      si = stock_indices.get(stock)
      if si is None:
        continue
      open_val = float(data['open'][trade_idx, si])
      close = float(data['close'][trade_idx, si])
      amount_val = float(data['amount'][trade_idx, si])
      if np.isnan(open_val) or open_val <= 0:
        if stock in current_position_codes and not np.isnan(close) and close > 0:
          prices[stock] = close
        elif stock in current_position_codes:
          delist_info = delist_stock_info.get(stock)
          is_delisted = delist_info is not None and trade_date > delist_info.delist_date
          if not is_delisted:
            for t in range(trade_idx - 1, -1, -1):
              past_close = float(data['close'][t, si])
              if not np.isnan(past_close) and past_close > 0:
                prices[stock] = past_close
                break
        continue
      prices[stock] = open_val
      if trade_idx == 0:
        continue
      pre_close = float(data['close'][trade_idx - 1, si])
      if np.isnan(pre_close) or pre_close <= 0:
        continue
      trade_bars[stock] = pd.Series({
        'open': open_val,
        'high': float(data['high'][trade_idx, si]),
        'low': float(data['low'][trade_idx, si]),
        'close': close,
        'preClose': pre_close,
        'issuePrice': float(data['issue_price'][si]) if 'issue_price' in data else np.nan,
        'volume': float(data['volume'][trade_idx, si]),
        'amount': amount_val,
        'suspendFlag': 0,
      })

    # 4. 卖出
    executed_sell_list: List[str] = []
    for stock in current_position_codes - set(sell_m_stocks):
      if stock not in prices:
        continue
      if freeze_days > 0:
        buy_date = account.positions[stock].get('buy_trade_date') or account.positions[stock].get('buy_date')
        if buy_date is not None:
          days_held = _count_holding_trading_days(buy_date, trade_date) - 1
          if days_held < freeze_days:
            continue
      bar = trade_bars.get(stock)
      if bar is None:
        continue
      result = validator.check_sell(stock, trade_datetime, bar=bar)
      if not result.allowed:
        skipped_sell_reasons[result.reason] = skipped_sell_reasons.get(result.reason, 0) + 1
        continue
      account.clear_stock(
        code=stock, price=prices[stock], clear_date=trade_date, clear_reason='调仓换出',
        signal_date=signal_date, price_field='open',
        signal_dividend_type='back', execution_dividend_type='none',
      )
      executed_sell_list.append(stock)

    # 5. 买入
    tradable_buy_stocks = []
    blocked_buy_details: List[Dict] = []
    for stock in buy_n_stocks:
      if stock not in prices:
        skipped_buy_reasons['missing_trade_bar'] = skipped_buy_reasons.get('missing_trade_bar', 0) + 1
        blocked_buy_details.append({
          'code': stock, 'reason': 'missing_trade_bar',
          'signal_date': signal_date.isoformat(), 'trade_date': trade_date.isoformat(),
        })
        continue
      bar = trade_bars.get(stock)
      if bar is None:
        skipped_buy_reasons['missing_bar'] = skipped_buy_reasons.get('missing_bar', 0) + 1
        continue
      result = validator.check_buy(stock, trade_datetime, bar=bar)
      if not result.allowed:
        skipped_buy_reasons[result.reason] = skipped_buy_reasons.get(result.reason, 0) + 1
        blocked_buy_details.append({
          'code': stock, 'reason': result.reason,
          'signal_date': signal_date.isoformat(), 'trade_date': trade_date.isoformat(),
        })
        continue
      tradable_buy_stocks.append(stock)

    executed_buy_records: List[Dict] = []
    if tradable_buy_stocks:
      stock_infos = [(s, prices[s]) for s in tradable_buy_stocks]
      effective_cash = account.current_cash
      if position_multipliers is not None and not np.isnan(position_multipliers[i]):
        effective_cash = account.current_cash * position_multipliers[i]
      allocations = Sizer.allocate(stock_infos, total_capital=effective_cash)
      for stock, volume in allocations.items():
        if stock in account.positions or volume <= 0:
          continue
        price = prices[stock]
        max_vol = int(account.current_cash / (price * (1 + account.commission)))
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

    assets = account.calc_assets(trade_datetime, prices)
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

  # 收盘估值
  final_signal_date = valid_dates[-1].date() if hasattr(valid_dates[-1], 'date') else valid_dates[-1]
  final_trade_date = final_signal_date
  final_datetime = datetime.combine(final_trade_date, AFTERNOON_END)
  final_assets = account.calc_assets(final_datetime, prices)
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

  stock_name_map = _get_stock_name_map(all_stock_codes)
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
  testback_logger.info(f"从文件加载 Individual_config: {args.individual_config}")

  stock_pool = individual_config.get('stock_pool')
  if stock_pool:
    if isinstance(stock_pool, list):
      stock_pool = tuple(stock_pool)
    all_stocks = [s for s in all_stocks if s.startswith(stock_pool)]
    testback_logger.info(f"stock_pool={_format_pool(stock_pool)}: {len(all_stocks)} 只")

  candidate_stock_pool = _extend_verify_stock_pool_with_historical_codes(
    all_stocks, backtest_datetime_list, verify_config)
  single_stock_pool = _resolve_single_stock_pool(candidate_stock_pool, verify_config)
  if verify_config:
    freeze_days_override = verify_config.get('freeze_days_override')
    if freeze_days_override is not None:
      individual_config['freeze_days'] = freeze_days_override
    testback_logger.info(
      f"启用 single 退市验证: {verify_config['force_stock_code']}, "
      f"候选池={len(single_stock_pool)} 只, freeze_days={individual_config.get('freeze_days', 0)}"
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

  init_stock_detail_cache(single_stock_pool)

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

  timing_multipliers = None
  timing_enabled = individual_config.get('timing_enabled', True)
  timing_sens = individual_config.get('timing_sensitivity')
  if timing_enabled and timing_sens is not None:
    from testback.market_timing import load_index_open, compute_position_multiplier, INDEX_INFO
    timing_index = individual_config.get('timing_index', 'sh000852')
    _, index_open = load_index_open(timing_index, valid_dates)
    ma_period = individual_config.get('ma_period', 60)
    direction = individual_config.get('timing_direction', 1)
    timing_multipliers = compute_position_multiplier(
      index_open, ma_period=ma_period, sensitivity=timing_sens, direction=direction)
    d_name = '顺势' if direction == 1 else '逆势'
    idx_name = INDEX_INFO.get(timing_index, timing_index)
    testback_logger.info(f"大盘择时启用: {idx_name} {d_name}, sensitivity={timing_sens}, ma={ma_period}, "
                         f"multiplier范围=[{np.nanmin(timing_multipliers):.2f}, {np.nanmax(timing_multipliers):.2f}]")

  result = _backtest_direct(
    data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
    weights=individual_config['weights'],
    buy_n=individual_config['buy_n'], sell_m=individual_config['sell_m'],
    temperatures=individual_config['temperatures'],
    freeze_days=individual_config.get('freeze_days', 0),
    verify_config=verify_config,
    position_multipliers=timing_multipliers,
    list_dates_map=list_dates_map,
  )

  testback_logger.info(f"回测完成: 总收益={result['total_return']:.2f}%")

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
    output_dir = _resolve_output_dir(args.output_dir, 'single')
    try:
      from testback.reportor import generate_single_report
      html_path = generate_single_report(report_data, output_dir)
      testback_logger.info(f"可视化报告已保存至: {html_path}")
      import webbrowser
      webbrowser.open(Path(html_path).resolve().as_uri())
    except ImportError:
      testback_logger.warning("testback.report 模块未找到，跳过可视化报告生成")
    except Exception as e:
      testback_logger.warning(f"可视化报告生成失败: {e}")

  return result


# === 文件后备 memmap 数据共享（Windows spawn，每数组独立文件 → offset=0 天然对齐） ===
_MEMMAP_DIRS: list = []

def _make_memmap_dir(parent: str) -> str:
  import tempfile
  d = tempfile.mkdtemp(prefix='ga_memmap_', dir=parent)
  _MEMMAP_DIRS.append(d)
  return d

def _arrays_to_memmap(arrays: dict, tmpdir: str) -> dict:
  """每数组独立 .bin 文件，offset=0 免对齐"""
  info = {}
  for name, arr in arrays.items():
    if not isinstance(arr, np.ndarray):
      continue
    filepath = Path(tmpdir) / f'{name}.bin'
    np.ascontiguousarray(arr).tofile(str(filepath))
    info[name] = (str(filepath), arr.shape, str(arr.dtype))
  return info

def _arrays_from_memmap(info: dict) -> dict:
  """从独立文件重建数组（offset=0，天然页对齐）"""
  result = {}
  for name, (filepath, shape, dtype_str) in info.items():
    result[name] = np.memmap(filepath, dtype=np.dtype(dtype_str), mode='r', shape=shape)
  return result

def _cleanup_memmap():
  import shutil
  for d in _MEMMAP_DIRS:
    shutil.rmtree(d, ignore_errors=True)
  _MEMMAP_DIRS.clear()

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

def _worker_evaluate(args):
  try:
    train_info, score_keys, valid_dates, date_indices, stock_indices, \
        all_stocks_list, config, index_data, list_dates_map = args

    all_arrays = _arrays_from_memmap(train_info)
    data = {k: v for k, v in all_arrays.items() if k not in score_keys}
    all_scores = {k: v for k, v in all_arrays.items() if k in score_keys}

    stock_pool = config.get('stock_pool')
    pool_stocks = [s for s in all_stocks_list if s.startswith(stock_pool)] if stock_pool else list(all_stocks_list)

    timing_multipliers = None
    timing_enabled = config.get('timing_enabled', True)
    timing_config = config.get('timing_sensitivity')
    if timing_enabled and timing_config is not None and index_data is not None:
      from testback.market_timing import compute_position_multiplier
      idx_symbol = config.get('timing_index', 'sh000852')
      idx_open = index_data.get(idx_symbol)
      if idx_open is not None:
        ma_period = config.get('ma_period', 60)
        direction = config.get('timing_direction', 1)
        timing_multipliers = compute_position_multiplier(
          idx_open, ma_period=ma_period, sensitivity=timing_config, direction=direction)

    r = _backtest_direct(
      data, all_scores, valid_dates, date_indices, pool_stocks, stock_indices,
      weights=config['weights'], buy_n=config['buy_n'], sell_m=config['sell_m'],
      temperatures=config['temperatures'], freeze_days=config.get('freeze_days', 0),
      position_multipliers=timing_multipliers, list_dates_map=list_dates_map)
    daily_rets = np.array(r['daily_returns'], dtype=float)
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    if len(daily_rets) > 1:
      mean_ret = float(np.mean(daily_rets))
      std_ret = float(np.std(daily_rets, ddof=1))
      sharpe = float(mean_ret / std_ret * np.sqrt(252.0)) if std_ret > 0 else 0.0
      cum = np.cumprod(1.0 + daily_rets / 100.0)
      peak = np.maximum.accumulate(cum)
      dd = np.min((cum - peak) / peak) * 100.0 if len(cum) > 0 else 0.0
      annualized = float(((cum[-1] / cum[0]) ** (252.0 / len(daily_rets)) - 1.0) * 100.0) if len(cum) > 1 else 0.0
    else:
      sharpe = 0.0
      dd = 0.0
      annualized = 0.0
    total_return = r['total_return']
    cleared_count = r['cleared_positions_count']
  except Exception:
    import traceback
    traceback.print_exc(file=sys.stderr)
    sharpe = -999.0
    total_return = -999.0
    cleared_count = 0
    dd = 0.0
    annualized = 0.0

  return {
    'individual_config': config,
    'total_return': total_return,
    'sharpe': sharpe,
    'annualized': annualized,
    'max_drawdown': dd,
    'cleared_positions_count': cleared_count,
    'current_positions_count': 0,
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
  names = {'60': '沪主', '00': '深主', '30': '创业板', '688': '科创板'}
  return '+'.join(names.get(p, p) for p in pool) if pool else 'all'

def _format_timing(config: dict) -> str:
  if not config.get('timing_enabled', True):
    return ', timing=OFF'
  sens = config.get('timing_sensitivity')
  if sens is None:
    return ''
  d = config.get('timing_direction', 1)
  d_str = '顺势' if d == 1 else '逆势'
  ma = config.get('ma_period', 60)
  idx_val = config.get('timing_index', 'sh000852')
  from testback.market_timing import INDEX_INFO as _IDX
  idx_name = _IDX.get(idx_val, idx_val)
  return f', timing={sens}({d_str}, ma={ma}, {idx_name})'


def _try_send_feishu(msg: str):
  try:
    from trading.lark.sender import lark_sender
    lark_sender.send_msg(msg)
  except Exception:
    pass


def _run_ga(args, mode_config, backtest_datetime_list, all_stocks, profile_name=DEFAULT_GA_PROFILE):
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
    cpu_count = os.cpu_count() or 4
    population_size = mode_config['population_size'] or 50

  mode_name = 'debug' if is_debug else 'ga'
  output_dir = _resolve_output_dir(args.output_dir, mode_name)
  factor_classes = get_profile_factor_classes(profile_name)

  ga_state = {'population': [], 'hall_of_fame': [], 'fitness_cache': {}}

  cache_path = output_dir / f'state.{os.getpid()}.json'
  _cache_fallback = output_dir / 'state.json'
  ga_cache = {}
  if cache_path.exists():
    try:
      with open(cache_path, 'r', encoding='utf-8') as f:
        ga_cache = json_mod.load(f)
    except Exception:
      pass
  if not ga_cache and _cache_fallback.exists():
    try:
      with open(_cache_fallback, 'r', encoding='utf-8') as f:
        ga_cache = json_mod.load(f)
    except Exception:
      pass
  for v in ga_cache.values():
    v.setdefault('annualized', 0.0)
    v.setdefault('max_drawdown', 0.0)
    sp = v['individual_config'].get('stock_pool')
    if isinstance(sp, list):
      v['individual_config']['stock_pool'] = tuple(sp)
  if ga_cache:
    testback_logger.info(f"加载 GA 缓存: {len(ga_cache)} 条")

  t0 = time.time()
  precompute = _compute_factor_scores(
    backtest_datetime_list, all_stocks, weights=None, factor_classes=factor_classes,
  )
  if precompute is None:
    raise RuntimeError('预计算失败：无可用 NPZ 数据')
  data, all_scores_df, valid_dates, date_indices, all_valid_stocks, stock_indices = precompute
  testback_logger.info(f"预计算完成 ({time.time() - t0:.1f}s), {len(all_valid_stocks)} 只, {len(valid_dates)} 天")

  # 主进程计算 list_dates，避免每个 worker 分配 14.5MiB boolean 数组
  train_list_dates = _compute_list_dates(data['stock_codes'], data['open'], data['trade_dates'])

  # 文件后备 memmap，不占页面文件配额
  data_tmpdir = _make_memmap_dir(str(output_dir))

  _BT_KEYS = ['open','high','low','close','volume','amount','st_mask','stock_codes','trade_dates','issue_price']
  all_scores_arr = {name: df.values for name, df in all_scores_df.items()}
  _train_arrays = {k: data[k] for k in _BT_KEYS if k in data}
  _train_arrays.update(all_scores_arr)
  _train_info = _arrays_to_memmap(_train_arrays, data_tmpdir)
  _score_keys = set(all_scores_arr.keys())

  from testback.market_timing import load_index_open, INDEX_INFO
  timing_index_symbols = list(INDEX_INFO.keys())
  index_data = {}
  for sym in timing_index_symbols:
    _, index_data[sym] = load_index_open(sym, valid_dates)

  # 测试集预计算（2020-2026，不参与优化，仅报告）
  test_start = date(2020, 1, 1)
  test_end = date(2026, 4, 30)
  test_datetime_list = [datetime.combine(d, datetime.min.time()) for d in get_trading_date_span(test_start, test_end)]
  testback_logger.info(f"测试集预计算: {test_start} - {test_end}")
  t1 = time.time()
  test_precompute = _compute_factor_scores(
    test_datetime_list, all_stocks, weights=None, factor_classes=factor_classes)
  test_data, test_scores_df, test_valid_dates, test_date_indices, test_valid_stocks, test_stock_indices = test_precompute
  test_list_dates = _compute_list_dates(test_data['stock_codes'], test_data['open'], test_data['trade_dates'])
  test_scores_arr = {name: df.values for name, df in test_scores_df.items()}
  test_tmpdir = _make_memmap_dir(str(output_dir))
  _test_arrays = {k: test_data[k] for k in _BT_KEYS if k in test_data}
  _test_arrays.update(test_scores_arr)
  _test_info = _arrays_to_memmap(_test_arrays, test_tmpdir)
  test_index_data = {}
  for sym in timing_index_symbols:
    _, test_index_data[sym] = load_index_open(sym, test_valid_dates)
  testback_logger.info(f"测试集预计算完成 ({time.time() - t1:.1f}s), {len(test_valid_stocks)} 只, {len(test_valid_dates)} 天")

  if args.warm_start:
    with open(args.warm_start, 'r', encoding='utf-8') as f:
      cache_data = json_mod.load(f)
    sorted_results = sorted(cache_data.values(), key=lambda v: v.get('sharpe', -999), reverse=True)
    next_configs = [v['individual_config'] for v in sorted_results[:2 * population_size]]
    for cfg in next_configs:
      if isinstance(cfg.get('stock_pool'), list):
        cfg['stock_pool'] = tuple(cfg['stock_pool'])
    testback_logger.info(f"热启动: 从 {args.warm_start} 加载 top {len(next_configs)} 个种子配置 (共 {len(cache_data)} 条缓存)")
  else:
    next_configs = generate_initial_configs(2 * population_size, profile_name=profile_name)

  all_results = []
  generation_results = []

  try:
    for generation in range(generations):
      generation_start_ts = time.time()
      n_configs = len(next_configs)
      testback_logger.info(f"\n{'=' * 60}")
      testback_logger.info(f"GA 第 {generation + 1}/{generations} 代{' (调试模式)' if is_debug else ''} (n={n_configs})")
      testback_logger.info(f"{'=' * 60}")

      # 分离缓存命中与待执行配置
      cached_results = []
      uncached_configs = []
      for cfg in next_configs:
        key = json_mod.dumps(_config_key(cfg), ensure_ascii=False)
        if key in ga_cache:
          cached_result = ga_cache[key]
          sp = cached_result['individual_config'].get('stock_pool')
          if isinstance(sp, list):
            cached_result['individual_config']['stock_pool'] = tuple(sp)
          cached_results.append(cached_result)
        else:
          uncached_configs.append(cfg)

      if cached_results:
        testback_logger.info(f"缓存命中: {len(cached_results)}/{n_configs} 个配置跳过回测")

      worker_args = [
        (_train_info, _score_keys, valid_dates, date_indices, stock_indices,
         all_valid_stocks, config, index_data, train_list_dates)
        for config in uncached_configs
      ]

      results_list = list(cached_results)
      if is_debug:
        for idx, wargs in enumerate(worker_args):
          cfg = uncached_configs[idx]
          testback_logger.info(f"  回测 {idx + 1}/{len(uncached_configs)}: {cfg['weights']}, pool={_format_pool(cfg.get('stock_pool'))}, pos={cfg['buy_n']}")
          result = _worker_evaluate(wargs)
          results_list.append(result)
          key = json_mod.dumps(_config_key(result['individual_config']), ensure_ascii=False)
          ga_cache[key] = result
          testback_logger.info(f"    夏普={result['sharpe']:.3f}, 总收益={result['total_return']:.2f}%")
      else:
        from multiprocessing import Pool
        n_workers = min(4, max(1, len(worker_args)))
        testback_logger.info(f"并行 workers: {n_workers} (共 {len(worker_args)} 个任务)")
        with Pool(processes=n_workers) as pool:
          for result in pool.imap_unordered(_worker_evaluate, worker_args):
            results_list.append(result)
            key = json_mod.dumps(_config_key(result['individual_config']), ensure_ascii=False)
            ga_cache[key] = result
      if worker_args:
        with open(cache_path, 'w', encoding='utf-8') as f:
          json_mod.dump(ga_cache, f, ensure_ascii=False)
        _out_tmp = str(output_dir / 'state.json') + '.tmp'
        with open(_out_tmp, 'w', encoding='utf-8') as f:
          json_mod.dump(ga_cache, f, ensure_ascii=False)
        os.replace(_out_tmp, output_dir / 'state.json')

      if not is_debug:
        # 训练集统计
        sharpes = [r['sharpe'] for r in results_list]
        best_idx = max(range(len(results_list)), key=lambda i: sharpes[i])
        best = results_list[best_idx]
        best_cfg = best['individual_config']
        best_m = {'sharpe': best['sharpe'], 'annualized': best['annualized'], 'max_drawdown': best['max_drawdown']}
        avg_sharpe = sum(sharpes) / len(sharpes)
        avg_ann = sum(r['annualized'] for r in results_list) / len(results_list)
        avg_dd = sum(r['max_drawdown'] for r in results_list) / len(results_list)

        # HS300 训练基线
        hs300_vals = index_data['sh000300'].astype(float)
        hs300_daily = np.diff(hs300_vals) / hs300_vals[:-1] * 100.0
        hs300_daily = hs300_daily[np.isfinite(hs300_daily)]
        hs300_m = _compute_metrics_simple(list(hs300_daily))

        gen_time = time.time() - generation_start_ts
        w_str = ', '.join(f'{k}={v:.2f}' for k, v in best_cfg['weights'].items())
        timing_str = _format_timing(best_cfg)

        # 测试集评估：第1代 + 每10代，不参与优化
        report_gen = ((generation + 1) % 10 == 0)
        if report_gen:
          test_worker_args = [
            (_test_info, _score_keys, test_valid_dates, test_date_indices, test_stock_indices,
             test_valid_stocks, config, test_index_data, test_list_dates)
            for config in next_configs
          ]
          test_results = [_worker_evaluate(a) for a in test_worker_args]
          test_sharpes = [r['sharpe'] for r in test_results]
          train_best_test_m = {'sharpe': test_results[best_idx]['sharpe'], 'annualized': test_results[best_idx]['annualized'], 'max_drawdown': test_results[best_idx]['max_drawdown']}
          test_best_idx = max(range(len(test_results)), key=lambda i: test_sharpes[i])
          test_best_m = {'sharpe': test_results[test_best_idx]['sharpe'], 'annualized': test_results[test_best_idx]['annualized'], 'max_drawdown': test_results[test_best_idx]['max_drawdown']}
          test_avg_sharpe = sum(test_sharpes) / len(test_sharpes)
          test_avg_ann = sum(r['annualized'] for r in test_results) / len(test_results)
          test_avg_dd = sum(r['max_drawdown'] for r in test_results) / len(test_results)

          testback_logger.info(
            f"  [训练] 最优: 夏普={best_m['sharpe']:.3f}, 年化={best_m['annualized']:.1f}%, 回撤={best_m['max_drawdown']:.1f}% | "
            f"参数: [{w_str}], pool={_format_pool(best_cfg.get('stock_pool'))}, pos={best_cfg['buy_n']}{timing_str}")
          testback_logger.info(
            f"  [训练] 均值: 夏普={avg_sharpe:.3f}, 年化={avg_ann:.1f}%, 回撤={avg_dd:.1f}%")
          testback_logger.info(
            f"  [训练] HS300: 夏普={hs300_m['sharpe']:.3f}, 年化={hs300_m['annualized']:.1f}%, 回撤={hs300_m['max_drawdown']:.1f}%")
          testback_logger.info(
            f"  [测试] 训练最优: 夏普={train_best_test_m['sharpe']:.3f}, 年化={train_best_test_m['annualized']:.1f}%, 回撤={train_best_test_m['max_drawdown']:.1f}%")
          testback_logger.info(
            f"  [测试] 全体最优: 夏普={test_best_m['sharpe']:.3f}, 年化={test_best_m['annualized']:.1f}%, 回撤={test_best_m['max_drawdown']:.1f}%")
          testback_logger.info(
            f"  [测试] 均值: 夏普={test_avg_sharpe:.3f}, 年化={test_avg_ann:.1f}%, 回撤={test_avg_dd:.1f}% | 耗时={gen_time:.0f}s")

          _try_send_feishu(
            f"GA 第 {generation + 1}/{generations} 代 ({gen_time:.0f}s)\n"
            f"[训练] 最优夏普={best_m['sharpe']:.3f} | 均值夏普={avg_sharpe:.3f}\n"
            f"[测试] 训练最优夏普={train_best_test_m['sharpe']:.3f} | 全体最优夏普={test_best_m['sharpe']:.3f} | 均值夏普={test_avg_sharpe:.3f}\n"
            f"参数: [{w_str}], {_format_pool(best_cfg.get('stock_pool'))}, pos={best_cfg['buy_n']}{timing_str}")
        else:
          testback_logger.info(
            f"  [训练] 最优: 夏普={best_m['sharpe']:.3f}, 年化={best_m['annualized']:.1f}%, 回撤={best_m['max_drawdown']:.1f}% | "
            f"均值: 夏普={avg_sharpe:.3f}, 年化={avg_ann:.1f}%, 回撤={avg_dd:.1f}% | "
            f"HS300: 夏普={hs300_m['sharpe']:.3f} | {gen_time:.0f}s | "
            f"[{w_str}], {_format_pool(best_cfg.get('stock_pool'))}, pos={best_cfg['buy_n']}{timing_str}")

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
          'all_individuals': results_list,
        }
        generation_results.append(generation_stats)

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

      next_configs = ga_optimizer(results_list, state=ga_state, population_size=population_size,
                                  hall_of_fame_size=population_size, profile_name=profile_name)

  finally:
    _cleanup_memmap()

  if not is_debug:
    all_individual_configs = []
    for gen_stat in generation_results:
      for ind in gen_stat['all_individuals']:
        all_individual_configs.append({
          'generation': ind['generation'],
          'individual_config': ind['individual_config'],
          'fitness': ind['fitness'],
          'total_return': ind['total_return'],
          'cleared_positions_count': ind['cleared_positions_count'],
          'current_positions_count': ind['current_positions_count'],
        })

    with open(output_dir / 'all_individuals.json', 'w', encoding='utf-8') as f:
      json_mod.dump(all_individual_configs, f, indent=2, ensure_ascii=False)
    testback_logger.info(f"已保存所有个体配置: {output_dir / 'all_individuals.json'}")

    best_config = ga_state['hall_of_fame'][0]
    key = _config_key(best_config)
    best_fitness = ga_state['fitness_cache'][key]

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
    testback_logger.info(f"最优夏普率: {best_fitness:.3f}, 参数: [{w_str}], pool={_format_pool(best_cfg.get('stock_pool'))}, pos={best_cfg['buy_n']}{timing_str}")
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

  # 最终缓存写入 output_dir
  try:
    with open(output_dir / 'state.json', 'w', encoding='utf-8') as f:
      json_mod.dump(ga_cache, f, ensure_ascii=False)
  except Exception:
    pass

  testback_logger.info(f"\n{'调试' if is_debug else 'GA'}模式执行完成，结果目录: {output_dir}")
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
  parser.add_argument('--profile', type=str, default=None)
  args = parser.parse_args()

  filtered_stocks = list(allow_buy_stock_code_list())
  profile_name = args.profile or DEFAULT_GA_PROFILE
  mode_configs = get_mode_configs(profile_name)
  mode_config = mode_configs[args.mode].copy()

  loguru_logger.remove()
  loguru_logger.add(sys.stderr, level=mode_config['log_level'])

  testback_logger.info(f"运行模式: {args.mode} - {mode_config['desc']}")
  testback_logger.info(f"回测周期: {mode_config['period_span']} 天")
  testback_logger.info(f"股票池 (allow_buy): {len(filtered_stocks)} 只")

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

  init_stock_detail_cache(filtered_stocks)

  if args.mode == 'single':
    result = run_single_mode(args, mode_config, backtest_datetime_list, filtered_stocks)
  else:
    result = _run_ga(args, mode_config, backtest_datetime_list, filtered_stocks, profile_name=profile_name)

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
