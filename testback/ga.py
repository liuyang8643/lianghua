import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from joblib import Parallel, delayed, parallel_backend

from core import (
  cleanup_shared_cache,
  get_market_data_batch,
  get_stock_detail,
  init_stock_detail_cache,
)
from core.database import allow_buy_stock_code_list, init_market_data_range
from core.database.history import get_history_data
from core.database.delist import get_delist_stock_info
from core.strategies.top_n import compute_topn_range, make_topn_range_cache_key
from utils.shared_memory import SharedMemoryCache
from utils.stock.time import get_latest_trading_time, get_next_trading_day, get_target_period_backward, get_trading_date_span
from utils.stock.info import evaluate_orderability
from utils.windows_awake import keep_windows_awake
from testback.account import StockAccountMocker
from testback.ga_config import (
  DEFAULT_GA_PROFILE,
  build_individual_config,
  generate_initial_configs,
  get_mode_configs,
  get_profile_factor_classes,
  get_profile_factor_names,
  get_profile_fixed_weights,
  get_profile_metadata,
  get_profile_preload_range,
  get_profile_weight_search_spaces,
  resolve_profile_name,
  sample_position_count,
  sample_weights,
)
from testback.metrics import compute_hs300_cumulative_returns, compute_strategy_metrics

os.environ['LOKY_PICKLER'] = 'pickle'  # 使用更快的pickle

from datetime import date, date as date_type, datetime, timedelta
from testback.logger import testback_logger
from core.strategies import TopN


# ========== GA 核心函数 ==========
def _sample_topn_window(total_days: int, window_days: int) -> tuple[int, int]:
  import random

  if total_days <= 0:
    return 0, 0

  window_size = min(total_days, max(1, window_days))
  if total_days == window_size:
    return 0, window_size

  offset = random.randint(0, total_days - window_size)
  return offset, window_size


def _individual_config_to_key(individual_config: dict) -> str:
  import json

  return json.dumps(individual_config, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

# GA 状态（全局）
_ga_state = {
  'population': [],  # 当前种群
  'hall_of_fame': [],  # 历史最优池
  'fitness_cache': {},  # 适应度缓存
}

def ga_optimizer(
    results,
    population_size: int = 24,
    hall_of_fame_size: int = 24,
    profile_name: str = DEFAULT_GA_PROFILE,
) -> list[dict]:
  """
  GA 优化器：根据回测结果生成下一代Individual_config

  Args:
    results: 回测结果，每个元素包含 {'individual_config': dict, 'total_return': float, ...}
    population_size: 种群大小
    hall_of_fame_size: 历史最优池大小

  Returns:
    下一代Individual_config列表
  """
  import random

  results_list = list(results) if not isinstance(results, list) else results

  for r in results_list:
    key = _individual_config_to_key(r['individual_config'])
    _ga_state['fitness_cache'][key] = r['total_return']

  testback_logger.info(f"GA 本轮有效结果: {len(results_list)} 个")

  if not _ga_state['population']:
    _ga_state['population'] = [r['individual_config'] for r in results_list][:population_size]
    testback_logger.info(f"GA 初始化种群: {len(_ga_state['population'])} 个个体")

  def get_fitness(config):
    key = _individual_config_to_key(config)
    return _ga_state['fitness_cache'][key]

  population_with_fitness = [(ind, get_fitness(ind)) for ind in _ga_state['population']]
  population_with_fitness.sort(key=lambda x: x[1], reverse=True)
  parents = [ind for ind, _ in population_with_fitness[:population_size]]

  all_individuals = _ga_state['hall_of_fame'] + parents
  unique_dict = {}
  for ind in all_individuals:
    key = _individual_config_to_key(ind)
    fitness = get_fitness(ind)
    if key not in unique_dict or fitness > unique_dict[key][1]:
      unique_dict[key] = (ind, fitness)
  sorted_hof = sorted(unique_dict.values(), key=lambda x: x[1], reverse=True)
  _ga_state['hall_of_fame'] = [ind for ind, _ in sorted_hof[:hall_of_fame_size]]

  has_weight_search = get_profile_weight_search_spaces(profile_name) is not None

  def crossover_config(p1, p2):
    position_count = p1['buy_n'] if random.random() < 0.5 else p2['buy_n']
    freeze_days = p1.get('freeze_days', 0)
    crossed_weights = None
    if has_weight_search:
      crossed_weights = {}
      for k in p1['weights']:
        crossed_weights[k] = p1['weights'][k] if random.random() < 0.5 else p2['weights'][k]
    return build_individual_config(position_count, freeze_days=freeze_days, weights=crossed_weights, profile_name=profile_name)

  def mutate_config(config, mutation_rate: float = 0.2):
    position_count = config['buy_n']
    if random.random() < mutation_rate:
      position_count = sample_position_count(current_value=position_count, profile_name=profile_name)
    mutated_weights = None
    if has_weight_search:
      mutated_weights = sample_weights(current_weights=config['weights'], mutation_rate=mutation_rate, profile_name=profile_name)
    return build_individual_config(position_count, freeze_days=config.get('freeze_days', 0), weights=mutated_weights, profile_name=profile_name)

  children = []
  while len(children) < population_size:
    if len(parents) == 1:
      child = mutate_config(parents[0], mutation_rate=0.2)
    else:
      p1, p2 = random.sample(parents, 2)
      child = crossover_config(p1, p2)
      child = mutate_config(child, mutation_rate=0.2)
    children.append(child)

  _ga_state['population'] = parents + children

  best = _ga_state['hall_of_fame'][0]
  best_fitness = get_fitness(best)
  weights_str = ', '.join(f'{k}={v:.1f}' for k, v in best['weights'].items())
  testback_logger.info(f"GA 当前最优收益率: {best_fitness:.2f}%, buy_n={best['buy_n']}, weights=[{weights_str}]")

  next_configs = parents + children + _ga_state['hall_of_fame']
  return next_configs[:3 * population_size]

# 全局缓存
testback_cache = SharedMemoryCache[list[TopN]]('testback_cache', compress_level=6)


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


def _signal_to_trade_date(signal_date: date_type) -> date_type:
  return get_next_trading_day(signal_date)


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


def _apply_single_verify_override(topn_list, verify_config: Dict[str, Any]):
  if not verify_config:
    return topn_list

  force_stock_code = verify_config['force_stock_code']
  candidate_stock_codes = [
    code for code in verify_config.get('candidate_stock_codes', [])
    if code != force_stock_code
  ]

  class _SingleVerifyTopNProxy:
    def __init__(self, base_topn, forced_code: str, candidate_codes: List[str]):
      self._base_topn = base_topn
      self.base_date = base_topn.base_date
      self._forced_code = forced_code
      self._candidate_codes = candidate_codes

    def __getattr__(self, item):
      return getattr(self._base_topn, item)

    def get_ordered_stocks(self, n: int, weights: Dict[str, float] = None,
                           temperatures: Dict[str, float] = None, norm_method: str = 'softmax') -> List[str]:
      base_list = list(self._base_topn.get_ordered_stocks(n, weights, temperatures, norm_method))
      ordered = []
      seen = set()
      for code in [self._forced_code, *self._candidate_codes, *base_list]:
        if not code or code in seen:
          continue
        seen.add(code)
        ordered.append(code)
      return ordered[:n] if n > 0 else []

  return [
    _SingleVerifyTopNProxy(topn, force_stock_code, candidate_stock_codes)
    for topn in topn_list
  ]


def _backtest_with_config(topn_list, weights, buy_n, sell_m, temperatures, freeze_days: int = 0):
  """ 独立回测函数，计算给定配置的收益

  语义：
  - TopN.base_date / 裸 date 代表 signal_date（T-1）
  - 回测执行日为 trade_date（T）
  - 信号层使用 back 复权（因子计算需要连续价格序列）
  - 执行层使用不复权（避免后复权价格异常导致仓位失真）
  - 成交价使用 trade_date 的不复权 open
  """
  from core.strategies.sizers.sizer import Sizer

  account = StockAccountMocker(
    cash=500_000.0,
    commission=2 / 1000,
    min_commission=5.0,
  )
  delist_stock_info = get_delist_stock_info()

  daily_snapshots: List[Dict] = []
  prices: dict[str, float] = {}
  skipped_buy_reasons: Dict[str, int] = {}
  skipped_sell_reasons: Dict[str, int] = {}
  delist_events: List[Dict] = []

  def _raise_if_trade_bars_stale(price_codes: set[str], trade_dt: datetime, signal_dt: date_type):
    if not price_codes:
      return

    stale_probe = get_history_data(
      list(price_codes),
      1,
      get_latest_trading_time(trade_dt),
      period='1d',
      dividend_type='none',
    )
    latest_dates = []
    missing_codes = []
    for code, data in stale_probe.items():
      if data is None or data.empty:
        missing_codes.append(code)
        continue
      latest_dates.append(datetime.fromtimestamp(int(data.iloc[-1]['time']) / 1000).date())

    if latest_dates:
      latest_available = max(latest_dates)
      if latest_available < trade_dt.date():
        raise RuntimeError(
          '执行层行情数据已过期：'
          f'signal_date={signal_dt.isoformat()}, '
          f'trade_date={trade_dt.date().isoformat()}, '
          f'latest_available_trade_bar={latest_available.isoformat()}, '
          f'price_universe={len(price_codes)} 只。'
          '请先更新 QMT 本地日线数据，或缩短回测区间。'
        )

    if missing_codes and len(missing_codes) == len(price_codes):
      raise RuntimeError(
        '执行层行情数据缺失：'
        f'signal_date={signal_dt.isoformat()}, '
        f'trade_date={trade_dt.date().isoformat()}, '
        f'price_universe={len(price_codes)} 只全部无可用日线。'
        '请检查 QMT 本地数据是否完整。'
      )

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
      clear_reason = '退市归零'

      account.write_off_stock(
        code=stock,
        write_off_date=trade_date,
        write_off_reason=clear_reason,
        signal_date=signal_date,
        price_field='delist_zero',
        signal_dividend_type='back',
        execution_dividend_type='none',
      )
      delist_events.append({
        'code': stock,
        'delist_date': delist_info.delist_date,
        'clear_signal_date': signal_date,
        'clear_trade_date': trade_date,
        'buy_trade_date': buy_trade_date,
        'holding_days': _count_holding_trading_days(buy_trade_date, trade_date),
        'volume': position.get('volume', 0),
        'cost': cost,
        'income': -cost,
        'income_pct': -100.0 if cost else 0.0,
        'clear_reason': clear_reason,
      })
      testback_logger.info(
        f'{stock} 已于 {delist_info.delist_date} 退市，{trade_date} 按零价值核销持仓'
      )

  for topn in topn_list:
    if not hasattr(topn, 'base_date'):
      raise ValueError('topn_list 必须为 TopN 实例列表（与实盘共用 compute_topn_range / 预计算窗口）')
    signal_date = topn.base_date.date()

    trade_date = _signal_to_trade_date(signal_date)
    trade_datetime = datetime.combine(trade_date, datetime.min.time())

    _write_off_delisted_positions(signal_date, trade_date)

    sell_m_stocks = topn.get_ordered_stocks(
      n=sell_m,
      weights=weights,
      temperatures=temperatures,
      norm_method='rank',
    )
    buy_n_stocks = topn.get_ordered_stocks(
      n=buy_n,
      weights=weights,
      temperatures=temperatures,
      norm_method='rank',
    )

    current_position_codes = set(account.positions.keys())
    price_universe = current_position_codes | set(sell_m_stocks) | set(buy_n_stocks)
    trade_bar_data = get_market_data_batch(
      list(price_universe),
      1,
      trade_datetime,
      period='1d',
      dividend_type='none',
      strict_trade_date=True,
    )
    trade_bars = {
      code: (data.iloc[-1] if data is not None and not data.empty else None)
      for code, data in trade_bar_data.items()
    }
    if price_universe and not any(bar is not None for bar in trade_bars.values()):
      _raise_if_trade_bars_stale(price_universe, trade_datetime, signal_date)

    prices = {}
    for stock, bar in trade_bars.items():
      if bar is not None and bar.get('open') is not None:
        open_price = float(bar['open'])
        if open_price > 0:
          prices[stock] = open_price

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

      orderability = evaluate_orderability('sell', stock, trade_datetime, bar=trade_bars.get(stock), dividend_type='none')
      if not orderability['allowed']:
        skipped_sell_reasons[orderability['reason']] = skipped_sell_reasons.get(orderability['reason'], 0) + 1
        continue

      account.clear_stock(
        code=stock,
        price=prices[stock],
        clear_date=trade_date,
        clear_reason='调仓换出',
        signal_date=signal_date,
        price_field='open',
        signal_dividend_type='back',
        execution_dividend_type='none',
      )
      executed_sell_list.append(stock)

    tradable_buy_stocks = []
    blocked_buy_details: List[Dict] = []
    for stock in buy_n_stocks:
      if stock not in prices:
        skipped_buy_reasons['missing_trade_bar'] = skipped_buy_reasons.get('missing_trade_bar', 0) + 1
        blocked_buy_details.append({
          'code': stock,
          'reason': 'missing_trade_bar',
          'signal_date': signal_date.isoformat(),
          'trade_date': trade_date.isoformat(),
        })
        continue

      orderability = evaluate_orderability('buy', stock, trade_datetime, bar=trade_bars.get(stock), dividend_type='none')
      if not orderability['allowed']:
        skipped_buy_reasons[orderability['reason']] = skipped_buy_reasons.get(orderability['reason'], 0) + 1
        blocked_buy_details.append({
          'code': stock,
          'reason': orderability['reason'],
          'signal_date': signal_date.isoformat(),
          'trade_date': trade_date.isoformat(),
        })
        continue
      tradable_buy_stocks.append(stock)

    executed_buy_records: List[Dict] = []
    if tradable_buy_stocks:
      stock_infos = [(s, prices[s]) for s in tradable_buy_stocks]
      allocations = Sizer.allocate(
        stock_infos,
        total_capital=account.calc_assets(trade_datetime, prices)['total_asset'],
      )
      for stock, volume in allocations.items():
        if stock in account.positions:
          continue
        if volume <= 0:
          continue

        price = prices[stock]
        while volume > 0:
          cost = volume * price
          commission = account.calc_commission(cost)
          if cost + commission <= account.current_cash:
            break
          volume -= 100

        if volume <= 0:
          skipped_buy_reasons['insufficient_cash'] = skipped_buy_reasons.get('insufficient_cash', 0) + 1
          blocked_buy_details.append({
            'code': stock,
            'reason': 'insufficient_cash',
            'signal_date': signal_date.isoformat(),
            'trade_date': trade_date.isoformat(),
          })
          continue

        if not account.buy_stock(
          code=stock,
          volume=volume,
          price=price,
          buy_date=trade_date,
          signal_date=signal_date,
          price_field='open',
          signal_dividend_type='back',
          execution_dividend_type='none',
        ):
          skipped_buy_reasons['insufficient_cash'] = skipped_buy_reasons.get('insufficient_cash', 0) + 1
          blocked_buy_details.append({
            'code': stock,
            'reason': 'insufficient_cash',
            'signal_date': signal_date.isoformat(),
            'trade_date': trade_date.isoformat(),
          })
          continue
        executed_buy_records.append({
          'code': stock,
          'signal_date': signal_date.isoformat(),
          'trade_date': trade_date.isoformat(),
          'price': price,
          'price_field': 'open',
          'volume': volume,
          'signal_dividend_type': 'back',
          'execution_dividend_type': 'none',
        })

    assets = account.calc_assets(trade_datetime, prices)
    prev_total_asset = daily_snapshots[-1]['total_asset'] if daily_snapshots else account.init_cash
    daily_ret = (assets['total_asset'] - prev_total_asset) / prev_total_asset * 100 if prev_total_asset else 0.0
    daily_snapshots.append({
      'date': trade_date.strftime('%Y-%m-%d') if isinstance(trade_date, date_type) else str(trade_date),
      'signal_date': signal_date.strftime('%Y-%m-%d') if isinstance(signal_date, date_type) else str(signal_date),
      'trade_date': trade_date.strftime('%Y-%m-%d') if isinstance(trade_date, date_type) else str(trade_date),
      'signal_dividend_type': 'back',
      'execution_dividend_type': 'none',
      'price_field': 'open',
      'cash': assets['cash'],
      'market_value': assets['market_value'],
      'total_asset': assets['total_asset'],
      'daily_return_pct': daily_ret,
      'cumulative_return_pct': 0.0,
      'sell_m_list': sell_m_stocks,
      'buy_n_list': buy_n_stocks,
      'buy_n_diff_list': [stock for stock in buy_n_stocks if stock not in sell_m_stocks],
      'executed_sell_list': executed_sell_list,
      'executed_buy_list': [record['code'] for record in executed_buy_records],
      'executed_buy_details': executed_buy_records,
      'blocked_buy_details': blocked_buy_details,
    })

  last_topn = topn_list[-1]
  final_signal_date = last_topn.base_date.date()
  final_trade_date = _signal_to_trade_date(final_signal_date)
  final_datetime = datetime.combine(final_trade_date, datetime.min.time())

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

  positions = account.calc_position_values(final_datetime, prices)
  for position in positions:
    position['holding_days'] = _count_holding_trading_days(
      position.get('buy_trade_date') or position.get('buy_date'),
      final_trade_date,
    )

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

  delist_count = len(delist_events)

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
    'delist_count': delist_count,
    'skipped_buy_reasons': skipped_buy_reasons,
    'skipped_sell_reasons': skipped_sell_reasons,
    'final_asset': final_assets['total_asset'],
  }


def _wrap_process_worker(individual_config: dict, mem_offset: int, mem_count: int, topn_cache_key: str = 'topn_window'):
  """独立进程计算最终收益：从共享内存读取当前窗口的 TopN 切片。"""
  import time
  # 生成 worker 标识
  import os
  worker_id = f"[Worker-{os.getpid()}]"
  weights = individual_config['weights']
  buy_n = individual_config['buy_n']
  sell_m = individual_config['sell_m']
  temperatures = individual_config['temperatures']
  freeze_days = individual_config.get('freeze_days', 0)

  testback_logger.info(f"{worker_id} 🚀 开始回测 | offset={mem_offset}, count={mem_count}, config={individual_config}")

  try:
    t_load = time.time()
    topn_list = testback_cache.get(topn_cache_key)
    testback_logger.debug(f"{worker_id} ⏱️  从共享内存读取窗口数据耗时: {time.time() - t_load:.2f}秒")

    if not topn_list or len(topn_list) == 0:
      testback_logger.error(f"{worker_id} ❌ 共享窗口数据为空: key={topn_cache_key}")
      return None

    testback_logger.info(
      f"{worker_id} 📊 窗口数据加载完成 | key={topn_cache_key}, "
      f"offset={mem_offset}, count={mem_count}, 实际={len(topn_list)}天, "
      f"周期: {topn_list[0].base_date} ~ {topn_list[-1].base_date}"
    )

    result = _backtest_with_config(topn_list, weights, buy_n, sell_m, temperatures, freeze_days=freeze_days)

    return {
      'individual_config': individual_config,
      'init_cash': 500_000.0,
      'final_cash': 0,
      'final_market_value': 0,
      'final_total_asset': 0,
      'total_return': result['total_return'],
      'target2_total_return': result['total_return'],
      'target2_cleared_positions_count': result['cleared_positions_count'],
      'target2_current_positions_count': result['current_positions_count'],
      'cleared_positions_count': result['cleared_positions_count'],
      'current_positions_count': result['current_positions_count'],
      'daily_returns': result['daily_returns'],
    }

  except Exception as e:
    testback_logger.error(f"回测时出错: {e}")
    import traceback
    testback_logger.error(traceback.format_exc())
    return None

# ========== 数据加载（所有模式共用）==========

def _load_shared_data(backtest_datetime_list, all_stocks, max_hist_days: int = 0):
  """按回测窗口预加载市场数据到共享内存（不含 TopN）"""
  if not backtest_datetime_list:
    return

  # 预加载股票详情到共享内存缓存
  init_stock_detail_cache(all_stocks)
  signal_start = backtest_datetime_list[0]
  signal_end = backtest_datetime_list[-1]
  preload_start = (
    get_target_period_backward(signal_start, '1d', max_hist_days)
    if max_hist_days > 0 else signal_start
  )

  init_market_data_range(
    all_stocks,
    preload_start,
    signal_end,
    '1d',
  )
  testback_logger.debug(
    f"市场数据窗口预加载完成，共 {len(all_stocks)} 只股票，"
    f"范围 {preload_start.date()} ~ {signal_end.date()}"
  )


def _compute_topn_for_range(
    backtest_datetime_list,
    all_stocks,
    weights=None,
    factor_classes=None,
    profile_name: str = DEFAULT_GA_PROFILE,
):
  """为指定日期范围计算 profile 定义的 TopN 实例。"""
  factor_classes = factor_classes or get_profile_factor_classes(profile_name)
  topn_weights = weights or get_profile_fixed_weights(profile_name)
  return compute_topn_range(
    backtest_datetime_list,
    all_stocks,
    weights=topn_weights,
    factor_classes=factor_classes,
  )


def _prepare_shared_topn(backtest_datetime_list, all_stocks, profile_name: str = DEFAULT_GA_PROFILE):
  """预计算 TopN，供主进程按代切片后写入共享内存窗口。"""
  factor_classes = get_profile_factor_classes(profile_name)
  topn_weights = get_profile_fixed_weights(profile_name)
  ordered_topns = _compute_topn_for_range(
    backtest_datetime_list,
    all_stocks,
    weights=topn_weights,
    factor_classes=factor_classes,
  )
  cache_key = make_topn_range_cache_key(
    backtest_datetime_list,
    all_stocks,
    weights=topn_weights,
    factor_classes=factor_classes,
  )
  if not ordered_topns:
    raise RuntimeError('TopN 预计算结果为空')
  testback_logger.info(
    f"TopN 预计算完成: key={cache_key}, 天数={len(ordered_topns)}, 股票数={len(all_stocks)}"
  )
  return ordered_topns


def _put_topn_window_slice(ordered_topns, mem_offset: int, mem_count: int, topn_cache_key: str = 'topn_window'):
  topn_window = ordered_topns[mem_offset:mem_offset + mem_count]
  if not topn_window:
    raise RuntimeError(f'TopN 窗口为空: offset={mem_offset}, count={mem_count}')
  if not testback_cache.put(topn_cache_key, topn_window):
    raise RuntimeError(f'共享内存写入 {topn_cache_key} 失败')
  return topn_window


def _make_run_cache_key(prefix: str, run_id: str, suffix: str) -> str:
  return f'{prefix}_{run_id}_{suffix}'


def _make_topn_window_cache_key(run_id: str, generation: int) -> str:
  return _make_run_cache_key('topn_window', run_id, f'gen_{generation}')


def _make_output_dir(mode: str) -> Path:
  """创建模式对应的结果输出目录"""
  results_dir = Path('results')
  results_dir.mkdir(exist_ok=True)
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  output_dir = results_dir / f'{mode}_{timestamp}'
  output_dir.mkdir(exist_ok=True)
  return output_dir


def _resolve_output_dir(output_dir_arg: str | None, mode: str) -> Path:
  """解析并确保结果输出目录存在"""
  output_dir = Path(output_dir_arg) if output_dir_arg else _make_output_dir(mode)
  output_dir.mkdir(parents=True, exist_ok=True)
  return output_dir


# ========== 模式执行函数 ==========

def run_single_mode(args, mode_config, backtest_datetime_list, all_stocks):
  """单次回测模式：加载指定配置文件，执行一次回测并生成可视化报告"""
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

  candidate_stock_pool = _extend_verify_stock_pool_with_historical_codes(
    all_stocks,
    backtest_datetime_list,
    verify_config,
  )
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

  # 根据 weights 中的因子名自动查找因子类
  import core.factors as _all_factors
  config_factor_classes = []
  for fname in individual_config['weights']:
    cls = getattr(_all_factors, fname, None)
    if cls is None:
      testback_logger.error(f"因子类 {fname} 不存在，请检查配置文件")
      sys.exit(1)
    config_factor_classes.append(cls)

  ordered_topNs = _compute_topn_for_range(
    backtest_datetime_list,
    single_stock_pool,
    weights=individual_config['weights'],
    factor_classes=config_factor_classes,
  )
  ordered_topNs = _apply_single_verify_override(ordered_topNs, verify_config)
  signal_dates = [topn.base_date.date() for topn in ordered_topNs]
  trade_dates = [_signal_to_trade_date(signal_date) for signal_date in signal_dates]
  signal_date_strs = [d.strftime('%Y-%m-%d') for d in signal_dates]
  trade_date_strs = [d.strftime('%Y-%m-%d') for d in trade_dates]
  testback_logger.info(
    f"回测信号范围: {signal_dates[0]} ~ {signal_dates[-1]}，"
    f"执行范围: {trade_dates[0]} ~ {trade_dates[-1]}，共 {len(ordered_topNs)} 个调仓日"
  )

  # 直接在主进程运行回测
  result = _backtest_with_config(
    topn_list=ordered_topNs,
    weights=individual_config['weights'],
    buy_n=individual_config['buy_n'],
    sell_m=individual_config['sell_m'],
    temperatures=individual_config['temperatures'],
    freeze_days=individual_config.get('freeze_days', 0),
  )

  testback_logger.info(f"回测完成: 总收益={result['total_return']:.2f}%")

  # 构造 report_data
  metrics = compute_strategy_metrics(
    cumulative_returns_pct=result.get('cumulative_returns', []),
    trade_dates=trade_date_strs,
    init_cash=500_000.0,
    final_asset=result.get('final_asset', 500_000.0),
    trade_log=result.get('trade_log', []),
  )
  hs300_returns = compute_hs300_cumulative_returns(trade_date_strs)

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
      'signal_timing': 'T-1',
      'trade_timing': 'T open',
      'signal_dividend_type': 'back',
      'execution_dividend_type': 'none',
      'price_field': 'open',
    },
    'period': {
      'signal_start': signal_date_strs[0],
      'signal_end': signal_date_strs[-1],
      'trade_start': trade_date_strs[0],
      'trade_end': trade_date_strs[-1],
      'start': trade_date_strs[0],
      'end': trade_date_strs[-1],
    },
  }

  # 生成可视化报告
  if mode_config['save_charts']:
    output_dir = _resolve_output_dir(args.output_dir, 'single')
    try:
      from testback.reportor import generate_single_report
      html_path = generate_single_report(report_data, output_dir)
      testback_logger.info(f"可视化报告已保存至: {html_path}")
      # 自动用浏览器打开报告
      import webbrowser, pathlib
      abs_path = pathlib.Path(html_path).resolve()
      # Windows file URL 格式: file:///D:/path/to/file.html
      file_url = abs_path.as_uri()
      webbrowser.open(file_url)
    except ImportError:
      testback_logger.warning("testback.report 模块未找到，跳过可视化报告生成")
    except Exception as e:
      testback_logger.warning(f"可视化报告生成失败: {e}")

  return result


def run_debug_mode(args, mode_config, ordered_topns, profile_name: str = DEFAULT_GA_PROFILE):
  """调试模式：使用极小种群和短周期，验证 GA 流程。"""
  cpu_count = os.cpu_count() or 4
  population_size = mode_config['population_size']
  generations = mode_config['generations']
  window_days = mode_config.get('window_days', mode_config['period_span'])
  run_id = f"{os.getpid()}_{int(datetime.now().timestamp() * 1000)}"

  testback_logger.info(
    f"调试模式参数: population_size={population_size}, generations={generations}, window_days={window_days}"
  )

  if not ordered_topns:
    testback_logger.error("TopN 预计算数据为空")
    sys.exit(1)

  output_dir = _resolve_output_dir(args.output_dir, 'debug')

  _ga_state['population'] = []
  _ga_state['hall_of_fame'] = []
  _ga_state['fitness_cache'] = {}

  next_configs = generate_initial_configs(3 * population_size, profile_name=profile_name)

  for generation in range(generations):
    mem_offset, mem_count = _sample_topn_window(len(ordered_topns), window_days)
    window_start = ordered_topns[mem_offset].base_date.date()
    window_end = ordered_topns[mem_offset + mem_count - 1].base_date.date()
    topn_cache_key = _make_topn_window_cache_key(run_id, generation)

    testback_logger.info(f"\n{'=' * 60}")
    testback_logger.info(f"GA 第 {generation + 1}/{generations} 代 (调试模式)")
    testback_logger.info(f"窗口: offset={mem_offset}, count={mem_count}, {window_start} ~ {window_end}")
    testback_logger.info(f"{'=' * 60}")

    new_params = [[config, mem_offset, mem_count, topn_cache_key] for config in next_configs]
    ga_worker_count = min(cpu_count, len(new_params))
    _put_topn_window_slice(ordered_topns, mem_offset, mem_count, topn_cache_key=topn_cache_key)
    try:
      with parallel_backend('loky', n_jobs=ga_worker_count):
        parallel_pool = Parallel(
          return_as='generator',
          n_jobs=ga_worker_count,
          prefer='processes',
          batch_size=1,
          verbose=0,
        )
        results_list = [
          result for result in parallel_pool(delayed(_wrap_process_worker)(*worker_args) for worker_args in new_params)
          if result is not None
        ]
    finally:
      testback_cache.remove(topn_cache_key)

    if not results_list:
      raise RuntimeError('调试模式未获得任何有效回测结果')

    for result in results_list:
      result['generation'] = generation
      result['fitness'] = result['total_return']
      result['target2_fitness'] = result.get('target2_total_return', result['total_return'])
      result['window_offset'] = mem_offset
      result['window_count'] = mem_count
      result['window_start'] = window_start.isoformat()
      result['window_end'] = window_end.isoformat()

    next_configs = ga_optimizer(
      results_list,
      population_size=population_size,
      hall_of_fame_size=population_size,
      profile_name=profile_name,
    )

  testback_logger.info(f"\n调试模式执行完成，结果目录: {output_dir}")
  return None


def run_ga_mode(args, mode_config, ordered_topns, profile_name: str = DEFAULT_GA_PROFILE):
  """GA 优化模式：按 profile 定义执行完整 GA 搜索。"""
  import json as json_mod
  import pickle
  import time

  cpu_count = os.cpu_count() or 4
  generations = mode_config['generations']
  window_days = mode_config.get('window_days', mode_config['period_span'])
  population_size = mode_config['population_size'] or max(8, ((cpu_count // 2) * 2) or 8)
  run_id = f"{os.getpid()}_{int(datetime.now().timestamp() * 1000)}"

  testback_logger.info(
    f"GA 模式参数: population_size={population_size}, generations={generations}, window_days={window_days}"
  )

  if not ordered_topns:
    testback_logger.error("TopN 预计算数据为空")
    sys.exit(1)

  output_dir = _resolve_output_dir(args.output_dir, 'ga')

  _ga_state['population'] = []
  _ga_state['hall_of_fame'] = []
  _ga_state['fitness_cache'] = {}

  profile_metadata = get_profile_metadata(profile_name)
  next_configs = generate_initial_configs(3 * population_size, profile_name=profile_name)

  all_results = []
  generation_results = []

  for generation in range(generations):
    generation_start_ts = time.time()
    mem_offset, mem_count = _sample_topn_window(len(ordered_topns), window_days)
    window_start = ordered_topns[mem_offset].base_date.date()
    window_end = ordered_topns[mem_offset + mem_count - 1].base_date.date()
    topn_cache_key = _make_topn_window_cache_key(run_id, generation)

    testback_logger.info(f"\n{'=' * 60}")
    testback_logger.info(f"GA 第 {generation + 1}/{generations} 代")
    testback_logger.info(f"窗口: offset={mem_offset}, count={mem_count}, {window_start} ~ {window_end}")
    testback_logger.info(f"{'=' * 60}")

    new_params = [[config, mem_offset, mem_count, topn_cache_key] for config in next_configs]
    ga_worker_count = min(cpu_count, len(new_params))
    _put_topn_window_slice(ordered_topns, mem_offset, mem_count, topn_cache_key=topn_cache_key)
    try:
      with parallel_backend('loky', n_jobs=ga_worker_count):
        parallel_pool = Parallel(
          return_as='generator',
          n_jobs=ga_worker_count,
          prefer='processes',
          batch_size=1,
          verbose=0,
        )
        results_list = [
          result for result in parallel_pool(delayed(_wrap_process_worker)(*worker_args) for worker_args in new_params)
          if result is not None
        ]
    finally:
      testback_cache.remove(topn_cache_key)

    if not results_list:
      raise RuntimeError(f'第 {generation + 1} 代未获得任何有效回测结果')

    generation_time = time.time() - generation_start_ts
    for result in results_list:
      result['generation'] = generation
      result['fitness'] = result['total_return']
      result['target2_fitness'] = result.get('target2_total_return', result['total_return'])
      result['window_offset'] = mem_offset
      result['window_count'] = mem_count
      result['window_start'] = window_start.isoformat()
      result['window_end'] = window_end.isoformat()

    all_results.extend(results_list)

    fitnesses = [ind['fitness'] for ind in results_list]
    target2_fitnesses = [ind.get('target2_fitness', 0) for ind in results_list]
    generation_stats = {
      'generation': generation,
      'generation_time': generation_time,
      'population_size': len(results_list),
      'window_offset': mem_offset,
      'window_count': mem_count,
      'window_start': window_start.isoformat(),
      'window_end': window_end.isoformat(),
      'max_fitness': max(fitnesses),
      'mean_fitness': sum(fitnesses) / len(fitnesses),
      'min_fitness': min(fitnesses),
      'max_target2_fitness': max(target2_fitnesses),
      'mean_target2_fitness': sum(target2_fitnesses) / len(target2_fitnesses),
      'min_target2_fitness': min(target2_fitnesses),
      'all_individuals': results_list,
    }
    generation_results.append(generation_stats)

    with open(output_dir / 'generation_results.pkl', 'wb') as f:
      pickle.dump(generation_results, f)

    best_in_gen = max(results_list, key=lambda x: x['fitness'])
    best_result = {
      **profile_metadata,
      'individual_config': best_in_gen['individual_config'],
      'fitness': best_in_gen['fitness'],
      'generation': generation + 1,
      'generation_time': generation_time,
      'population_size': len(results_list),
      'window_offset': mem_offset,
      'window_count': mem_count,
      'window_start': window_start.isoformat(),
      'window_end': window_end.isoformat(),
    }
    with open(output_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
      json_mod.dump(best_result, f, indent=2, ensure_ascii=False)

    next_configs = ga_optimizer(
      results_list,
      population_size=population_size,
      hall_of_fame_size=population_size,
      profile_name=profile_name,
    )

  all_individual_configs = []
  for gen_stat in generation_results:
    for ind in gen_stat['all_individuals']:
      all_individual_configs.append({
        'generation': ind['generation'],
        'window_offset': ind.get('window_offset'),
        'window_count': ind.get('window_count'),
        'window_start': ind.get('window_start'),
        'window_end': ind.get('window_end'),
        'individual_config': ind['individual_config'],
        'fitness': ind['fitness'],
        'target2_fitness': ind.get('target2_fitness', 0),
        'total_return': ind['total_return'],
        'target2_total_return': ind.get('target2_total_return', 0),
        'init_cash': ind['init_cash'],
        'final_cash': ind['final_cash'],
        'final_market_value': ind['final_market_value'],
        'final_total_asset': ind['final_total_asset'],
        'cleared_positions_count': ind['cleared_positions_count'],
        'current_positions_count': ind['current_positions_count'],
      })

  with open(output_dir / 'all_individuals.json', 'w', encoding='utf-8') as f:
    json_mod.dump(all_individual_configs, f, indent=2, ensure_ascii=False)
  testback_logger.info(f"已保存所有个体配置: {output_dir / 'all_individuals.json'}")

  best_config = _ga_state['hall_of_fame'][0]
  key = _individual_config_to_key(best_config)
  best_fitness = _ga_state['fitness_cache'][key]

  best_result = {
    **profile_metadata,
    'individual_config': best_config,
    'fitness': best_fitness,
    'generation': generations,
    'population_size': population_size,
  }
  with open(output_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
    json_mod.dump(best_result, f, indent=2, ensure_ascii=False)

  testback_logger.info(f"\n最优参数已保存:")
  testback_logger.info(f"  - {output_dir / 'best_individual_config.json'}")
  testback_logger.info(f"最优收益率: {best_fitness:.2f}%")
  testback_logger.info(f"最优buy_n: {best_config['buy_n']}, sell_m: {best_config['sell_m']}")

  returns = [r['total_return'] for r in all_results]
  testback_logger.info(f"\n{'=' * 60}")
  testback_logger.info("回测执行完成")
  testback_logger.info("\n统计信息:")
  testback_logger.info(f"  总回测次数: {len(all_results)}")
  testback_logger.info(f"  平均收益率: {sum(returns) / len(returns):.2f}%")
  testback_logger.info(f"  最大收益率: {max(returns):.2f}%")
  testback_logger.info(f"  最小收益率: {min(returns):.2f}%")
  testback_logger.info(f"  正收益策略: {len([r for r in returns if r > 0])} 个")
  testback_logger.info(f"  负收益策略: {len([r for r in returns if r < 0])} 个")
  testback_logger.info(f"{'=' * 60}")

  return None


# ========== 主入口 ==========

def _main_impl():
  """主入口 - 根据模式参数执行"""
  import argparse
  import random

  from utils.stock.time import get_trading_date_span
  from loguru import logger as loguru_logger

  ts = datetime.now()
  result = None

  # 解析参数
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default='ga', choices=['single', 'debug', 'ga'],
                      help='运行模式')
  parser.add_argument('--individual-config', type=str, default=None,
                      help='single 模式使用的配置文件路径')
  parser.add_argument('--output-dir', type=str, default=None,
                      help='结果输出目录')
  parser.add_argument('--start-date', type=str, default=None,
                      help='回测开始日期，格式 YYYYMMDD 或 YYYY-MM-DD（默认 2020-06-30）')
  parser.add_argument('--end-date', type=str, default=None,
                      help='回测结束日期，格式 YYYYMMDD 或 YYYY-MM-DD（默认 2024-12-31）')
  parser.add_argument('--profile', type=str, default=None,
                      help=f'GA profile 名称（默认 {DEFAULT_GA_PROFILE}）')
  args = parser.parse_args()

  # 获取模式配置
  # 与 trading/main.py 一致：可买 A 股池（排除科创板等），避免回测在全市场列表上选股指与实盘漂移
  filtered_stocks = list(allow_buy_stock_code_list())
  profile_name = args.profile or DEFAULT_GA_PROFILE
  mode_configs = get_mode_configs(profile_name)
  mode_config = mode_configs[args.mode].copy()

  # 设置日志级别
  loguru_logger.remove()
  loguru_logger.add(sys.stderr, level=mode_config['log_level'])

  testback_logger.info(f"运行模式: {args.mode} - {mode_config['desc']}")
  testback_logger.info(f"回测周期: {mode_config['period_span']} 天")
  testback_logger.info(f"股票池（与实盘 allow_buy 一致）: {len(filtered_stocks)} 只")

  # 解析日期范围
  def parse_date(s):
    if s is None:
      return None
    s = s.replace('-', '')
    if len(s) == 8:
      return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    raise ValueError(f'日期格式错误: {s}，应为 YYYYMMDD 或 YYYY-MM-DD')

  start_date = parse_date(args.start_date) or date(2020, 6, 30)
  end_date = parse_date(args.end_date) or date(2024, 12, 31)

  if args.mode == 'ga':
    start_date, end_date = get_profile_preload_range(profile_name)
    testback_logger.info(
      f"GA 模式固定预加载区间: {start_date.strftime('%Y%m%d')} - {end_date.strftime('%Y%m%d')}"
    )

  # 数据准备：预加载到共享内存（所有模式共用）
  backtest_datetime_list = [
    datetime.combine(d, datetime.min.time())
    for d in get_trading_date_span(start_date, end_date)
  ]

  # 打印因子历史需求
  factor_classes = get_profile_factor_classes(profile_name)
  factor_histories = {factor_cls.__name__: factor_cls().hist_days for factor_cls in factor_classes}
  max_hist_days = max(factor_histories.values(), default=0)
  hist_detail = ', '.join(f'{name}={days}天' for name, days in factor_histories.items())
  testback_logger.info(f"因子历史需求: {hist_detail}，最大需求={max_hist_days}天")

  try:
    _load_shared_data(backtest_datetime_list, filtered_stocks, max_hist_days)

    from core.database.stock_name import prefetch_stock_histories
    prefetch_stock_histories(filtered_stocks)

    ordered_topns = None
    if args.mode in {'debug', 'ga'}:
      ordered_topns = _prepare_shared_topn(backtest_datetime_list, filtered_stocks, profile_name=profile_name)

    # 根据模式执行
    if args.mode == 'single':
      result = run_single_mode(args, mode_config, backtest_datetime_list, filtered_stocks)
    elif args.mode == 'debug':
      result = run_debug_mode(args, mode_config, ordered_topns, profile_name=profile_name)
    else:
      result = run_ga_mode(args, mode_config, ordered_topns, profile_name=profile_name)

    te = datetime.now()
    testback_logger.info(f"总耗时: {(te - ts).total_seconds():.2f} 秒")
    return result
  finally:
    testback_cache.cleanup()
    cleanup_shared_cache()


def main():
  with keep_windows_awake() as keep_awake_enabled:
    if keep_awake_enabled:
      testback_logger.info('已启用 Windows 防休眠，任务结束后自动恢复')
    else:
      testback_logger.warning('未能启用 Windows 防休眠，系统可能仍按当前电源策略休眠')
    return _main_impl()


if __name__ == "__main__":
  main()
