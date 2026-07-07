"""回测核心函数：因子计算、直接回测、择时、指标、single模式入口。"""

import json
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)

from core.strategy_config import load_strategy_config
from core.runtime import load_runtime_npz
from core.scoring import scores_to_ranks, compute_weighted_scores
from core.prefilter import apply_prefilter
from core.legality import LegalityChecker
from core.sim.account import StockAccountMocker
from core.strategy import build_rebalance_day
from data.db.delist import get_delist_stock_info
from testback.logger import testback_logger
from testback.metrics import compute_hs300_cumulative_returns, compute_strategy_metrics, compute_per_year_metrics
from utils.stock.time import get_trading_date_span

_ALL_A_SHARE_PREFIXES = ('60', '00', '30', '688')


# ========== 工具函数 ==========

def _count_holding_trading_days(start_date, end_date) -> int:
  if start_date is None or end_date is None:
    return 0
  if end_date < start_date:
    return 0
  return len(get_trading_date_span(start_date, end_date))


def is_rebalance_day_index(day_index: int, holding_period) -> bool:
  return not (holding_period and holding_period > 1) or day_index % holding_period == 0


def stock_pool_prefixes(stock_pool=None):
  pool = stock_pool or _ALL_A_SHARE_PREFIXES
  return tuple(pool) if isinstance(pool, list) else pool


def filter_stock_pool(stocks: list[str], stock_pool=None) -> list[str]:
  prefixes = stock_pool_prefixes(stock_pool)
  return [s for s in stocks if s.startswith(prefixes)]


def build_list_dates_map(data) -> dict:
  return _compute_list_dates(data['stock_codes'], data['open'], data['trade_dates'])


def build_delist_dates_map() -> dict:
  return {c: info.delist_date for c, info in get_delist_stock_info().items()}


def _get_stock_name_map(traded_codes: set[str], stock_names=None, stock_codes=None) -> Dict[str, str]:
  if stock_names is None or stock_codes is None:
    return {}
  name_map: Dict[str, str] = {}
  for i, code in enumerate(stock_codes):
    if code in traded_codes:
      name_map[str(code)] = str(stock_names[i]) if stock_names[i] else ''
  return name_map


def _calc_holding_stats(current_positions: List[Dict], cleared_positions: List[Dict]) -> Dict:
  current_days = [p['holding_days'] for p in current_positions]
  cleared_days = [p['holding_days'] for p in cleared_positions]
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


# ========== 核心回测 ==========

def _compute_factor_scores(backtest_datetime_list, all_stocks, weights, factor_classes,
                           data=None, kline_data=None, filter_factor_classes=None,
                           enable_nan_filter=True):
  """加载 NPZ 并批量计算因子分数，返回 (data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices)。

  data 可传入预加载面板（GA 整轮复用同一份 NPZ，避免每个因子重复加载 580MB 文件）。
  kline_data 用于内存 overlay 今日 K 线（实盘/盘后共用，不落盘 NPZ）。
  """
  if data is None:
    all_factor_classes = list(factor_classes) + list(filter_factor_classes or [])
    max_lookback = max((c.hist_days for c in all_factor_classes), default=0) or None
    data = load_runtime_npz(backtest_datetime_list, max_lookback=max_lookback)
  if data is None:
    first_d = backtest_datetime_list[0].strftime('%Y%m%d')
    last_d = backtest_datetime_list[-1].strftime('%Y%m%d')
    raise FileNotFoundError(f"未找到覆盖 {first_d}~{last_d} 的 runtime npz 文件")

  if kline_data:
    from data.update_live import apply_kline_overlay
    data, _, _ = apply_kline_overlay(data, kline_data)

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
    d = dt.date()
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
    if weights is not None and name not in weights:
      continue
    if weights is not None and weights[name] == 0:
      continue
    factor_meta.append((name, f))

  py_dates = [d.astype('datetime64[D]').item() for d in npz_dates]
  factor_data = {**data, 'stock_codes': npz_stocks, 'trade_dates': py_dates}
  all_scores: dict[str, np.ndarray] = {}
  all_raw = []
  for name, f in factor_meta:
    raw = f.calc_batch(factor_data)
    all_scores[name] = scores_to_ranks(raw.astype(np.float32, copy=False))
    all_raw.append(raw.astype(np.float32, copy=False))

  # 多因子 NaN 并集过滤：任一因子对某股票返回 NaN，该股票就该天排除
  filter_masks: dict[str, np.ndarray] = {}
  if enable_nan_filter:
    if len(all_raw) > 1:
      stacked = np.stack(all_raw, axis=0)
      any_nan = np.any(np.isnan(stacked), axis=0)
      filter_masks['_nan_union'] = ~any_nan
    elif len(all_raw) == 1:
      filter_masks['_nan_union'] = np.isfinite(all_raw[0])

  for f_cls in filter_factor_classes or []:
    f = f_cls()
    raw = f.calc_batch(factor_data)
    filter_masks[f.__class__.__name__] = np.isfinite(raw) & (raw > 0)

  testback_logger.info(f"因子批量+预排名完成 ({time.time() - t0:.1f}s), {len(valid_dates)} 个调仓日")
  return data, all_scores, filter_masks, valid_dates, date_indices, valid_stocks, stock_indices


def apply_t1_prefilter_for_date(prev_date, all_stocks, weights, factor_classes, prefilter_n, positions=None):
  if not prefilter_n:
    return all_stocks, None
  prev_dt = datetime.combine(prev_date, datetime.min.time())
  t1 = _compute_factor_scores([prev_dt], all_stocks, weights, factor_classes)
  if t1 is None:
    raise ValueError(f"prefilter T-1 {prev_date} 不在 runtime npz 范围内")
  t1_data, t1_scores, _, _, t1_di, t1_stocks, t1_si = t1
  t1_cols = np.array([t1_si[s] for s in t1_stocks], dtype=np.intp)
  t1_w = compute_weighted_scores(t1_scores, t1_di[0], t1_cols, weights)
  t1_ranking = [t1_stocks[i] for i in np.argsort(-t1_w)]
  pos_codes = list(positions or {})
  return apply_prefilter(t1_ranking, prefilter_n, all_stocks, pos_codes), t1_data


def _backtest_direct(data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
                     weights, buy_n, sell_m, holding_period=None,
                     verify_config=None,
                     position_multipliers=None, list_dates_map=None,
                     lightweight=False, init_cash=1_000_000.0, init_positions=None,
                     init_total_asset=None,
                     market_order_freeze=True, limit_up_protection=False,
                     rebalance=True, filter_masks=None, prefilter_n=None,
                     rebalance_start_index=0):
  """直接 numpy 回测，不创建 TopN 对象。lightweight=True 跳过明细组装，仅返回收益序列。

  init_cash / init_positions: 种子参数，用于「单日回放」对账（盘后用实盘 T-1 真实
  现金+持仓做起点，只回放 T 日多退少补，使手数与实盘可比）。默认保持 70 万空仓，
  不影响 GA / 单次研究回测。init_positions 形如 {code: {'volume', 'avg_price'}}。

  market_order_freeze: 默认 True。模拟「市价买单按涨停价冻结资金」的真实约束：
    1. 买入可用校验用涨停价(前收×(1+板块涨跌幅))而非收盘价，序贯下单、成交即释放；
       末位标的现金不足以覆盖涨停价冻结就买不进，与实盘一致。
    2. base_target 预留 reserve_L = max(板块涨跌幅) 份额，即 total_eq/(buy_n+reserve_L)，
       让最后一只也冻结得起，实现尽量均匀的满仓。
    置 False 可复现旧口径(收盘价校验、total_eq/buy_n)。

  limit_up_protection: 默认 False。收盘涨停保护——当日收盘涨停的股票不买也不卖，
    被过滤的买单从排名中补位至 buy_n 满额。

  rebalance: 默认 True。均仓多退少补——每只持仓目标=total_eq/buy_n，超出则卖、不足则补。
    置 False 则仅替换：只清仓不在 topN 的持仓 + 均分可用现金买入 topN 中缺失的标的，
    topN 内已有持仓不动（不追平市值）。

  成交价与估值全用原始真实价（不复权）。preClose 已吸收除权除息调整，收益计算无误。
  """

  account = StockAccountMocker(cash=init_cash)
  if init_positions:
    seed_date = valid_dates[0].date() if valid_dates else None
    for code, info in init_positions.items():
      vol = int(info.get('volume', 0) or 0)
      if vol <= 0:
        continue
      avg = float(info.get('avg_price', 0) or 0.0)
      account.positions[code] = {
        'code': code, 'volume': vol, 'cost': avg * vol, 'commission': 0.0,
        'buy_date': seed_date, 'buy_signal_date': seed_date, 'buy_trade_date': seed_date,
        'avg_price': avg, 'price_field': 'close',
      }
  initial_equity = float(init_total_asset) if init_total_asset is not None else account.init_cash
  delist_stock_info = get_delist_stock_info()
  if list_dates_map is None:
    list_dates_map = build_list_dates_map(data)

  daily_snapshots: List[Dict] = []
  prices: dict[str, float] = {}
  delist_events: List[Dict] = []

  full_cols = np.array([stock_indices[s] for s in valid_stocks], dtype=np.intp)

  # 临退禁买：把退市股摘牌/暂停日传入合法性闸门（实盘不传，由 allow_buy 名单剔除）
  delist_dates_map = {c: info.delist_date for c, info in delist_stock_info.items()}
  # 合法性一律用原始 OHLC + 官方 preClose
  checker = LegalityChecker(data, stock_indices, list_dates_map, delist_dates_map,
                             limit_up_protection=limit_up_protection)
  force_codes = []
  if verify_config:
    force_codes = [verify_config['force_stock_code']]
    force_codes += [c for c in verify_config.get('candidate_stock_codes', []) if c != verify_config['force_stock_code']]

  def _write_off_delisted_positions(signal_date, trade_date):
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
      )
      delist_events.append({
        'code': stock, 'delist_date': delist_info.delist_date,
        'clear_signal_date': signal_date, 'clear_trade_date': trade_date,
        'buy_trade_date': buy_trade_date,
        'holding_days': _count_holding_trading_days(buy_trade_date, trade_date),
        'volume': position.get('volume', 0), 'cost': cost, 'income': -cost,
        'income_pct': -100.0 if cost else 0.0, 'clear_reason': '退市归零',
      })

  _last_valid_close_price: dict[str, float] = {}
  if lightweight:
    _lw_daily_returns = []
    _lw_topn: List[List[str]] = []
    _lw_last_asset = initial_equity

  t1_ranking: list[str] = []
  for i, dt in enumerate(valid_dates):
    signal_date = dt.date()
    date_idx = date_indices[i]
    trade_idx = date_idx
    trade_date = signal_date

    _write_off_delisted_positions(signal_date, trade_date)

    is_rebalance_day = is_rebalance_day_index(rebalance_start_index + i, holding_period)

    pos_volumes = {c: p['volume'] for c, p in account.positions.items()}
    timing_mult = position_multipliers[i] if (position_multipliers is not None and not np.isnan(position_multipliers[i])) else 1.0

    # prefilter：首日全量，后续用 T-1 排名限制候选池
    if prefilter_n and i > 0 and t1_ranking:
      day_stocks = apply_prefilter(t1_ranking, prefilter_n, valid_stocks, pos_volumes)
    else:
      day_stocks = valid_stocks
    day_cols = np.array([stock_indices[s] for s in day_stocks], dtype=np.intp)

    buy_filter_mask = None
    if filter_masks:
      rows = [mask[date_idx][day_cols] for mask in filter_masks.values()]
      buy_filter_mask = np.logical_and.reduce(rows) if rows else None
    day_plan = build_rebalance_day(
        data=data, all_scores=all_scores, date_idx=date_idx, trade_idx=trade_idx,
        signal_date=signal_date, valid_stocks=day_stocks, valid_cols=day_cols,
        stock_indices=stock_indices, weights=weights, buy_n=buy_n, sell_m=sell_m,
        checker=checker, positions=pos_volumes, sellable_volumes=pos_volumes,
        cash=account.current_cash, rebalance=rebalance,
        is_rebalance_day=is_rebalance_day,
        force_codes=force_codes if force_codes else None,
        position_multiplier=timing_mult,
        last_valid_close_prices=_last_valid_close_price,
        market_order_freeze=market_order_freeze,
        limit_up_protection=limit_up_protection,
        buy_filter_mask=buy_filter_mask,
    )
    buy_n_stocks = day_plan.buy_n_stocks
    sell_m_stocks = day_plan.sell_m_stocks
    prices = day_plan.prices
    close_prices = day_plan.close_prices

    # 全量排名供 T+1 prefilter
    if prefilter_n:
      t_full = compute_weighted_scores(all_scores, date_idx, full_cols, weights)
      t1_ranking = [valid_stocks[i] for i in np.argsort(-t_full)]
    tradable_buy_stocks = day_plan.tradable_buy_stocks

    executed_sell_list: List[str] = []
    executed_sell_details: List[Dict] = []
    executed_buy_records: List[Dict] = []

    prev_positions = set(account.positions.keys())

    if day_plan.sell_orders or day_plan.buy_orders:
      buy_n_set = set(buy_n_stocks)
      for code, sv in day_plan.sell_orders:
        vol = account.positions[code]['volume'] if sv == -1 else sv
        tgt = day_plan.base_target if (rebalance and code in buy_n_set) else 0.0
        account.sell_stock(code, vol, prices[code], trade_date,
                           clear_reason='换出', signal_date=signal_date)
        executed_sell_list.append(code)
        executed_sell_details.append({
          'code': code, 'shares': vol, 'price': prices[code],
          'reason': '换出', 'cv_before': day_plan.pos_vals.get(code, 0),
          'target': tgt,
        })

      for code, bv in day_plan.buy_orders.items():
        if account.buy_stock(code, bv, prices[code], trade_date, signal_date=signal_date):
          executed_buy_records.append({
            'code': code, 'signal_date': signal_date.isoformat(),
            'trade_date': trade_date.isoformat(), 'price': prices[code],
            'price_field': 'close', 'shares': bv,
          })

    curr_positions = set(account.positions.keys())
    entered_stocks = [c for c in curr_positions if c not in prev_positions]
    exited_stocks = [c for c in prev_positions if c not in curr_positions]

    if lightweight:
      mkt_val = sum(close_prices[c] * p['volume'] for c, p in account.positions.items() if c in close_prices)
      total_asset = account.current_cash + mkt_val
      if _lw_last_asset > 0:
        daily_ret = (total_asset - _lw_last_asset) / _lw_last_asset * 100
      else:
        daily_ret = 0.0
      _lw_daily_returns.append(daily_ret)
      _lw_topn.append(list(buy_n_stocks))
      _lw_last_asset = total_asset
    else:
      assets = account.calc_assets(close_prices)
      prev_total_asset = daily_snapshots[-1]['total_asset'] if daily_snapshots else initial_equity
      daily_ret = (assets['total_asset'] - prev_total_asset) / prev_total_asset * 100 if prev_total_asset else 0.0
      # positions_eod: 当日收盘后持仓快照，用于多日回测的 T 日 daily_pnl 重建
      positions_eod = account.calc_position_values(close_prices)
      daily_snapshots.append({
        'date': trade_date.strftime('%Y-%m-%d'),
        'signal_date': signal_date.strftime('%Y-%m-%d'),
        'trade_date': trade_date.strftime('%Y-%m-%d'),
        'price_field': 'close',
        'cash': assets['cash'], 'market_value': assets['market_value'],
        'total_asset': assets['total_asset'], 'daily_return_pct': daily_ret, 'cumulative_return_pct': 0.0,
        'sell_m_list': sell_m_stocks,
        'raw_buy_n_list': buy_n_stocks,
        'buy_n_list': tradable_buy_stocks,
        'buy_n_diff_list': [s for s in tradable_buy_stocks if s not in sell_m_stocks],
        'executed_sell_list': executed_sell_list,
        'executed_sell_details': executed_sell_details,
        'executed_buy_list': [r['code'] for r in executed_buy_records],
        'executed_buy_details': executed_buy_records,
        'entered_stocks': entered_stocks,
        'exited_stocks': exited_stocks,
        'positions_eod': positions_eod,
      })

  if lightweight:
    mkt_val = sum(close_prices[c] * p['volume'] for c, p in account.positions.items() if c in close_prices)
    total_asset = account.current_cash + mkt_val
    total_return = (total_asset - initial_equity) / initial_equity * 100
    return {
      'total_return': total_return,
      'cleared_positions_count': len(account.cleared_positions),
      'daily_returns': _lw_daily_returns,
      'daily_topn': _lw_topn,
      'daily_snapshots': [], 'cumulative_returns': [], 'trade_log': [],
      'positions': [], 'cleared_positions': [], 'delist_events': [],
      'stock_name_map': {}, 'holding_stats': {},
      'executed_buy_count': 0, 'executed_sell_count': 0,
      'delist_count': 0, 'round_trip_count': 0, 'current_positions_count': 0,
      'final_asset': total_asset,
    }

  final_signal_date = valid_dates[-1].date()
  final_trade_date = final_signal_date
  final_assets = account.calc_assets(close_prices)
  total_return = (final_assets['total_asset'] - initial_equity) / initial_equity * 100

  cumulative_returns = []
  if daily_snapshots:
    for snap in daily_snapshots:
      cum = (snap['total_asset'] - initial_equity) / initial_equity * 100
      cumulative_returns.append(cum)
    for i, snap in enumerate(daily_snapshots):
      snap['cumulative_return_pct'] = cumulative_returns[i]

  daily_returns = [snap['daily_return_pct'] for snap in daily_snapshots]

  positions = account.calc_position_values(close_prices)
  for position in positions:
    position['holding_days'] = _count_holding_trading_days(
      position['buy_trade_date'] or position['buy_date'], final_trade_date)

  cleared_positions = []
  for cleared in account.cleared_positions:
    buy_trade_date = cleared['pos']['buy_trade_date'] or cleared['pos']['buy_date']
    clear_trade_date = cleared['clear_trade_date'] or cleared['clear_date']
    holding_days = _count_holding_trading_days(buy_trade_date, clear_trade_date)
    cost = cleared['pos']['cost']
    income = cleared['income']
    income_pct = (income / cost * 100) if cost else 0.0
    cleared_positions.append({
      'code': cleared['code'],
      'buy_date': cleared['pos']['buy_date'],
      'buy_signal_date': cleared['pos']['buy_signal_date'],
      'buy_trade_date': buy_trade_date,
      'clear_date': cleared['clear_date'],
      'clear_signal_date': cleared['clear_signal_date'],
      'clear_trade_date': clear_trade_date,
      'holding_days': holding_days,
      'volume': cleared['pos']['volume'],
      'avg_price': cleared['pos']['avg_price'],
      'cost': cost,
      'clear_price': cleared['clear_price'],
      'income': income,
      'income_pct': income_pct,
      'clear_reason': cleared['clear_reason'],
      'price_field': cleared['price_field'],
    })

  trade_log = account.get_trade_log()
  executed_buy_count = len([t for t in trade_log if t['action'] == 'buy'])
  executed_sell_count = len([t for t in trade_log if t['action'] == 'sell'])

  all_stock_codes = set()
  for trade in trade_log:
    if trade['code']:
      all_stock_codes.add(trade['code'])
  for snapshot in daily_snapshots:
    all_stock_codes.update(snapshot['buy_n_list'])
    all_stock_codes.update(snapshot['executed_buy_list'])
  for position in positions:
    if position['code']:
      all_stock_codes.add(position['code'])
  for cleared in cleared_positions:
    if cleared['code']:
      all_stock_codes.add(cleared['code'])

  stock_name_map = _get_stock_name_map(all_stock_codes, data['stock_names'], data['stock_codes'])
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
    'final_asset': final_assets['total_asset'],
  }


# ========== 辅助函数 ==========

def _compute_list_dates(stock_codes_arr, open_arr, trade_dates_arr) -> dict:
  result = {}
  valid = ~np.isnan(open_arr) & (open_arr > 0)
  first_idx = np.argmax(valid, axis=0)
  has_valid = np.any(valid, axis=0)
  for i, code in enumerate(stock_codes_arr):
    if has_valid[i]:
      result[str(code)] = trade_dates_arr[first_idx[i]].astype('datetime64[D]').astype(date)
  return result


def _compute_timing_multipliers(config, valid_dates, index_data=None):
    from core.timing import load_index_close, compute_position_multiplier, compute_calendar_empty_mask, INDEX_INFO

    empty_months = config.get('empty_months')
    calendar_mask = compute_calendar_empty_mask(valid_dates, empty_months)

    timing_enabled = config.get('timing_enabled', False)
    timing_base = config.get('timing_base')

    result = None
    if timing_enabled and timing_base is not None:
        idx_symbol = config.get('timing_index', 'sh000852')
        if index_data is not None:
            idx_close = index_data.get(idx_symbol)
        else:
            _, idx_close = load_index_close(idx_symbol, valid_dates)
        if idx_close is not None:
            window = config.get('timing_window', 20)
            leverage = config.get('timing_leverage', 10)
            direction = config.get('timing_direction', 1)
            result = compute_position_multiplier(idx_close, window=window, base=timing_base, leverage=leverage, direction=direction)
    if calendar_mask is not None:
        result = calendar_mask if result is None else result * calendar_mask

    cash_reserve = float(config.get('cash_reserve_ratio', 0.0))
    if cash_reserve > 0:
        invest_ratio = np.float64(1.0 - cash_reserve)
        if result is not None:
            result = result * invest_ratio
        else:
            result = np.full(len(valid_dates), invest_ratio, dtype=np.float64)
    return result


def _load_all_index_data(valid_dates):
    import pyarrow.parquet as pq
    from core.timing import load_index_close, INDEX_INFO, _index_parquet_path

    index_data = {}
    start = valid_dates[0].date()
    end = valid_dates[-1].date()
    for sym in INDEX_INFO:
        path = _index_parquet_path(sym)
        if not path.exists():
            continue
        table = pq.read_table(path)
        idx_dates = table.column('trade_date').to_numpy().astype('datetime64[D]')
        idx_start = idx_dates[0].item()
        idx_end = idx_dates[-1].item()
        if start < idx_start or end > idx_end:
            continue
        _, index_data[sym] = load_index_close(sym, valid_dates)
    return index_data



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
  from core.timing import INDEX_INFO as _IDX
  idx_name = _IDX.get(idx_val, idx_val)
  return f', timing={base}/{leverage}x({d_str}, win={window}, {idx_name})'


# ========== 模式执行函数 ==========

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


def run_single_mode(args, mode_config, backtest_datetime_list, all_stocks):
  """单次回测模式：直接 numpy 回测，无 TopN 对象中间层"""
  if not args.individual_config:
    testback_logger.error('--individual-config 参数在 single 模式下必须指定')
    sys.exit(1)

  strategy_config = load_strategy_config(args.individual_config)
  individual_config = strategy_config['individual_config']
  config_data = {
    'ga_profile': strategy_config['profile_name'],
    'individual_config': individual_config,
  }
  verify_config = _parse_single_verify_config(config_data)

  output_dir = _resolve_output_dir(args.output_dir, 'single')
  testback_logger.add_file_sink(str(output_dir / 'single.log'))
  testback_logger.info(f"日志文件: {output_dir / 'single.log'}")
  testback_logger.info(f"从文件加载 Individual_config: {args.individual_config}")

  stock_pool = individual_config.get('stock_pool') or _ALL_A_SHARE_PREFIXES
  pool_tuple = stock_pool_prefixes(stock_pool)
  all_stocks = filter_stock_pool(all_stocks, stock_pool)
  testback_logger.info(f"stock_pool={_format_pool(pool_tuple)}: {len(all_stocks)} 只")

  candidate_stock_pool = _extend_verify_stock_pool_with_historical_codes(
    all_stocks, backtest_datetime_list, verify_config)
  single_stock_pool = _resolve_single_stock_pool(candidate_stock_pool, verify_config)
  if verify_config:
    testback_logger.info(
      f"启用 single 退市验证: {verify_config['force_stock_code']}, "
      f"候选池={len(single_stock_pool)} 只"
    )

  testback_logger.info(f"使用配置进行单次回测: buy_n={individual_config['buy_n']}, sell_m={individual_config['sell_m']}")

  config_factor_classes = strategy_config['factor_classes']
  filter_factor_classes = strategy_config.get('filter_factor_classes') or []
  if filter_factor_classes:
    testback_logger.info(f"启用买入过滤器: {', '.join(strategy_config.get('filter_factor_names', []))}")

  # NaN 并集过滤：config 默认 True，CLI --filter/--no-filter 可覆盖
  enable_nan_filter = individual_config.get('filter', True)
  if getattr(args, 'filter_enabled', None) is not None:
    enable_nan_filter = args.filter_enabled
  if not enable_nan_filter:
    testback_logger.info("NaN 并集过滤已关闭 (--no-filter)，缺失数据因子不参与加权、不排除股票")

  scores_result = _compute_factor_scores(
    backtest_datetime_list, single_stock_pool,
    weights=individual_config['weights'], factor_classes=config_factor_classes,
    filter_factor_classes=filter_factor_classes if filter_factor_classes else None,
    enable_nan_filter=enable_nan_filter,
  )
  if scores_result is None:
    testback_logger.error("因子计算失败，无有效交易日")
    sys.exit(1)
  data, all_scores, filter_masks, valid_dates, date_indices, valid_stocks, stock_indices = scores_result

  if '_nan_union' in filter_masks:
    total_cells = filter_masks['_nan_union'].size
    filtered_cells = int((~filter_masks['_nan_union']).sum())
    avg_filtered = filtered_cells / max(len(valid_dates), 1)
    testback_logger.info(
      f"NaN并集过滤: {filtered_cells}/{total_cells} 个股票·日被排除 "
      f"({filtered_cells/total_cells*100:.1f}%, 日均 {avg_filtered:.0f} 只)")

  list_dates_map = build_list_dates_map(data)

  signal_dates = [d.date() for d in valid_dates]
  trade_dates = list(signal_dates)
  testback_logger.info(
    f"回测信号范围: {signal_dates[0]} ~ {signal_dates[-1]}，"
    f"执行范围: {trade_dates[0]} ~ {trade_dates[-1]}，共 {len(valid_dates)} 个调仓日"
  )

  timing_multipliers = _compute_timing_multipliers(individual_config, valid_dates)
  if timing_multipliers is not None:
    parts = []
    empty_months = individual_config.get('empty_months')
    if empty_months:
      parts.append(f"日历空仓: {','.join(str(m) for m in empty_months)}月")
    timing_enabled = individual_config.get('timing_enabled', False)
    if timing_enabled:
      from core.timing import INDEX_INFO
      idx_name = INDEX_INFO.get(individual_config.get('timing_index', 'sh000852'), '?')
      direction = individual_config.get('timing_direction', 1)
      d_name = '顺势' if direction == 1 else '逆势'
      parts.append(f"动量择时: {idx_name} {d_name} base={individual_config.get('timing_base', 0.5)}")
    cash_reserve = individual_config.get('cash_reserve_ratio', 0.0)
    if cash_reserve > 0:
      parts.append(f"闲钱保留: {cash_reserve*100:.0f}%")
    testback_logger.info(f"{' + '.join(parts)}, multiplier范围=[{np.nanmin(timing_multipliers):.2f}, {np.nanmax(timing_multipliers):.2f}]")

  prefilter_n = individual_config.get('prefilter_n')
  if prefilter_n:
    testback_logger.info(f"启用 prefilter: T-1 排名 top {prefilter_n} + 持仓")

  result = _backtest_direct(
    data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
    weights=individual_config['weights'],
    buy_n=individual_config['buy_n'], sell_m=individual_config['sell_m'],
    holding_period=individual_config.get('holding_period'),
    verify_config=verify_config,
    position_multipliers=timing_multipliers,
    list_dates_map=list_dates_map,
    limit_up_protection=individual_config.get('limit_up_protection', False),
    rebalance=individual_config.get('rebalance', True),
    filter_masks=filter_masks,
    prefilter_n=prefilter_n,
  )

  signal_date_strs = [d.strftime('%Y-%m-%d') for d in signal_dates]
  trade_date_strs = [d.strftime('%Y-%m-%d') for d in trade_dates]

  metrics = compute_strategy_metrics(
    cumulative_returns_pct=result['cumulative_returns'],
    trade_dates=trade_date_strs,
    trade_log=result['trade_log'],
  )
  hs300_returns = compute_hs300_cumulative_returns(trade_date_strs)
  per_year_metrics = compute_per_year_metrics(result['cumulative_returns'], trade_date_strs)

  testback_logger.info(
    f"回测: 年化={metrics['annualized']:.2f}% 夏普={metrics['sharpe_ratio']:.2f} "
    f"最大回撤={metrics['max_drawdown']:.2f}% 卡玛={metrics['calmar_ratio']:.2f} "
    f"胜率={metrics['win_rate']:.1f}% 总成交={metrics['total_trades']}"
  )

  report_data = {
    'individual_config': individual_config,
    'total_return': result['total_return'],
    'daily_returns': result['daily_returns'],
    'cumulative_returns': result['cumulative_returns'],
    'signal_dates': signal_date_strs,
    'trade_dates': trade_date_strs,
    'trade_log': result['trade_log'],
    'daily_snapshots': result['daily_snapshots'],
    'positions': result['positions'],
    'cleared_positions': result['cleared_positions'],
    'delist_events': result['delist_events'],
    'stock_name_map': result['stock_name_map'],
    'holding_stats': result['holding_stats'],
    'executed_buy_count': result['executed_buy_count'],
    'executed_sell_count': result['executed_sell_count'],
    'delist_count': result['delist_count'],
    'round_trip_count': result['round_trip_count'],
    'final_asset': result['final_asset'],
    'metrics': metrics,
    'per_year_metrics': per_year_metrics,
    'hs300_returns': hs300_returns,
    'cleared_positions_count': result['cleared_positions_count'],
    'current_positions_count': result['current_positions_count'],
    'init_cash': 1_000_000.0,
    'verify_config': verify_config,
    'report_metadata': {
      'config_path': str(Path(args.individual_config).resolve()),
      'stock_pool_size': len(single_stock_pool),
    },
    'rebalance_rule': {
      'signal_timing': 'T', 'trade_timing': 'T close', 'price_field': 'close',
    },
    'period': {
      'signal_start': signal_date_strs[0], 'signal_end': signal_date_strs[-1],
      'trade_start': trade_date_strs[0], 'trade_end': trade_date_strs[-1],
      'start': trade_date_strs[0], 'end': trade_date_strs[-1],
    },
  }

  if mode_config['save_charts']:
    from testback.reportor import generate_single_report
    html_path = generate_single_report(report_data, output_dir)
    testback_logger.info(f"可视化报告已保存至: {html_path}")

  _save_single_record(report_data, output_dir, individual_config, metrics)

  testback_logger.remove_file_sink()
  return result


def _save_single_record(report_data, output_dir, individual_config, metrics):
  """落机器可读 record.json（每日日期/收益/topN 持仓 + 摘要），并在单因子时入 factor_runs。"""
  daily_snapshots = report_data.get('daily_snapshots', [])
  record = {
    'weights': individual_config.get('weights', {}),
    'buy_n': individual_config.get('buy_n'),
    'stock_pool': individual_config.get('stock_pool'),
    'period': report_data.get('period', {}),
    'dates': report_data.get('signal_dates', []),
    'daily_returns': report_data.get('daily_returns', []),
    'topn': [s.get('raw_buy_n_list', []) for s in daily_snapshots],
    'metrics': {
      'sharpe': metrics.get('sharpe_ratio'),
      'annualized': metrics.get('annualized'),
      'max_dd': metrics.get('max_drawdown'),
      'n_trades': metrics.get('round_trip_count', report_data.get('round_trip_count', 0)),
    },
  }
  record_path = Path(output_dir) / 'record.json'
  record_path.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
  testback_logger.info(f"回测明细记录已保存至: {record_path}")

  weights = individual_config.get('weights', {})
  if len(weights) == 1:
    from factor_db import records as _records
    name = next(iter(weights))
    period = report_data.get('period', {})
    m = record['metrics']
    _records.add_run(
      name, bt_start=period.get('start', ''), bt_end=period.get('end', ''),
      buy_n=individual_config.get('buy_n', 0),
      stock_pool=str(individual_config.get('stock_pool') or ''),
      dates=record['dates'], daily_returns=record['daily_returns'], topn=record['topn'],
      sharpe=m['sharpe'], annualized=m['annualized'],
      max_dd=m['max_dd'], n_trades=m['n_trades'], record_dir=str(Path(output_dir).resolve()),
    )
    testback_logger.info(f"已登记单因子回测到 factor_runs: {name}")


