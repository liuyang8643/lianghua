"""单日调仓决策链。

回测和实盘只能通过本模块生成调仓计划；`trading` 只负责把计划交给券商执行。
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable

import numpy as np

from core.rebalance import compute_rebalance_plan, freeze_unit_price
from core.scoring import (
  select_selection_sleeves_legal,
  select_topn_legal,
  top_level_factor_filter_masks,
  validate_selection_sleeves,
)
from utils.stock.info import board_limit_ratio


@dataclass
class RebalanceDayPlan:
  buy_n_stocks: list[str]
  sell_m_stocks: list[str]
  final_score: np.ndarray
  prices: dict[str, float]
  close_prices: dict[str, float]
  limit_prices: dict[str, float]
  pos_vals: dict[str, float]
  total_eq: float
  base_target: float
  reserve_L: float
  tradable_buy_stocks: list[str]
  sellable_ok: set[str]
  sell_orders: list[tuple[str, int]]
  buy_orders: dict[str, int]
  skip_reasons: dict[str, str]
  t1_ranking: list[str]


@dataclass
class StrategyDayResult:
  trade_date: date
  position_multiplier: float
  plan: RebalanceDayPlan
  data: dict
  all_scores: dict
  filter_masks: dict
  date_idx: int
  valid_stocks: list[str]
  stock_indices: dict[str, int]
  kline_data: dict | None


def build_strategy_day(
    *, trade_date: date, all_stocks: list[str], individual_config: dict,
    factor_classes: list, filter_factor_classes: list,
    positions: dict[str, int], sellable_volumes: dict[str, int], cash: float,
    kline_loader: Callable[[list[str]], dict | None] | None = None,
    kline_data: dict | None = None, target_cash: float | None = None,
    target_positions: dict[str, int] | None = None,
    is_rebalance_day: bool = True, position_multiplier: float = 1.0,
) -> StrategyDayResult:
  """Build the authoritative strategy target for one trading day."""
  from core.backtest import _compute_factor_scores, _compute_list_dates
  from core.legality import LegalityChecker

  weights = individual_config['weights']
  selection_sleeves = individual_config.get('selection_sleeves')
  sleeve_factor_names: set[str] = set()
  if selection_sleeves is not None:
    sleeves = validate_selection_sleeves(
      selection_sleeves,
      individual_config['buy_n'],
      available_factor_names={
        factor_class.__name__ for factor_class in factor_classes
      },
    )
    sleeve_factor_names = {
      name
      for sleeve in sleeves
      for name, weight in sleeve['weights'].items()
      if weight != 0.0
    }
  overlay_settings = individual_config.get('trend_risk_overlay') or {}
  overlay_mode = str(overlay_settings.get('mode', '')).lower()
  if kline_loader is not None:
    if kline_data is not None:
      raise ValueError('pass either kline_loader or kline_data')
    kline_data = kline_loader(all_stocks)

  score_data = None
  completed_factor_history = 0
  completed_timing_history = 0
  if overlay_settings.get('enabled') and overlay_mode == 'dual_completed':
    from core.runtime import load_runtime_npz

    active_factor_classes = [
      factor_class for factor_class in factor_classes
      if (
        weights.get(factor_class.__name__, 0) != 0
        or factor_class.__name__ in sleeve_factor_names
      )
    ]
    history_classes = active_factor_classes + list(filter_factor_classes or [])
    completed_factor_history = max(
      (int(factor_class.hist_days) for factor_class in history_classes),
      default=0,
    )
    completed_timing_history = max(
      int(overlay_settings.get('momentum_window', 20)),
      int(overlay_settings.get('ma_window', 20)),
      int(overlay_settings.get('strategy_momentum_window', 3)),
      int(overlay_settings.get('strategy_ma_window', 20)),
    )
    preload_lookback = (
      completed_factor_history + completed_timing_history + 10
    )
    score_data = load_runtime_npz(
      [datetime.combine(trade_date, datetime.min.time())],
      max_lookback=preload_lookback,
    )

  scored = _compute_factor_scores(
    [datetime.combine(trade_date, datetime.min.time())], all_stocks,
    weights, factor_classes, data=score_data, kline_data=kline_data,
    filter_factor_classes=filter_factor_classes or None,
    selection_sleeves=selection_sleeves,
    buy_n=individual_config['buy_n'],
  )
  if scored is None:
    raise ValueError(f'signal date {trade_date} is outside runtime data')
  data, all_scores, filter_masks, valid_dates, date_indices, valid_stocks, stock_indices = scored
  date_idx = date_indices[0]
  if overlay_settings.get('enabled') and overlay_mode in {
      'dual_strategy', 'dual_completed',
  }:
    if overlay_mode == 'dual_completed':
      from core.trend_timing import compute_dual_completed_trend_multipliers

      compute_dual_multiplier = compute_dual_completed_trend_multipliers
    else:
      from core.trend_timing import compute_dual_trend_multipliers

      compute_dual_multiplier = compute_dual_trend_multipliers

    history_start = 0
    if overlay_mode == 'dual_completed':
      history_start = date_idx - completed_timing_history
      if history_start < completed_factor_history:
        raise ValueError(
          'dual_completed runtime history is shorter than factor and timing '
          'warmup requirements'
        )
    history_indices = list(range(history_start, date_idx + 1))
    history_dates = [
      datetime.combine(value.astype('datetime64[D]').item(), datetime.min.time())
      for value in np.asarray(data['trade_dates'])[history_start:date_idx + 1]
    ]
    timing_filter_masks = top_level_factor_filter_masks(
      all_scores, filter_masks, weights,
    )
    dual = compute_dual_multiplier(
      data=data, all_scores=all_scores,
      valid_dates=history_dates, date_indices=history_indices,
      valid_stocks=valid_stocks, stock_indices=stock_indices,
      weights=weights, buy_n=individual_config['buy_n'],
      settings=overlay_settings,
      limit_up_protection=individual_config.get('limit_up_protection', False),
      filter_masks=timing_filter_masks,
    )
    position_multiplier *= float(dual[-1])
  else:
    from core.trend_timing import trend_overlay_multiplier_for_row
    position_multiplier *= trend_overlay_multiplier_for_row(
      data, date_idx, individual_config
    )
  valid_cols = np.array([stock_indices[s] for s in valid_stocks], dtype=np.intp)
  buy_filter_mask = None
  if filter_masks:
    buy_filter_mask = np.logical_and.reduce(
      [mask[date_idx][valid_cols] for mask in filter_masks.values()]
    )
  checker = LegalityChecker(
    data, stock_indices,
    _compute_list_dates(data['stock_codes'], data['open'], data['trade_dates']),
    limit_up_protection=individual_config['limit_up_protection'],
  )
  plan = build_rebalance_day(
    data=data, all_scores=all_scores, date_idx=date_idx, trade_idx=date_idx,
    signal_date=trade_date, valid_stocks=valid_stocks, valid_cols=valid_cols,
    stock_indices=stock_indices, weights=weights,
    buy_n=individual_config['buy_n'], sell_m=individual_config['sell_m'],
    checker=checker, positions=positions, sellable_volumes=sellable_volumes,
    cash=cash, rebalance=individual_config['rebalance'],
    is_rebalance_day=is_rebalance_day,
    position_multiplier=position_multiplier,
    target_cash=target_cash, target_positions=target_positions,
    price_codes_extra=set(target_positions or {}),
    buy_filter_mask=buy_filter_mask,
    selection_sleeves=selection_sleeves,
    slippage_bps=individual_config.get('slippage_bps', 10.0),
    rebalance_band_pct=individual_config.get('rebalance_band_pct', 0.01),
    enforce_position_multiplier_on_sell_m=individual_config.get(
      'enforce_position_multiplier_on_sell_m', False
    ),
  )
  return StrategyDayResult(
    trade_date=trade_date, position_multiplier=position_multiplier,
    plan=plan, data=data, all_scores=all_scores, filter_masks=filter_masks,
    date_idx=date_idx, valid_stocks=valid_stocks, stock_indices=stock_indices,
    kline_data=kline_data,
  )


def _map_open_prices(data, stock_indices, trade_idx: int, price_codes) -> dict[str, float]:
  open_all = data['open']
  day_open = open_all[trade_idx]
  prices: dict[str, float] = {}
  for code in price_codes:
    if code not in stock_indices:
      continue
    si = stock_indices[code]
    open_val = day_open[si]
    if not np.isnan(open_val) and open_val > 0:
      prices[code] = float(open_val)
  return prices


def _map_close_prices(data, stock_indices, trade_idx: int, price_codes,
                      prices: dict[str, float],
                      last_valid_close_prices: dict[str, float] | None) -> dict[str, float]:
  close_all = data['close']
  day_close = close_all[trade_idx]
  close_prices: dict[str, float] = {}
  for code in price_codes:
    if code not in stock_indices:
      continue
    si = stock_indices[code]
    close_val = day_close[si]
    if not np.isnan(close_val) and close_val > 0:
      close_prices[code] = float(close_val)
      if last_valid_close_prices is not None:
        last_valid_close_prices[code] = close_prices[code]
    elif last_valid_close_prices is not None and code in last_valid_close_prices:
      close_prices[code] = last_valid_close_prices[code]
    elif code in prices:
      close_prices[code] = prices[code]
  return close_prices


def _map_equity_prices(data, stock_indices, trade_idx: int, price_codes,
                       prices: dict[str, float],
                       last_valid_close_prices: dict[str, float] | None) -> dict[str, float]:
  """目标权益估值价：优先 open[T]；停牌/缺 open 时只用 close[T-1]。"""
  close_all = data['close']
  prev_close = close_all[trade_idx - 1] if trade_idx >= 1 else None
  equity_prices: dict[str, float] = {}
  for code in price_codes:
    if code in prices:
      equity_prices[code] = prices[code]
      continue
    si = stock_indices.get(code)
    if si is None:
      continue
    if prev_close is not None:
      pc = prev_close[si]
      if not np.isnan(pc) and pc > 0:
        equity_prices[code] = float(pc)
        continue
    if last_valid_close_prices is not None and code in last_valid_close_prices:
      equity_prices[code] = last_valid_close_prices[code]
  return equity_prices


def _freeze_prices(data, stock_indices, trade_idx: int, prices: dict[str, float],
                   market_order_freeze: bool) -> dict[str, float]:
  if not market_order_freeze:
    return {}
  preclose_row = data['preClose'][trade_idx]
  limit_prices: dict[str, float] = {}
  for code, price in prices.items():
    si = stock_indices[code]
    pc = float(preclose_row[si])
    if np.isnan(pc):
      pc = 0.0
    limit_prices[code] = freeze_unit_price(code, price, pc)
  return limit_prices


def build_rebalance_day(
    *,
    data,
    all_scores,
    date_idx: int,
    trade_idx: int,
    signal_date: date,
    valid_stocks: list[str],
    valid_cols: np.ndarray,
    stock_indices: dict[str, int],
    weights: dict,
    buy_n: int,
    sell_m: int,
    checker,
    positions: dict[str, int],
    sellable_volumes: dict[str, int],
    cash: float,
    rebalance: bool = True,
    is_rebalance_day: bool = True,
    force_codes: list[str] | None = None,
    position_multiplier: float = 1.0,
    target_cash: float | None = None,
    target_positions: dict[str, int] | None = None,
    price_codes_extra=None,
    last_valid_close_prices: dict[str, float] | None = None,
    market_order_freeze: bool = True,
    limit_up_protection: bool = False,
    buy_filter_mask: np.ndarray | None = None,
    sell_all_scores: dict[str, np.ndarray] | None = None,
    sell_weights: dict[str, float] | None = None,
    sell_filter_mask: np.ndarray | None = None,
    selection_sleeves: list[dict] | None = None,
    slippage_bps: float = 10.0,
    rebalance_band_pct: float = 0.01,
    enforce_position_multiplier_on_sell_m: bool = False,
) -> RebalanceDayPlan:
  if is_rebalance_day:
    day_open = data['open'][trade_idx]
    if selection_sleeves is not None:
      if sell_all_scores is not None:
        raise ValueError(
          'selection_sleeves cannot be combined with separate sell scores'
        )
      buy_n_stocks, sell_m_stocks, final_score, t1_ranking = (
        select_selection_sleeves_legal(
          all_scores=all_scores,
          score_idx=date_idx,
          valid_stocks=valid_stocks,
          valid_cols=valid_cols,
          selection_sleeves=selection_sleeves,
          buy_n=buy_n,
          sell_m=sell_m,
          checker=checker,
          trade_idx=trade_idx,
          signal_date=signal_date,
          day_open=day_open,
          common_filter_mask=buy_filter_mask,
        )
      )
    elif sell_all_scores is None:
      buy_n_stocks, sell_m_stocks, final_score, t1_ranking = select_topn_legal(
        all_scores, date_idx, valid_stocks, valid_cols,
        weights, buy_n, sell_m,
        checker=checker, trade_idx=trade_idx, signal_date=signal_date,
        day_open=day_open, stock_indices=stock_indices,
        filter_mask=buy_filter_mask,
      )
    else:
      buy_n_stocks, _, final_score, t1_ranking = select_topn_legal(
        all_scores, date_idx, valid_stocks, valid_cols,
        weights, buy_n, 0,
        checker=checker, trade_idx=trade_idx, signal_date=signal_date,
        day_open=day_open, stock_indices=stock_indices,
        filter_mask=buy_filter_mask,
      )
      _, sell_m_stocks, _, _ = select_topn_legal(
        sell_all_scores, date_idx, valid_stocks, valid_cols,
        sell_weights or {}, 0, sell_m,
        checker=checker, trade_idx=trade_idx, signal_date=signal_date,
        day_open=day_open, stock_indices=stock_indices,
        filter_mask=sell_filter_mask,
      )
  else:
    final_score = np.zeros(len(valid_stocks), dtype=np.float32)
    buy_n_stocks = []
    sell_m_stocks = []
    t1_ranking = []

  extra = set(price_codes_extra or [])
  price_universe = set(positions) | set(sell_m_stocks) | set(buy_n_stocks) | extra
  prices = _map_open_prices(data, stock_indices, trade_idx, price_universe)
  close_prices = _map_close_prices(
    data, stock_indices, trade_idx, price_universe,
    prices, last_valid_close_prices,
  )
  equity_prices = _map_equity_prices(
    data, stock_indices, trade_idx, price_universe,
    prices, last_valid_close_prices,
  )

  pos_vals = {c: positions[c] * prices[c] for c in positions if c in prices}
  if target_cash is not None and target_positions is not None:
    total_eq = target_cash + sum(sh * equity_prices[c] for c, sh in target_positions.items() if c in equity_prices)
  else:
    total_eq = cash + sum(positions[c] * equity_prices[c] for c in positions if c in equity_prices)

  reserve_L = max((board_limit_ratio(c) for c in buy_n_stocks), default=0.0) if market_order_freeze else 0.0
  base_target = total_eq * position_multiplier / (buy_n + reserve_L)
  limit_prices = _freeze_prices(data, stock_indices, trade_idx, prices, market_order_freeze)

  sellable_ok: set[str] = set()
  sell_check = [c for c in positions if c in prices and c in stock_indices]
  if sell_check:
    ok, _ = checker.check([stock_indices[c] for c in sell_check],
                          trade_idx, signal_date, is_buy=False)
    sellable_ok = {c for c, o in zip(sell_check, ok) if o}

  tradable_buy_stocks = buy_n_stocks

  sell_orders: list[tuple[str, int]] = []
  buy_orders: dict[str, int] = {}
  skip_reasons: dict[str, str] = {}
  if is_rebalance_day and (positions or buy_n_stocks):
    position_control_active = (
      enforce_position_multiplier_on_sell_m and position_multiplier < 1.0
    )
    sell_orders, buy_orders, skip_reasons = compute_rebalance_plan(
      positions=positions, sellable_volumes=sellable_volumes,
      pos_vals=pos_vals, cash=cash,
      buy_n_stocks=buy_n_stocks, tradable_buy_stocks=tradable_buy_stocks,
      sellable_ok=sellable_ok, prices=prices, limit_prices=limit_prices,
      base_target=base_target,
      keep_stocks=buy_n_stocks if position_control_active else sell_m_stocks,
      rebalance=rebalance or position_control_active,
      slippage_bps=slippage_bps,
      rebalance_band_pct=rebalance_band_pct,
    )

  return RebalanceDayPlan(
    buy_n_stocks=buy_n_stocks,
    sell_m_stocks=sell_m_stocks,
    final_score=final_score,
    prices=prices,
    close_prices=close_prices,
    limit_prices=limit_prices,
    pos_vals=pos_vals,
    total_eq=total_eq,
    base_target=base_target,
    reserve_L=reserve_L,
    tradable_buy_stocks=tradable_buy_stocks,
    sellable_ok=sellable_ok,
    sell_orders=sell_orders,
    buy_orders=buy_orders,
    skip_reasons=skip_reasons,
    t1_ranking=t1_ranking,
  )
