"""单日调仓决策链。

回测和实盘只能通过本模块生成调仓计划；`trading` 只负责把计划交给券商执行。
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable

import numpy as np

from core.rebalance import compute_rebalance_plan, freeze_unit_price
from core.scoring import select_topn_legal
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
  from core.backtest import _compute_factor_scores, apply_t1_prefilter_for_date, build_list_dates_map
  from core.legality import LegalityChecker
  from utils.stock.time import get_last_trading_day

  weights = individual_config['weights']
  prefilter_n = individual_config['prefilter_n']
  data = None
  if prefilter_n:
    prev_date = get_last_trading_day(trade_date - timedelta(days=1))
    all_stocks, data = apply_t1_prefilter_for_date(
      prev_date, all_stocks, weights, factor_classes, prefilter_n,
      filter_factor_classes=filter_factor_classes,
    )
  if kline_loader is not None:
    if kline_data is not None:
      raise ValueError('pass either kline_loader or kline_data')
    kline_data = kline_loader(all_stocks)

  scored = _compute_factor_scores(
    [datetime.combine(trade_date, datetime.min.time())], all_stocks,
    weights, factor_classes, data=data, kline_data=kline_data,
    filter_factor_classes=filter_factor_classes or None,
  )
  if scored is None:
    raise ValueError(f'signal date {trade_date} is outside runtime data')
  data, all_scores, filter_masks, valid_dates, date_indices, valid_stocks, stock_indices = scored
  date_idx = date_indices[0]
  valid_cols = np.array([stock_indices[s] for s in valid_stocks], dtype=np.intp)
  buy_filter_mask = None
  if filter_masks:
    buy_filter_mask = np.logical_and.reduce(
      [mask[date_idx][valid_cols] for mask in filter_masks.values()]
    )
  checker = LegalityChecker(
    data, stock_indices, build_list_dates_map(data),
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
    position_multiplier: float = 1.0,
    target_cash: float | None = None,
    target_positions: dict[str, int] | None = None,
    price_codes_extra=None,
    last_valid_close_prices: dict[str, float] | None = None,
    market_order_freeze: bool = True,
    buy_filter_mask: np.ndarray | None = None,
) -> RebalanceDayPlan:
  if is_rebalance_day:
    day_open = data['open'][trade_idx]
    buy_n_stocks, sell_m_stocks, final_score, t1_ranking = select_topn_legal(
      all_scores, date_idx, valid_stocks, valid_cols,
      weights, buy_n, sell_m,
      checker=checker, trade_idx=trade_idx, signal_date=signal_date,
      day_open=day_open, stock_indices=stock_indices,
      filter_mask=buy_filter_mask,
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
    sell_orders, buy_orders, skip_reasons = compute_rebalance_plan(
      positions=positions, sellable_volumes=sellable_volumes,
      pos_vals=pos_vals, cash=cash,
      buy_n_stocks=buy_n_stocks, tradable_buy_stocks=tradable_buy_stocks,
      sellable_ok=sellable_ok, prices=prices, limit_prices=limit_prices,
      base_target=base_target, keep_stocks=sell_m_stocks, rebalance=rebalance,
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
