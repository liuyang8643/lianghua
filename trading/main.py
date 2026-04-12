from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import argparse
import json
import sys
import threading
from types import SimpleNamespace

from xtquant import xtconstant, xtdata

from configs import TRADE_ACCOUNT
from core import allow_buy_stock_code_list, get_market_data_batch
from core.strategies import TopN
from core.strategies.sizers import Sizer
from testback.ga_config import get_profile_factor_classes, resolve_profile_name
from trading.logger import trading_logger
from utils.recorder import recorder
from utils.stock.time import AFTERNOON_END, get_last_trading_day
from utils.stock.info import evaluate_orderability

from .lark.receiver import create_lark_handler
from .manual_confirm import build_manual_confirmation_text, is_manual_confirmation_approved
from .scheduler import TradingScheduler
from .trader import Trader


def _get_signal_date(trade_date):
  signal_date = get_last_trading_day(trade_date)
  if signal_date >= trade_date:
    signal_date = get_last_trading_day(trade_date - timedelta(days=1))
  return signal_date


def _submit_sell_order(store: TradingScheduler, code: str, signal_date, trade_date):
  try:
    store.trader.clear_position(code, reason=f'调仓换出 signal={signal_date.isoformat()} trade={trade_date.isoformat()}')
    trading_logger.info(f"已提交卖出委托: {code}")
    recorder.mark("提交卖出委托")
  except ValueError as e:
    trading_logger.info(f"{code} 卖出前校验拦截: {e}")
  except Exception as e:
    trading_logger.exception(f"{code} 卖出委托失败: {e}")


def _submit_buy_order(store: TradingScheduler, code: str, shares: int, signal_date, trade_date):
  try:
    store.trader.order(
      xtconstant.STOCK_BUY,
      code,
      shares,
      None,
      order_remark=f'调仓买入 signal={signal_date.isoformat()} trade={trade_date.isoformat()}'
    )
    trading_logger.info(f"已提交买入委托: {code} * {shares} 股")
    recorder.mark("提交买入委托")
  except ValueError as e:
    trading_logger.info(f"{code} 买入前校验拦截: {e}")
  except Exception as e:
    trading_logger.exception(f"{code} 买入委托失败: {e}")


def _print_and_confirm_manual_plan(pending: dict, execute_buy: bool, execute_sell: bool) -> bool:
  message = build_manual_confirmation_text(pending, execute_buy=execute_buy, execute_sell=execute_sell)
  print(message)
  user_input = input("确认执行> ")
  return is_manual_confirmation_approved(user_input)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--individual-config', type=str, required=True, help='Individual_config JSON文件路径')
  parser.add_argument('--buy', action='store_true', help='忽略时间窗口，立即执行一次买入（需手工yes确认）')
  parser.add_argument('--sell', action='store_true', help='忽略时间窗口，立即执行一次卖出（需手工yes确认）')
  args = parser.parse_args()

  with open(args.individual_config, 'r', encoding='utf-8') as f:
    config_data = json.load(f)
  profile_name = resolve_profile_name(config_data)
  factor_classes = get_profile_factor_classes(profile_name)
  individual_config = config_data['individual_config']
  weights = individual_config['weights']
  temperatures = individual_config['temperatures']
  buy_n = individual_config['buy_n']
  sell_m = individual_config['sell_m']

  trading_logger.info(f"加载Individual_config: {args.individual_config}")
  trading_logger.info(f"配置参数: buy_n={buy_n}, sell_m={sell_m}, factors={sorted(weights.keys())}")

  td = Trader(TRADE_ACCOUNT)

  threading.Thread(target=create_lark_handler, args=[td], daemon=True).start()

  def before_trade(store: TradingScheduler):
    trade_now = datetime.now()
    trade_date = trade_now.date()
    signal_date = _get_signal_date(trade_date)
    signal_datetime = datetime.combine(signal_date, datetime.min.time())

    trading_logger.info(
      f"开始预计算调仓: signal_date={signal_date.isoformat()}, trade_date={trade_date.isoformat()}, "
      f"signal_dividend_type=back, execution_price_type=peer_price_first"
    )
    recorder.mark("开始选股")

    if store.whole_sub_id is None:
      store.whole_sub_id = xtdata.subscribe_whole_quote(['SH', 'SZ'])

    asset = store.trader.query_asset()
    if asset is None:
      trading_logger.warning("资产查询失败，跳过本轮调仓")
      store.pending_rebalance = None
      return

    all_stocks = allow_buy_stock_code_list()
    filtered_stocks = list(all_stocks)
    trading_logger.info(f"候选股票池: {len(filtered_stocks)} 只")
    get_market_data_batch(filtered_stocks, 2, base_time=signal_datetime, dividend_type='back')

    trade_bar_time = datetime.combine(trade_date, AFTERNOON_END)

    def _load_trade_bars(stock_codes: list[str]):
      trade_bar_data = get_market_data_batch(
        stock_codes,
        1,
        base_time=trade_bar_time,
        period='1d',
        allow_tainted=True,
        dividend_type='none',
        strict_trade_date=True,
      )
      return {
        code: (data.iloc[-1] if data is not None and not data.empty else None)
        for code, data in trade_bar_data.items()
      }

    topn = TopN(
      filtered_stocks,
      signal_datetime,
      weights=weights,
      factor_classes=factor_classes,
      dividend_type='back'
    )
    sell_m_stocks = topn.get_ordered_stocks(
      n=sell_m,
      temperatures=temperatures,
      norm_method='rank'
    )
    buy_n_stocks = topn.get_ordered_stocks(
      n=buy_n,
      temperatures=temperatures,
      norm_method='rank'
    )

    positions = {p.stock_code: p for p in store.trader.query_positions()}
    sell_candidates = sorted(set(positions) - set(sell_m_stocks))
    sell_bars = _load_trade_bars(sell_candidates)
    allowed_sell_codes = []
    sell_details = []
    for code in sell_candidates:
      orderability = evaluate_orderability('sell', code, trade_date, bar=sell_bars.get(code), dividend_type='none')
      if not orderability['allowed']:
        trading_logger.info(
          f"{code} 跳过卖出: reason={orderability['reason']}, regime={orderability['regime']}, "
          f"up_limit={orderability['up_limit']}, down_limit={orderability['down_limit']}"
        )
        continue
      allowed_sell_codes.append(code)
      position = positions.get(code)
      bar = sell_bars.get(code)
      volume = int(position.can_use_volume) if position is not None else 0
      est_price = float(bar['open']) if bar and bar.get('open') is not None else 0.0
      sell_details.append({
        'code': code,
        'volume': volume,
        'est_price': est_price,
        'est_amount': volume * est_price,
      })

    buy_trade_bars = _load_trade_bars(buy_n_stocks)
    tradable_buy_stocks = []
    sizing_prices = {}
    for code in buy_n_stocks:
      orderability = evaluate_orderability('buy', code, trade_date, bar=buy_trade_bars.get(code), dividend_type='none')
      if not orderability['allowed']:
        trading_logger.info(
          f"{code} 跳过买入: reason={orderability['reason']}, regime={orderability['regime']}, "
          f"up_limit={orderability['up_limit']}, down_limit={orderability['down_limit']}"
        )
        continue
      if positions.get(code):
        continue
      sizing_bar = buy_trade_bars.get(code)
      if sizing_bar is None or sizing_bar.get('open') is None:
        trading_logger.info(f"{code} 跳过买入: live sizing 缺少 open")
        continue
      sizing_prices[code] = float(sizing_bar['open'])
      tradable_buy_stocks.append(code)

    allocations = Sizer.allocate(
      [(s, sizing_prices[s]) for s in tradable_buy_stocks],
      asset.total_asset
    )
    allocations = {code: shares for code, shares in allocations.items() if shares > 0}
    buy_details = [
      {
        'code': code,
        'shares': int(shares),
        'est_price': float(sizing_prices[code]),
        'est_amount': int(shares) * float(sizing_prices[code]),
      }
      for code, shares in allocations.items()
    ]

    store.pending_rebalance = {
      'signal_date': signal_date,
      'trade_date': trade_date,
      'sell_codes': allowed_sell_codes,
      'buy_allocations': allocations,
      'sell_details': sell_details,
      'buy_details': buy_details,
    }

    trading_logger.info(
      f"调仓预计算完成: sell_candidates={len(allowed_sell_codes)}, buy_candidates={len(allocations)}, "
      f"signal_date={signal_date.isoformat()}, trade_date={trade_date.isoformat()}"
    )
    recorder.mark("完成调仓预计算")

  def execute_trade(store: TradingScheduler, execute_sell: bool = True, execute_buy: bool = True):
    pending = getattr(store, 'pending_rebalance', None)
    if not pending:
      trading_logger.warning("没有可执行的调仓计划，跳过本轮下单")
      return

    signal_date = pending['signal_date']
    trade_date = pending['trade_date']
    sell_codes = pending['sell_codes']
    buy_allocations = pending['buy_allocations']

    planned_sell_count = len(sell_codes) if execute_sell else 0
    planned_buy_count = len(buy_allocations) if execute_buy else 0
    trading_logger.info(
      f"开始执行调仓: sell_orders={planned_sell_count}, buy_orders={planned_buy_count}, "
      f"signal_date={signal_date.isoformat()}, trade_date={trade_date.isoformat()}"
    )

    if execute_sell and sell_codes:
      with ThreadPoolExecutor(max_workers=min(16, len(sell_codes))) as executor:
        futures = [executor.submit(_submit_sell_order, store, code, signal_date, trade_date) for code in sell_codes]
        for future in as_completed(futures):
          future.result()

    if execute_buy and buy_allocations:
      with ThreadPoolExecutor(max_workers=min(16, len(buy_allocations))) as executor:
        futures = [
          executor.submit(_submit_buy_order, store, code, shares, signal_date, trade_date)
          for code, shares in buy_allocations.items()
        ]
        for future in as_completed(futures):
          future.result()

    store.pending_rebalance = None
    trading_logger.debug("调仓执行结束")

  def after_trade(store: TradingScheduler):
    if store.whole_sub_id is not None:
      xtdata.unsubscribe_quote(store.whole_sub_id)
      store.whole_sub_id = None
    store.pending_rebalance = None

  scheduler = TradingScheduler(
    td,
    before_trade=before_trade,
    execute_trade=execute_trade,
    while_trade=[],
    after_trade=after_trade,
  )

  manual_trigger = args.buy or args.sell
  if manual_trigger:
    execute_buy = args.buy
    execute_sell = args.sell
    if args.buy and args.sell:
      execute_buy = True
      execute_sell = True

    store = SimpleNamespace(trader=td, whole_sub_id=None, pending_rebalance=None)
    try:
      before_trade(store)
      pending = getattr(store, 'pending_rebalance', None)
      if not pending:
        trading_logger.warning("没有生成可执行计划，已取消本次手动触发")
        sys.exit(0)

      confirmed = _print_and_confirm_manual_plan(
        pending,
        execute_buy=execute_buy,
        execute_sell=execute_sell,
      )
      if not confirmed:
        trading_logger.warning("手动确认未通过，已取消本次交易")
        sys.exit(0)

      execute_trade(store, execute_sell=execute_sell, execute_buy=execute_buy)
      trading_logger.info("手动单次触发执行完成")
    finally:
      after_trade(store)
    sys.exit(0)

  scheduler.start_check_trading()
