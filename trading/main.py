from core import allow_buy_stock_code_list, get_market_data, get_market_data_batch
from core.strategies import TopN
from core.strategies.sizers import Sizer
from utils.stock.time import is_current_trading

if __name__ == '__main__':
  import threading
  import json
  import argparse
  from xtquant import xtconstant, xtdata
  from datetime import datetime, date, time

  from configs import TRADE_ACCOUNT
  from trading.logger import trading_logger
  from utils.recorder import recorder
  from .lark.receiver import create_lark_handler
  from .scheduler import TradingScheduler
  from .trader import Trader

  parser = argparse.ArgumentParser()
  parser.add_argument('--individual-config', type=str, required=True, help='Individual_config JSON文件路径')
  args = parser.parse_args()

  with open(args.individual_config, 'r', encoding='utf-8') as f:
    config_data = json.load(f)
  individual_config = config_data['individual_config']
  weights = individual_config['weights']
  temperatures = individual_config['temperatures']
  buy_n = individual_config['buy_n']
  sell_m = individual_config['sell_m']

  trading_logger.info(f"加载Individual_config: {args.individual_config}")
  trading_logger.info(f"配置参数: buy_n={buy_n}, sell_m={sell_m}")

  td = Trader(TRADE_ACCOUNT)

  threading.Thread(target=create_lark_handler, args=[td], daemon=True).start()

  def before_trade(store: TradingScheduler):
    store.whole_sub_id = xtdata.subscribe_whole_quote(['SH', 'SZ'])
    ''' 选股 '''
    asset = store.trader.query_asset()
    trading_logger.debug(f"开始选股")
    recorder.mark(f"开始选股")

    all_stocks = allow_buy_stock_code_list(date.today())
    get_market_data_batch(all_stocks, 2, dividend_type='front')  # 预加载数据

    # 获取当日 Top sell_m 只股票用于判断卖出
    sell_m_stocks = TopN(all_stocks, datetime.now()).get_ordered_stocks(
      n=sell_m,
      weights=weights,
      temperatures=temperatures,
      norm_method='rank'
    )
    trading_logger.debug(f"待卖出前 {sell_m} 名股票: {sell_m_stocks}")

    # 卖出不在Top sell_m中的股票
    for code in set([p.stock_code for p in store.trader.query_positions()]) - set(sell_m_stocks):
      store.trader.clear_position(code)
      trading_logger.info(f"已清仓股票: {code}")
      recorder.mark("清仓股票")

    # 获取当日 Top buy_n 只股票用于买入
    buy_n_stocks = TopN(all_stocks, datetime.now()).get_ordered_stocks(
      n=buy_n,
      weights=weights,
      temperatures=temperatures,
      norm_method='rank'
    )

    trading_logger.debug(f"待买入前 {buy_n} 名股票: {buy_n_stocks}")
    prices = {}
    for code in buy_n_stocks:
      try:
        data = get_market_data(code, 1, dividend_type='front')
        if data is not None and len(data) > 0:
          prices[code] = float(data.iloc[-1]['close'])
      except Exception as e:
        trading_logger.warning(f"{code} 价格获取失败: {e}")
    trading_logger.debug(f"开始计算仓位分配...")
    allocations = Sizer.allocate(
      [(s, prices[s]) for s in buy_n_stocks],
      asset.total_asset
    )
    trading_logger.debug(f"计算仓位分配完成: {allocations}")
    positions = {p.stock_code: p for p in store.trader.query_positions()}
    for code, shares in allocations.items():
      if shares <= 0:
        continue
      if positions[code]:
        continue
      if is_current_trading():
        store.trader.order(xtconstant.STOCK_BUY, code, shares, None)
        trading_logger.info(f"下单买入 {code} * {shares} 股")
        recorder.mark("下单买入股票")
      else:
        trading_logger.warning(f"{code}当前非交易时间，中断买入...")
    trading_logger.debug(f"选股结束! ")

  def after_trade(store: TradingScheduler):
    # 取消订阅
    xtdata.unsubscribe_quote(store.whole_sub_id)

  # 创建交易调度器
  scheduler = TradingScheduler(
    td,
    before_trade=before_trade,
    while_trade=[],
    after_trade=after_trade,
  )

  scheduler.start_check_trading()
