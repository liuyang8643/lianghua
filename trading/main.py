from core import allow_buy_stock_code_list
from core.strategies import TopN

if __name__ == '__main__':
  import threading
  import json
  import argparse
  from pathlib import Path
  from xtquant import xtdata
  from datetime import datetime, date, time

  from configs import TRADE_ACCOUNT
  from trading.logger import trading_logger
  from core.strategies.sizers.sizer import MIN_BUY_AMOUNT
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

  def buy_task(store: TradingScheduler):
    if (
        not store.finding_stocks
        and time(14, 40) <= datetime.now().time() <= time(14, 55)
    ):
      try:
        store.finding_stocks = True
        asset = store.trader.query_asset()
        if asset.cash > MIN_BUY_AMOUNT * 1.1 if asset else False:
          trading_logger.debug(f"开始选股")
          recorder.mark(f"开始选股")
          all_stocks = allow_buy_stock_code_list(date.today())
          topn = TopN(all_stocks, datetime.now())
          sorted_stocks = topn.get_ordered_stocks(
            n=buy_n,
            weights=weights,
            temperatures=temperatures,
            norm_method='rank'
          )
          trading_logger.debug(f"选股结束! 选出{len(sorted_stocks)}只股票")
      finally:
        store.finding_stocks = False

  def after_trade(store: TradingScheduler):
    # 取消订阅
    xtdata.unsubscribe_quote(store.whole_sub_id)

  # 创建交易调度器
  scheduler = TradingScheduler(
    td,
    before_trade=before_trade,
    while_trade=[buy_task],
    after_trade=after_trade,
  )

  scheduler.start_check_trading()
