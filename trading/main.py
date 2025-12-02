if __name__ == '__main__':
  import threading
  from xtquant import xtdata
  from datetime import datetime, time

  from configs import TRADE_ACCOUNT
  from trading.logger import trading_logger
  from core.sizer import MIN_BUY_AMOUNT
  from core.database import get_stock_detail
  from utils.stock.info import is_stock_trading
  from utils.recorder import recorder
  from .lark.receiver import create_lark_handler
  from .subscribe import subscribe_stock, unsubscribe_stock
  from .scheduler import TradingScheduler
  from .trader import Trader, get_position
  from .actions.buy import to_quick_buy_stocks
  from .actions.sell import to_sell_handler

  td = Trader(TRADE_ACCOUNT)

  # 创建飞书事件处理器
  threading.Thread(target=create_lark_handler, args=[td], daemon=True).start()

  def before_trade(store: TradingScheduler):
    # 监听全市场行情
    store.whole_sub_id = xtdata.subscribe_whole_quote(['SH', 'SZ'])
    # 监听持仓
    for position in get_position(store.trader, True):
      if position.can_use_volume > 0 and is_stock_trading(get_stock_detail(position.stock_code)):
        subscribe_stock(position.stock_code, to_sell_handler(store.trader, position.stock_code))

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
          bought_list = to_quick_buy_stocks(store.trader, store.trader.order)
          trading_logger.debug(f"选股结束! 共选出 {len(bought_list)} 只股票")
      finally:
        store.finding_stocks = False

  def after_trade(store: TradingScheduler):
    store.finding_stocks = False
    # 取消订阅
    xtdata.unsubscribe_quote(store.whole_sub_id)
    for position in get_position(store.trader, True):
      unsubscribe_stock(position.stock_code)

  # 创建交易调度器
  scheduler = TradingScheduler(
    td,
    before_trade=before_trade,
    while_trade=[buy_task],
    after_trade=after_trade,
  )

  scheduler.start_check_trading()
