if __name__ == '__main__':
  import threading
  from xtquant import xtdata
  from datetime import datetime, time

  from configs import TRADE_ACCOUNT
  from trading.logger import trading_logger
  from core.sizer import MIN_BUY_AMOUNT
  from utils.recorder import recorder
  from .lark.receiver import create_lark_handler
  from .subscribe import subscribe_stock, unsubscribe_stock
  from .scheduler import TradingScheduler
  from .trader import Trader, get_position

  td = Trader(TRADE_ACCOUNT)

  # 创建飞书事件处理器
  threading.Thread(target=create_lark_handler, args=[td], daemon=True).start()

  def before_trade(store: TradingScheduler):
    # 监听全市场行情
    store.whole_sub_id = xtdata.subscribe_whole_quote(['SH', 'SZ'])

  def buy_task(store: TradingScheduler):
    if (
        not store.finding_stocks
        and time(14, 40) <= datetime.now().time() <= time(14, 55)
    ):
      # 尾盘买入
      try:
        store.finding_stocks = True
        asset = store.trader.query_asset()
        if asset.cash > MIN_BUY_AMOUNT * 1.1 if asset else False:
          trading_logger.debug(f"开始选股")
          recorder.mark(f"开始选股")
          # TODO 实现选股及购买逻辑
          trading_logger.debug(f"选股结束! ")
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
