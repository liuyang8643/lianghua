from core.judge.sellSignal import get_sell_signal
from trading.trader import Trader
from trading.logger import trading_logger
from trading.subscribe import is_stock_subscribed
from utils.recorder import recorder
from utils.stock.format import get_stock_desc
from utils.stock.time import is_current_trading

def to_sell_handler(trader: Trader, code: str):
  """ 获取卖出处理函数 """
  executing = False

  @trading_logger.catch
  def handler(datas):
    """ 订阅持仓行情 """
    nonlocal executing

    if executing or not is_stock_subscribed(code):
      return

    try:
      # 加锁
      executing = True
      # 获取卖出原因
      result = get_sell_signal(code)
      recorder.mark(f"获取卖出信号 {code}")
      if result is not None:
        if not is_current_trading():
          trading_logger.warning(f"{get_stock_desc(result['detail'])} 当前非交易时间，中断处理...")
          return
        if not is_stock_subscribed(code):
          trading_logger.warning(f"{get_stock_desc(result['detail'])} 已取消订阅，中断处理...")
          return
        trader.clear_position(code, result['mark'])
        trading_logger.info(f"清仓 {get_stock_desc(result['detail'])}({result['expect_sell_price']:.2f}): {result['reason']}")
        recorder.mark("执行清仓指令")
    except Exception as e:
      trading_logger.error(f"处理过程中发生异常: {e}")
    finally:
      executing = False  # 解锁

  return handler
