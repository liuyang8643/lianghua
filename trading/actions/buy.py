from typing import Any, Callable, Optional
from xtquant import xtconstant

from core.judge.buySignal import get_buy_signal
from trading.trader import Trader, get_position
from trading.logger import trading_logger
from utils.recorder import recorder
from core.sizer import Sizer, get_quick_allocation
from core.database import allow_buy_stock_code_list
from utils.stock.format import get_stock_desc
from utils.stock.time import is_current_trading

@trading_logger.catch
def to_quick_buy_stocks(
    trader: Trader,
    order_callback: Callable[[int, str, int, Optional[float], Optional[str]], Any] = None
) -> list[ExpectBuyStock]:
  """ 筛出就买 """

  bought_stocks = [p.stock_code for p in get_position(trader, False)]
  trading_logger.debug(f"当前已持仓 {len(bought_stocks)} 只股票")
  allow_buy_list = [x for x in allow_buy_stock_code_list() if x not in bought_stocks]
  check_stock_count = 0

  def did_buy(x: str) -> Optional[ExpectBuyStock]:
    buy_info = get_buy_signal(x)
    if buy_info is not None:
      # 值得买
      asset = trader.query_asset()
      allocation = get_quick_allocation(
        {
          "code": buy_info["code"],
          "price": buy_info["latest_trade_data"]["close"]
        },
        asset.cash * 0.98 if asset else 0,
      )
      if allocation and allocation['count'] > 0:
        if order_callback:
          stock_desc = get_stock_desc(buy_info['detail'])
          if is_current_trading():
            order_callback(xtconstant.STOCK_BUY, allocation['code'], allocation['count'], None, buy_info['result']['factor'])
            trading_logger.info(f"下单买入 {stock_desc} * {allocation['count']}({buy_info['current_price']}) 股: {buy_info['result']['reason']}")
            recorder.mark("下单买入股票")
          else:
            trading_logger.warning(f"{stock_desc}当前非交易时间，中断买入...")
            return None

      return to_expect_buy(buy_info, allocation)

    # 每十分一汇报一次进度
    nonlocal check_stock_count
    check_stock_count += 1
    if check_stock_count % (len(allow_buy_list) // 10) == 0:
      trading_logger.debug(f"当前选股进度：{check_stock_count}/{len(allow_buy_list)}")
      return None
    return None

  trading_logger.debug(f"准备开始选股！共 {len(allow_buy_list)} 只股票可供选择！")
  worth_buy_list = [buy_info for buy_info in [did_buy(code) for code in allow_buy_list] if buy_info is not None]

  if len(worth_buy_list) > 0:
    trading_logger.debug(f"选股结束！已筛出 {len(worth_buy_list)} 只股票！")
  else:
    trading_logger.debug("选股结束！当前没有可以购买的股票！")

  return worth_buy_list

if __name__ == '__main__':
  from configs import TRADE_ACCOUNT
  from utils.sys import with_time_count

  td = Trader(TRADE_ACCOUNT)
  with_time_count('买入耗时', lambda: to_quick_buy_stocks(td, None))
