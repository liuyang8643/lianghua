from xtquant import xtconstant
from xtquant.xttrader import XtQuantTraderCallback
from xtquant.xttype import XtOrder, XtTrade, XtOrderError, XtCancelError

from trading.helper import get_order_status_label, get_order_type_label, get_price_type_label
from trading.lark.sender import LarkMsgLevel, lark_sender
from trading.logger import trading_logger
from core.database import get_stock_detail
from utils.stock.format import get_stock_desc
from trading.subscribe import unsubscribe_stock

class TraderCallback(XtQuantTraderCallback):
  """ 订阅账户变更 """

  def __init__(self, trader):
    self.trader = trader

  def on_connected(self):
    trading_logger.success("交易已连接")

  def on_disconnected(self):
    trading_logger.error("交易连接已断开")

  def on_stock_order(self, order: XtOrder):
    if order.order_status not in [
      xtconstant.ORDER_REPORTED,
      xtconstant.ORDER_CANCELED,
    ]:
      # 只处理已报和已撤的订单
      return

    detail = get_stock_detail(order.stock_code)
    sub_title = f"{get_order_type_label(order.order_type)} {get_stock_desc(detail)}"
    content = f"""\
委托信息：{f'{get_price_type_label(order.price_type)} {order.order_volume} 股' if order.price is None else f'￥{order.price:.2f} * {order.order_volume}股 ≈ ￥{(order.price * order.order_volume):.2f}'}
委托状态：[{get_order_status_label(order.order_status)}]{order.status_msg}
<hr/>{order.strategy_name} {order.order_remark}"""

    lark_sender.send_notification_card(
      level=LarkMsgLevel.Info,
      title="订单已提交",
      sub_title=sub_title,
      content=content
    )
    trading_logger.debug(f"订单已提交：\n{sub_title}\n{content}")

  def on_stock_trade(self, trade: XtTrade):
    detail = get_stock_detail(trade.stock_code)
    sub_title = f"{get_order_type_label(trade.order_type)} {get_stock_desc(detail)}"
    content = f"""\
成交信息：￥{trade.traded_price:.2f} * {trade.traded_volume}股
成交金额：￥{trade.traded_amount:.2f} + 手续费 ￥{trade.commission:.2f}
<hr/>{trade.strategy_name} {trade.order_remark}"""

    lark_sender.send_notification_card(
      level=LarkMsgLevel.Success,
      title="订单已成交",
      sub_title=sub_title,
      content=content
    )
    trading_logger.success(f"订单已成交：\n{sub_title}\n{content}")

    # 完全清仓则取消订阅
    if trade.order_type == xtconstant.STOCK_SELL:
      position = self.trader.query_stock_position(trade.stock_code)
      if not position or not position.can_use_volume:
        unsubscribe_stock(trade.stock_code)

  def on_order_error(self, order_error: XtOrderError):
    trading_logger.error(f"订单错误：{order_error.error_msg}")

  def on_cancel_error(self, cancel_error: XtCancelError):
    trading_logger.error(f"撤单失败：{cancel_error.error_msg}")
