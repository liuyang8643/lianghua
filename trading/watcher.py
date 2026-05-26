from xtquant import xtconstant
from xtquant.xttrader import XtQuantTraderCallback
from xtquant.xttype import XtOrder, XtTrade, XtOrderError, XtCancelError

from trading.helper import get_order_status_label, get_order_type_label, get_price_type_label
from trading.lark.sender import LarkMsgLevel, lark_sender
from trading.logger import trading_logger
from data.db import get_stock_detail
from utils.stock.format import get_stock_desc


class TraderCallback(XtQuantTraderCallback):

  def __init__(self, trader):
    self.trader = trader

  def on_connected(self):
    trading_logger.success("交易已连接")

  def on_disconnected(self):
    trading_logger.error("交易连接已断开")

  def on_stock_order(self, order: XtOrder):
    detail = get_stock_detail(order.stock_code)
    desc = f"{get_order_type_label(order.order_type)} {get_stock_desc(detail)}"
    status_label = get_order_status_label(order.order_status)
    price_info = f'{get_price_type_label(order.price_type)} {order.order_volume} 股'
    status = order.order_status

    if status == xtconstant.ORDER_SUCCEEDED:
        trading_logger.success(f"已成: {desc} {price_info}")
        lark_sender.send_notification_card(
            level=LarkMsgLevel.Success, title="已成", sub_title=desc,
            content=f"{price_info}")
    elif status in (xtconstant.ORDER_CANCELED, xtconstant.ORDER_JUNK, xtconstant.ORDER_PART_CANCEL):
        trading_logger.error(f"废单/已撤: {desc} [{status_label}] {order.status_msg}")
        lark_sender.send_notification_card(
            level=LarkMsgLevel.Danger, title="废单/已撤", sub_title=desc,
            content=f"{price_info} {order.status_msg}")
    elif status == xtconstant.ORDER_REPORTED:
        trading_logger.info(f"已报: {desc} {price_info}")
        lark_sender.send_notification_card(
            level=LarkMsgLevel.Info, title="已报", sub_title=desc,
            content=f"{price_info}")

  def on_stock_trade(self, trade: XtTrade):
    detail = get_stock_detail(trade.stock_code)
    desc = f"{get_order_type_label(trade.order_type)} {get_stock_desc(detail)}"
    lark_sender.send_notification_card(
        level=LarkMsgLevel.Success, title="成交", sub_title=desc,
        content=f"￥{trade.traded_price:.2f} * {trade.traded_volume}股 ≈ ￥{trade.traded_amount:.2f}")

    try:
      from .persistence import live_trade_recorder
      direction = 'buy' if trade.order_type == xtconstant.STOCK_BUY else 'sell'
      amt = float(trade.traded_amount)
      fee = max(amt * 0.0000854, 0.1) + amt * 0.00002
      if direction == 'sell':
        fee += amt * 0.0005
      live_trade_recorder.record_fill(
          code=trade.stock_code, direction=direction,
          price=float(trade.traded_price), shares=int(trade.traded_volume),
          amount=amt, order_id=trade.order_id,
          name=detail.get('InstrumentName', '') if detail else '', fee=fee)
    except Exception as e:
      trading_logger.warning(f"记录成交失败: {e}")

  def on_order_error(self, order_error: XtOrderError):
    trading_logger.error(f"订单错误：{order_error.error_msg}")
    lark_sender.send_notification_card(
        level=LarkMsgLevel.Danger, title="订单错误", content=order_error.error_msg)

  def on_cancel_error(self, cancel_error: XtCancelError):
    trading_logger.error(f"撤单失败：{cancel_error.error_msg}")
