"""QMT 交易回调 —— 4 个 callback 统一走 `live_trade_recorder.record_event`
落 `events_{T}.parquet`，再分发到 day_board 战报；本地日志保留用于排查。

设计原则（v3）：
  1. 单一事实来源：所有 QMT 推送先入 events，下游（fills/positions/report）皆派生
  2. 不在 callback 里做任何聚合 / 计算 / 网络 IO
  3. callback 异常不影响 QMT 主线程（每个分发块独立 try）
"""
from xtquant import xtconstant
from xtquant.xttrader import XtQuantTraderCallback
from xtquant.xttype import XtOrder, XtTrade, XtOrderError, XtCancelError

from trading.helper import get_order_status_label, get_order_type_label, get_price_type_label
from trading.logger import trading_logger
from data.db import get_stock_detail
from trading.persistence import (
    live_trade_recorder,
    EVT_ORDER, EVT_TRADE, EVT_ORDER_ERROR, EVT_CANCEL_ERROR,
)


def _stock_name(code: str) -> str:
  detail = get_stock_detail(code) if code else None
  return (detail.get('InstrumentName', '') if detail else '').strip()


class TraderCallback(XtQuantTraderCallback):

  def __init__(self, trader):
    self.trader = trader

  def on_connected(self):
    trading_logger.success("交易已连接")

  def on_disconnected(self):
    trading_logger.error("交易连接已断开")

  def on_stock_order(self, order: XtOrder):
    name = _stock_name(order.stock_code) or order.stock_code
    op_label = get_order_type_label(order.order_type)
    status_label = get_order_status_label(order.order_status)
    price_label = get_price_type_label(order.price_type)
    status = order.order_status

    if status == xtconstant.ORDER_SUCCEEDED:
      trading_logger.success(f"已成: {op_label} {name} {price_label} {order.order_volume}股")
    elif status in (xtconstant.ORDER_CANCELED, xtconstant.ORDER_JUNK, xtconstant.ORDER_PART_CANCEL):
      trading_logger.warning(f"废单/已撤: {op_label} {name} [{status_label}] {order.status_msg}")
    elif status == xtconstant.ORDER_REPORTED:
      trading_logger.info(f"已报: {op_label} {name} {price_label} {order.order_volume}股")
    else:
      trading_logger.info(f"订单状态变更: {op_label} {name} [{status_label}]")

    try:
      live_trade_recorder.record_event(
          EVT_ORDER,
          code=order.stock_code, name=name,
          order_id=int(order.order_id),
          order_type=int(order.order_type),
          order_status=int(order.order_status),
          order_volume=int(order.order_volume or 0),
          traded_volume=int(getattr(order, 'traded_volume', 0) or 0),
          price=float(order.price or 0),
          traded_price=float(getattr(order, 'traded_price', 0) or 0),
          status_msg=order.status_msg or '',
      )
    except Exception as e:
      trading_logger.warning(f"events.order 落盘失败: {e}")

    try:
      from .day_board import day_board
      day_board.record_order(order)
    except Exception as e:
      trading_logger.warning(f"战报订单状态更新失败: {e}")

  def on_stock_trade(self, trade: XtTrade):
    name = _stock_name(trade.stock_code) or trade.stock_code
    op_label = get_order_type_label(trade.order_type)
    trading_logger.success(
        f"成交: {op_label} {name} ¥{trade.traded_price:.2f} × {trade.traded_volume}股 "
        f"≈ ¥{trade.traded_amount:.2f}"
    )

    direction = 'buy' if trade.order_type == xtconstant.STOCK_BUY else 'sell'
    try:
      live_trade_recorder.record_event(
          EVT_TRADE,
          code=trade.stock_code, name=name,
          order_id=int(trade.order_id),
          traded_id=str(getattr(trade, 'traded_id', '') or ''),
          order_type=int(trade.order_type),
          direction=direction,
          traded_price=float(trade.traded_price),
          traded_volume=int(trade.traded_volume),
          amount=float(trade.traded_amount or 0),
      )
    except Exception as e:
      trading_logger.warning(f"events.trade 落盘失败: {e}")

    try:
      from .day_board import day_board
      day_board.record_trade(trade)
    except Exception as e:
      trading_logger.warning(f"战报成交记录失败: {e}")

  def on_order_error(self, order_error: XtOrderError):
    """订单错误 — events + 战报聚合（不再发独立卡）。"""
    trading_logger.warning(f"订单错误：{order_error.error_msg}")
    code = getattr(order_error, 'stock_code', '') or ''
    try:
      live_trade_recorder.record_event(
          EVT_ORDER_ERROR,
          code=code, name=_stock_name(code) if code else '',
          order_id=int(getattr(order_error, 'order_id', 0) or 0),
          status_msg=getattr(order_error, 'error_msg', '') or '',
      )
    except Exception as e:
      trading_logger.warning(f"events.order_error 落盘失败: {e}")

    try:
      from .day_board import day_board
      day_board.record_order_error(order_error)
    except Exception as e:
      trading_logger.warning(f"战报错误聚合失败: {e}")

  def on_cancel_error(self, cancel_error: XtCancelError):
    trading_logger.warning(f"撤单失败：{cancel_error.error_msg}")
    code = getattr(cancel_error, 'stock_code', '') or ''
    try:
      live_trade_recorder.record_event(
          EVT_CANCEL_ERROR,
          code=code, name=_stock_name(code) if code else '',
          order_id=int(getattr(cancel_error, 'order_id', 0) or 0),
          status_msg=getattr(cancel_error, 'error_msg', '') or '',
      )
    except Exception as e:
      trading_logger.warning(f"events.cancel_error 落盘失败: {e}")
