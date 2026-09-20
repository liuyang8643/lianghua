"""QMT callbacks for operator logs.

Broker truth is persisted only by ``trade.journal.DecisionJournal`` after the
canonical executor has reconciled terminal orders. Callback notifications are
never a second fill or position store.
"""

from xtquant import xtconstant
from xtquant.xttrader import XtQuantTraderCallback
from xtquant.xttype import XtCancelError, XtOrder, XtOrderError, XtTrade

from data.db import get_stock_detail
from trade.broker.helper import (
    get_order_status_label,
    get_order_type_label,
    get_price_type_label,
)
from trade.logger import trading_logger


def _stock_name(code: str) -> str:
    detail = get_stock_detail(code) if code else None
    return (detail.get("InstrumentName", "") if detail else "").strip()


class TraderCallback(XtQuantTraderCallback):
    def __init__(self, trader):
        self.trader = trader
        self._seen_order_status: dict[int, tuple[object, ...]] = {}
        self._seen_errors: set[tuple[int, str]] = set()

    def on_connected(self):
        trading_logger.success("交易已连接")

    def on_disconnected(self):
        trading_logger.error("交易连接已断开")

    def on_stock_order(self, order: XtOrder):
        name = _stock_name(order.stock_code) or order.stock_code
        operation = get_order_type_label(order.order_type)
        status_label = get_order_status_label(order.order_status)
        price_label = get_price_type_label(order.price_type)
        traded_volume = int(getattr(order, "traded_volume", 0) or 0)
        traded_price = float(getattr(order, "traded_price", 0) or 0)
        if order.order_status == xtconstant.ORDER_SUCCEEDED:
            trading_logger.success(
                f"已成: order_id={order.order_id} {operation} {name} {price_label} "
                f"委托{order.order_volume}股 成交{traded_volume}股 "
                f"委托价={float(order.price or 0):.4f} 成交价={traded_price:.4f}"
            )
        elif order.order_status in (
            xtconstant.ORDER_CANCELED,
            xtconstant.ORDER_JUNK,
            xtconstant.ORDER_PART_CANCEL,
        ):
            key = (order.order_status, order.status_msg)
            if self._seen_order_status.get(order.order_id) != key:
                self._seen_order_status[order.order_id] = key
                trading_logger.warning(
                    f"废单/已撤: order_id={order.order_id} {operation} {name} "
                    f"[{status_label}] 委托{order.order_volume}股 成交{traded_volume}股 "
                    f"委托价={float(order.price or 0):.4f} msg={order.status_msg}"
                )
        elif order.order_status == xtconstant.ORDER_REPORTED:
            trading_logger.info(
                f"已报: order_id={order.order_id} {operation} {name} {price_label} "
                f"委托{order.order_volume}股 委托价={float(order.price or 0):.4f}"
            )
        else:
            trading_logger.info(
                f"订单状态变更: order_id={order.order_id} {operation} {name} "
                f"[{status_label}] 委托{order.order_volume}股 成交{traded_volume}股 "
                f"msg={order.status_msg}"
            )

    def on_stock_trade(self, trade: XtTrade):
        name = _stock_name(trade.stock_code) or trade.stock_code
        operation = get_order_type_label(trade.order_type)
        trading_logger.success(
            f"成交: order_id={trade.order_id} "
            f"traded_id={getattr(trade, 'traded_id', '')} {operation} {name} "
            f"¥{trade.traded_price:.4f} × {trade.traded_volume}股 "
            f"≈ ¥{trade.traded_amount:.2f}"
        )

    def on_order_error(self, order_error: XtOrderError):
        order_id = int(getattr(order_error, "order_id", 0) or 0)
        message = getattr(order_error, "error_msg", "") or ""
        key = (order_id, message)
        if key not in self._seen_errors:
            self._seen_errors.add(key)
            trading_logger.warning(f"订单错误：{message}")

    def on_cancel_error(self, cancel_error: XtCancelError):
        trading_logger.warning(f"撤单失败：{cancel_error.error_msg}")
