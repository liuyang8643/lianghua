"""Project-side compatibility adapter for the shared execution state machine.

Autonomous brokerage-app execution uses execution_engine directly.
"""
from data.db import get_stock_detail
from trading.logger import trading_logger
from trading.persistence import live_trade_recorder
from utils.recorder import recorder
from utils.stock.time import is_current_trading
from trading.execution_engine import (
    OrderMonitor, RebalanceExecutor as ExecutionEngine,
    TERMINAL_STATUS, CANCELABLE_STATUS, QMT_INSUFFICIENT_FUNDS_CODE,
    _is_insufficient_funds,
)


class RebalanceExecutor(ExecutionEngine):
    @property
    def logger(self):
        return trading_logger

    def _is_current_trading(self, now):
        return is_current_trading(now)

    def _record_event(self, *args, **kwargs):
        return live_trade_recorder.record_event(*args, **kwargs)

    def _stock_detail(self, code):
        return get_stock_detail(code)

    def _mark(self, message):
        recorder.mark(message)
