"""One-time mechanical extraction; keep source comments and method bodies."""
import ast
from pathlib import Path

path = Path('trading/executor.py')
source = path.read_text(encoding='utf-8')
Path('tmp/executor_before_file_mode.py').write_text(source, encoding='utf-8')
tree = ast.parse(source)
lines = source.splitlines(keepends=True)


def compatible(node):
    start = node.lineno - 1
    text = ''.join(lines[start:node.end_lineno])
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    # AST columns are UTF-8 bytes, while edits below use unicode offsets.
    def offset(row, col):
        line = lines[row - 1]
        return offsets[row - 1 - start] + len(line.encode()[:col].decode())
    edits = []
    for item in ast.walk(node):
        if isinstance(item, ast.arg) and item.annotation:
            a = item.annotation
            begin = offset(item.lineno, item.col_offset) + len(item.arg)
            edits.append((begin, offset(a.end_lineno, a.end_col_offset), ''))
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.returns:
            a = item.returns
            end = offset(a.end_lineno, a.end_col_offset)
            begin = text.rfind('->', 0, offset(a.lineno, a.col_offset))
            edits.append((begin, end, ''))
        elif isinstance(item, ast.AnnAssign):
            a = item.annotation
            begin = offset(item.target.end_lineno, item.target.end_col_offset)
            edits.append((begin, offset(a.end_lineno, a.end_col_offset), ''))
    for begin, end, value in sorted(edits, reverse=True):
        text = text[:begin] + value + text[end:]
    return text


parts = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
            ('TERMINAL_STATUS', 'CANCELABLE_STATUS', 'QMT_INSUFFICIENT_FUNDS_CODE') for t in node.targets):
        parts.append(compatible(node))
    elif isinstance(node, ast.FunctionDef) and node.name == '_is_insufficient_funds':
        parts.append(compatible(node))
    elif isinstance(node, ast.ClassDef):
        text = compatible(node)
        if node.name == 'OrderMonitor':
            text = text.replace('trading_logger.', 'self.executor.logger.')
            text = text.replace('live_trade_recorder.record_event(', 'self.executor._record_event(')
            text = text.replace('is_current_trading(current)', 'self.executor._is_current_trading(current)')
            begin = text.index('        return any(\n', text.index('def _has_unhandled_terminal'))
            end = text.index('\n    #', begin)
            text = text[:begin] + '''        for oid in self.orders_by_code.get(code, []):
            state = self._last_order_state.get(oid)
            if oid not in self.handled_terminal and state is not None and state[0] in TERMINAL_STATUS:
                return True
        return False
''' + text[end:]
        else:
            text = text.replace('trading_logger.', 'self.logger.')
            text = text.replace('live_trade_recorder.record_event(', 'self._record_event(')
            text = text.replace('get_stock_detail(', 'self._stock_detail(')
            text = text.replace('recorder.mark(', 'self._mark(')
            begin = text.index('            pending_orders = [', text.index('def _wait_terminal'))
            end = text.index('\n            if ', begin)
            text = text[:begin] + '''            pending_orders = []
            for submitted_order in submitted:
                order = self.trader.query_order(submitted_order['order_id'])
                if order and order.order_status not in TERMINAL_STATUS:
                    pending_orders.append(submitted_order['code'])
''' + text[end:]
            hooks = '''
    @property
    def logger(self):
        return logging.getLogger('wbr.execution')

    def _is_current_trading(self, now):
        t = now.time()
        return dt_time(9, 30) <= t < dt_time(11, 30) or dt_time(13) <= t < dt_time(15)

    def _record_event(self, *args, **kwargs):
        raise NotImplementedError('Execution audit sink is required')

    def _stock_detail(self, code):
        return {}

    def _mark(self, message):
        pass
'''
            text += hooks
        parts.append(text)

header = '''"""Shared execution state machine. Python 3.6; no strategy/dataframe imports.

Broker operations and audit sinks are supplied by the application adapter.
The local compatibility adapter remains in trading.executor.
"""
import time
import math
import logging
from datetime import datetime, time as dt_time
from xtquant import xtconstant
from core.fees import LIVE_BUY_FEE_RATE
from trading.helper import get_order_status_label
from utils.stock.info import floor_buy_shares, min_buy_shares

EVT_EXECUTION_ACTION = 'execution_action'
SRC_EXECUTOR = 'executor'

'''
output = header + '\n\n'.join(parts) + '\n'
ast.parse(output, feature_version=(3, 6))
Path('trading/execution_engine.py').write_text(output, encoding='utf-8')
wrapper = '''"""Project-side compatibility adapter for the shared execution state machine.

Autonomous brokerage-app execution uses execution_engine directly.
"""
import time
import math
from datetime import datetime, time as dt_time
from xtquant import xtconstant
from data.db import get_stock_detail
from trading.logger import trading_logger
from trading.persistence import live_trade_recorder, EVT_EXECUTION_ACTION, SRC_EXECUTOR
from utils.recorder import recorder
from utils.stock.time import is_current_trading
from utils.stock.info import floor_buy_shares, min_buy_shares
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
'''
path.write_text(wrapper, encoding='utf-8')
