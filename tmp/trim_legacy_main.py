"""Remove unreachable legacy CLI while retaining imported offline helpers."""
import ast
from pathlib import Path

path = Path('trading/main.py')
source = path.read_text(encoding='utf-8')
tree = ast.parse(source)
lines = source.splitlines(keepends=True)
keep = {'_build_plan_rows', '_load_prev_eod_baseline', 'MutableTime',
        '_parse_skip_datetime', '_require_previous_session'}
helpers = [''.join(lines[n.lineno - 1:n.end_lineno]) for n in tree.body
           if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in keep]
header = '''"""File-only production entry plus retained offline diagnostic helpers."""
from datetime import datetime, timedelta
import numpy as np

from data.db.stock_name import get_stock_name_at_date
from utils.stock.time import get_last_trading_day
from trading.scheduler import PREPARE_REBALANCE_START

'''
footer = '''

if __name__ == '__main__':
    from trading.planner_watchdog import cli
    cli()
'''
path.write_text(header + '\n\n'.join(helpers) + footer, encoding='utf-8')
