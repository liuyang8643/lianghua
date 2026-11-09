"""File-only handoff. Python 3.6; no brokerage or strategy dependencies."""
import datetime
import hashlib
import json
import math
import os
import re
import uuid

# XtQuant's locked order-status protocol. Shared by local fact checks and the
# Python 3.6 executor without importing a brokerage SDK in the local planner.
TERMINAL_ORDER_STATUSES = frozenset((53, 54, 56, 57))

def read_document(path):
    raw = path.read_bytes()
    return json.loads(raw.decode('utf-8')), hashlib.sha256(raw).hexdigest()


def broker_fact_date(value):
    """Broker order/trade timestamps are epoch seconds (some feeds use ms)."""
    try:
        stamp = float(value)
        if not math.isfinite(stamp) or stamp < 1000000000:
            raise ValueError('Broker timestamp is missing or not an epoch time')
        if stamp >= 1000000000000:
            stamp /= 1000
        return datetime.datetime.fromtimestamp(stamp).date()
    except (TypeError, OverflowError, OSError) as error:
        raise ValueError('Broker fact timestamp is invalid') from error


def validate_target(document, account, today, now, allow_simulation=False):
    if document.get('simulation', False) and not allow_simulation:
        raise ValueError('Simulation target is prohibited in production')
    if (document['version'] != 1 or document['account'] != account
            or document['trade_date'] != today.isoformat()
            or not re.fullmatch(r'[0-9a-f]{12}', document['id'])
            or type(document['generated_at']) not in (int, float)
            or type(document['expires']) not in (int, float)
            or not math.isfinite(document['generated_at'])
            or not math.isfinite(document['expires'])
            or not document['generated_at'] <= now < document['expires']
            or not isinstance(document['targets'], dict)):
        raise ValueError('Invalid, future, expired or wrong-account target file')
    seen = set()
    for code in document['buy_priority']:
        if code in seen or code not in document['targets']:
            raise ValueError('Invalid or repeated buy priority')
        seen.add(code)
    for code, row in document['targets'].items():
        if (not re.fullmatch(r'\d{6}\.(SH|SZ|BJ)', code)
                or type(row['shares']) is not int or row['shares'] < 0
                or type(row['open_price']) not in (int, float)
                or not math.isfinite(row['open_price']) or row['open_price'] <= 0
                or type(row['freeze_price']) not in (int, float)
                or not math.isfinite(row['freeze_price']) or row['freeze_price'] <= 0):
            raise ValueError('Invalid absolute target holding or execution price')


def write_immutable_target(path, document):
    if path.exists():
        previous, _ = read_document(path)
        if previous != document:
            raise RuntimeError('Daily target file is immutable; refuse overwrite')
        return
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(document, stream, ensure_ascii=True, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link publishes the complete file with create-once semantics.
            # Unlike replace(), it cannot overwrite a concurrent publisher.
            os.link(str(temporary), str(path))
        except FileExistsError:
            previous, _ = read_document(path)
            if previous != document:
                raise RuntimeError('Concurrent publisher already froze a different daily target')
    finally:
        if temporary.exists():
            temporary.unlink()


def targets_to_pending(document, positions):
    """Mechanical difference only. Selection, sizing and legality stay local.

    Unlisted holdings are unmanaged. Freeze full differences, including T+1
    unsellable shares; the execution engine waits for actual sellable quantity.
    """
    today = datetime.datetime.strptime(document['trade_date'], '%Y-%m-%d').date()
    sales, purchases, prices, freeze = [], {}, {}, {}
    for code, target in document['targets'].items():
        current = positions.get(code, 0)
        difference = target['shares'] - current
        if difference < 0:
            sales.append((code, -difference))
        elif difference > 0:
            purchases[code] = difference
        prices[code] = target['open_price']
        freeze[code] = target['freeze_price']
    priority = list(document['buy_priority'])
    priority.extend(code for code in purchases if code not in priority)
    return dict(signal_date=today, trade_date=today, sell_orders=sales,
                buy_allocations=purchases, buy_n_stocks=priority,
                prices=prices, limit_prices=freeze)
