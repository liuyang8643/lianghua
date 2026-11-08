"""Python 3.6 compatible, local durable protocol for the QMT execution bridge."""
import json
import os
import sqlite3


class ConfirmedNotSubmitted(RuntimeError):
    """Durable bridge evidence proves this intent never reached the order API."""


class BridgeUnavailable(SystemExit):
    """A broker connection failure requires rebuilding and journal recovery."""
    def __init__(self, message, *, not_submitted=False):
        self.not_submitted = not_submitted
        super().__init__(message)


class BrokerConnectionLost(RuntimeError):
    """Rebuild the client SDK session and recover durable order intents."""


SCHEMA = '''
CREATE TABLE IF NOT EXISTS requests (
 id TEXT PRIMARY KEY, created REAL NOT NULL, expires REAL NOT NULL,
 client_pid INTEGER NOT NULL, instance TEXT NOT NULL, operation TEXT NOT NULL,
 payload TEXT NOT NULL, state TEXT NOT NULL, response TEXT
);
CREATE TABLE IF NOT EXISTS order_intents (
 remark TEXT PRIMARY KEY, specification TEXT NOT NULL, state TEXT NOT NULL,
 result INTEGER
);
CREATE TABLE IF NOT EXISTS events (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,
 kind TEXT NOT NULL, payload TEXT NOT NULL
);
'''


def open_database(path):
    connection = sqlite3.connect(str(path), timeout=5)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA synchronous=FULL')
    connection.executescript(SCHEMA)
    return connection


def atomic_json(path, value):
    temporary = str(path) + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=True, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, str(path))


def encode(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    result = {}
    for key in dir(value):
        if key.startswith('_'):
            continue
        item = getattr(value, key)
        if item is None or isinstance(item, (str, bool, int, float)):
            result[key] = item
    return result
