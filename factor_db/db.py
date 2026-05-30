"""因子库 DB 读写封装（append-only）。

设计约束：
- 只提供 add_factor() 写入与若干只读查询接口。
- 绝不提供 update / delete —— 模块级未定义任何此类函数，且 SQLite 触发器在底层
  拦截 UPDATE / DELETE，保证 factors 表 append-only。
- SQLite 文件固定在 factor_db/registry.db。
"""
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DB_PATH = Path(__file__).resolve().parent / 'registry.db'
_FACTORS_DIR = Path(__file__).resolve().parent / 'factors'

_SCHEMA = """
CREATE TABLE IF NOT EXISTS factors (
    factor_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    file_path    TEXT    NOT NULL,
    code_sha256  TEXT    NOT NULL,
    parent_ids   TEXT,
    op           TEXT    NOT NULL,
    generation   INTEGER NOT NULL,
    created_at   TEXT    NOT NULL,
    bt_start     TEXT,
    bt_end       TEXT,
    stock_pool   TEXT,
    train_sharpe REAL,
    total_return REAL,
    max_dd       REAL,
    n_trades     INTEGER,
    params_count INTEGER NOT NULL,
    status       TEXT    NOT NULL
);
"""

# append-only：底层触发器拦截任何 UPDATE / DELETE。
_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS factors_no_update
BEFORE UPDATE ON factors
BEGIN
    SELECT RAISE(ABORT, 'factors 表 append-only：禁止 UPDATE');
END;
CREATE TRIGGER IF NOT EXISTS factors_no_delete
BEFORE DELETE ON factors
BEGIN
    SELECT RAISE(ABORT, 'factors 表 append-only：禁止 DELETE');
END;
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """创建表与 append-only 触发器（幂等）。"""
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        conn.executescript(_TRIGGERS)


def file_sha256(file_path: Path) -> str:
    return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()


def add_factor(
    name: str,
    file_path: str,
    *,
    op: str,
    generation: int,
    params_count: int,
    status: str = 'active',
    parent_ids: Optional[str] = None,
    code_sha256: Optional[str] = None,
    bt_start: Optional[str] = None,
    bt_end: Optional[str] = None,
    stock_pool: Optional[str] = None,
    train_sharpe: Optional[float] = None,
    total_return: Optional[float] = None,
    max_dd: Optional[float] = None,
    n_trades: Optional[int] = None,
    created_at: Optional[str] = None,
) -> int:
    """登记一个新因子，返回 factor_id。name 唯一，重复登记会抛 IntegrityError。

    code_sha256 缺省时按 file_path 对应文件内容计算。
    """
    init_db()
    if code_sha256 is None:
        code_sha256 = file_sha256(_FACTORS_DIR.parent.parent / file_path)
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO factors (
                name, file_path, code_sha256, parent_ids, op, generation,
                created_at, bt_start, bt_end, stock_pool, train_sharpe,
                total_return, max_dd, n_trades, params_count, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name, file_path, code_sha256, parent_ids, op, generation,
                created_at, bt_start, bt_end, stock_pool, train_sharpe,
                total_return, max_dd, n_trades, params_count, status,
            ),
        )
        return int(cur.lastrowid)


def get_factor(name: str) -> Optional[dict]:
    init_db()
    with _connect() as conn:
        row = conn.execute('SELECT * FROM factors WHERE name = ?', (name,)).fetchone()
        return dict(row) if row else None


def get_factor_by_id(factor_id: int) -> Optional[dict]:
    init_db()
    with _connect() as conn:
        row = conn.execute('SELECT * FROM factors WHERE factor_id = ?', (factor_id,)).fetchone()
        return dict(row) if row else None


def list_factors(generation: Optional[int] = None, op: Optional[str] = None) -> list[dict]:
    init_db()
    sql = 'SELECT * FROM factors'
    conds, params = [], []
    if generation is not None:
        conds.append('generation = ?'); params.append(generation)
    if op is not None:
        conds.append('op = ?'); params.append(op)
    if conds:
        sql += ' WHERE ' + ' AND '.join(conds)
    sql += ' ORDER BY factor_id'
    with _connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def exists(name: str) -> bool:
    return get_factor(name) is not None
