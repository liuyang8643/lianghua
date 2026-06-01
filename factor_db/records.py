"""因子回测明细记录（append-only）。

与 db.py（只存因子血缘 + 摘要指标）互补：本模块为每次回测落一条完整明细，
含每日日期、每日收益、每日 topN 持仓（gzip 压缩 blob），用于：
  - 报告里回填 seed 因子缺失的夏普等指标（每日收益仅用于算夏普/净值，不做相关性）
  - 持仓明细留档

注：因子之间的"相同/不同"（去重、多样性、GA NSGA 目标）统一由 factor_db.similarity 的
每日截面股票 rank 指纹判定，本模块不参与任何收益相关性计算。

设计约束（同 db.py）：
- 只提供 add_run() 写入与只读查询，绝不提供 update / delete。
- SQLite 触发器在底层拦截 factor_runs 表的 UPDATE / DELETE，保证 append-only。
- 复用同一个 factor_db/registry.db 文件。
"""
import gzip
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DB_PATH = Path(__file__).resolve().parent / 'registry.db'

_SCHEMA = """
CREATE TABLE IF NOT EXISTS factor_runs (
    run_pk         INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name    TEXT    NOT NULL,
    run_id         TEXT,
    bt_start       TEXT    NOT NULL,
    bt_end         TEXT    NOT NULL,
    buy_n          INTEGER NOT NULL,
    stock_pool     TEXT,
    sharpe         REAL,
    annualized     REAL,
    max_dd         REAL,
    n_trades       INTEGER,
    n_days         INTEGER,
    record_dir     TEXT,
    dates_blob     BLOB,
    daily_ret_blob BLOB,
    topn_blob      BLOB,
    created_at     TEXT    NOT NULL
);
"""

_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS factor_runs_no_update
BEFORE UPDATE ON factor_runs
BEGIN
    SELECT RAISE(ABORT, 'factor_runs 表 append-only：禁止 UPDATE');
END;
CREATE TRIGGER IF NOT EXISTS factor_runs_no_delete
BEFORE DELETE ON factor_runs
BEGIN
    SELECT RAISE(ABORT, 'factor_runs 表 append-only：禁止 DELETE');
END;
"""

_BLOB_COLS = ('dates_blob', 'daily_ret_blob', 'topn_blob')


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_records() -> None:
    """创建 factor_runs 表与 append-only 触发器（幂等）。"""
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        conn.executescript(_TRIGGERS)


def _pack(obj) -> bytes:
    return gzip.compress(json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def _unpack(blob: Optional[bytes]):
    if blob is None:
        return None
    return json.loads(gzip.decompress(blob).decode('utf-8'))


def add_run(
    factor_name: str,
    *,
    bt_start: str,
    bt_end: str,
    buy_n: int,
    dates: list[str],
    daily_returns: list[float],
    topn: list[list[str]],
    sharpe: Optional[float] = None,
    annualized: Optional[float] = None,
    max_dd: Optional[float] = None,
    n_trades: Optional[int] = None,
    stock_pool: Optional[str] = None,
    run_id: Optional[str] = None,
    record_dir: Optional[str] = None,
    created_at: Optional[str] = None,
) -> int:
    """登记一次回测明细，返回 run_pk。同一因子可有多条（不同区间/参数），append-only。"""
    init_records()
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO factor_runs (
                factor_name, run_id, bt_start, bt_end, buy_n, stock_pool,
                sharpe, annualized, max_dd, n_trades, n_days,
                record_dir, dates_blob, daily_ret_blob, topn_blob, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                factor_name, run_id, bt_start, bt_end, buy_n, stock_pool,
                sharpe, annualized, max_dd, n_trades, len(dates),
                record_dir, _pack(dates), _pack(daily_returns), _pack(topn), created_at,
            ),
        )
        return int(cur.lastrowid)


def has_run(factor_name: str) -> bool:
    init_records()
    with _connect() as conn:
        row = conn.execute(
            'SELECT 1 FROM factor_runs WHERE factor_name = ? LIMIT 1', (factor_name,)
        ).fetchone()
        return row is not None


_SUMMARY_COLS = (
    'run_pk, factor_name, run_id, bt_start, bt_end, buy_n, stock_pool, '
    'sharpe, annualized, max_dd, n_trades, n_days, record_dir, created_at'
)


def list_runs() -> list[dict]:
    """返回所有回测摘要（不含明细 blob），按 run_pk 升序。"""
    init_records()
    with _connect() as conn:
        return [dict(r) for r in conn.execute(
            f'SELECT {_SUMMARY_COLS} FROM factor_runs ORDER BY run_pk'
        ).fetchall()]


def _row_to_dict(row, detail: bool) -> dict:
    d = dict(row)
    if detail:
        d['dates'] = _unpack(d.pop('dates_blob'))
        d['daily_returns'] = _unpack(d.pop('daily_ret_blob'))
        d['topn'] = _unpack(d.pop('topn_blob'))
    return d


def get_latest_run(factor_name: str, *, detail: bool = False) -> Optional[dict]:
    """取某因子最近一次回测。detail=True 时附带解压后的 dates/daily_returns/topn。"""
    init_records()
    cols = '*' if detail else _SUMMARY_COLS
    with _connect() as conn:
        row = conn.execute(
            f'SELECT {cols} FROM factor_runs WHERE factor_name = ? ORDER BY run_pk DESC LIMIT 1',
            (factor_name,),
        ).fetchone()
        return _row_to_dict(row, detail) if row else None


def get_run(factor_name: str, bt_start: str, bt_end: str, *, detail: bool = False) -> Optional[dict]:
    """取某因子在指定回测区间的最近一次记录（fitness/多样性需同基准区间，避免跨区间不可比）。"""
    init_records()
    cols = '*' if detail else _SUMMARY_COLS
    with _connect() as conn:
        row = conn.execute(
            f'SELECT {cols} FROM factor_runs WHERE factor_name = ? AND bt_start = ? AND bt_end = ? '
            f'ORDER BY run_pk DESC LIMIT 1',
            (factor_name, bt_start, bt_end),
        ).fetchone()
        return _row_to_dict(row, detail) if row else None


def latest_runs_by_factor(*, detail: bool = False) -> dict[str, dict]:
    """每个因子取最近一次回测，返回 {factor_name: run}。"""
    runs = list_runs()
    names = {r['factor_name'] for r in runs}
    return {n: get_latest_run(n, detail=detail) for n in names}
