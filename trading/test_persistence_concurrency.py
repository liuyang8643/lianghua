"""并发写 parquet 回归测试 —— 防止 2026-06-02 的 events parquet 损坏复发。

根因:09:30 时 SellMonitor 线程池 / BuyMonitor 线程 / watcher 回调线程并发调用
record_event → _append_event,旧实现用固定 .tmp 文件名 + 无锁的读改写,
导致多线程写花同一个 tmp(footer 损坏)与丢更新。
"""
import multiprocessing
import threading
from pathlib import Path

import pandas as pd
import pytest

import trading.persistence as persistence
from trading.persistence import EVT_ORDER, EVT_TRADE, LiveTradeRecorder


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "_TRADE_DIR", tmp_path)
    return LiveTradeRecorder()


def _process_write_events(directory: str, start: int, count: int):
    persistence._TRADE_DIR = Path(directory)
    recorder = LiveTradeRecorder()
    for oid in range(start, start + count):
        recorder.record_event(
            EVT_ORDER, code="600000.SH", order_id=oid,
            order_status=50, order_volume=100, price=10.0,
        )


def test_concurrent_record_event_no_corruption_no_lost_update(recorder, tmp_path):
    n_threads = 12
    per_thread = 20
    total = n_threads * per_thread
    barrier = threading.Barrier(n_threads)

    def worker(tid: int):
        barrier.wait()  # 尽量让所有线程同一瞬间开火,放大并发竞争
        for i in range(per_thread):
            oid = tid * 1000 + i  # 全局唯一 → 不会被去重 key 折叠
            recorder.record_event(
                EVT_ORDER,
                code="600000.SH",
                order_id=oid,
                order_volume=100,
                price=float(oid),  # 价格也唯一,进一步保证 5 元组去重键唯一
                status_msg="concurrent",
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    path = tmp_path / f"events_{pd.Timestamp.now().date().isoformat()}.parquet"
    # 文件必须可读(没损坏),且不曾被隔离成 .corrupt
    assert path.exists()
    assert not (tmp_path / (path.name + ".corrupt")).exists()
    df = pd.read_parquet(path)
    # 全部写入、无丢更新
    assert len(df) == total
    assert df["order_id"].nunique() == total
    # 不留临时文件
    assert list(tmp_path.glob("*.tmp*")) == []


def test_concurrent_record_event_and_fill_mixed(recorder, tmp_path):
    """events 与 fills 两条写路径并发交织也不互相破坏。"""
    n = 60
    barrier = threading.Barrier(2)

    def write_events():
        barrier.wait()
        for i in range(n):
            recorder.record_event(EVT_ORDER, code="000001.SZ",
                                   order_id=i, order_volume=100, price=float(i))

    def write_fills():
        barrier.wait()
        for i in range(n):
            recorder.record_fill("000002.SZ", "buy", price=10.0 + i,
                                  shares=100, amount=(10.0 + i) * 100,
                                  order_id=10_000 + i, name="x")

    te = threading.Thread(target=write_events)
    tf = threading.Thread(target=write_fills)
    te.start(); tf.start()
    te.join(); tf.join()

    today = pd.Timestamp.now().date().isoformat()
    ev = pd.read_parquet(tmp_path / f"events_{today}.parquet")
    fl = pd.read_parquet(tmp_path / f"fills_{today}.parquet")
    # fills 派生的 trade 也会写进 events,故 events 至少包含 n 条 order 事件
    assert (ev["event_type"] == EVT_ORDER).sum() == n
    assert len(fl) == n
    assert list(tmp_path.glob("*.tmp*")) == []


def test_cross_process_record_event_no_lost_update(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    workers = [
        ctx.Process(target=_process_write_events, args=(str(tmp_path), i * 100, 20))
        for i in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0

    today = pd.Timestamp.now().date().isoformat()
    events = pd.read_parquet(tmp_path / f"events_{today}.parquet")
    assert len(events) == 40
    assert events["order_id"].nunique() == 40
    assert list(tmp_path.glob("*.tmp*")) == []


def test_duplicate_qmt_callbacks_use_stable_business_keys(recorder, tmp_path):
    order = dict(
        code="600000.SH", order_id=42, order_status=50,
        order_volume=100, traded_volume=0, price=10.0,
    )
    recorder.record_event(EVT_ORDER, **order)
    recorder.record_event(EVT_ORDER, **order)
    recorder.record_event(EVT_ORDER, **{**order, "order_status": 56, "traded_volume": 100})

    trade = dict(
        code="600000.SH", order_id=42, traded_id="T42", direction="buy",
        traded_volume=100, traded_price=10.0, amount=1000.0,
    )
    recorder.record_event(EVT_TRADE, **trade)
    recorder.record_event(EVT_TRADE, **trade)

    today = pd.Timestamp.now().date().isoformat()
    events = pd.read_parquet(tmp_path / f"events_{today}.parquet")
    fills = pd.read_parquet(tmp_path / f"fills_{today}.parquet")
    assert (events["event_type"] == EVT_ORDER).sum() == 2
    assert (events["event_type"] == EVT_TRADE).sum() == 1
    assert len(fills) == 1
    assert len(recorder._today_fills) == 1


def test_atomic_replace_retries_transient_windows_permission_error(tmp_path, monkeypatch):
    path = tmp_path / "events.parquet"
    real_replace = persistence.os.replace
    attempts = 0

    def flaky_replace(src, dst):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "temporarily locked")
        return real_replace(src, dst)

    monkeypatch.setattr(persistence.os, "replace", flaky_replace)
    monkeypatch.setattr(persistence.time, "sleep", lambda _: None)
    persistence._atomic_write_parquet(pd.DataFrame([{"x": 1}]), path)

    assert attempts == 3
    assert pd.read_parquet(path).iloc[0]["x"] == 1
