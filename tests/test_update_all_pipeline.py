from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


class _FakeStore:
    _skip_update = False


class _FakeRecorder:
    def __init__(self):
        self.marks = []

    def mark(self, name):
        self.marks.append(name)


def test_current_update_date_is_not_frozen_at_import(monkeypatch):
    import data.update_all as update_all

    values = iter([date(2026, 9, 3), date(2026, 9, 4)])

    class MovingDate:
        @classmethod
        def today(cls):
            return next(values)

    monkeypatch.setattr(update_all, 'date', MovingDate)

    assert update_all.current_update_date() == date(2026, 9, 3)
    assert update_all.current_update_date() == date(2026, 9, 4)


def test_post_close_update_all_failure_not_reported_success(tmp_path, monkeypatch):
    import data.update_all as update_all
    from trading import post_close

    today = date.today()
    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    monkeypatch.setattr(update_all, "current_update_date", lambda: today)

    kline_path = tmp_path / "k-line" / "000001.SZ.parquet"
    kline_path.parent.mkdir(parents=True)
    kline_path.write_text("placeholder")

    called = []

    def noop():
        return None

    def fail_kline():
        called.append("kline")
        raise RuntimeError("kline failed")

    build_calls = []
    for name in [
        "_update_stock_list",
        "_update_delist",
        "_update_stock_name",
        "_update_balance",
        "_update_financial_deep",
        "_update_issue_price",
        "_update_indices",
        "_update_trading_calendar",
    ]:
        monkeypatch.setattr(update_all, name, noop)
    monkeypatch.setattr(update_all, "_build_runtime", lambda: build_calls.append("runtime"))
    monkeypatch.setattr(update_all, "_update_kline", fail_kline)

    def retry_once(func, max_retries=3, base_delay=2.0):
        try:
            return func(), True
        except Exception:
            return None, False

    fake_recorder = _FakeRecorder()
    monkeypatch.setattr(post_close, "_retry_with_backoff", retry_once)
    monkeypatch.setattr(post_close, "recorder", fake_recorder)

    assert post_close.run_update_all(_FakeStore()) is False
    assert called == ["kline"]
    assert build_calls == []
    run_checkpoint = tmp_path / "runtime" / ".update_checkpoints" / today.isoformat()
    assert (run_checkpoint / "step_stock_list.done").exists()
    assert not (run_checkpoint / "step_kline.done").exists()
    assert not (run_checkpoint / "step_issue_price.done").exists()
    assert not (run_checkpoint / "final.done").exists()
    assert fake_recorder.marks[-1] == "全量更新失败"
    assert "全量更新完成" not in fake_recorder.marks


def test_update_all_resumes_failed_step_then_validates_runtime(tmp_path, monkeypatch):
    import data.update_all as update_all
    from trading import post_close

    today = date(2026, 9, 3)
    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    monkeypatch.setattr(update_all, "current_update_date", lambda: today)
    pd.DataFrame({'trade_date': [today]}).to_parquet(
        tmp_path / "trading_calendar.parquet", index=False)

    calls = []
    fail_kline = {'value': True}

    def make_step(name):
        def _step():
            calls.append(name)
            if name == 'kline' and fail_kline['value']:
                raise RuntimeError('temporary mootdx failure')
        return _step

    for attr, key in [
        ("_update_stock_list", "stock_list"),
        ("_update_delist", "delist"),
        ("_update_stock_name", "stock_name"),
        ("_update_kline", "kline"),
        ("_update_balance", "balance"),
        ("_update_financial_deep", "financial_deep"),
        ("_update_issue_price", "issue_price"),
        ("_update_indices", "indices"),
        ("_update_trading_calendar", "trading_calendar"),
    ]:
        monkeypatch.setattr(update_all, attr, make_step(key))

    runtime_path = tmp_path / "runtime" / f"runtime_2020-01-01_{today}.npz"

    def build_runtime():
        calls.append('runtime')
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            runtime_path,
            trade_dates=np.array([today], dtype='datetime64[D]'),
            stock_codes=np.array(['000001.SZ']),
        )
        return runtime_path

    monkeypatch.setattr(update_all, "_build_runtime", build_runtime)

    def retry_once(func, max_retries=3, base_delay=2.0):
        try:
            return func(), True
        except Exception:
            return None, False

    monkeypatch.setattr(post_close, "_retry_with_backoff", retry_once)
    monkeypatch.setattr(post_close, "recorder", _FakeRecorder())

    assert post_close.run_update_all(_FakeStore()) is False
    assert 'runtime' not in calls

    first_counts = {name: calls.count(name) for name in set(calls)}
    fail_kline['value'] = False
    assert post_close.run_update_all(_FakeStore()) is True
    assert calls.count('kline') == 2
    for name, count in first_counts.items():
        if name != 'kline':
            assert calls.count(name) == count
    assert calls.count('runtime') == 1

    final_marker = (
        tmp_path / "runtime" / ".update_checkpoints" /
        today.isoformat() / "final.done"
    )
    assert final_marker.read_text(encoding='utf-8') == runtime_path.name

    # A valid final marker is still checked, then the whole pipeline is skipped.
    calls_before = list(calls)
    assert post_close.run_update_all(_FakeStore()) is True
    assert calls == calls_before


def test_runtime_date_validation_rejects_stale_archive(tmp_path):
    from trading.post_close import _validate_runtime_date

    runtime_path = tmp_path / "runtime_stale.npz"
    np.savez_compressed(
        runtime_path,
        trade_dates=np.array(['2026-09-02'], dtype='datetime64[D]'),
    )

    with pytest.raises(RuntimeError, match="actual=2026-09-02, expected=2026-09-03"):
        _validate_runtime_date(runtime_path, date(2026, 9, 3))


def test_issue_price_source_errors_are_not_reported_as_success(tmp_path, monkeypatch):
    import data.update_all as update_all

    monkeypatch.setattr(update_all, 'DATA_DIR', tmp_path)
    kline_dir = tmp_path / 'k-line'
    kline_dir.mkdir(parents=True)
    (kline_dir / '000001.SZ.parquet').touch()
    (kline_dir / '600000.SH.parquet').touch()

    def source_down(*args, **kwargs):
        raise ConnectionError('source unavailable')

    monkeypatch.setattr(update_all, '_post_json_with_retry', source_down)

    with pytest.raises(RuntimeError, match="2/2 只下载失败"):
        update_all._update_issue_price()
    assert not (tmp_path / 'issue_price' / 'issue_price.parquet').exists()


def test_failed_stock_list_blocks_dependent_steps(tmp_path, monkeypatch):
    import data.update_all as update_all
    from trading import post_close

    today = date(2026, 9, 3)
    monkeypatch.setattr(update_all, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(update_all, 'current_update_date', lambda: today)
    dependent_calls = []

    def fail_stock_list():
        raise RuntimeError('QMT stock list unavailable')

    monkeypatch.setattr(update_all, '_update_stock_list', fail_stock_list)
    monkeypatch.setattr(update_all, '_update_stock_name', lambda: dependent_calls.append('stock_name'))
    monkeypatch.setattr(update_all, '_update_kline', lambda: dependent_calls.append('kline'))
    monkeypatch.setattr(update_all, '_update_balance', lambda: dependent_calls.append('balance'))
    monkeypatch.setattr(update_all, '_update_issue_price', lambda: dependent_calls.append('issue_price'))
    for name in [
        '_update_delist', '_update_financial_deep', '_update_indices',
        '_update_trading_calendar',
    ]:
        monkeypatch.setattr(update_all, name, lambda: None)
    runtime_calls = []
    monkeypatch.setattr(update_all, '_build_runtime', lambda: runtime_calls.append('runtime'))

    def retry_once(func, max_retries=3, base_delay=2.0):
        try:
            return func(), True
        except Exception:
            return None, False

    monkeypatch.setattr(post_close, '_retry_with_backoff', retry_once)
    monkeypatch.setattr(post_close, 'recorder', _FakeRecorder())

    assert post_close.run_update_all(_FakeStore()) is False
    assert dependent_calls == []
    assert runtime_calls == []
    checkpoint = (
        tmp_path / 'runtime' / '.update_checkpoints' / today.isoformat()
    )
    assert not (checkpoint / 'step_stock_name.done').exists()
    assert not (checkpoint / 'step_kline.done').exists()
    assert not (checkpoint / 'step_balance.done').exists()
    assert not (checkpoint / 'step_issue_price.done').exists()


def test_deep_financial_partial_failure_is_persisted_but_raised(tmp_path, monkeypatch):
    import data.update_financial_deep as deep

    output = tmp_path / 'financial' / 'deep_indicators.parquet'
    monkeypatch.setattr(deep, 'OUT_PATH', output)
    monkeypatch.setattr(
        deep,
        '_ordered_symbols',
        lambda: [('000001.SZ', '000001'), ('600000.SH', '600000')],
    )

    def parse(symbol):
        if symbol == '600000':
            raise ConnectionError('THS unavailable')
        return pd.DataFrame([{'report_period': 20260630, 'bps': 10.0}])

    monkeypatch.setattr(deep, '_parse_one', parse)
    monkeypatch.setattr(deep.time, 'sleep', lambda seconds: None)

    with pytest.raises(RuntimeError, match="1/2 只下载失败"):
        deep.main()

    saved = pd.read_parquet(output)
    assert saved['stock_code'].tolist() == ['000001.SZ']


def test_delist_kline_missing_is_completed_by_mootdx(tmp_path, monkeypatch):
    import data.update_all as update_all
    import data.db.delist as delist_db
    import data.update_kline as kline_mootdx

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    (tmp_path / "k-line").mkdir(parents=True)
    calls = []

    monkeypatch.setattr(
        delist_db,
        "get_delist_stock_info",
        lambda: {"000003.SZ": SimpleNamespace()},
    )

    def fake_update_full(codes, end=None):
        calls.append(list(codes))
        (tmp_path / "k-line" / "000003.SZ.parquet").write_text("ok")

    monkeypatch.setattr(kline_mootdx, "update_full", fake_update_full)

    update_all._ensure_delist_kline_mootdx()

    assert calls == [["000003.SZ"]]


def test_delist_kline_missing_after_mootdx_raises(tmp_path, monkeypatch):
    import data.update_all as update_all
    import data.db.delist as delist_db
    import data.update_kline as kline_mootdx

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    (tmp_path / "k-line").mkdir(parents=True)
    monkeypatch.setattr(
        delist_db,
        "get_delist_stock_info",
        lambda: {"000003.SZ": SimpleNamespace()},
    )
    monkeypatch.setattr(kline_mootdx, "update_full", lambda codes, end=None: None)

    with pytest.raises(RuntimeError, match="mootdx 未补齐"):
        update_all._ensure_delist_kline_mootdx()
