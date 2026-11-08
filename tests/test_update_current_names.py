from datetime import date

import pandas as pd
import pytest


def test_update_current_names_uses_tencent_fetcher(tmp_path, monkeypatch):
    import data.update_all as update_all

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    monkeypatch.setattr(update_all, "current_update_date", date.today)

    def fail_akshare():
        raise AssertionError("akshare current-name fetch should not be called")

    monkeypatch.setattr(update_all.ak, "stock_info_a_code_name", fail_akshare)
    monkeypatch.setattr(
        update_all,
        "_fetch_current_stock_names",
        lambda codes: pd.DataFrame({
            "code": ["000001", "600000", "430001"],
            "name": ["平安银行", "浦发银行", "北交测试"],
        }),
    )

    update_all._update_current_names(["000001.SZ", "600000.SH", "430001.BJ", "900901.SH"])

    df = pd.read_parquet(tmp_path / "stock_name" / "current_names.parquet")
    assert df.to_dict("records") == [
        {"bare_code": "000001", "stock_code": "000001.SZ", "name": "平安银行"},
        {"bare_code": "600000", "stock_code": "600000.SH", "name": "浦发银行"},
        {"bare_code": "430001", "stock_code": "430001.BJ", "name": "北交测试"},
    ]


def test_tencent_quote_symbol_routes_boards():
    import data.update_all as update_all

    assert update_all._tencent_quote_symbol("000001.SZ") == "sz000001"
    assert update_all._tencent_quote_symbol("600000.SH") == "sh600000"
    assert update_all._tencent_quote_symbol("688001.SH") == "sh688001"
    assert update_all._tencent_quote_symbol("430001.BJ") == "bj430001"
    assert update_all._tencent_quote_symbol("830001.BJ") == "bj830001"
    assert update_all._tencent_quote_symbol("870001.BJ") == "bj870001"
    assert update_all._tencent_quote_symbol("920001.BJ") == "bj920001"


def test_name_history_resumes_verified_empty_responses_after_network_failure(tmp_path, monkeypatch):
    import data.update_all as module
    import data.db.stock_list as stocks
    monkeypatch.setattr(module, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(module, 'current_update_date', date.today)
    monkeypatch.setattr(stocks, 'get_all_stock_code_list', lambda: ['000001.SZ', '000002.SZ'])
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(module, '_update_current_names', lambda codes: None)
    calls = []
    def fetch(url, **kwargs):
        if url.endswith('p_stock2117'):
            return {'records': [dict(SECCODE='000001', VARYDATE='2020-01-01', F006V='test', F002V='test')]}
        bare = kwargs['params']['scode']
        calls.append(bare)
        if bare == '000002' and calls.count(bare) == 1:
            raise RuntimeError('network interrupted')
        return {'records': [] if bare == '000001' else [
            dict(SECCODE=bare, STARTDATE='2020-01-01', F002V='Original name')]}
    monkeypatch.setattr(module, '_post_json_with_retry', fetch)
    module._update_stock_name()
    import json
    status = json.loads((tmp_path / ('stock_name/name_history_status_' + date.today().isoformat() + '.json')).read_text())
    assert status['state'] == 'skipped' and status['required'] is False
    module._update_stock_name()
    assert calls == ['000001', '000002', '000002']
    frame = pd.read_parquet(tmp_path / 'stock_name/name_changes.parquet')
    assert frame['bare_code'].tolist() == ['000002']


def test_startup_status_does_not_complete_full_name_history(tmp_path, monkeypatch):
    import data.update_all as module
    import data.db.stock_list as stocks
    monkeypatch.setattr(module, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(stocks, 'get_all_stock_code_list', lambda: ['000001.SZ'])
    monkeypatch.setattr(module, '_fetch_current_stock_names', lambda codes: pd.DataFrame([
        dict(code='000001', name='Current name')]))
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        return {'records': [dict(SECCODE='000001', VARYDATE='2020-01-01', F006V='新股上市', F002V='正常上市')]
                if url.endswith('p_stock2117') else []}
    monkeypatch.setattr(module, '_post_json_with_retry', fetch)
    module._update_stock_name(include_history=False)
    assert not any(url.endswith('p_stock2109') for url in calls)
    assert (tmp_path / 'stock_name/current_names.parquet').exists()
    module._update_stock_name(include_history=True)
    assert sum(url.endswith('p_stock2109') for url in calls) == 1


def test_st_failure_remains_fatal_and_cannot_be_optional(tmp_path, monkeypatch):
    import data.update_all as module
    import data.db.stock_list as stocks
    monkeypatch.setattr(module, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(stocks, 'get_all_stock_code_list', lambda: ['000001.SZ'])
    def fail(*args, **kwargs):
        raise RuntimeError('ST source unavailable')
    monkeypatch.setattr(module, '_post_json_with_retry', fail)
    monkeypatch.setattr(module, '_update_name_history', lambda *a, **k: pytest.fail('Must not reach history'))
    with pytest.raises(RuntimeError, match='ST source unavailable'):
        module._update_stock_name()
    assert not list((tmp_path / 'stock_name').glob('name_history_status*'))


def test_optional_history_budget_preserves_previous_file(tmp_path, monkeypatch):
    import data.update_all as module
    monkeypatch.setattr(module, 'DATA_DIR', tmp_path)
    folder = tmp_path / 'stock_name'; folder.mkdir()
    old = folder / 'name_changes.parquet'
    pd.DataFrame([dict(bare_code='000003',start_date=date(2020,1,1),old_name='Existing')]).to_parquet(old)
    before = old.read_bytes()
    monkeypatch.setattr(module, '_post_json_with_retry', lambda *a, **k: pytest.fail('Budget expired'))
    with pytest.raises(TimeoutError):
        module._update_name_history(['000001'], js=None, headers={}, budget_seconds=0)
    assert old.read_bytes() == before


@pytest.mark.parametrize('failure', ['status', 'parquet'])
def test_optional_writes_cannot_block_required_status_or_damage_old_history(tmp_path, monkeypatch, failure):
    from pathlib import Path
    from types import SimpleNamespace
    import data.update_all as module
    import data.db.stock_list as stocks
    monkeypatch.setattr(module, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(stocks, 'get_all_stock_code_list', lambda: ['000001.SZ'])
    monkeypatch.setattr(module, '_update_current_names', lambda codes: None)
    folder = tmp_path / 'stock_name'; folder.mkdir()
    old = folder / 'name_changes.parquet'
    pd.DataFrame([dict(bare_code='000003',start_date=date(2020,1,1),old_name='Keep')]).to_parquet(old)
    before = old.read_bytes()
    monkeypatch.setattr(module, '_post_json_with_retry', lambda url, **kwargs: {'records':
        [dict(SECCODE='000001', VARYDATE='2020-01-01', F006V='新股上市', F002V='正常上市')]
        if url.endswith('p_stock2117') else [dict(SECCODE='000001',STARTDATE='2020-01-01',F002V='Name')]})
    if failure == 'status':
        replace = Path.replace
        def failing_replace(path, target):
            if path.name.startswith('name_history_status_'):
                raise PermissionError('busy optional status')
            return replace(path, target)
        monkeypatch.setattr(Path, 'replace', failing_replace)
        monkeypatch.setattr(module, '_update_name_history', lambda *a, **k: None)
    else:
        write = pd.DataFrame.to_parquet
        def failing_write(frame, path, **kwargs):
            if str(path).endswith('.parquet.tmp'):
                Path(path).write_bytes(b'partial bytes')
                raise OSError('disk write interrupted')
            return write(frame, path, **kwargs)
        monkeypatch.setattr(pd.DataFrame, 'to_parquet', failing_write)
    module._update_stock_name()
    assert (folder / 'st_changes.parquet').exists()
    assert old.read_bytes() == before


def test_current_name_partial_response_does_not_publish_success(tmp_path, monkeypatch):
    import data.update_all as module
    import data.db.delist as delist
    monkeypatch.setattr(module, 'DATA_DIR', tmp_path)
    (tmp_path / 'k-line').mkdir()
    (tmp_path / 'k-line/000001.SZ.parquet').touch()
    monkeypatch.setattr(delist, 'get_delist_stock_info', lambda: {})
    monkeypatch.setattr(module, '_fetch_current_stock_names', lambda codes: pd.DataFrame([
        dict(code='600000', name='Another stock')]))
    with pytest.raises(RuntimeError, match='Current names missing'):
        module._update_current_names(['000001.SZ', '600000.SH'])
    assert not (tmp_path / 'stock_name/current_names.parquet').exists()


def test_fetch_current_stock_names_parses_tencent_response(monkeypatch):
    import data.update_all as update_all

    calls = []

    class _Resp:
        def __init__(self, text):
            self.content = text.encode("gbk")
        def raise_for_status(self):
            pass

    def fake_get(url, headers, timeout):
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        symbols = url.split("q=", 1)[1].split(",")
        lines = []
        for sym in symbols:
            bare = sym[2:]
            name = {"000001": "平安银行", "600000": "浦发银行", "430001": "北交测试"}[bare]
            lines.append(f'v_{sym}="51~{name}~{bare}~0";')
        return _Resp("".join(lines))

    monkeypatch.setattr(update_all, "CURRENT_NAMES_CHUNK_SIZE", 2)
    monkeypatch.setattr(update_all.requests, "get", fake_get)

    df = update_all._fetch_current_stock_names(["000001.SZ", "600000.SH", "430001.BJ"])

    assert calls == [
        {
            "url": "https://qt.gtimg.cn/q=sz000001,bj430001",
            "headers": {"User-Agent": "Mozilla/5.0"},
            "timeout": update_all.CURRENT_NAMES_REQUEST_TIMEOUT,
        },
        {
            "url": "https://qt.gtimg.cn/q=sh600000",
            "headers": {"User-Agent": "Mozilla/5.0"},
            "timeout": update_all.CURRENT_NAMES_REQUEST_TIMEOUT,
        },
    ]
    assert df.to_dict("records") == [
        {"code": "000001", "name": "平安银行"},
        {"code": "430001", "name": "北交测试"},
        {"code": "600000", "name": "浦发银行"},
    ]
