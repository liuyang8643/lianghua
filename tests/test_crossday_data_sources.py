import numpy as np
import pandas as pd
import pytest


def test_issue_price_cninfo_fields_and_transfer(tmp_path, monkeypatch):
    import data.update_all as module
    monkeypatch.setattr(module, 'DATA_DIR', tmp_path)
    (tmp_path / 'k-line').mkdir()
    (tmp_path / 'issue_price').mkdir()
    from datetime import date
    pd.DataFrame([{'stock_code': '000001', 'issue_price': 40.0,
                   'list_date': date(1991, 4, 3), 'source': 'legacy',
                   'source_as_of': date(2026, 5, 16)}]).to_parquet(
                       tmp_path / 'issue_price' / 'issue_price.parquet')
    for code in ('301192.SZ', '688287.SH'):
        (tmp_path / 'k-line' / f'{code}.parquet').touch()
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs)
        assert kwargs['timeout'] == 15
        return {'resultcode': 200, 'records': [{'SECCODE': '301192', 'F008N': 16.42, 'F006D': '2022-08-11'}]}
    monkeypatch.setattr(module, '_post_json_with_retry', post)
    module._update_issue_price()
    rows = pd.read_parquet(tmp_path / 'issue_price' / 'issue_price.parquet').set_index('stock_code')
    assert rows.loc['301192', 'issue_price'] == 16.42
    assert pd.isna(rows.loc['688287', 'issue_price'])
    assert rows.loc['688287', 'issue_price_status'] == 'not_applicable_transfer'
    assert len(calls) == 1
    module._update_issue_price()
    assert len(calls) == 1


def test_financial_sina_fallback_keeps_unknown_fields_missing(monkeypatch):
    import data.update_financial_deep as module
    def broken(*args, **kwargs):
        raise AttributeError('old THS page missing')
    monkeypatch.setattr(module, '_fetch_ths', broken)
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'result': {'status': {'code': 0}, 'data': {'report_list': {'19961231': {'data': [
                {'item_title': '基本每股收益', 'item_value': '0.35'},
                {'item_title': '净利润', 'item_value': '1.2亿', 'item_tongbi': '10%'},
            ]}}}}}
    monkeypatch.setattr(module.requests.Session, 'get', lambda *a, **k: Response())
    result = module._parse_one('000508')
    assert result.iloc[0]['report_period'] == 19961231
    assert result.iloc[0]['eps'] == 0.35
    assert result.iloc[0]['net_profit'] == 1.2e8
    assert result.iloc[0]['profit_yoy'] == 10
    assert np.isnan(result.iloc[0]['revenue'])


def test_financial_empty_backup_is_not_success(monkeypatch):
    import data.update_financial_deep as module
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'result': {'status': {'code': 0}, 'data': {'report_list': {}}}}
    monkeypatch.setattr(module.requests.Session, 'get', lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match='无报告'):
        module._parse_sina('000508')


def test_ths_actual_request_has_timeout(monkeypatch):
    import data.update_financial_deep as module
    class Response:
        text = '<p id="main">{"title":["报告期",["基本每股收益"]],"report":[["2026-06-30"],["1.2"]]}</p>'
        def raise_for_status(self):
            pass
    def get(self, url, **kwargs):
        assert kwargs['timeout'] == 15
        assert self.trust_env is False
        return Response()
    monkeypatch.setattr(module.requests.Session, 'get', get)
    frame = module._parse_one('600000')
    assert frame.iloc[0]['eps'] == 1.2
