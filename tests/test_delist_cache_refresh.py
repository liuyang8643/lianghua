"""常驻实盘必须看到更新后的退市表，不能要求退市后继续提供行情。"""
from datetime import date

import pandas as pd
import pytest

from data.db import delist


@pytest.fixture
def delist_path(tmp_path, monkeypatch):
    monkeypatch.setattr(delist, '_DATA_DIR', tmp_path)
    monkeypatch.setattr(delist, '_DELIST_CACHE', None)
    monkeypatch.setattr(delist, '_DELIST_CACHE_VERSION', None)
    path = tmp_path / 'delist' / 'delist.parquet'
    path.parent.mkdir()
    return path


def write_table(path, codes):
    pd.DataFrame({
        '公司代码': codes, '公司简称': ['已退市'] * len(codes),
        '上市日期': ['2023-02-01'] * len(codes),
        '暂停上市日期': ['2026-10-19'] * len(codes),
        'exchange': ['SH'] * len(codes),
    }).to_parquet(path, index=False)


def test_long_running_reader_sees_new_delisting_after_file_replacement(delist_path):
    write_table(delist_path, ['600001'])
    cached = delist.get_delist_stock_info()
    assert delist.get_delist_stock_info() is cached
    replacement = delist_path.with_suffix('.tmp.parquet')
    write_table(replacement, ['600001', '601059'])
    replacement.replace(delist_path)
    refreshed = delist.get_delist_stock_info()
    assert refreshed['601059.SH'].delist_date == date(2026, 10, 19)
    assert '601059.SH' not in cached
    assert delist.get_delist_stock_info() is refreshed


def test_deleted_table_does_not_silently_use_old_cache(delist_path):
    write_table(delist_path, ['600001'])
    delist.get_delist_stock_info()
    delist_path.unlink()
    with pytest.raises(FileNotFoundError):
        delist.get_delist_stock_info()


def test_invalid_new_table_does_not_silently_use_old_cache(delist_path):
    write_table(delist_path, ['600001'])
    delist.get_delist_stock_info()
    pd.DataFrame({'bad': [1]}).to_parquet(delist_path)
    with pytest.raises(ValueError, match='exchange'):
        delist.get_delist_stock_info()
