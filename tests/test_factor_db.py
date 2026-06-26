"""factor_db.db 单元测试：验证 append-only 语义（add_factor 可用，update/delete 不存在/被拦截）。"""
import sqlite3

import pytest

from factor_db import db


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, '_DB_PATH', tmp_path / 'registry.db')
    db.init_db()
    return db


def test_add_and_get(temp_db):
    fid = temp_db.add_factor(
        name='MyFactor', file_path='factor_db/factors/MyFactor.py',
        op='seed', generation=0, params_count=0, code_sha256='abc',
    )
    assert fid == 1
    row = temp_db.get_factor('MyFactor')
    assert row['name'] == 'MyFactor'
    assert row['op'] == 'seed'
    assert row['generation'] == 0
    assert row['status'] == 'active'
    assert temp_db.get_factor_by_id(fid)['name'] == 'MyFactor'
    assert temp_db.exists('MyFactor') is True
    assert temp_db.exists('Nope') is False


def test_autoincrement_and_list(temp_db):
    temp_db.add_factor(name='A', file_path='a.py', op='seed', generation=0, params_count=0, code_sha256='1')
    temp_db.add_factor(name='B', file_path='b.py', op='crossover', generation=1, params_count=2, code_sha256='2', parent_ids='1')
    rows = temp_db.list_factors()
    assert [r['name'] for r in rows] == ['A', 'B']
    assert [r['factor_id'] for r in rows] == [1, 2]
    assert temp_db.list_factors(generation=1)[0]['name'] == 'B'
    assert temp_db.list_factors(op='seed')[0]['name'] == 'A'


def test_unique_name_rejected(temp_db):
    temp_db.add_factor(name='Dup', file_path='d.py', op='seed', generation=0, params_count=0, code_sha256='x')
    with pytest.raises(sqlite3.IntegrityError):
        temp_db.add_factor(name='Dup', file_path='d2.py', op='seed', generation=0, params_count=0, code_sha256='y')


def test_no_update_delete_api(temp_db):
    """append-only：模块绝不暴露 update / delete 接口。"""
    for forbidden in ('update_factor', 'delete_factor', 'remove_factor', 'update', 'delete'):
        assert not hasattr(db, forbidden), f'append-only 违例：不应存在 {forbidden}'


def test_trigger_blocks_update(temp_db):
    temp_db.add_factor(name='T', file_path='t.py', op='seed', generation=0, params_count=0, code_sha256='x')
    with sqlite3.connect(temp_db._DB_PATH) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE factors SET status='dead' WHERE name='T'")


def test_trigger_blocks_delete(temp_db):
    temp_db.add_factor(name='D', file_path='d.py', op='seed', generation=0, params_count=0, code_sha256='x')
    with sqlite3.connect(temp_db._DB_PATH) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM factors WHERE name='D'")
        # 确认数据仍在
        assert conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0] == 1
