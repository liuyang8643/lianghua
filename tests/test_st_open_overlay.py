from datetime import date

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize('already_present', [False, True])
def test_open_overlay_applies_only_effective_st_events_and_matches_full_runtime(tmp_path, monkeypatch, already_present):
    from data import build_runtime, update_live
    from data.db import stock_name
    events = pd.DataFrame([
        ('600001', date(2026, 10, 1), '新股上市', '正常上市'),
        ('600001', date(2026, 11, 3), '戴帽披*', '正常上市'),
        ('600001', date(2026, 11, 4), '摘帽', '正常上市'),
        ('600002', date(2026, 10, 1), '戴帽', '正常上市'),
        ('600002', date(2026, 11, 3), '摘帽', '正常上市'),
        ('600002', date(2026, 11, 4), '戴帽', '正常上市'),
    ], columns=['bare_code', 'date', 'event', 'status'])
    (tmp_path / 'stock_name').mkdir()
    events.to_parquet(tmp_path / 'stock_name/st_changes.parquet', index=False)
    monkeypatch.setattr(build_runtime, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(stock_name, '_load_st_changes', lambda: events)
    codes = np.array(['600001.SH', '600002.SH'])
    dates = np.array(['2026-11-02', '2026-11-03'] if already_present else ['2026-11-02'], dtype='datetime64[D]')
    data = dict(stock_codes=codes, trade_dates=dates,
                st_mask=np.tile([False, True], (len(dates), 1)))
    for field in update_live._RAW_PATCH_FIELDS:
        data[field] = np.ones((len(dates), 2), dtype=float)
    bars = {'time': np.array([pd.Timestamp('2026-11-03').value // 1000000])}
    bars.update({field: np.array([10. if field in ('open', 'preClose') else np.nan])
                 for field in update_live._RAW_PATCH_FIELDS})
    result, _, _ = update_live.apply_kline_overlay(data, {codes[0]: bars})
    assert result['st_mask'].tolist() == [[False, True], [True, False]]
    full = build_runtime.build_st_mask(codes, np.array(['2026-11-02', '2026-11-03', '2026-11-04'], dtype='datetime64[D]'))
    assert full.tolist() == [[False, True], [True, False], [False, True]]
    np.testing.assert_array_equal(result['st_mask'], full[:2])
