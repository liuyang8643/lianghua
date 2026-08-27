"""成交去重回归测试 —— 防止「QMT 回填撞不上无 traded_id 的重建行 → 整批翻倍」复发。"""
import pandas as pd

from trading.persistence import (
    FILL_COLS,
    _build_existing_fill_index,
    _consume_existing_fill,
    _coarse_key,
)


def _fills(rows):
    return pd.DataFrame(rows, columns=FILL_COLS)


def _row(order_id, price, shares, traded_id='', direction='sell'):
    return {
        'date': None, 'code': '688060.SH', 'name': '', 'direction': direction,
        'price': price, 'shares': shares, 'amount': price * shares, 'fee_est': 0.0,
        'order_id': order_id, 'traded_id': traded_id, 'fill_time': None,
        'est_price': None, 'slippage_pct': None,
    }


def test_qmt_backfill_dedups_against_rebuilt_rows_without_traded_id():
    """既有 4 笔等量同价、traded_id 为空(从 events 重建);QMT 回填同 4 笔(带 traded_id)
    应全部判为重复,不再翻倍。"""
    df_old = _fills([_row(403701761, 38.8, 200, traded_id='') for _ in range(4)])
    tids, coarse = _build_existing_fill_index(df_old)

    dup = 0
    for i in range(4):  # QMT 返回的 4 笔,带各自唯一 traded_id
        if _consume_existing_fill(f'TID{i}', 403701761, 38.8, 200, tids, coarse):
            dup += 1
    assert dup == 4  # 4 笔全部识别为已存在,零新增


def test_traded_id_hit_is_duplicate():
    df_old = _fills([_row(1, 10.0, 100, traded_id='X1')])
    tids, coarse = _build_existing_fill_index(df_old)
    assert _consume_existing_fill('X1', 1, 10.0, 100, tids, coarse) is True


def test_distinct_traded_ids_with_same_coarse_key_are_both_kept():
    df_old = _fills([_row(1, 10.0, 100, traded_id='X1')])
    tids, coarse = _build_existing_fill_index(df_old)
    assert _consume_existing_fill(
        'X2', 1, 10.0, 100, tids, coarse
    ) is False


def test_genuinely_new_fill_not_deduped():
    """既有 2 笔等量同价,QMT 来了 3 笔同键 → 前 2 笔消费存量、第 3 笔是真新增。"""
    df_old = _fills([_row(7, 5.0, 100, traded_id='') for _ in range(2)])
    tids, coarse = _build_existing_fill_index(df_old)
    results = [_consume_existing_fill(f'N{i}', 7, 5.0, 100, tids, coarse) for i in range(3)]
    assert results == [True, True, False]  # 第 3 笔未被去重 → 会被追加


def test_coarse_key_normalizes_types():
    assert _coarse_key('1', '10.0', '100') == _coarse_key(1, 10.0, 100)
