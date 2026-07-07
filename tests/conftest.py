"""真实数据测试共用 fixture。

加载生产 runtime NPZ（session 内只加载一次），构建与回测完全一致的 LegalityChecker
（list_dates_map 取 K 线首个有效开盘日、delist_dates_map 取 db.delist 退市日），
并提供按 (代码, 日期) 查询买卖合法性与原始 bar 的便捷 API，供 test_legality_realdata_*.py 使用。
"""
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from core.legality import LegalityChecker

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def runtime_data():
    files = sorted((_ROOT / 'data' / 'runtime').glob('runtime_*.npz'))
    if not files:
        pytest.skip('runtime npz 不存在，跳过真实数据测试')
    npz = np.load(files[-1], allow_pickle=True)
    return {k: npz[k] for k in npz.files}


class RealMarket:
    """封装真实 runtime + LegalityChecker，按 (code, date) 查询。"""

    def __init__(self, data):
        self.data = data
        self.codes = [str(s) for s in data['stock_codes']]
        self.stock_indices = {c: i for i, c in enumerate(self.codes)}
        self.dates = data['trade_dates'].astype('datetime64[D]')

        # list_dates_map：K 线首个有效开盘日（与 core/backtest._compute_list_dates 一致）
        valid = ~np.isnan(data['open']) & (data['open'] > 0)
        first_idx = np.argmax(valid, axis=0)
        has_valid = np.any(valid, axis=0)
        list_map = {self.codes[i]: self.dates[first_idx[i]].item()
                    for i in range(len(self.codes)) if has_valid[i]}

        # delist_dates_map：退市/暂停日（与回测一致）
        from data.db.delist import get_delist_stock_info
        delist_map = {c: info.delist_date for c, info in get_delist_stock_info().items()}

        self.list_map = list_map
        self.delist_map = delist_map
        self.checker = LegalityChecker(data, self.stock_indices, list_map, delist_map)

    def has(self, code):
        return code in self.stock_indices

    def didx(self, d):
        """日期→交易日行索引（必须是交易日，否则报错）。"""
        d64 = np.datetime64(d)
        i = int(np.searchsorted(self.dates, d64))
        assert i < len(self.dates) and self.dates[i] == d64, f'{d} 不是交易日或越界'
        return i

    def list_date(self, code):
        return self.list_map.get(code)

    def delist_date(self, code):
        return self.delist_map.get(code)

    def buy(self, code, d):
        """该股 d 日收盘能否买入。"""
        ok, _ = self.checker.check([self.stock_indices[code]], self.didx(d), d, is_buy=True)
        return bool(ok[0])

    def sell(self, code, d):
        """该股 d 日收盘能否卖出。"""
        ok, _ = self.checker.check([self.stock_indices[code]], self.didx(d), d, is_buy=False)
        return bool(ok[0])

    def bar(self, code, d):
        """原始 OHLC + 前收 + ST + 发行价（用于在测试里核对样本是否符合预期形态）。"""
        ti = self.didx(d); ci = self.stock_indices[code]
        return dict(
            open=float(self.data['open'][ti, ci]),
            high=float(self.data['high'][ti, ci]),
            low=float(self.data['low'][ti, ci]),
            close=float(self.data['close'][ti, ci]),
            preclose=float(self.data['close'][ti - 1, ci]) if ti > 0 else np.nan,
            st=bool(self.data['st_mask'][ti, ci]),
            issue_price=float(self.data['issue_price'][ci]),
            board=int(self.checker.board_type[ci]),
        )


@pytest.fixture(scope='session')
def market(runtime_data):
    return RealMarket(runtime_data)


def test_conftest_smoke(market):
    # 冒烟：603690.SH IPO 首日(2017-01-13) 收盘封 +44% 涨停 → 禁买（真实数据）
    assert market.buy('603690.SH', date(2017, 1, 13)) is False
