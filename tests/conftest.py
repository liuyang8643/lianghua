"""真实数据测试共用 fixture，只接受当前完整 runtime 契约。"""
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from env.legality import classify_board_types, evaluate_trade_legality
from offline_data.runtime import RUNTIME_FIELDS
from offline_data.financial_versions import FINANCIAL_PANEL_FIELDS

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def compatible_runtime_path():
    files = sorted((_ROOT / 'data' / 'runtime').glob('runtime_*.npz'))
    if not files:
        pytest.skip('runtime npz 不存在，跳过真实数据测试')
    path = files[-1]
    required = {'stock_codes', 'trade_dates', *(field.name for field in RUNTIME_FIELDS), *FINANCIAL_PANEL_FIELDS}
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(required - set(payload.files))
    if missing:
        pytest.skip(f'生产 runtime 契约已过期，缺少字段: {missing}')
    return path


@pytest.fixture(scope='session')
def runtime_data(compatible_runtime_path):
    with np.load(compatible_runtime_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


class RealMarket:
    """封装真实 runtime，并调用生产唯一合法性函数。"""

    def __init__(self, data):
        self.data = data
        self.codes = [str(s) for s in data['stock_codes']]
        self.stock_indices = {c: i for i, c in enumerate(self.codes)}
        self.dates = data['trade_dates'].astype('datetime64[D]')
        self.board_types = classify_board_types(self.codes)

        ages = np.asarray(data['listing_age'], dtype=np.int32)
        self.list_map = {}
        for stock_index, code in enumerate(self.codes):
            first = np.flatnonzero(ages[:, stock_index] == 0)
            if first.size:
                self.list_map[code] = self.dates[int(first[0])].item()

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

    def buy(self, code, d):
        """该股 d 日开盘能否买入。"""
        return bool(self._legality(code, d).buy_allowed[0])

    def sell(self, code, d):
        """该股 d 日开盘能否卖出。"""
        result = self._legality(code, d)
        assert result.sell_allowed is not None
        return bool(result.sell_allowed[0])

    def _legality(self, code, d):
        ti = self.didx(d)
        ci = self.stock_indices[code]
        issue_price = (
            self.data['issue_price'][ci:ci + 1]
            if self.data['issue_date'][ci] == self.dates[ti]
            else np.asarray([np.nan])
        )
        return evaluate_trade_legality(
            decision_date=d,
            stock_codes=[code],
            listing_age=self.data['listing_age'][ti, ci:ci + 1],
            open_prices=self.data['open'][ti, ci:ci + 1],
            preclose_prices=self.data['preClose'][ti, ci:ci + 1],
            issue_prices=issue_price,
            st_mask=self.data['st_mask'][ti, ci:ci + 1],
            delisted_mask=self.data['delisted_mask'][ti, ci:ci + 1],
        )

    def bar(self, code, d):
        """原始 OHLC + 前收 + ST + 发行价（用于在测试里核对样本是否符合预期形态）。"""
        ti = self.didx(d); ci = self.stock_indices[code]
        return dict(
            open=float(self.data['open'][ti, ci]),
            high=float(self.data['high'][ti, ci]),
            low=float(self.data['low'][ti, ci]),
            close=float(self.data['close'][ti, ci]),
            preclose=float(self.data['preClose'][ti, ci]),
            st=bool(self.data['st_mask'][ti, ci]),
            issue_price=float(self.data['issue_price'][ci]),
            issue_date=self.data['issue_date'][ci],
            listing_age=int(self.data['listing_age'][ti, ci]),
            board=int(self.board_types[ci]),
        )


@pytest.fixture(scope='session')
def market(runtime_data):
    return RealMarket(runtime_data)


def test_conftest_smoke(market):
    # 冒烟：603690.SH IPO 首日(2017-01-13)一字/秒封 → 禁买（真实数据）
    assert market.buy('603690.SH', date(2017, 1, 13)) is False
