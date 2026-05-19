"""
合法性校验模块 — 回测/实盘统一的可交易性检查。

设计原则：
- 回测和实盘完全统一：都按开盘价执行，直接从 bar['open'] 取值
- 涨跌停价格使用交易所规则四舍五入到分（Decimal ROUND_HALF_UP）
- 回测路径使用 NPZ st_mask 查 ST，完全离线无联网
- 实盘路径使用 CNINFO 实时 ST 数据
- preClose 来源于 bar 数据，除权日需上游传入正确的除权参考价
"""

import numpy as np
from typing import Optional, Dict, Tuple, List

from utils.stock.info import get_limit_band_from_ratio

EPSILON = 0.001


class LegalityResult:
    __slots__ = ('allowed', 'reason', 'up_limit', 'down_limit', 'regime_name')

    def __init__(self, allowed: bool, reason: str,
                 up_limit: Optional[float] = None, down_limit: Optional[float] = None,
                 regime_name: str = 'unknown'):
        self.allowed = allowed
        self.reason = reason
        self.up_limit = up_limit
        self.down_limit = down_limit
        self.regime_name = regime_name

    def __repr__(self):
        return (f"LegalityResult(allowed={self.allowed}, reason='{self.reason}', "
                f"up={self.up_limit}, down={self.down_limit}, regime={self.regime_name})")


class LegalityValidator:
    """
    统一合法性校验器。

    两种模式：
    - 回测模式：传入 st_mask + stock_codes + trade_dates，ST 从 NPZ 查表，完全离线
    - 实盘模式：不传 st_mask，调 resolve_limit_regime → CNINFO 实时 ST
    """

    def __init__(self, st_mask: Optional[np.ndarray] = None,
                 stock_codes: Optional[List[str]] = None,
                 trade_dates: Optional[List] = None, *,
                 offline: bool = False,
                 list_dates: Optional[dict] = None):
        self._st_mask = st_mask
        self._offline = offline
        self._list_dates = list_dates or {}
        self._stock_idx: Dict[str, int] = {}
        self._date_idx: Dict[str, int] = {}
        if st_mask is not None and stock_codes is not None and trade_dates is not None:
            self._stock_idx = {str(c): i for i, c in enumerate(stock_codes)}
            self._date_idx = {str(d)[:10]: i for i, d in enumerate(trade_dates)}

    def _is_st(self, stock_code: str, trade_date) -> Optional[bool]:
        """回测模式：从 NPZ st_mask 查表。找不到返回 None，fallback 到联网。"""
        if not self._stock_idx or self._st_mask is None:
            return None
        si = self._stock_idx.get(stock_code)
        if si is None:
            return None
        date_str = str(trade_date)[:10]
        di = self._date_idx.get(date_str)
        if di is None:
            # 精确日期不在 NPZ 里，找最近的已过去日期
            candidates = [(d, i) for d, i in self._date_idx.items() if d <= date_str]
            if not candidates:
                return None
            di = max(candidates, key=lambda x: x[0])[1]
        return bool(self._st_mask[di, si])

    def _get_limit_band(self, stock_code: str, trade_date, bar
                        ) -> Tuple[Optional[float], Optional[float], dict]:
        is_st = self._is_st(stock_code, trade_date)
        list_date = self._list_dates.get(stock_code)
        if is_st is not None:
            return _get_limit_band_with_st(stock_code, trade_date, bar, is_st, list_date=list_date)
        if self._st_mask is not None or self._offline:
            return _get_limit_band_with_st(stock_code, trade_date, bar, False, list_date=list_date)
        return get_limit_band_from_ratio(stock_code, trade_date, bar)

    def check_buy(self, stock_code: str, trade_date, bar) -> LegalityResult:
        if bar is None:
            return LegalityResult(False, 'no_data')

        open_p = float(bar['open'])
        if np.isnan(open_p) or open_p <= 0:
            return LegalityResult(False, 'suspended')

        if int(bar.get('suspendFlag', 0)) == 1:
            return LegalityResult(False, 'suspended')

        up_limit, down_limit, regime = self._get_limit_band(stock_code, trade_date, bar)
        if regime['has_price_limit'] and up_limit is not None:
            if open_p >= up_limit - EPSILON:
                return LegalityResult(False, 'limit_up', up_limit, down_limit, regime['name'])
            # IPO 首日秒封：开盘未涨停但当日触及涨停且 open==low → 实盘无法买入
            list_date = self._list_dates.get(stock_code)
            if list_date is not None and str(trade_date)[:10] == str(list_date):
                low_p = float(bar.get('low', np.nan))
                high_p = float(bar.get('high', np.nan))
                if (not np.isnan(low_p) and not np.isnan(high_p)
                        and abs(open_p - low_p) < EPSILON
                        and high_p >= up_limit - EPSILON):
                    return LegalityResult(False, 'instant_limit_up', up_limit, down_limit, regime['name'])

        return LegalityResult(True, 'ok', up_limit, down_limit, regime['name'])

    def check_sell(self, stock_code: str, trade_date, bar) -> LegalityResult:
        if bar is None:
            return LegalityResult(False, 'no_data')

        open_p = float(bar['open'])
        if np.isnan(open_p) or open_p <= 0:
            return LegalityResult(False, 'suspended')

        if int(bar.get('suspendFlag', 0)) == 1:
            return LegalityResult(False, 'suspended')

        up_limit, down_limit, regime = self._get_limit_band(stock_code, trade_date, bar)
        if regime['has_price_limit'] and down_limit is not None:
            if open_p <= down_limit + EPSILON:
                return LegalityResult(False, 'limit_down', up_limit, down_limit, regime['name'])

        return LegalityResult(True, 'ok', up_limit, down_limit, regime['name'])


def _get_limit_band_with_st(stock_code: str, trade_date, bar, is_st: bool,
                            list_date=None
                            ) -> Tuple[Optional[float], Optional[float], dict]:
    """离线版 get_limit_band_from_ratio：ST 状态从外部传入，不走 CNINFO。"""
    from utils.stock.info import (
        _to_trade_date,
        _is_limit_exempt_window,
        _parse_list_date,
        _round_limit_price,
        is_bse_stock, is_cyb_stock, is_kcb_stock,
    )
    from datetime import date as _date

    trade_day = _to_trade_date(trade_date)

    # 模拟 StockDetail 只包含 list_date，避免调用 get_stock_detail（SharedMemory 页面文件压力）
    detail = {'OpenDate': list_date.strftime('%Y%m%d')} if list_date else None

    # 1. 豁免窗口
    if _is_limit_exempt_window(stock_code, trade_day, detail):
        return None, None, {
            'name': 'unlimited', 'ratio': None, 'is_st': is_st,
            'has_price_limit': False, 'is_limit_exempt': True,
        }

    # 2. IPO 首日 44%
    list_date = _parse_list_date(detail)
    if list_date is not None and trade_day == list_date and trade_day >= _date(2014, 1, 1):
        regime = {
            'name': 'ipo_first_day', 'ratio': 0.44, 'is_st': is_st,
            'has_price_limit': True, 'is_limit_exempt': False,
        }
        pre_close = float(bar['preClose'])
        if pre_close <= 0 or np.isnan(pre_close):
            issue_p = bar.get('issuePrice')
            if issue_p is not None and not np.isnan(float(issue_p)) and float(issue_p) > 0:
                pre_close = float(issue_p)
        if pre_close <= 0 or np.isnan(pre_close):
            return None, None, regime
        return (_round_limit_price(pre_close, 0.44, True),
                _round_limit_price(pre_close, 0.44, False), regime)

    # 3. 北交所 30%
    if is_bse_stock(stock_code):
        ratio = 0.30
    # 4. 创业板 20%/10%
    elif is_cyb_stock(stock_code):
        ratio = 0.20 if trade_day >= _date(2020, 8, 24) else 0.10
    # 5. 科创板 20%
    elif is_kcb_stock(stock_code):
        ratio = 0.20
    # 6. ST 5%
    elif is_st:
        ratio = 0.05
    # 7. 主板 10%
    else:
        ratio = 0.10

    regime_name = 'st' if is_st else {
        0.30: 'bse', 0.20: 'cyb', 0.10: 'main_board',
        0.05: 'st',
    }.get(ratio, 'main_board')
    if is_cyb_stock(stock_code):
        regime_name = 'cyb'
    elif is_kcb_stock(stock_code):
        regime_name = 'kcb'

    regime = {
        'name': regime_name, 'ratio': ratio, 'is_st': is_st,
        'has_price_limit': True, 'is_limit_exempt': False,
    }

    pre_close = float(bar['preClose'])
    if pre_close <= 0 or np.isnan(pre_close):
        return None, None, regime

    return (_round_limit_price(pre_close, ratio, True),
            _round_limit_price(pre_close, ratio, False), regime)
