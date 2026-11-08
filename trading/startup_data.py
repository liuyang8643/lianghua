"""启动/跨日的数据预检与历史补齐；不触发交易或当日行情落盘。"""
from datetime import timedelta
from zipfile import BadZipFile
from zlib import error as ZlibError

import numpy as np

from core.runtime import latest_runtime_npz_path, _1D_FIELDS, _2D_FIELDS
from trading.logger import trading_logger
from utils.stock.time import get_last_trading_day, get_trading_date_span


def validate_startup_runtime(previous, path=None):
    """校验可读性、面板形状、近期交易日连续性及 T-1 有效行情。

    停牌、未上市、未披露财务等位置允许 NaN，不能当作下载失败。
    """
    path = path or latest_runtime_npz_path()
    with np.load(path, allow_pickle=False) as runtime:
        dates = runtime['trade_dates'].astype('datetime64[D]')
        codes = runtime['stock_codes']
        if dates.ndim != 1 or not len(dates) or np.any(dates[1:] <= dates[:-1]):
            raise ValueError('Runtime 交易日为空、重复或乱序')
        if codes.ndim != 1 or not len(codes) or len(np.unique(codes)) != len(codes):
            raise ValueError('Runtime 股票列表为空或重复')
        expected = np.array(get_trading_date_span(previous - timedelta(days=60), previous),
                            dtype='datetime64[D]')
        missing = expected[~np.isin(expected, dates)]
        if len(missing):
            raise ValueError(f'Runtime 缺少历史交易日 {missing[0]} ~ {missing[-1]}')
        row = int(np.searchsorted(dates, np.datetime64(previous)))
        valid_prices = np.ones(len(codes), dtype=bool)
        for field in _2D_FIELDS:
            values = runtime[field]
            if values.shape != (len(dates), len(codes)):
                raise ValueError(f'Runtime {field} 面板形状错误: {values.shape}')
            if field in ('open', 'close', 'preClose'):
                valid_prices &= np.isfinite(values[row]) & (values[row] > 0)
        if not np.any(valid_prices):
            raise ValueError(f'Runtime {previous} 无完整有效行情')
        for field in _1D_FIELDS:
            if runtime[field].shape != (len(codes),):
                raise ValueError(f'Runtime {field} 股票维度错误')
    return path


def ensure_startup_data(store):
    """立即补齐缺失历史，成功重检才允许开始新的选股计划。"""
    from trading.post_close import run_update_all
    from utils.stock import time as stock_time
    try:
        _, _, calendar_end = stock_time._get_trading_calendar_state()
    except (OSError, ValueError, KeyError):
        calendar_end = None
    if calendar_end is None or calendar_end < store._now().date():
        from data.update_all import _update_trading_calendar
        trading_logger.warning('[启动数据检查] 交易日历缺失或过期，立即更新')
        _update_trading_calendar()
        _, _, calendar_end = stock_time._get_trading_calendar_state()
        if calendar_end is None or calendar_end < store._now().date():
            raise ValueError('交易日历更新后仍未覆盖当前日期')
    previous = get_last_trading_day(store._now().date() - timedelta(days=1))
    trading_logger.info(f'[启动数据检查] 校验 Runtime，至少覆盖至 {previous}')
    try:
        validate_startup_runtime(previous)
    except (OSError, ValueError, KeyError, EOFError, BadZipFile, ZlibError) as exc:
        trading_logger.warning(f'[启动数据检查] 数据不完整，立即补齐至 {previous}: {exc}')
        # 不能固定只补三天：停机多日时覆盖整个缺口。
        days = 60
        try:
            with np.load(latest_runtime_npz_path(), allow_pickle=False) as runtime:
                dates = runtime['trade_dates'].astype('datetime64[D]')
                if dates.ndim == 1 and len(dates):
                    days = max(days, int((np.datetime64(previous) - dates[-1])
                                        / np.timedelta64(1, 'D')) + 3)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile, ZlibError):
            pass  # 文件缺失/损坏由下面的下载与重建修复。
        if not run_update_all(store, history_end=previous, history_days=days):
            return False
        validate_startup_runtime(previous)
    trading_logger.info(f'[启动数据检查] 通过，历史数据已覆盖至 {previous}')
    return True
