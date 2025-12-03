from datetime import datetime, date
from typing import Optional
from pandas import DataFrame
import talib
import numpy as np

from core.database import StockDetail, StockTradingData

class FactorCtx:
  """判断上下文，包含股票代码、时间"""

  def __init__(self, code: str, base_time: datetime):
    self.code = code
    self.base_time = base_time

  def get_stock_detail(self) -> StockDetail:
    """获取股票详情"""
    from core.database import get_stock_detail

    detail = get_stock_detail(self.code)
    if detail is None:
      raise ValueError(f"获取股票详情失败: {self.code}")
    return detail

  def get_daily_data(self, pass_days: int) -> DataFrame:
    """获取日线数据"""
    from core.database import get_market_data_from_cache, get_market_data

    if self.base_time < datetime.combine(date.today(), datetime.min.time()):
      # 历史数据，使用缓存
      return get_market_data_from_cache(self.code, pass_days, self.base_time, '1d')
    else:
      # 今天或未来的数据，不使用缓存
      return get_market_data(self.code, pass_days, self.base_time, '1d')

  def get_today_data(self) -> StockTradingData:
    """获取今日数据"""
    return self.get_daily_data(1).iloc[-1]

  def get_current_price(self) -> float:
    """获取当前价格"""
    return self.get_today_data()['close']

  def get_yesterday_data(self) -> StockTradingData:
    """获取昨日数据"""
    return self.get_daily_data(2).iloc[-2]

  def get_minute_data(self, pass_minutes: int) -> DataFrame:
    """获取分钟数据"""
    from core.database import get_market_data_from_cache, get_market_data

    if self.base_time < datetime.combine(date.today(), datetime.min.time()):
      # 历史数据，使用缓存
      return get_market_data_from_cache(self.code, pass_minutes, self.base_time, '1m')
    else:
      # 今天或未来的数据，不使用缓存
      return get_market_data(self.code, pass_minutes, self.base_time, '1m')

  def get_minute_data_today(self, base_datetime: datetime) -> Optional[DataFrame]:
    """获取今日分钟数据"""
    from utils.stock.time import get_trading_pass_minute

    pass_minutes = get_trading_pass_minute(base_datetime) + 1  # 包含0930开盘的k线
    if pass_minutes < 2:
      # 如果当前时间在开盘前或非交易时间，直接返回 None
      return None

    return self.get_minute_data(pass_minutes)

  def get_minute_data_pass_days(self, pass_days: int) -> DataFrame:
    """获取过去N天的分钟数据"""
    if pass_days < 1:
      raise ValueError(f'[get_minute_data_pass_days]传入的天数必须大于0: {pass_days}')
    bar_count_day = 4 * 60 + 1  # 当日k线数量，+1是因为0930开盘的k线
    return self.get_minute_data(pass_days * bar_count_day)

  def get_amount_pass_days(self, pass_days: int) -> list[float]:
    """当前时间下过去 pass_days 天同期成交额"""
    from utils.stock.time import get_trading_pass_minute

    if pass_days < 1:
      raise ValueError(f'[get_amount_pass_days]传入的天数必须大于0: {pass_days}')

    bar_count = 4 * 60 + 1  # 每天的K线数量
    history_data = self.get_minute_data_pass_days(pass_days)
    latest_time = datetime.fromtimestamp(int(history_data.iloc[-1]['time']) // 1000)
    bar_count_pass = get_trading_pass_minute(latest_time) + 1
    return [sum(history_data['amount'].iloc[- bar_count_pass - d * bar_count:(-d * bar_count) if d else None]) for d in reversed(range(pass_days))]

  def get_maw(self, period: int) -> DataFrame:
    """获取MAW指标"""
    history_data = self.get_daily_data(period * 2)
    return ((history_data['open'] + history_data['close'] + history_data['low'] + history_data['high']) / 4 * history_data['amount']).rolling(window=period).mean() / \
      history_data['amount'].rolling(window=period).mean()

  def get_raise_ratio(self, period: int) -> float:
    """获取从底部涨幅比例"""
    history_data = self.get_daily_data(period)
    lowest_d = history_data[-period:]['low'].min()
    current_price = history_data.iloc[-1]['close']
    return (current_price - lowest_d) / lowest_d  # 从底部涨幅比例

  def get_raise_ratio_max(self, period: int) -> float:
    """获取最大涨幅比例"""
    history_data = self.get_daily_data(period)
    highest_d = history_data[-period:-1]['high'].max()
    lowest_d = history_data[-period:-1]['low'].min()
    return (highest_d - lowest_d) / lowest_d  # 从底部涨幅比例

  def get_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[float, float, float]:
    """
    获取MACD指标
    :param fast_period: 快速EMA周期
    :param slow_period: 慢速EMA周期
    :param signal_period: 信号线EMA周期
    :return: MACD线、信号线、柱状图序列
    """
    history_data = self.get_daily_data(slow_period)  # 获取足够的历史数据
    close_prices = np.array(history_data['close'].values, dtype=np.float64)
    macd, signal, hist = talib.MACD(
      close_prices,
      fastperiod=fast_period,
      slowperiod=slow_period,
      signalperiod=signal_period
    )
    return macd[-1], signal[-1], hist[-1]

  def get_bbi(self, period: int) -> DataFrame:
    """
    获取BBI指标（多空指数）
    BBI = (MA3 + MA6 + MA12 + MA24) / 4
    :param period: 额外获取的历史数据周期
    :return: BBI序列
    """
    history_data = self.get_daily_data(period + 24)  # 需要额外24天计算MA24
    close = (history_data['open'] + history_data['close'] + history_data['low'] + history_data['high']) / 4
    ma3 = close.rolling(window=3).mean()
    ma6 = close.rolling(window=6).mean()
    ma12 = close.rolling(window=12).mean()
    ma24 = close.rolling(window=24).mean()
    bbi = (ma3 + ma6 + ma12 + ma24) / 4
    return bbi

  def get_cci(self, period: int) -> DataFrame:
    """
    获取CCI指标（顺势指标）
    CCI = (TP - MA) / (0.015 * MD)
    其中：TP = (最高价 + 最低价 + 收盘价) / 3
         MA = TP的N日简单移动平均
         MD = TP的N日平均绝对偏差
    :param period: CCI计算周期
    :return: CCI序列
    """
    history_data = self.get_daily_data(period * 2)
    high = history_data['high']
    low = history_data['low']
    close = history_data['close']

    # 计算典型价格TP
    tp = (high + low + close) / 3

    # 计算TP的移动平均MA
    ma = tp.rolling(window=period).mean()

    # 计算平均绝对偏差MD
    md = tp.rolling(window=period).apply(lambda x: (abs(x - x.mean())).mean(), raw=False)

    # 计算CCI
    cci = (tp - ma) / (0.015 * md)

    return cci
