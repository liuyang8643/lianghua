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
    from core.database import get_market_data

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
    from core.database import get_market_data

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

  def get_wma(self, period: int) -> DataFrame:
    """获取WMA指标"""
    history_data = self.get_daily_data(period * 2)
    return ((history_data['open'] + history_data['close'] + history_data['low'] + history_data['high']) / 4 * history_data['amount']).rolling(window=period).mean() / \
      history_data['amount'].rolling(window=period).mean()

  def get_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[float, float, float]:
    """
    获取MACD指标
    :param fast_period: 快速EMA周期
    :param slow_period: 慢速EMA周期
    :param signal_period: 信号线EMA周期
    :return: MACD线、信号线、柱状图序列
    """
    date_span = slow_period + signal_period - 1
    history_data = self.get_daily_data(date_span)  # 获取足够的历史数据

    if history_data is None or history_data.empty:
      raise ValueError(f"获取MACD指标失败，历史数据不足: 需要至少{date_span}天")
    close_prices = np.array(history_data['close'].values, dtype=np.float64)
    macd, signal, hist = talib.MACD(
      close_prices,
      fastperiod=fast_period,
      slowperiod=slow_period,
      signalperiod=signal_period
    )
    return macd[-1], signal[-1], hist[-1]

  def get_bbi(self) -> float:
    """
    获取BBI指标（多空指数）- 只返回最新值
    BBI = (MA3 + MA6 + MA12 + MA24) / 4
    :return: 当前BBI值
    """
    history_data = self.get_daily_data(25)  # 需要24天数据+当前1天
    # 使用numpy加速：直接在数组上计算，避免多次DataFrame操作
    close_array = ((history_data['open'].values + history_data['close'].values +
                    history_data['low'].values + history_data['high'].values) * 0.25)

    # 向量化计算所有MA值（只计算最后一个点）
    ma3 = close_array[-3:].mean()
    ma6 = close_array[-6:].mean()
    ma12 = close_array[-12:].mean()
    ma24 = close_array[-24:].mean()

    return (ma3 + ma6 + ma12 + ma24) * 0.25

  def get_cci(self) -> float:
    """
    获取CCI指标（顺势指标）- 只返回最新值
    CCI = (TP - MA) / (0.015 * MD)
    其中：TP = (最高价 + 最低价 + 收盘价) / 3
         MA = TP的N日简单移动平均
         MD = TP的N日平均绝对偏差
    :return: 当前CCI值
    """
    period = 14
    history_data = self.get_daily_data(period + 1)

    # 使用numpy直接计算，避免创建中间Series
    high = history_data['high'].values
    low = history_data['low'].values
    close = history_data['close'].values

    # 计算典型价格TP（向量化）
    tp = (high + low + close) / 3.0

    # 只需要最后period个数据点
    tp_window = tp[-period:]

    # 计算MA和MD（纯numpy，更快）
    ma = tp_window.mean()
    md = np.abs(tp_window - ma).mean()

    # 计算当前CCI值
    current_tp = tp[-1]
    cci = (current_tp - ma) / (0.015 * md) if md > 0 else 0.0

    return cci
