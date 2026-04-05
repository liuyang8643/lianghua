from datetime import datetime, date
from typing import Optional
from pandas import DataFrame
import talib
import numpy as np

from core.database import StockDetail, StockTradingData

class FactorCtx:
  """判断上下文，包含股票代码、时间"""

  def __init__(self, code: str, base_time: datetime, dividend_type: str = 'back'):
    self.code = code
    self.base_time = base_time
    self.dividend_type = dividend_type

  def get_stock_detail(self) -> StockDetail:
    """获取股票详情"""
    from core.database import get_stock_detail

    detail = get_stock_detail(self.code)
    if detail is None:
      raise ValueError(f"获取股票详情失败: {self.code}")
    return detail

  def get_daily_data(self, pass_days: int) -> DataFrame:
    """获取日线数据"""
    from core.database import get_market_data_from_cache

    return get_market_data_from_cache(self.code, pass_days, self.base_time, '1d', dividend_type=self.dividend_type)

  def get_today_data(self) -> StockTradingData:
    """获取今日数据"""
    return self.get_daily_data(1).iloc[-1]

  def get_current_price(self) -> float:
    """获取当前价格"""
    return self.get_today_data()['close']

  def get_raw_close(self) -> Optional[float]:
    """获取不复权收盘价（实际交易价格，用于面值退市风险判断）"""
    from core.database import get_market_data_from_cache

    try:
      data = get_market_data_from_cache(
        self.code, 1, self.base_time, '1d',
        allow_tainted=True, dividend_type='none',
      )
      if data is not None and not data.empty:
        return float(data.iloc[-1]['close'])
    except Exception:
      pass
    return None

  def get_yesterday_data(self) -> StockTradingData:
    """获取昨日数据"""
    return self.get_daily_data(2).iloc[-2]

  def get_minute_data(self, pass_minutes: int) -> DataFrame:
    """获取分钟数据"""
    from core.database import get_market_data_from_cache

    return get_market_data_from_cache(self.code, pass_minutes, self.base_time, '1m', dividend_type=self.dividend_type)

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

  def get_wma(self, period: int) -> Optional[DataFrame]:
    """获取WMA指标"""
    history_data = self.get_daily_data(period * 2)
    if history_data is None or len(history_data) < period:
      return None
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

  def get_kdj(self, period: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    获取KDJ指标的原始数据序列
    :param period: KDJ周期，默认9
    :return: K值序列、D值序列、J值序列
    """
    # talib.STOCH 需要 fastk_period + slowk_period + slowd_period 的数据
    # period(9) + slowk(3) + slowd(3) = 15 天，再加一些buffer用于差分计算
    date_span = period + 10
    history_data = self.get_daily_data(date_span)

    if history_data is None or history_data.empty:
      raise ValueError(f"获取KDJ指标失败，历史数据不足: 需要至少{date_span}天")

    high = history_data['high'].values
    low = history_data['low'].values
    close = history_data['close'].values

    # 使用talib计算KDJ（STOCH指标）
    k, d = talib.STOCH(
      high,
      low,
      close,
      fastk_period=period,
      slowk_period=3,
      slowd_period=3,
    )

    # 计算J值: J = 3K - 2D
    j = 3 * k - 2 * d

    # 返回完整的K、D、J序列
    return k, d, j

  def get_rsi(self, period: int = 14) -> float:
    """
    获取RSI指标（相对强弱指标）
    :param period: 计算周期，默认14天
    :return: 当前RSI值 [0, 100]
    """
    history_data = self.get_daily_data(period + 1)
    close_prices = np.array(history_data['close'].values, dtype=np.float64)
    rsi = talib.RSI(close_prices, timeperiod=period)
    return float(rsi[-1])

  def get_adx(self, period: int = 14) -> tuple[float, float, float]:
    """
    获取ADX指标（平均趋向指数）
    :param period: 计算周期，默认14天
    :return: (adx, plus_di, minus_di)
    """
    history_data = self.get_daily_data(period * 2)
    high = np.array(history_data['high'].values, dtype=np.float64)
    low = np.array(history_data['low'].values, dtype=np.float64)
    close = np.array(history_data['close'].values, dtype=np.float64)

    adx = talib.ADX(high, low, close, timeperiod=period)
    plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
    minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)

    return float(adx[-1]), float(plus_di[-1]), float(minus_di[-1])

  def get_bollinger_bands(self, period: int = 20, nbdevup: float = 2, nbdevdn: float = 2) -> tuple[float, float, float, float, float]:
    """
    获取布林带指标
    :param period: 计算周期，默认20天
    :param nbdevup: 上轨标准差倍数，默认2
    :param nbdevdn: 下轨标准差倍数，默认2
    :return: (upper, middle, lower, bb_position, bb_bandwidth)
            bb_position: 价格在布林带中的位置 [0, 1]，0=下轨，1=上轨
            bb_bandwidth: 布林带宽度 (upper - lower) / middle
    """
    history_data = self.get_daily_data(period + 1)
    close = np.array(history_data['close'].values, dtype=np.float64)

    upper, middle, lower = talib.BBANDS(close,
                                        timeperiod=period,
                                        nbdevup=nbdevup,
                                        nbdevdn=nbdevdn,
                                        matype=0)

    current_price = close[-1]
    upper_val = upper[-1]
    middle_val = middle[-1]
    lower_val = lower[-1]

    # 计算价格在布林带中的位置
    if upper_val > lower_val:
      bb_position = (current_price - lower_val) / (upper_val - lower_val)
    else:
      bb_position = 0.5

    # 计算布林带宽度
    if middle_val > 0:
      bb_bandwidth = (upper_val - lower_val) / middle_val
    else:
      bb_bandwidth = 0.0

    return float(upper_val), float(middle_val), float(lower_val), float(bb_position), float(bb_bandwidth)

  def get_willr(self, period: int = 14) -> float:
    """
    获取威廉指标（Williams %R）
    :param period: 计算周期，默认14天
    :return: 当前WILLR值 [-100, 0]
    """
    history_data = self.get_daily_data(period + 1)
    high = np.array(history_data['high'].values, dtype=np.float64)
    low = np.array(history_data['low'].values, dtype=np.float64)
    close = np.array(history_data['close'].values, dtype=np.float64)

    willr = talib.WILLR(high, low, close, timeperiod=period)
    return float(willr[-1])

  def get_sar(self, acceleration: float = 0.02, maximum: float = 0.2) -> float:
    """
    获取抛物线SAR指标
    :param acceleration: 加速因子，默认0.02
    :param maximum: 最大加速因子，默认0.2
    :return: 当前SAR值
    """
    history_data = self.get_daily_data(50)  # SAR需要较多历史数据
    high = np.array(history_data['high'].values, dtype=np.float64)
    low = np.array(history_data['low'].values, dtype=np.float64)

    sar = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
    return float(sar[-1])

  def get_trix(self, period: int = 30) -> float:
    """
    获取TRIX指标（三重指数平滑平均线）
    :param period: 计算周期，默认30天
    :return: 当前TRIX值（百分比形式）
    """
    history_data = self.get_daily_data(period * 3 + 1)
    close = np.array(history_data['close'].values, dtype=np.float64)

    trix = talib.TRIX(close, timeperiod=period)
    return float(trix[-1])

  def get_mom(self, period: int = 10) -> float:
    """
    获取MOM动量指标（Momentum）
    :param period: 计算周期，默认10天
    :return: 当前动量值（价格变化）
    """
    history_data = self.get_daily_data(period + 1)
    close = np.array(history_data['close'].values, dtype=np.float64)

    mom = talib.MOM(close, timeperiod=period)
    return float(mom[-1])

  def get_mom_ratio(self, period: int = 10) -> float:
    """
    获取MOM动量比率（Momentum Ratio，百分比形式）
    :param period: 计算周期，默认10天
    :return: 动量比率 (当前价格 - N日前价格) / N日前价格 * 100
    """
    history_data = self.get_daily_data(period + 1)
    close = history_data['close'].values

    if len(close) < period + 1:
      raise ValueError(f"历史数据不足，需要至少{period + 1}天")

    current_price = close[-1]
    past_price = close[-(period + 1)]

    if past_price == 0:
      return 0.0

    mom_ratio = ((current_price - past_price) / past_price) * 100.0
    return float(mom_ratio)

  def get_roc(self, period: int = 10) -> float:
    """
    获取ROC指标（Rate of Change，变动率）
    :param period: 计算周期，默认10天
    :return: 当前ROC值（百分比形式）
    """
    history_data = self.get_daily_data(period + 1)
    close = np.array(history_data['close'].values, dtype=np.float64)

    roc = talib.ROC(close, timeperiod=period)
    return float(roc[-1])

  def get_benchmark_pct_change(self, period: int) -> Optional[np.ndarray]:
    """获取大盘涨跌幅序列"""
    from core.database import get_market_data_from_cache
    from utils.stock.info import baseline_stock_code

    benchmark_data = get_market_data_from_cache(
      baseline_stock_code,
      period,
      self.base_time,
      '1d',
      allow_tainted=True,
      dividend_type=self.dividend_type,
    )
    if benchmark_data is None or benchmark_data.empty:
      return None

    closes = benchmark_data['close'].values
    pre_closes = benchmark_data['preClose'].values
    return ((closes - pre_closes) / pre_closes) * 100
