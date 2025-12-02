from .helpers import *

class MAWGoldCross(BaseFactor):
  """
  判断当前股票是否MAW金叉
  """

  def __init__(self, fast_period, slow_period):
    super().__init__()
    self.fast_period = fast_period
    self.slow_period = slow_period

  def real_trade(self, ctx: JudgeCtx) -> JudgeResult:
    current_price = ctx.get_current_price()
    maw_fast = ctx.get_maw(self.fast_period)
    maw_slow = ctx.get_maw(self.slow_period)

    if maw_fast.iloc[-2] < maw_slow.iloc[-2] and maw_fast.iloc[-1] >= maw_slow.iloc[-1]:
      return JudgeResult(
        signal=TradeSignal.Buy,
        trigger_price=current_price,
        factor=self.__class__.__name__,
        reason=f'当前 MAW({self.fast_period},{self.slow_period}) 金叉'
      )

    return JudgeResult(
      signal=TradeSignal.Hold,
      trigger_price=current_price,
      factor=self.__class__.__name__,
      reason=f'当前非 MAW({self.fast_period},{self.slow_period}) 金叉'
    )

  def mock_trade_daily(self, ctx: JudgeCtx) -> JudgeResult:
    return self.real_trade(ctx)

class MAWShortArrangement(BaseFactor):
  """
  判断当前股票是否空头排列
  """

  def __init__(self, fast_period, slow_period, buffer=0):
    super().__init__()
    self.fast_period = fast_period
    self.slow_period = slow_period
    self.buffer = buffer

  def real_trade(self, ctx: JudgeCtx) -> JudgeResult:
    current_price = ctx.get_current_price()
    maw_fast = ctx.get_maw(self.fast_period)
    maw_slow = ctx.get_maw(self.slow_period)

    if maw_fast.iloc[-1] < maw_slow.iloc[-1] / (1 + self.buffer):
      return JudgeResult(
        signal=TradeSignal.Sell,
        trigger_price=current_price,
        factor=self.__class__.__name__,
        reason=f'当前 MAW({self.fast_period},{self.slow_period}) 空头排列（~{self.buffer:.2%}）'
      )

    return JudgeResult(
      signal=TradeSignal.Hold,
      trigger_price=current_price,
      factor=self.__class__.__name__,
      reason=f'当前非 MAW({self.fast_period},{self.slow_period}) 空头排列（~{self.buffer:.2%}）'
    )

  def mock_trade_daily(self, ctx: JudgeCtx) -> JudgeResult:
    return self.real_trade(ctx)

class MAWDropped(BaseFactor):
  """
  判断当前股票是否MAW下降
  """

  def __init__(self, period, count=2):
    super().__init__()
    self.period = period
    self.count = count
    if count < 1:
      raise ValueError("count must be greater than 0")
    if count > period - 1:
      raise ValueError("count must be less than period - 1")

  def real_trade(self, ctx: JudgeCtx) -> JudgeResult:
    maw = ctx.get_maw(self.period)
    current_price = ctx.get_current_price()

    if all([maw.iloc[-i - 1] > maw.iloc[-i] for i in range(1, self.count)]) and maw.iloc[-1] > current_price:
      return JudgeResult(
        signal=TradeSignal.Sell,
        trigger_price=current_price,
        factor=self.__class__.__name__,
        reason=f'股价 {current_price:.2f} 低于 MAW({self.period}) 且下降 {self.count} 天'
      )
    return JudgeResult(
      signal=TradeSignal.Hold,
      trigger_price=current_price,
      factor=self.__class__.__name__,
      reason=f'MAW({self.period}) 未下降 {self.count} 天'
    )

  def mock_trade_daily(self, ctx: JudgeCtx) -> JudgeResult:
    return self.real_trade(ctx)

class BelowMAW(BaseFactor):
  """
  判断当前股票是否股价持续低于MAW
  """

  def __init__(self, period, last_days=3):
    """
    判断当前股票是否股价持续低于MAW
    :param period: MAW周期
    :param last_days: 连续多少天低于MAW
    """
    super().__init__()
    self.period = period
    self.last_days = last_days

  def real_trade(self, ctx: JudgeCtx) -> JudgeResult:
    maw = ctx.get_maw(self.period)
    history_close_data = ctx.get_daily_data(self.period)['close']
    current_price = ctx.get_current_price()
    # 检查过去 last_days 天是否都低于 MAW
    if all(maw.iloc[-i - 1] > history_close_data.iloc[-i - 1] for i in range(self.last_days)):
      return JudgeResult(
        signal=TradeSignal.Sell,
        trigger_price=current_price,
        factor=self.__class__.__name__,
        reason=f'股价 {current_price:.2f} 连续低于 MAW({self.period}) {self.last_days} 天'
      )

    return JudgeResult(
      signal=TradeSignal.Hold,
      trigger_price=current_price,
      factor=self.__class__.__name__,
      reason=f'股价 {current_price:.2f} 未连续低于 MAW({self.period}) {self.last_days} 天'
    )

  def mock_trade_daily(self, ctx: JudgeCtx) -> JudgeResult:
    maw = ctx.get_maw(self.period)
    history_close_data = ctx.get_daily_data(self.period)['close']
    target_price = ctx.get_daily_data(self.period)['low'].iloc[-1]
    # 检查过去 last_days 天是否都低于 MAW
    if all(maw.iloc[-i - 2] > history_close_data.iloc[-i - 2] for i in range(self.last_days - 1)) and maw.iloc[-1] > target_price:
      return JudgeResult(
        signal=TradeSignal.Sell,
        trigger_price=target_price,
        factor=self.__class__.__name__,
        reason=f'股价 {target_price:.2f} 连续低于 MAW({self.period}) {self.last_days} 天'
      )

    return JudgeResult(
      signal=TradeSignal.Hold,
      trigger_price=history_close_data.iloc[-1],
      factor=self.__class__.__name__,
      reason=f'股价 {history_close_data.iloc[-1]:.2f} 未连续低于 MAW({self.period}) {self.last_days} 天'
    )

