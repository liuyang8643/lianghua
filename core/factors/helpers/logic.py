from typing import Callable

from .interface import *

class LogicEvery(BaseFactor):
  """
  判断传入策略均为目标信号，否则返回 None 信号
  """

  def __init__(self, target_signal: TradeSignal, factors: list[BaseFactor]):
    super().__init__()
    self.target_signal = target_signal
    self.factors = factors

  def execute(self, method_name: str, ctx: JudgeCtx) -> JudgeResult:
    res: list[JudgeResult] = []
    for factor in self.factors:
      result = getattr(factor, method_name)(ctx)
      res.append(result)
      if result['signal'] != self.target_signal:
        return JudgeResult(
          signal=None,
          trigger_price=result['trigger_price'],
          factor=result['factor'],
          reason=f'[{result["signal"]}]{factor.__class__.__name__}->{result["reason"]}'
        )

    trigger_prices = [x['trigger_price'] for x in filter(lambda x: x['trigger_price'] is not None, res)]
    return JudgeResult(
      signal=self.target_signal,
      trigger_price=(max if self.target_signal == TradeSignal.Buy else min)(trigger_prices) if trigger_prices else 0,
      factor=",".join(map(lambda x: x['factor'], filter(lambda x: x['factor'], reversed(res)))).strip(","),
      reason=",".join(map(lambda x: x['reason'], filter(lambda x: x['reason'], reversed(res)))).strip(","),
    )

  def real_trade(self, ctx: JudgeCtx) -> JudgeResult:
    return self.execute('real_trade', ctx)

  def mock_trade_daily(self, ctx: JudgeCtx) -> JudgeResult:
    return self.execute('mock_trade_daily', ctx)

class LogicAny(BaseFactor):
  """
  判断传入策略任意为目标信号，否则返回 None 信号
  """

  def __init__(self, target_signal: TradeSignal, factors: list[BaseFactor]):
    super().__init__()
    self.target_signal = target_signal
    self.factors = factors

  def execute(self, method_name: str, ctx: JudgeCtx) -> JudgeResult:
    res: list[JudgeResult] = []
    for factor in self.factors:
      result = getattr(factor, method_name)(ctx)
      res.append(result)
      if result['signal'] == self.target_signal:
        return JudgeResult(
          signal=self.target_signal,
          trigger_price=result['trigger_price'],
          factor=result['factor'],
          reason=f'[{result["signal"]}]{factor.__class__.__name__}->{result["reason"]}'
        )

    trigger_prices = [x['trigger_price'] for x in filter(lambda x: x['trigger_price'] is not None, res)]
    return JudgeResult(
      signal=None,
      trigger_price=(max if self.target_signal == TradeSignal.Buy else min)(trigger_prices) if trigger_prices else 0,
      factor="|".join(map(lambda x: x['factor'], reversed(res))),
      reason="|".join(map(lambda x: f"[{x['signal']}]{x['reason']}", reversed(res)))
    )

  def real_trade(self, ctx: JudgeCtx) -> JudgeResult:
    return self.execute('real_trade', ctx)

  def mock_trade_daily(self, ctx: JudgeCtx) -> JudgeResult:
    return self.execute('mock_trade_daily', ctx)

class LogicNot(BaseFactor):
  """
  判断传入策略不为目标信号，否则返回 None 信号
  """

  def __init__(self, target_signal: TradeSignal, factor: BaseFactor):
    super().__init__()
    self.target_signal = target_signal
    self.factor = factor

  def execute(self, method_name: str, ctx: JudgeCtx) -> JudgeResult:
    result = getattr(self.factor, method_name)(ctx)
    if result['signal'] != self.target_signal and result['signal'] is not None:
      return JudgeResult(
        signal=self.target_signal,
        trigger_price=None,
        factor=result['factor'],
        reason=''
      )
    return JudgeResult(
      signal=None,
      trigger_price=result['trigger_price'],
      factor=result['factor'],
      reason=f'[Not][{result["signal"]}]{self.factor.__class__.__name__}->{result["reason"]}'
    )

  def real_trade(self, ctx: JudgeCtx) -> JudgeResult:
    return self.execute('real_trade', ctx)

  def mock_trade_daily(self, ctx: JudgeCtx) -> JudgeResult:
    return self.execute('mock_trade_daily', ctx)

class LogicExpect(BaseFactor):
  """
  判断传入策略返回值为期望信号，则返回目标信号，否则返回 None 信号
  """

  def __init__(self, target_signal: TradeSignal, expect_signal: TradeSignal, factor: BaseFactor):
    super().__init__()
    self.target_signal = target_signal
    self.expect_signal = expect_signal
    self.factor = factor

  def execute(self, method_name: str, ctx: JudgeCtx) -> JudgeResult:
    result = getattr(self.factor, method_name)(ctx)
    if result['signal'] == self.expect_signal:
      return JudgeResult(
        signal=self.target_signal,
        trigger_price=result['trigger_price'],
        factor=result['factor'],
        reason=''
      )
    return JudgeResult(
      signal=None,
      trigger_price=result['trigger_price'],
      factor=result['factor'],
      reason=f'[Unexpected:{self.target_signal}][{result["signal"]}]{self.factor.__class__.__name__}->{result["reason"]}'
    )

  def real_trade(self, ctx: JudgeCtx) -> JudgeResult:
    return self.execute('real_trade', ctx)

  def mock_trade_daily(self, ctx: JudgeCtx) -> JudgeResult:
    return self.execute('mock_trade_daily', ctx)

class LogicBool(BaseFactor):
  """
  判断传入函数返回值为 True 则返回目标信号，否则返回 None 信号
  """

  def __init__(
      self,
      target_signal: TradeSignal,
      judge_desc: str,
      judge_func: Callable[[JudgeCtx], bool],
      trigger_price_func: Callable[[JudgeCtx], float] = lambda ctx: ctx.get_current_price()
  ):
    super().__init__()
    self.target_signal = target_signal
    self.judge_desc = judge_desc
    self.judge_func = judge_func
    self.trigger_price_func = trigger_price_func

  def execute(self, ctx: JudgeCtx) -> JudgeResult:
    result = self.judge_func(ctx)
    if result:
      return JudgeResult(
        signal=self.target_signal,
        trigger_price=self.trigger_price_func(ctx),
        factor=self.__class__.__name__,
        reason=f'[Is]{self.judge_desc}'
      )
    return JudgeResult(
      signal=None,
      trigger_price=self.trigger_price_func(ctx),
      factor=self.__class__.__name__,
      reason=f'[Not]{self.judge_desc}'
    )

  def real_trade(self, ctx: JudgeCtx) -> JudgeResult:
    return self.execute(ctx)

  def mock_trade_daily(self, ctx: JudgeCtx) -> JudgeResult:
    return self.execute(ctx)
