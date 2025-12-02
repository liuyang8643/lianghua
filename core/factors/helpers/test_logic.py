import unittest
from unittest.mock import MagicMock

from .logic import *

class TestLogicEvery(unittest.TestCase):
  def test_all_factors_match_target_signal(self):
    target_signal = TradeSignal.Buy
    factors = [MagicMock(BaseFactor), MagicMock(BaseFactor)]
    factors[0].real_trade.return_value = JudgeResult(signal=TradeSignal.Buy, trigger_price=100, factor="Factor1", reason="Match1")
    factors[1].real_trade.return_value = JudgeResult(signal=TradeSignal.Buy, trigger_price=110, factor="Factor2", reason="Match2")

    logic = LogicEvery(target_signal, factors)
    ctx = MagicMock(JudgeCtx)
    result = logic.real_trade(ctx)

    self.assertEqual(TradeSignal.Buy, result['signal'])
    self.assertEqual(110, result['trigger_price'])  # 买入信号取最大价格
    self.assertEqual("Factor1,Factor2", result['factor'])
    self.assertEqual("Match1,Match2", result['reason'])

  def test_all_factors_match_target_signal_sell(self):
    target_signal = TradeSignal.Sell
    factors = [MagicMock(BaseFactor), MagicMock(BaseFactor)]
    factors[0].real_trade.return_value = JudgeResult(signal=TradeSignal.Sell, trigger_price=90, factor="Factor1", reason="Match1")
    factors[1].real_trade.return_value = JudgeResult(signal=TradeSignal.Sell, trigger_price=80, factor="Factor2", reason="Match2")

    logic = LogicEvery(target_signal, factors)
    ctx = MagicMock(JudgeCtx)
    result = logic.real_trade(ctx)

    self.assertEqual(TradeSignal.Sell, result['signal'])
    self.assertEqual(80, result['trigger_price'])  # 卖出信号取最小价格
    self.assertEqual("Factor1,Factor2", result['factor'])
    self.assertEqual("Match1,Match2", result['reason'])

  def test_not_all_factors_match_target_signal(self):
    target_signal = TradeSignal.Buy
    factors = [MagicMock(BaseFactor), MagicMock(BaseFactor)]
    factors[0].real_trade.return_value = JudgeResult(signal=TradeSignal.Buy, trigger_price=100, factor="Factor1", reason="Match")
    factors[1].real_trade.return_value = JudgeResult(signal=TradeSignal.Sell, trigger_price=90, factor="Factor2", reason="Mismatch")

    logic = LogicEvery(target_signal, factors)
    ctx = MagicMock(JudgeCtx)
    result = logic.real_trade(ctx)

    self.assertEqual(None, result['signal'])
    self.assertEqual(90, result['trigger_price'])
    self.assertEqual("Factor2", result['factor'])
    self.assertEqual("[TradeSignal.Sell]BaseStrategy->Mismatch", result['reason'])

class TestLogicAny(unittest.TestCase):
  def test_any_factor_matches_target_signal(self):
    target_signal = TradeSignal.Buy
    factors = [MagicMock(BaseFactor), MagicMock(BaseFactor)]
    factors[0].real_trade.return_value = JudgeResult(signal=TradeSignal.Sell, trigger_price=90, factor="Factor1", reason="Mismatch")
    factors[1].real_trade.return_value = JudgeResult(signal=TradeSignal.Buy, trigger_price=100, factor="Factor2", reason="Match")

    logic = LogicAny(target_signal, factors)
    ctx = MagicMock(JudgeCtx)
    result = logic.real_trade(ctx)

    self.assertEqual(TradeSignal.Buy, result['signal'])
    self.assertEqual(100, result['trigger_price'])
    self.assertEqual("Factor2", result['factor'])
    self.assertEqual("[TradeSignal.Buy]BaseStrategy->Match", result['reason'])

  def test_no_factor_matches_target_signal(self):
    target_signal = TradeSignal.Buy
    factors = [MagicMock(BaseFactor), MagicMock(BaseFactor)]
    factors[0].real_trade.return_value = JudgeResult(signal=TradeSignal.Sell, trigger_price=90, factor="Factor1", reason="Mismatch1")
    factors[1].real_trade.return_value = JudgeResult(signal=TradeSignal.Hold, trigger_price=100, factor="Factor2", reason="Mismatch2")

    logic = LogicAny(target_signal, factors)
    ctx = MagicMock(JudgeCtx)
    result = logic.real_trade(ctx)

    self.assertEqual(None, result['signal'])
    self.assertEqual(100, result['trigger_price'])  # 买入信号取最大价格
    self.assertEqual("Factor1|Factor2", result['factor'])
    self.assertEqual("[TradeSignal.Sell]Mismatch1|[TradeSignal.Hold]Mismatch2", result['reason'])

  def test_no_factor_matches_target_signal_sell(self):
    target_signal = TradeSignal.Sell
    factors = [MagicMock(BaseFactor), MagicMock(BaseFactor)]
    factors[0].real_trade.return_value = JudgeResult(signal=TradeSignal.Buy, trigger_price=110, factor="Factor1", reason="Mismatch1")
    factors[1].real_trade.return_value = JudgeResult(signal=TradeSignal.Hold, trigger_price=100, factor="Factor2", reason="Mismatch2")

    logic = LogicAny(target_signal, factors)
    ctx = MagicMock(JudgeCtx)
    result = logic.real_trade(ctx)

    self.assertEqual(None, result['signal'])
    self.assertEqual(100, result['trigger_price'])  # 卖出信号取最小价格
    self.assertEqual("Factor1|Factor2", result['factor'])
    self.assertEqual("[TradeSignal.Buy]Mismatch1|[TradeSignal.Hold]Mismatch2", result['reason'])

class TestLogicNot(unittest.TestCase):
  def test_target_signal_when_factor_signal_is_not_target(self):
    target_signal = TradeSignal.Buy
    factor = MagicMock(BaseFactor)
    factor.real_trade.return_value = JudgeResult(signal=TradeSignal.Sell, trigger_price=90, factor="Factor", reason="Test reason")
    factor.__class__.__name__ = "BaseStrategy"
    logic = LogicNot(target_signal, factor)
    ctx = MagicMock(JudgeCtx)

    result = logic.real_trade(ctx)

    self.assertEqual(target_signal, result['signal'])
    self.assertEqual(90, result['trigger_price'])
    self.assertEqual("Factor", result['factor'])
    self.assertEqual("", result['reason'])

  def test_none_signal_when_factor_signal_is_target(self):
    target_signal = TradeSignal.Buy
    factor = MagicMock(BaseFactor)
    factor.real_trade.return_value = JudgeResult(signal=target_signal, trigger_price=100, factor="Factor", reason="Test reason")
    factor.__class__.__name__ = "BaseStrategy"
    logic = LogicNot(target_signal, factor)
    ctx = MagicMock(JudgeCtx)

    result = logic.real_trade(ctx)

    self.assertEqual(None, result['signal'])
    self.assertEqual(100, result['trigger_price'])
    self.assertEqual("Factor", result['factor'])
    self.assertEqual("[Not][TradeSignal.Buy]BaseStrategy->Test reason", result['reason'])

  def test_none_signal_when_factor_signal_is_none(self):
    target_signal = TradeSignal.Buy
    factor = MagicMock(BaseFactor)
    factor.real_trade.return_value = JudgeResult(signal=None, trigger_price=100, factor="Factor", reason="Test reason")
    factor.__class__.__name__ = "BaseStrategy"
    logic = LogicNot(target_signal, factor)
    ctx = MagicMock(JudgeCtx)

    result = logic.real_trade(ctx)

    self.assertEqual(None, result['signal'])
    self.assertEqual(100, result['trigger_price'])
    self.assertEqual("Factor", result['factor'])
    self.assertEqual("[Not][None]BaseStrategy->Test reason", result['reason'])

class TestLogicExpect(unittest.TestCase):
  def test_logic_expect_true(self):
    target_signal = TradeSignal.Buy
    factor = MagicMock(BaseFactor)
    factor.real_trade.return_value = JudgeResult(signal=TradeSignal.Sell, trigger_price=90, factor="Factor", reason="Test reason")
    factor.__class__.__name__ = "BaseStrategy"

    logic = LogicExpect(target_signal, TradeSignal.Sell, factor)
    ctx = MagicMock(JudgeCtx)
    result = logic.real_trade(ctx)

    self.assertEqual(target_signal, result['signal'])
    self.assertEqual(90, result['trigger_price'])
    self.assertEqual("Factor", result['factor'])
    self.assertEqual("", result['reason'])

  def test_logic_expect_false(self):
    target_signal = TradeSignal.Buy
    factor = MagicMock(BaseFactor)
    factor.real_trade.return_value = JudgeResult(signal=TradeSignal.Hold, trigger_price=100, factor="Factor", reason="Test reason")
    factor.__class__.__name__ = "BaseStrategy"

    logic = LogicExpect(target_signal, TradeSignal.Sell, factor)
    ctx = MagicMock(JudgeCtx)
    result = logic.real_trade(ctx)

    self.assertEqual(None, result['signal'])
    self.assertEqual(100, result['trigger_price'])
    self.assertEqual("Factor", result['factor'])
    self.assertEqual("[Unexpected:TradeSignal.Buy][TradeSignal.Hold]BaseStrategy->Test reason", result['reason'])

class TestLogicBool(unittest.TestCase):
  def test_logic_bool_true(self):
    target_signal = TradeSignal.Buy
    ctx = MagicMock(JudgeCtx)
    trigger_price_func = lambda _: 100

    logic = LogicBool(target_signal, 'ShouldTrue', lambda _: True, trigger_price_func)
    result = logic.real_trade(ctx)

    self.assertEqual(target_signal, result['signal'])
    self.assertEqual(100, result['trigger_price'])
    self.assertEqual("ShouldTrue", result['factor'])
    self.assertEqual("", result['reason'])

  def test_logic_bool_false(self):
    target_signal = TradeSignal.Sell
    ctx = MagicMock(JudgeCtx)
    trigger_price_func = lambda _: 90

    logic = LogicBool(target_signal, 'ShouldFalse', lambda _: False, trigger_price_func)
    result = logic.real_trade(ctx)

    self.assertEqual(None, result['signal'])
    self.assertEqual(90, result['trigger_price'])
    self.assertEqual("ShouldFalse", result['factor'])
    self.assertEqual("[False]ShouldFalse", result['reason'])

if __name__ == '__main__':
  unittest.main()
