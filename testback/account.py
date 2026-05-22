from typing import Dict, List, Optional, TypedDict
from datetime import date as sys_date, datetime


class MockStockPosition(TypedDict):
  code: str
  volume: int
  cost: float
  commission: float
  buy_date: sys_date
  buy_signal_date: sys_date
  buy_trade_date: sys_date
  avg_price: float
  price_field: str
  signal_dividend_type: str
  execution_dividend_type: str


class MockStockClearedPosition(TypedDict):
  code: str
  income: float
  clear_date: sys_date
  clear_signal_date: sys_date
  clear_trade_date: sys_date
  clear_price: float
  clear_reason: Optional[str]
  price_field: str
  signal_dividend_type: str
  execution_dividend_type: str
  pos: MockStockPosition


class DailyReturnRecord(TypedDict):
  date: sys_date
  total_asset: float
  daily_return: float


class TradeRecord(TypedDict):
  code: str
  action: str  # 'buy' or 'sell'
  date: sys_date
  signal_date: sys_date
  trade_date: sys_date
  price: float
  price_field: str
  volume: int
  amount: float
  commission: float
  cost: Optional[float]
  income: Optional[float]
  reason: Optional[str]
  signal_dividend_type: str
  execution_dividend_type: str


class StockAccountMocker:
  def __init__(
      self,
      cash: float,
      commission: float = 0.0000854,
      min_commission: float = 0.1,
      stamp_tax: float = 0.0005,
      transfer_fee: float = 0.00002,
      slippage: float = 0.001,
  ):
    self.init_cash = cash
    self.current_cash = cash
    self.commission = commission
    self.min_commission = min_commission
    self.stamp_tax = stamp_tax
    self.transfer_fee = transfer_fee
    self.slippage = slippage
    self.cleared_positions: list[MockStockClearedPosition] = []
    self.positions: Dict[str, MockStockPosition] = {}
    self.daily_returns: Dict[sys_date, float] = {}
    self.trade_log: List[TradeRecord] = []
    self._last_total_asset: float = cash

  def calc_commission(self, cost: float):
    return max(cost * self.commission, self.min_commission)

  def calc_stamp_tax(self, amount: float):
    return amount * self.stamp_tax

  def calc_transfer_fee(self, amount: float):
    return amount * self.transfer_fee

  def calc_slippage(self, amount: float):
    return amount * self.slippage

  def buy_stock(
      self,
      code: str,
      volume: int,
      price: float,
      buy_date: sys_date,
      signal_date: sys_date | None = None,
      price_field: str = 'open',
      signal_dividend_type: str = 'back',
      execution_dividend_type: str = 'back',
      reason: str | None = None,
  ):
    """ 买入股票 """
    cost = volume * price
    commission = self.calc_commission(cost)
    transfer_fee = self.calc_transfer_fee(cost)
    slippage = self.calc_slippage(cost)
    total_fee = commission + transfer_fee + slippage
    total_cost = cost + total_fee
    if total_cost > self.current_cash:
      # testback_logger = __import__('testback.logger', fromlist=['testback_logger']).testback_logger
      # testback_logger.warning(f'Cash not enough, skip buy: {code}, cost: {total_cost:.2f}, cash: {self.current_cash:.2f}')
      return False

    signal_date = signal_date or buy_date

    self.current_cash -= total_cost

    if code in self.positions:
      pos = self.positions[code]
      pos['volume'] += volume
      pos['cost'] += cost
      pos['commission'] += total_fee
      pos['avg_price'] = pos['cost'] / pos['volume']
    else:
      self.positions[code] = {
        'code': code,
        'volume': volume,
        'cost': cost,
        'commission': total_fee,
        'buy_date': buy_date,
        'buy_signal_date': signal_date,
        'buy_trade_date': buy_date,
        'avg_price': price,
        'price_field': price_field,
        'signal_dividend_type': signal_dividend_type,
        'execution_dividend_type': execution_dividend_type,
      }

    self.trade_log.append({
      'code': code,
      'action': 'buy',
      'date': buy_date,
      'signal_date': signal_date,
      'trade_date': buy_date,
      'price': price,
      'price_field': price_field,
      'volume': volume,
      'amount': cost,
      'commission': total_fee,
      'cost': cost,
      'income': None,
      'reason': reason,
      'signal_dividend_type': signal_dividend_type,
      'execution_dividend_type': execution_dividend_type,
    })
    return True

  def sell_stock(
      self,
      code: str,
      volume: int,
      price: float,
      sell_date: sys_date,
      clear_reason: str = None,
      signal_date: sys_date | None = None,
      price_field: str = 'open',
      signal_dividend_type: str = 'back',
      execution_dividend_type: str = 'back',
  ):
    """ 卖出股票 """
    if code not in self.positions:
      raise Exception(f'Position not found for code: {code}')

    pos = self.positions[code]
    if pos['volume'] < volume:
      raise Exception(f'Volume not enough, volume: {volume}, position volume: {pos["volume"]}')

    signal_date = signal_date or sell_date
    original_pos = dict(pos)
    cost_basis = original_pos['avg_price'] * volume
    gain = volume * price if price is not None else 0
    commission = self.calc_commission(gain)
    stamp_tax_fee = self.calc_stamp_tax(gain)
    transfer_fee = self.calc_transfer_fee(gain)
    slippage = self.calc_slippage(gain)
    total_fee = commission + stamp_tax_fee + transfer_fee + slippage
    total_gain = gain - total_fee
    realized_income = total_gain - cost_basis

    self.current_cash += total_gain

    self.trade_log.append({
      'code': code,
      'action': 'sell',
      'date': sell_date,
      'signal_date': signal_date,
      'trade_date': sell_date,
      'price': price,
      'price_field': price_field,
      'volume': volume,
      'amount': gain,
      'commission': total_fee,
      'cost': cost_basis,
      'income': realized_income,
      'reason': clear_reason,
      'signal_dividend_type': signal_dividend_type,
      'execution_dividend_type': execution_dividend_type,
    })

    pos['commission'] += total_fee
    pos['volume'] -= volume
    pos['cost'] -= cost_basis

    if pos['volume'] == 0:
      cleared_pos: MockStockPosition = {
        **original_pos,
        'commission': original_pos['commission'] + total_fee,
      }
      del self.positions[code]
      self.cleared_positions.append(MockStockClearedPosition(
        code=code,
        income=realized_income,
        clear_date=sell_date,
        clear_signal_date=signal_date,
        clear_trade_date=sell_date,
        clear_price=price,
        pos=cleared_pos,
        clear_reason=clear_reason,
        price_field=price_field,
        signal_dividend_type=signal_dividend_type,
        execution_dividend_type=execution_dividend_type,
      ))
    else:
      pos['avg_price'] = pos['cost'] / pos['volume']

  def clear_stock(
      self,
      code: str,
      price: float,
      clear_date: sys_date,
      clear_reason: str = None,
      signal_date: sys_date | None = None,
      price_field: str = 'open',
      signal_dividend_type: str = 'back',
      execution_dividend_type: str = 'back',
  ):
    """ 清仓股票 """
    if code not in self.positions:
      raise Exception(f'Position not found for code: {code}')
    self.sell_stock(
      code,
      self.positions[code]['volume'],
      price,
      clear_date,
      clear_reason,
      signal_date=signal_date,
      price_field=price_field,
      signal_dividend_type=signal_dividend_type,
      execution_dividend_type=execution_dividend_type,
    )

  def write_off_stock(
      self,
      code: str,
      write_off_date: sys_date,
      write_off_reason: str = '退市归零',
      signal_date: sys_date | None = None,
      price_field: str = 'delist_zero',
      signal_dividend_type: str = 'back',
      execution_dividend_type: str = 'back',
  ):
    """将持仓按零价值核销（如退市）。"""
    if code not in self.positions:
      raise Exception(f'Position not found for code: {code}')

    signal_date = signal_date or write_off_date
    pos = self.positions[code]
    original_pos = dict(pos)
    cost_basis = original_pos['cost']
    realized_income = -cost_basis

    self.trade_log.append({
      'code': code,
      'action': 'sell',
      'date': write_off_date,
      'signal_date': signal_date,
      'trade_date': write_off_date,
      'price': 0.0,
      'price_field': price_field,
      'volume': original_pos['volume'],
      'amount': 0.0,
      'commission': 0.0,
      'cost': cost_basis,
      'income': realized_income,
      'reason': write_off_reason,
      'signal_dividend_type': signal_dividend_type,
      'execution_dividend_type': execution_dividend_type,
    })

    del self.positions[code]
    self.cleared_positions.append(MockStockClearedPosition(
      code=code,
      income=realized_income,
      clear_date=write_off_date,
      clear_signal_date=signal_date,
      clear_trade_date=write_off_date,
      clear_price=0.0,
      pos=original_pos,
      clear_reason=write_off_reason,
      price_field=price_field,
      signal_dividend_type=signal_dividend_type,
      execution_dividend_type=execution_dividend_type,
    ))

  def calc_position_values(self, prices: dict = None):
    """获取持仓市值，直接构造返回值避免 **pos dict 展开。"""
    res = []
    for code, pos in self.positions.items():
      price = prices.get(code) if prices is not None else None
      if price is not None:
        res.append({
          'code': pos['code'], 'volume': pos['volume'], 'cost': pos['cost'],
          'commission': pos['commission'], 'buy_date': pos['buy_date'],
          'buy_signal_date': pos['buy_signal_date'], 'buy_trade_date': pos['buy_trade_date'],
          'avg_price': pos['avg_price'], 'price_field': pos['price_field'],
          'signal_dividend_type': pos['signal_dividend_type'],
          'execution_dividend_type': pos['execution_dividend_type'],
          'current_price': price, 'current_value': price * pos['volume'],
        })
    return res

  def get_position(self, code: str):
    """ 获取指定股票持仓 """
    return self.positions.get(code)

  def calc_assets(self, cur_time, prices: dict = None):
    """计算资产。cur_time 可为 date 或 datetime。"""
    market_value = sum(pos['current_value'] for pos in self.calc_position_values(prices))
    total_asset = self.current_cash + market_value

    if self._last_total_asset > 0:
      daily_return = (total_asset - self._last_total_asset) / self._last_total_asset * 100
      cur_date = cur_time.date() if hasattr(cur_time, 'date') else cur_time
      self.daily_returns[cur_date] = daily_return

    self._last_total_asset = total_asset

    return {
      'cash': self.current_cash,
      'market_value': market_value,
      'total_asset': total_asset,
    }

  def get_daily_returns(self) -> Dict[sys_date, float]:
    """获取所有交易日的每日收益率"""
    return self.daily_returns.copy()

  def get_cumulative_returns(self) -> List[float]:
    """获取累计收益率列表（%，从第1天开始）

    用于可视化报告。
    第1天收益率为0（或第1天的日收益率），
    后续天为复利累计。
    """
    if not self.daily_returns:
      return []

    sorted_dates = sorted(self.daily_returns.keys())
    cumulative = []
    cumulative_value = 1.0

    for d in sorted_dates:
      daily_ret = self.daily_returns[d]
      cumulative_value *= (1 + daily_ret / 100)
      cumulative.append((cumulative_value - 1) * 100)

    return cumulative

  def get_trade_log(self) -> List[Dict]:
    """获取交易日志"""
    return [dict(record) for record in self.trade_log]
