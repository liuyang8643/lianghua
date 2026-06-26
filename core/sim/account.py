from datetime import date as sys_date

from core.fees import (
  COMMISSION_RATE,
  MIN_COMMISSION,
  SIM_SLIPPAGE_RATE,
  STAMP_TAX_RATE,
  TRANSFER_FEE_RATE,
)


class StockAccountMocker:
  def __init__(
      self,
      cash: float,
      commission: float = COMMISSION_RATE,
      min_commission: float = MIN_COMMISSION,
      stamp_tax: float = STAMP_TAX_RATE,
      transfer_fee: float = TRANSFER_FEE_RATE,
      slippage: float = SIM_SLIPPAGE_RATE,
  ):
    self.init_cash = cash
    self.current_cash = cash
    self.commission = commission
    self.min_commission = min_commission
    self.stamp_tax = stamp_tax
    self.transfer_fee = transfer_fee
    self.slippage = slippage
    self.cleared_positions: list[dict] = []
    self.positions: dict[str, dict] = {}
    self.trade_log: list[dict] = []

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
      reason: str | None = None,
  ):
    cost = volume * price
    commission = self.calc_commission(cost)
    transfer_fee = self.calc_transfer_fee(cost)
    slippage = self.calc_slippage(cost)
    total_fee = commission + transfer_fee + slippage
    total_cost = cost + total_fee
    if total_cost > self.current_cash:
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
  ):
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
    })

    pos['commission'] += total_fee
    pos['volume'] -= volume
    pos['cost'] -= cost_basis

    if pos['volume'] == 0:
      cleared_pos = {
        **original_pos,
        'commission': original_pos['commission'] + total_fee,
      }
      del self.positions[code]
      self.cleared_positions.append({
        'code': code,
        'income': realized_income,
        'clear_date': sell_date,
        'clear_signal_date': signal_date,
        'clear_trade_date': sell_date,
        'clear_price': price,
        'pos': cleared_pos,
        'clear_reason': clear_reason,
        'price_field': price_field,
      })
    else:
      pos['avg_price'] = pos['cost'] / pos['volume']

  def write_off_stock(
      self,
      code: str,
      write_off_date: sys_date,
      write_off_reason: str = '退市归零',
      signal_date: sys_date | None = None,
      price_field: str = 'delist_zero',
  ):
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
    })

    del self.positions[code]
    self.cleared_positions.append({
      'code': code,
      'income': realized_income,
      'clear_date': write_off_date,
      'clear_signal_date': signal_date,
      'clear_trade_date': write_off_date,
      'clear_price': 0.0,
      'pos': original_pos,
      'clear_reason': write_off_reason,
      'price_field': price_field,
    })

  def calc_position_values(self, prices: dict):
    res = []
    for code, pos in self.positions.items():
      if code not in prices:
        continue
      price = prices[code]
      res.append({
        'code': pos['code'], 'volume': pos['volume'], 'cost': pos['cost'],
        'commission': pos['commission'], 'buy_date': pos['buy_date'],
        'buy_signal_date': pos['buy_signal_date'], 'buy_trade_date': pos['buy_trade_date'],
        'avg_price': pos['avg_price'], 'price_field': pos['price_field'],
        'current_price': price, 'current_value': price * pos['volume'],
      })
    return res

  def calc_assets(self, prices: dict):
    market_value = sum(pos['current_value'] for pos in self.calc_position_values(prices))
    return {
      'cash': self.current_cash,
      'market_value': market_value,
      'total_asset': self.current_cash + market_value,
    }

  def get_trade_log(self) -> list[dict]:
    return [dict(record) for record in self.trade_log]
