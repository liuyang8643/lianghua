from typing import Dict, Optional, TypedDict
from datetime import date as sys_date, datetime

from core.database import get_market_data_from_cache

class MockStockPosition(TypedDict):
  code: str
  volume: int
  cost: float
  commission: float
  buy_date: sys_date
  avg_price: float

class MockStockClearedPosition(TypedDict):
  code: str
  income: float
  clear_date: sys_date
  clear_price: float
  clear_reason: Optional[str]
  pos: MockStockPosition

class StockAccountMocker:
  def __init__(
      self,
      # 初始资金
      cash: float,
      # 交易费率
      commission: float,
      # 最小交易费
      min_commission: float,
  ):
    self.init_cash = cash
    self.current_cash = cash
    self.commission = commission
    self.min_commission = min_commission
    self.cleared_positions: list[MockStockClearedPosition] = []
    self.positions: Dict[str, MockStockPosition] = {}
    self.daily_cash: Dict[sys_date, float] = {}
    self.daily_assets: Dict[sys_date, float] = {}

  def calc_commission(self, cost: float):
    return max(cost * self.commission, self.min_commission)

  def buy_stock(self, code: str, volume: int, price: float, buy_date: sys_date):
    """ 买入股票 """
    cost = volume * price
    commission = self.calc_commission(cost)
    total_cost = cost + self.calc_commission(cost)
    if total_cost > self.current_cash:
      raise Exception(f'Cash not enough, cost: {total_cost}, cash: {self.current_cash}')
    self.current_cash -= total_cost
    self.daily_cash[buy_date] = self.current_cash
    if code in self.positions:
      pos = self.positions[code]
      pos['volume'] += volume
      pos['cost'] = pos['cost'] + cost
      pos['commission'] = pos['commission'] + commission
      pos['avg_price'] = pos['cost'] / pos['volume']
    else:
      self.positions[code] = {
        'code': code,
        'volume': volume,
        'cost': cost,
        'commission': commission,
        'buy_date': buy_date,
        'avg_price': price,
      }

  def sell_stock(self, code: str, volume: int, price: float, sell_date: sys_date, clear_reason: str = None):
    """ 卖出股票 """
    if code not in self.positions:
      raise Exception(f'Position not found for code: {code}')
    pos = self.positions[code]
    if pos['volume'] < volume:
      raise Exception(f'Volume not enough, volume: {volume}, position volume: {pos["volume"]}')
    gain = volume * price
    commission = self.calc_commission(gain)
    total_gain = gain - commission
    self.current_cash += total_gain
    self.daily_cash[sell_date] = self.current_cash
    pos['volume'] -= volume
    pos['commission'] += commission
    if pos['volume'] == 0:
      del self.positions[code]
      self.cleared_positions.append(
        {
          'code': code,
          'income': gain,
          'clear_date': sell_date,
          'clear_price': price,
          'pos': pos,
          'clear_reason': clear_reason,
        }
      )

  def clear_stock(self, code: str, price: float, clear_date: sys_date, clear_reason: str = None):
    """ 清仓股票 """
    if code not in self.positions:
      raise Exception(f'Position not found for code: {code}')
    self.sell_stock(code, self.positions[code]['volume'], price, clear_date, clear_reason)

  def calc_position_values(self, cur_time: datetime):
    """ 获取持仓 """
    res = []
    for pos in self.positions.values():
      latest_data = get_market_data_from_cache(pos['code'], 1, cur_time)
      current_price = latest_data.iloc[-1]['close'] if latest_data is not None and latest_data.size > 0 else pos['avg_price']
      res.append(
        {
          **pos,
          'current_price': current_price,
          'current_value': current_price * pos['volume']
        }
      )
    return res

  def get_position(self, code: str):
    """ 获取指定股票持仓 """
    return self.positions.get(code)

  def calc_assets(self, cur_time: datetime):
    market_value = sum([pos['current_value'] for pos in self.calc_position_values(cur_time)])
    total_asset = self.current_cash + market_value
    self.daily_assets[cur_time.date()] = total_asset
    return {
      'cash': self.current_cash,
      'market_value': market_value,
      'total_asset': total_asset,
    }
