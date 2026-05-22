from typing import Optional
from xtquant import xtconstant

from data.db import get_stock_detail

def get_order_type_label(order_type):
  if order_type == xtconstant.STOCK_BUY:
    return '买入'
  elif order_type == xtconstant.STOCK_SELL:
    return '卖出'
  else:
    return '未知'

def get_order_status_label(order_status):
  """
  订单状态
  https://dict.thinktrader.net/nativeApi/xttrader.html?id=e2M5nZ#%E5%A7%94%E6%89%98%E7%8A%B6%E6%80%81-order-status
  """
  if order_status == xtconstant.ORDER_UNREPORTED:
    return '未报'
  elif order_status == xtconstant.ORDER_WAIT_REPORTING:
    return '待报'
  elif order_status == xtconstant.ORDER_REPORTED:
    return '已报'
  elif order_status == xtconstant.ORDER_REPORTED_CANCEL:
    return '已报待撤'
  elif order_status == xtconstant.ORDER_PARTSUCC_CANCEL:
    return '部成待撤'
  elif order_status == xtconstant.ORDER_PART_CANCEL:
    return '部撤（已经有一部分成交，剩下的已经撤单）'
  elif order_status == xtconstant.ORDER_CANCELED:
    return '已撤'
  elif order_status == xtconstant.ORDER_PART_SUCC:
    return '部成（已经有一部分成交，剩下的待成交）'
  elif order_status == xtconstant.ORDER_SUCCEEDED:
    return '已成'
  elif order_status == xtconstant.ORDER_JUNK:
    return '废单'
  else:
    return '未知'

def get_price_type_label(price_type: int) -> str:
  if price_type == xtconstant.MARKET_PEER_PRICE_FIRST:
    return '对手方最优价格委托'
  elif price_type == xtconstant.MARKET_SH_CONVERT_5_CANCEL:
    return '上交所五档即成剩撤'
  elif price_type == xtconstant.MARKET_SZ_CONVERT_5_CANCEL:
    return '深交所五档即成剩撤'
  elif price_type == xtconstant.FIX_PRICE:
    return '限价委托'
  else:
    return str(price_type)

def get_price_type(order_type: int, stock_code: str, price: float = None) -> Optional[int]:
  if price:
    # 固定价格委托
    return xtconstant.FIX_PRICE

  if order_type == xtconstant.STOCK_BUY:
    ''' 对手方最优价格委托 '''
    return xtconstant.MARKET_PEER_PRICE_FIRST
  elif order_type == xtconstant.STOCK_SELL:
    ''' 五档即成剩撤 '''
    detail = get_stock_detail(stock_code)
    if detail['ExchangeID'] == 'SH':
      return xtconstant.MARKET_SH_CONVERT_5_CANCEL
    elif detail['ExchangeID'] == 'SZ':
      return xtconstant.MARKET_SZ_CONVERT_5_CANCEL
    else:
      return xtconstant.MARKET_PEER_PRICE_FIRST

  return None  # 未知订单类型，返回 None
