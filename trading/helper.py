from xtquant import xtconstant, xtdata


def timing_symbol_to_qmt(symbol: str) -> str:
  return f"{symbol[2:]}.{symbol[:2].upper()}"


def get_index_close_today(symbol: str, trade_date) -> float:
  qmt_code = timing_symbol_to_qmt(symbol)
  ds = trade_date.strftime('%Y%m%d')
  data = xtdata.get_market_data_ex(['close'], [qmt_code], period='1d', start_time=ds, end_time=ds)
  df = data[qmt_code]
  return float(df['close'].iloc[-1])



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
  elif price_type == xtconstant.OPT_AFTER_FIX_BUY:
    return '盘后固定价格买入'
  elif price_type == xtconstant.OPT_AFTER_FIX_SELL:
    return '盘后固定价格卖出'
  else:
    return str(price_type)

def get_price_type(order_type: int, stock_code: str, price: float = None) -> int:
  """尾盘收盘价成交：盘后固定价格委托（15:05-15:30 撮合）。
  prType=49(盘后定价)，QMT 策略框架文档的枚举值，1043/1044 在旧版 xtquant 不支持。"""
  if price is not None:
    return 49  # 盘后定价
  return xtconstant.MARKET_PEER_PRICE_FIRST
