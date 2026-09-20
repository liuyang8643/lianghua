import os
from datetime import datetime
from typing import List, Optional
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount, XtOrder, XtAsset, XtPosition, XtTrade
from xtquant import xtconstant, xtdata

from trade.broker.helper import get_price_type
from trade.broker.watcher import TraderCallback
from trade.logger import trading_logger
from utils.stock.info import is_convertible_bond

qmt_data_dir = os.path.dirname(xtdata.get_data_dir())

class Trader:
  strategy_name = 'FlashMan'

  def __init__(self, account_id: str):
    # 创建交易对象
    self.client = XtQuantTrader(
      qmt_data_dir,
      int(datetime.now().timestamp())
    )
    # 开启主动请求接口的专用线程 开启后在on_stock_xxx回调函数里调用XtQuantTrader.query_xxx函数不会卡住回调线程，但是查询和推送的数据在时序上会变得不确定
    # 详见: https://dict.thinktrader.net/nativeApi/xttrader.html?id=e2M5nZ#%E5%BC%80%E5%90%AF%E4%B8%BB%E5%8A%A8%E8%AF%B7%E6%B1%82%E6%8E%A5%E5%8F%A3%E7%9A%84%E4%B8%93%E7%94%A8%E7%BA%BF%E7%A8%8B
    self.client.set_relaxed_response_order_enabled(True)
    # 创建交易回调类对象，并声明接收回调
    self.client.register_callback(TraderCallback(self))
    # 启动交易线程
    self.client.start()
    # 连接交易服务器
    connect_res = self.client.connect()
    if connect_res != 0:
      raise Exception(f'Connect failed, connect_res: {connect_res}')
    # 创建账号对象
    self.account = StockAccount(account_id, 'STOCK')
    # 订阅账号
    subscribe_res = self.client.subscribe(self.account)
    if subscribe_res != 0:
      raise Exception(f'Subscribe failed, subscribe_res: {subscribe_res}')

  def order(
      self,
      order_type: int,  # xtconstant.STOCK_BUY, xtconstant.STOCK_SELL
      stock_code: str,
      volume: int,
      price: float = None,
      order_remark=''
  ):
    """ 下单 """
    trading_logger.info(f"[QMT] order_stock {stock_code} {'买' if order_type == xtconstant.STOCK_BUY else '卖'} {volume}股 ...")
    price_type = get_price_type(order_type, stock_code, price)
    order_id = self.client.order_stock(
      self.account,
      stock_code,
      order_type,
      volume,
      price_type=price_type,
      price=0 if price is None else price,
      strategy_name=self.strategy_name,
      order_remark=order_remark,
    )

    if order_id > 0:
      return order_id
    else:
      raise Exception(f'Order failed, order_id: {order_id}')

  def cancel_order(self, order_id: int):
    """ 撤单 """
    trading_logger.info(f"[QMT] cancel_order_stock {order_id} ...")
    res = self.client.cancel_order_stock(self.account, order_id)
    if res != 0:
      raise Exception(f'Cancel order({order_id}) failed: {res}')

  def query_order(self, order_id: int) -> Optional[XtOrder]:
    """查询委托"""
    return self.client.query_stock_order(self.account, order_id)

  def query_asset(self) -> Optional[XtAsset]:
    """查询证券资产"""
    return self.client.query_stock_asset(self.account)

  def query_positions(self) -> Optional[List[XtPosition]]:
    """查询持仓"""
    return self.client.query_stock_positions(self.account)

  def query_all_trades(self) -> List[XtTrade]:
    """查询当日全部成交，由 BrokerExecutor 匹配已登记委托并写入 journal。"""
    trading_logger.info("[QMT] query_stock_trades (全部成交) ...")
    all_trades = self.client.query_stock_trades(self.account)
    return list(all_trades) if all_trades is not None else []

def get_position(trader: Trader, only_can_sell: bool) -> List[XtPosition]:
  return [
    p
    for p in trader.query_positions()
    if (not only_can_sell or p.can_use_volume > 0) and not is_convertible_bond(p.stock_code)
  ]

