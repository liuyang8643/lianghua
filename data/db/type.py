from typing import TypedDict

class StockDetail(TypedDict):
  """ 股票详情 """
  ExchangeID: str  # 合约市场代码
  InstrumentID: str  # 合约代码
  InstrumentName: str  # 合约名称
  ProductID: str  # 合约的品种ID (期货)
  ProductName: str  # 合约的品种名称 (期货)
  ProductType: int  # 合约的类型, 默认-1，枚举值可参考文档说明
  ExchangeCode: str  # 交易所代码
  UniCode: str  # 统一规则代码
  CreateDate: str  # 创建日期
  OpenDate: str  # 上市日期 (特殊值情况见表末)
  ExpireDate: str  # 退市日或者到期日 (特殊值情况见表末)
  PreClose: float  # 前收盘价格
  SettlementPrice: float  # 前结算价格
  UpStopPrice: float  # 当日涨停价
  DownStopPrice: float  # 当日跌停价
  FloatVolume: float  # 流通股本 (注意, 部分低等级客户端中此字段为 FloatVolumn)
  TotalVolume: float  # 总股本 (注意, 部分低等级客户端中此字段为 FloatVolumn)
  PriceTick: float  # 最小价格变动单位
  VolumeMultiple: int  # 合约乘数 (对期货以外的品种, 默认值为1)
  MainContract: int  # 主力合约标记, 1、2、3分别表示第一主力合约、第二主力合约、第三主力合约
  LastVolume: int  # 昨日持仓量
  InstrumentStatus: int  # 合约挂牌状态 (<=0:正常交易, -1:复牌, >=1:停牌天数)

class StockTradingData(TypedDict):
  """ 股票交易数据 """
  time: int  # 时间,ms
  open: float  # 开盘价
  high: float  # 最高价
  low: float  # 最低价
  close: float  # 收盘价
  volume: float  # 成交量
  amount: float  # 成交额
  preClose: float  # 前收盘价
