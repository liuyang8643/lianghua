from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal, Optional, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
  from core.database import StockDetail, StockTradingData

LimitRegimeName = Literal['main_board', 'cyb', 'kcb', 'bse', 'st', 'unlimited']
TradeSide = Literal['buy', 'sell']
TradabilityReason = Literal['ok', 'suspended', 'limit_up_locked', 'limit_down_locked', 'missing_trade_bar']


class LimitRegime(TypedDict):
  """涨跌停制度类型定义"""
  name: LimitRegimeName
  ratio: Optional[float]
  is_st: bool
  has_price_limit: bool
  is_limit_exempt: bool


class OrderabilityResult(TypedDict):
  """可交易性评估结果"""
  allowed: bool
  reason: TradabilityReason
  side: TradeSide
  stock_code: str
  trade_date: str
  regime: LimitRegimeName
  limit_ratio: Optional[float]
  up_limit: Optional[float]
  down_limit: Optional[float]
  source: str


def is_st_stock(stock_code: str) -> bool:
  """
  判断股票名称是否含 ST（*ST / ST 等）

  Args:
    stock_code: 股票代码

  Returns:
    True 表示是 ST 股票
  """
  from core.database import get_stock_detail
  detail = get_stock_detail(stock_code)
  return bool(detail and 'ST' in detail.get('InstrumentName', ''))

def is_cyb_stock(stock_code: str) -> bool:
  """
  判断股票是否为创业板股票（300/301 开头）

  创业板涨跌停限制：上市后前 5 个交易日不设涨跌幅限制，此后为 ±20%；
  创业板风险警示股票仍沿用 20% 涨跌幅限制。

  参考文档：
  - 深交所《创业板改革并试点注册制相关问题答记者问》(2020-08-21)
    https://www.szse.cn/aboutus/trends/news/t20200821_580924.html
  - 创业板公司公告《证券代码：300096》(2024-01-03)，明确“股票交易的日涨跌幅限制不变，仍为20%”
    https://disc.static.szse.cn/download/disc/disk03/finalpage/2024-01-03/70e90ff3-2f2a-49cc-aa90-1e9bfbef063d.PDF

  Args:
    stock_code: 股票代码

  Returns:
    True 表示是创业板股票
  """
  return stock_code.startswith('300') or stock_code.startswith('301')

def is_kcb_stock(stock_code: str) -> bool:
  """
  判断股票是否为科创板股票（688 开头）

  科创板涨跌停限制：上市后前 5 个交易日不设涨跌幅限制，此后为 ±20%；
  科创板公司被实施 ST 后，不进入风险警示板交易，涨跌幅限制仍为 20%。

  参考文档：
  - 上交所《科创板开市初期交易制度的答记者问》(2019-07-19)
    https://www.sse.com.cn/star/media/news/c/c_20190719_4866789.shtml
  - 上交所《就<股票发行上市审核规则>等7项业务规则公开征求意见答记者问》(2024-04-12)
    https://www.sse.com.cn/listing/announcement/notification/c/c_20240412_10764653.shtml

  Args:
    stock_code: 股票代码

  Returns:
    True 表示是科创板股票
  """
  return stock_code.startswith('688')

def is_bse_stock(stock_code: str) -> bool:
  """
  判断股票是否为北交所股票（43/83/87/92 开头）

  北交所股票上市首日不设涨跌幅限制，此后为 ±30%。
  代码段上既要兼容历史的 43/83/87，也要兼容 2024-04-22 起启用的 920 新号段。

  参考文档：
  - 北交所《为上市公司股票启用920代码号段》(2023-11-17)
    https://www.bse.cn/important_news/200019643.html
  - 北交所《关于做好存量上市公司代码切换准备工作的通知》(2024-04-22 起新增股票启用 920)
    https://www.bse.cn/important_news/200024109.html
  - 北交所《关于存量上市公司代码切换试点上线的通知》(2025-04-25)
    https://www.bse.cn/important_news/200025603.html

  Args:
    stock_code: 股票代码

  Returns:
    True 表示是北交所股票
  """
  bare_code = stock_code.split('.')[0]
  return bare_code.startswith(('43', '83', '87', '92'))

def is_b_stock(stock_code: str) -> bool:
  """
  判断股票是否为 B 股（900/200 开头）

  Args:
    stock_code: 股票代码

  Returns:
    True 表示是 B 股
  """
  return stock_code.startswith('900') or stock_code.startswith('200')

def is_convertible_bond(stock_code: str) -> bool:
  """
  判断股票是否为可转债（11/12/13 开头）

  Args:
    stock_code: 股票代码

  Returns:
    True 表示是可转债
  """
  return stock_code.startswith('11') or stock_code.startswith('12') or stock_code.startswith('13')


def _to_trade_date(trade_date: datetime | date) -> date:
  """将 datetime 或 date 转换为 date 对象"""
  return trade_date.date() if isinstance(trade_date, datetime) else trade_date


def _round_limit_price(price: float, ratio: float, is_up: bool) -> float:
  """
  计算涨跌停价格并按规则四舍五入到分

  Args:
    price: 前收盘价
    ratio: 涨跌幅比例（如 0.10 表示 10%）
    is_up: True 表示涨停，False 表示跌停

  Returns:
    四舍五入到分的涨跌停价格
  """
  price_decimal = Decimal(str(price))
  ratio_decimal = Decimal(str(ratio))
  multiplier = Decimal('1') + ratio_decimal if is_up else Decimal('1') - ratio_decimal
  raw_price = price_decimal * multiplier
  return float(raw_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def _parse_list_date(detail: Optional['StockDetail']) -> Optional[date]:
  """
  从股票详情中解析上市日期

  Args:
    detail: 股票详情数据

  Returns:
    上市日期，解析失败返回 None
  """
  if not detail:
    return None

  raw = detail.get('OpenDate') or detail.get('CreateDate')
  if not raw or raw in ('0', '00000000'):
    return None

  try:
    return datetime.strptime(raw, '%Y%m%d').date()
  except ValueError:
    return None


def _is_st_from_detail(detail: Optional['StockDetail']) -> bool:
  """从股票详情判断是否为 ST 股票"""
  return bool(detail and 'ST' in detail.get('InstrumentName', ''))


def _get_historical_st_checker():
  """延迟加载历史 ST 判断函数，避免常规路径引入额外依赖。"""
  from core.database.stock_name import is_st_at_date

  return is_st_at_date


def _resolve_st_status(
    stock_code: str,
    trade_day: date,
    detail: Optional['StockDetail'],
) -> bool:
  """
  按指定交易日解析 ST 状态。

  对于历史日期，优先使用历史名称/ST 事件记录；实时路径退回当前详情。
  """
  if detail is not None and trade_day >= date.today():
    return _is_st_from_detail(detail)

  try:
    historical_checker = _get_historical_st_checker()
    return bool(historical_checker(stock_code, trade_day))
  except Exception:
    if detail is not None:
      return _is_st_from_detail(detail)
    return is_st_stock(stock_code)


def _is_limit_exempt_window(stock_code: str, trade_date: date, detail: Optional['StockDetail']) -> bool:
  """
  判断股票在指定交易日是否处于涨跌停豁免期

  新股上市后有涨跌停豁免期：
  - 主板/创业板/科创板：上市后前 5 个交易日（含上市日）
  - 北交所：仅上市首日不设涨跌幅限制

  Args:
    stock_code: 股票代码
    trade_date: 交易日期
    detail: 股票详情（可选，用于获取上市日期）

  Returns:
    是否在豁免期内

  参考文档：
  - 深交所《主板投资入市手册（十四）：主板股票交易机制（二）》(2023-06-29)
    https://investor.szse.cn/institute/rules/t20230629_601434.html
  - 深交所《创业板改革并试点注册制相关问题答记者问》(2020-08-21)
    https://www.szse.cn/aboutus/trends/news/t20200821_580924.html
  - 上交所《科创板开市初期交易制度的答记者问》(2019-07-19)
    https://www.sse.com.cn/star/media/news/c/c_20190719_4866789.shtml
  - 北交所发行上市文件/风险揭示材料均沿用“上市首日不设涨跌幅限制，其后涨跌幅限制为30%”表述，例如：
    https://www.bse.cn/disclosure/2024/2024-08-26/1724666892_999832.pdf
  """
  from utils.stock.time import get_target_forward_day

  list_date = _parse_list_date(detail)
  if list_date is None or trade_date < list_date:
    return False

  if is_bse_stock(stock_code):
    exempt_days = 0
  else:
    exempt_days = 4

  return trade_date <= get_target_forward_day(list_date, exempt_days)


def resolve_limit_regime(
    stock_code: str,
    trade_date: datetime | date,
    detail: Optional['StockDetail'] = None,
) -> LimitRegime:
  """
  解析股票在指定交易日的涨跌停制度

  涨跌停制度：
  - 主板：±10%
  - 创业板（300/301开头）：±20%
  - 科创板（688开头）：±20%
  - 北交所（43/83/87/92开头）：±30%
  - 主板 ST 股票：±5%
  - 创业板/科创板/北交所 ST 股票：沿用板块限制
  - 新股豁免期：无涨跌停限制

  Args:
    stock_code: 股票代码
    trade_date: 交易日期
    detail: 股票详情（可选，用于判断 ST 和上市日期）

  Returns:
    涨跌停制度信息

  参考文档：
  - 深交所《主板股票的涨跌幅比例是多少？》(2023-03-08)
    https://investor.szse.cn/knowledge/t20230308_599141.html
  - 深交所《关于修改<深圳证券交易所交易规则>的通知》(2021-03-31)，主板风险警示股票涨跌幅限制为 5%
    https://investor.szse.cn/lawrules/rule/allrules/bussiness/t20210331_585336.html
  - 深交所《创业板改革并试点注册制相关问题答记者问》(2020-08-21)
    https://www.szse.cn/aboutus/trends/news/t20200821_580924.html
  - 上交所《科创板开市初期交易制度的答记者问》(2019-07-19)
    https://www.sse.com.cn/star/media/news/c/c_20190719_4866789.shtml
  - 上交所《就<股票发行上市审核规则>等7项业务规则公开征求意见答记者问》(2024-04-12)
    https://www.sse.com.cn/listing/announcement/notification/c/c_20240412_10764653.shtml
  - 北交所《关于做好存量上市公司代码切换准备工作的通知》(2024-04-22 起新增股票启用 920)
    https://www.bse.cn/important_news/200024109.html
  """
  trade_day = _to_trade_date(trade_date)
  if detail is None:
    from core.database import get_stock_detail

    detail = get_stock_detail(stock_code)
  is_st = _resolve_st_status(stock_code, trade_day, detail)

  if _is_limit_exempt_window(stock_code, trade_day, detail):
    return {
      'name': 'unlimited',
      'ratio': None,
      'is_st': is_st,
      'has_price_limit': False,
      'is_limit_exempt': True,
    }

  if is_bse_stock(stock_code):
    return {
      'name': 'bse',
      'ratio': 0.30,
      'is_st': is_st,
      'has_price_limit': True,
      'is_limit_exempt': False,
    }

  if is_cyb_stock(stock_code):
    return {
      'name': 'cyb',
      'ratio': 0.20,
      'is_st': is_st,
      'has_price_limit': True,
      'is_limit_exempt': False,
    }

  if is_kcb_stock(stock_code):
    return {
      'name': 'kcb',
      'ratio': 0.20,
      'is_st': is_st,
      'has_price_limit': True,
      'is_limit_exempt': False,
    }

  if is_st:
    return {
      'name': 'st',
      'ratio': 0.05,
      'is_st': True,
      'has_price_limit': True,
      'is_limit_exempt': False,
    }

  return {
    'name': 'main_board',
    'ratio': 0.10,
    'is_st': False,
    'has_price_limit': True,
    'is_limit_exempt': False,
  }


def get_stock_estimate_up_limit(stock_code: str, current_price: float) -> float:
  """
  估算股票涨停价（基于当前价格的简化版本）

  注意：此函数使用当前价格作为基准，仅用于快速估算。
  实际涨跌停价格应基于前收盘价计算，请使用 resolve_limit_regime 获取准确制度。

  Args:
    stock_code: 股票代码
    current_price: 当前价格

  Returns:
    估算的涨停价
  """
  regime = resolve_limit_regime(stock_code, datetime.now())
  if not regime['has_price_limit'] or regime['ratio'] is None:
    return current_price * 1.5  # 无涨跌停限制时返回一个较大值
  return _round_limit_price(current_price, regime['ratio'], True)

def get_stock_estimate_down_limit(stock_code: str, current_price: float) -> float:
  """
  估算股票跌停价（基于当前价格的简化版本）

  注意：此函数使用当前价格作为基准，仅用于快速估算。
  实际涨跌停价格应基于前收盘价计算，请使用 resolve_limit_regime 获取准确制度。

  Args:
    stock_code: 股票代码
    current_price: 当前价格

  Returns:
    估算的跌停价
  """
  regime = resolve_limit_regime(stock_code, datetime.now())
  if not regime['has_price_limit'] or regime['ratio'] is None:
    return current_price * 0.5  # 无涨跌停限制时返回一个较小值
  return _round_limit_price(current_price, regime['ratio'], False)


def get_limit_band_from_ratio(
    stock_code: str,
    trade_date: datetime | date,
    bar: 'StockTradingData',
    detail: Optional['StockDetail'] = None,
) -> tuple[Optional[float], Optional[float], LimitRegime]:
  """
  基于实际 bar 数据计算涨跌停价格

  Args:
    stock_code: 股票代码
    trade_date: 交易日期
    bar: 交易数据
    detail: 股票详情（可选）

  Returns:
    (涨停价, 跌停价, 涨跌停制度)
  """
  regime = resolve_limit_regime(stock_code, trade_date, detail)
  if not regime['has_price_limit']:
    return None, None, regime

  pre_close = float(bar['preClose'])
  if pre_close <= 0:
    return None, None, regime

  ratio = regime['ratio']
  if ratio is None:
    return None, None, regime

  return (
    _round_limit_price(pre_close, ratio, True),
    _round_limit_price(pre_close, ratio, False),
    regime,
  )


def _get_trade_bar(
    stock_code: str,
    trade_day: date,
    dividend_type: str,
) -> Optional['StockTradingData']:
  """
  获取指定交易日的 bar 数据

  Args:
    stock_code: 股票代码
    trade_day: 交易日期
    dividend_type: 复权类型

  Returns:
    交易数据，获取失败返回 None
  """
  from core.database import get_market_data_batch
  from utils.stock.time import AFTERNOON_END

  trade_data = get_market_data_batch(
    [stock_code],
    1,
    datetime.combine(trade_day, AFTERNOON_END),
    period='1d',
    dividend_type=dividend_type,
    strict_trade_date=True,
  ).get(stock_code)
  if trade_data is None or trade_data.empty:
    return None
  return trade_data.iloc[-1]


def evaluate_orderability(
    side: TradeSide,
    stock_code: str,
    trade_date: datetime | date,
    *,
    detail: Optional['StockDetail'] = None,
    bar: Optional['StockTradingData'] = None,
    dividend_type: str = 'none',
) -> OrderabilityResult:
  """
  评估股票在指定日期是否可买入/卖出

  检查项：
  - 停牌状态
  - 涨停锁定（买入时）
  - 跌停锁定（卖出时）
  - 交易数据可用性

  Args:
    side: 交易方向（'buy' 或 'sell'）
    stock_code: 股票代码
    trade_date: 交易日期
    detail: 股票详情（可选）
    bar: 交易数据（可选）
    dividend_type: 复权类型

  Returns:
    可交易性评估结果
  """
  trade_day = _to_trade_date(trade_date)
  if detail is None:
    from core.database import get_stock_detail

    detail = get_stock_detail(stock_code)

  if detail is not None and trade_day == date.today() and not is_stock_trading(detail):
    regime = resolve_limit_regime(stock_code, trade_day, detail)
    return {
      'allowed': False,
      'reason': 'suspended',
      'side': side,
      'stock_code': stock_code,
      'trade_date': trade_day.isoformat(),
      'regime': regime['name'],
      'limit_ratio': regime['ratio'],
      'up_limit': None,
      'down_limit': None,
      'source': 'detail_status',
    }

  trade_bar = bar if bar is not None else _get_trade_bar(stock_code, trade_day, dividend_type)
  if trade_bar is None:
    regime = resolve_limit_regime(stock_code, trade_day, detail)
    return {
      'allowed': False,
      'reason': 'missing_trade_bar',
      'side': side,
      'stock_code': stock_code,
      'trade_date': trade_day.isoformat(),
      'regime': regime['name'],
      'limit_ratio': regime['ratio'],
      'up_limit': None,
      'down_limit': None,
      'source': 'trade_bar',
    }

  if int(trade_bar.get('suspendFlag', 0)) == 1:
    regime = resolve_limit_regime(stock_code, trade_day, detail)
    up_limit, down_limit, regime = get_limit_band_from_ratio(stock_code, trade_day, trade_bar, detail)
    return {
      'allowed': False,
      'reason': 'suspended',
      'side': side,
      'stock_code': stock_code,
      'trade_date': trade_day.isoformat(),
      'regime': regime['name'],
      'limit_ratio': regime['ratio'],
      'up_limit': up_limit,
      'down_limit': down_limit,
      'source': 'trade_bar',
    }

  up_limit, down_limit, regime = get_limit_band_from_ratio(stock_code, trade_day, trade_bar, detail)
  if regime['has_price_limit'] and up_limit is not None and down_limit is not None:
    day_low = float(trade_bar['low'])
    day_high = float(trade_bar['high'])

    if side == 'buy' and day_low >= up_limit:
      return {
        'allowed': False,
        'reason': 'limit_up_locked',
        'side': side,
        'stock_code': stock_code,
        'trade_date': trade_day.isoformat(),
        'regime': regime['name'],
        'limit_ratio': regime['ratio'],
        'up_limit': up_limit,
        'down_limit': down_limit,
        'source': 'back_adjusted_daily_bar',
      }

    if side == 'sell' and day_high <= down_limit:
      return {
        'allowed': False,
        'reason': 'limit_down_locked',
        'side': side,
        'stock_code': stock_code,
        'trade_date': trade_day.isoformat(),
        'regime': regime['name'],
        'limit_ratio': regime['ratio'],
        'up_limit': up_limit,
        'down_limit': down_limit,
        'source': 'back_adjusted_daily_bar',
      }

  return {
    'allowed': True,
    'reason': 'ok',
    'side': side,
    'stock_code': stock_code,
    'trade_date': trade_day.isoformat(),
    'regime': regime['name'],
    'limit_ratio': regime['ratio'],
    'up_limit': up_limit,
    'down_limit': down_limit,
    'source': 'back_adjusted_daily_bar',
  }


def is_stock_trading(detail: Optional['StockDetail']) -> bool:
  """
  判断股票是否正常交易

  Args:
    detail: 股票详情数据

  Returns:
    True 表示正常交易，False 表示停牌或退市
  """
  if detail is None:
    return False

  return (
      detail['InstrumentStatus'] <= 0  # 未停牌
      # 未退市
      and (detail['ExpireDate'] in ('0', '99999999'))
  )


# baseline_stock_code = '000300.SH'  # 基准股票代码，沪深300
baseline_stock_code = '000852.SH'  # 基准股票代码，中证1000

def get_baseline_data(base_time: datetime = None) -> Optional['StockTradingData']:
  """
  获取基准指数的交易数据

  当前使用中证1000（000852.SH）作为基准指数

  Args:
    base_time: 基准时间，默认为当前时间

  Returns:
    基准指数的交易数据，获取失败返回 None
  """
  from core.database import get_market_data_from_cache

  input_time = base_time or datetime.now()
  data = get_market_data_from_cache(baseline_stock_code, 1, input_time, '1d')
  return data.iloc[-1] if data is not None and data.size > 0 else None
