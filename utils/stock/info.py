from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
from typing import Literal, Optional, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
  from data.db import StockDetail, StockTradingData

LimitRegimeName = Literal['main_board', 'cyb', 'kcb', 'bse', 'st', 'unlimited']


class LimitRegime(TypedDict):
  """涨跌停制度类型定义"""
  name: LimitRegimeName
  ratio: Optional[float]
  is_st: bool
  has_price_limit: bool
  is_limit_exempt: bool


def is_st_stock(stock_code: str) -> bool:
  """
  判断股票名称是否含 ST（*ST / ST 等）

  Args:
    stock_code: 股票代码

  Returns:
    True 表示是 ST 股票
  """
  from data.db import get_stock_detail
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
  - 创业板公司公告《证券代码：300096》(2024-01-03)，明确"股票交易的日涨跌幅限制不变，仍为20%"
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


def min_buy_shares(stock_code: str) -> int:
  """
  市价委托的最小买入数量（股）。

  规则来源 — QMT 实盘报错验证：
  - 科创板 688：「上海科创板市价委托最小买入数量为200」
  - 创业板 300/301：深交所市价委托最小数量同为 200 股
  - 主板 / 北交所：100 股

  Args:
    stock_code: 股票代码

  Returns:
    最小买入股数（100 或 200）
  """
  if is_kcb_stock(stock_code) or is_cyb_stock(stock_code):
    return 200
  return 100


def board_limit_ratio(stock_code: str) -> float:
  """该股所属板块的常规涨跌幅比例，用于「市价买单按涨停价冻结资金」的估算。

  A股券商对市价买单申报时按涨停价冻结资金做可用校验，故下单预算 / 回测资金校验需用
  涨停价 = 前收 × (1 + 本比例)。这里取板块常规上限（不含 ST 的 5%——用更大的板块
  上限做保守冻结，避免低估资金占用导致废单）：
    - 科创板 688 / 创业板 300,301：0.20
    - 北交所 43/83/87/92：0.30
    - 其余（主板 60/00 等）：0.10
  """
  if is_kcb_stock(stock_code) or is_cyb_stock(stock_code):
    return 0.20
  if is_bse_stock(stock_code):
    return 0.30
  return 0.10


def limit_up_price(stock_code: str, prev_close: float) -> float:
  """市价买单的资金冻结单价估算 = 前收 × (1 + 板块涨跌幅)。prev_close<=0 返回 0.0。"""
  if not prev_close or prev_close <= 0:
    return 0.0
  return float(prev_close) * (1.0 + board_limit_ratio(stock_code))


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
  from data.db.stock_name import is_st_at_date

  return is_st_at_date


def _resolve_st_status(
    stock_code: str,
    trade_day: date,
    detail: Optional['StockDetail'],
) -> bool:
  """
  按指定交易日解析 ST 状态。

  历史日期使用 parquet 中的历史 ST 事件记录；实时路径(>=today)用当前详情。
  历史数据缺失时一律返回 False（非 ST），避免把当前 ST 状态回投到历史造成数据泄露。
  """
  if detail is not None and trade_day >= date.today():
    return _is_st_from_detail(detail)

  historical_checker = _get_historical_st_checker()
  return bool(historical_checker(stock_code, trade_day))


def _is_limit_exempt_window(stock_code: str, trade_date: date, detail: Optional['StockDetail']) -> bool:
  """
  判断股票在指定交易日是否处于涨跌停豁免期（按板块和年份区分）。

  历史规则概要：
  - 北交所(43/83/87/92)：2021.11.15 起，仅上市首日不设涨跌幅限制
  - 科创板(688)：2019.07.22 起，上市前 5 个交易日不设限制
  - 创业板(300/301)：2020.08.24 起，上市前 5 个交易日不设限制；此前无豁免
  - 主板(600/601/603/000/001/002/003)：2023.04.10 起，上市前 5 个交易日不设限制
  - 2014 年之前所有板块：首日无固定涨跌停（仅有三档临停），近似为首日豁免
  """
  from utils.stock.time import get_target_forward_day

  list_date = _parse_list_date(detail)
  if list_date is None or trade_date < list_date:
    return False

  # 北交所：仅上市首日不设涨跌幅限制
  if is_bse_stock(stock_code):
    return trade_date == list_date

  # 科创板：2019.07.22 起前 5 个交易日不设限制
  if is_kcb_stock(stock_code):
    return trade_date >= date(2019, 7, 22) and trade_date <= get_target_forward_day(list_date, 4)

  # 创业板：2020.08.24 起前 5 个交易日不设限制
  if is_cyb_stock(stock_code):
    if trade_date >= date(2020, 8, 24):
      return trade_date <= get_target_forward_day(list_date, 4)
    # 2014 年之前首日无固定涨跌停（仅有三档临停）
    if trade_date < date(2014, 1, 1):
      return trade_date == list_date
    return False

  # 主板：2023.04.10 起前 5 个交易日不设限制
  if trade_date >= date(2023, 4, 10):
    return trade_date <= get_target_forward_day(list_date, 4)
  # 2014 年之前首日无固定涨跌停
  if trade_date < date(2014, 1, 1):
    return trade_date == list_date
  return False


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
    from data.db import get_stock_detail

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

  # IPO 首日（非豁免窗口）：2014 年起首日 ±44% 以发行价为基准
  list_date = _parse_list_date(detail)
  if list_date is not None and trade_day == list_date and trade_day >= date(2014, 1, 1):
    return {
      'name': 'ipo_first_day',
      'ratio': 0.44,
      'is_st': is_st,
      'has_price_limit': True,
      'is_limit_exempt': False,
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
    # 创业板涨跌幅限制历史变更：
    # 2020-08-24 前：±10%（与主板相同）
    # 2020-08-24 起：±20%
    ratio = 0.20 if trade_day >= date(2020, 8, 24) else 0.10
    return {
      'name': 'cyb',
      'ratio': ratio,
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
  # IPO 首日 44%：preClose=NaN（上市前无交易），用发行价作基准
  if regime['name'] == 'ipo_first_day' and (pre_close <= 0 or np.isnan(pre_close)):
    issue_p = bar.get('issuePrice')
    if issue_p is not None and not np.isnan(float(issue_p)) and float(issue_p) > 0:
      pre_close = float(issue_p)
  if pre_close <= 0 or np.isnan(pre_close):
    return None, None, regime

  ratio = regime['ratio']
  if ratio is None:
    return None, None, regime

  return (
    _round_limit_price(pre_close, ratio, True),
    _round_limit_price(pre_close, ratio, False),
    regime,
  )


baseline_stock_code = '000852.SH'
