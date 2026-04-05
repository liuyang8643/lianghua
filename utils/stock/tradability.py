from datetime import date, datetime
from typing import Literal, Optional, TypedDict

from core.database import StockDetail, StockTradingData, get_market_trade_bar, get_stock_detail
from utils.stock.info import is_bse_stock, is_cyb_stock, is_kcb_stock, is_st_stock, is_stock_trading
from utils.stock.time import get_target_forward_day

TradeSide = Literal['buy', 'sell']
LimitRegimeName = Literal['main_board', 'cyb', 'kcb', 'bse', 'st', 'unlimited']
TradabilityReason = Literal['ok', 'suspended', 'limit_up_locked', 'limit_down_locked', 'missing_trade_bar']


class LimitRegime(TypedDict):
  name: LimitRegimeName
  ratio: Optional[float]
  is_st: bool
  has_price_limit: bool
  is_limit_exempt: bool


class OrderabilityResult(TypedDict):
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


def _to_trade_date(trade_date: datetime | date) -> date:
  return trade_date.date() if isinstance(trade_date, datetime) else trade_date


def _round_limit_price(price: float, ratio: float, is_up: bool) -> float:
  raw_price = price * (1 + ratio if is_up else 1 - ratio)
  return round(raw_price + (1e-8 if is_up else -1e-8), 2)


def _parse_list_date(detail: Optional[StockDetail]) -> Optional[date]:
  if not detail:
    return None

  raw = detail.get('OpenDate') or detail.get('CreateDate')
  if not raw or raw in ('0', '00000000'):
    return None

  try:
    return datetime.strptime(raw, '%Y%m%d').date()
  except ValueError:
    return None


def _is_st_from_detail(detail: Optional[StockDetail]) -> bool:
  return bool(detail and 'ST' in detail.get('InstrumentName', ''))


def _is_limit_exempt_window(stock_code: str, trade_date: date, detail: Optional[StockDetail]) -> bool:
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
    detail: Optional[StockDetail] = None,
) -> LimitRegime:
  trade_day = _to_trade_date(trade_date)
  detail = detail or get_stock_detail(stock_code)
  is_st = _is_st_from_detail(detail) if detail is not None else is_st_stock(stock_code)

  if _is_limit_exempt_window(stock_code, trade_day, detail):
    return {
      'name': 'unlimited',
      'ratio': None,
      'is_st': is_st,
      'has_price_limit': False,
      'is_limit_exempt': True,
    }

  if is_st:
    return {
      'name': 'st',
      'ratio': 0.05,
      'is_st': True,
      'has_price_limit': True,
      'is_limit_exempt': False,
    }

  if is_bse_stock(stock_code):
    return {
      'name': 'bse',
      'ratio': 0.30,
      'is_st': False,
      'has_price_limit': True,
      'is_limit_exempt': False,
    }

  if is_cyb_stock(stock_code):
    return {
      'name': 'cyb',
      'ratio': 0.20,
      'is_st': False,
      'has_price_limit': True,
      'is_limit_exempt': False,
    }

  if is_kcb_stock(stock_code):
    return {
      'name': 'kcb',
      'ratio': 0.20,
      'is_st': False,
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
    bar: StockTradingData,
    detail: Optional[StockDetail] = None,
) -> tuple[Optional[float], Optional[float], LimitRegime]:
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


def evaluate_orderability(
    side: TradeSide,
    stock_code: str,
    trade_date: datetime | date,
    *,
    detail: Optional[StockDetail] = None,
    bar: Optional[StockTradingData] = None,
    dividend_type: str = 'none',
) -> OrderabilityResult:
  trade_day = _to_trade_date(trade_date)
  detail = detail or get_stock_detail(stock_code)

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

  trade_bar = bar if bar is not None else get_market_trade_bar(stock_code, trade_day, dividend_type=dividend_type)
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
