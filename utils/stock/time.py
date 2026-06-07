from datetime import date, datetime, time, timedelta
from pathlib import Path

DAY_START = time(0, 0)
MORNING_START = time(9, 30)
MORNING_END = time(11, 30)
AFTERNOON_START = time(13, 0)
AFTERNOON_END = time(15, 0)


_TRADING_CALENDAR_STATE: tuple[frozenset[date], date | None, date | None] | None = None


def _get_trading_calendar_state() -> tuple[frozenset[date], date | None, date | None]:
  global _TRADING_CALENDAR_STATE
  if _TRADING_CALENDAR_STATE is not None:
    return _TRADING_CALENDAR_STATE

  import pyarrow.parquet as pq
  path = Path(__file__).resolve().parents[2] / "data" / "trading_calendar.parquet"
  if not path.exists():
    return frozenset(), None, None
  dates = sorted(pq.read_table(path).column('trade_date').to_pylist())
  if not dates:
    return frozenset(), None, None
  _TRADING_CALENDAR_STATE = (frozenset(dates), dates[0], dates[-1])
  return _TRADING_CALENDAR_STATE


def _is_weekday(target_date: date) -> bool:
  return target_date.weekday() < 5


def _get_last_weekday(base_date: date) -> date:
  input_date = base_date
  while not _is_weekday(input_date):
    input_date -= timedelta(days=1)
  return input_date


def _get_next_weekday(base_date: date) -> date:
  input_date = base_date + timedelta(days=1)
  while not _is_weekday(input_date):
    input_date += timedelta(days=1)
  return input_date


def _weekday_span(start_date: date, end_date: date) -> list[date]:
  res_span: list[date] = []
  current_date = start_date
  while current_date <= end_date:
    if _is_weekday(current_date):
      res_span.append(current_date)
    current_date += timedelta(days=1)
  return res_span


def is_trading_day(target_date: date = None) -> bool:
  if target_date is None:
    target_date = date.today()
  if not _is_weekday(target_date):
    return False
  trading_dates, _, last_known_date = _get_trading_calendar_state()
  if last_known_date is None or target_date > last_known_date:
    return True
  return target_date in trading_dates

def is_current_trading(base_time: datetime = None) -> bool:
  input_time = base_time or datetime.now()
  current_time = input_time.time()
  trading_hours = (
    MORNING_START <= current_time < MORNING_END or
    AFTERNOON_START <= current_time < AFTERNOON_END
  )
  return trading_hours and is_trading_day(input_time.date())

def get_last_trading_day(base_date: date = None) -> date:
  if base_date is None:
    base_date = date.today()
  trading_dates, first_known_date, last_known_date = _get_trading_calendar_state()
  if first_known_date is None or last_known_date is None:
    return _get_last_weekday(base_date)
  if base_date < first_known_date or base_date > last_known_date:
    return _get_last_weekday(base_date)
  current_date = base_date
  while current_date >= first_known_date:
    if current_date in trading_dates:
      return current_date
    current_date -= timedelta(days=1)
  return _get_last_weekday(base_date)

def get_next_trading_day(base_date: date = None) -> date:
  if base_date is None:
    base_date = date.today()
  trading_dates, first_known_date, last_known_date = _get_trading_calendar_state()
  if first_known_date is None or last_known_date is None:
    return _get_next_weekday(base_date)
  current_date = base_date + timedelta(days=1)
  if current_date < first_known_date:
    return first_known_date
  while current_date <= last_known_date:
    if current_date in trading_dates:
      return current_date
    current_date += timedelta(days=1)
  return _get_next_weekday(base_date)

def get_trading_date_span(
    start_date: date,
    end_date: date,
) -> list[date]:
  if not isinstance(start_date, date) or not isinstance(end_date, date):
    raise ValueError('start_date和end_date必须是date类型')
  if start_date > end_date:
    raise ValueError('start_date不能大于end_date')

  trading_dates, first_known_date, last_known_date = _get_trading_calendar_state()
  if first_known_date is None or last_known_date is None:
    return _weekday_span(start_date, end_date)

  result: list[date] = []
  current_date = start_date

  if current_date < first_known_date:
    before_end = min(end_date, first_known_date - timedelta(days=1))
    result.extend(_weekday_span(current_date, before_end))
    current_date = first_known_date

  historical_end = min(end_date, last_known_date)
  while current_date <= historical_end:
    if current_date in trading_dates:
      result.append(current_date)
    current_date += timedelta(days=1)

  if current_date <= end_date:
    result.extend(_weekday_span(current_date, end_date))

  return result
