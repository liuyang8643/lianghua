import re
from datetime import date, datetime, time, timedelta
from pathlib import Path

DAY_START = time(0, 0)
MORNING_START = time(9, 30)
MORNING_END = time(11, 30)
AFTERNOON_START = time(13, 0)
AFTERNOON_END = time(15, 0)


def _get_trading_calendar_state() -> tuple[frozenset[date], date | None, date | None]:
  import pyarrow.parquet as pq
  path = Path(__file__).resolve().parents[2] / "data" / "trading_calendar.parquet"
  if not path.exists():
    return frozenset(), None, None
  dates = sorted(pq.read_table(path).column('trade_date').to_pylist())
  if not dates:
    return frozenset(), None, None
  return frozenset(dates), dates[0], dates[-1]


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
  if last_known_date is None:
    return True

  # QMT 这里拿到的是已完成的历史交易日；当前/未来日期退化为工作日判断。
  if target_date > last_known_date:
    return True

  return target_date in trading_dates

def is_current_trading(base_time: datetime = None) -> bool:
  input_time = base_time or datetime.now()
  current_time = input_time.time()
  trading_hours = (
      MORNING_START <= current_time < MORNING_END or
      AFTERNOON_START <= current_time < AFTERNOON_END
  )
  return is_trading_day(input_time.date()) and trading_hours

def get_last_trading_day(base_date: date = None) -> date:
  """
  获取最近的交易日
  """
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
  """
  获取下一个交易日
  """
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

def get_target_forward_day(base_date: date, count: int = 1) -> date:
  """
  获取向前(未来)指定天数的交易日日期
  :param base_date: 基准日期
  :param count: 向前的交易日天数
  :return: 目标交易日日期
  """
  if count <= 0:
    return base_date

  result_date = base_date
  remaining_days = count
  while remaining_days > 0:
    # 直接调用优化后的函数，避免重复计算
    result_date = get_next_trading_day(result_date)
    remaining_days -= 1
  return result_date

def get_latest_trading_time(base_time: datetime = None) -> datetime:
  """
  获取最近的交易时间点
  如果当前是交易时间，返回当前时间
  如果当前不在交易时间，返回最近一个交易日的收盘时间
  """
  input_time = base_time or datetime.now()
  input_date = input_time.date()
  input_time_obj = input_time.time()

  if is_trading_day(input_date):
    # 是交易日
    if MORNING_START <= input_time_obj < MORNING_END:
      # 早盘交易时间
      return input_time
    elif MORNING_END <= input_time_obj < AFTERNOON_START:
      # 午休时间
      return datetime.combine(input_date, MORNING_END)
    elif AFTERNOON_START <= input_time_obj < AFTERNOON_END:
      # 下午交易时间
      return input_time
    elif input_time_obj >= AFTERNOON_END or (input_time_obj == DAY_START and input_date < date.today()):
      # 收盘后 或 历史交易日
      return datetime.combine(input_date, AFTERNOON_END)
    else:
      # 开盘前，返回前一个交易日收盘时间
      prev_trading_date = get_last_trading_day(input_date - timedelta(days=1))
      return datetime.combine(prev_trading_date, AFTERNOON_END)

  # 非交易日，寻找前一个交易日
  prev_trading_date = get_last_trading_day(input_date - timedelta(days=1))
  return datetime.combine(prev_trading_date, AFTERNOON_END)

def is_latest_data(data_time: datetime, base_time: datetime, period='1d') -> bool:
  if period == '1d':
    return data_time.date() >= get_latest_trading_time(base_time).date()

  # 允许 20s 误差
  return data_time + timedelta(seconds=20) >= get_target_period_backward(base_time, period)

# 缓存正则表达式编译结果
_PERIOD_REGEX = re.compile(r'(\d+)([md])')

def get_target_period_backward(base_time: datetime, period: str, count=1) -> datetime:
  """
  获取周期开始时间，考虑到交易时间
  :param base_time: 基准时间
  :param period: 周期
  :param count: 周期数
  :return: 周期开始时间，不能解析时返回基准时间
  """
  # 使用预编译的正则表达式
  match = _PERIOD_REGEX.match(period)
  if not match:
    raise Exception(f"不支持的周期格式: {period}")

  period_value = int(match.group(1))
  period_unit = match.group(2)

  if period_unit == 'd':
    # 日周期处理 - 优化版本
    result_time = base_time
    remaining_days = count
    while remaining_days > 0:
      # 直接减一天，然后找交易日
      prev_date = get_last_trading_day(result_time.date() - timedelta(days=1))
      result_time = datetime.combine(prev_date, result_time.time())
      remaining_days -= 1
    return result_time

  elif period_unit == 'm':
    # 分钟周期处理
    result_time = base_time
    minutes_to_subtract = period_value * count

    while minutes_to_subtract > 0:
      # 获取当前时间在交易日已经过去的分钟数
      curr_day_passed = get_trading_pass_minute(result_time)

      # 如果当天已经过去的交易分钟足够减去所需分钟
      if curr_day_passed >= minutes_to_subtract:
        # 计算目标时间
        target_pass = curr_day_passed - minutes_to_subtract
        trading_date=result_time.date()
        if target_pass < 120:  # 上午交易时段的分钟数
          # 目标时间在上午交易时段
          total_minutes = target_pass + 30  # 加上9:30之前的30分钟
          target_hour = 9 + total_minutes // 60
          target_minute = total_minutes % 60
          return datetime.combine(trading_date, time(target_hour, target_minute, result_time.second))
        else:
          # 目标时间在下午交易时段
          afternoon_minutes = target_pass - 120
          target_hour = 13 + afternoon_minutes // 60
          target_minute = afternoon_minutes % 60
          return datetime.combine(trading_date, time(target_hour, target_minute, result_time.second))

      # 减去当天已经交易的分钟数，然后前往前一个交易日
      minutes_to_subtract -= curr_day_passed

      # 使用优化后的函数查找前一个交易日
      prev_date = get_last_trading_day(result_time.date() - timedelta(days=1))

      # 设置为前一个交易日的收盘时间
      result_time = datetime.combine(prev_date, time(15, 0, result_time.second))

    return result_time

  else:
    raise Exception(f"不支持的周期单位: {period_unit}")

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

def get_trading_pass_minute(target_datetime: datetime) -> int:
  """ 返回当前时间在交易日已经过去的分钟K线数 """
  target_time = target_datetime.time()

  if target_time < MORNING_START:
    return 0
  elif MORNING_START <= target_time < MORNING_END:
    # 上午交易
    return (target_time.hour - 9) * 60 + target_time.minute - 30
  elif MORNING_END <= target_time < AFTERNOON_START:
    # 中午休息时间
    return 120  # 2小时 * 60分钟
  elif AFTERNOON_START <= target_time < AFTERNOON_END:
    # 下午交易
    return 120 + (target_time.hour - 13) * 60 + target_time.minute
  else:
    # 盘后
    return 240  # 4小时 * 60分钟
