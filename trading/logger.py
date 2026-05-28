import os
from datetime import datetime

from configs import LOGGER_PATH

from utils.logger import BaseLogger

# 已聚合到 day_board 战报的高频交易日志源 — 从 lark_sink 过滤，避免飞书刷屏。
# 这些信息在 day_board 卡片里已经按状态分类+实时更新，无需再单独成卡。
_LARK_SKIP_KEYS = {
  ('watcher', 'on_stock_order'),
  ('watcher', 'on_order_error'),
  ('watcher', 'on_cancel_error'),
  ('main', 'execute_trade'),
}


def lark_sink(message):
  from configs import LARK_APP_ID
  if not LARK_APP_ID:
    return
  record = message.record
  if (record['module'], record['function']) in _LARK_SKIP_KEYS:
    return
  from .lark.sender import LarkMsgLevel, lark_sender
  lark_sender.send_notification_card(
    level=LarkMsgLevel.Danger,
    title=f"{record['level'].icon} {record['level'].name} {record['function']}@{record['module']}:{record['line']}",
    content=record['message'],
  )

trading_logger = BaseLogger()

# 设置文件日志格式
trading_logger.real_logger.add(
  sink=os.path.join(LOGGER_PATH, f"qmt-{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), # 日志文件输出路径
  format=trading_logger.log_format,
  level="INFO",  # 日志级别
  rotation="00:00",  # 文件分片
  encoding='utf-8',
  enqueue=True,
  backtrace=True,
  diagnose=True
)

# 设置上报到飞书卡片
trading_logger.real_logger.add(
  sink=lark_sink,
  format=trading_logger.log_format,
  level="ERROR",
  backtrace=True,
  diagnose=True
)