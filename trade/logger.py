import os
from datetime import datetime

from configs import LOGGER_PATH

from utils.logger import LOG_FORMAT, configure_logger

def lark_sink(message):
  from configs import LARK_APP_ID
  if not LARK_APP_ID:
    return
  record = message.record
  from .lark.sender import LarkMsgLevel, lark_sender
  lark_sender.send_notification_card(
    level=LarkMsgLevel.Danger,
    title=f"{record['level'].icon} {record['level'].name} {record['function']}@{record['module']}:{record['line']}",
    content=record['message'],
  )

trading_logger = configure_logger()
LOG_FILE_PATH = os.path.join(LOGGER_PATH, f"qmt-{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 设置文件日志格式
trading_logger.add(
  sink=LOG_FILE_PATH,
  format=LOG_FORMAT,
  level="INFO",  # 日志级别
  rotation="00:00",  # 文件分片
  encoding='utf-8',
  enqueue=True,
  backtrace=True,
  diagnose=True
)

# 设置上报到飞书卡片
trading_logger.add(
  sink=lark_sink,
  format=LOG_FORMAT,
  level="ERROR",
  backtrace=True,
  diagnose=True
)
