import os
from datetime import datetime

from configs import LOGGER_PATH

from utils.logger import BaseLogger

def lark_sink(message):
  from configs import LARK_APP_ID
  if not LARK_APP_ID:
    return
  from .lark.sender import LarkMsgLevel, lark_sender
  """飞书卡片日志输出"""
  record = message.record

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