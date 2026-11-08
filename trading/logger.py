import os
from datetime import datetime

from configs import LOGGER_PATH

from utils.logger import configure_logger, LOG_FORMAT

# 报错只写终端和本地日志，不自动转发飞书。
# 终端默认只展示实盘关键状态；需要逐阶段诊断时设置 WBR_LOG_LEVEL=DEBUG。
trading_logger = configure_logger(level="INFO")

# 设置文件日志格式
trading_logger.add(
  sink=os.path.join(LOGGER_PATH, f"qmt-{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), # 日志文件输出路径
  format=LOG_FORMAT,
  level=os.getenv("WBR_FILE_LOG_LEVEL", "INFO").upper(),
  rotation="00:00",  # 文件分片
  encoding='utf-8',
  enqueue=False,
  backtrace=True,
  diagnose=True
)
