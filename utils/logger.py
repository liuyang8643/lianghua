import sys
from typing import Any
from loguru import logger as real_logger

class BaseLogger:
  def __init__(self):
    self.real_logger = real_logger
    self.catch = self.real_logger.catch
    self.log_format = "<w>{time:YYYY-MM-DD HH:mm:ss}</w> | <level>{level}</level> | {function}@{module}:{line} | <level>{message}</level>"
    # 移除默认的 handler
    self.real_logger.remove()
    # 设置控制台日志格式
    self.real_logger.add(
      sink=sys.stdout,
      format=self.log_format,
      level="DEBUG",
      backtrace=True,
      diagnose=True
    )

  def debug(self, message: str, *args: Any, **kwargs: Any):
    """记录 DEBUG 级别的日志"""
    self.real_logger.opt(depth=1).debug(message, *args, **kwargs)

  def info(self, message: str, *args: Any, **kwargs: Any):
    """记录 INFO 级别的日志"""
    self.real_logger.opt(depth=1).info(message, *args, **kwargs)

  def success(self, message: str, *args: Any, **kwargs: Any):
    """记录 SUCCESS 级别的日志"""
    self.real_logger.opt(depth=1).success(message, *args, **kwargs)

  def warning(self, message: str, *args: Any, **kwargs: Any):
    """记录 WARNING 级别的日志"""
    self.real_logger.opt(depth=1).warning(message, *args, **kwargs)

  def error(self, message: str, *args: Any, **kwargs: Any):
    """记录 ERROR 级别的日志"""
    self.real_logger.opt(depth=1).error(message, *args, **kwargs)

  def exception(self, message: str, *args: Any, **kwargs: Any):
    """记录 EXCEPTION 级别的日志"""
    self.real_logger.opt(depth=1).exception(message, *args, **kwargs)
