import sys
import os
from typing import Any
from loguru import logger as real_logger

class BaseLogger:
  def __init__(self, level="DEBUG"):
    self.real_logger = real_logger
    self.catch = self.real_logger.catch
    self.log_format = "<w>{time:YYYY-MM-DD HH:mm:ss}</w> | <level>{level}</level> | {function}@{module}:{line} | <level>{message}</level>"
    effective_level = os.getenv('WBR_LOG_LEVEL', level).upper()
    self.real_logger.remove()
    self.real_logger.add(
      sink=sys.stdout,
      format=self.log_format,
      level=effective_level,
      backtrace=True,
      diagnose=True,
      colorize=False,
    )
    self._file_sink_id: int | None = None

  def add_file_sink(self, filepath: str, level: str = "DEBUG"):
    self._file_sink_id = self.real_logger.add(
      sink=filepath,
      format=self.log_format,
      level=level.upper(),
      backtrace=True,
      diagnose=True,
      colorize=False,
      rotation="50 MB",
      retention="7 days",
    )

  def remove_file_sink(self):
    if self._file_sink_id is not None:
      self.real_logger.remove(self._file_sink_id)
      self._file_sink_id = None

  def debug(self, message: str, *args: Any, **kwargs: Any):
    self.real_logger.opt(depth=1).debug(message, *args, **kwargs)

  def info(self, message: str, *args: Any, **kwargs: Any):
    self.real_logger.opt(depth=1).info(message, *args, **kwargs)

  def success(self, message: str, *args: Any, **kwargs: Any):
    self.real_logger.opt(depth=1).success(message, *args, **kwargs)

  def warning(self, message: str, *args: Any, **kwargs: Any):
    self.real_logger.opt(depth=1).warning(message, *args, **kwargs)

  def error(self, message: str, *args: Any, **kwargs: Any):
    self.real_logger.opt(depth=1).error(message, *args, **kwargs)

  def exception(self, message: str, *args: Any, **kwargs: Any):
    self.real_logger.opt(depth=1).exception(message, *args, **kwargs)
