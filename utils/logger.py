import sys
import os
from loguru import logger

LOG_FORMAT = "<w>{time:YYYY-MM-DD HH:mm:ss}</w> | <level>{level}</level> | {function}@{module}:{line} | <level>{message}</level>"


def configure_utf8_stdio():
  """Keep Windows console output readable for Chinese, currency symbols and emoji."""
  for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
      reconfigure(encoding="utf-8", errors="replace")


def configure_logger(level="DEBUG"):
  """Configure the process console at its entry point; use Loguru directly."""
  configure_utf8_stdio()
  logger.remove()
  logger.add(
    sink=sys.stdout,
    format=LOG_FORMAT,
    level=os.getenv('WBR_LOG_LEVEL', level).upper(),
    backtrace=True,
    diagnose=True,
    colorize=False,
  )
  return logger
