"""Network utilities for timeout and retry."""
import socket
import time
import functools
from core.logger import core_logger


def set_socket_timeout(seconds: float = 30):
  """Set default socket timeout."""
  old = socket.getdefaulttimeout()
  socket.setdefaulttimeout(seconds)
  return old


def restore_socket_timeout(old):
  """Restore previous default socket timeout."""
  socket.setdefaulttimeout(old)


def _is_network_error(err: Exception) -> bool:
  """Check if an exception is a network-related error worth retrying."""
  if isinstance(err, (ConnectionError, TimeoutError, OSError)):
    return True
  try:
    from requests import RequestException
    from urllib3.exceptions import HTTPError
    if isinstance(err, (RequestException, HTTPError)):
      return True
  except ImportError:
    pass
  cls_name = type(err).__name__
  if cls_name in ('RemoteDisconnected', 'ProtocolError', 'NewConnectionError'):
    return True
  return False


def retry_on_network_error(max_retries=1, delay=2.0):
  """Decorator: retry on common network errors, with socket timeout."""
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      last_err = None
      for attempt in range(max_retries + 1):
        try:
          old = set_socket_timeout(30)
          try:
            return func(*args, **kwargs)
          finally:
            restore_socket_timeout(old)
        except Exception as e:
          restore_socket_timeout(socket.getdefaulttimeout())
          if _is_network_error(e) and attempt < max_retries:
            last_err = e
            core_logger.debug(
              f"Retrying {func.__name__} after {e} (attempt {attempt + 1})"
            )
            time.sleep(delay)
          else:
            raise
      raise last_err

    return wrapper
  return decorator
