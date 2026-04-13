import ctypes
import os
import threading
from contextlib import contextmanager
from typing import Iterator

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

_thread_state = threading.local()


def _set_thread_execution_state(flags: int) -> None:
  result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
  if result == 0:
    raise ctypes.WinError()


@contextmanager
def keep_windows_awake() -> Iterator[bool]:
  """
  在上下文期间阻止 Windows 因空闲而自动休眠。
  不覆盖诸如“合上盖子时睡眠”之类的电源策略。
  """
  if os.name != 'nt':
    yield False
    return

  depth = getattr(_thread_state, 'depth', 0)
  activated = depth > 0

  if depth == 0:
    try:
      _set_thread_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
      activated = True
    except OSError:
      yield False
      return

  _thread_state.depth = depth + 1
  try:
    yield activated
  finally:
    next_depth = _thread_state.depth - 1
    _thread_state.depth = next_depth
    if next_depth == 0:
      try:
        _set_thread_execution_state(ES_CONTINUOUS)
      except OSError:
        pass
