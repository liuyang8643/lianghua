import sys
import threading
import re

from utils.logger import BaseLogger

core_logger = BaseLogger()

class BufferedStderrHandler:
  """A stderr proxy that groups traceback lines into a single log block.

  - Buffers lines after detecting a traceback header.
  - Flushes the whole traceback as one ERROR log entry.
  - Falls back to per-line ERROR logs for other stderr messages.
  """
  def __init__(self):
    self._lock = threading.Lock()
    self._buffer: list[str] = []
    self._in_traceback = False
    # Provide common stream attributes for better compatibility
    self.encoding = "utf-8"
    # Pattern like "TypeError: ..." or "module.Error: ..." with no spaces before colon
    self._exc_line_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*: .+")

  def write(self, message: str):
    if not message:
      return 0
    # Normalize newlines so we can handle partial writes
    text = message.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    with self._lock:
      for i, line in enumerate(lines):
        is_last_fragment = (i == len(lines) - 1)
        if not line and is_last_fragment:
          # Trailing newline at end of message, just triggers potential flush
          self._maybe_flush_on_newline()
          continue

        if self._detect_traceback_start(line):
          self._in_traceback = True
          self._buffer.append(line)
          continue

        if self._in_traceback:
          self._buffer.append(line)
          if self._detect_traceback_end(line):
            self._flush_buffer()
          continue

        # Not in traceback mode: log per complete line
        if line:
          core_logger.error(f"STDERR: {line}")

    # Return the number of characters "written" to mimic file-like behavior
    return len(message)

  def flush(self):
    with self._lock:
      if self._buffer:
        self._flush_buffer()

  # Compatibility helpers
  def isatty(self):
    return False

  def writelines(self, lines):
    for line in lines:
      self.write(line)

  # Internal helpers
  def _maybe_flush_on_newline(self):
    # End of line encountered. If we're in traceback mode but the end wasn't
    # detected yet, keep buffering; otherwise no-op.
    if self._in_traceback:
      return

  def _detect_traceback_start(self, line: str) -> bool:
    return line.startswith("Traceback (most recent call last):")

  def _detect_traceback_end(self, line: str) -> bool:
    # Heuristic: end when we encounter an exception summary line like
    # "TypeError: message" (no indentation, no spaces before colon).
    if not line or line.startswith(" ") or line.startswith("\t"):
      return False
    return bool(self._exc_line_pattern.match(line))

  def _flush_buffer(self):
    text = "\n".join(self._buffer)
    self._buffer.clear()
    self._in_traceback = False
    core_logger.error("STDERR:\n" + text)

# Install the buffered handler for stderr
sys.stderr = BufferedStderrHandler()

# Install a global exception hook to avoid Python printing the traceback line-by-line to stderr
# and instead log it once via loguru.

def _global_excepthook(exc_type, exc_value, exc_traceback):
  # Allow Ctrl+C to behave normally
  if issubclass(exc_type, KeyboardInterrupt):
    return sys.__excepthook__(exc_type, exc_value, exc_traceback)
  # Log the full exception once. loguru will include the traceback.
  core_logger.real_logger.opt(exception=(exc_type, exc_value, exc_traceback)).error("Uncaught exception")
  return None

sys.excepthook = _global_excepthook
