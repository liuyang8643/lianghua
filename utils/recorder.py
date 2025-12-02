class Recorder:
  def __init__(self):
    self._record_data: dict[str, int] = {}

  def mark(self, data: str):
    """ 记录数据 """
    if data not in self._record_data:
      self._record_data[data] = 0
    else:
      self._record_data[data] += 1
    return self._record_data[data]

  def flush(self):
    """ 消费所有记录的数据 """
    recorded_data = self._record_data.copy()
    self._record_data.clear()
    return recorded_data

recorder = Recorder()

if __name__ == "__main__":
  recorder.mark("test_event")
  recorder.mark("test_event")
  recorder.mark("test_event2")
  recorder.mark("test_event1")
  rc = recorder.flush()
  msg = f"{'\n'.join([f'{k}@{v}' for k, v in rc.items()])}"
  print(msg)
