import time
from typing import Any, Callable, List, Optional, Dict
import concurrent.futures

def batch_run_threads(
    func: Callable[..., Optional[Any]],  # 要执行的函数
    args_list: List[List[Any]],  # 位置参数列表
    kwargs_list: List[Dict[str, Any]] = None,  # 关键字参数列表
    timeout: Optional[int] = None,  # 超时时间，默认为 None
    max_workers: Optional[int] = None,  # 最大线程数，默认为 None（自动选择）
) -> List[Optional[Any]]:
  """
  使用线程池并发执行多个函数任务，并支持传递位置参数和关键字参数。

  :param func: 要执行的函数。
  :param args_list: 包含函数位置参数的列表，列表中的每一项都是一次调用的参数。
  :param kwargs_list: 包含函数关键字参数的列表，列表中的每一项都是一次调用的关键字参数。
  :param timeout: 超时时间，默认为 None。
  :param max_workers: 最大线程数，默认为 None（自动选择，通常为 CPU核心数 * 5）。对于IO密集型任务，可以设置更大的值。
  :return: 所有任务的执行结果（保持与输入顺序一致）。
  """
  if kwargs_list is None:
    kwargs_list = [{}] * len(args_list)  # 如果没有提供 kwargs_list，则默认为空字典

  total_tasks = len(args_list)

  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 提交所有任务，保存 future 和索引的映射
    future_to_index = {
      executor.submit(func, *args, **kwargs): idx
      for idx, (args, kwargs) in enumerate(zip(args_list, kwargs_list))
    }

    # 按索引顺序收集结果
    results = [None] * total_tasks

    for future in concurrent.futures.as_completed(future_to_index):
      idx = future_to_index[future]
      try:
        result = future.result(timeout=timeout)
        results[idx] = result

      except concurrent.futures.TimeoutError:
        future.cancel()  # 取消任务
        results[idx] = None
      except Exception as e:
        results[idx] = None

    return results

def exec_with_retry(
    func: Callable[..., Optional[Any]],
    *args,
    timeout: Optional[int] = None,
    max_retries: int = 3,
    delay: int = 0,
    **kwargs
) -> Optional[Any]:
  """
  Execute a function in a thread pool with retries on failure.

  :param func: Function to execute.
  :param max_retries: Maximum number of attempts (including the initial attempt).
  :param delay: Delay between retries (seconds).
  :param timeout: Task execution timeout.
  :param args: Positional arguments for the function.
  :param kwargs: Keyword arguments for the function.
  :return: Result of the function, or None if all attempts fail.
  """
  with concurrent.futures.ThreadPoolExecutor() as executor:
    for attempt in range(max_retries):
      future = executor.submit(func, *args, **kwargs)
      try:
        return future.result(timeout=timeout)
      except concurrent.futures.TimeoutError:
        future.cancel()
      except Exception as e:
        pass
      if delay > 0:
        time.sleep(delay)
    return None