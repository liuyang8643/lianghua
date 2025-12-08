import os

from joblib import Parallel, delayed, parallel_backend

from utils.parallel import batch_run_threads

os.environ['LOKY_PICKLER'] = 'pickle'  # 使用更快的pickle

from datetime import timedelta
from testback.logger import testback_logger
from core.strategies import TopN
import pickle
from multiprocessing import shared_memory

def _wrap_process_worker(shm_name: str, data_size: int, weights: dict[str, float]):
  """ 独立进程计算最终收益 - 从共享内存读取数据 """
  try:
    # 延迟导入：每个 worker 只导入自己需要的模块
    from xtquant import xtdata
    xtdata.enable_hello = False

    # 连接到共享内存
    shm = shared_memory.SharedMemory(name=shm_name)

    try:
      # 从共享内存反序列化数据
      s_list = pickle.loads(shm.buf[:data_size])

      for i in s_list:
        # testback_logger.info(f"回测日期: {i.base_date.strftime('%Y-%m-%d')}")
        for f in i.factors:
          factor_name = f.__class__.__name__
          # testback_logger.info(f"  因子: {factor_name} 权重: {weights.get(factor_name, 0.0):.4f}")
          # TODO 使用 weights 计算最终收益，此处获取的是每日的因子计算结果
    finally:
      # 关闭共享内存连接（不删除，因为其他进程还在使用）
      shm.close()

  except Exception as e:
    testback_logger.error(f"回测时出错: {e}")
    return None

if __name__ == "__main__":
  import random
  from core.database import allow_buy_stock_code_list
  from utils.stock.time import get_latest_trading_time

  all_stocks = allow_buy_stock_code_list()
  back_dates = []
  base_date = get_latest_trading_time()

  TASK_COUNT = 640  # FIXME 模拟 GA 迭代
  worker_count = min(os.cpu_count() or 4, TASK_COUNT)

  for _ in range(10):  # FIXME 模拟多日期回测
    base_date = get_latest_trading_time(base_date - timedelta(days=1))
    back_dates.append(base_date)

  testback_logger.debug(f'回测日期列表: {[d.strftime("%Y-%m-%d") for d in back_dates]}')

  # 多线程获取 TopN 实例
  topNs = batch_run_threads(
    func=TopN,
    args_list=[[all_stocks, d] for d in back_dates],
    max_workers=64,  # 最大并发线程数
  )

  # 序列化 topNs 到共享内存
  testback_logger.info("topNs 获取完成，正在序列化到共享内存...")
  serialized_data = pickle.dumps(topNs)
  data_size = len(serialized_data)

  # 创建共享内存
  shm = shared_memory.SharedMemory(create=True, size=data_size)
  testback_logger.info(f"创建共享内存: {shm.name}, 大小: {data_size / 1024 / 1024:.2f} MB")

  try:
    # 将序列化数据写入共享内存
    shm.buf[:data_size] = serialized_data

    testback_logger.info(f"开始回测：{TASK_COUNT}个任务，{worker_count}个进程，共{len(all_stocks)}只股票")

    with parallel_backend('loky', n_jobs=worker_count, inner_max_num_threads=1):
      results = Parallel(
        n_jobs=worker_count,
        prefer='processes',  # 明确指定使用进程而不是线程
        batch_size=1,  # 每次发送1个任务，减少序列化开销
      )(
        delayed(_wrap_process_worker)(
          shm.name,  # 传递共享内存名称
          data_size,  # 传递数据大小
          {  # FIXME 模拟每个任务使用不同权重
            'MACD': random.random(),
            'BBI': random.random(),
            'CCI': random.random(),
          },
        )
        for idx in range(TASK_COUNT)
      )

    testback_logger.info(f"回测执行完成")

  finally:
    # 清理共享内存
    shm.close()
    shm.unlink()
