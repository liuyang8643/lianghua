import os

from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm

from core import get_market_data_from_cache, init_stock_detail_cache
from core.database import init_full_data
from utils.parallel import batch_run_threads
from utils.shared_memory import SharedMemoryCache
from testback.account import StockAccountMocker

os.environ['LOKY_PICKLER'] = 'pickle'  # 使用更快的pickle

from datetime import date, datetime, timedelta
from testback.logger import testback_logger
from core.strategies import TopN

# 全局缓存
testback_cache = SharedMemoryCache[list[TopN]]('testback_cache', compress_level=6)

def _wrap_process_worker(weights: dict[str, float], mem_offset: int, mem_count: int):
  """ 独立进程计算最终收益 - 从共享内存读取数据
  
  Args:
    weights: 因子权重
  """

  rank_n = 20
  use_rl_env = False
  try:
    # 延迟导入：每个 worker 只导入自己需要的模块
    from xtquant import xtdata
    xtdata.enable_hello = False
    from core.strategies.sizers.sizer import Sizer
    from core.database import get_market_data

    # 创建缓存实例并从共享内存读取
    topn_list_all = testback_cache.get('topn_data')

    # 检查是否成功写入
    if not topn_list_all or topn_list_all is None or len(topn_list_all) == 0:
      testback_logger.error("请先调用 testback_cache.put('topn_data', data) 写入数据")
      exit(1)

    # 读取指定范围的 TopN 实例
    topn_list = topn_list_all[mem_offset:mem_offset + mem_count]
    testback_logger.debug(f'回测周期: {len(topn_list)}天 ({topn_list[0].base_date} ~ {topn_list[-1].base_date})')

    # === 可选：使用RL环境 ===
    if use_rl_env:
      from testback.rl_env import RLEnv
      from testback.rl_example import TopNAgent, run_episode_with_agent

      # 创建RL环境（使用alpha奖励）
      env = RLEnv(
        topn_list=topn_list,
        init_cash=500_000.0,
        rank_n=rank_n,
        weights=weights,
        commission=2 / 1000,
        min_commission=5.0,
        use_alpha_reward=True,  # 启用超额收益作为奖励
      )

      # 使用TopN策略
      agent = TopNAgent(n=10)

      # 运行回测
      summary = run_episode_with_agent(env, agent)
      summary['weights'] = weights

      return summary

    # === 原有的逐日回测逻辑 ===
    # 创建账户模拟器
    account = StockAccountMocker(
      cash=500_000.0,  # 初始资金50万元
      commission=2 / 1000,  # 千2交易费率
      min_commission=5.0,  # 最小5元交易费
    )

    # 记录上一日持仓
    prev_holdings = set()

    # 遍历每个交易日
    for topn in topn_list:
      trade_date = topn.base_date

      # 1. 获取当日 Top N 股票（使用传入的权重）
      top_stocks = topn.get_ordered_stocks(
        n=rank_n,
        weights=weights,
        temperatures={k: 1.0 for k in weights.keys()},  # 使用默认温度
        norm_method='rank'
      )

      if not top_stocks:
        continue

      # 2. 获取股票价格
      prices = {}
      for stock in top_stocks:
        try:
          # 将 date 转换为 datetime
          trade_datetime = datetime.combine(trade_date, datetime.min.time())
          data = get_market_data_from_cache(stock, 1, trade_datetime)
          if data is not None and len(data) > 0:
            prices[stock] = float(data.iloc[-1]['close'])
        except Exception as e:
          testback_logger.warning(f"{stock} 价格获取失败: {e}")

      if not prices:
        continue

      # 3. 计算仓位分配
      target_holdings = set(top_stocks)
      allocations = Sizer.allocate(
        stocks=top_stocks,
        total_capital=account.current_cash,
        prices=prices
      )

      # 4. 卖出不在目标持仓中的股票
      stocks_to_sell = prev_holdings - target_holdings
      for stock in stocks_to_sell:
        pos = account.get_position(stock)
        if pos and pos['volume'] > 0:
          try:
            sell_price = prices.get(stock)
            if not sell_price:
              # 如果没有价格，使用持仓均价
              sell_price = pos['avg_price']
            account.sell_stock(
              code=stock,
              volume=pos['volume'],
              price=sell_price,
              sell_date=trade_date,
              clear_reason='调仓'
            )
          except Exception as e:
            testback_logger.warning(f"卖出 {stock} 失败: {e}")

      # 5. 买入新股票
      for stock, shares in allocations.items():
        if shares <= 0:
          continue

        # 检查是否已持仓
        pos = account.get_position(stock)
        if pos:
          # 已持仓，跳过（不加仓）
          continue

        try:
          account.buy_stock(
            code=stock,
            volume=shares,
            price=prices[stock],
            buy_date=trade_date
          )
        except Exception as e:
          # 资金不足或其他错误，跳过
          testback_logger.debug(f"买入 {stock} 失败: {e}")

      # 6. 记录当日资产
      trade_datetime = datetime.combine(trade_date, datetime.min.time())
      account.calc_assets(trade_datetime)

      # 更新持仓记录
      prev_holdings = target_holdings

    # 7. 计算最终收益
    final_datetime = datetime.combine(topn_list[-1].base_date, datetime.min.time())
    final_assets = account.calc_assets(final_datetime)
    total_return = (final_assets['total_asset'] - account.init_cash) / account.init_cash * 100

    return {
      'weights': weights,
      'init_cash': account.init_cash,
      'final_cash': final_assets['cash'],
      'final_market_value': final_assets['market_value'],
      'final_total_asset': final_assets['total_asset'],
      'total_return': total_return,
      'cleared_positions_count': len(account.cleared_positions),
      'current_positions_count': len(account.positions),
    }

  except Exception as e:
    testback_logger.error(f"回测时出错: {e}")
    import traceback
    testback_logger.error(traceback.format_exc())
    return None

if __name__ == "__main__":
  import random
  from core.database import allow_buy_stock_code_list
  from utils.stock.time import get_trading_date_span

  ts = datetime.now()
  all_stocks = allow_buy_stock_code_list()

  # 预加载股票详情到共享内存缓存
  init_stock_detail_cache(all_stocks)
  # 初始化并预加载数据到共享内存
  init_full_data(all_stocks, '1d')

  backtest_datetime_list = [
    datetime.combine(d, datetime.min.time())
    for d in get_trading_date_span(date(2025, 11, 1), date(2025, 12, 15))]

  # 多进程获取 TopN 实例
  topn_worker_count = min(os.cpu_count(), len(backtest_datetime_list))
  with parallel_backend('loky', n_jobs=topn_worker_count):
    parallel_pool = Parallel(
      return_as='generator',  # 结果以生成器形式返回
      n_jobs=topn_worker_count,
      prefer='processes',
      batch_size=1,
      verbose=0,
    )
    topNs = list(tqdm(
      parallel_pool(delayed(TopN)(all_stocks, d) for d in backtest_datetime_list),
      total=len(backtest_datetime_list),
      maxinterval=30,
      desc=f"预热 TopN 股票：{len(all_stocks)} 只股票在 {backtest_datetime_list[0]} ~ {backtest_datetime_list[-1]}，{len(backtest_datetime_list)}天"
    ))

  # 创建共享内存缓存并存入数据
  ordered_topNs = sorted(topNs, key=lambda x: x.base_date)
  testback_cache.put('topn_data', ordered_topNs)
  testback_logger.debug(f"已将 {len(ordered_topNs)} 天的 TopN 实例存入共享内存缓存")

  # 任务数量：模拟 GA 一代的评估量（3k，k=24时为72）
  TASK_COUNT = 2  # 对应 population=24 时一代的评估量
  GA_PERIOD_SPAN = 5  # 天，模拟每个任务使用不同时间段的数据
  ga_worker_count = min(os.cpu_count(), TASK_COUNT)
  ga_task_args = [
    [
      {  # FIXME 模拟每个任务使用不同权重
        'MACD': random.random(),
        'BBI': random.random(),
        'CCI': random.random(),
      },
      data_idx,
      GA_PERIOD_SPAN
    ]
    for data_idx in random.sample(range(len(ordered_topNs) - GA_PERIOD_SPAN), TASK_COUNT)
  ]
  testback_logger.debug(f"开始回测：{TASK_COUNT}个任务，{ga_worker_count}个进程，共{len(all_stocks)}只股票")
  with parallel_backend('loky', n_jobs=ga_worker_count):
    results = Parallel(
      prefer='processes',  # 明确指定使用进程而不是线程
      n_jobs=ga_worker_count,
      batch_size=1,
      verbose=0,
    )(
      delayed(_wrap_process_worker)(*args)
      for args in ga_task_args
    )

  # 输出回测结果
  testback_logger.info(f"{'=' * 60}")
  testback_logger.info(f"回测执行完成")
  returns = [r['total_return'] for r in [r for r in results if r is not None]]
  testback_logger.info(f"\n统计信息:")
  testback_logger.info(f"  平均收益率: {sum(returns) / len(returns):.2f}%")
  testback_logger.info(f"  最大收益率: {max(returns):.2f}%")
  testback_logger.info(f"  最小收益率: {min(returns):.2f}%")
  testback_logger.info(f"  正收益策略: {len([r for r in returns if r > 0])} 个")
  testback_logger.info(f"  负收益策略: {len([r for r in returns if r < 0])} 个")
  testback_logger.info(f"{'=' * 60}")
  te = datetime.now()
  testback_logger.info(f"总耗时: {(te - ts).total_seconds():.2f} 秒")
