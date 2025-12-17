import os

from joblib import Parallel, delayed, parallel_backend

from utils.parallel import batch_run_threads
from utils.shared_memory import SharedMemoryCache
from testback.account import StockAccountMocker

os.environ['LOKY_PICKLER'] = 'pickle'  # 使用更快的pickle

from datetime import date, datetime, timedelta
from testback.logger import testback_logger
from core.strategies import TopN

# 全局缓存
testback_cache = SharedMemoryCache('testback_cache', compress_level=6)

def _wrap_process_worker(weights: dict[str, float], rank_n: int = 20):
  """ 独立进程计算最终收益 - 从共享内存读取数据 """
  try:
    # 延迟导入：每个 worker 只导入自己需要的模块
    from xtquant import xtdata
    xtdata.enable_hello = False
    from core.strategies.sizers.sizer import Sizer
    from core.database import get_market_data

    # 创建缓存实例并从共享内存读取
    topn_list = testback_cache.get('topn_data')

    # 检查是否成功写入
    if not topn_list:
      testback_logger.error("请先调用 testback_cache.put('topn_data', data) 写入数据")
      exit(1)

    if topn_list is None:
      testback_logger.error("无法从共享内存读取数据")
      return None

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
          data = get_market_data(stock, 1, trade_datetime)
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

  # 最小配置测试
  all_stocks = allow_buy_stock_code_list()[:5]  # 只用前5只股票
  # 使用最近3个交易日（确保有足够历史数据）
  from datetime import timedelta

  today = date.today()
  start_date = today - timedelta(days=7)  # 往前推7天，确保能找到3个交易日
  back_dates = get_trading_date_span(start_date, today)[-3:]  # 取最近3个交易日

  TASK_COUNT = 2  # 只运行2个任务
  worker_count = min(os.cpu_count() or 4, TASK_COUNT)

  testback_logger.info(f'=' * 60)
  testback_logger.info(f'最小配置测试')
  testback_logger.info(f'股票数量: {len(all_stocks)}')
  testback_logger.info(f'回测日期: {[d.strftime("%Y-%m-%d") for d in back_dates]}')
  testback_logger.info(f'任务数量: {TASK_COUNT}')
  testback_logger.info(f'=' * 60)

  # 多线程获取 TopN 实例
  topNs = batch_run_threads(
    func=TopN,
    args_list=[[all_stocks, datetime.combine(d, datetime.min.time())] for d in back_dates],  # 转换为datetime
    max_workers=64,  # 最大并发线程数
  )

  # 创建共享内存缓存并存入数据
  testback_logger.info("topNs 获取完成，正在写入共享内存...")
  testback_cache.put('topn_data', topNs)

  testback_logger.info(f"共享内存写入完成")

  testback_logger.info(f"开始回测：{TASK_COUNT}个任务，{worker_count}个进程，共{len(all_stocks)}只股票")
  with parallel_backend('loky', n_jobs=worker_count, inner_max_num_threads=1):
    results = Parallel(
      n_jobs=worker_count,
      prefer='processes',  # 明确指定使用进程而不是线程
      batch_size=1,  # 每次发送1个任务，减少序列化开销
    )(
      delayed(_wrap_process_worker)(
        {  # FIXME 模拟每个任务使用不同权重
          'MACD': random.random(),
          'BBI': random.random(),
          'CCI': random.random(),
        },
        rank_n=20,  # Top 20 选股
      )
      for idx in range(TASK_COUNT)
    )

  testback_logger.info(f"回测执行完成")

  # 输出回测结果
  valid_results = [r for r in results if r is not None]
  if valid_results:
    testback_logger.info(f"\n{'=' * 60}")
    testback_logger.info(f"回测结果汇总 ({len(valid_results)}/{TASK_COUNT} 个任务成功)")
    testback_logger.info(f"{'=' * 60}")

    # 按收益率排序
    sorted_results = sorted(valid_results, key=lambda x: x['total_return'], reverse=True)

    # 输出 Top 5 最佳策略
    testback_logger.info(f"\nTop 5 最佳策略:")
    for i, result in enumerate(sorted_results[:5], 1):
      testback_logger.info(
        f"\n#{i} 收益率: {result['total_return']:.2f}%"
      )
      testback_logger.info(f"  初始资金: {result['init_cash']:,.2f}")
      testback_logger.info(f"  最终资产: {result['final_total_asset']:,.2f}")
      testback_logger.info(f"  现金: {result['final_cash']:,.2f}")
      testback_logger.info(f"  市值: {result['final_market_value']:,.2f}")
      testback_logger.info(f"  已平仓: {result['cleared_positions_count']} 笔")
      testback_logger.info(f"  持仓中: {result['current_positions_count']} 只")
      testback_logger.info(f"  因子权重: {result['weights']}")

    # 统计信息
    returns = [r['total_return'] for r in valid_results]
    testback_logger.info(f"\n统计信息:")
    testback_logger.info(f"  平均收益率: {sum(returns) / len(returns):.2f}%")
    testback_logger.info(f"  最大收益率: {max(returns):.2f}%")
    testback_logger.info(f"  最小收益率: {min(returns):.2f}%")
    testback_logger.info(f"  正收益策略: {len([r for r in returns if r > 0])} 个")
    testback_logger.info(f"  负收益策略: {len([r for r in returns if r < 0])} 个")
  else:
    testback_logger.error("所有回测任务均失败！")
