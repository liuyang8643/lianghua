"""
适应度评估函数（独立模块，用于多进程）

在 Windows 上，worker 函数必须定义在可导入的模块中，而不是主模块中。
"""

import time
import numpy as np
from datetime import datetime
from typing import Dict, Tuple

from testback.logger import testback_logger
from testback.account import StockAccountMocker
from core.strategies._weights import RANK_N
from utils.shared_memory import SharedMemoryCache

# 全局缓存（与 ga_main.py 共享）
ga_cache = SharedMemoryCache('testback_cache', compress_level=6)


def evaluate_individual_worker(
    genes: Dict[str, float],
    topn_cache_key: str,
    initial_cash: float = 500_000.0,
    rank_n: int = RANK_N,
    commission_rate: float = 0.0003,
    min_commission: float = 5.0,
    individual_idx: int = 0,
    generation: int = 0
) -> Tuple[float, float, float, Dict]:
    """独立进程评估个体 - 从共享内存读取数据（完全对齐 main.py）"""
    t_start = time.time()

    try:
        # 延迟导入：每个 worker 只导入自己需要的模块
        from xtquant import xtdata
        xtdata.enable_hello = False
        from core.strategies.sizers.sizer import Sizer
        from core import get_market_data_from_cache
        from utils.stock.info import get_baseline_data

        # 从共享内存读取数据
        topn_list = ga_cache.get(topn_cache_key)
        if not topn_list:
            testback_logger.error(f"[个体{individual_idx}] 无法从共享内存读取数据")
            return (-999.0, -999.0, -999.0, {})

        # 创建账户模拟器
        account = StockAccountMocker(cash=initial_cash, commission=commission_rate, min_commission=min_commission)

        # 记录上一日持仓
        prev_holdings = set()

        # 回测统计
        daily_returns = []
        baseline_returns = []

        # 遍历每个交易日
        for topn in topn_list:
            trade_date = topn.base_date

            # 1. 获取当日 Top N 股票（使用传入的权重）
            top_stocks = topn.get_ordered_stocks(
                n=rank_n,
                weights=genes,
                temperatures={k: 1.0 for k in genes.keys()},
                norm_method='rank'
            )

            if not top_stocks:
                daily_returns.append(0.0)
                baseline_returns.append(0.0)
                continue

            # 2. 获取股票价格
            prices = {}
            for stock in top_stocks:
                try:
                    trade_datetime = datetime.combine(trade_date, datetime.min.time())
                    data = get_market_data_from_cache(stock, 1, trade_datetime)
                    if data is not None and len(data) > 0:
                        prices[stock] = float(data.iloc[-1]['close'])
                except Exception as e:
                    testback_logger.debug(f"{stock} 价格获取失败: {e}")

            if not prices:
                daily_returns.append(0.0)
                baseline_returns.append(0.0)
                continue

            # 3. 计算仓位分配
            target_holdings = set(top_stocks)
            allocations = Sizer.allocate(stocks=top_stocks, total_capital=account.current_cash, prices=prices)

            # 4. 卖出不在目标持仓中的股票
            stocks_to_sell = prev_holdings - target_holdings
            for stock in stocks_to_sell:
                pos = account.get_position(stock)
                if pos and pos['volume'] > 0:
                    try:
                        sell_price = prices.get(stock) or pos['avg_price']
                        account.sell_stock(code=stock, volume=pos['volume'], price=sell_price, sell_date=trade_date, clear_reason='调仓')
                    except Exception as e:
                        testback_logger.debug(f"卖出 {stock} 失败: {e}")

            # 5. 买入新股票
            for stock, shares in allocations.items():
                if shares <= 0 or account.get_position(stock):
                    continue  # 跳过无效分配或已持仓

                try:
                    account.buy_stock(code=stock, volume=shares, price=prices[stock], buy_date=trade_date)
                except Exception as e:
                    testback_logger.debug(f"买入 {stock} 失败: {e}")

            # 6. 记录当日资产并计算收益率
            trade_datetime = datetime.combine(trade_date, datetime.min.time())
            assets = account.calc_assets(trade_datetime)

            # 计算日收益率
            prev_date_list = sorted(account.daily_assets.keys())
            if len(prev_date_list) >= 2:
                prev_value = account.daily_assets[prev_date_list[-2]]
                daily_return = (assets['total_asset'] - prev_value) / prev_value if prev_value > 0 else 0.0
            else:
                daily_return = 0.0

            daily_returns.append(daily_return)

            # 计算基准收益
            baseline_data = get_baseline_data(trade_datetime)
            baseline_return = 0.0
            if baseline_data is not None and len(prev_date_list) >= 2:
                prev_baseline = get_baseline_data(prev_date_list[-2])
                if prev_baseline is not None:
                    baseline_return = (float(baseline_data['close']) - float(prev_baseline['close'])) / float(prev_baseline['close'])

            baseline_returns.append(baseline_return)

            # 更新持仓记录
            prev_holdings = target_holdings

        # 7. 计算最终收益
        final_assets = account.calc_assets(datetime.combine(topn_list[-1].base_date, datetime.min.time()))
        total_return = (final_assets['total_asset'] - initial_cash) / initial_cash if initial_cash > 0 else 0.0

        # 计算回测指标
        trading_days = len(daily_returns)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0.0
        baseline_annualized = (1 + sum(baseline_returns)) ** (252 / trading_days) - 1 if trading_days > 0 else 0.0
        annualized_excess = annualized_return - baseline_annualized

        # 最大回撤
        cumulative_values = [initial_cash]
        for ret in daily_returns:
            cumulative_values.append(cumulative_values[-1] * (1 + ret))

        max_drawdown = 0.0
        peak = cumulative_values[0]
        for value in cumulative_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

        # 夏普比率
        if trading_days > 1:
            returns_std = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = (annualized_return - 0.03) / returns_std if returns_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Calmar比率
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'baseline_return': baseline_annualized,
            'annualized_excess': annualized_excess,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(account.cleared_positions),
            'trading_days': trading_days
        }

        # 输出日志
        testback_logger.info(
            f"[个体{individual_idx}] 超额={annualized_excess:+.4f} 夏普={sharpe_ratio:.2f} Calmar={calmar_ratio:.2f} | "
            f"年化={annualized_return:.4f} 基准={baseline_annualized:.4f} 回撤={max_drawdown:.4f} | "
            f"交易={metrics['total_trades']}次 耗时={time.time()-t_start:.1f}秒"
        )

        return (annualized_excess, sharpe_ratio, calmar_ratio, metrics)

    except Exception as e:
        testback_logger.error(f"[个体{individual_idx}] 回测时出错: {e}")
        import traceback
        testback_logger.error(traceback.format_exc())
        return (-999.0, -999.0, -999.0, {})
