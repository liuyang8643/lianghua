"""
适应度函数

评估个体（因子权重组合）的适应度，返回多个优化目标：
1. 年化超额收益率（相对基准）
2. 夏普比率
3. Calmar比率（年化收益/最大回撤）

流程：
1. 从共享内存加载TopN数据
2. 使用权重加权求和计算综合得分
3. 选取Top N股票
4. 模拟回测计算收益指标
"""

import time
import numpy as np
from typing import Dict, Tuple
from datetime import datetime

from testback.logger import testback_logger
from testback.account import StockAccountMocker
from core.strategies._weights import RANK_N
from utils.stock.format import format_qmt_date


def evaluate_individual(
    genes: Dict[str, float],
    topn_cache_key: str,
    initial_cash: float = 500_000.0,
    rank_n: int = RANK_N,
    commission_rate: float = 0.0003,
    min_commission: float = 5.0,
    individual_idx: int = 0,
    generation: int = 0
) -> Tuple[float, float, float, Dict]:
    """评估单个个体的适应度

    Args:
        genes: 因子权重字典 {'MACD': 1.0, 'BBI': 0.5, ...}
        topn_cache_key: TopN数据共享内存key
        initial_cash: 初始资金
        rank_n: 选股数量
        commission_rate: 佣金费率
        min_commission: 最小佣金
        individual_idx: 个体索引（用于日志）
        generation: 代数（用于日志）

    Returns:
        (annualized_excess, sharpe_ratio, calmar_ratio, detailed_metrics)
    """
    t_start = time.time()

    # 从共享内存加载TopN数据
    from utils.shared_memory import SharedMemoryCache
    topn_cache = SharedMemoryCache('testback_cache')
    topn_list = topn_cache.get(topn_cache_key)

    # 创建回测账户
    account = StockAccountMocker(
        cash=initial_cash,
        commission=commission_rate,
        min_commission=min_commission
    )

    # 回测统计
    daily_returns = []
    baseline_returns = []  # 基准收益（等权平均）
    total_trades = 0
    trade_dates = []

    # 遍历每个交易日
    for topn in topn_list:
        current_date = topn.base_date
        # 转换为date对象
        if isinstance(current_date, datetime):
            trade_date = current_date.date()
        else:
            trade_date = current_date
        date_str = format_qmt_date(trade_date)
        trade_dates.append(date_str)

        # === 核心：加权求和计算综合得分 ===
        final_scores = _calculate_weighted_scores(topn, genes)

        if not final_scores:
            # 无有效分数，跳过
            daily_returns.append(0.0)
            baseline_returns.append(0.0)
            continue

        # 选取Top N
        sorted_stocks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_n_stocks = [stock for stock, score in sorted_stocks[:rank_n]]

        # 获取当前持仓
        current_positions = account.positions

        # 生成交易动作（差集调仓）
        target_holdings = set(top_n_stocks)
        current_holdings = set(current_positions.keys())

        # 卖出不在目标持仓中的股票
        stocks_to_sell = current_holdings - target_holdings
        for stock in stocks_to_sell:
            pos = current_positions[stock]
            shares = pos['volume']
            price = _get_stock_price(topn, stock, date_str)
            if price and price > 0:
                account.sell_stock(stock, shares, price, trade_date, clear_reason='rebalance')
                total_trades += 1

        # 买入新股票
        stocks_to_buy = target_holdings - current_holdings
        if stocks_to_buy:
            available_cash = account.current_cash
            buy_amount_per_stock = available_cash / len(stocks_to_buy)

            for stock in stocks_to_buy:
                price = _get_stock_price(topn, stock, date_str)

                if price and price > 0:
                    shares = int(buy_amount_per_stock / price / 100) * 100  # 整手
                    if shares >= 100:
                        account.buy_stock(stock, shares, price, trade_date)
                        total_trades += 1

        # 计算当日资产
        assets = account.calc_assets(current_date)
        total_value = assets['total_asset']

        # 计算日收益率
        if len(daily_returns) == 0:
            daily_return = 0.0
        else:
            prev_date_list = sorted(account.daily_assets.keys())
            if len(prev_date_list) >= 2:
                prev_value = account.daily_assets[prev_date_list[-2]]
            else:
                prev_value = initial_cash
            daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0.0

        daily_returns.append(daily_return)

        # 计算基准收益（等权平均所有候选股票）
        baseline_return = _calculate_baseline_return(topn, date_str)
        baseline_returns.append(baseline_return)

    # 计算回测指标
    # 获取最终资产
    if topn_list:
        last_date = topn_list[-1].base_date
        final_assets = account.calc_assets(last_date)
        final_value = final_assets['total_asset']
    else:
        final_value = initial_cash

    metrics = _calculate_metrics(
        daily_returns=daily_returns,
        baseline_returns=baseline_returns,
        initial_cash=initial_cash,
        final_value=final_value,
        total_trades=total_trades,
        trade_dates=trade_dates
    )

    t_end = time.time()

    # 输出日志
    testback_logger.info(
        f"[个体{individual_idx}] "
        f"超额={metrics['annualized_excess']:.4f} "
        f"夏普={metrics['sharpe_ratio']:.2f} "
        f"Calmar={metrics['calmar_ratio']:.2f} | "
        f"年化={metrics['annualized_return']:.4f} "
        f"基准={metrics['baseline_return']:.4f} "
        f"回撤={metrics['max_drawdown']:.4f} | "
        f"交易={total_trades}次 "
        f"耗时={t_end-t_start:.1f}秒"
    )

    return (
        metrics['annualized_excess'],
        metrics['sharpe_ratio'],
        metrics['calmar_ratio'],
        metrics
    )


def _calculate_weighted_scores(topn, weights: Dict[str, float]) -> Dict[str, float]:
    """加权求和计算综合得分

    Args:
        topn: TopN实例
        weights: 因子权重字典

    Returns:
        {stock_code: final_score}
    """
    if not hasattr(topn, 'factor_scores') or not topn.factor_scores:
        return {}

    # 获取所有股票
    all_stocks = set()
    for factor_name, stock_scores in topn.factor_scores.items():
        if stock_scores:
            all_stocks.update(stock_scores.keys())

    if not all_stocks:
        return {}

    # 加权求和
    final_scores = {}
    for stock in all_stocks:
        weighted_sum = 0.0
        valid_factors = 0

        for factor_name, weight in weights.items():
            if factor_name in topn.factor_scores:
                stock_scores = topn.factor_scores[factor_name]
                if stock and stock in stock_scores:
                    factor_result = stock_scores[stock]
                    # FactorResult是一个dict，包含score和err字段
                    score = factor_result['score'] if isinstance(factor_result, dict) else factor_result
                    # 检查score是否为有效数值
                    if score is not None:
                        score_float = float(score)
                        if np.isfinite(score_float):
                            weighted_sum += weight * score_float
                            valid_factors += 1

        # 只保留至少有一个有效因子的股票
        if valid_factors > 0:
            final_scores[stock] = weighted_sum

    return final_scores


def _get_stock_price(topn, stock_code: str, date_str: str) -> float:
    """获取股票价格（从缓存获取）

    Args:
        topn: TopN实例
        stock_code: 股票代码
        date_str: 日期字符串 (格式: YYYYMMDD)

    Returns:
        股票价格，如果获取失败则返回None
    """
    from datetime import datetime
    from core.database import get_market_data_from_cache

    # 将日期字符串转换为datetime对象
    trade_datetime = datetime.strptime(date_str, '%Y%m%d')

    try:
        # 从缓存获取市场数据
        data = get_market_data_from_cache(stock_code, 1, trade_datetime)
        if data is not None and len(data) > 0:
            price = float(data.iloc[-1]['close'])
            if price and price > 0:
                return price
    except Exception as e:
        testback_logger.debug(f"[价格获取] {stock_code} @ {date_str} 失败: {e}")

    return None


def _calculate_baseline_return(topn, date_str: str) -> float:
    """计算基准收益（等权平均）

    Args:
        topn: TopN实例
        date_str: 日期字符串

    Returns:
        基准日收益率
    """
    # 简化实现：返回0（可以改为实际的基准指数收益）
    return 0.0


def _calculate_metrics(
    daily_returns: list,
    baseline_returns: list,
    initial_cash: float,
    final_value: float,
    total_trades: int,
    trade_dates: list
) -> Dict:
    """计算回测指标

    Args:
        daily_returns: 日收益率列表
        baseline_returns: 基准日收益率列表
        initial_cash: 初始资金
        final_value: 最终资产
        total_trades: 总交易次数
        trade_dates: 交易日期列表

    Returns:
        指标字典
    """
    # 总收益率
    total_return = (final_value - initial_cash) / initial_cash if initial_cash > 0 else 0.0

    # 年化收益率
    trading_days = len(daily_returns)
    if trading_days > 0:
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
    else:
        annualized_return = 0.0

    # 基准年化收益
    baseline_total_return = sum(baseline_returns)
    if trading_days > 0:
        baseline_annualized = (1 + baseline_total_return) ** (252 / trading_days) - 1
    else:
        baseline_annualized = 0.0

    # 超额收益
    annualized_excess = annualized_return - baseline_annualized

    # 最大回撤
    max_drawdown = _calculate_max_drawdown(daily_returns, initial_cash)

    # 夏普比率（假设无风险利率=3%）
    risk_free_rate = 0.03
    if trading_days > 0 and len(daily_returns) > 1:
        returns_std = np.std(daily_returns) * np.sqrt(252)
        if returns_std > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / returns_std
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Calmar比率
    if max_drawdown > 0:
        calmar_ratio = annualized_return / max_drawdown
    else:
        calmar_ratio = 0.0

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'baseline_return': baseline_annualized,
        'annualized_excess': annualized_excess,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'total_trades': total_trades,
        'trading_days': trading_days
    }


def _calculate_max_drawdown(daily_returns: list, initial_cash: float) -> float:
    """计算最大回撤

    Args:
        daily_returns: 日收益率列表
        initial_cash: 初始资金

    Returns:
        最大回撤（0-1之间）
    """
    if not daily_returns:
        return 0.0

    # 计算累计净值曲线
    cumulative_values = [initial_cash]
    for ret in daily_returns:
        cumulative_values.append(cumulative_values[-1] * (1 + ret))

    # 计算最大回撤
    max_drawdown = 0.0
    peak = cumulative_values[0]

    for value in cumulative_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown
