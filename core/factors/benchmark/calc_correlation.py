import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr

from core import FactorCtx, core_logger, init_full_data, init_stock_detail_cache
from core.database import allow_buy_stock_code_list, get_full_market_data, get_stock_detail
from utils.stock.time import get_trading_date_span

@dataclass
class StockFactorScore:
  """单只股票在某日的因子得分"""
  stock_code: str
  stock_name: str
  factor_score: float
  close_price: float
  future_close_prices: dict[int, Optional[float]]  # {m_days: close_price}
  return_rates: dict[int, Optional[float]]  # {m_days: return_rate}

@dataclass
class DailyCorrelation:
  """单日的相关性结果"""
  trade_date: date
  m_days: int
  correlation: Optional[float]
  rank_correlation: Optional[float]
  p_value: Optional[float]
  valid_stock_count: int
  stock_scores: list[StockFactorScore] = None  # 可选：默认不保存以节省内存

@dataclass
class PeriodStatistics:
  """单个持有期的统计结果"""
  m_days: int
  daily_correlations: list[DailyCorrelation]
  avg_correlation: float
  median_correlation: float
  avg_rank_correlation: float
  positive_days: int
  negative_days: int
  valid_days: int

@dataclass
class FactorCorrelationReport:
  """因子相关性报告"""
  factor_name: str
  start_date: date
  end_date: date
  m_days_list: list[int]
  total_stocks: int
  period_statistics: list[PeriodStatistics]

def _calculate_daily_correlation(
    trade_date: date,
    stock_codes: list[str],
    factor_cls,
    m_days_list: list[int],
    trading_dates: list[date],
    save_stock_scores: bool = False  # 新增：是否保存详细 stock_scores
) -> dict[int, DailyCorrelation]:
  """计算单日的因子得分与多个T+M日收益率的相关性"""
  try:
    from xtquant import xtdata
    xtdata.enable_hello = False

    # 计算所有T+M日
    current_idx = trading_dates.index(trade_date)
    future_dates = {m: trading_dates[current_idx + m]
                    for m in m_days_list
                    if current_idx + m < len(trading_dates)}

    if not future_dates:
      return {m: DailyCorrelation(trade_date, m, None, None, None, 0, None)
              for m in m_days_list}

    factor = factor_cls()
    stock_scores = []

    # 计算每只股票的因子得分和收益率
    for stock_code in stock_codes:
      try:
        detail = get_stock_detail(stock_code)
        if not detail:
          continue

        full_data = get_full_market_data(stock_code, '1d')
        if full_data is None or full_data.empty:
          continue

        # 构建日期到收盘价的映射（优化：使用向量化操作替代 iterrows）
        timestamps = full_data['time'].values
        closes = full_data['close'].values
        dates = [datetime.fromtimestamp(ts / 1000).date() for ts in timestamps]
        time_to_close = dict(zip(dates, closes))

        if trade_date not in time_to_close:
          continue

        # 计算因子得分
        ctx = FactorCtx(stock_code, datetime.combine(trade_date, datetime.max.time()))
        result = factor.calc(ctx)

        if result['score'] is None or np.isnan(result['score']):
          continue

        # 计算所有持有期的收益率
        close_price = time_to_close[trade_date]
        future_close_prices = {}
        return_rates = {}

        for m, future_date in future_dates.items():
          if future_date in time_to_close:
            future_close = time_to_close[future_date]
            future_close_prices[m] = future_close
            return_rates[m] = (future_close - close_price) / close_price
          else:
            future_close_prices[m] = None
            return_rates[m] = None

        if any(r is not None for r in return_rates.values()):
          stock_scores.append(StockFactorScore(
            stock_code=stock_code,
            stock_name=detail.get('InstrumentName', stock_code),
            factor_score=result['score'],
            close_price=close_price,
            future_close_prices=future_close_prices,
            return_rates=return_rates
          ))
      except:
        continue

    # 为每个持有期计算相关性
    results = {}
    for m in m_days_list:
      valid_scores = [s for s in stock_scores if s.return_rates.get(m) is not None]

      if len(valid_scores) < 10:
        # 不保存 stock_scores 以节省内存
        results[m] = DailyCorrelation(
          trade_date, m, None, None, None, len(valid_scores), 
          stock_scores if save_stock_scores else None
        )
        continue

      factor_scores = np.array([s.factor_score for s in valid_scores])
      return_rates_array = np.array([s.return_rates[m] for s in valid_scores])

      correlation, p_value = pearsonr(factor_scores, return_rates_array)
      rank_correlation, _ = spearmanr(factor_scores, return_rates_array)

      results[m] = DailyCorrelation(
        trade_date, m, correlation, rank_correlation, p_value, len(valid_scores),
        stock_scores if save_stock_scores else None  # 默认不保存以节省内存
      )

    return results

  except Exception as e:
    core_logger.error(f"计算日期 {trade_date} 的相关性时出错: {e}")
    return {m: DailyCorrelation(trade_date, m, None, None, None, 0, None)
            for m in m_days_list}

def calculate_factor_correlation(
    factor_cls,
    start_date: date,
    end_date: date,
    m_days: int | list[int] = 5,
    stock_codes: list[str] = None,
    save_stock_scores: bool = False  # 新增：是否保存详细的 stock_scores（默认否以节省内存）
) -> FactorCorrelationReport:
  """计算因子得分与T+M日收益率的相关性（一次性计算所有持有期）
  
  注意：该函数会在内部使用多进程并行计算，共享内存缓存已自动启用。
  如果在外部多进程环境中调用，请确保主进程已调用 init_full_data()
  """
  
  stock_codes = stock_codes or allow_buy_stock_code_list()
  m_days_list = [m_days] if isinstance(m_days, int) else sorted(m_days)

  # 预加载股票详情到共享内存缓存
  init_stock_detail_cache(stock_codes)
  # 初始化并预加载数据到共享内存
  init_full_data(stock_codes, '1d')

  core_logger.info(f"开始计算 {factor_cls.__name__} 相关性: {start_date} 至 {end_date}")
  core_logger.info(f"股票数: {len(stock_codes)}, 持有期: {m_days_list}")

  date_list = get_trading_date_span(start_date, end_date)
  worker_count = min(os.cpu_count() or 4, len(date_list))

  # 并行计算所有日期和持有期
  daily_results_list = Parallel(n_jobs=worker_count, backend='loky')(
    delayed(_calculate_daily_correlation)(d, stock_codes, factor_cls, m_days_list, date_list, save_stock_scores)
    for d in date_list
  )

  # 重组结果
  period_daily_results = {m: [] for m in m_days_list}
  for daily_dict in daily_results_list:
    for m, daily_corr in daily_dict.items():
      period_daily_results[m].append(daily_corr)

  # 计算统计指标
  period_statistics = []
  for m in m_days_list:
    daily_results = period_daily_results[m]
    valid_results = [r for r in daily_results if r.correlation is not None]
    valid_correlations = [r.correlation for r in valid_results]
    valid_rank_correlations = [r.rank_correlation for r in valid_results]

    avg_corr = np.mean(valid_correlations) if valid_correlations else 0.0
    median_corr = np.median(valid_correlations) if valid_correlations else 0.0
    avg_rank_corr = np.mean(valid_rank_correlations) if valid_rank_correlations else 0.0
    pos_days = sum(1 for c in valid_correlations if c > 0)
    neg_days = sum(1 for c in valid_correlations if c < 0)

    core_logger.info(f"T+{m}日: 相关系数={avg_corr:.4f}, 有效天数={len(valid_results)}/{len(date_list)}")

    period_statistics.append(PeriodStatistics(
      m, daily_results, avg_corr, median_corr, avg_rank_corr, pos_days, neg_days, len(valid_results)
    ))

  return FactorCorrelationReport(
    factor_cls.__name__, start_date, end_date, m_days_list, len(stock_codes), period_statistics
  )
