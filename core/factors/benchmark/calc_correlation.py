import os
import pickle
import sys
import traceback
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from core import FactorCtx, core_logger, init_stock_detail_cache
from core.database import get_market_data_range_from_cache, get_stock_detail, init_market_data_range
from core.factors.helpers import CacheKey, DiskCache
from utils.hash import hash_function_code
from utils.stock.format import format_qmt_datetime
from utils.stock.time import get_target_forward_day, get_target_period_backward, get_trading_date_span
from utils.windows_awake import keep_windows_awake

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
  """单个持有期的统计结果（类型1：同一天但不同股票）"""
  m_days: int
  daily_correlations: list[DailyCorrelation]
  avg_correlation: float  # 加权平均Pearson相关系数
  median_correlation: float  # 中位数Pearson相关系数
  avg_rank_correlation: float  # 加权平均Spearman秩相关系数
  positive_days: int  # 正相关天数
  negative_days: int  # 负相关天数
  valid_days: int  # 有效天数
  total_samples: int  # 总样本量（有效天数，即相关性观察次数）
  positive_count: int  # 正相关样本量（正相关天数）
  negative_count: int  # 负相关样本量（负相关天数）
  total_data_points: int  # 总数据点数（所有有效股票数之和，用于计算相关性的数据点总数）
  positive_data_points: int  # 正相关数据点数（正相关天的有效股票数之和）
  negative_data_points: int  # 负相关数据点数（负相关天的有效股票数之和）
  ic_mean: float  # IC均值（Information Coefficient）
  ic_std: float  # IC标准差
  ir: float  # 信息比率（IR = IC均值 / IC标准差）
  ic_ir: float  # ICIR（衡量因子稳定性）

@dataclass
class StockCorrelation:
  """单只股票的相关性结果（类型2：同一个股票但不同天数）"""
  stock_code: str
  stock_name: str
  m_days: int
  correlation: Optional[float]
  rank_correlation: Optional[float]
  p_value: Optional[float]
  valid_date_count: int  # 有效日期数量

@dataclass
class StockPeriodStatistics:
  """单个持有期的股票相关性统计结果（类型2）"""
  m_days: int
  stock_correlations: list[StockCorrelation]
  avg_correlation: float  # 加权平均Pearson相关系数
  median_correlation: float  # 中位数Pearson相关系数
  avg_rank_correlation: float  # 加权平均Spearman秩相关系数
  positive_stocks: int  # 正相关股票数
  negative_stocks: int  # 负相关股票数
  valid_stocks: int  # 有效股票数
  total_samples: int  # 总样本量（有效股票数，即相关性观察次数）
  positive_count: int  # 正相关样本量（正相关股票数）
  negative_count: int  # 负相关样本量（负相关股票数）
  total_data_points: int  # 总数据点数（所有有效日期数之和，用于计算相关性的数据点总数）
  positive_data_points: int  # 正相关数据点数（正相关股票的有效日期数之和）
  negative_data_points: int  # 负相关数据点数（负相关股票的有效日期数之和）
  ic_mean: float  # IC均值
  ic_std: float  # IC标准差
  ir: float  # 信息比率
  ic_ir: float  # ICIR

@dataclass
class FactorCorrelationReport:
  """因子相关性报告"""
  factor_name: str
  start_date: date
  end_date: date
  m_days_list: list[int]
  total_stocks: int
  period_statistics: list[PeriodStatistics]  # 类型1：同一天但不同股票
  stock_period_statistics: list[StockPeriodStatistics]  # 类型2：同一个股票但不同天数
  show_stock_correlation: bool = False  # 是否显示类型2（股票相关性详情），默认不显示

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
    factor_name = factor_cls.__name__

    # 创建缓存
    func_hash = hash_function_code(factor.calc)
    base_datetime = datetime.combine(trade_date, datetime.max.time())
    cache_key = CacheKey.make_key(
      [f"factor-{factor_name}-{func_hash}", format_qmt_datetime(base_datetime)],
      stocks=stock_codes
    )

    # 直接从磁盘加载（内存映射，操作系统页面缓存）
    cached_factor_stocks = DiskCache.load_pickle(cache_key) or {}

    stock_scores = []

    # 计算每只股票的因子得分和收益率
    for stock_code in stock_codes:
      try:
        detail = get_stock_detail(stock_code)
        if not detail:
          continue

        price_end_date = max(future_dates.values())
        price_data = get_market_data_range_from_cache(
          stock_code,
          datetime.combine(trade_date, datetime.min.time()),
          datetime.combine(price_end_date, datetime.max.time()),
          '1d',
          dividend_type='back',
        )
        if price_data is None or price_data.empty:
          continue

        # 构建日期到收盘价的映射（优化：使用向量化操作替代 iterrows）
        timestamps = price_data['time'].values
        closes = price_data['close'].values
        dates = [datetime.fromtimestamp(ts / 1000).date() for ts in timestamps]
        time_to_close = dict(zip(dates, closes))

        if trade_date not in time_to_close:
          continue

        # 从缓存获取或计算因子得分
        cached_stock_value = cached_factor_stocks.get(stock_code)
        if cached_stock_value is not None:
          # 命中缓存
          result = cached_stock_value
        else:
          # 计算因子得分
          ctx = FactorCtx(stock_code, base_datetime)
          result = factor.calc(ctx)
          # 保存到缓存
          cached_factor_stocks[stock_code] = result

        if result['score'] is None or np.isnan(result['score']):
          core_logger.warning(f"股票 {stock_code} 在日期 {trade_date} 的因子得分无效：{result['err']}")
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
        core_logger.error(f"股票 {stock_code} 在日期 {trade_date} 计算因子时异常：\n{traceback.format_exc()}")
        continue

    # 保存缓存
    DiskCache.save_pickle(cache_key, cached_factor_stocks)

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

def _calculate_stock_correlation(
    stock_code: str,
    trade_dates: list[date],
    factor_cls,
    m_days_list: list[int],
    trading_dates: list[date],
    stock_name: str = None
) -> dict[int, StockCorrelation]:
  """计算单只股票在不同买入日期的相关性（类型2：同一个股票但不同天数）"""
  try:
    from xtquant import xtdata
    xtdata.enable_hello = False

    factor = factor_cls()

    # 获取股票数据
    if not trade_dates:
      return {m: StockCorrelation(stock_code, stock_name or stock_code, m, None, None, None, 0)
              for m in m_days_list}

    range_end_date = get_target_forward_day(trade_dates[-1], max(m_days_list, default=0))
    price_data = get_market_data_range_from_cache(
      stock_code,
      datetime.combine(trade_dates[0], datetime.min.time()),
      datetime.combine(range_end_date, datetime.max.time()),
      '1d',
      dividend_type='back',
    )
    if price_data is None or price_data.empty:
      return {m: StockCorrelation(stock_code, stock_name or stock_code, m, None, None, None, 0)
              for m in m_days_list}

    # 构建日期到收盘价的映射
    timestamps = price_data['time'].values
    closes = price_data['close'].values
    dates = [datetime.fromtimestamp(ts / 1000).date() for ts in timestamps]
    time_to_close = dict(zip(dates, closes))

    # 收集该股票在不同买入日期的因子得分和收益
    stock_scores_by_date = {}  # {trade_date: (factor_score, return_rates)}

    for trade_date in trade_dates:
      if trade_date not in time_to_close:
        continue

      try:
        # 计算因子得分
        base_datetime = datetime.combine(trade_date, datetime.max.time())
        ctx = FactorCtx(stock_code, base_datetime)
        result = factor.calc(ctx)

        if result['score'] is None or np.isnan(result['score']):
          core_logger.warning(f"股票 {stock_code} 在日期 {trade_date} 的因子得分无效：${result['err']}")
          continue

        # 计算所有持有期的收益率
        close_price = time_to_close[trade_date]
        return_rates = {}

        current_idx = trading_dates.index(trade_date)
        for m in m_days_list:
          if current_idx + m < len(trading_dates):
            future_date = trading_dates[current_idx + m]
            if future_date in time_to_close:
              future_close = time_to_close[future_date]
              return_rates[m] = (future_close - close_price) / close_price
            else:
              return_rates[m] = None
          else:
            return_rates[m] = None

        if any(r is not None for r in return_rates.values()):
          stock_scores_by_date[trade_date] = (result['score'], return_rates)
      except Exception as e:
        core_logger.error(f"股票 {stock_code} 在日期 {trade_date} 计算因子时异常：\n{traceback.format_exc()}")
        continue

    if not stock_scores_by_date:
      return {m: StockCorrelation(stock_code, stock_name or stock_code, m, None, None, None, 0)
              for m in m_days_list}

    # 获取股票名称
    if stock_name is None:
      detail = get_stock_detail(stock_code)
      stock_name = detail.get('InstrumentName', stock_code) if detail else stock_code

    # 为每个持有期计算相关性
    results = {}
    for m in m_days_list:
      # 收集该持有期的有效数据
      valid_data = []
      for trade_date, (factor_score, return_rates) in stock_scores_by_date.items():
        if return_rates.get(m) is not None:
          valid_data.append((factor_score, return_rates[m]))

      if len(valid_data) < 10:
        results[m] = StockCorrelation(
          stock_code, stock_name, m, None, None, None, len(valid_data)
        )
        continue

      factor_scores = np.array([d[0] for d in valid_data])
      return_rates_array = np.array([d[1] for d in valid_data])

      correlation, p_value = pearsonr(factor_scores, return_rates_array)
      rank_correlation, _ = spearmanr(factor_scores, return_rates_array)

      results[m] = StockCorrelation(
        stock_code, stock_name, m, correlation, rank_correlation, p_value, len(valid_data)
      )

    return results

  except Exception as e:
    core_logger.error(f"计算股票 {stock_code} 的相关性时出错: {e}")
    return {m: StockCorrelation(stock_code, stock_name or stock_code, m, None, None, None, 0)
            for m in m_days_list}

def _calculate_factor_correlation_impl(
    factor_cls,
    start_date: date,
    end_date: date,
    m_days: int | list[int] = 5,
    stock_codes: list[str] = None,
    save_stock_scores: bool = False,  # 新增：是否保存详细的 stock_scores（默认否以节省内存）
    show_stock_correlation: bool = False  # 是否计算并显示类型2（股票相关性详情），默认不显示
) -> FactorCorrelationReport:
  """计算因子得分与T+M日收益率的相关性（一次性计算所有持有期）
  
  注意：该函数会在内部使用多进程并行计算，共享内存缓存已自动启用。
  如果在外部多进程环境中调用，请确保主进程已按回测窗口预热日线缓存。
  """

  m_days_list = [m_days] if isinstance(m_days, int) else sorted(m_days)
  max_m_days = max(m_days_list, default=0)

  # 预加载股票详情到共享内存缓存
  init_stock_detail_cache(stock_codes)
  # 仅预加载相关性计算所需窗口，避免整段历史常驻内存
  factor_history_days = factor_cls().hist_days
  preload_start = get_target_period_backward(
    datetime.combine(start_date, datetime.max.time()),
    '1d',
    factor_history_days,
  )
  preload_end = datetime.combine(get_target_forward_day(end_date, max_m_days), datetime.max.time())
  init_market_data_range(stock_codes, preload_start, preload_end, '1d')

  core_logger.info(f"开始计算 {factor_cls.__name__} 相关性: {start_date} 至 {end_date}")
  core_logger.info(f"股票数: {len(stock_codes)}, 持有期: {m_days_list}")

  date_list = get_trading_date_span(start_date, end_date)
  worker_count = min(os.cpu_count() or 4, len(date_list))

  # 并行计算所有日期和持有期（带进度条）
  parallel_pool = Parallel(
    return_as='generator',
    n_jobs=worker_count,
    backend='loky',
    prefer='processes',
    batch_size=1,
    verbose=0,
  )
  daily_results_list = list(tqdm(
    parallel_pool(
      delayed(_calculate_daily_correlation)(d, stock_codes, factor_cls, m_days_list, date_list, save_stock_scores)
      for d in date_list
    ),
    total=len(date_list),
    maxinterval=30,
    desc=f"计算 {factor_cls.__name__} 相关性: {len(stock_codes)}只股票, 持有期{m_days_list}"
  ))

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

    if not valid_results:
      # 没有有效结果
      period_statistics.append(PeriodStatistics(
        m_days=m,
        daily_correlations=daily_results,
        avg_correlation=0.0,
        median_correlation=0.0,
        avg_rank_correlation=0.0,
        positive_days=0,
        negative_days=0,
        valid_days=0,
        total_samples=0,
        positive_count=0,
        negative_count=0,
        total_data_points=0,
        positive_data_points=0,
        negative_data_points=0,
        ic_mean=0.0,
        ic_std=0.0,
        ir=0.0,
        ic_ir=0.0
      ))
      continue

    # 使用加权平均：权重为有效股票数量
    weights = np.array([r.valid_stock_count for r in valid_results])
    valid_correlations = np.array([r.correlation for r in valid_results])
    valid_rank_correlations = np.array([r.rank_correlation for r in valid_results])

    # 加权平均相关系数
    total_weight = weights.sum()
    avg_corr = np.average(valid_correlations, weights=weights) if total_weight > 0 else 0.0
    avg_rank_corr = np.average(valid_rank_correlations, weights=weights) if total_weight > 0 else 0.0

    # 中位数不使用加权（保持原有逻辑）
    median_corr = np.median(valid_correlations)

    # 统计天数和样本量
    pos_days_list = [r for r in valid_results if r.correlation > 0]
    neg_days_list = [r for r in valid_results if r.correlation < 0]

    pos_days = len(pos_days_list)
    neg_days = len(neg_days_list)

    # 样本量 = 观察次数（天数）
    total_samples = len(valid_results)
    positive_count = pos_days
    negative_count = neg_days

    # 数据点数 = 用于计算相关性的数据点总数
    total_data_points = int(total_weight)
    positive_data_points = sum(r.valid_stock_count for r in pos_days_list)
    negative_data_points = sum(r.valid_stock_count for r in neg_days_list)

    # 计算IC指标
    # IC (Information Coefficient) = 因子得分与收益率的相关系数
    # IC均值：反映因子的平均预测能力
    # IC标准差：反映因子的稳定性
    # IR (Information Ratio) = IC均值 / IC标准差：衡量因子的风险调整后收益
    ic_mean = np.mean(valid_correlations)  # 简单平均IC
    ic_std = np.std(valid_correlations, ddof=1) if len(valid_correlations) > 1 else 0.0
    ir = ic_mean / ic_std if ic_std > 0 else 0.0

    # ICIR = |IC均值| / IC标准差 * sqrt(有效天数)：年化信息比率的近似
    ic_ir = abs(ic_mean) / ic_std * np.sqrt(len(valid_results)) if ic_std > 0 else 0.0

    # 计算加权有效天数（等效样本量）
    # 使用公式: 等效样本量 = (总权重)^2 / (权重平方和)
    # 这考虑了样本量不均匀的情况
    effective_days = (total_weight ** 2) / (weights ** 2).sum() if total_weight > 0 else len(valid_results)

    core_logger.info(
      f"T+{m}日: 加权相关系数={avg_corr:.4f}, IC={ic_mean:.4f}, IR={ir:.2f}, ICIR={ic_ir:.2f}, "
      f"有效天数={len(valid_results)}/{len(date_list)}, "
      f"样本量={total_samples}, 数据点总数={int(total_weight)}, 等效天数={effective_days:.1f}"
    )

    period_statistics.append(PeriodStatistics(
      m_days=m,
      daily_correlations=daily_results,
      avg_correlation=avg_corr,
      median_correlation=median_corr,
      avg_rank_correlation=avg_rank_corr,
      positive_days=pos_days,
      negative_days=neg_days,
      valid_days=len(valid_results),
      total_samples=total_samples,
      positive_count=positive_count,
      negative_count=negative_count,
      total_data_points=total_data_points,
      positive_data_points=positive_data_points,
      negative_data_points=negative_data_points,
      ic_mean=ic_mean,
      ic_std=ic_std,
      ir=ir,
      ic_ir=ic_ir
    ))

  # ========== 计算类型2：同一个股票但不同天数的相关性 ==========
  stock_period_statistics = []
  if show_stock_correlation:
    core_logger.info(f"开始计算类型2相关性（同一个股票但不同天数）")

    # 获取股票名称映射
    stock_name_map = {}
    for stock_code in stock_codes:
      try:
        detail = get_stock_detail(stock_code)
        if detail:
          stock_name_map[stock_code] = detail.get('InstrumentName', stock_code)
      except:
        stock_name_map[stock_code] = stock_code

    # 并行计算所有股票的相关性（类型2）
    stock_worker_count = min(os.cpu_count() or 4, len(stock_codes))
    stock_parallel_pool = Parallel(
      return_as='generator',
      n_jobs=stock_worker_count,
      backend='loky',
      prefer='processes',
      batch_size=1,
      verbose=0,
    )
    stock_results_list = list(tqdm(
      stock_parallel_pool(
        delayed(_calculate_stock_correlation)(
          stock_code, date_list, factor_cls, m_days_list, date_list, stock_name_map.get(stock_code)
        )
        for stock_code in stock_codes
      ),
      total=len(stock_codes),
      maxinterval=30,
      desc=f"计算类型2相关性: {len(stock_codes)}只股票, 持有期{m_days_list}"
    ))

    # 重组结果（类型2）
    stock_period_results = {m: [] for m in m_days_list}
    for stock_dict in stock_results_list:
      for m, stock_corr in stock_dict.items():
        stock_period_results[m].append(stock_corr)

    # 计算类型2的统计指标
    for m in m_days_list:
      stock_results = stock_period_results[m]
      valid_results = [r for r in stock_results if r.correlation is not None]

      if not valid_results:
        stock_period_statistics.append(StockPeriodStatistics(
          m_days=m,
          stock_correlations=stock_results,
          avg_correlation=0.0,
          median_correlation=0.0,
          avg_rank_correlation=0.0,
          positive_stocks=0,
          negative_stocks=0,
          valid_stocks=0,
          total_samples=0,
          positive_count=0,
          negative_count=0,
          total_data_points=0,
          positive_data_points=0,
          negative_data_points=0,
          ic_mean=0.0,
          ic_std=0.0,
          ir=0.0,
          ic_ir=0.0
        ))
        continue

      # 使用加权平均：权重为有效日期数量
      weights = np.array([r.valid_date_count for r in valid_results])
      valid_correlations = np.array([r.correlation for r in valid_results])
      valid_rank_correlations = np.array([r.rank_correlation for r in valid_results])

      # 加权平均相关系数
      total_weight = weights.sum()
      avg_corr = np.average(valid_correlations, weights=weights) if total_weight > 0 else 0.0
      avg_rank_corr = np.average(valid_rank_correlations, weights=weights) if total_weight > 0 else 0.0

      # 中位数不使用加权
      median_corr = np.median(valid_correlations)

      # 统计股票数和样本量
      pos_stocks_list = [r for r in valid_results if r.correlation > 0]
      neg_stocks_list = [r for r in valid_results if r.correlation < 0]

      pos_stocks = len(pos_stocks_list)
      neg_stocks = len(neg_stocks_list)

      # 样本量 = 观察次数（股票数）
      total_samples = len(valid_results)
      positive_count = pos_stocks
      negative_count = neg_stocks

      # 数据点数 = 用于计算相关性的数据点总数
      total_data_points = int(total_weight)
      positive_data_points = sum(r.valid_date_count for r in pos_stocks_list)
      negative_data_points = sum(r.valid_date_count for r in neg_stocks_list)

      # 计算IC指标
      ic_mean = np.mean(valid_correlations)
      ic_std = np.std(valid_correlations, ddof=1) if len(valid_correlations) > 1 else 0.0
      ir = ic_mean / ic_std if ic_std > 0 else 0.0
      ic_ir = abs(ic_mean) / ic_std * np.sqrt(len(valid_results)) if ic_std > 0 else 0.0

      core_logger.info(
        f"类型2 T+{m}日: 加权相关系数={avg_corr:.4f}, IC={ic_mean:.4f}, IR={ir:.2f}, ICIR={ic_ir:.2f}, "
        f"有效股票数={len(valid_results)}/{len(stock_codes)}, "
        f"样本量={total_samples}, 数据点总数={int(total_weight)}"
      )

      stock_period_statistics.append(StockPeriodStatistics(
        m_days=m,
        stock_correlations=stock_results,
        avg_correlation=avg_corr,
        median_correlation=median_corr,
        avg_rank_correlation=avg_rank_corr,
        positive_stocks=pos_stocks,
        negative_stocks=neg_stocks,
        valid_stocks=len(valid_results),
        total_samples=total_samples,
        positive_count=positive_count,
        negative_count=negative_count,
        total_data_points=total_data_points,
        positive_data_points=positive_data_points,
        negative_data_points=negative_data_points,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ir=ir,
        ic_ir=ic_ir
      ))
  else:
    core_logger.info(f"跳过类型2相关性计算（show_stock_correlation=False）")

  return FactorCorrelationReport(
    factor_cls.__name__, start_date, end_date, m_days_list, len(stock_codes),
    period_statistics, stock_period_statistics, show_stock_correlation
  )


def calculate_factor_correlation(
    factor_cls,
    start_date: date,
    end_date: date,
    m_days: int | list[int] = 5,
    stock_codes: list[str] = None,
    save_stock_scores: bool = False,
    show_stock_correlation: bool = False
) -> FactorCorrelationReport:
  """计算因子得分与T+M日收益率的相关性（一次性计算所有持有期）。"""
  with keep_windows_awake() as keep_awake_enabled:
    if keep_awake_enabled:
      core_logger.info('已启用 Windows 防休眠，相关性计算结束后自动恢复')
    else:
      core_logger.warning('未能启用 Windows 防休眠，系统可能仍按当前电源策略休眠')
    return _calculate_factor_correlation_impl(
      factor_cls=factor_cls,
      start_date=start_date,
      end_date=end_date,
      m_days=m_days,
      stock_codes=stock_codes,
      save_stock_scores=save_stock_scores,
      show_stock_correlation=show_stock_correlation,
    )

if __name__ == '__main__':
  """
  从本地 pkl 文件加载报告并生成 HTML
  """
  from core.factors.benchmark.report import generate_html_report

  # Hardcoded pkl file path
  pkl_file = "reports/factor-correlation-SmallCap-20251229_010317.pkl"

  # 检查文件是否存在
  if not os.path.exists(pkl_file):
    print(f"错误: 文件不存在: {pkl_file}")
    sys.exit(1)

  # 检查文件扩展名
  if not pkl_file.endswith('.pkl'):
    print(f"错误: 文件必须是 .pkl 格式")
    sys.exit(1)

  # 加载 pkl 文件
  try:
    with open(pkl_file, 'rb') as f:
      report = pickle.load(f)

    # 生成 HTML 报告
    html_file = generate_html_report(report)
    print(f"HTML 报告已生成: {html_file}")

  except Exception as e:
    print(f"错误: 加载或生成报告失败: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
