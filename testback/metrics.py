from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np

from testback.logger import testback_logger

HS300_CODE = '000300.SH'


def compute_hs300_cumulative_returns(trade_dates: List[str]) -> List[float]:
  """获取沪深300在回测期间的累计收益率（%）。

  注意：指数不走 get_market_data_from_cache（会被 check_stock_valid_at_date 拦截），
  直接用 get_full_market_data 取全量日线再按日期切片。
  """
  if not trade_dates:
    return []

  try:
    from core.database.data import get_full_market_data

    base_time = datetime.strptime(trade_dates[-1], '%Y-%m-%d')
    data = get_full_market_data(HS300_CODE, '1d', target_time=base_time,
                                allow_tainted=True, dividend_type='back')

    if data is None or data.empty:
      testback_logger.warning(f'沪深300 ({HS300_CODE}) 数据获取失败，返回全0基准线')
      return [0.0] * len(trade_dates)

    date_to_close: Dict[str, float] = {}
    for ts, close in zip(data['time'].values, data['close'].values):
      d = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d')
      date_to_close[d] = float(close)

    result: List[float] = []
    first_close = None
    for d in trade_dates:
      if d in date_to_close:
        if first_close is None:
          first_close = date_to_close[d]
        if first_close and first_close != 0:
          cumulative = (date_to_close[d] - first_close) / first_close * 100
          result.append(round(cumulative, 4))
        else:
          result.append(0.0)
      else:
        result.append(0.0 if not result else result[-1])

    hit_count = sum(1 for d in trade_dates if d in date_to_close)
    if hit_count == 0:
      testback_logger.warning(f'沪深300数据无日期命中（共 {len(date_to_close)} 条），返回全0')
    else:
      testback_logger.info(f'沪深300基准加载成功: {hit_count}/{len(trade_dates)} 日命中')
    return result
  except Exception as e:
    testback_logger.warning(f'获取沪深300数据时出错: {e}，返回全0基准线')
    return [0.0] * len(trade_dates)


def compute_strategy_metrics(
    cumulative_returns_pct: List[float],
    trade_dates: List[str],
    init_cash: float,
    final_asset: float,
    trade_log: List[Dict],
) -> Dict:
  """计算策略核心指标。"""
  if not cumulative_returns_pct or not trade_dates:
    return {}

  n = len(trade_dates)
  total_return = (final_asset - init_cash) / init_cash * 100

  first_date = datetime.strptime(trade_dates[0], '%Y-%m-%d')
  last_date = datetime.strptime(trade_dates[-1], '%Y-%m-%d')
  calendar_days = (last_date - first_date).days
  years = calendar_days / 365.25 if calendar_days > 0 else 0.0

  if years > 0 and total_return > -100:
    annualized = ((1 + total_return / 100) ** (1.0 / years) - 1) * 100
  else:
    annualized = total_return / years if years > 0 else 0.0

  cumulative_arr = np.array(cumulative_returns_pct)
  if len(cumulative_arr) >= 2:
    daily_rets = np.diff(cumulative_arr)
  else:
    daily_rets = np.array([])

  if len(daily_rets) > 0:
    cum = np.concatenate([[0.0], daily_rets]).cumsum()
    peak = np.maximum.accumulate(cum)
    drawdown = cum - peak
    max_dd = float(np.min(drawdown))
  else:
    max_dd = 0.0

  if len(daily_rets) > 1:
    mean_ret = float(np.mean(daily_rets))
    std_ret = float(np.std(daily_rets, ddof=1))
    periods_per_year = n / years if years > 0 else 252.0
    sharpe = (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0
  else:
    sharpe = 0.0

  calmar = abs(annualized / max_dd) if max_dd < 0 else 0.0

  sell_trades = [t for t in trade_log if t.get('action') == 'sell']
  buy_trades = [t for t in trade_log if t.get('action') == 'buy']
  incomes = [t['income'] for t in sell_trades if t.get('income') is not None]
  wins = [i for i in incomes if i > 0]
  losses = [i for i in incomes if i <= 0]
  win_rate = len(wins) / len(incomes) * 100 if incomes else 0.0
  total_commission = sum(t.get('commission', 0) for t in trade_log if t.get('commission') is not None)

  return {
    'total_return': round(total_return, 2),
    'annualized': round(annualized, 2),
    'max_drawdown': round(max_dd, 2),
    'sharpe_ratio': round(sharpe, 2),
    'calmar_ratio': round(calmar, 2),
    'win_rate': round(win_rate, 1),
    'total_trades': len(buy_trades) + len(sell_trades),
    'buy_trades': len(buy_trades),
    'sell_trades': len(sell_trades),
    'wins': len(wins),
    'losses': len(losses),
    'avg_profit': round(float(np.mean(incomes)), 2) if incomes else 0.0,
    'avg_win': round(float(np.mean(wins)), 2) if wins else 0.0,
    'avg_loss': round(float(np.mean(losses)), 2) if losses else 0.0,
    'max_profit': round(max(incomes), 2) if incomes else 0.0,
    'max_loss': round(min(incomes), 2) if incomes else 0.0,
    'total_commission': round(total_commission, 2),
  }
