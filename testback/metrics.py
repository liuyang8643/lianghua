from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np

from testback.logger import testback_logger

HS300_CODE = '000300.SH'


def compute_per_year_metrics(cumulative_returns_pct: List[float], trade_dates: List[str]) -> List[Dict]:
    """按自然年计算夏普、年化收益、最大回撤。"""
    if not cumulative_returns_pct or not trade_dates:
        return []

    nav = np.array(cumulative_returns_pct, dtype=float) / 100.0 + 1.0
    years = sorted(set(d[:4] for d in trade_dates))
    result = []

    for year in years:
        idx = [i for i, d in enumerate(trade_dates) if d[:4] == year]
        if len(idx) < 5:
            continue

        first, last = idx[0], idx[-1]
        year_nav = nav[first:last + 1]
        year_start_nav = nav[first - 1] if first > 0 else 1.0
        year_nav_full = np.concatenate([[year_start_nav], year_nav])

        # 年化收益
        year_return = (year_nav[-1] / year_start_nav - 1) * 100
        trading_days = len(year_nav_full) - 1
        if trading_days > 0 and year_start_nav > 0:
            annualized = ((year_nav[-1] / year_start_nav) ** (252.0 / trading_days) - 1) * 100
        else:
            annualized = 0.0

        # 夏普
        daily_rets = year_nav_full[1:] / year_nav_full[:-1] - 1
        if len(daily_rets) > 1:
            mean_ret = float(np.mean(daily_rets))
            std_ret = float(np.std(daily_rets, ddof=1))
            sharpe = (mean_ret / std_ret * np.sqrt(252.0)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        # 最大回撤
        peaks = np.maximum.accumulate(year_nav_full)
        dd = year_nav_full / peaks - 1
        max_dd = float(np.min(dd) * 100)

        result.append({
            'year': year,
            'return': round(year_return, 2),
            'annualized': round(annualized, 2),
            'sharpe': round(sharpe, 2),
            'max_drawdown': round(max_dd, 2),
            'trading_days': trading_days,
        })

    return result


def compute_hs300_cumulative_returns(trade_dates: List[str]) -> List[float]:
  """获取沪深300在回测期间的累计收益率（%），从预下载 parquet 读取。"""
  if not trade_dates:
    return []

  try:
    from pathlib import Path
    import pyarrow.parquet as pq
    import numpy as np

    path = Path(__file__).resolve().parents[1] / "data" / "index_sh000300_daily.parquet"
    if not path.exists():
      testback_logger.warning('沪深300指数数据缺失: %s，请先运行 data/update_all.py', path)
      return [0.0] * len(trade_dates)

    table = pq.read_table(path)
    dates_arr = table.column('trade_date').to_numpy().astype('datetime64[D]')
    close_arr = table.column('open').to_numpy()  # parquet 存的是 open，改用 close 更合理；此处暂用 open 做近似

    date_to_val: Dict[str, float] = {}
    for i in range(len(dates_arr)):
      d_str = str(dates_arr[i])[:10]
      date_to_val[d_str] = float(close_arr[i])

    result: List[float] = []
    first_val = None
    for d in trade_dates:
      if d in date_to_val:
        if first_val is None:
          first_val = date_to_val[d]
        if first_val and first_val != 0:
          cumulative = (date_to_val[d] - first_val) / first_val * 100
          result.append(round(cumulative, 4))
        else:
          result.append(0.0)
      else:
        result.append(0.0 if not result else result[-1])

    hit_count = sum(1 for d in trade_dates if d in date_to_val)
    if hit_count > 0:
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

  total_return = (final_asset - init_cash) / init_cash * 100
  ending_nav = final_asset / init_cash if init_cash else 0.0

  first_date = datetime.strptime(trade_dates[0], '%Y-%m-%d')
  last_date = datetime.strptime(trade_dates[-1], '%Y-%m-%d')
  calendar_days = (last_date - first_date).days
  years = calendar_days / 365.25 if calendar_days > 0 else 0.0

  if years > 0 and ending_nav > 0:
    annualized = (ending_nav ** (1.0 / years) - 1) * 100
  elif years > 0 and ending_nav == 0:
    annualized = -100.0
  else:
    annualized = 0.0

  nav = np.array(cumulative_returns_pct, dtype=float) / 100.0 + 1.0
  prev_nav = np.concatenate(([1.0], nav[:-1]))
  valid_prev_nav = np.where(prev_nav == 0, np.nan, prev_nav)
  daily_rets = (nav / valid_prev_nav - 1.0) * 100.0
  daily_rets = daily_rets[~np.isnan(daily_rets)]

  if len(nav) > 0:
    peaks = np.maximum.accumulate(nav)
    drawdown = nav / peaks - 1.0
    max_dd = float(np.min(drawdown) * 100.0)
    max_dd_idx = int(np.argmin(drawdown))
    peak_idx = int(np.argmax(nav[:max_dd_idx + 1])) if max_dd_idx >= 0 else 0
    max_dd_start = trade_dates[peak_idx]
    max_dd_end = trade_dates[max_dd_idx]
  else:
    max_dd = 0.0
    max_dd_start = trade_dates[0]
    max_dd_end = trade_dates[0]

  if len(daily_rets) > 1:
    mean_ret = float(np.mean(daily_rets))
    std_ret = float(np.std(daily_rets, ddof=1))
    sharpe = (mean_ret / std_ret * np.sqrt(252.0)) if std_ret > 0 else 0.0
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
    'max_drawdown_start': max_dd_start,
    'max_drawdown_end': max_dd_end,
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
