from __future__ import annotations


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
  """获取沪深300在回测期间的累计收益率（%），从预下载 parquet 读取。

  数据缺失（FileNotFoundError）属预下载问题，直接抛错由调用方决定如何处理；
  不再静默返回全 0，避免基准失真造成夏普虚高。
  """
  if not trade_dates:
    return []

  from pathlib import Path
  import pyarrow.parquet as pq

  path = Path(__file__).resolve().parents[1] / "data" / "index_sh000300_daily.parquet"
  if not path.exists():
    raise FileNotFoundError(
        f'沪深300指数数据缺失: {path}，请先运行 data/update_all.py'
    )

  table = pq.read_table(path)
  dates_arr = table.column('trade_date').to_numpy().astype('datetime64[D]')
  close_arr = table.column('close').to_numpy()

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


def compute_strategy_metrics(
    cumulative_returns_pct: List[float],
    trade_dates: List[str],
    trade_log: List[Dict],
) -> Dict:
  """计算策略核心指标。"""
  if not cumulative_returns_pct or not trade_dates:
    return {}

  nav = np.array(cumulative_returns_pct, dtype=float) / 100.0 + 1.0
  prev_nav = np.concatenate(([1.0], nav[:-1]))
  daily_rets = (nav / np.where(prev_nav == 0, np.nan, prev_nav) - 1.0) * 100.0
  daily_rets_clean = daily_rets[~np.isnan(daily_rets)]

  from core.metrics import compute_core_metrics
  core = compute_core_metrics(daily_rets_clean)
  annualized = core['annualized']
  max_dd = core['max_drawdown']
  sharpe = core['sharpe']

  if len(nav) > 0:
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    max_dd_idx = int(np.argmin(drawdown))
    peak_idx = int(np.argmax(nav[:max_dd_idx + 1])) if max_dd_idx >= 0 else 0
    max_dd_start = trade_dates[peak_idx]
    max_dd_end = trade_dates[max_dd_idx]
  else:
    max_dd_start = trade_dates[0]
    max_dd_end = trade_dates[0]

  calmar = abs(annualized / max_dd) if max_dd < 0 else 0.0

  sell_trades = [t for t in trade_log if t.get('action') == 'sell']
  buy_trades = [t for t in trade_log if t.get('action') == 'buy']
  incomes = [t['income'] for t in sell_trades if t.get('income') is not None]
  wins = [i for i in incomes if i > 0]
  losses = [i for i in incomes if i <= 0]
  win_rate = len(wins) / len(incomes) * 100 if incomes else 0.0
  total_commission = sum(t.get('commission', 0) for t in trade_log if t.get('commission') is not None)

  return {
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
