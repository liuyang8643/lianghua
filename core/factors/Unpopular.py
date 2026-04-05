import numpy as np
from core.factors.helpers import BaseFactor, FactorResult, FactorCtx

class Unpopular(BaseFactor):
  """冷门股因子 - 基于大盘相关性和持续时间 (独立于市值)"""

  def calc(self, ctx: FactorCtx) -> FactorResult:
    period = 60
    history_data = ctx.get_daily_data(period)

    closes = history_data['close'].values
    pre_closes = history_data['preClose'].values
    pct_changes = ((closes - pre_closes) / pre_closes) * 100

    # 大盘相关性
    benchmark_pct = ctx.get_benchmark_pct_change(period)
    if benchmark_pct is None:
      raise ValueError(f"无法获取大盘数据: {ctx.code}")

    min_len = min(len(pct_changes), len(benchmark_pct))
    corr_60d = np.corrcoef(pct_changes[-min_len:], benchmark_pct[-min_len:])[0, 1]
    corr_5d = np.corrcoef(pct_changes[-5:], benchmark_pct[-5:])[0, 1] if len(pct_changes) >= 5 else corr_60d

    # 冷门持续时间 (相关性高于0.5的天数占比)
    rolling_corr = np.array([np.corrcoef(pct_changes[i:i+20], benchmark_pct[i:i+20])[0,1]
                             if i+20 <= len(pct_changes) else corr_60d
                             for i in range(0, len(pct_changes), 5)])
    is_unpopular = rolling_corr > 0.5
    unpopular_days = is_unpopular.sum()
    duration_factor = unpopular_days / len(rolling_corr) if len(rolling_corr) > 0 else 0

    # base_score: 冷门度 (越跟随大盘越冷门)
    unpopularity = 0.60 * (corr_60d + 0.1) + 0.40 * duration_factor
    base_score = min(100, unpopularity * 100)

    # signal_bonus: 相关性下降 (从跟随到独立)
    signal_bonus = 0
    if corr_60d > 0.6 and corr_5d < 0.4:
      signal_bonus = 50
    elif corr_60d > 0.5 and corr_5d < 0.3:
      signal_bonus = 30

    # 最终分数
    final_score = 0.6 * base_score + 0.4 * signal_bonus

    return FactorResult(
      score=final_score,
      err=None,
      raw_value=unpopularity,
      metadata={
        'corr_60d': corr_60d,
        'corr_5d': corr_5d,
        'unpopular_ratio': duration_factor,
      }
    )
