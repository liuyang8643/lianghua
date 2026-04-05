import numpy as np
from datetime import datetime

from .helpers import BaseFactor, FactorCtx, FactorResult
from core.database.money_flow import get_retail_flow_amount, get_retail_net_flow

_INTERACTION_SCALE = 0.001


class WMACrossShort(BaseFactor):
  """
  Short-cycle WMACross composite:
  - WMACross mean-reversion core (anchor).
  - Retail participation level.
  - Retail flow momentum.
  """

  def __init__(
      self,
      fast_period: int = 5,
      slow_period: int = 20,
      momentum_lookback: int = 5,
      momentum_weight: float = 1.0,
      contrarian_weight: float = 0.5,
      retail_smooth_days: int = 3,
      flow_mom_period: int = 5,
      weight_wma: float = 2.0,
      weight_retail_ratio: float = 1.0,
      weight_flow_momentum: float = 0.2,
      weight_interaction: float = -1.0,
      retail_ratio_center: float = 87.88,
      retail_ratio_scale: float = 45.87,
      flow_mom_center: float = -0.175,
      flow_mom_scale: float = 18.61,
  ):
    super().__init__()
    self.fast_period = fast_period
    self.slow_period = slow_period
    self.momentum_lookback = momentum_lookback
    self.momentum_weight = momentum_weight
    self.contrarian_weight = contrarian_weight
    self.retail_smooth_days = retail_smooth_days
    self.flow_mom_period = flow_mom_period

    self.weight_wma = weight_wma
    self.weight_retail_ratio = weight_retail_ratio
    self.weight_flow_momentum = weight_flow_momentum
    self.weight_interaction = weight_interaction

    self.retail_ratio_center = retail_ratio_center
    self.retail_ratio_scale = retail_ratio_scale
    self.flow_mom_center = flow_mom_center
    self.flow_mom_scale = flow_mom_scale

  @staticmethod
  def _tanh_norm(value: float, center: float, scale: float) -> float:
    return float(np.tanh((value - center) / (scale + 1e-8)))

  def _calc_wma_core(self, ctx: FactorCtx, history_data) -> float | None:
    typical_price = (
      history_data['open'].astype(float)
      + history_data['close'].astype(float)
      + history_data['low'].astype(float)
      + history_data['high'].astype(float)
    ) / 4.0
    amounts = history_data['amount'].astype(float)
    vwtp = typical_price * amounts

    wma_fast = (
      vwtp.rolling(window=self.fast_period).mean() /
      amounts.rolling(window=self.fast_period).mean()
    )
    wma_slow = (
      vwtp.rolling(window=self.slow_period).mean() /
      amounts.rolling(window=self.slow_period).mean()
    )

    valid_mask = wma_fast.notna() & wma_slow.notna()
    wma_fast_valid = wma_fast[valid_mask]
    wma_slow_valid = wma_slow[valid_mask]
    if len(wma_fast_valid) < self.momentum_lookback + 1:
      return None

    current_fast = float(wma_fast_valid.iloc[-1])
    current_slow = float(wma_slow_valid.iloc[-1])
    spread_pct = (current_fast - current_slow) / (abs(current_slow) + 1e-8)

    lookback = min(self.momentum_lookback, len(wma_fast_valid) - 1)
    prev_fast = float(wma_fast_valid.iloc[-(lookback + 1)])
    prev_slow = float(wma_slow_valid.iloc[-(lookback + 1)])
    prev_spread_pct = (prev_fast - prev_slow) / (abs(prev_slow) + 1e-8)
    spread_momentum = spread_pct - prev_spread_pct

    base_score = -(spread_pct + self.momentum_weight * spread_momentum)

    amplifier = 1.0
    if self.contrarian_weight > 0:
      retail_net_pcts = []
      for i in range(self.retail_smooth_days):
        idx = -(i + 1)
        if abs(idx) > len(history_data):
          break
        row = history_data.iloc[idx]
        amount = float(row['amount'])
        if amount <= 0:
          continue
        ts = int(row['time'])
        trade_date = datetime.fromtimestamp(ts / 1000).date()
        retail_net = get_retail_net_flow(ctx.code, trade_date)
        if retail_net is not None:
          retail_net_pcts.append(retail_net / amount)

      if retail_net_pcts:
        avg_retail_net_pct = float(np.mean(retail_net_pcts))
        interaction = spread_pct * avg_retail_net_pct
        norm_interaction = interaction / _INTERACTION_SCALE
        amplifier = max(0.1, 1.0 + self.contrarian_weight * norm_interaction)

    return float(base_score * amplifier)

  def _calc_retail_ratio_and_momentum(self, ctx: FactorCtx, history_data) -> tuple[float, float]:
    ratios = []
    lookback = max(self.flow_mom_period + 1, 3)

    for i in range(lookback):
      idx = -(i + 1)
      if abs(idx) > len(history_data):
        break
      row = history_data.iloc[idx]
      amount = float(row['amount'])
      if amount <= 0:
        continue

      ts = int(row['time'])
      trade_date = datetime.fromtimestamp(ts / 1000).date()
      retail_amount = get_retail_flow_amount(ctx.code, trade_date)
      if retail_amount is None:
        continue

      ratios.append((retail_amount / amount) * 100.0)

    if not ratios:
      return self.retail_ratio_center, self.flow_mom_center

    today_ratio = float(ratios[0])
    mom_window = ratios[:min(self.flow_mom_period, len(ratios))]
    ma_ratio = float(np.mean(mom_window)) if mom_window else today_ratio
    if abs(ma_ratio) <= 1e-8:
      flow_mom = self.flow_mom_center
    else:
      flow_mom = -((today_ratio - ma_ratio) / ma_ratio) * 100.0

    return today_ratio, float(flow_mom)

  def calc(self, ctx: FactorCtx) -> FactorResult:
    try:
      days_needed = max(
        self.slow_period + max(self.momentum_lookback, self.retail_smooth_days) + 2,
        self.flow_mom_period + 2,
      )
      history_data = ctx.get_daily_data(days_needed)
      if history_data is None or len(history_data) < self.slow_period + 2:
        return FactorResult(score=None, err=None)

      wma_score = self._calc_wma_core(ctx, history_data)
      if wma_score is None:
        return FactorResult(score=None, err=None)

      retail_ratio, flow_mom = self._calc_retail_ratio_and_momentum(ctx, history_data)

      retail_ratio_n = self._tanh_norm(retail_ratio, self.retail_ratio_center, self.retail_ratio_scale)
      flow_mom_n = self._tanh_norm(flow_mom, self.flow_mom_center, self.flow_mom_scale)

      interaction = wma_score * flow_mom_n

      final_score = (
        self.weight_wma * wma_score
        + self.weight_retail_ratio * retail_ratio_n
        + self.weight_flow_momentum * flow_mom_n
        + self.weight_interaction * interaction
      )

      return FactorResult(score=float(final_score), err=None)

    except Exception as e:
      return FactorResult(score=None, err=e)
