import numpy as np
from datetime import datetime

from .helpers import BaseFactor, FactorCtx, FactorResult
from core.database.money_flow import get_retail_flow_amount


class SmallCapShortV2(BaseFactor):
  """
  Frozen version of the current best short-cycle SmallCap composite.
  """

  def __init__(
      self,
      base_window: int = 60,
      base_scale_yi: float = 5.0,
      flow_mom_period: int = 5,
      weight_size: float = 0.8,
      weight_retail_ratio: float = 1.2,
      weight_flow_momentum: float = 0.4,
      weight_interaction: float = -0.4,
      size_center: float = 74.48,
      size_scale: float = 49.58,
      retail_ratio_center: float = 87.88,
      retail_ratio_scale: float = 45.87,
      flow_mom_center: float = -0.175,
      flow_mom_scale: float = 18.61,
  ):
    super().__init__()
    self.base_window = base_window
    self.base_scale_yi = base_scale_yi
    self.flow_mom_period = flow_mom_period

    self.weight_size = weight_size
    self.weight_retail_ratio = weight_retail_ratio
    self.weight_flow_momentum = weight_flow_momentum
    self.weight_interaction = weight_interaction

    self.size_center = size_center
    self.size_scale = size_scale
    self.retail_ratio_center = retail_ratio_center
    self.retail_ratio_scale = retail_ratio_scale
    self.flow_mom_center = flow_mom_center
    self.flow_mom_scale = flow_mom_scale

  @staticmethod
  def _tanh_norm(value: float, center: float, scale: float) -> float:
    return float(np.tanh((value - center) / (scale + 1e-8)))

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
      days_needed = max(self.base_window, self.flow_mom_period + 2)
      history_data = ctx.get_daily_data(days_needed)
      if history_data is None or len(history_data) < self.base_window:
        return FactorResult(score=None, err=None)

      amounts = history_data['amount'].astype(float).values
      avg_amount = float(np.mean(amounts[-self.base_window:]))
      if avg_amount <= 0:
        return FactorResult(score=None, err=None)

      avg_amount_yi = avg_amount / 1e8
      size_score = 100.0 * np.exp(-(avg_amount_yi / self.base_scale_yi))

      retail_ratio, flow_mom = self._calc_retail_ratio_and_momentum(ctx, history_data)

      size_n = self._tanh_norm(size_score, self.size_center, self.size_scale)
      retail_ratio_n = self._tanh_norm(retail_ratio, self.retail_ratio_center, self.retail_ratio_scale)
      flow_mom_n = self._tanh_norm(flow_mom, self.flow_mom_center, self.flow_mom_scale)

      interaction = size_n * retail_ratio_n

      final_score = (
        self.weight_size * size_n
        + self.weight_retail_ratio * retail_ratio_n
        + self.weight_flow_momentum * flow_mom_n
        + self.weight_interaction * interaction
      )

      return FactorResult(score=float(final_score), err=None)

    except Exception as e:
      return FactorResult(score=None, err=e)
