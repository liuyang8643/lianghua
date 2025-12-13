from .helpers import *
import numpy as np

class KDJ(BaseFactor):
  """
  KDJ指标因子
  评分基于KDJ交叉：金叉时数值最大（正峰值），死叉时数值最小（负峰值）
  """

  def __init__(self, period: int = 9, sensitivity: float = 0.5):
    """
    :param period: KDJ周期，默认9
    :param sensitivity: 敏感度系数，默认0.5。数值越大，非交叉时刻的分数衰减越快
    """
    super().__init__()
    self.period = period
    self.sensitivity = sensitivity

  def calculate_cross_score(self, k: np.ndarray, d: np.ndarray, j: np.ndarray) -> float:
    """
    计算KDJ交叉评分：金叉时数值最大（正峰值），死叉时数值最小（负峰值）

    :param k: K值序列
    :param d: D值序列
    :param j: J值序列
    :return: 交叉评分（最新值）
    """
    # 1. 计算差值 (Spread)
    # 使用 J - D，J线最快
    diff = j - d

    # 2. 计算差值的变化速度 (Velocity / Momentum)
    # fillna(0) 是为了防止第一行出现 NaN
    diff_velocity = np.diff(diff, prepend=diff[0])

    # 3. 计算"距离权重" (Proximity Weight)
    # 使用高斯/指数衰减：当 diff 为 0 时，权重为 1；diff 越大，权重越小
    weight = np.exp(-self.sensitivity * np.abs(diff))

    # 4. 最终得分 = 速度 * 权重
    # 金叉：diff约为0(权重高) + 速度向上(正数) -> 正高分
    # 死叉：diff约为0(权重高) + 速度向下(负数) -> 负高分
    score = diff_velocity * weight

    return score[-1]

  @cached_factor('KDJ')
  def calc(self, ctx: FactorCtx) -> FactorResult:
    try:
      k, d, j = ctx.get_kdj(self.period)
      score = self.calculate_cross_score(k, d, j)
      return FactorResult(score=score, err=None)
    except Exception as e:
      return FactorResult(score=None, err=e)

