"""
布林带因子 - 加法混合设计

从qmt-trade项目迁移，输出原始连续分数。

金融逻辑:
- bb_position=0 (下轨) → 超卖买入机会
- bb_position=1 (上轨) → 超买卖出信号
- bb_bandwidth小 → 波动率收缩，可能突破
- 低位缩口 (bb_position < 0.3 and bb_bandwidth < 0.05) → 强买入信号

评分设计:
- base_score (50%): 布林带位置 (连续值，越低越好)
- signal_bonus (50%): 低位缩口突破 (离散信号)

注意: 本因子输出原始分数，需要在框架层使用batch_norm进行归一化
"""

from .helpers import *
import numpy as np


class BollingerBandsFactor(BaseFactor):
    """
    布林带因子

    评分逻辑:
    1. base_score (50%): 布林带位置 (连续值，越低越好)
    2. signal_bonus (50%): 低位缩口突破 (离散信号)

    输出: 原始分数 [0, 1]，由框架层进行batch norm归一化
    """

    def __init__(self,
                 period: int = 20,
                 nbdevup: float = 2.0,
                 nbdevdn: float = 2.0,
                 squeeze_threshold: float = 0.05,
                 low_position_threshold: float = 0.3):
        """
        :param period: 计算周期，默认20天
        :param nbdevup: 上轨标准差倍数，默认2
        :param nbdevdn: 下轨标准差倍数，默认2
        :param squeeze_threshold: 缩口阈值，默认0.05
        :param low_position_threshold: 低位阈值，默认0.3
        """
        super().__init__()
        self.period = period
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn
        self.squeeze_threshold = squeeze_threshold
        self.low_position_threshold = low_position_threshold

    @cached_factor('BollingerBandsFactor')
    def calc(self, ctx: FactorCtx) -> FactorResult:
        try:
            # 获取布林带指标
            upper, middle, lower, bb_position, bb_bandwidth = ctx.get_bollinger_bands(
                period=self.period,
                nbdevup=self.nbdevup,
                nbdevdn=self.nbdevdn
            )

            # === 1. 连续基础分 (50%) ===
            base_score = (1.0 - bb_position) * 0.5  # 位置越低分越高

            # === 2. 信号加成 (50%) ===
            signal_bonus = 0.0

            # 低位缩口：必须同时满足位置低 + 带宽窄
            if bb_position < self.low_position_threshold and bb_bandwidth < self.squeeze_threshold:
                # 缩口强度：带宽越窄 + 位置越低，加成越高
                squeeze_strength = 1.0 - (bb_bandwidth / self.squeeze_threshold)  # [0, 1]
                position_strength = 1.0 - (bb_position / self.low_position_threshold)  # [0, 1]
                signal_bonus = (squeeze_strength * position_strength) * 0.5

            # === 3. 最终分数 = base + bonus ===
            final_score = base_score + signal_bonus

            return FactorResult(score=final_score, err=None)

        except Exception as e:
            return FactorResult(score=None, err=e)
