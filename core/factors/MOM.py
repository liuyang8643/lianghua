"""
MOM动量因子 - 纯趋势追踪（加法混合设计）

从qmt-trade项目迁移，输出原始连续分数。

金融逻辑:
- 追涨逻辑，正动量（价格上涨）时给高分
- 负动量（价格下跌）时给低分
- ROC加速（连续上涨）时给予额外加分

评分设计:
- base_score (50%): sigmoid映射（中心点5%，正动量高分）
- signal_bonus (50%): ROC加速确认（连续上涨 + ROC>0前置条件）

注意: 本因子输出原始分数，需要在框架层使用batch_norm进行归一化
"""

from .helpers import *
import numpy as np
import talib


def safe_sigmoid(x, clip_range=20):
    """安全的sigmoid函数，避免exp溢出"""
    x_clipped = np.clip(x, -clip_range, clip_range)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class MOMFactor(BaseFactor):
    """
    MOM动量因子

    评分逻辑:
    1. base_score (50%): sigmoid映射（中心点5%，正动量高分）
    2. signal_bonus (50%): ROC加速确认（连续上涨 + ROC>0前置条件）

    输出: 原始分数 [0, 1]，由框架层进行batch norm归一化
    """

    def __init__(self,
                 mom_period: int = 10,
                 roc_period: int = 10,
                 mom_center: float = 5.0,
                 acceleration_lookback: int = 5):
        """
        :param mom_period: 动量计算周期，默认10天
        :param roc_period: ROC计算周期，默认10天
        :param mom_center: sigmoid映射的中心点（百分比），默认5.0
        :param acceleration_lookback: ROC加速检测回溯窗口，默认5天
        """
        super().__init__()
        self.mom_period = mom_period
        self.roc_period = roc_period
        self.mom_center = mom_center
        self.acceleration_lookback = acceleration_lookback

    def calc(self, ctx: FactorCtx) -> FactorResult:
        try:
            # 获取动量比率（百分比形式）
            mom_ratio = ctx.get_mom_ratio(period=self.mom_period)

            # === 1. 连续基础分 (50%) ===
            # sigmoid映射：中心点5%（正常追涨起点）
            # mom_ratio=5 → 0.25, mom_ratio=15 → ~0.43, mom_ratio=-5 → ~0.07
            base_score = safe_sigmoid((mom_ratio - self.mom_center) / self.mom_center) * 0.5

            # === 2. 信号加成 (50%) ===
            signal_bonus = 0.0

            # ROC加速检测（连续上涨 + ROC>0前置条件）
            # 需要足够的历史数据来计算ROC序列
            history_data = ctx.get_daily_data(self.roc_period + self.acceleration_lookback)

            if len(history_data) >= self.roc_period + self.acceleration_lookback:
                close = np.array(history_data['close'].values, dtype=np.float64)
                roc_series = talib.ROC(close, timeperiod=self.roc_period)

                # 提取最近的有效数据（去掉NaN）
                valid_mask = ~np.isnan(roc_series)
                valid_roc = roc_series[valid_mask]

                if len(valid_roc) >= self.acceleration_lookback:
                    # 取最近5天的ROC
                    recent_roc = valid_roc[-self.acceleration_lookback:]
                    current_roc = recent_roc[-1]

                    # 前置条件：当前ROC必须为正（追涨逻辑）
                    if current_roc > 0:
                        # 检测最近5天ROC是否连续上升
                        is_accelerating = all(
                            recent_roc[i] > recent_roc[i-1]
                            for i in range(1, len(recent_roc))
                        )

                        if is_accelerating:
                            roc_base = abs(recent_roc[0])
                            if roc_base != 0:  # 只有base非0才计算加速度
                                acceleration = (recent_roc[-1] - recent_roc[0]) / roc_base
                                signal_bonus = min(acceleration / 0.1, 1.0) * 0.5

            # === 3. 最终分数 = base + bonus ===
            final_score = base_score + signal_bonus

            return FactorResult(score=final_score, err=None)

        except Exception as e:
            return FactorResult(score=None, err=e)
