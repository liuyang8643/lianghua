"""
RSI超卖因子 - 加法混合设计

从qmt-trade项目迁移，输出原始连续分数。

金融逻辑:
- base_score (50%): RSI超卖程度 (连续值，RSI越低分越高)
- signal_bonus (50%): 底背离 (离散信号)

注意: 本因子输出原始分数，需要在框架层使用batch_norm进行归一化
"""

from .helpers import *
import numpy as np


def safe_sigmoid(x, clip_range=20):
    """安全的sigmoid函数，避免exp溢出"""
    x_clipped = np.clip(x, -clip_range, clip_range)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class RSIOversold(BaseFactor):
    """
    RSI超卖因子

    评分逻辑:
    1. base_score (50%): RSI越低分越高（全范围连续评分）
    2. signal_bonus (50%): 底背离检测（离散信号）

    输出: 原始分数 [0, 1]，由框架层进行batch norm归一化
    """

    def __init__(self,
                 rsi_period: int = 14,
                 divergence_lookback: int = 10,
                 divergence_order: int = 3):
        """
        :param rsi_period: RSI计算周期，默认14天
        :param divergence_lookback: 背离检测回溯窗口，默认10天
        :param divergence_order: 极值点检测窗口，默认3
        """
        super().__init__()
        self.rsi_period = rsi_period
        self.divergence_lookback = divergence_lookback
        self.divergence_order = divergence_order

    @cached_factor('RSIOversold')
    def calc(self, ctx: FactorCtx) -> FactorResult:
        try:
            # === 1. 连续基础分 (50%) ===
            rsi_value = ctx.get_rsi(period=self.rsi_period)

            # RSI越低越好，全范围连续评分
            if rsi_value < 70:
                base_score = (70 - rsi_value) / 70  # RSI=70→0, RSI=0→1
                base_score = base_score * 0.5  # 基础分占50%
            else:
                base_score = 0

            # === 2. 信号加成 (50%) ===
            signal_bonus = 0.0

            # 获取历史数据用于背离检测
            history_data = ctx.get_daily_data(self.divergence_lookback + self.rsi_period)

            if len(history_data) >= self.divergence_lookback + self.rsi_period:
                # 计算历史RSI
                close_prices = np.array(history_data['close'].values, dtype=np.float64)
                import talib
                rsi_series = talib.RSI(close_prices, timeperiod=self.rsi_period)

                # 提取最近lookback个点（去掉NaN）
                valid_mask = ~np.isnan(rsi_series)
                valid_close = close_prices[valid_mask]
                valid_rsi = rsi_series[valid_mask]

                if len(valid_rsi) >= self.divergence_lookback:
                    price_history = valid_close[-self.divergence_lookback:]
                    rsi_history = valid_rsi[-self.divergence_lookback:]

                    # 底背离检测
                    bullish_div = detect_bullish_divergence(
                        price_history,
                        rsi_history,
                        lookback=self.divergence_lookback,
                        order=self.divergence_order
                    )

                    if bullish_div['exists']:
                        divergence_strength = bullish_div['strength'] * time_decay(
                            bullish_div['days_ago'], valid_period=5
                        )
                        signal_bonus = divergence_strength * 0.5

            # === 3. 最终分数 = base + bonus ===
            final_score = base_score + signal_bonus

            return FactorResult(score=final_score, err=None)

        except Exception as e:
            return FactorResult(score=None, err=e)
