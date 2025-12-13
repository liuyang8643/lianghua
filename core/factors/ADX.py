"""
ADX因子 - 加法混合设计

从qmt-trade项目迁移，输出原始连续分数。

金融逻辑:
- ADX衡量趋势强度(不分方向), +DI/-DI衡量方向
- 强上涨趋势: 高分 (ADX高 + plus_di>minus_di)
- 震荡无趋势: 中等分 (ADX低或DI差值小)
- 强下跌趋势: 低分但不为0 (超跌反弹机会, ADX高 + minus_di>plus_di)

评分设计:
- base_score (50%): ADX趋势强度 × DI方向 (连续值)
- signal_bonus (50%): +DI上穿-DI金叉 (离散信号)

注意: 本因子输出原始分数，需要在框架层使用batch_norm进行归一化
"""

from .helpers import *
import numpy as np


def safe_sigmoid(x, clip_range=20):
    """安全的sigmoid函数，避免exp溢出"""
    x_clipped = np.clip(x, -clip_range, clip_range)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class ADXFactor(BaseFactor):
    """
    ADX因子

    评分逻辑:
    1. base_score (50%): ADX趋势强度 × DI方向分数
    2. signal_bonus (50%): +DI上穿-DI金叉

    输出: 原始分数 [0, 1]，由框架层进行batch norm归一化
    """

    def __init__(self,
                 adx_period: int = 14,
                 cross_valid_days: int = 5):
        """
        :param adx_period: ADX计算周期，默认14天
        :param cross_valid_days: 金叉有效期，默认5天
        """
        super().__init__()
        self.adx_period = adx_period
        self.cross_valid_days = cross_valid_days

    @cached_factor('ADXFactor')
    def calc(self, ctx: FactorCtx) -> FactorResult:
        try:
            # 获取ADX相关指标
            adx, plus_di, minus_di = ctx.get_adx(period=self.adx_period)

            # === 1. 连续基础分 (50%) ===
            # 方向评分: (plus_di - minus_di) / 100 ∈ [-1, 1]
            # 正值=上涨趋势, 负值=下跌趋势, 0=震荡
            direction_raw = (plus_di - minus_di) / 100.0

            # sigmoid映射到[0, 1]: 上涨高分, 震荡中等, 下跌低分但不为0
            # direction_raw = 0.5 → sigmoid ≈ 0.92 (强上涨)
            # direction_raw = 0   → sigmoid = 0.50 (震荡)
            # direction_raw = -0.5 → sigmoid ≈ 0.08 (强下跌, 仍有超跌反弹价值)
            direction_normalized = safe_sigmoid(direction_raw * 5)

            # ADX趋势强度 × 方向分数
            adx_strength = adx / 100.0
            base_score = adx_strength * direction_normalized * 0.5  # 基础分占50%

            # === 2. 信号加成 (50%) ===
            signal_bonus = 0.0

            # +DI上穿-DI金叉检测
            history_data = ctx.get_daily_data(self.cross_valid_days + self.adx_period * 2)

            if len(history_data) >= self.cross_valid_days + self.adx_period * 2:
                high = np.array(history_data['high'].values, dtype=np.float64)
                low = np.array(history_data['low'].values, dtype=np.float64)
                close = np.array(history_data['close'].values, dtype=np.float64)

                import talib
                plus_di_series = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
                minus_di_series = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)

                # 提取最近的有效数据（去掉NaN）
                valid_mask = ~(np.isnan(plus_di_series) | np.isnan(minus_di_series))
                valid_plus_di = plus_di_series[valid_mask]
                valid_minus_di = minus_di_series[valid_mask]

                if len(valid_plus_di) >= self.cross_valid_days + 1:
                    plus_di_history = valid_plus_di[-(self.cross_valid_days + 1):]
                    minus_di_history = valid_minus_di[-(self.cross_valid_days + 1):]

                    di_cross = detect_golden_cross(
                        plus_di_history,
                        minus_di_history,
                        lookback=self.cross_valid_days
                    )

                    if di_cross['exists']:
                        cross_strength = time_decay(di_cross['days_ago'], valid_period=self.cross_valid_days)
                        signal_bonus = cross_strength * 0.5

            # === 3. 最终分数 = base + bonus ===
            final_score = base_score + signal_bonus

            return FactorResult(score=final_score, err=None)

        except Exception as e:
            return FactorResult(score=None, err=e)
