"""
TRIX因子 - 加法混合设计

从qmt-trade项目迁移，输出原始连续分数。

金融逻辑:
- TRIX > 0: 三重平滑趋势向上 → 多头信号
- TRIX < 0: 三重平滑趋势向下 → 空头信号（买入因子应返回0）
- TRIX上穿0: 趋势由空转多 → 强买入信号

评分设计:
- base_score (50%): TRIX正值强度 (连续值，负值返回0)
- signal_bonus (50%): TRIX上穿零轴 (离散信号)

注意: 本因子输出原始分数，需要在框架层使用batch_norm进行归一化
"""

from .helpers import *
import numpy as np
import talib


def safe_sigmoid(x, clip_range=20):
    """安全的sigmoid函数，避免exp溢出"""
    x_clipped = np.clip(x, -clip_range, clip_range)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class TRIXFactor(BaseFactor):
    """
    TRIX因子（三重指数平滑平均线）

    评分逻辑:
    1. base_score (50%): TRIX正值强度 (连续值，负值返回0)
    2. signal_bonus (50%): TRIX上穿零轴 (离散信号)

    输出: 原始分数 [0, 1]，由框架层进行batch norm归一化
    """

    def __init__(self,
                 trix_period: int = 30,
                 cross_valid_days: int = 5,
                 trix_center: float = 0.005):
        """
        :param trix_period: TRIX计算周期，默认30天
        :param cross_valid_days: 金叉有效期，默认5天
        :param trix_center: sigmoid映射的中心点，默认0.005
        """
        super().__init__()
        self.trix_period = trix_period
        self.cross_valid_days = cross_valid_days
        self.trix_center = trix_center

    def calc(self, ctx: FactorCtx) -> FactorResult:
        try:
            # 获取TRIX指标
            trix_value = ctx.get_trix(period=self.trix_period)

            # === 1. 连续基础分 (50%) ===
            # 只有TRIX>0才有正向贡献
            if trix_value > 0:
                # 使用sigmoid平滑映射，避免硬阈值
                # trix_value范围通常在[-0.01, 0.01]
                # sigmoid中心设在0.005（中等强度）
                x = (trix_value - self.trix_center) / self.trix_center  # 标准化到sigmoid合理区间
                base_score = safe_sigmoid(x * 2) * 0.5  # sigmoid映射到[0, 0.5]
            else:
                base_score = 0.0  # 下跌趋势不给基础分

            # === 2. 信号加成 (50%) ===
            signal_bonus = 0.0

            # TRIX上穿零轴检测
            history_data = ctx.get_daily_data(self.cross_valid_days + self.trix_period * 3 + 1)

            if len(history_data) >= self.cross_valid_days + self.trix_period * 3 + 1:
                close = np.array(history_data['close'].values, dtype=np.float64)
                trix_series = talib.TRIX(close, timeperiod=self.trix_period)

                # 提取最近的有效数据（去掉NaN）
                valid_mask = ~np.isnan(trix_series)
                valid_trix = trix_series[valid_mask]

                if len(valid_trix) >= self.cross_valid_days + 1:
                    trix_history = valid_trix[-(self.cross_valid_days + 1):]
                    zero_line = np.zeros_like(trix_history)

                    golden_cross = detect_golden_cross(
                        trix_history,
                        zero_line,
                        lookback=self.cross_valid_days
                    )

                    if golden_cross['exists']:
                        # 使用时间衰减，不依赖TRIX当前值（因为刚穿过时TRIX接近0）
                        time_factor = time_decay(golden_cross['days_ago'], valid_period=self.cross_valid_days)
                        signal_bonus = time_factor * 0.5

            # === 3. 最终分数 = base + bonus ===
            final_score = base_score + signal_bonus

            return FactorResult(score=final_score, err=None)

        except Exception as e:
            return FactorResult(score=None, err=e)
