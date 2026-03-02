"""
WMA金叉因子 - WMA5/WMA20 价差连续评分

理论基础:
  动量效应 (Jegadeesh & Titman, 1993): 短期均线偏离中期均线的幅度
  反映个股动量强度，与未来短期收益正相关。

设计思路:
  使用纯连续函数，不依赖离散事件（金叉/死叉），保证分数平滑。
  分数可以为负——负分对应预期收益为负的空头趋势。

  两个分量:

  1. 价差水平 spread_pct = (WMA_fast - WMA_slow) / |WMA_slow|
     - 反映当前趋势方向和强度
     - 正值 = 多头趋势（短期强于中期）→ 预期正收益
     - 负值 = 空头趋势（短期弱于中期）→ 预期负收益

  2. 价差动量 spread_momentum = spread_pct(today) - spread_pct(N日前)
     - 反映趋势的加速/减速
     - 正值 = 动能增强（价差在扩大 或 从负值收窄）
     - 负值 = 动能减弱（价差在收窄 或 从正值下降）

  最终分数 = spread_pct + momentum_weight × spread_momentum

  关于金叉事件:
    金叉本质是 spread_pct 从负转正。此信息已完全被 spread_pct
    和 spread_momentum 两个连续分量捕获:
    - 金叉时刻: spread_pct ≈ 0 且 spread_momentum 处于正向峰值
    - 无需额外的离散事件检测
    - 去掉事件信号避免了在金叉时刻产生不自然的分数跳变

  分数行为示例 (假设 momentum_weight = 1.0):
    金叉时刻:   spread_pct ≈ 0,    momentum > 0  → 正分 (动量驱动)
    多头扩张:   spread_pct > 0,    momentum > 0  → 高正分
    多头收窄:   spread_pct > 0,    momentum < 0  → 正分但在降低
    死叉时刻:   spread_pct ≈ 0,    momentum < 0  → 负分
    空头扩张:   spread_pct < 0,    momentum < 0  → 高负分
    空头收窄:   spread_pct < 0,    momentum > 0  → 负分但在回升
"""

from .helpers import *

class WMACross(BaseFactor):
  """
  WMA金叉因子

  基于WMA5与WMA20的价差及其变化率进行连续评分。
  分数可以为负，负分表示预期T+N日收益为负。

  不使用sigmoid/事件等非线性变换，保持分数与价差的线性关系，
  确保分数分布平滑且与未来收益的相关性最大化。
  最终排序由框架层 batch_normalize (Rank归一化 + 温度) 完成。
  """

  def __init__(self,
               fast_period: int = 5,
               slow_period: int = 20,
               momentum_lookback: int = 5,
               momentum_weight: float = 1.0):
    """
    :param fast_period: 快线WMA周期（默认5日）
    :param slow_period: 慢线WMA周期（默认20日）
    :param momentum_lookback: 计算价差动量的回溯天数（默认5日）
    :param momentum_weight: 价差动量相对于价差水平的权重（默认1.0，即等权）
    """
    super().__init__()
    self.fast_period = fast_period
    self.slow_period = slow_period
    self.momentum_lookback = momentum_lookback
    self.momentum_weight = momentum_weight

  def calc(self, ctx: FactorCtx) -> FactorResult:
    try:
      # ----------------------------------------------------------------
      # 数据准备: 一次性获取足够天数的日线数据
      # slow_period 天用于计算 WMA_slow 的 rolling window
      # momentum_lookback 天用于计算价差动量
      # +2 作为边界缓冲
      # ----------------------------------------------------------------
      days_needed = self.slow_period + self.momentum_lookback + 2
      history_data = ctx.get_daily_data(days_needed)

      if history_data is None or len(history_data) < self.slow_period + 2:
        return FactorResult(score=None, err=None)

      # ----------------------------------------------------------------
      # 计算两条WMA（与 ctx.get_wma() 相同算法）
      #
      # WMA 定义: 成交额加权典型价格均线
      #   typical_price = (Open + High + Low + Close) / 4
      #   WMA(period) = rolling_mean(typical_price × amount) / rolling_mean(amount)
      #
      # 之所以不直接调用 ctx.get_wma()，是因为 get_wma(period) 内部
      # 只获取 period*2 天的数据，导致 WMA5 只有 ~6 个有效点，
      # 不足以计算 momentum_lookback 天的动量。
      # 这里从同一份数据计算两条WMA，保证时间对齐。
      # ----------------------------------------------------------------
      typical_price = (history_data['open'] + history_data['close'] +
                       history_data['low'] + history_data['high']) / 4
      vwtp = typical_price * history_data['amount']

      wma_fast = (vwtp.rolling(window=self.fast_period).mean() /
                  history_data['amount'].rolling(window=self.fast_period).mean())
      wma_slow = (vwtp.rolling(window=self.slow_period).mean() /
                  history_data['amount'].rolling(window=self.slow_period).mean())

      # ----------------------------------------------------------------
      # 对齐: 只保留两条WMA都有效（非NaN）的部分
      # WMA_slow 的前 slow_period-1 个点是 NaN（rolling 不足窗口）
      # 对齐后序列从 WMA_slow 第一个有效点开始
      # ----------------------------------------------------------------
      valid_mask = wma_fast.notna() & wma_slow.notna()
      wma_fast_valid = wma_fast[valid_mask]
      wma_slow_valid = wma_slow[valid_mask]

      # 需要至少 momentum_lookback + 1 个有效点（当前 + 回溯）
      if len(wma_fast_valid) < self.momentum_lookback + 1:
        return FactorResult(score=None, err=None)

      # ----------------------------------------------------------------
      # 分量1: 价差水平 (spread_pct)
      #
      # spread_pct = (WMA_fast - WMA_slow) / |WMA_slow|
      #
      # 物理含义: 短期均线相对中期均线的偏离百分比
      # - 正值 → 短期趋势强于中期 → 多头
      # - 负值 → 短期趋势弱于中期 → 空头
      # - 绝对值大小 → 趋势强度
      #
      # 典型范围: [-5%, +5%]（即 [-0.05, +0.05]）
      # ----------------------------------------------------------------
      current_fast = wma_fast_valid.iloc[-1]
      current_slow = wma_slow_valid.iloc[-1]
      spread_pct = (current_fast - current_slow) / (abs(current_slow) + 1e-8)

      # ----------------------------------------------------------------
      # 分量2: 价差动量 (spread_momentum)
      #
      # spread_momentum = spread_pct(today) - spread_pct(N日前)
      #
      # 物理含义: 价差的变化方向和速度
      # - 正值 → 价差在改善（趋势增强 或 空头减弱正在收窄）
      # - 负值 → 价差在恶化（趋势减弱 或 空头加强）
      # - 零值 → 价差稳定不变
      #
      # 这个分量让因子具备前瞻性:
      #   即使当前处于空头 (spread_pct < 0)，如果价差在收窄
      #   (spread_momentum > 0)，分数会提前回升，
      #   而不用等到金叉才开始给正分。
      # ----------------------------------------------------------------
      lookback = min(self.momentum_lookback, len(wma_fast_valid) - 1)
      prev_fast = wma_fast_valid.iloc[-(lookback + 1)]
      prev_slow = wma_slow_valid.iloc[-(lookback + 1)]
      prev_spread_pct = (prev_fast - prev_slow) / (abs(prev_slow) + 1e-8)
      spread_momentum = spread_pct - prev_spread_pct

      # ----------------------------------------------------------------
      # 最终分数
      #
      # score = spread_pct + momentum_weight × spread_momentum
      #
      # 使用线性组合而非 sigmoid 等非线性变换，理由:
      # 1. 动量研究表明因子值与收益率在横截面上近似线性关系
      # 2. 线性保持了原始信号的分布特性，有利于后续 batch_normalize
      # 3. 框架层的 Rank归一化 + 温度参数 已经处理了非线性映射
      #
      # momentum_weight 的作用:
      # - = 0: 纯价差水平因子（只看趋势方向和强度）
      # - = 1: 等权混合（兼顾水平和变化率）
      # - > 1: 偏重动量变化（更灵敏但也更波动）
      # 该参数可通过 GA 优化搜索最优值
      # ----------------------------------------------------------------
      final_score = spread_pct + self.momentum_weight * spread_momentum

      return FactorResult(score=final_score, err=None)

    except Exception as e:
      return FactorResult(score=None, err=e)
