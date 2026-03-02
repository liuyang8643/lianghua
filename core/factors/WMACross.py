"""
WMA均值回归因子 - WMA5/WMA20 价差连续评分 + 散户追涨杀跌放大

理论基础:
  1. A股短期反转效应 (Jegadeesh, 1990):
     在散户主导的A股市场中，短期价格动量呈现反转而非延续。
     WMA5远高于WMA20 = 短期超买 → 后续回调；反之超卖 → 后续反弹。

  2. 噪声交易者放大效应 (De Long et al., 1990):
     散户（噪声交易者）的追涨杀跌行为放大了价格偏离基本面的程度。
     当散户净买入方向与价格超买/超卖方向一致时（追涨或杀跌），
     说明价格偏离由散户推动，均值回归概率更高、幅度更大。

评分设计:
  三层结构:

  1. base_score = -(spread_pct + momentum_weight × spread_momentum)
     纯均值回归信号，超卖→正分，超买→负分。

  2. 散户追涨杀跌放大器 (multiplicative amplifier):
     interaction = spread_pct × avg_retail_net_pct
     - 正值 = 散户在追涨(超买时净买入)或杀跌(超卖时净卖出)
       → 价格偏离由散户推动 → 放大均值回归信号
     - 负值 = 散户在做"对的事"(超买时卖出/超卖时买入)
       → 价格偏离可能有基本面支撑 → 减弱均值回归信号

     amplifier = clip(1 + contrarian_weight × interaction / SCALE, 0.1, ∞)

  3. final_score = base_score × amplifier

  四象限行为:
    超买 + 散户净买入: base<0, amplifier>1 → 更负 (强回调预期)
    超买 + 散户净卖出: base<0, amplifier<1 → 较少负 (弱回调,机构可能在撑)
    超卖 + 散户净卖出: base>0, amplifier>1 → 更正 (强反弹预期)
    超卖 + 散户净买入: base>0, amplifier<1 → 较少正 (弱反弹,散户在接盘)

  散户数据缺失时: amplifier=1.0，退化为纯WMA均值回归因子。
"""

from .helpers import *
from core.database.money_flow import get_retail_net_flow
from datetime import datetime
import numpy as np

# 交互项的归一化尺度
# spread_pct 典型范围 ±0.02, retail_net_pct 典型范围 ±0.01
# 乘积典型范围 ±0.0002, 最大约 ±0.001
# 除以 INTERACTION_SCALE 后变为 ±1 量级，方便 contrarian_weight 控制
_INTERACTION_SCALE = 0.001


class WMACross(BaseFactor):
  """
  WMA均值回归因子（含散户追涨杀跌放大）

  基于WMA5与WMA20价差的反向评分，结合散户净流向方向性确认。
  - 纯价格信号: 超买→负分，超卖→正分（均值回归）
  - 散户放大器: 散户追涨杀跌时放大信号，散户"做对"时减弱信号
  - 散户数据缺失时自动退化为纯WMA因子

  最终排序由框架层 batch_normalize (Rank归一化 + 温度) 完成。
  """

  def __init__(self,
               fast_period: int = 5,
               slow_period: int = 20,
               momentum_lookback: int = 5,
               momentum_weight: float = 1.0,
               contrarian_weight: float = 0.5,
               retail_smooth_days: int = 3):
    """
    :param fast_period: 快线WMA周期（默认5日）
    :param slow_period: 慢线WMA周期（默认20日）
    :param momentum_lookback: 计算价差动量的回溯天数（默认5日）
    :param momentum_weight: 价差动量权重（默认1.0）
    :param contrarian_weight: 散户追涨杀跌放大系数（默认0.5）
        - 0: 禁用散户信号，退化为纯WMA因子
        - 0.5: 适度放大（默认，典型放大幅度 ±15%~30%）
        - 1.0: 激进放大（典型放大幅度 ±30%~50%）
        可通过 GA 优化搜索最优值
    :param retail_smooth_days: 散户净流向平滑天数（默认3日，减少单日噪声）
    """
    super().__init__()
    self.fast_period = fast_period
    self.slow_period = slow_period
    self.momentum_lookback = momentum_lookback
    self.momentum_weight = momentum_weight
    self.contrarian_weight = contrarian_weight
    self.retail_smooth_days = retail_smooth_days

  def calc(self, ctx: FactorCtx) -> FactorResult:
    try:
      # ----------------------------------------------------------------
      # 数据准备: 一次性获取足够天数的日线数据
      # slow_period 天用于 WMA_slow 的 rolling window
      # momentum_lookback 天用于价差动量
      # retail_smooth_days 天用于散户净流向平滑
      # +2 作为边界缓冲
      # ----------------------------------------------------------------
      extra_days = max(self.momentum_lookback, self.retail_smooth_days)
      days_needed = self.slow_period + extra_days + 2
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
      # 只获取 period*2 天的数据，WMA5只有~6个有效点，不足以回溯。
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
      # ----------------------------------------------------------------
      valid_mask = wma_fast.notna() & wma_slow.notna()
      wma_fast_valid = wma_fast[valid_mask]
      wma_slow_valid = wma_slow[valid_mask]

      if len(wma_fast_valid) < self.momentum_lookback + 1:
        return FactorResult(score=None, err=None)

      # ----------------------------------------------------------------
      # 分量1: 价差水平 (spread_pct)
      #
      # spread_pct = (WMA_fast - WMA_slow) / |WMA_slow|
      # - 正值 → 短期超买
      # - 负值 → 短期超卖
      # 典型范围: [-5%, +5%]
      # ----------------------------------------------------------------
      current_fast = wma_fast_valid.iloc[-1]
      current_slow = wma_slow_valid.iloc[-1]
      spread_pct = (current_fast - current_slow) / (abs(current_slow) + 1e-8)

      # ----------------------------------------------------------------
      # 分量2: 价差动量 (spread_momentum)
      #
      # spread_momentum = spread_pct(today) - spread_pct(N日前)
      # - 正值 → 超买加深 或 超卖减轻
      # - 负值 → 超买减轻 或 超卖加深
      # ----------------------------------------------------------------
      lookback = min(self.momentum_lookback, len(wma_fast_valid) - 1)
      prev_fast = wma_fast_valid.iloc[-(lookback + 1)]
      prev_slow = wma_slow_valid.iloc[-(lookback + 1)]
      prev_spread_pct = (prev_fast - prev_slow) / (abs(prev_slow) + 1e-8)
      spread_momentum = spread_pct - prev_spread_pct

      # ----------------------------------------------------------------
      # 基础分: 取反（均值回归）
      # 超卖 → 正分（预期反弹），超买 → 负分（预期回调）
      # ----------------------------------------------------------------
      base_score = -(spread_pct + self.momentum_weight * spread_momentum)

      # ----------------------------------------------------------------
      # 分量3: 散户追涨杀跌放大器 (contrarian amplifier)
      #
      # 从 money_flow 数据获取最近 N 日的散户净流向:
      #   retail_net_pct = (小单买入 - 小单卖出) / 当日总成交额
      #   正值 = 散户净买入，负值 = 散户净卖出
      #   取多日平均以减少单日噪声
      #
      # 交互项 interaction = spread_pct × avg_retail_net_pct:
      #   正值 = 散户追涨杀跌（与超买/超卖同方向）
      #     - 超买(+) × 散户净买入(+) = (+) → 散户FOMO追涨
      #     - 超卖(-) × 散户净卖出(-) = (+) → 散户恐慌杀跌
      #   负值 = 散户"做对了"（逆向操作）
      #     - 超买(+) × 散户净卖出(-) = (-) → 散户理性卖出
      #     - 超卖(-) × 散户净买入(+) = (-) → 散户抄底
      #
      # 放大器 = 1 + contrarian_weight × 归一化交互项
      #   > 1: 散户追涨杀跌 → 放大基础分（强化均值回归信号）
      #   < 1: 散户逆向操作 → 衰减基础分（弱化均值回归信号）
      #   = 1: 无散户数据 → 不影响（退化为纯WMA因子）
      #
      # 乘法效果:
      #   base < 0（超买）× amplifier > 1 → 更负 → 更强回调预期 ✓
      #   base > 0（超卖）× amplifier > 1 → 更正 → 更强反弹预期 ✓
      #   base < 0（超买）× amplifier < 1 → 减弱负值 → 较弱回调预期 ✓
      #   base > 0（超卖）× amplifier < 1 → 减弱正值 → 较弱反弹预期 ✓
      # ----------------------------------------------------------------
      amplifier = 1.0  # 默认: 无散户数据时不放大

      if self.contrarian_weight > 0:
        retail_net_pcts = []
        # 遍历最近 retail_smooth_days 天，收集散户净流向占比
        for i in range(self.retail_smooth_days):
          idx = -(i + 1)
          if abs(idx) > len(history_data):
            break
          row = history_data.iloc[idx]
          total_amount = float(row['amount'])
          if total_amount <= 0:
            continue
          # 从 K 线 time 列获取交易日期
          ts = int(row['time'])
          trade_date = datetime.fromtimestamp(ts / 1000).date()
          retail_net = get_retail_net_flow(ctx.code, trade_date)
          if retail_net is not None:
            retail_net_pcts.append(retail_net / total_amount)

        if retail_net_pcts:
          avg_retail_net_pct = np.mean(retail_net_pcts)
          # 交互项: 正 = 追涨杀跌，负 = 逆向操作
          interaction = spread_pct * avg_retail_net_pct
          # 归一化到 ±1 量级后乘以权重
          norm_interaction = interaction / _INTERACTION_SCALE
          # 下限 0.1 防止符号翻转（不应将看涨变看跌）
          amplifier = max(0.1, 1.0 + self.contrarian_weight * norm_interaction)

      final_score = base_score * amplifier

      return FactorResult(score=final_score, err=None)

    except Exception as e:
      return FactorResult(score=None, err=e)
