"""
WMA短周期均值回归因子 - 2/28成交额加权均线偏离 + 1日偏离扩张确认 + 极轻散户追涨杀跌放大

设计目标:
  1. 面向全市场日频打分 + TopN 选股场景。
     这类场景最重要的不是拟合单只股票的价格路径，而是让横截面排序与
     T+N收益率保持稳定正相关。因此因子需要同时满足两点：
     - 短周期机会足够敏感，能尽快识别超买/超卖后的反转；
     - 分数变化不能过于跳跃，避免轻微噪声导致TopN频繁洗牌。

  2. 使用更短的 2/28 WMA 结构。
     该组合比传统 5/20 更偏短线，能更快捕捉短期拥挤和超调，实测在当前
     step=5 样本 benchmark 上，对 T+3/T+5/T+10 的相关性更高。

理论依据:
  1. A股短期反转效应:
     散户主导市场中，极短期涨跌常常过度，随后在未来数日内均值回归。
     因此，快线显著偏离慢线时，更适合做反向打分：
     - 快线高于慢线很多 -> 超买 -> 负分
     - 快线低于慢线很多 -> 超卖 -> 正分

  2. 偏离扩张比偏离本身更重要:
     仅有“已经超买/超卖”还不够；如果偏离仍在继续扩大，通常代表情绪与
     拥挤交易还在发酵，后续反转空间更大。因此本因子对“偏离继续扩张”给
     额外奖励，而对“偏离开始修复”只做缓慢减弱，不让分数大幅抖动。

  3. 散户追涨杀跌会放大超调:
     若价格已经偏离，而散户净流向又与偏离方向一致，则说明偏离更可能由
     噪声交易推动，未来均值回归概率更高；反之若散户在逆向交易，则减弱
     该反转信号。

当前实现的三层结构:
  1. spread_pct:
     spread_pct = (WMA_fast - WMA_slow) / |WMA_slow|
     它衡量短周期价格相对中周期价格的“偏离程度”，是整个因子的主信号。

  2. momentum adjustment:
     spread_momentum = spread_pct(today) - spread_pct(N日前)
     再将动量转成 signed_extension = sign(spread_pct) * spread_momentum：
     - signed_extension > 0: 偏离仍在继续扩大 -> 增强反转分数
     - signed_extension < 0: 偏离开始回归 -> 只缓慢减弱反转分数

  3. retail amplifier:
     interaction = spread_pct * avg_retail_net_pct
     - interaction > 0: 散户在追涨/杀跌 -> 放大均值回归信号
     - interaction < 0: 散户在逆向操作 -> 减弱均值回归信号

最终:
  final_score = base_score * amplifier

解释方向:
  - 分数 > 0: 更偏向未来几日反弹
  - 分数 < 0: 更偏向未来几日回调
  - |分数| 越大: 该方向的把握越强

散户数据缺失时:
  amplifier = 1.0，自动退化为纯价格型 WMA 均值回归因子。
"""

from .helpers import *
from core.database.money_flow import get_retail_net_flow
from datetime import datetime
import numpy as np

# 交互项的归一化尺度。
#
# interaction = spread_pct * avg_retail_net_pct
# - spread_pct 在当前 2/28 结构下，常见量级大约在 ±1% ~ ±3%
# - retail_net_pct 常见量级大约在 ±0.5% ~ ±1%
# - 因此二者乘积通常在 1e-4 ~ 1e-3 量级
#
# 用 0.001 做归一化后，norm_interaction 大致落在 [-1, 1] 附近，
# contrarian_weight 就可以直接表达“散户信号影响 base_score 的强度”。
_INTERACTION_SCALE = 0.001

# 价差动量缩放尺度。
#
# extension_bonus / reversion_drag 在短周期下通常很小，若直接线性使用，
# 少数极端样本会导致分数骤变。这里用 tanh(x / _MOMENTUM_SCALE) 做饱和：
# - 小变化仍保留方向信息
# - 大变化逐步饱和，避免单日异常把排序拉爆
_MOMENTUM_SCALE = 0.0058

# spread_pct -> spread_strength 的缩放尺度。
#
# spread_strength = tanh(abs(spread_pct) / _GATE_SCALE)
# 它不是主信号，而是一个“强弱门控器”：
# - 偏离很小: spread_strength 接近 0，说明只是轻微信号
# - 偏离够大: spread_strength 接近 1，说明已进入更可信的超调区间
_GATE_SCALE = 0.014

# 基础分门控权重。
#
# base_score 先由 -spread_pct 给出主方向，再乘上：
#   (1 - _BASE_GATE + _BASE_GATE * spread_strength)
# 含义是：
# - 弱偏离时，对原始 spread_pct 只做非常轻度折扣
# - 强偏离时，几乎保持原始强度
#
# 之所以只给 0.02，是因为 benchmark 显示：TopN 场景里如果把弱信号压得
# 过重，会损失横截面排序区分度，反而降低 T+N 收益率相关性。
_BASE_GATE = 0.00

# 散户放大器门控权重。
#
# interaction 先用 spread_strength 轻微门控，再进入 amplifier。
# 这表示：当价格本身偏离并不明显时，不希望散户单日流向过度主导因子；
# 只有在价格已经进入偏离区后，散户行为才应被更多地视作确认信号。
_RETAIL_GATE = 0.00

# “偏离继续扩大”时，对 base_score 的增强系数。
#
# 数值越大，因子越偏向追踪极短期拥挤反转；
# 数值过大则会让分数对单日变动过于敏感。
_BONUS_MULT = 0.96

# “偏离开始修复”时，对 base_score 的减弱速度。
#
# 这里刻意设得较小：
# - 一旦偏离停止扩大，不代表马上失效
# - TopN 场景里，更希望分数平缓衰减，而不是一天内剧烈翻转
_DRAG_MULT = 0.02

# base_score 的最小保留比例。
#
# 当 reversion_drag 很大时，减弱也不能无限向下压，否则会因为短暂修复就
# 让分数塌陷。0.986 表示即使进入“开始修复”阶段，主信号仍保留至少 98.6%。
_DRAG_FLOOR = 0.986


class WMACross(BaseFactor):
  """
  WMA短周期均值回归因子（含偏离扩张确认与散户放大）

  因子计算顺序:
  1. 用 2/28 成交额加权典型价格均线计算 spread_pct
  2. 用 1 日前的 spread_pct 计算 spread_momentum
  3. 将 momentum 拆成:
     - extension_bonus: 偏离继续扩大
     - reversion_drag: 偏离开始修复
  4. 用散户净流向作为乘法放大器做方向确认

  响应逻辑:
  - 快线高于慢线很多 -> 因子给负分 -> 预期 T+N 更易回调
  - 快线低于慢线很多 -> 因子给正分 -> 预期 T+N 更易反弹
  - 若偏离仍在扩大 -> 强化该判断
  - 若散户还在追涨/杀跌 -> 再进一步强化该判断

  最终排序由框架层 batch_normalize (Rank归一化 + 温度) 完成。
  """

  def __init__(self,
               fast_period: int = 2,
               slow_period: int = 28,
               momentum_lookback: int = 1,
               momentum_weight: float = 1.0,
               contrarian_weight: float = 0.03,
               retail_smooth_days: int = 1):
    """
    :param fast_period: 快线WMA周期（默认2日）
    :param slow_period: 慢线WMA周期（默认28日）
    :param momentum_lookback: 计算价差动量的回溯天数（默认1日）
    :param momentum_weight: 价差动量权重（默认1.0）
    :param contrarian_weight: 散户追涨杀跌放大系数（默认0.03）
        - 0: 禁用散户信号，退化为纯WMA因子
        - 0.03: 极轻放大（默认，优先保证横截面稳定性）
        - 0.1: 轻度放大（更偏激进）
        可通过 GA 优化搜索最优值
    :param retail_smooth_days: 散户净流向平滑天数（默认1日，更强调信号新鲜度）
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

      # spread_sign 只负责记录当前偏离方向：
      # +1 表示超买，-1 表示超卖，0 表示几乎无偏离
      spread_sign = float(np.sign(spread_pct))

      # ----------------------------------------------------------------
      # 基础分: 保留 spread_pct 的原始横截面区分度，避免弱信号被压平。
      #
      # 对 TopN 选股来说，只要没有明显劣化，就不希望分数因为轻微波动
      # 被大幅改写。因此这里只让动量做非对称调节，而不是直接线性叠加。
      # ----------------------------------------------------------------
      # 第一层: 以 spread_pct 作为主信号。
      # 这里保留原始横截面区分度，不直接把 spread 和 momentum 线性相加，
      # 目的是减少因子分数被短期噪声来回拉扯。
      base_score = -spread_pct

      # ----------------------------------------------------------------
      # 动量只在"偏离继续扩大"时明显增强；若偏离开始回归，只缓慢减弱。
      # 这样可以保留短周期反转的机会，同时减少排序剧烈翻动。
      # ----------------------------------------------------------------
      # signed_extension 把“动量方向”映射到“是否继续偏离”：
      # - 超买且 spread 继续走高 -> signed_extension > 0
      # - 超卖且 spread 继续走低 -> signed_extension > 0
      # 上述两种都说明价格在继续远离均值，后续反转空间更大。
      signed_extension = spread_sign * spread_momentum
      extension_bonus = max(0.0, signed_extension)
      reversion_drag = max(0.0, -signed_extension)

      if base_score != 0.0:
        # bonus_scale / drag_scale 都通过 tanh 限制到 [0, 1) 区间。
        # 这样可以把“继续扩大/开始修复”的信息稳定地映射为乘法调节，
        # 避免个别极端股票在截面排序中产生异常跳跃。
        bonus_scale = np.tanh(extension_bonus / _MOMENTUM_SCALE) if extension_bonus > 0 else 0.0
        drag_scale = np.tanh(reversion_drag / _MOMENTUM_SCALE) if reversion_drag > 0 else 0.0

        # spread_strength 衡量当前偏离是否已经足够大。
        # 它不会改变分数方向，只决定下面的增强/减弱有多明显。
        spread_strength = np.tanh(abs(spread_pct) / _GATE_SCALE)

        # 先做极轻的基础门控。
        # 理由: benchmark 表明完全不门控会略增噪声，但门控太强会损失截面
        # 区分度，因此这里仅做很小的平滑处理。
        base_score *= (1.0 - _BASE_GATE + _BASE_GATE * spread_strength)

        # 偏离继续扩大 -> 增强反转分数。
        base_score *= (1.0 + _BONUS_MULT * self.momentum_weight * bonus_scale)

        # 偏离开始修复 -> 缓慢减弱，而不是直接打折过猛。
        # 这样可以让 TopN 排名更稳定，更贴近“除非确认劣化否则不要剧烈变化”。
        base_score *= max(_DRAG_FLOOR, 1.0 - _DRAG_MULT * self.momentum_weight * drag_scale)

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

          # interaction > 0:
          #   超买时散户净买入，或超卖时散户净卖出
          #   -> 说明散户在追涨/杀跌 -> 偏离更可能是情绪推动 -> 反转更强
          # interaction < 0:
          #   散户在逆向交易 -> 说明当前偏离未必是纯噪声 -> 反转信号减弱
          interaction = spread_pct * avg_retail_net_pct

          # 散户项也只做轻度门控，避免在弱偏离状态下被资金流单独主导。
          interaction *= (1.0 - _RETAIL_GATE + _RETAIL_GATE * np.tanh(abs(spread_pct) / _GATE_SCALE))
          norm_interaction = interaction / _INTERACTION_SCALE

          # amplifier 是最终的方向确认层：
          # - > 1: 放大 base_score
          # - < 1: 衰减 base_score
          # 下限 0.1 避免极端情况下把信号完全压没。
          amplifier = max(0.1, 1.0 + self.contrarian_weight * norm_interaction)

      # final_score 是最终原始因子值，后续还会进入框架层归一化排序。
      final_score = base_score * amplifier

      return FactorResult(score=final_score, err=None)

    except Exception as e:
      return FactorResult(score=None, err=e)
