from .helpers import *
from core.database.money_flow import *


def _get_retail_ratio_series(code: str, base_time: datetime, days: int) -> Optional[np.ndarray]:
    """
    获取散户占比的历史序列

    Args:
        code: 股票代码
        base_time: 基准时间 (datetime)
        days: 获取天数

    Returns:
        散户占比序列 (从旧到新)，如果数据不足返回 None
    """
    from core.database import get_market_data_from_cache

    # 获取历史日线数据（用于成交额和日期对齐）
    history_data = get_market_data_from_cache(code, days, base_time, '1d')
    if history_data is None or len(history_data) < days:
        return None

    ratios = []
    for _, row in history_data.iterrows():
        # 从 time 列获取日期
        ts = int(row['time'])
        trade_date = datetime.fromtimestamp(ts / 1000).date()

        # 获取散户资金金额
        retail_amount = get_retail_flow_amount(code, trade_date)
        if retail_amount is None:
            return None  # 数据不完整

        total_amount = float(row['amount'])
        if total_amount <= 0:
            return None

        ratio = (retail_amount / total_amount) * 100
        ratios.append(ratio)

    return np.array(ratios)


class RetailFlowMomentum(BaseFactor):
    """
    散户资金流动量因子 (短期逆向版本)

    设计目标：捕捉 T+1 ~ T+5 的短期收益

    核心逻辑（逆向思维）：
    - 散户涌入 = 韭菜进场 = 卖出信号
    - 散户撤离 = 聪明钱建仓 = 买入信号

    计算方式：
    1. 主动量: -((today - MA5) / MA5 * 100)
       - 负号反转：散户流出时得分为正
    2. 加速度: 动量的变化率（同样取负）

    输出：
    - score: 综合得分，约 -50 ~ +50 范围
    - 正值 = 散户在撤离 = 买入信号
    - 负值 = 散户在涌入 = 卖出信号
    """

    def __init__(
        self,
        momentum_period: int = 5,      # 动量计算的 MA 周期（MA5）
        accel_period: int = 3,         # 加速度计算周期
        momentum_weight: float = 0.7,  # 动量权重
        accel_weight: float = 0.3,     # 加速度权重
        use_volume_weight: bool = False  # 是否使用成交量加权
    ):
        super().__init__()
        self.momentum_period = momentum_period
        self.accel_period = accel_period
        self.momentum_weight = momentum_weight
        self.accel_weight = accel_weight
        self.use_volume_weight = use_volume_weight

        # 需要的历史数据天数
        self._required_days = max(momentum_period, accel_period) + 5

    def calc(self, ctx: FactorCtx) -> FactorResult:
        # 1. 获取散户占比历史序列
        ratios = _get_retail_ratio_series(ctx.code, ctx.base_time, self._required_days)
        if ratios is None or len(ratios) < self._required_days:
            raise ValueError(f"散户占比历史数据不足: {ctx.code}，需要{self._required_days}天，实际{len(ratios) if ratios is not None else 0}天")

        # 2. 计算主动量：(today - MA) / MA * 100
        today_ratio = ratios[-1]
        ma = np.mean(ratios[-self.momentum_period:])

        if ma <= 0:
            raise ValueError(f"MA 异常: {ctx.code}，MA={ma}")

        momentum = ((today_ratio - ma) / ma) * 100

        # 3. 计算加速度：今日动量 - N日前动量
        # 先计算历史动量序列
        momentum_series = []
        for i in range(self.accel_period + 1):
            idx = -(i + 1)
            end_idx = idx if idx < -1 else None
            start_idx = idx - self.momentum_period + 1

            if abs(start_idx) > len(ratios):
                break

            period_data = ratios[start_idx:end_idx] if end_idx else ratios[start_idx:]
            period_ma = np.mean(period_data)
            if period_ma > 0:
                m = ((ratios[idx] - period_ma) / period_ma) * 100
                momentum_series.append(m)

        # 计算加速度（动量的变化）
        if len(momentum_series) >= 2:
            acceleration = momentum_series[0] - momentum_series[-1]
        else:
            acceleration = 0.0

        # 4. 组合得分（取负：散户流出 = 正分 = 买入信号）
        score = -(momentum * self.momentum_weight +
                  acceleration * self.accel_weight)

        # 5. 成交量加权（可选）
        if self.use_volume_weight:
            today_data = ctx.get_today_data()
            yesterday_data = ctx.get_yesterday_data()
            volume_ratio = float(today_data['volume']) / float(yesterday_data['volume'])
            # 放量时增强信号，缩量时削弱
            volume_factor = np.clip(volume_ratio, 0.5, 2.0)
            score *= volume_factor

        return FactorResult(score=score, err=None)


class RetailFlowAcceleration(BaseFactor):
    """
    散户资金加速度因子 (拐点捕捉版)

    专注于捕捉拐点：当散户从涌入转为撤离的那一刻

    核心逻辑（逆向）：
    - 散户加速撤离（负加速度）= 聪明钱加速建仓 = 买入信号
    - 取负后：加速度越正 = 买入信号越强

    输出：
    - score: 加速度值（已取负）
    - 正值越大 = 散户撤离加速 = 买入信号越强
    """

    def __init__(self, period: int = 3):
        """
        Args:
            period: 加速度计算周期，越小越敏感
        """
        super().__init__()
        self.period = period
        self._required_days = period + 3

    def calc(self, ctx: FactorCtx) -> FactorResult:
        ratios = _get_retail_ratio_series(ctx.code, ctx.base_time, self._required_days)
        if ratios is None or len(ratios) < self._required_days:
            raise ValueError(f"数据不足: {ctx.code}，需要{self._required_days}天，实际{len(ratios) if ratios is not None else 0}天")

        # 一阶导数（变化率）
        velocity = np.diff(ratios)

        # 二阶导数（加速度）
        acceleration = np.diff(velocity)

        # 取最近的加速度平均值
        recent_accel = np.mean(acceleration[-self.period:])

        # 取负并放大：散户撤离加速 = 正分
        score = -recent_accel * 10

        return FactorResult(score=score, err=None)


class RetailFlowRSI(BaseFactor):
    """
    散户资金 RSI 因子（逆向版本）

    类似于价格 RSI，但计算的是散户占比的相对强弱

    核心逻辑（逆向）：
    - RSI 低 = 散户撤离过度 = 买入信号
    - RSI 高 = 散户涌入过度 = 卖出信号
    - 输出: 100 - RSI，使得高分 = 买入信号

    输出：
    - score: 反转后的 RSI 值 (0-100)
    - 高分 = 散户撤离 = 买入信号
    """

    def __init__(self, period: int = 5):
        """
        Args:
            period: RSI 计算周期
        """
        super().__init__()
        self.period = period
        self._required_days = period + 5

    def calc(self, ctx: FactorCtx) -> FactorResult:
        ratios = _get_retail_ratio_series(ctx.code, ctx.base_time, self._required_days)
        if ratios is None or len(ratios) < self._required_days:
            raise ValueError(f"数据不足: {ctx.code}，需要{self._required_days}天，实际{len(ratios) if ratios is not None else 0}天")

        # 计算变化
        changes = np.diff(ratios)

        # 分离涨跌
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        # 计算平均涨跌
        avg_gain = np.mean(gains[-self.period:])
        avg_loss = np.mean(losses[-self.period:])

        # 计算 RSI
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # 反转：低 RSI = 高分 = 买入信号
        score = 100 - rsi

        return FactorResult(score=score, err=None)


class RetailFlowBreakout(BaseFactor):
    """
    散户资金突破因子（逆向版本）

    检测散户占比是否突破近期区间

    核心逻辑（逆向）：
    - 向下突破 = 散户突然大量撤离 = 买入信号
    - 向上突破 = 散户突然涌入 = 卖出信号
    - 取负后：正值 = 散户撤离 = 买入信号

    输出：
    - score: 突破强度 (-100 ~ +100)，已取负
    - 正值 = 散户向下突破（撤离）= 买入信号
    """

    def __init__(self, lookback: int = 10, threshold: float = 1.5):
        """
        Args:
            lookback: 回看周期
            threshold: 突破阈值（标准差倍数）
        """
        super().__init__()
        self.lookback = lookback
        self.threshold = threshold
        self._required_days = lookback + 5

    def calc(self, ctx: FactorCtx) -> FactorResult:
        ratios = _get_retail_ratio_series(ctx.code, ctx.base_time, self._required_days)
        if ratios is None or len(ratios) < self._required_days:
            raise ValueError(f"数据不足: {ctx.code}，需要{self._required_days}天，实际{len(ratios) if ratios is not None else 0}天")

        # 历史区间
        history = ratios[-(self.lookback + 1):-1]
        today = ratios[-1]

        mean = np.mean(history)
        std = np.std(history)

        if std <= 0:
            raise ValueError(f"标准差为0: {ctx.code}，无法计算Z-score")

        # 计算 Z-score
        z_score = (today - mean) / std

        # 转换为突破强度
        if abs(z_score) >= self.threshold:
            score = z_score * 20
        else:
            score = z_score * 5

        # 取负并限制范围：散户撤离突破 = 正分
        score = np.clip(-score, -100, 100)

        return FactorResult(score=score, err=None)
