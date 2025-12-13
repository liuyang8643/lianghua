"""
信号检测通用工具函数

从qmt-trade项目迁移，用于检测技术分析信号。

功能:
1. 背离检测(底背离/顶背离)
2. 金叉/死叉检测
3. 时效衰减函数
4. 局部极值点查找
5. SAR翻转检测
6. 价格突破检测
"""

import numpy as np
from typing import Dict, List
from scipy.signal import argrelextrema


def find_local_minima(series: np.ndarray, order: int = 3) -> List[Dict]:
    """
    查找局部最小值点

    Args:
        series: 数据序列
        order: 局部窗口大小(前后order个点都比它大才算极小值)

    Returns:
        [{'index': int, 'value': float}, ...]
    """
    if len(series) < order * 2 + 1:
        raise ValueError(f"序列长度{len(series)}不足，需要至少{order * 2 + 1}个点（order={order}）")

    # 使用scipy查找局部极小值
    minima_indices = argrelextrema(series, np.less_equal, order=order)[0]

    return [
        {'index': int(idx), 'value': float(series[idx])}
        for idx in minima_indices
    ]


def find_local_maxima(series: np.ndarray, order: int = 3) -> List[Dict]:
    """
    查找局部最大值点

    Args:
        series: 数据序列
        order: 局部窗口大小

    Returns:
        [{'index': int, 'value': float}, ...]
    """
    if len(series) < order * 2 + 1:
        raise ValueError(f"序列长度{len(series)}不足，需要至少{order * 2 + 1}个点（order={order}）")

    maxima_indices = argrelextrema(series, np.greater_equal, order=order)[0]

    return [
        {'index': int(idx), 'value': float(series[idx])}
        for idx in maxima_indices
    ]


def detect_bullish_divergence(
    price_series: np.ndarray,
    indicator_series: np.ndarray,
    lookback: int = 10,
    order: int = 3
) -> Dict:
    """
    检测底背离(看涨背离)

    定义:
    - 价格创新低(最近低点 < 前期低点)
    - 指标未创新低(最近指标低点 > 前期指标低点)

    Args:
        price_series: 价格序列(最近lookback个点)
        indicator_series: 指标序列(最近lookback个点)
        lookback: 回溯窗口
        order: 极值点检测窗口

    Returns:
        {
            'exists': bool,
            'strength': float [0, 1],
            'days_ago': int,
            'price_low_diff': float,
            'indicator_low_diff': float
        }
    """
    if len(price_series) < lookback or len(indicator_series) < lookback:
        raise ValueError(f"价格序列长度{len(price_series)}或指标序列长度{len(indicator_series)}不足，需要至少{lookback}个点")

    # 只看最近lookback个点
    recent_prices = price_series[-lookback:]
    recent_indicators = indicator_series[-lookback:]

    # 查找局部低点
    price_lows = find_local_minima(recent_prices, order=order)
    indicator_lows = find_local_minima(recent_indicators, order=order)

    # 如果极值点不足，返回"不存在背离"（这不是错误，只是模式不存在）
    if len(price_lows) < 2 or len(indicator_lows) < 2:
        return {'exists': False, 'strength': 0.0, 'days_ago': 0, 'price_low_diff': 0.0, 'indicator_low_diff': 0.0}

    # 取最近两个低点
    recent_price_low = price_lows[-1]
    prev_price_low = price_lows[-2]
    recent_indicator_low = indicator_lows[-1]
    prev_indicator_low = indicator_lows[-2]

    # 判断背离
    price_diff = recent_price_low['value'] - prev_price_low['value']
    indicator_diff = recent_indicator_low['value'] - prev_indicator_low['value']

    if price_diff < 0 and indicator_diff > 0:
        # 背离存在
        # 强度 = 指标改善幅度 / 价格恶化幅度
        price_deterioration = abs(price_diff) / (abs(prev_price_low['value']) + 1e-8)
        indicator_improvement = indicator_diff / (abs(prev_indicator_low['value']) + 1e-8)
        strength = min(indicator_improvement / (price_deterioration + 1e-8), 1.0)

        # 距离现在多少天(最近低点距离序列末尾的距离)
        days_ago = lookback - recent_price_low['index'] - 1

        return {
            'exists': True,
            'strength': float(max(0.0, min(strength, 1.0))),
            'days_ago': int(max(0, days_ago)),
            'price_low_diff': float(price_diff),
            'indicator_low_diff': float(indicator_diff)
        }

    return {'exists': False, 'strength': 0.0, 'days_ago': 0, 'price_low_diff': 0.0, 'indicator_low_diff': 0.0}


def detect_bearish_divergence(
    price_series: np.ndarray,
    indicator_series: np.ndarray,
    lookback: int = 10,
    order: int = 3
) -> Dict:
    """
    检测顶背离(看跌背离)

    定义:
    - 价格创新高(最近高点 > 前期高点)
    - 指标未创新高(最近指标高点 < 前期指标高点)

    Returns: 同detect_bullish_divergence
    """
    if len(price_series) < lookback or len(indicator_series) < lookback:
        raise ValueError(f"价格序列长度{len(price_series)}或指标序列长度{len(indicator_series)}不足，需要至少{lookback}个点")

    recent_prices = price_series[-lookback:]
    recent_indicators = indicator_series[-lookback:]

    price_highs = find_local_maxima(recent_prices, order=order)
    indicator_highs = find_local_maxima(recent_indicators, order=order)

    # 如果极值点不足，返回"不存在背离"（这不是错误，只是模式不存在）
    if len(price_highs) < 2 or len(indicator_highs) < 2:
        return {'exists': False, 'strength': 0.0, 'days_ago': 0, 'price_high_diff': 0.0, 'indicator_high_diff': 0.0}

    recent_price_high = price_highs[-1]
    prev_price_high = price_highs[-2]
    recent_indicator_high = indicator_highs[-1]
    prev_indicator_high = indicator_highs[-2]

    price_diff = recent_price_high['value'] - prev_price_high['value']
    indicator_diff = recent_indicator_high['value'] - prev_indicator_high['value']

    if price_diff > 0 and indicator_diff < 0:
        # 顶背离存在
        price_improvement = price_diff / (abs(prev_price_high['value']) + 1e-8)
        indicator_deterioration = abs(indicator_diff) / (abs(prev_indicator_high['value']) + 1e-8)
        strength = min(indicator_deterioration / (price_improvement + 1e-8), 1.0)

        days_ago = lookback - recent_price_high['index'] - 1

        return {
            'exists': True,
            'strength': float(max(0.0, min(strength, 1.0))),
            'days_ago': int(max(0, days_ago)),
            'price_high_diff': float(price_diff),
            'indicator_high_diff': float(indicator_diff)
        }

    return {'exists': False, 'strength': 0.0, 'days_ago': 0, 'price_high_diff': 0.0, 'indicator_high_diff': 0.0}


def detect_golden_cross(
    fast_line: np.ndarray,
    slow_line: np.ndarray,
    lookback: int = 5
) -> Dict:
    """
    检测金叉(快线上穿慢线)

    Args:
        fast_line: 快线序列
        slow_line: 慢线序列
        lookback: 回溯天数(在最近lookback天内发生的金叉)

    Returns:
        {
            'exists': bool,
            'days_ago': int,
            'cross_strength': float  # 穿越强度(快慢线差值)
        }
    """
    if len(fast_line) < lookback + 1 or len(slow_line) < lookback + 1:
        raise ValueError(f"快线长度{len(fast_line)}或慢线长度{len(slow_line)}不足，需要至少{lookback + 1}个点")

    # 从最近的往前找
    for i in range(lookback):
        idx = -(i + 1)
        prev_idx = -(i + 2)

        # 前一天: fast <= slow
        # 当天: fast > slow
        if fast_line[prev_idx] <= slow_line[prev_idx] and fast_line[idx] > slow_line[idx]:
            cross_strength = (fast_line[idx] - slow_line[idx]) / (abs(slow_line[idx]) + 1e-8)
            return {
                'exists': True,
                'days_ago': i,
                'cross_strength': float(abs(cross_strength))
            }

    return {'exists': False, 'days_ago': 0, 'cross_strength': 0.0}


def detect_death_cross(
    fast_line: np.ndarray,
    slow_line: np.ndarray,
    lookback: int = 5
) -> Dict:
    """
    检测死叉(快线下穿慢线)

    Returns: 同detect_golden_cross
    """
    if len(fast_line) < lookback + 1 or len(slow_line) < lookback + 1:
        raise ValueError(f"快线长度{len(fast_line)}或慢线长度{len(slow_line)}不足，需要至少{lookback + 1}个点")

    for i in range(lookback):
        idx = -(i + 1)
        prev_idx = -(i + 2)

        # 前一天: fast >= slow
        # 当天: fast < slow
        if fast_line[prev_idx] >= slow_line[prev_idx] and fast_line[idx] < slow_line[idx]:
            cross_strength = (slow_line[idx] - fast_line[idx]) / (abs(slow_line[idx]) + 1e-8)
            return {
                'exists': True,
                'days_ago': i,
                'cross_strength': float(abs(cross_strength))
            }

    return {'exists': False, 'days_ago': 0, 'cross_strength': 0.0}


def time_decay(
    days_ago: int,
    valid_period: int = 5,
    decay_type: str = 'linear'
) -> float:
    """
    时效衰减函数

    将离散信号转换为连续值的核心函数

    Args:
        days_ago: 信号发生在几天前
        valid_period: 信号有效期(天)
        decay_type: 衰减类型
            - 'linear': 线性衰减
            - 'exponential': 指数衰减
            - 'step': 阶跃衰减(有效期内=1, 超过=0)

    Returns:
        float [0, 1]: 衰减后的信号强度

    Examples:
        >>> time_decay(0, 5, 'linear')  # 刚发生
        1.0
        >>> time_decay(2, 5, 'linear')  # 2天前
        0.6
        >>> time_decay(5, 5, 'linear')  # 5天前
        0.0
        >>> time_decay(10, 5, 'linear')  # 超过有效期
        0.0
    """
    if days_ago > valid_period:
        return 0.0

    if days_ago < 0:
        raise ValueError(f"days_ago不能为负数，当前值为{days_ago}")

    if decay_type == 'linear':
        return 1.0 - days_ago / valid_period

    elif decay_type == 'exponential':
        # 半衰期 = valid_period / 2
        half_life = valid_period / 2.0
        return np.exp(-np.log(2) * days_ago / half_life)

    elif decay_type == 'step':
        return 1.0 if days_ago < valid_period else 0.0

    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")


def detect_price_breakout(
    prices: np.ndarray,
    resistance_lookback: int = 20
) -> Dict:
    """
    检测价格突破阻力位

    Args:
        prices: 价格序列
        resistance_lookback: 阻力位计算窗口

    Returns:
        {
            'exists': bool,
            'days_ago': int,
            'breakout_strength': float  # 突破幅度
        }
    """
    if len(prices) < resistance_lookback + 1:
        raise ValueError(f"价格序列长度{len(prices)}不足，需要至少{resistance_lookback + 1}个点")

    current_price = prices[-1]
    historical_high = np.max(prices[-(resistance_lookback+1):-1])

    if current_price > historical_high:
        breakout_strength = (current_price - historical_high) / historical_high
        return {
            'exists': True,
            'days_ago': 0,
            'breakout_strength': float(breakout_strength)
        }

    return {'exists': False, 'days_ago': 0, 'breakout_strength': 0.0}


def detect_sar_flip(
    sar_values: np.ndarray,
    prices: np.ndarray,
    lookback: int = 5
) -> Dict:
    """
    检测SAR翻转

    Args:
        sar_values: SAR指标序列
        prices: 价格序列
        lookback: 回溯窗口

    Returns:
        {
            'exists': bool,
            'days_ago': int,
            'flip_type': 'bullish' or 'bearish'
        }
    """
    if len(sar_values) < lookback + 1 or len(prices) < lookback + 1:
        raise ValueError(f"SAR序列长度{len(sar_values)}或价格序列长度{len(prices)}不足，需要至少{lookback + 1}个点")

    for i in range(lookback):
        idx = -(i + 1)
        prev_idx = -(i + 2)

        # 看涨翻转: SAR从价格上方跳到下方
        if sar_values[prev_idx] > prices[prev_idx] and sar_values[idx] < prices[idx]:
            return {
                'exists': True,
                'days_ago': i,
                'flip_type': 'bullish'
            }

        # 看跌翻转: SAR从价格下方跳到上方
        if sar_values[prev_idx] < prices[prev_idx] and sar_values[idx] > prices[idx]:
            return {
                'exists': True,
                'days_ago': i,
                'flip_type': 'bearish'
            }

    return {'exists': False, 'days_ago': 0, 'flip_type': 'none'}
