"""统一交易费率口径。

回测账户、调仓计划和实盘执行器都从这里取费率，避免同一口径散落三份。
"""

import math

COMMISSION_RATE = 0.0000854
MIN_COMMISSION = 0.1
STAMP_TAX_RATE = 0.0005
TRANSFER_FEE_RATE = 0.00002
SIM_SLIPPAGE_RATE = 0.001
SIM_SLIPPAGE_BPS = SIM_SLIPPAGE_RATE * 10_000


def slippage_rate_from_bps(slippage_bps: float = SIM_SLIPPAGE_BPS) -> float:
    """Convert one-way simulated slippage from basis points to a decimal rate."""
    if (
        isinstance(slippage_bps, bool)
        or not isinstance(slippage_bps, (int, float))
        or not math.isfinite(float(slippage_bps))
        or not 0 <= float(slippage_bps) < 10_000
    ):
        raise ValueError('slippage_bps must be finite and in [0, 10000)')
    return float(slippage_bps) / 10_000


def simulated_buy_fee_rate(slippage_bps: float = SIM_SLIPPAGE_BPS) -> float:
    return COMMISSION_RATE + TRANSFER_FEE_RATE + slippage_rate_from_bps(slippage_bps)


def simulated_sell_fee_rate(slippage_bps: float = SIM_SLIPPAGE_BPS) -> float:
    return simulated_buy_fee_rate(slippage_bps) + STAMP_TAX_RATE


BUY_FEE_RATE = simulated_buy_fee_rate()
SELL_FEE_RATE = simulated_sell_fee_rate()

LIVE_BUY_FEE_RATE = COMMISSION_RATE + TRANSFER_FEE_RATE
# 实盘买单缺少涨停冻结价时的保守开盘价缓冲。
LIVE_BUY_PRICE_BUFFER = 1.0
