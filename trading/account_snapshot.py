"""Account snapshot checks shared by local Python and client Python 3.6."""
import math


def validate_account_snapshot(asset, positions):
    if asset is None or positions is None:
        raise ValueError('Asset or positions response is missing')
    values = [float(asset.cash), float(asset.total_asset), float(asset.market_value)]
    if not all(math.isfinite(v) and v >= 0 for v in values) or values[1] <= 0:
        raise ValueError('Account assets are not ready or invalid')
    positive = [p for p in positions if p.volume > 0]
    if len({p.stock_code for p in positive}) != len(positive):
        raise ValueError('Duplicate position code')
    if any(p.volume < 0 or p.can_use_volume < 0 or p.can_use_volume > p.volume for p in positions):
        raise ValueError('Invalid holdings or sellable shares')
    market_value = sum(float(p.market_value) for p in positive)
    if (not math.isfinite(market_value)
            or abs(market_value - values[2]) > max(1.0, values[2] * 0.01)):
        raise ValueError('Asset market value and position details disagree; partial snapshot possible')
    return asset, positive
