"""核心指标计算 — 年化/夏普/回撤，统一 252 交易日年化，无外部依赖。"""
import numpy as np


def compute_core_metrics(daily_returns_pct) -> dict:
    """日收益率(%) → {annualized, max_drawdown, sharpe}"""
    daily = np.asarray(daily_returns_pct, dtype=float)
    daily = daily[np.isfinite(daily)]
    n = len(daily)
    if n < 2:
        return {'annualized': 0.0, 'max_drawdown': 0.0, 'sharpe': 0.0}

    mean_ret = float(np.mean(daily))
    std_ret = float(np.std(daily, ddof=1))
    sharpe = float(mean_ret / std_ret * np.sqrt(252.0)) if std_ret > 0 else 0.0

    terminal_nav_path = np.cumprod(1.0 + daily / 100.0)
    years = n / 252.0
    annualized = (
        float((terminal_nav_path[-1] ** (1.0 / years) - 1) * 100)
        if years > 0 and terminal_nav_path[-1] > 0
        else 0.0
    )

    # Every independently reported period starts with NAV=1.  Without this
    # anchor, a loss on the first day is silently omitted from drawdown and can
    # materially inflate short-fold Calmar.
    nav = np.concatenate(([1.0], terminal_nav_path))
    peaks = np.maximum.accumulate(nav)
    max_dd = float(np.min(nav / peaks - 1.0) * 100)

    return {'annualized': annualized, 'max_drawdown': max_dd, 'sharpe': sharpe}
