"""Research signals transcribed from the author's public price-pattern videos.

BV1qX4y1d7kG: prior raw close > 2, low price; the disclosed best variant
intersects the lowest 200 prices and 200 market caps. Rank ties follow the
runtime stock axis. Video monthly/variable holdings are explicitly adapted
to the current env's daily fixed-20 execution by the research runner. These
are selection scores, not hard pool filters; zero-signal fill-ins retain the
canonical full-investment behavior. The core price scores use the current
env's ST/IPO legality rather than copying the video's one-year exclusion.
BV1sa411g7RU: nine consecutive close-vs-four-bars-ago comparisons. Signals
remain active while the last nine comparisons hold (the video does not
specify post-nine resetting). Raw rather than adjusted close is an explicit
assumption for these price patterns.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from factor.base import FactorDefinition, FactorMetadata


def _member(panel):
    return (panel['listing_age'] >= 0) & (panel['delisted_mask'] == 0)


def _lowest_n(values, valid, n):
    """Exact cardinality with stable axis-order ties, without full sorting."""
    safe = np.where(valid, values, np.inf)
    n = min(n, values.shape[1])
    threshold = np.partition(safe, n - 1, axis=1)[:, n - 1, None]
    below = safe < threshold
    equal = (safe == threshold) & valid
    room = n - below.sum(axis=1, keepdims=True)
    return valid & (below | (equal & (np.cumsum(equal, axis=1) <= room)))


class BiliLowPriceAbove2:
    hist_days = 1
    def calc_batch(self, panel):
        close = np.asarray(panel['close'], dtype=np.float64)
        out = np.full(close.shape, np.nan)
        valid = np.isfinite(close) & (close > 2) & _member(panel)
        out[1:] = np.where(valid[:-1], -close[:-1], np.nan)
        return out


class BiliLowPriceSmallCap200:
    hist_days = 1
    def calc_batch(self, panel):
        close, shares = (np.asarray(panel[k], dtype=np.float64) for k in ('close', 'total_share'))
        if close.ndim != 2 or shares.shape != close.shape:
            raise ValueError('price and shares must share date-stock axes')
        member = _member(panel)
        valid_price = member & np.isfinite(close) & (close > 0)
        cap = close * shares
        valid_cap = valid_price & np.isfinite(shares) & (shares > 0) & np.isfinite(cap)
        # Ranking precedes the price/ST/IPO filters, as described in the video.
        signal = _lowest_n(close, valid_price, 200) & _lowest_n(cap, valid_cap, 200) & (close > 2)
        out = np.full(close.shape, np.nan)
        out[1:] = np.where(valid_cap[:-1], signal[:-1], np.nan)
        return out


def _nine_comparisons(panel, upward):
    close = np.asarray(panel['close'], dtype=np.float64)
    if close.ndim != 2:
        raise ValueError('close must have date-stock axes')
    out = np.full(close.shape, np.nan)
    if len(close) <= 13:
        return out
    good = np.isfinite(close) & (close > 0)
    comparisons = close[4:] > close[:-4] if upward else close[4:] < close[:-4]
    valid = good[4:] & good[:-4]
    count = len(close) - 13
    event = np.ones((count, close.shape[1]), dtype=bool)
    available = event.copy()
    for offset in range(9):
        event &= comparisons[offset:offset + count]
        available &= valid[offset:offset + count]
    out[13:] = np.where(available, event, np.nan)
    return out


class BiliNineUpContinuation:
    hist_days = 13
    def calc_batch(self, panel):
        return _nine_comparisons(panel, True)


class BiliNineDownReversal:
    hist_days = 13
    def calc_batch(self, panel):
        return _nine_comparisons(panel, False)


_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEFINITIONS = tuple(FactorDefinition(FactorMetadata(
    name=cls.__name__, version='bili-price-patterns-v1-raw-completed', hist_days=cls.hist_days,
    required_fields=fields, implementation_hash=_SHA, score_semantics=semantics), cls)
    for cls, fields, semantics in (
        (BiliLowPriceAbove2, ('close', 'listing_age', 'delisted_mask'), 'continuous'),
        (BiliLowPriceSmallCap200, ('close', 'total_share', 'listing_age', 'delisted_mask'), 'binary'),
        (BiliNineUpContinuation, ('close',), 'binary'),
        (BiliNineDownReversal, ('close',), 'binary'),
    ))


def prepare_bullish_broken_board(runtime, limit_up_prices, ordinary_universe):
    """BV1Cz4y1x7Jj's best *disclosed* variant: broken limit-up + bull bar.

    Limit prices must be supplied by the caller's public env legality engine;
    the factor layer never duplicates exchange rules or imports env. The
    video later advertises an undisclosed extra condition, which is omitted.
    The source's staggered two-day sleeves are not this ranking adaptation.
    """
    fields = ('open', 'high', 'low', 'close', 'listing_age', 'delisted_mask', 'st_mask')
    opening, high, low, close = (np.asarray(runtime.field(k), dtype=np.float64) for k in fields[:4])
    limits = np.asarray(limit_up_prices, dtype=np.float64)
    if limits.shape != close.shape or ordinary_universe.shape != close.shape or ordinary_universe.dtype != np.bool_:
        raise ValueError('limit prices and universe must match the sealed runtime')
    age = runtime.field('listing_age')
    # With full-history preload, the first PIT membership day is the actual
    # observed listing day; no future price/return enters the seasoned mask.
    if runtime.manifest.actual_preload_rows >= runtime.manifest.requested_preload_rows:
        raise ValueError('listing-year exclusion requires full-history preload')
    first = (age >= 0).argmax(axis=0)
    seasoned = (age >= 0) & (runtime.trade_dates[:, None] >= runtime.trade_dates[first][None, :] + np.timedelta64(365, 'D'))
    member = (age >= 0) & ~runtime.field('delisted_mask')
    valid = member & np.isfinite(opening) & (opening > 0) & np.isfinite(high) & np.isfinite(low) & (low > 0) & np.isfinite(close) & (close > 0)
    valid &= (high >= np.maximum(opening, close)) & (low <= np.minimum(opening, close))
    # Round observed highs/closes to the exchange's cent grid so canonical
    # float32 representation of e.g. 11.00 cannot change an equality boundary.
    event = (np.rint(high * 100) >= np.rint(limits * 100)) & (np.rint(close * 100) < np.rint(limits * 100))
    event &= (close > opening) & ordinary_universe & seasoned & ~runtime.field('st_mask') & np.isfinite(limits)
    scores = np.full(close.shape, np.nan)
    scores[1:] = np.where(valid[:-1], event[:-1], np.nan)
    scores.flags.writeable = False

    class BiliBullishBrokenLimitUp:
        def calc_batch(self, panel):
            if any(panel[k] is not runtime.field(k) or panel[k].flags.writeable for k in fields):
                raise ValueError('broken-board factor is bound to another runtime')
            return scores

    return FactorDefinition(FactorMetadata(name='BiliBullishBrokenLimitUp',
        version='bili-broken-board-v1-completed-bull-10pct-calendar365', hist_days=1,
        required_fields=fields, implementation_hash=_SHA, score_semantics='binary'),
        BiliBullishBrokenLimitUp, raw_runtime_view=True)
