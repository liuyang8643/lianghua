"""参数化 size+finance blend 因子，四个命名变体在此定义。"""

import numpy as np
import pandas as pd

from core.factors.helpers import BaseFactor

MIN_RAW_PRICE = 2.0

FIN_FIELDS = ['eps', 'roe', 'gross_margin', 'operating_cf_ps', 'profit_yoy', 'revenue_yoy']


class SmallCapMVFinanceBlend(BaseFactor):
  """size pct rank + Σ(weight × finance_field pct rank) 的泛化模板。

  子类只需设 `_field_weights`：{字段名: 权重}，支持 eps/roe/gross_margin/
  operating_cf_ps/profit_yoy/revenue_yoy。
  eps 自动做 price 除法，其余直接取 rank。
  """

  hist_days = 0
  _field_weights: dict[str, float] = {}

  def calc_batch(self, panel: dict) -> pd.DataFrame:
    stock_codes = panel["stock_codes"]
    trade_dates = panel["trade_dates"]
    raw_open = panel["open"]
    total_share = panel["total_share"]
    st_mask = panel["st_mask"]

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= MIN_RAW_PRICE)
      & ~np.isnan(total_share)
      & (total_share > 0)
      & ~st_mask
    )

    total_mv_yi = (raw_open * total_share) / 1e8
    size_rank = pd.DataFrame(total_mv_yi, index=trade_dates, columns=stock_codes) \
      .where(pd.DataFrame(base_valid, index=trade_dates, columns=stock_codes)) \
      .rank(axis=1, pct=True)

    score = 1 - size_rank
    for field, weight in self._field_weights.items():
      if field == "eps":
        fin_arr = panel["eps"] / np.where(raw_open > 0, raw_open, np.nan)
      else:
        fin_arr = panel[field]
      rank = pd.DataFrame(fin_arr, index=trade_dates, columns=stock_codes) \
        .where(pd.DataFrame(base_valid, index=trade_dates, columns=stock_codes)) \
        .rank(axis=1, pct=True)
      score += weight * rank.fillna(0.0)

    return score.where(pd.DataFrame(base_valid, index=trade_dates, columns=stock_codes))


class SmallCapDailyMVMaskRoe2xBottom10(SmallCapMVFinanceBlend):
  """ROE 2x weight variant — roe=0.20, selects bottom 10% by composite score."""
  _field_weights = {"roe": 0.20}


class TrueMarketCap(SmallCapMVFinanceBlend):
  """纯真市值因子 — open × total_share，无财务叠加"""
  _field_weights = {}


class TMC_ROE_10(SmallCapMVFinanceBlend):
  _field_weights = {"roe": 0.10}


class TMC_ROE_30(SmallCapMVFinanceBlend):
  _field_weights = {"roe": 0.30}


class TMC_GM_15(SmallCapMVFinanceBlend):
  _field_weights = {"gross_margin": 0.15}


class TMC_ProfitYoy_10(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.10}


class TMC_RevYoy_10(SmallCapMVFinanceBlend):
  _field_weights = {"revenue_yoy": 0.10}


class TMC_ROE_CFQ(SmallCapMVFinanceBlend):
  _field_weights = {"roe": 0.15, "operating_cf_ps": 0.10}


class TMC_Quality(SmallCapMVFinanceBlend):
  _field_weights = {"roe": 0.10, "gross_margin": 0.05, "operating_cf_ps": 0.05}

class TMC_ProfitYoy_Roe_10_10(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.10, "roe": 0.10}

class TMC_ProfitYoy_15(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.15}

class TMC_ProfitYoy_20(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.20}

class TMC_ProfitYoy_Eps_10_05(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.10, "eps": 0.05}

class TMC_ProfitYoy_Roe_CFQ(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.10, "roe": 0.10, "operating_cf_ps": 0.05}

class TMC_ProfitYoy_25(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.25}

class TMC_ProfitYoy_30(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.30}

class TMC_ProfitYoy_15_Eps_05(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.15, "eps": 0.05}

class TMC_ProfitYoy_RevYoy_10_05(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.10, "revenue_yoy": 0.05}

class TMC_ProfitYoy_15_neg_GM(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.15, "gross_margin": -0.05}

class TMC_ProfitYoy_25_neg_GM(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.25, "gross_margin": -0.05}

class TMC_ProfitYoy_20_CFQ(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.20, "operating_cf_ps": 0.05}

class TMC_ProfitYoy_20_Eps_neg_GM(SmallCapMVFinanceBlend):
  _field_weights = {"profit_yoy": 0.20, "eps": 0.05, "gross_margin": -0.03}


class PureProfitYoy(BaseFactor):
  hist_days = 0

  def calc_batch(self, panel: dict) -> pd.DataFrame:
    stock_codes = panel["stock_codes"]
    trade_dates = panel["trade_dates"]
    raw_open = panel["open"]
    st_mask = panel["st_mask"]
    profit_yoy = panel["profit_yoy"]

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= MIN_RAW_PRICE)
      & ~st_mask
    )
    valid_df = pd.DataFrame(base_valid, index=trade_dates, columns=stock_codes)
    py_rank = pd.DataFrame(profit_yoy, index=trade_dates, columns=stock_codes) \
      .where(valid_df).rank(axis=1, pct=True)
    return py_rank.where(valid_df)


class PureHighVol(BaseFactor):
  hist_days = 22

  def calc_batch(self, panel: dict) -> pd.DataFrame:
    stock_codes = panel["stock_codes"]
    trade_dates = panel["trade_dates"]
    raw_open = panel["open"]
    st_mask = panel["st_mask"]

    n_dates, n_stocks = len(trade_dates), len(stock_codes)
    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= MIN_RAW_PRICE)
      & ~st_mask
    )

    daily_ret = np.full((n_dates, n_stocks), np.nan)
    daily_ret[1:] = raw_open[1:] / raw_open[:-1] - 1.0

    vol = np.full((n_dates, n_stocks), np.nan)
    for t in range(21, n_dates):
      vol[t] = np.nanstd(daily_ret[t - 20:t + 1], axis=0)

    valid_df = pd.DataFrame(base_valid, index=trade_dates, columns=stock_codes)
    vol_df = pd.DataFrame(vol, index=trade_dates, columns=stock_codes).where(valid_df)
    vol_rank = vol_df.rank(axis=1, pct=True)
    return vol_rank.where(valid_df)


class TMC_ProfitYoy_25_LowVol(BaseFactor):
  """TMC + profit_yoy(0.25) + 低波动惩罚 — 在小盘成长中剔除高波动"""
  hist_days = 22

  def calc_batch(self, panel: dict) -> pd.DataFrame:
    stock_codes = panel["stock_codes"]
    trade_dates = panel["trade_dates"]
    raw_open = panel["open"]
    total_share = panel["total_share"]
    st_mask = panel["st_mask"]
    profit_yoy = panel["profit_yoy"]

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= 2.0)
      & ~np.isnan(total_share)
      & (total_share > 0)
      & ~st_mask
    )

    n_dates, n_stocks = len(trade_dates), len(stock_codes)

    # daily open-to-open returns
    daily_ret = np.full((n_dates, n_stocks), np.nan)
    daily_ret[1:] = raw_open[1:] / raw_open[:-1] - 1.0

    # rolling 20-day volatility (forward-filled to avoid lookahead at T, but open[t] is known)
    vol = np.full((n_dates, n_stocks), np.nan)
    for t in range(21, n_dates):
      window = daily_ret[t - 20:t + 1]
      vol[t] = np.nanstd(window, axis=0)

    valid_df = pd.DataFrame(base_valid, index=trade_dates, columns=stock_codes)

    # size pct_rank
    total_mv_yi = (raw_open * total_share) / 1e8
    size_rank = pd.DataFrame(total_mv_yi, index=trade_dates, columns=stock_codes) \
      .where(valid_df).rank(axis=1, pct=True)

    # profit_yoy pct_rank
    py_rank = pd.DataFrame(profit_yoy, index=trade_dates, columns=stock_codes) \
      .where(valid_df).rank(axis=1, pct=True).fillna(0.0)

    # vol pct_rank (low vol = good)
    vol_df = pd.DataFrame(vol, index=trade_dates, columns=stock_codes).where(valid_df)
    vol_rank = vol_df.rank(axis=1, pct=True)

    score = (1 - size_rank) + 0.25 * py_rank - 0.10 * vol_rank.fillna(0.5)
    return score.where(valid_df)
