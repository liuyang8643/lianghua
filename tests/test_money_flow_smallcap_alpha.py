import importlib

import numpy as np
import pandas as pd


def test_day_score_rewards_passive_inflow_and_penalizes_retail_inflow():
    mod = importlib.import_module("factor_db.factors.MoneyFlowSmallcapAlpha")
    passive = np.array([0.10, -0.05, 0.00, 0.03, -0.02])
    main = np.array([0.02, -0.02, 0.01, 0.03, 0.00])
    small = np.array([-0.03, 0.12, 0.02, 0.04, 0.08])
    large_share = np.array([0.10, 0.50, 0.20, 0.30, 0.40])
    low_amount_rank = np.array([0.10, 0.10, 0.50, 0.80, 0.20])

    score = mod._day_score(passive, main, small, large_share, low_amount_rank)

    assert score[0] > score[1]
    assert score[0] > score[4]


def test_calc_batch_uses_money_flow_on_next_trading_day(tmp_path, monkeypatch):
    mod = importlib.import_module("factor_db.factors.MoneyFlowSmallcapAlpha")
    money_dir = tmp_path / "money-flow"
    day_dir = money_dir / "2024" / "01"
    day_dir.mkdir(parents=True)

    n_days = 70
    n_stocks = 25
    trade_dates = np.arange(np.datetime64("2023-11-01"), np.datetime64("2023-11-01") + n_days)
    flow_day = np.datetime64("2024-01-02")
    flow_idx = int(np.where(trade_dates == flow_day)[0][0])
    codes = np.array([f"{i:06d}.SZ" for i in range(1, n_stocks + 1)], dtype="U12")

    rows = []
    for i in range(1, n_stocks + 1):
        rows.append({
            "code": f"{i:06d}",
            "主动买入特大单金额（元）": 1000.0 + i,
            "被动买入特大单金额（元）": 5000.0 + i * 100,
            "主动买入大单金额（元）": 1000.0 + i,
            "被动买入大单金额（元）": 5000.0 + i * 100,
            "主动买入中单金额（元）": 1000.0 + i,
            "被动买入中单金额（元）": 5000.0 + i * 100,
            "主动卖出特大单金额（元）": 1000.0,
            "被动卖出特大单金额（元）": 500.0,
            "主动卖出大单金额（元）": 1000.0,
            "被动卖出大单金额（元）": 500.0,
            "主动卖出中单金额（元）": 1000.0,
            "被动卖出中单金额（元）": 500.0,
            "小单买入金额（元）": 1000.0,
            "小单卖出金额（元）": 4000.0 + i * 20,
            "DDE大单净额（元）": 1000.0 + i * 100,
        })
    pd.DataFrame(rows).to_csv(day_dir / "2024-01-02.csv", index=False)

    panel = {
        "trade_dates": trade_dates,
        "stock_codes": codes,
        "open": np.full((n_days, n_stocks), 10.0),
        "amount": np.full((n_days, n_stocks), 1_000_000.0),
        "st_mask": np.zeros((n_days, n_stocks), dtype=bool),
    }
    monkeypatch.setattr(mod, "MONEY_FLOW_DIR", money_dir)
    monkeypatch.setattr(mod, "CACHE_PATH", tmp_path / "cache.npz")

    score = mod.MoneyFlowSmallcapAlpha().calc_batch(panel)

    assert np.isnan(score[flow_idx]).all()
    assert np.isfinite(score[flow_idx + 1]).all()


def test_select_topn_filter_excludes_existing_position():
    from core.scoring import select_topn

    all_scores = {"base": np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)}
    valid_stocks = ["A", "B", "C", "D"]
    valid_cols = np.array([0, 1, 2, 3])
    filter_mask = np.array([False, True, True, True])

    without_exempt, _ = select_topn(
        all_scores, 0, valid_stocks, valid_cols, {"base": 1.0}, 2,
        filter_mask=filter_mask,
    )
    assert without_exempt == ["B", "C"]
