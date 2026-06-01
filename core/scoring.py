"""共享打分模块 — 回测和实盘共用，纯 numpy 无外部依赖。

红线（CLAUDE.md §2.2）：T 日开盘契约——选股 **只允许使用 open[T]**，当日的
close/high/low/volume/amount 全部视为前视野泄露。需要"前收"统一用 close[T-1]。

买卖合法性闸门（涨跌停/IPO 首日/退市整理/ST）已独立到 core/legality.py 的
LegalityChecker 类，回测与实盘共用唯一实现。
"""
import numpy as np
from typing import Optional


def select_topn(
    all_scores: dict,
    score_idx: int,
    valid_stocks: list[str],
    valid_cols: np.ndarray,
    weights: dict[str, float],
    temperatures: dict[str, float],
    top_n: int,
    force_codes: Optional[list[str]] = None,
) -> tuple[list[str], np.ndarray]:
    """加权打分 + 截面排序 + 取前 N。回测和实盘必须共用此函数。

    Args:
        all_scores: {factor_name: ranks_mat (n_dates, n_full_stocks)} 归一化排名矩阵
        score_idx: 信号日索引（回测=date_idx, 实盘=score_date_idx, 都等于 T 日 NPZ 索引）
        valid_stocks: 候选股 codes（已经过 stock_indices 过滤）
        valid_cols: 候选股在 ranks_mat 中的列索引 np.intp 数组
        weights: {factor_name: weight}
        temperatures: {factor_name: temperature}
        top_n: 取前 N 只
        force_codes: 强制纳入的 codes（保留位次，前置）；None 表示不启用

    Returns:
        (topn_stocks, final_score_arr)
            topn_stocks: 选出的 top-N 股票代码列表
            final_score_arr: 全候选股的加权打分（按 valid_stocks 顺序，用于落地 plan）
    """
    final_score = np.zeros(len(valid_stocks))
    for name, ranks_mat in all_scores.items():
        w = weights.get(name, 0.0)
        if w == 0:
            continue
        ranks = ranks_mat[score_idx][valid_cols]
        temp = temperatures.get(name, 1.0)
        if temp != 1.0:
            np.power(ranks, 1.0 / temp, out=ranks)
        final_score += ranks * w

    top_idx = np.argsort(-final_score)
    topn = [valid_stocks[i] for i in top_idx[:top_n]]

    if force_codes:
        ordered, seen = [], set()
        for code in force_codes:
            if code and code not in seen:
                seen.add(code)
                ordered.append(code)
        for code in topn:
            if code not in seen:
                seen.add(code)
                ordered.append(code)
        topn = ordered[:top_n]

    return topn, final_score


def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """每日截面排名归一化 (0~1, 1=最优), NaN → 0。原地修改。"""
    n_days = scores.shape[0]
    ranks = np.empty_like(scores, dtype=np.float32)
    for d in range(n_days):
        row = scores[d]
        nans = np.isnan(row)
        valid_mask = ~nans
        n_valid = valid_mask.sum()
        if n_valid == 0:
            ranks[d] = 0.0
            continue
        order = np.argsort(row[valid_mask])[::-1]
        ranks[d, nans] = 0.0
        col_idx = np.where(valid_mask)[0]
        ranks[d, col_idx[order]] = 1.0 - np.arange(n_valid, dtype=np.float32) / n_valid
    return ranks
