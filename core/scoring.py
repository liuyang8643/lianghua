"""共享打分模块 — 回测和实盘共用，纯 numpy 无外部依赖。

红线（CLAUDE.md §2.2）：T 日开盘契约——选股 **只允许使用 open[T]**，当日的
close/high/low/volume/amount 全部视为前视野泄露。需要"前收"统一用 close[T-1]。

买卖合法性闸门（涨跌停/IPO 首日/退市整理/ST）已独立到 core/legality.py 的
LegalityChecker 类，回测与实盘共用唯一实现。
"""
import numpy as np
from typing import Optional


def compute_weighted_scores(
    all_scores: dict,
    score_idx: int,
    valid_cols: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    """加权求和：各因子排名 × 权重，返回 (n_stocks,) 数组。"""
    final_score = np.zeros(len(valid_cols))
    for name, ranks_mat in all_scores.items():
        w = weights[name]
        if w == 0:
            continue
        final_score += ranks_mat[score_idx][valid_cols] * w
    return final_score


def select_topn(
    all_scores: dict,
    score_idx: int,
    valid_stocks: list[str],
    valid_cols: np.ndarray,
    weights: dict[str, float],
    top_n: int,
    force_codes: Optional[list[str]] = None,
    filter_mask: Optional[np.ndarray] = None,
    filter_exempt_codes: Optional[set[str]] = None,
) -> tuple[list[str], np.ndarray]:
    """加权打分 + 截面排序 + 取前 N。回测和实盘必须共用此函数。

    Args:
        all_scores: {factor_name: ranks_mat (n_dates, n_full_stocks)} 归一化排名矩阵
        score_idx: 信号日索引（回测=date_idx, 实盘=score_date_idx, 都等于 T 日 NPZ 索引）
        valid_stocks: 候选股 codes（已经过 stock_indices 过滤）
        valid_cols: 候选股在 ranks_mat 中的列索引 np.intp 数组
        weights: {factor_name: weight}
        top_n: 取前 N 只
        force_codes: 强制纳入的 codes（保留位次，前置）；None 表示不启用
        filter_mask: (n_valid_stocks,) bool 数组，True=保留；None 表示不过滤

    Returns:
        (topn_stocks, final_score_arr)
            topn_stocks: 选出的 top-N 股票代码列表
            final_score_arr: 全候选股的加权打分（按 valid_stocks 顺序，用于落地 plan）
    """
    final_score = compute_weighted_scores(all_scores, score_idx, valid_cols, weights)

    if filter_mask is not None:
        if filter_exempt_codes:
            exempt = np.array([s in filter_exempt_codes for s in valid_stocks], dtype=bool)
            final_score[~filter_mask & ~exempt] = -np.inf
        else:
            final_score[~filter_mask] = -np.inf

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


def scores_to_ranks(scores: np.ndarray, total_n: int | None = None) -> np.ndarray:
    """每日截面排名归一化 (0~1, 1=最优), NaN → 0。原地修改。

    Args:
        scores: (n_dates, n_stocks) 原始因子值
        total_n: 全量股票数，用于排名间距修正。非空时用 total_n 做分母
                 而非本地有效股票数，保证子集排名与全量排名的间距对齐。
    """
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
        denom = max(total_n - 1, 1) if total_n is not None else n_valid
        ranks[d, col_idx[order]] = 1.0 - np.arange(n_valid, dtype=np.float32) / denom
    return ranks
