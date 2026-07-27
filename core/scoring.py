"""共享打分模块 — 回测和实盘共用，纯 numpy 无外部依赖。

红线（CLAUDE.md §2.2）：T 日开盘契约——选股 **只允许使用 open[T]**，当日的
close/high/low/volume/amount 全部视为前视野泄露。需要"前收"统一用 close[T-1]。

买卖合法性闸门（涨跌停/IPO 首日/退市整理/ST）已独立到 core/legality.py 的
LegalityChecker 类，回测与实盘共用唯一实现。
"""
import numpy as np


class FactorScoreMatrices(dict):
    """Rank matrices with optional raw values for candidate-local reranking."""

    def __init__(self, *args, raw_scores=None, pre_ranked_names=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_scores = raw_scores
        self.pre_ranked_names = frozenset(pre_ranked_names)


def candidate_local_score_matrices(all_scores, score_idx, candidate_cols):
    """Re-rank regular factors inside the fixed T-day candidate pool."""
    raw_scores = getattr(all_scores, 'raw_scores', None)
    if raw_scores is None:
        return all_scores, score_idx

    candidate_cols = np.asarray(candidate_cols, dtype=np.intp)
    width = next(iter(all_scores.values())).shape[1]
    local_scores = {}
    for name, ranked in all_scores.items():
        if name in all_scores.pre_ranked_names:
            values = ranked[score_idx, candidate_cols]
        else:
            raw = raw_scores[name][score_idx, candidate_cols]
            values = scores_to_ranks(raw[None, :])[0]
        matrix = np.zeros((1, width), dtype=np.float32)
        matrix[0, candidate_cols] = values
        local_scores[name] = matrix
    return local_scores, 0


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


def compute_weighted_score_matrix(
    all_scores: dict,
    date_indices: np.ndarray,
    valid_cols: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    """Vectorized weighted scores for multiple dates and stocks."""
    rows = np.asarray(date_indices, dtype=np.intp)
    cols = np.asarray(valid_cols, dtype=np.intp)
    result = np.zeros((len(rows), len(cols)), dtype=np.float64)
    selection = np.ix_(rows, cols)
    for name, ranks_mat in all_scores.items():
        weight = float(weights[name])
        if weight != 0.0:
            result += np.asarray(ranks_mat[selection], dtype=np.float64) * weight
    return result


def select_topn(
    all_scores: dict,
    score_idx: int,
    valid_stocks: list[str],
    valid_cols: np.ndarray,
    weights: dict[str, float],
    top_n: int,
    filter_mask: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    """加权打分 + 截面排序 + 取前 N。回测和实盘必须共用此函数。

    Args:
        all_scores: {factor_name: ranks_mat (n_dates, n_full_stocks)} 归一化排名矩阵
        score_idx: 信号日索引（回测=date_idx, 实盘=score_date_idx, 都等于 T 日 NPZ 索引）
        valid_stocks: 候选股 codes（已经过 stock_indices 过滤）
        valid_cols: 候选股在 ranks_mat 中的列索引 np.intp 数组
        weights: {factor_name: weight}
        top_n: 取前 N 只
        filter_mask: (n_valid_stocks,) bool 数组，True=保留；None 表示不过滤

    Returns:
        (topn_stocks, final_score_arr)
            topn_stocks: 选出的 top-N 股票代码列表
            final_score_arr: 全候选股的加权打分（按 valid_stocks 顺序，用于落地 plan）
    """
    final_score = compute_weighted_scores(all_scores, score_idx, valid_cols, weights)

    if filter_mask is not None:
        final_score[~filter_mask] = -np.inf

    top_idx = np.flatnonzero(np.isfinite(final_score))[np.argsort(-final_score[np.isfinite(final_score)])]
    topn = [valid_stocks[i] for i in top_idx[:top_n]]

    return topn, final_score


def select_topn_legal(
    all_scores: dict,
    score_idx: int,
    valid_stocks: list[str],
    valid_cols: np.ndarray,
    weights: dict[str, float],
    buy_n: int,
    sell_m: int,
    checker,
    trade_idx: int,
    signal_date,
    day_open: np.ndarray,
    stock_indices: dict[str, int],
    filter_mask: np.ndarray | None = None,
) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    """合并排名 + 批量合法性检查 + 凑够 N 即停，替代 select_topn + select_tradable_buys。

    1. compute_weighted_scores 排名所有候选股
    2. np.argsort 降序排列
    3. 从高到低遍历，每批 BATCH 只调 checker.check
    4. 合法的加入 buy_n_stocks / sell_m_stocks，凑够即停
    """
    final_score = compute_weighted_scores(all_scores, score_idx, valid_cols, weights)

    if filter_mask is not None:
        final_score[~filter_mask] = -np.inf

    buy, sell, ranking = select_topn_legal_from_scores(
        final_score, valid_stocks, valid_cols, buy_n, sell_m,
        checker, trade_idx, signal_date, day_open,
    )
    return buy, sell, final_score, ranking


def select_topn_legal_from_scores(
    final_score: np.ndarray,
    valid_stocks: list[str],
    valid_cols: np.ndarray,
    buy_n: int,
    sell_m: int,
    checker,
    trade_idx: int,
    signal_date,
    day_open: np.ndarray,
) -> tuple[list[str], list[str], list[str]]:
    """Select legal TopN from an already-combined score row."""

    ranked_idx = np.flatnonzero(np.isfinite(final_score))[np.argsort(-final_score[np.isfinite(final_score)])]
    buy_n_stocks: list[str] = []
    sell_m_stocks: list[str] = []

    BATCH = 80
    for start in range(0, len(ranked_idx), BATCH):
        if len(buy_n_stocks) >= buy_n and len(sell_m_stocks) >= sell_m:
            break
        batch = ranked_idx[start:start + BATCH]
        batch_cols = valid_cols[batch]
        # 筛掉停牌（open 为 NaN 或 ≤0）
        opens = day_open[batch_cols]
        valid_price = ~np.isnan(opens) & (opens > 0)
        if not np.any(valid_price):
            continue
        batch_idx = batch[valid_price]
        batch_si = batch_cols[valid_price]
        ok, _ = checker.check(batch_si, trade_idx, signal_date, is_buy=True)
        for i, is_ok in enumerate(ok):
            if is_ok:
                s = valid_stocks[batch_idx[i]]
                if len(buy_n_stocks) < buy_n:
                    buy_n_stocks.append(s)
                if len(sell_m_stocks) < sell_m:
                    sell_m_stocks.append(s)
            if len(buy_n_stocks) >= buy_n and len(sell_m_stocks) >= sell_m:
                break

    # sell_m 不足时用排名补（不检查合法性——卖出的合法性单独判断）
    for idx_val in ranked_idx:
        if len(sell_m_stocks) >= sell_m:
            break
        s = valid_stocks[idx_val]
        if s not in buy_n_stocks and s not in sell_m_stocks:
            sell_m_stocks.append(s)

    # 全量排名供 T+1 prefilter 复用（零额外计算——就是用已经排好的 ranked_idx）
    t1_ranking = [valid_stocks[i] for i in ranked_idx]

    return buy_n_stocks, sell_m_stocks[:sell_m], t1_ranking


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
