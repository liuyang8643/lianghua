"""共享打分模块 — 回测和实盘共用，纯 numpy 无外部依赖。

红线（CLAUDE.md §2.2）：T 日开盘契约——选股 **只允许使用 open[T]**，当日的
close/high/low/volume/amount 全部视为前视野泄露。需要"前收"统一用 close[T-1]。

买卖合法性闸门（涨跌停/IPO 首日/退市整理/ST）已独立到 core/legality.py 的
LegalityChecker 类，回测与实盘共用唯一实现。
"""
import numpy as np


class FactorScoreMatrices(dict):
    """Rank matrices with raw values and per-factor validity kept out of filters."""

    def __init__(
        self,
        *args,
        raw_scores=None,
        pre_ranked_names=(),
        factor_validity=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.raw_scores = raw_scores
        self.pre_ranked_names = frozenset(pre_ranked_names)
        self.factor_validity = factor_validity


def top_level_factor_filter_masks(
    all_scores,
    common_filter_masks: dict,
    weights: dict[str, float],
) -> dict:
    """Restore the ordinary top-level validity mask for timing only.

    Sleeve selection keeps factor validity local to each sleeve.  The existing
    timing target, however, still uses the top-level strategy weights and must
    therefore receive exactly the validity intersection those weights would
    have used without sleeves.
    """
    factor_validity = getattr(all_scores, 'factor_validity', None)
    if factor_validity is None:
        return common_filter_masks
    active = []
    for name, weight in weights.items():
        if weight == 0.0:
            continue
        if name not in factor_validity:
            raise ValueError(
                f'missing factor validity for top-level factor: {name}'
            )
        active.append(np.asarray(factor_validity[name], dtype=bool))
    result = dict(common_filter_masks)
    if active:
        result['_active_factor_intersection'] = np.logical_and.reduce(active)
    return result


def candidate_local_score_matrices(all_scores, score_idx, candidate_cols):
    """Re-rank regular factors inside the fixed T-day candidate pool."""
    raw_scores = getattr(all_scores, 'raw_scores', None)
    if raw_scores is None:
        return all_scores, score_idx

    candidate_cols = np.asarray(candidate_cols, dtype=np.intp)
    width = next(iter(all_scores.values())).shape[1]
    local_scores = {}
    local_validity = {}
    for name, ranked in all_scores.items():
        if name in all_scores.pre_ranked_names:
            values = ranked[score_idx, candidate_cols]
        else:
            raw = raw_scores[name][score_idx, candidate_cols]
            values = scores_to_ranks(raw[None, :])[0]
        matrix = np.zeros((1, width), dtype=np.float32)
        matrix[0, candidate_cols] = values
        local_scores[name] = matrix
        if all_scores.factor_validity is not None:
            validity = np.zeros((1, width), dtype=bool)
            validity[0, candidate_cols] = np.asarray(
                all_scores.factor_validity[name][score_idx, candidate_cols],
                dtype=bool,
            )
            local_validity[name] = validity
    return FactorScoreMatrices(
        local_scores,
        raw_scores=None,
        pre_ranked_names=all_scores.pre_ranked_names,
        factor_validity=local_validity if local_validity else None,
    ), 0


def validate_selection_sleeves(
    selection_sleeves,
    buy_n: int,
    available_factor_names=None,
) -> list[dict]:
    """Validate and normalize fixed-slot sleeve selection configuration."""
    if not isinstance(selection_sleeves, list) or not selection_sleeves:
        raise ValueError('selection_sleeves must be a non-empty list')
    if isinstance(buy_n, bool) or not isinstance(buy_n, (int, np.integer)) or buy_n <= 0:
        raise ValueError('buy_n must be a positive integer')

    available = (
        None if available_factor_names is None
        else set(available_factor_names)
    )
    normalized: list[dict] = []
    names: set[str] = set()
    total_slots = 0
    for sleeve in selection_sleeves:
        if not isinstance(sleeve, dict):
            raise ValueError('each selection sleeve must be an object')
        name = sleeve.get('name')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('selection sleeve name must be a non-empty string')
        name = name.strip()
        if name in names:
            raise ValueError(f'duplicate selection sleeve name: {name}')
        names.add(name)

        slots = sleeve.get('slots')
        if (
            isinstance(slots, bool)
            or not isinstance(slots, (int, np.integer))
            or slots <= 0
        ):
            raise ValueError(f'selection sleeve {name} slots must be a positive integer')

        weights = sleeve.get('weights')
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f'selection sleeve {name} weights must be a non-empty object')
        normalized_weights: dict[str, float] = {}
        for factor_name, weight in weights.items():
            if not isinstance(factor_name, str) or not factor_name:
                raise ValueError(f'selection sleeve {name} has an invalid factor name')
            if available is not None and factor_name not in available:
                raise ValueError(
                    f'selection sleeve {name} factor does not exist: {factor_name}'
                )
            if isinstance(weight, bool) or not isinstance(
                weight, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    f'selection sleeve {name} factor {factor_name} weight must be numeric'
                )
            value = float(weight)
            if not np.isfinite(value):
                raise ValueError(
                    f'selection sleeve {name} factor {factor_name} weight must be finite'
                )
            normalized_weights[factor_name] = value
        if not any(weight != 0.0 for weight in normalized_weights.values()):
            raise ValueError(f'selection sleeve {name} weights cannot all be zero')

        normalized.append({
            'name': name,
            'slots': int(slots),
            'weights': normalized_weights,
        })
        total_slots += int(slots)

    if total_slots != int(buy_n):
        raise ValueError(
            f'selection sleeve slots must sum to buy_n: {total_slots} != {buy_n}'
        )
    return normalized


def compute_weighted_scores(
    all_scores: dict,
    score_idx: int,
    valid_cols: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    """加权求和：各因子排名 × 权重，返回 (n_stocks,) 数组。"""
    final_score = np.zeros(len(valid_cols))
    for name, ranks_mat in all_scores.items():
        w = weights.get(name, 0.0)
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
        weight = float(weights.get(name, 0.0))
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


def select_selection_sleeves_legal(
    all_scores: dict,
    score_idx: int,
    valid_stocks: list[str],
    valid_cols: np.ndarray,
    selection_sleeves: list[dict],
    buy_n: int,
    sell_m: int,
    checker,
    trade_idx: int,
    signal_date,
    day_open: np.ndarray,
    common_filter_mask: np.ndarray | None = None,
) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    """Select fixed slots sleeve-by-sleeve, excluding earlier sleeve targets."""
    sleeves = validate_selection_sleeves(
        selection_sleeves, buy_n, available_factor_names=all_scores.keys()
    )
    factor_validity = getattr(all_scores, 'factor_validity', None)
    if factor_validity is None:
        raise ValueError(
            'selection_sleeves require per-factor validity on all_scores'
        )

    n_stocks = len(valid_stocks)
    stock_to_local = {code: idx for idx, code in enumerate(valid_stocks)}
    if common_filter_mask is not None and len(common_filter_mask) != n_stocks:
        raise ValueError('common sleeve filter mask length must match valid_stocks')
    selected: list[str] = []
    selected_indices: set[int] = set()
    sleeve_rankings: list[list[str]] = []

    for sleeve in sleeves:
        scores = compute_weighted_scores(
            all_scores, score_idx, valid_cols, sleeve['weights']
        )
        eligible = (
            np.ones(n_stocks, dtype=bool)
            if common_filter_mask is None
            else np.asarray(common_filter_mask, dtype=bool).copy()
        )
        for factor_name, weight in sleeve['weights'].items():
            if weight == 0.0:
                continue
            if factor_name not in factor_validity:
                raise ValueError(
                    f'missing factor validity for sleeve factor: {factor_name}'
                )
            eligible &= np.asarray(
                factor_validity[factor_name][score_idx, valid_cols],
                dtype=bool,
            )
        if selected_indices:
            eligible[np.fromiter(selected_indices, dtype=np.intp)] = False
        scores[~eligible] = -np.inf

        sleeve_targets, _, sleeve_ranking = select_topn_legal_from_scores(
            scores, valid_stocks, valid_cols, sleeve['slots'], 0,
            checker, trade_idx, signal_date, day_open,
        )
        selected_indices.update(stock_to_local[code] for code in sleeve_targets)
        selected.extend(sleeve_targets)
        sleeve_rankings.append(sleeve_ranking)

    # One deterministic display row: actual targets first, then remaining
    # candidates in sleeve-priority order.
    t1_ranking = list(selected)
    ranked_seen = set(selected)
    for ranking in sleeve_rankings:
        for code in ranking:
            if code not in ranked_seen:
                ranked_seen.add(code)
                t1_ranking.append(code)
    final_score = np.full(n_stocks, -np.inf, dtype=np.float64)
    for rank, code in enumerate(t1_ranking):
        final_score[stock_to_local[code]] = float(len(t1_ranking) - rank)

    return selected, selected[:sell_m], final_score, t1_ranking


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


def factor_scores_to_rank_matrix(
    raw: np.ndarray,
    valid_cols: np.ndarray,
    *,
    scores_are_ranks: bool = False,
) -> np.ndarray:
    """Convert raw factor values to the common rank matrix representation.

    ``scores_are_ranks`` is an explicit opt-in for factors that already emit
    comparable [0, 1] scores, such as missing-value-neutral composites.
    Keeping this conversion shared prevents GA and single backtests from using
    different score semantics.
    """
    columns = np.asarray(valid_cols, dtype=np.intp)
    values = np.asarray(raw)[:, columns].astype(np.float32, copy=False)
    ranks = np.zeros(np.asarray(raw).shape, dtype=np.float32)
    if not scores_are_ranks:
        ranks[:, columns] = scores_to_ranks(values)
        return ranks

    finite = values[np.isfinite(values)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise ValueError(
            "scores_are_ranks requires finite scores within [0, 1]"
        )
    ranks[:, columns] = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return ranks


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
