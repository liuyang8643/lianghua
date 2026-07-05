"""T-1 策略排名预过滤：回测与实盘共用，确保股票池对齐逻辑唯一。"""


def apply_prefilter(t1_ranking: list[str], prefilter_n: int,
                    all_stocks_full: list[str],
                    positions: dict | None = None) -> list[str]:
    """根据 T-1 全量排名取 top prefilter_n，并合并当前持仓。

    Args:
        t1_ranking: T-1 全量排名（按分数降序的 code 列表）
        prefilter_n: 取前 N 只
        all_stocks_full: 当日全量候选池
        positions: 当前持仓 {code: volume}，不在 prefilter 内的持仓会被补入

    Returns:
        过滤后的股票列表（保持 all_stocks_full 中的出现顺序，持仓补在末尾）
    """
    prefilter_set = set(t1_ranking[:prefilter_n])
    filtered = [s for s in all_stocks_full if s in prefilter_set]

    if positions:
        for code in positions:
            if code not in prefilter_set and code in all_stocks_full:
                prefilter_set.add(code)
                filtered.append(code)

    return filtered
