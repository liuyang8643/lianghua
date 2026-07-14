"""T-1 策略排名预过滤：回测与实盘共用，确保股票池对齐逻辑唯一。"""


def apply_prefilter(t1_ranking: list[str], prefilter_n: int,
                    all_stocks_full: list[str]) -> list[str]:
    """T-1 全量排名取前 N，并保持当日股票池顺序。"""
    prefilter_set = set(t1_ranking[:prefilter_n])
    return [code for code in all_stocks_full if code in prefilter_set]
