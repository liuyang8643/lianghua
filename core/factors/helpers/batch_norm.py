"""批量归一化工具 - 对一个因子的所有股票进行归一化"""

import numpy as np
from typing import Dict
from .interface import FactorResult


class BatchNormFactor:
    """批量归一化 - 应用温度参数平滑单个因子内股票分数的差异"""

    @staticmethod
    def batch_normalize(
        raw_scores: Dict[str, FactorResult],
        temperature: float = 1.0,
        method: str = 'rank'
    ) -> Dict[str, float]:
        if not raw_scores:
            return {}

        stocks = []
        scores = []
        for stock, result in raw_scores.items():
            err = result.get('err') if isinstance(result, dict) else result.get('err')
            score = result.get('score') if isinstance(result, dict) else result.get('score')
            if err is None and score is not None:
                stocks.append(stock)
                scores.append(score)

        if not stocks:
            return {}

        n = len(stocks)
        arr = np.array(scores)
        order = np.argsort(arr)[::-1]  # descending
        norm = np.empty(n)
        norm[order] = 1.0 - (np.arange(n) / n)  # rank=0 → 1.0, rank=n-1 → ~0

        if temperature != 1.0:
            norm = norm ** (1.0 / temperature)

        return dict(zip(stocks, norm.tolist()))
