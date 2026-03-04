"""批量归一化工具 - 对一个因子的所有股票进行归一化"""

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
        """
        对一批股票的因子分数进行Rank归一化 + 温度调节

        Args:
            raw_scores: {stock_code: FactorResult}
            temperature: 温度系数
                - T < 1.0: 压缩差异，排名更稳定
                - T = 1.0: 保持原始差异
                - T > 1.0: 放大差异，排名更灵敏

        Returns:
            {stock_code: normalized_score}  # [0, 1] 区间
        """
        # 1. 提取有效分数（排除计算失败的）
        valid_scores = {}
        for stock, result in raw_scores.items():
            err = result.get('err') if isinstance(result, dict) else result.get('err')
            score = result.get('score') if isinstance(result, dict) else result.get('score')
            if err is None and score is not None:
                valid_scores[stock] = score

        if not valid_scores:
            return {}

        # 2. Rank归一化：按分数排序，转换为 [0, 1] 区间
        sorted_stocks = sorted(
            valid_scores.items(),
            key=lambda x: x[1],
            reverse=True  # 分数高的排前面
        )

        total = len(sorted_stocks)
        rank_scores = {}
        for rank, (stock, _) in enumerate(sorted_stocks):
            # rank=0 是最高分 → 1.0
            # rank=total-1 是最低分 → 接近0
            rank_scores[stock] = 1.0 - (rank / total)

        # 3. 应用温度参数：score_new = score^(1/T)
        if temperature != 1.0:
            rank_scores = {
                stock: score ** (1.0 / temperature)
                for stock, score in rank_scores.items()
            }

        return rank_scores
