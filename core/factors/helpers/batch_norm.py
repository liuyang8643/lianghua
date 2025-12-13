"""
BatchNorm因子标准化模块

从qmt-trade项目迁移，用于对因子的原始分数进行批量归一化处理。
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np
from scipy.stats import rankdata


@dataclass
class FactorOutput:
    """标准化因子输出"""
    code: str
    norm_score: float      # [0, 1] 标准化分数
    raw_value: float       # 原始因子值
    batch_mean: float      # 当天市场均值
    batch_std: float       # 当天市场标准差
    batch_size: int        # 批次大小
    z_score: float         # Z-Score

    __slots__ = ('code', 'norm_score', 'raw_value', 'batch_mean', 'batch_std', 'batch_size', 'z_score')


class BatchNormFactor:
    """BatchNorm因子标准化"""

    @staticmethod
    def batch_normalize(values: Dict[str, float], method: str = 'rank') -> Dict[str, FactorOutput]:
        """
        对一个batch（当天所有股票）进行BatchNorm标准化

        Args:
            values: {code: raw_value}
            method: 'rank' (排名归一化) 或 'minmax' (值归一化)

        Returns:
            {code: FactorOutput}
        """
        if not values:
            raise ValueError("批量归一化输入值为空")

        # 向量化数据转换（按股票代码排序，确保顺序稳定）
        codes = sorted(values.keys())
        raw_values = np.array([values[code] for code in codes], dtype=float)

        # NaN检查
        nan_mask = np.isnan(raw_values)
        if nan_mask.any():
            nan_codes = [codes[i] for i in np.where(nan_mask)[0][:3]]
            raise ValueError(f"【BatchNorm】{nan_mask.sum()}只股票因子值为nan: {nan_codes}")

        # 批次统计量（一次性计算）
        batch_mean = float(raw_values.mean())
        batch_std = float(raw_values.std())
        batch_size = len(raw_values)

        # 检查所有值是否相同（使用快速方法）
        if raw_values.min() == raw_values.max():
            val = float(raw_values[0])
            norm_scores = np.full(batch_size, 0.5 if val < 0 or val > 1 else val, dtype=float)
            z_scores = np.zeros(batch_size, dtype=float)
        else:
            if method == 'rank':
                # 使用scipy.stats.rankdata向量化实现排名归一化（C实现，比Python排序快10倍）
                ranks = rankdata(raw_values, method='ordinal')
                norm_scores = (ranks - 1) / (batch_size - 1) if batch_size > 1 else np.array([0.5] * batch_size)
            else:
                # Min-Max归一化（向量化）
                min_val = raw_values.min()
                max_val = raw_values.max()
                norm_scores = (raw_values - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else np.array([0.5] * batch_size)

            # Z-score计算（向量化）
            z_scores = (raw_values - batch_mean) / (batch_std if batch_std > 1e-8 else 1.0)

        # 批量构建输出（使用列表推导+zip避免重复索引）
        return {
            code: FactorOutput(code, float(ns), float(rv), batch_mean, batch_std, batch_size, float(zs))
            for code, ns, rv, zs in zip(codes, norm_scores, raw_values, z_scores)
        }


def apply_batch_norm_to_factor(results: Dict, value_key: str = 'score', method: str = 'minmax') -> Dict:
    """
    通用函数：为因子结果应用BatchNorm

    Args:
        results: {code: {value_key: float, ...}} or {code: float}
        value_key: 要标准化的值的键名
        method: 'rank' (排名归一化) 或 'minmax' (值归一化)

    Returns:
        {code: {..., 'score': float, 'batch_mean': float, ...}}
    """
    if not results:
        raise ValueError("因子结果为空")

    # 提取原始值（向量化，避免多次类型检查）
    first_val = next(iter(results.values()))
    if isinstance(first_val, dict):
        # 字典类型结果
        raw_values = {code: result[value_key] for code, result in results.items()}
    else:
        # 标量类型结果
        raw_values = {code: float(result) for code, result in results.items()}

    # BatchNorm
    norm_results = BatchNormFactor.batch_normalize(raw_values, method=method)

    # 合并结果（优化：避免重复字典访问）
    if isinstance(first_val, dict):
        return {
            code: {
                **results[code],
                'norm_score': (norm_out := norm_results[code]).norm_score,
                'score': norm_out.norm_score,
                'batch_mean': norm_out.batch_mean,
                'batch_std': norm_out.batch_std,
                'z_score': norm_out.z_score
            }
            for code in results
        }
    else:
        return {
            code: {
                value_key: results[code],
                'norm_score': (norm_out := norm_results[code]).norm_score,
                'score': norm_out.norm_score,
                'batch_mean': norm_out.batch_mean,
                'batch_std': norm_out.batch_std,
                'z_score': norm_out.z_score
            }
            for code in results
        }
