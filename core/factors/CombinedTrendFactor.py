"""
组合趋势因子 - 基于 makemoney GA 优化权重

将多个趋势因子按 GA 优化权重线性组合，测试组合 IC 是否优于单因子。

方案1 (CombinedTrendV1): TRIX×1.0 + ADX×0.9 + MOM×0.9
方案3 (CombinedTrendV3): TRIX×1.0 + ADX×0.9 + MOM×0.9 + CCI×0.7 + BollingerBands×0.6

设计说明:
- 各子因子本身输出 [0, 1] 分数（TRIX/ADX/MOM/BB）
- CCI 输出原始值，使用 tanh(cci/100) 映射到 [0, 1] 后参与加权
- 缺失子因子的分数直接跳过，按剩余因子的权重比例重新归一化
"""

import numpy as np
from .helpers import *
from .TRIX import TRIXFactor
from .ADX import ADXFactor
from .MOM import MOMFactor
from .CCI import CCI as CCIFactor
from .BollingerBands import BollingerBandsFactor


def _normalize_cci(cci_raw: float) -> float:
    """将 CCI 原始值通过 tanh 映射到 [0, 1]
    CCI=0 → 0.5, CCI=100 → 0.76, CCI=-100 → 0.24
    """
    return (np.tanh(cci_raw / 100.0) + 1.0) / 2.0


class CombinedTrendV1(BaseFactor):
    """
    方案1: TRIX + ADX + MOM 线性组合因子

    权重来自 makemoney GA 优化结果 (config214-223.yaml):
      TRIX: 1.0 / ADX: 0.9 / MOM: 0.9

    输出: 加权平均分 [0, 1]，某子因子失败时跳过并重新归一化权重
    """

    def __init__(self):
        super().__init__()
        self._trix = TRIXFactor()
        self._adx = ADXFactor()
        self._mom = MOMFactor()
        self._weights = {'trix': 1.0, 'adx': 0.9, 'mom': 0.9}

    def calc(self, ctx: FactorCtx) -> FactorResult:
        try:
            raw = {
                'trix': self._trix.calc(ctx),
                'adx':  self._adx.calc(ctx),
                'mom':  self._mom.calc(ctx),
            }

            scores, weights = [], []
            for key, result in raw.items():
                s = result['score']
                if s is not None and not np.isnan(s):
                    scores.append(s)
                    weights.append(self._weights[key])

            if not scores:
                return FactorResult(score=None, err=ValueError("所有子因子均返回 None"))

            combined = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return FactorResult(score=combined, err=None)

        except Exception as e:
            return FactorResult(score=None, err=e)


class CombinedTrendV3(BaseFactor):
    """
    方案3: TRIX + ADX + MOM + CCI + BollingerBands 线性组合因子

    权重来自 makemoney GA 优化结果 (config214-223.yaml):
      TRIX: 1.0 / ADX: 0.9 / MOM: 0.9 / CCI: 0.7 / BollingerBands: 0.6

    CCI 原始值通过 tanh(cci/100) 映射到 [0, 1] 再参与加权。
    输出: 加权平均分 [0, 1]
    """

    def __init__(self):
        super().__init__()
        self._trix = TRIXFactor()
        self._adx = ADXFactor()
        self._mom = MOMFactor()
        self._cci = CCIFactor()
        self._bb = BollingerBandsFactor()
        self._weights = {'trix': 1.0, 'adx': 0.9, 'mom': 0.9, 'cci': 0.7, 'bb': 0.6}

    def calc(self, ctx: FactorCtx) -> FactorResult:
        try:
            raw = {
                'trix': self._trix.calc(ctx),
                'adx':  self._adx.calc(ctx),
                'mom':  self._mom.calc(ctx),
                'cci':  self._cci.calc(ctx),
                'bb':   self._bb.calc(ctx),
            }

            scores, weights = [], []
            for key, result in raw.items():
                s = result['score']
                if s is None or np.isnan(s):
                    continue
                if key == 'cci':
                    s = _normalize_cci(s)
                scores.append(s)
                weights.append(self._weights[key])

            if not scores:
                return FactorResult(score=None, err=ValueError("所有子因子均返回 None"))

            combined = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return FactorResult(score=combined, err=None)

        except Exception as e:
            return FactorResult(score=None, err=e)
