from typing import Optional, TypedDict


class FactorResult(TypedDict):
    score: Optional[float]
    err: Optional[Exception]


class BaseFactor:
    hist_days: int = 0
