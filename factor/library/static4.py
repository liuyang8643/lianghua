"""Static-live four-factor comparison, including its embedded validity gates.

Reuse the original score calculations. The live configuration's filter=false
does not disable these per-factor gates; apply them before percentile ranking.
"""

import numpy as np

from factor_db.factors.AmihudIlliquidity import AmihudIlliquidity as OriginalAmihud
from factor_db.factors.AmountBasedSmallCap import AmountBasedSmallCap as OriginalAmount
from factor_db.factors.TrueMarketCap import TrueMarketCap as OriginalMarketCap
from factor_db.factors.VolumeCV import VolumeCV as OriginalVolume


class _LiveValidity:
    def calc_batch(self, panel: dict) -> np.ndarray:
        values = super().calc_batch(panel)
        valid = (panel['open'] >= 2.0) & (panel['st_mask'] == 0)
        return np.where(valid, values, np.nan)


class TrueMarketCap(_LiveValidity, OriginalMarketCap):
    pass


class VolumeCV(_LiveValidity, OriginalVolume):
    pass


class AmountBasedSmallCap(_LiveValidity, OriginalAmount):
    pass


class AmihudIlliquidity(_LiveValidity, OriginalAmihud):
    pass
