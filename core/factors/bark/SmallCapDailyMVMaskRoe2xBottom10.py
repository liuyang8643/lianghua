import numpy as np
from core.factors.helpers.blend_base import SmallCapMVFinanceBlend

class SmallCapDailyMVMaskRoe2xBottom10(SmallCapMVFinanceBlend):
  """ROE 2x weight variant — roe=0.20, selects bottom 10% by composite score."""
  _field_weights = {"roe": 0.20}
