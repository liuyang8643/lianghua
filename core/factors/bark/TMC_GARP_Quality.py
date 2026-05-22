import numpy as np
from core.factors.helpers.blend_base import SmallCapMVFinanceBlend

class TMC_GARP_Quality(SmallCapMVFinanceBlend):
  """小盘+增长+价值+质量: profit_yoy(0.20) + eps(0.10) + operating_cf_ps(0.05)"""
  _field_weights = {"profit_yoy": 0.20, "eps": 0.10, "operating_cf_ps": 0.05}
