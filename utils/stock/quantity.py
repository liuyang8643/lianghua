"""Exchange quantity rules shared by research and Python 3.6 execution."""
from functools import lru_cache

QUANTITY_SCHEMA_VERSION = 'wbr-board-quantity-v3-kcb-bj-one-share-step'

def is_kcb_stock(code):
    return str(code).split('.', 1)[0].startswith(('688', '689'))

def is_bj_stock(code):
    return str(code).upper().endswith('.BJ')

@lru_cache(maxsize=8192)
def _quantity_rules(code):
    kcb = is_kcb_stock(code)
    return (200 if kcb else 100, 1 if kcb or is_bj_stock(code) else 100)

def minimum_buy_quantity(code):
    return _quantity_rules(code)[0]

def buy_quantity_step(code):
    return _quantity_rules(code)[1]

def floor_quantity(quantity, minimum, step):
    quantity = int(quantity)
    if quantity < minimum:
        return 0
    return minimum + (quantity - minimum) // step * step

def floor_buy_quantity(code, quantity):
    return floor_quantity(quantity, *_quantity_rules(code))

def floor_partial_sell_quantity(code, quantity):
    return floor_quantity(quantity, *_quantity_rules(code))

def quantity_schema_manifest():
    def rule(minimum, step):
        return dict(minimum_buy=minimum, buy_step=step,
                    minimum_partial_sell=minimum, partial_sell_step=step)
    return dict(schema_version=QUANTITY_SCHEMA_VERSION,
                standard_a_share=rule(100, 100), kcb_688_689=rule(200, 1),
                beijing_exchange=rule(100, 1), full_liquidation_sentinel=-1)
