"""Numba registration for the shared Python 3.6 quantity rules."""
from numba.extending import register_jitable
from utils.stock.quantity import (
    QUANTITY_SCHEMA_VERSION, buy_quantity_step, floor_buy_quantity,
    floor_partial_sell_quantity, floor_quantity, is_bj_stock, is_kcb_stock,
    minimum_buy_quantity, quantity_schema_manifest,
)
register_jitable(inline="always")(floor_quantity)
