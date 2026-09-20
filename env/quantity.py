"""Single authority for exchange order-quantity semantics."""

from __future__ import annotations

from functools import lru_cache
from numba.extending import register_jitable


QUANTITY_SCHEMA_VERSION = "wbr-board-quantity-v3-kcb-bj-one-share-step"


def _bare_code(code: str) -> str:
    return str(code).split(".", 1)[0]


def is_kcb_stock(code: str) -> bool:
    return _bare_code(code).startswith(("688", "689"))


def is_bj_stock(code: str) -> bool:
    return str(code).upper().endswith(".BJ")


@lru_cache(maxsize=8192)
def _quantity_rules(code: str) -> tuple[int, int]:
    kcb = is_kcb_stock(code)
    return (200 if kcb else 100, 1 if kcb or is_bj_stock(code) else 100)


def minimum_buy_quantity(code: str) -> int:
    return _quantity_rules(code)[0]


def buy_quantity_step(code: str) -> int:
    return _quantity_rules(code)[1]


def floor_buy_quantity(code: str, quantity: int | float) -> int:
    minimum, step = _quantity_rules(code)
    return floor_quantity(quantity, minimum, step)


def floor_partial_sell_quantity(code: str, quantity: int | float) -> int:
    minimum, step = _quantity_rules(code)
    return floor_quantity(quantity, minimum, step)


@register_jitable(inline='always')
def floor_quantity(quantity: int | float, minimum: int, step: int) -> int:
    """Apply already resolved exchange terms to a requested share count."""
    quantity = int(quantity)
    if quantity < minimum:
        return 0
    return minimum + (quantity - minimum) // step * step


def quantity_schema_manifest() -> dict[str, object]:
    """Return the versioned rules consumed by planner and executor alike."""

    return {
        "schema_version": QUANTITY_SCHEMA_VERSION,
        "standard_a_share": {
            "minimum_buy": 100,
            "buy_step": 100,
            "minimum_partial_sell": 100,
            "partial_sell_step": 100,
        },
        "kcb_688_689": {
            "minimum_buy": 200,
            "buy_step": 1,
            "minimum_partial_sell": 200,
            "partial_sell_step": 1,
        },
        "beijing_exchange": {
            "minimum_buy": 100,
            "buy_step": 1,
            "minimum_partial_sell": 100,
            "partial_sell_step": 1,
        },
        "full_liquidation_sentinel": -1,
    }


__all__ = [
    "QUANTITY_SCHEMA_VERSION",
    "buy_quantity_step",
    "floor_buy_quantity",
    "floor_quantity",
    "floor_partial_sell_quantity",
    "is_bj_stock",
    "is_kcb_stock",
    "minimum_buy_quantity",
    "quantity_schema_manifest",
]
