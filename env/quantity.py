"""Single authority for exchange order-quantity semantics."""

from __future__ import annotations


QUANTITY_SCHEMA_VERSION = "wbr-board-quantity-v1"


def _bare_code(code: str) -> str:
    return str(code).split(".", 1)[0]


def is_kcb_stock(code: str) -> bool:
    return _bare_code(code).startswith("688")


def minimum_buy_quantity(code: str) -> int:
    return 200 if is_kcb_stock(code) else 100


def buy_quantity_step(code: str) -> int:
    return 1 if is_kcb_stock(code) else 100


def minimum_partial_sell_quantity(code: str) -> int:
    return 200 if is_kcb_stock(code) else 100


def partial_sell_quantity_step(code: str) -> int:
    return 1 if is_kcb_stock(code) else 100


def floor_buy_quantity(code: str, quantity: int | float) -> int:
    quantity = int(quantity)
    minimum = minimum_buy_quantity(code)
    if quantity < minimum:
        return 0
    step = buy_quantity_step(code)
    return minimum + (quantity - minimum) // step * step


def round_buy_quantity(code: str, quantity: int | float) -> int:
    quantity = int(quantity)
    if quantity <= 0:
        return 0
    minimum = minimum_buy_quantity(code)
    if quantity < minimum:
        return minimum
    return floor_buy_quantity(code, quantity)


def floor_partial_sell_quantity(code: str, quantity: int | float) -> int:
    quantity = int(quantity)
    minimum = minimum_partial_sell_quantity(code)
    if quantity < minimum:
        return 0
    step = partial_sell_quantity_step(code)
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
        "kcb_688": {
            "minimum_buy": 200,
            "buy_step": 1,
            "minimum_partial_sell": 200,
            "partial_sell_step": 1,
        },
        "full_liquidation_sentinel": -1,
    }


__all__ = [
    "QUANTITY_SCHEMA_VERSION",
    "buy_quantity_step",
    "floor_buy_quantity",
    "floor_partial_sell_quantity",
    "is_kcb_stock",
    "minimum_buy_quantity",
    "minimum_partial_sell_quantity",
    "partial_sell_quantity_step",
    "quantity_schema_manifest",
    "round_buy_quantity",
]
