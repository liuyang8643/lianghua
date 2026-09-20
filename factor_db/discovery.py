"""Candidate-factor discovery used only by the offline research registry."""

from __future__ import annotations

import importlib
import pkgutil


def get_all_factor_classes() -> dict[str, type]:
    package = importlib.import_module("factor_db.factors")
    discovered: dict[str, type] = {}
    for module_info in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        if module_info.ispkg:
            continue
        module = importlib.import_module(module_info.name)
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and hasattr(value, "calc_batch")
                and hasattr(value, "hist_days")
            ):
                discovered[value.__name__] = value
    return discovered
