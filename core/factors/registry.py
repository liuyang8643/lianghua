"""因子自动发现 — 扫描 core.factors 包下所有含 calc_batch 的类"""

import importlib
import pkgutil
from typing import Dict, Type, Any


_registry: Dict[str, Type] = {}


def _discover():
    """扫描 core/factors/ 下所有子模块，自动注册含 calc_batch 的类"""
    global _registry
    if _registry:
        return _registry

    import core.factors as pkg
    for _, name, is_pkg in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + '.'):
        if is_pkg:
            continue  # 跳过 helpers/bark/results 等子包
        mod = importlib.import_module(name)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and hasattr(obj, 'calc_batch') and hasattr(obj, 'hist_days'):
                _registry[obj.__name__] = obj

    return _registry


def get_factor_class(name: str) -> Type:
    return _discover()[name]


def get_factor_names() -> list:
    return list(_discover().keys())


def get_all_factor_classes() -> Dict[str, Type]:
    return dict(_discover())
