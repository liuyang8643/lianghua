"""因子自动发现入口：扫描 append-only 因子库 `factor_db.factors`。"""

import importlib
import pkgutil
from typing import Dict, Type, Any


_registry: Dict[str, Type] = {}

_FACTOR_PACKAGE = 'factor_db.factors'


def _discover():
    """扫描所有因子包下子模块，自动注册含 calc_batch 的类"""
    global _registry
    if _registry:
        return _registry

    pkg = importlib.import_module(_FACTOR_PACKAGE)
    for _, name, is_pkg in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + '.'):
        if is_pkg:
            continue
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
