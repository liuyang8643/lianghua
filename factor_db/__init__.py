"""因子库代码与本地研究产物路径。

- factor_db/factors/  : 现役因子代码；已禁用因子不能注册或进入计算。
- artifacts/factor_discovery/registry.db: 本地 SQLite 血缘/指标记录。
- factor_db/db.py      : 读写封装，仅提供 add_factor / 查询，绝不提供 update/delete。
"""

from pathlib import Path


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "factor_discovery"
REGISTRY_PATH = ARTIFACT_DIR / "registry.db"
REPORT_PATH = ARTIFACT_DIR / "report.html"


FORBIDDEN_FACTOR_NAMES = frozenset({"OvernightGapDown"})


def require_allowed_factor_names(names) -> None:
    """One admission rule for factors, candidate registration and ML inputs."""
    names = (names,) if isinstance(names, str) else names
    forbidden = {name.casefold() for name in FORBIDDEN_FACTOR_NAMES}
    for name in names:
        if not isinstance(name, str):
            raise TypeError("factor names must be strings")
        if any(part.casefold() in forbidden for part in name.split("__")):
            raise ValueError(f"factor is prohibited by user policy: {name}")


__all__ = ["ARTIFACT_DIR", "REGISTRY_PATH", "REPORT_PATH", "require_allowed_factor_names"]
