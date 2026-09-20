"""因子库代码与本地研究产物路径。

- factor_db/factors/  : 因子代码（不可变，只增不删不改），类名与原生产因子完全一致。
- artifacts/factor_discovery/registry.db: 本地 SQLite 血缘/指标记录。
- factor_db/db.py      : 读写封装，仅提供 add_factor / 查询，绝不提供 update/delete。
"""

from pathlib import Path


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "factor_discovery"
REGISTRY_PATH = ARTIFACT_DIR / "registry.db"
REPORT_PATH = ARTIFACT_DIR / "report.html"


__all__ = ["ARTIFACT_DIR", "REGISTRY_PATH", "REPORT_PATH"]
