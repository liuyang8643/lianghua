"""因子库 DB：append-only 的因子代码仓库 + SQLite 登记表。

- factor_db/factors/  : 因子代码（不可变，只增不删不改），类名与原生产因子完全一致。
- factor_db/registry.db: SQLite 登记表，记录每个因子的血缘/指标元数据。
- factor_db/db.py      : 读写封装，仅提供 add_factor / 查询，绝不提供 update/delete。
"""
