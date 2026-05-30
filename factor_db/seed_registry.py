"""把 factor_db/factors 下所有现存因子作为 seed(generation=0) 登记进 registry.db。

幂等：已登记（name 已存在）的因子跳过，不重复写入（append-only，不更新）。
用法: python factor_db/seed_registry.py
"""
import inspect
from pathlib import Path

from core.factors.registry import get_all_factor_classes
from factor_db import db

_ROOT = Path(__file__).resolve().parent.parent


def main():
    db.init_db()
    classes = get_all_factor_classes()
    added, skipped = [], []
    for name in sorted(classes):
        if db.exists(name):
            skipped.append(name)
            continue
        cls = classes[name]
        abs_path = Path(inspect.getfile(cls)).resolve()
        rel_path = abs_path.relative_to(_ROOT).as_posix()
        fid = db.add_factor(
            name=name,
            file_path=rel_path,
            op='seed',
            generation=0,
            params_count=0,
            status='active',
            parent_ids=None,
        )
        added.append((fid, name, rel_path))

    print(f'新增 {len(added)} 个 seed 因子:')
    for fid, name, rel in added:
        print(f'  #{fid:<3} {name:<22} {rel}')
    if skipped:
        print(f'已存在跳过 {len(skipped)} 个: {", ".join(skipped)}')


if __name__ == '__main__':
    main()
