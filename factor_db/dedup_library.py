"""因子库去重：删除"行为近似一样"的重复因子，每簇保留最早的一个。

判定口径（唯一标准）：因子的**每日截面股票 rank 相关**。两因子的截面 rank 指纹
（similarity.signature）相关 >= CLONE_CORR（0.99，近克隆）即视为行为克隆。
不使用收益序列 / 收益相关性——选股能力的"相同/不同"只看截面 rank。

无指纹（不在 signatures 缓存）或零指纹（输出全 NaN/常数）的因子无法判定，保留不动。

破坏性操作：会删除 factors/<name>.py、registry.db 中 factors 与 factor_runs 的对应行，
并精简指纹缓存。registry.db 默认 append-only（触发器禁 DELETE），本脚本临时摘除触发器删除后
再装回。执行前自动备份 registry.db -> registry.db.bak。

用法:
  uv run python -m factor_db.dedup_library          # 执行删除
  uv run python -m factor_db.dedup_library --dry-run # 只看将删除哪些
"""
import argparse
import shutil
import sqlite3

import numpy as np

from factor_db import db, records, similarity

_FACTORS_DIR = db._FACTORS_DIR


def find_duplicates() -> tuple[list[str], dict[int, list[dict]]]:
    """返回 (待删除因子名列表, 重复簇 {root: [factor,...]，仅含 size>1 的簇})。

    簇 = 截面 rank 指纹两两相关 >= CLONE_CORR 的连通分量；每簇保留 factor_id 最小者。
    """
    names, sigs, _ = similarity.load_cache()
    by_name = {f['name']: f for f in db.list_factors()}

    items = [(n, i) for i, n in enumerate(names)
             if n in by_name and float(np.linalg.norm(sigs[i])) >= 1e-6]
    if len(items) < 2:
        return [], {}

    sub_names = [n for n, _ in items]
    sub_sigs = np.array([sigs[i] for _, i in items], dtype=np.float32)
    corr = similarity.correlation_matrix(sub_sigs)

    n = len(sub_names)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] >= similarity.CLONE_CORR:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[dict]] = {}
    for i, name in enumerate(sub_names):
        groups.setdefault(find(i), []).append(by_name[name])

    def _sh(f):
        return f['train_sharpe'] if f['train_sharpe'] is not None else float('-inf')

    delete, clusters = [], {}
    for root, fs in groups.items():
        if len(fs) < 2:
            continue
        # 保留夏普最高的代表（与 llm-ga“新高覆盖旧”一致）；夏普相同则保留 factor_id 最小（最早）。
        fs.sort(key=lambda x: (_sh(x), -x['factor_id']), reverse=True)
        clusters[root] = fs
        delete.extend(f['name'] for f in fs[1:])
    return delete, clusters


def _delete_rows(names: list[str]) -> None:
    """临时摘除 append-only 触发器，删除 factors / factor_runs 行，再装回触发器。"""
    conn = sqlite3.connect(db._DB_PATH)
    try:
        for trg in ('factors_no_update', 'factors_no_delete',
                    'factor_runs_no_update', 'factor_runs_no_delete'):
            conn.execute(f'DROP TRIGGER IF EXISTS {trg}')
        qmarks = ','.join('?' * len(names))
        conn.execute(f'DELETE FROM factors WHERE name IN ({qmarks})', names)
        conn.execute(f'DELETE FROM factor_runs WHERE factor_name IN ({qmarks})', names)
        conn.commit()
        conn.executescript(db._TRIGGERS)
        conn.executescript(records._TRIGGERS)
        conn.commit()
    finally:
        conn.close()


def _prune_cache(deleted: set[str]) -> None:
    names, sigs, meta = similarity.load_cache()
    if not names:
        return
    keep = [i for i, n in enumerate(names) if n not in deleted]
    similarity.save_cache([names[i] for i in keep], sigs[keep], meta)


def remove_factors(names: list[str]) -> int:
    """彻底移除若干因子：删 factors/<name>.py、registry.db 行(factors+factor_runs)、指纹缓存条目。
    返回删除的因子文件数。无备份——调用方按需自行备份（库去重 run() 会先备份）。"""
    if not names:
        return 0
    removed = 0
    for n in names:
        p = _FACTORS_DIR / f'{n}.py'
        if p.exists():
            p.unlink()
            removed += 1
    _delete_rows(names)
    _prune_cache(set(names))
    return removed


def run(dry_run: bool = False) -> None:
    delete, clusters = find_duplicates()
    print(f'发现 {len(clusters)} 个"截面 rank 完全相同"的簇，共 {len(delete)} 个重复因子待删除：')
    for fs in sorted(clusters.values(), key=lambda x: -len(x)):
        rep = fs[0]['name']
        dups = [f['name'] for f in fs[1:]]
        print(f'  保留 #{fs[0]["factor_id"]} {rep}（夏普={fs[0]["train_sharpe"]}）'
              f' → 删除 {len(dups)} 个: {", ".join(dups[:4])}{" ..." if len(dups) > 4 else ""}')

    if not delete:
        print('无重复因子，无需删除。')
        return
    if dry_run:
        print('\n[dry-run] 未执行删除。')
        return

    backup = db._DB_PATH.with_suffix('.db.bak')
    shutil.copyfile(db._DB_PATH, backup)
    print(f'\n已备份 registry.db -> {backup}')

    removed_files = remove_factors(delete)

    print(f'删除完成：因子文件 {removed_files} 个，db 行 {len(delete)} 个（factors + factor_runs），指纹缓存已精简。')
    print(f'剩余因子数：{len(db.list_factors())}')


def main():
    p = argparse.ArgumentParser(description='因子库去重（按截面 rank 指纹删除行为完全一样的重复因子）')
    p.add_argument('--dry-run', action='store_true', help='只列出将删除哪些，不实际删除')
    args = p.parse_args()
    run(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
