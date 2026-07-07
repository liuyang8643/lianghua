"""因子代码静态红线校验（评测前的第一道闸门，纯静态、确定性）。

逐条对应 CLAUDE.md 红线：
1. T 日价格红线（尾盘收盘交易）：禁止引用 open/high/low/volume/amount。新因子只能用 close[T] + 财务/滞后量。
2. 矩阵计算红线：calc_batch 必须纯向量化，禁止任何 for/while 逐股票 / 逐日循环。
3. 自包含 / 数据源红线：除 numpy 外禁止 import 任何库（含 akshare/requests/xtdata 等联网库），
   禁止 open()/np.load 等读取外部文件。
4. 参数预算 cap：模块级 UPPER_CASE 常量个数不得超过 param_cap，抑制过度参数化导致的过拟合。

任一违规 → 直接 reject，不进回测。
"""
import ast
import re

from llm_ga.config import FORBIDDEN_FIELDS

_FIELD_ACCESS = re.compile(
    r"""(?:panel|data|factor_data)\s*\[\s*['"](%s)['"]\s*\]""" % '|'.join(FORBIDDEN_FIELDS)
)
_FIELD_GET = re.compile(
    r"""\.get\(\s*['"](%s)['"]""" % '|'.join(FORBIDDEN_FIELDS)
)
_UPPER_CONST = re.compile(r"""^[A-Z][A-Z0-9_]*\s*=""", re.MULTILINE)
_CLASS = re.compile(r"""^\s*class\s+(\w+)""", re.MULTILINE)

_ALLOWED_IMPORTS = {'numpy'}
# 禁止的文件读取调用：open(...) 与 numpy 的磁盘读取族。
_FORBIDDEN_FILE_CALLS = {'open', 'load', 'loadtxt', 'fromfile', 'genfromtxt', 'memmap', 'fromregex'}


def count_params(code: str) -> int:
    return len(_UPPER_CONST.findall(code))


def find_leak_fields(code: str) -> list[str]:
    hits = _FIELD_ACCESS.findall(code) + _FIELD_GET.findall(code)
    return sorted(set(hits))


def _find_bad_imports(tree: ast.AST) -> list[str]:
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split('.')[0] not in _ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or '').split('.')[0]
            if root not in _ALLOWED_IMPORTS:
                bad.append(node.module or '?')
    return sorted(set(bad))


def _has_loop(tree: ast.AST) -> bool:
    return any(isinstance(n, (ast.For, ast.AsyncFor, ast.While)) for n in ast.walk(tree))


def _find_file_calls(tree: ast.AST) -> list[str]:
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == 'open':
            bad.append('open')
        elif isinstance(fn, ast.Attribute) and fn.attr in _FORBIDDEN_FILE_CALLS:
            bad.append(fn.attr)
    return sorted(set(bad))


def check(code: str, param_cap: int) -> tuple[bool, int, str]:
    """返回 (是否通过, 参数个数, 原因)。通过时原因为空串。"""
    if 'calc_batch' not in code:
        return False, 0, '缺少 calc_batch 方法'
    if not _CLASS.search(code):
        return False, 0, '缺少因子类定义'

    leaks = find_leak_fields(code)
    if leaks:
        return False, 0, f'T 日数据泄露：引用了禁止字段 {leaks}（仅允许 close[T]）'

    tree = ast.parse(code)

    bad_imports = _find_bad_imports(tree)
    if bad_imports:
        return False, 0, f'非自包含：禁止 import {bad_imports}（仅允许 numpy）'

    file_calls = _find_file_calls(tree)
    if file_calls:
        return False, 0, f'禁止读取外部文件：检出 {file_calls}（因子只能用 panel 字段）'

    if _has_loop(tree):
        return False, 0, '矩阵计算红线：检出 for/while 循环（calc_batch 必须纯向量化，禁止逐股票/逐日遍历）'

    n_params = count_params(code)
    if n_params > param_cap:
        return False, n_params, f'参数超预算：{n_params} > {param_cap} 个 UPPER_CASE 常量'

    return True, n_params, ''
