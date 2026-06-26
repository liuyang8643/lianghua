"""llm_ga.guard 静态红线校验单元测试。"""
from llm_ga import guard

_OK = '''
import numpy as np
ALPHA = 0.5
class Good:
    hist_days = 0
    def calc_batch(self, panel):
        o = panel['open']
        base_valid = np.isfinite(o) & (o >= 2.0) & ~panel['st_mask']
        score = np.log(o) + ALPHA * np.log(panel['total_share'] + 1.0)
        return np.where(base_valid & np.isfinite(score), score, np.nan)
'''


def test_good_factor_passes():
    ok, n, reason = guard.check(_OK, param_cap=20)
    assert ok and reason == '' and n == 1


def test_leak_close_rejected():
    code = _OK.replace("np.log(o)", "panel['close']")
    ok, _, reason = guard.check(code, 20)
    assert not ok and '泄露' in reason


def test_loop_rejected():
    code = _OK.replace(
        "score = np.log(o) + ALPHA * np.log(panel['total_share'] + 1.0)",
        "score = np.zeros_like(o)\n        for i in range(o.shape[1]):\n            score[:, i] = o[:, i]",
    )
    ok, _, reason = guard.check(code, 20)
    assert not ok and '向量化' in reason


def test_bad_import_rejected():
    code = "import requests\n" + _OK
    ok, _, reason = guard.check(code, 20)
    assert not ok and 'import' in reason


def test_file_read_rejected():
    code = _OK.replace("o = panel['open']", "o = np.load('x.npy')")
    ok, _, reason = guard.check(code, 20)
    assert not ok and '外部文件' in reason


def test_param_cap_rejected():
    extra = '\n'.join(f'P{i} = {i}' for i in range(25))
    code = extra + '\n' + _OK
    ok, n, reason = guard.check(code, 20)
    assert not ok and '参数超预算' in reason and n > 20
