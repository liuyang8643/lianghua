"""Claude Code（headless CLI）封装 —— 受限工具，只在 scratch 目录新增因子文件。

复用本地 claude code 配置（~/.claude/settings.json，已指向 deepseek）。Python claude-agent-sdk
与本项目 mootdx 的 httpx 版本不可共存，故直接调用本地 `claude` CLI 的 -p（headless）模式，
等价于 SDK 形式且零依赖冲突。

权限边界：cwd 设为本代次专属 scratch 目录，allowedTools 仅 Write/Read，permission-mode
acceptEdits（自动接受写入）。模型只能在 scratch 下新增文件；不可 Bash/联网/改动仓库其它文件。
"""
import json
import os
import subprocess
from pathlib import Path

from llm_ga.config import ALLOWED_FIELDS, SCRATCH_DIR

_PROMPTS = Path(__file__).resolve().parent / 'prompts'
_SETTINGS = Path.home() / '.claude' / 'settings.json'

_CONTRACT = """因子契约（必须严格遵守）：
- 定义 `class <Name>:`，含类属性 `hist_days = <int>`（无滚动窗口则为 0），以及方法
  `def calc_batch(self, panel: dict) -> np.ndarray`。
- 返回 float 数组，形状 (n_dates, n_stocks)；分数【越高越好】（越优先买入）；
  无效/不合格的股票置为 np.nan。
- 纯 numpy、完全【向量化】。禁止任何 python 逐股票 / 逐日循环（含 for/while 对股票或日期维度的遍历）。
- T 日泄露红线：当日价格【只允许】用 panel['open']。绝对禁止引用
  panel['close']、panel['high']、panel['low']、panel['volume']、panel['amount']。
- 可用 panel 字段：{fields}。其中财务字段（eps/roe/gross_margin/operating_cf_ps/
  profit_yoy/revenue_yoy）是 point-in-time 滞后口径，当日使用安全。
- 必须【自包含】：只能用上面 panel 字典里的字段计算。禁止调用 _aligned/_load 等辅助函数，
  禁止读取任何外部文件 / npz，除 numpy 外禁止 import 任何库（尤其禁止 akshare/requests/
  mootdx/xtdata 等联网库），禁止 open()/文件读写。
- 合法域：base_valid = ~np.isnan(panel['open']) & (panel['open'] >= 2.0) & ~panel['st_mask']。
- 连续分数红线（硬性闸门，不满足直接拒绝）：分数对【每一只 base_valid 股票】都必须是
  【连续、无重复】的实数。具体要求：
  * 覆盖率：每日至少 90% 的 base_valid 股票要拿到有限分数。禁止用基本面条件收缩股票池
    （例如 eps>0、或要求 eps/roe/... 全部 finite）。要对【整个 base_valid 全集】打分。
  * 禁止 tie：两只 base_valid 股票几乎不应出现相同分数。因此禁止离散化：不得用
    np.sign / np.round / 分位分桶标签 / clip 成常数 / 布尔×常数 / 把大量股票置为 0.0 或同一常数。
  * 财务缺失值：在截面上做插补（如 NaN -> 当日 nanmedian，或 z-score 时把 NaN 当 0），
    不要因此丢弃该股票。插补后必须再【混入一个对所有股票都存在的连续项】（如对
    z-score 化的 eps/open 给小权重，或用 total_share*open 取对数市值），避免被插补的股票挤成同一个值。
  * 禁止对【可能为负】的输入做 sqrt/log/分数次幂（会产生 NaN 而丢股票）。改用带符号变换，
    如 np.sign(x)*np.sqrt(np.abs(x)) 或 np.tanh(x)。
  * 最后一行：return np.where(base_valid & np.isfinite(score), score, np.nan)，
    且 score 对（几乎）所有 base_valid 股票都为有限值。
- 可调参数必须是模块级 UPPER_CASE 常量；总数 <= {cap}。
- 在模块级定义 `__thesis__ = "..."`：一句话中文思路（单行、<=40 字），描述该因子的核心逻辑。
  只用双引号，引号内不要再出现引号或换行。这是元数据，不算可调参数（小写 dunder，不计入）。
- 文件顶部 `import numpy as np`。无需 docstring、无需注释。""".format(
    fields=', '.join(ALLOWED_FIELDS), cap='{cap}')


def _settings_env() -> dict:
    env = dict(os.environ)
    if _SETTINGS.exists():
        data = json.loads(_SETTINGS.read_text(encoding='utf-8'))
        for k, v in (data.get('env') or {}).items():
            env[str(k)] = str(v)
    return env


def _format_parents(parents: list[dict]) -> str:
    blocks = []
    for p in parents:
        head = f"### {p['name']}"
        if p.get('train_sharpe') is not None:
            head += f"  (train Sharpe = {p['train_sharpe']:.3f})"
        blocks.append(f"{head}\n```python\n{p['code']}\n```")
    return '\n\n'.join(blocks)


def _format_inspirations(inspirations: list[dict]) -> str:
    if not inspirations:
        return '（无）'
    return _format_parents(inspirations)


def _build_prompt(op: str, parents: list[dict], names: list[str], param_cap: int,
                  inspirations: list[dict] | None = None) -> str:
    template = (_PROMPTS / f'{op}.md').read_text(encoding='utf-8')
    return (
        template
        .replace('<<N>>', str(len(names)))
        .replace('<<CONTRACT>>', _CONTRACT.format(cap=param_cap))
        .replace('<<PARENTS>>', _format_parents(parents))
        .replace('<<INSPIRATIONS>>', _format_inspirations(inspirations or []))
        .replace('<<NAMES>>', ', '.join(names))
    )


def _run_claude(prompt: str, scratch: Path, model: str) -> subprocess.CompletedProcess:
    cmd = (
        f'claude -p --output-format json --permission-mode acceptEdits '
        f'--allowedTools Write Read --model {model}'
    )
    return subprocess.run(
        cmd, shell=True, cwd=str(scratch), input=prompt, text=True,
        capture_output=True, env=_settings_env(), timeout=900,
    )


def propose(op: str, parents: list[dict], names: list[str], model: str,
            param_cap: int, tag: str, inspirations: list[dict] | None = None) -> dict[str, Path]:
    """让 claude 在 scratch 目录生成期望命名的因子文件，返回 {name: path}（只含实际写出的）。

    op='mutation' 时 parents 应为单个父代，inspirations 为若干灵感因子；op='crossover' 时
    parents 为多个父代。多个并发调用各用独立 tag → scratch 目录隔离，线程并发安全。
    """
    scratch = SCRATCH_DIR / tag
    scratch.mkdir(parents=True, exist_ok=True)
    prompt = _build_prompt(op, parents, names, param_cap, inspirations)
    proc = _run_claude(prompt, scratch, model)

    written = {}
    for name in names:
        f = scratch / f'{name}.py'
        if f.exists():
            written[name] = f
    if not written:
        tail = (proc.stdout or proc.stderr or '')[-500:]
        raise RuntimeError(f'claude 未产出任何期望文件（rc={proc.returncode}）: {tail}')
    return written
