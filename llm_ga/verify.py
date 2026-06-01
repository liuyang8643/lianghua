"""LLM verify-agent（claude -p 硬否决闸门）：对生成的因子代码做红线评审。

定位：确定性 guard（静态）通过后的【第二道硬闸门】。模型只做只读代码审查，返回 PASS/FAIL，
因子必须两道闸门都过才进入回测与入库。无法解析的输出按 FAIL 处理（保守）。

复用本地 claude 配置（~/.claude/settings.json，已指向 deepseek），与 agent.py 同一套 CLI 形式。
"""
import json
import re
import subprocess
from pathlib import Path

from llm_ga.agent import _settings_env
from llm_ga.config import ALLOWED_FIELDS, FORBIDDEN_FIELDS

_PROMPTS = Path(__file__).resolve().parent / 'prompts'
_VERDICT = re.compile(r'VERDICT\s*[:：]\s*(PASS|FAIL)\s*[:：]?\s*(.*)', re.IGNORECASE)


def _build_prompt(code: str) -> str:
    tpl = (_PROMPTS / 'verify.md').read_text(encoding='utf-8')
    return (
        tpl
        .replace('<<FIELDS>>', ', '.join(ALLOWED_FIELDS))
        .replace('<<FORBIDDEN>>', ', '.join(FORBIDDEN_FIELDS))
        .replace('<<CODE>>', code)
    )


def _extract_text(proc: subprocess.CompletedProcess) -> str:
    try:
        return json.loads(proc.stdout)['result']
    except (json.JSONDecodeError, KeyError, TypeError):
        return proc.stdout or proc.stderr or ''


def review(code: str, model: str, timeout: int = 600) -> tuple[bool, str]:
    """对因子代码做红线评审，返回 (是否通过, 原因)。

    超时 / 子进程异常一律按【保守 FAIL】处理并返回，绝不向上抛异常（单次 LLM 调用失败
    不能崩掉整条进化长跑）。
    """
    try:
        proc = subprocess.run(
            f'claude -p --output-format json --model {model}',
            shell=True, input=_build_prompt(code), text=True,
            capture_output=True, env=_settings_env(), timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f'verify 超时(>{timeout}s)，保守判 FAIL'
    except Exception as e:
        return False, f'verify 调用异常({type(e).__name__}: {e})，保守判 FAIL'
    text = _extract_text(proc)
    for line in reversed(text.strip().splitlines()):
        m = _VERDICT.search(line)
        if m:
            return m.group(1).upper() == 'PASS', m.group(2).strip()
    return False, f'verify 输出无法解析为 VERDICT（rc={proc.returncode}）: {text[-200:]}'
