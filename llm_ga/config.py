"""LLM-GA 运行配置。

fitness 只看夏普、只用训练周期（默认 1993-2018，可经 CLI 覆盖）。
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FACTORS_DIR = REPO_ROOT / 'factor_db' / 'factors'
PROPOSALS_DIR = REPO_ROOT / 'llm_ga' / 'proposals'
SCRATCH_DIR = REPO_ROOT / 'llm_ga' / '_scratch'

# 训练周期（fitness 来源）
TRAIN_START = '19930101'
TRAIN_END = '20181231'

# T 日仅允许 open；以下字段为前视野泄露，新因子禁止使用（guard 强制）。
FORBIDDEN_FIELDS = ('close', 'high', 'low', 'volume', 'amount')

# 因子可用 panel 字段（写进 prompt 上下文）
ALLOWED_FIELDS = (
    'open', 'st_mask', 'total_share', 'issue_price',
    'eps', 'roe', 'gross_margin', 'operating_cf_ps', 'profit_yoy', 'revenue_yoy',
)


@dataclass
class RunConfig:
    """进化运行配置。

    种群语义（population = 父代 + 子代）：每代选 n_parents 个父代（n_elite 个上一轮最优保留 +
    其余从全库 top50% 随机挑），父代【不重新评测/不再过 LLM】；用随机的交叉/变异让 LLM 产
    n_offspring 个全新子代，仅子代过闸门 + 回测 + 入库。默认 population=10 → 5 父代 + 5 子代。
    """
    start: str = TRAIN_START
    end: str = TRAIN_END
    buy_n: int = 20
    pool_prefixes: tuple = ('60', '00', '30', '688')
    pool_label: str = 'all_A'
    generations: int = 1
    population: int = 10
    n_offspring: int = 5          # 每代期望 LLM 产出的子代数
    n_parents: int = 5            # 每代父代数（含 elite）
    n_elite: int = 1              # 上一轮最优保留为父代的个数
    param_cap: int = 20
    crossover_ratio: float = 0.3  # 每个子代按此概率走交叉，否则走变异（MadEvolve ~30%交叉/70%变异）
    n_parents_crossover: int = 5  # 交叉子代参考的父代数（≤ n_parents）
    n_inspirations: int = 3       # 变异时附带的"灵感因子"数（全局最优若干，仅供参考勿照抄）
    concurrency: int = 5          # 同一代内并发的 LLM 产因子子进程数（I/O 并发，不违反 CPU 多进程红线）
    model: str = 'deepseek-v4-pro'         # 产因子（创新，防克隆）
    verify_model: str = 'deepseek-v4-flash'  # verify 红线审查（静态检查，轻模型即可，快得多）
    seed: int = 42
    llm_verify: bool = True  # 是否启用 claude -p 的 LLM verify-agent 硬否决闸门
    run_id: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))
    extra_env: dict = field(default_factory=dict)
