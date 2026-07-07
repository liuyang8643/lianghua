"""LLM-GA 运行配置。

fitness 只看夏普、只用训练周期（默认 2010-至今，可经 CLI 覆盖）。
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FACTORS_DIR = REPO_ROOT / 'factor_db' / 'factors'
PROPOSALS_DIR = REPO_ROOT / 'llm_ga' / 'proposals'
SCRATCH_DIR = REPO_ROOT / 'llm_ga' / '_scratch'

# 训练周期（fitness / factor_db 榜单默认区间）
TRAIN_START = '20100101'
TRAIN_END = datetime.now().strftime('%Y%m%d')

# 尾盘收盘交易：T 日仅允许 close；以下当日字段禁止用于选股（guard 强制，统一口径）。
FORBIDDEN_FIELDS = ('open', 'high', 'low', 'volume', 'amount')

# 因子可用 panel 字段（写进 prompt 上下文）
ALLOWED_FIELDS = (
    'close', 'st_mask', 'total_share', 'issue_price',
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
    model: str = 'deepseek-v4-pro'         # 产因子（创新，防克隆）
    verify_model: str = 'deepseek-v4-flash'  # verify 红线审查（静态检查，轻模型即可，快得多）
    seed: int = 42
    llm_verify: bool = True  # 是否启用 claude -p 的 LLM verify-agent 硬否决闸门
    run_id: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))
    extra_env: dict = field(default_factory=dict)
    core_factors: list[str] | None = None  # 限定候选父代池；None=全库


# ── 预设：限定因子池的 GA 配置 ──

_CORE5 = [
    'TrueMarketCap',
    'High52Week',
    'Factor_20260531_005210_g17_4',
    'ProfitGrowth',
    'CashFlowYield',
]

_CORE9 = _CORE5 + [
    'Factor_20260531_005210_g39_1',
    'Factor_20260531_005210_g16_1',
    'ROEQuality',
    'Factor_20260531_005210_g29_2',
]

_CORE14 = _CORE9 + [
    'Factor_20260531_005210_g29_1',
    'Factor_20260531_005210_g37_2',
    'Factor_20260531_005210_g17_0',
    'EarningsYield',
    'AmountBasedSmallCap',
]


def preset_core5(**kwargs) -> RunConfig:
    """精简版：5因子，覆盖规模/动量/价值/增长/GA最佳。"""
    return RunConfig(
        n_parents=3, n_offspring=3, n_elite=1, population=6,
        n_inspirations=2, core_factors=list(_CORE5), **kwargs)


def preset_core9(**kwargs) -> RunConfig:
    """标准版：9因子，补齐质量/相位套利/高多样性锚。"""
    return RunConfig(
        n_parents=4, n_offspring=4, n_elite=1, population=8,
        n_inspirations=3, core_factors=list(_CORE9), **kwargs)


def preset_core14(**kwargs) -> RunConfig:
    """完整版：14因子，最全alpha模板库。"""
    return RunConfig(
        n_parents=6, n_offspring=6, n_elite=2, population=12,
        n_inspirations=4, core_factors=list(_CORE14), **kwargs)
