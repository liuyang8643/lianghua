"""
GA 合成景观模块 — 替换回测 evaluator，秒级测量收敛速度。

景观类型:
  - rastrigin:   多峰，局部最优密度极高，经典 GA 早熟测试
  - ackley:      外平内陡，一个全局最优 + 大量浅局部
  - griewank:    大尺度单峰 + 小尺度多峰叠加，模拟"趋势+噪声"
  - nk:          Kauffman NK 景观，可调 ruggedness (K=0 平滑 → K=N-1 混沌)
  - deceptive:   欺骗陷阱，全局最优被低适应度盆地包围

用法:
  from testback.ga_landscape import GALandscape, evaluate_population

  land = GALandscape(kind='rastrigin', dim=10, seed=42)
  # landscape.evaluate(config) → {'sharpe': ..., 'annualized': ..., ...}
"""

import hashlib
import json
import math
import random
from typing import Dict, List

import numpy as np


# ============================================================
# 景观函数（均为最小化问题，返回 negate 后映射到 sharpe 量纲）
# ============================================================

def _rastrigin(x: np.ndarray) -> float:
    """Rastrigin: 全局 min=0 at x=0, ~10^n 个局部最优"""
    n = len(x)
    return 10.0 * n + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x))


def _ackley(x: np.ndarray) -> float:
    """Ackley: 外部近乎平坦，中心陡峭单峰 + 大量浅局部"""
    n = len(x)
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)


def _griewank(x: np.ndarray) -> float:
    """Griewank: 大尺度二次型 + 小尺度 cos 乘积 = 趋势+噪声"""
    sum_sq = np.sum(x ** 2)
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1.0 + sum_sq / 4000.0 - prod_cos


def _sphere(x: np.ndarray) -> float:
    """Sphere: 纯凸单峰，测收敛速度下界"""
    return np.sum(x ** 2)


# ============================================================
# NK 景观 (Kauffman, 1993)
# ============================================================

class _NKLandscape:
    """N 位基因，每位受 K 个其他位上位效应影响。K=0 单峰，K=N-1 混沌。

    预计算适应度查表 (2^N 项)，评估 O(1)。
    """

    def __init__(self, n: int, k: int, seed: int = 42):
        if k >= n:
            k = n - 1
        self.n = n
        self.k = k
        self.table_size = 1 << n
        rng = np.random.default_rng(seed)
        # 每位基因的贡献表：形状 (n, 2^(k+1))
        self.contributions = rng.uniform(0, 1, (n, 1 << (k + 1))).astype(np.float32)
        # 上位邻接：每位受哪 K 个其他位影响
        self.epistasis = np.array([
            [(i + 1 + j) % n for j in range(k)]
            for i in range(n)
        ], dtype=np.int32)

    def _bits_to_int(self, bits: np.ndarray) -> int:
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return val

    def evaluate_bits(self, bits: np.ndarray) -> float:
        """bits: (n,) 0/1 数组 → fitness [0, n]"""
        total = 0.0
        for i in range(self.n):
            idx_bits = [bits[i]]
            for j in self.epistasis[i]:
                idx_bits.append(bits[j])
            idx = self._bits_to_int(idx_bits)
            total += float(self.contributions[i, idx])
        return total

    def evaluate_float(self, x: np.ndarray) -> float:
        """连续 x ∈ [-5,5] → 离散化为 bits → fitness"""
        bits = (x > 0).astype(np.int32)
        return self.evaluate_bits(bits)


# ============================================================
# 欺骗陷阱
# ============================================================

def _deceptive_trap(x: np.ndarray) -> float:
    """每个维度独立 trap: 全局最优在 x≈0，但 x≈±4 处有高适应度盆地"""
    def trap(xi):
        xi_abs = abs(xi)
        if xi_abs <= 1.0:
            return -xi_abs  # 全局最优: xi=0 → 0, 越小越好
        elif xi_abs <= 3.0:
            return xi_abs - 2.0  # 局部盆地
        else:
            return 4.0 - xi_abs  # 远端又被拉低（欺骗性）
    return np.sum([trap(xi) for xi in x])


# ============================================================
# 景观工厂
# ============================================================

_LANDSCAPE_BUILDERS = {
    'rastrigin': lambda dim, seed: ('continuous', lambda x: _rastrigin(x)),
    'ackley': lambda dim, seed: ('continuous', lambda x: _ackley(x)),
    'griewank': lambda dim, seed: ('continuous', lambda x: _griewank(x)),
    'sphere': lambda dim, seed: ('continuous', lambda x: _sphere(x)),
    'nk': lambda dim, seed: ('binary', _NKLandscape(dim, k=max(1, dim // 3), seed=seed).evaluate_float),
    'nk_chaotic': lambda dim, seed: ('binary', _NKLandscape(dim, k=dim - 1, seed=seed).evaluate_float),
    'deceptive': lambda dim, seed: ('continuous', lambda x: _deceptive_trap(x)),
}


# ============================================================
# config → 连续向量 编码器
# ============================================================

class _ConfigEncoder:
    """将 GA individual_config 编码为固定维度的连续向量 [-5, 5]^dim。

    编码规则：
      - 连续权重: 线性映射 [min, max] → [-5, 5]
      - 离散选择: 等距分桶映射到 [-5, 5]
      - buy_n / sell_m / holding_period: 同上
    """

    def __init__(self, profile_name: str | None = None):
        from core.ga import (
            get_profile_search_spaces,
            get_profile_weight_search_spaces,
        )
        self.mappings: list[dict] = []  # 每维一个映射描述

        spaces = get_profile_search_spaces(profile_name)
        weight_spaces = get_profile_weight_search_spaces(profile_name)

        # 离散维度
        for key in ['buy_n', 'holding_period',
                     'timing_base', 'timing_leverage', 'timing_direction',
                     'timing_window', 'timing_index', 'stock_pool', 'factor_choice']:
            space = spaces.get(key)
            if not space:
                continue
            if isinstance(space[0], (int, float)):
                lo, hi = min(space), max(space)
                self.mappings.append({
                    'key': key, 'type': 'numeric',
                    'lo': float(lo), 'hi': float(hi),
                })
            else:
                # 字符串类选择 → 等距分桶
                n = len(space)
                self.mappings.append({
                    'key': key, 'type': 'categorical',
                    'values': list(space),
                })

        # 连续权重维度
        for wk, wv in weight_spaces.items():
            lo, hi = min(wv), max(wv)
            self.mappings.append({
                'key': f'weight_{wk}', 'type': 'weight',
                'lo': float(lo), 'hi': float(hi),
            })

    @property
    def dim(self) -> int:
        return len(self.mappings)

    def encode(self, config: dict) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for i, m in enumerate(self.mappings):
            key = m['key']
            if m['type'] == 'weight':
                val = config.get('weights', {}).get(key.replace('weight_', ''), m['lo'])
            elif key in ('buy_n',):
                val = config.get('buy_n', config.get('sell_m', m['lo']))
            elif key in ('stock_pool',):
                raw = config.get('stock_pool')
                if isinstance(raw, list):
                    raw = str(raw)
                val = m['values'].index(raw) if raw in m['values'] else 0
            else:
                val = config.get(key, m.get('lo', 0) if m['type'] == 'numeric' else m['values'][0])

            if m['type'] == 'numeric':
                lo, hi = m['lo'], m['hi']
                if hi == lo:
                    vec[i] = 0.0
                else:
                    vec[i] = -5.0 + 10.0 * (float(val) - lo) / (hi - lo)
            elif m['type'] == 'weight':
                lo, hi = m['lo'], m['hi']
                if hi == lo:
                    vec[i] = 0.0
                else:
                    vec[i] = -5.0 + 10.0 * (float(val) - lo) / (hi - lo)
            elif m['type'] == 'categorical':
                n = len(m['values'])
                idx = val if isinstance(val, int) else m['values'].index(val) if val in m['values'] else 0
                if n == 1:
                    vec[i] = 0.0
                else:
                    vec[i] = -5.0 + 10.0 * idx / (n - 1)

        return vec


# ============================================================
# Hash 编码器（显式 dim 时使用）
# ============================================================

class _HashEncoder:
    """将 config 通过确定性哈希映射到固定维度连续向量 [-5, 5]^dim。

    相同 config_key 总是得到相同向量，相似 config 得到相近向量。
    """

    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim

    @property
    def dim_int(self) -> int:
        return self.dim

    def encode(self, config: dict) -> np.ndarray:
        weights = config.get('weights', {})
        vec = np.zeros(self.dim, dtype=np.float64)

        for i, (k, val) in enumerate([
            ('buy_n', config.get('buy_n', 10)),
            ('holding_period', config.get('holding_period', 1)),
            ('timing_base', config.get('timing_base', 0) or 0),
            ('timing_leverage', config.get('timing_leverage', 0) or 0),
            ('timing_direction', config.get('timing_direction', 0) or 0),
            ('timing_window', config.get('timing_window', 0) or 0),
        ]):
            idx = abs(hash(k)) % self.dim
            vec[idx] += float(abs(hash(str(val))) % 1000) / 100.0 - 5.0

        for wk, wv in weights.items():
            idx = abs(hash(wk)) % self.dim
            vec[idx] += float(wv) * 5.0

        vec = np.clip(vec, -5.0, 5.0)
        return vec


# ============================================================
# 主类
# ============================================================

class GALandscape:
    """合成适应度景观，接口对齐回测 evaluator。

    景观值越小越好 → negate 后线性映射到 sharpe ∈ [-1, 5]。
    """

    def __init__(self, kind: str = 'rastrigin', dim: int | None = None,
                 profile_name: str | None = None, seed: int = 42,
                 noise: float = 0.0):
        """
        kind:          景观类型 (rastrigin/ackley/griewank/sphere/nk/nk_chaotic/deceptive)
        dim:           维度（None 则从 profile 自动推断）
        profile_name:  GA profile（用于推断搜索空间维度）
        seed:          随机种子
        noise:         评估噪声标准差（模拟金融数据噪声），0=无噪声
        """
        self.kind = kind
        self.noise = noise
        self.rng = np.random.default_rng(seed)

        # 编码器：显式 dim 优先 → 用 hash 编码器（任意维度）
        #         仅 profile → 用 ConfigEncoder（真实搜索空间维度）
        self.dim = dim
        if dim is not None:
            self.encoder = _HashEncoder(dim, seed)
        elif profile_name is not None:
            self.encoder = _ConfigEncoder(profile_name)
            self.dim = self.encoder.dim
        else:
            self.dim = 10
            self.encoder = _HashEncoder(self.dim, seed)

        # 景观函数
        if kind not in _LANDSCAPE_BUILDERS:
            raise ValueError(f"未知景观类型: {kind}，可选: {list(_LANDSCAPE_BUILDERS)}")
        self._type_tag, self._fn = _LANDSCAPE_BUILDERS[kind](dim, seed)

        # 预计算统计（用于映射到 sharpe 量纲）
        self._sample_min = None
        self._sample_max = None
        self._calibrate(10000, seed + 1)

    def _calibrate(self, n_samples: int, seed: int):
        rng = np.random.default_rng(seed)
        samples = rng.uniform(-5, 5, (n_samples, self.dim))
        vals = np.array([float(self._fn(s)) for s in samples])
        self._sample_min = float(np.min(vals))
        self._sample_max = float(np.max(vals))
        self._sample_range = self._sample_max - self._sample_min or 1.0

    def _raw_fitness(self, x: np.ndarray) -> float:
        """原始景观值（越小越好）"""
        val = float(self._fn(x))
        if self.noise > 0:
            val += self.rng.normal(0, self.noise)
        return val

    def _to_sharpe(self, raw: float) -> float:
        """原始值 → sharpe 映射 (raw 越小 → sharpe 越高)"""
        normalized = (raw - self._sample_min) / self._sample_range  # 0=best, 1=worst
        return 5.0 - 6.0 * normalized  # sharpe ∈ [-1, 5]

    def evaluate(self, config: dict) -> dict:
        """接口对齐 _worker_evaluate 返回值"""
        if self.encoder is not None:
            x = self.encoder.encode(config)
        else:
            # 兜底：从 config 中提取连续值
            x = self._fallback_encode(config)

        if len(x) < self.dim:
            x = np.pad(x, (0, self.dim - len(x)), mode='constant')
        elif len(x) > self.dim:
            x = x[:self.dim]

        raw = self._raw_fitness(x)
        sharpe = self._to_sharpe(raw)

        return {
            'sharpe': sharpe,
            'annualized': sharpe * 0.5,  # 近似
            'max_drawdown': -0.15 + (1.0 - (raw - self._sample_min) / self._sample_range) * 0.05,
            'total_return': sharpe * 1.2,
            'raw_fitness': raw,  # 调试用
            'x': x.tolist(),     # 调试用
        }

    def _fallback_encode(self, config: dict) -> np.ndarray:
        weights = config.get('weights', {})
        vals = list(weights.values())
        if not vals:
            vals = [float(config.get('buy_n', 10))]
        return np.array(vals, dtype=np.float64)

    def global_optimum_sharpe(self) -> float:
        """理论全局最优的 sharpe（仅对已知解析解的景观有效）"""
        if self.kind in ('rastrigin', 'ackley', 'griewank', 'sphere'):
            return self._to_sharpe(0.0)  # 全局最优在 x=0
        return self._to_sharpe(self._sample_min)

    def landscape_stats(self) -> dict:
        """景观特征，用于评估搜索难度"""
        # 采样估计 ruggedness
        rng = np.random.default_rng(42)
        xs = rng.uniform(-5, 5, (2000, self.dim))
        vals = np.array([float(self._fn(x)) for x in xs])
        # 局部最优密度：随机扰动后 fitness 上升的次数占比
        n_local = 0
        n_trials = min(500, len(xs))
        for i in range(n_trials):
            x0 = xs[i]
            f0 = float(self._fn(x0))
            for _ in range(10):
                dx = rng.normal(0, 0.5, self.dim)
                f1 = float(self._fn(x0 + dx))
                if f1 > f0:  # 因为是 min 问题，f1 > f0 说明 f0 是局部最小
                    n_local += 1
                    break
        local_optima_density = n_local / n_trials

        return {
            'kind': self.kind,
            'dim': self.dim,
            'noise': self.noise,
            'sample_min': self._sample_min,
            'sample_max': self._sample_max,
            'global_optimum_sharpe': self.global_optimum_sharpe(),
            'local_optima_density_estimate': round(local_optima_density, 3),
            'valley_depth_ratio': round(1.0 - self._sample_min / (self._sample_max or 1.0), 3),
        }


# ============================================================
# 种群收敛指标（直接用于现有 GA 日志输出）
# ============================================================

def compute_convergence_metrics(configs: List[dict], encoder: _ConfigEncoder | None = None) -> dict:
    """计算种群多样性/收敛指标，用于逐代监控。

    不修改 GA 逻辑，仅作为观测器。
    """
    if not configs:
        return {}

    if encoder is not None:
        xs = np.array([encoder.encode(c) for c in configs])
    else:
        # 从 config 权重提取连续向量
        xs = []
        for c in configs:
            w = c.get('weights', {})
            if w:
                xs.append(list(w.values()))
            else:
                xs.append([float(c.get('buy_n', 10))])
        xs = np.array(xs, dtype=np.float64)

    n, d = xs.shape

    # 基因型多样性
    centroid = np.mean(xs, axis=0)
    genotypic_variance = float(np.mean(np.sum((xs - centroid) ** 2, axis=1)))

    # 逐维方差（归一化）
    dim_range = np.max(xs, axis=0) - np.min(xs, axis=0)
    dim_range[dim_range == 0] = 1.0
    dim_variance = float(np.mean(np.var(xs, axis=0) / dim_range))

    # 唯一个体比例
    unique_count = len({hashlib.md5(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()
                        for c in configs})
    uniqueness = unique_count / n

    # 种群半径（最远个体到中心距离）
    distances = np.sqrt(np.sum((xs - centroid) ** 2, axis=1))
    population_radius = float(np.max(distances))

    return {
        'genotypic_variance': round(genotypic_variance, 4),
        'dim_variance_norm': round(dim_variance, 4),
        'uniqueness': round(uniqueness, 3),
        'population_radius': round(population_radius, 3),
        'n_unique': unique_count,
        'n_total': n,
    }


# ============================================================
# 批量实验接口
# ============================================================

def run_convergence_experiment(
    landscape: GALandscape,
    ga_runner,  # 你的 ga_optimizer 或 _run_ga 的函数引用
    n_trials: int = 50,
    n_generations: int = 100,
) -> dict:
    """在合成景观上反复运行 GA，统计收敛指标。

    返回每代的平均 genotypic_variance / uniqueness 等。
    """
    all_metrics: list[list[dict]] = []  # [trial][gen]

    for trial in range(n_trials):
        # 这里需要接入你现有的 GA 循环，把 evaluator 替换成 landscape.evaluate
        # 返回每代的 convergence_metrics
        gen_metrics = ga_runner(landscape, n_generations, seed=trial)
        all_metrics.append(gen_metrics)

    # 按代数聚合
    agg = {}
    for gen_idx in range(n_generations):
        gen_vals = {}
        for trial_metrics in all_metrics:
            if gen_idx < len(trial_metrics):
                for k, v in trial_metrics[gen_idx].items():
                    gen_vals.setdefault(k, []).append(v)
        agg[f'gen_{gen_idx}'] = {
            k: {'mean': np.mean(v), 'std': np.std(v)}
            for k, v in gen_vals.items()
        }
    return agg
