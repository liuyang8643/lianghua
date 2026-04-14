import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, FIRST_COMPLETED, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from typing import Dict, List

from core.factors import SmallCap, WMACross, FactorCtx, FactorResult, BatchNormFactor
from utils.hash import hash_function_code
from utils.stock.format import format_qmt_date, format_qmt_datetime
from core.factors.helpers import BaseFactor, CacheKey, DiskCache
from ..logger import core_logger

DEFAULT_FACTOR_CLASSES = [SmallCap, WMACross]

_STOCK_BATCH_SIZE = 100
_TOPN_CACHE_SCHEMA_VERSION = 'topn-v2'
_FACTOR_SCORE_CACHE_SCHEMA_VERSION = 'factor-score-v2'
_CPU_COUNT = os.cpu_count() or 4
_STALL_TIMEOUT_SEC = 180


def _read_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == '':
        return default
    try:
        value = int(raw_value)
    except ValueError:
        core_logger.warning(f'{name}={raw_value!r} 不是正整数，回退到默认值 {default}')
        return default
    if value < 1:
        core_logger.warning(f'{name}={raw_value!r} 小于 1，回退到默认值 {default}')
        return default
    return value


_MAX_TOPN_WORKERS = _read_positive_int_env('WBR_TOPN_MAX_WORKERS', max(1, _CPU_COUNT - 2))
_MAX_FACTOR_THREADS = _read_positive_int_env('WBR_TOPN_FACTOR_THREADS', max(1, _CPU_COUNT - 2))


def _make_factor_cache_key(name: str, func_hash: str, base_date: datetime, stock_list: List[str], dividend_type: str) -> str:
    return CacheKey.make_key(
        [
            f"factor-{name}-{_FACTOR_SCORE_CACHE_SCHEMA_VERSION}-{func_hash}",
            format_qmt_datetime(base_date),
            f"dividend-{dividend_type}",
        ],
        stocks=stock_list,
    )


def _calc_topn_worker(args):
    """进程 worker：计算单个 TopN 实例（内部用线程并行）"""
    stock_list, base_date, weights, factor_classes, dividend_type = args
    topn = TopN(stock_list, base_date, weights=weights, factor_classes=factor_classes, dividend_type=dividend_type)
    return topn


class TopN:
    """多因子选股策略 - 选取综合得分排名前N的股票"""

    def __init__(
        self,
        stock_list: List[str],
        base_date: datetime,
        weights: Dict[str, float] = None,
        factor_classes: List[type] = None,
        dividend_type: str = 'back',
    ):
        """
        初始化TopN策略

        Args:
            stock_list: 股票池列表
            base_date: 基准日期（必须是datetime对象）
            weights: 因子权重字典，为0的因子跳过计算
            factor_classes: 因子类列表，默认使用 DEFAULT_FACTOR_CLASSES
        """
        self.stock_list = stock_list
        self.base_date = base_date
        self.weights = weights
        self.dividend_type = dividend_type

        self._factor_classes: List[type] = factor_classes or DEFAULT_FACTOR_CLASSES

        self.factors: List[BaseFactor] = []
        self._factor_names: List[str] = []
        for f_cls in self._factor_classes:
            f = f_cls()
            name = f.__class__.__name__
            if self.weights is None or self.weights.get(name, 0.0) != 0:
                self.factors.append(f)
                self._factor_names.append(name)

        self.factor_scores: Dict[str, Dict[str, FactorResult]] = {
            name: {} for name in self._factor_names
        }

        self.max_hist_days = max((f.hist_days for f in self.factors), default=0)

        self._calculate_all_factors()

    def _calculate_all_factors(self):
        """使用多线程并行计算所有因子的所有股票"""
        if not self.factors:
            return

        func_hashes = [hash_function_code(f.calc) for f in self.factors]

        cache_data: Dict[str, Dict[str, FactorResult]] = {}
        for name, fhash in zip(self._factor_names, func_hashes):
            cache_key = _make_factor_cache_key(name, fhash, self.base_date, self.stock_list, self.dividend_type)
            cached = DiskCache.load_pickle(cache_key)
            cache_data[name] = cached if cached else {}

        stocks_to_calc = [
            s for s in self.stock_list
            if all(s not in cache_data.get(name, {}) for name in self._factor_names)
        ]

        for name, cached in cache_data.items():
            for stock, val in cached.items():
                self.factor_scores[name][stock] = val

        if not stocks_to_calc:
            core_logger.debug(f"因子分数全部命中缓存，跳过计算")
            return

        n_threads = min(_MAX_FACTOR_THREADS, max(1, (os.cpu_count() or 4) // 2))
        # 动态计算批处理大小：确保每个线程至少有2个批次，但单批不超过100只股票
        batch_size = max(10, min(_STOCK_BATCH_SIZE, len(stocks_to_calc) // (n_threads * 2)))
        batches = [stocks_to_calc[i:i + batch_size]
                   for i in range(0, len(stocks_to_calc), batch_size)]

        def _calc_batch(batch_stocks: List[str]):
            """线程 worker：计算一批股票的所有因子"""
            results: Dict[str, FactorResult] = {}
            for stock_code in batch_stocks:
                for f in self.factors:
                    name = f.__class__.__name__
                    ctx = FactorCtx(stock_code, self.base_date, dividend_type=self.dividend_type)
                    results[(name, stock_code)] = f.calc(ctx)
            return results

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(_calc_batch, batch) for batch in batches]
            for future in as_completed(futures):
                batch_result = future.result()
                for (name, stock), result in batch_result.items():
                    self.factor_scores[name][stock] = result

        for name, fhash in zip(self._factor_names, func_hashes):
            cache_key = _make_factor_cache_key(name, fhash, self.base_date, self.stock_list, self.dividend_type)
            DiskCache.save_pickle(cache_key, self.factor_scores[name])

    def get_normalized_scores(
        self,
        temperatures: Dict[str, float],
        method: str = 'softmax'
    ) -> Dict[str, Dict[str, float]]:
        """对所有因子进行批量归一化"""
        normalized = {}
        for factor_name, raw_scores in self.factor_scores.items():
            temperature = temperatures.get(factor_name, 1.0)
            norm_scores = BatchNormFactor.batch_normalize(
                raw_scores=raw_scores,
                temperature=temperature,
                method=method
            )
            normalized[factor_name] = norm_scores
        return normalized

    def get_ordered_stocks(
        self,
        n: int,
        weights: Dict[str, float] = None,
        temperatures: Dict[str, float] = None,
        norm_method: str = 'softmax',
    ) -> List[str]:
        """返回按综合得分排序的前N只股票"""
        weights = self.weights if weights is None else weights
        temperatures = temperatures or {}

        normalized_scores = self.get_normalized_scores(temperatures, norm_method)

        final_scores: Dict[str, float] = {}
        for stock_code in self.stock_list:
            score = 0.0
            valid_factors = 0
            for factor_name in self.factor_scores.keys():
                if stock_code in normalized_scores.get(factor_name, {}):
                    weight = weights.get(factor_name, 0.0)
                    if weight == 0:
                        continue
                    score += normalized_scores[factor_name][stock_code] * weight
                    valid_factors += 1
            if valid_factors > 0:
                final_scores[stock_code] = score

        if not final_scores:
            core_logger.warning(f"没有股票有有效因子分数！日期: {format_qmt_date(self.base_date)}")
            return []

        sorted_stocks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_n = [stock for stock, _ in sorted_stocks[:n]]

        core_logger.info(
            f"{format_qmt_date(self.base_date)} TopN选股完成: "
            f"候选{len(final_scores)}只, 选出{len(top_n)}只"
        )
        if top_n:
            top_5 = sorted_stocks[:min(5, len(sorted_stocks))]
            core_logger.debug(
                "Top 5: " + ", ".join([f"{s}({sc:.6f})" for s, sc in top_5])
            )

        return top_n


def make_topn_range_cache_key(
    backtest_datetime_list: List[datetime],
    stock_list: List[str],
    weights: Dict[str, float] = None,
    factor_classes: List[type] = None,
    dividend_type: str = 'back',
) -> str:
    """生成 TopN 范围共享内存缓存键。"""
    if not backtest_datetime_list:
        return 'topn_empty'

    factor_classes = factor_classes or DEFAULT_FACTOR_CLASSES
    first_d = backtest_datetime_list[0].strftime('%Y%m%d')
    last_d = backtest_datetime_list[-1].strftime('%Y%m%d')
    factor_key = '_'.join(sorted(cls.__name__ for cls in factor_classes))
    weight_key = '_'.join(f'{k}={v}' for k, v in sorted((weights or {}).items()))
    stock_hash = hashlib.md5('\n'.join(stock_list).encode('utf-8')).hexdigest()[:12]
    return (
        f'topn_{_TOPN_CACHE_SCHEMA_VERSION}_{first_d}_{last_d}_{dividend_type}'
        f'_f{factor_key}_n{len(stock_list)}_{stock_hash}'
        + (f'_w{weight_key}' if weight_key else '')
    )


def _compute_topn_sequential(worker_args_list: List[tuple]) -> List["TopN"]:
    total_tasks = len(worker_args_list)
    started_at = time.time()
    result_topns = []
    for idx, args in enumerate(worker_args_list, start=1):
        result_topns.append(_calc_topn_worker(args))
        now = time.time()
        elapsed = now - started_at
        speed = (idx / elapsed) if elapsed > 0 else 0.0
        eta_sec = ((total_tasks - idx) / speed) if speed > 0 else 0.0
        core_logger.info(
            f"TopN 串行进度: {idx}/{total_tasks} "
            f"({idx / total_tasks * 100:.1f}%), 已耗时 {elapsed:.1f}s, 预计剩余 {eta_sec:.1f}s"
        )
    return result_topns


def _compute_topn_parallel(worker_args_list: List[tuple], n_workers: int) -> List["TopN"]:
    core_logger.info(
        f"启动 {n_workers} 个进程并行计算 TopN，未计算 {len(worker_args_list)} 天"
    )
    heartbeat_interval_sec = 15
    total_tasks = len(worker_args_list)
    started_at = time.time()
    last_heartbeat_at = started_at
    last_progress_at = started_at
    completed_count = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_calc_topn_worker, args) for args in worker_args_list]
        uncached_topns = []
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=heartbeat_interval_sec, return_when=FIRST_COMPLETED)
            now = time.time()
            if done:
                for future in done:
                    uncached_topns.append(future.result())
                    completed_count += 1
                last_progress_at = now

            elapsed = now - started_at
            stalled_for = now - last_progress_at
            if done or (now - last_heartbeat_at) >= heartbeat_interval_sec:
                progress_pct = (completed_count / total_tasks * 100) if total_tasks else 100.0
                speed = (completed_count / elapsed) if elapsed > 0 else 0.0
                eta_sec = ((total_tasks - completed_count) / speed) if speed > 0 else 0.0
                core_logger.info(
                    f"TopN 并行进度: {completed_count}/{total_tasks} "
                    f"({progress_pct:.1f}%), 已耗时 {elapsed:.1f}s, 预计剩余 {eta_sec:.1f}s"
                )
                last_heartbeat_at = now

            if pending and stalled_for >= _STALL_TIMEOUT_SEC:
                core_logger.error(
                    f"TopN 并行超过 {_STALL_TIMEOUT_SEC}s 无进度，触发熔断"
                )
                for proc in list(getattr(executor, '_processes', {}).values()):
                    if proc is not None and proc.is_alive():
                        proc.kill()
                executor.shutdown(wait=False, cancel_futures=True)
                raise TimeoutError(
                    f"TopN parallel stalled for {_STALL_TIMEOUT_SEC}s "
                    f"(completed={completed_count}/{total_tasks})"
                )
    return uncached_topns


def compute_topn_range(
    backtest_datetime_list: List[datetime],
    stock_list: List[str],
    weights: Dict[str, float] = None,
    factor_classes: List[type] = None,
    dividend_type: str = 'back',
) -> List["TopN"]:
    """计算指定日期范围的 TopN 实例列表。

    进程级并行：每天一个 TopN 实例，不同日期分配到不同进程。
    线程级并行：每个 TopN 内部，多线程并行计算因子分数。
    """
    if not backtest_datetime_list:
        return []

    factor_classes = factor_classes or DEFAULT_FACTOR_CLASSES
    first_d = backtest_datetime_list[0].strftime('%Y%m%d')
    last_d = backtest_datetime_list[-1].strftime('%Y%m%d')

    from utils.shared_memory import SharedMemoryCache

    testback_cache = SharedMemoryCache('testback_cache', compress_level=6)
    cache_key = make_topn_range_cache_key(
        backtest_datetime_list,
        stock_list,
        weights=weights,
        factor_classes=factor_classes,
        dividend_type=dividend_type,
    )

    cached = testback_cache.get(cache_key)
    if cached and len(cached) == len(backtest_datetime_list):
        core_logger.info(f"从缓存加载 TopN：{first_d}~{last_d}，{len(cached)} 天")
        return cached

    # 预计算因子元数据，避免在循环中重复实例化
    factor_metadata = []
    for f_cls in factor_classes:
        f = f_cls()
        name = f.__class__.__name__
        if weights is not None and weights.get(name, 0.0) == 0:
            continue
        fhash = hash_function_code(f.calc)
        factor_metadata.append((name, fhash))

    # 检查缓存状态
    topn_params = []
    for d in backtest_datetime_list:
        all_cached = True
        for name, fhash in factor_metadata:
            ck = _make_factor_cache_key(name, fhash, d, stock_list, dividend_type)
            cached_factors = DiskCache.load_pickle(ck) or {}
            if len(cached_factors) < len(stock_list):
                all_cached = False
                break
        topn_params.append((d, all_cached))

    uncached_dates = [p[0] for p in topn_params if not p[1]]
    cached_dates = [p[0] for p in topn_params if p[1]]

    core_logger.info(
        f"TopN 范围 {first_d}~{last_d}：{len(cached_dates)} 天命中缓存，"
        f"{len(uncached_dates)} 天需计算"
    )

    result_topns = []
    for d in cached_dates:
        topn = TopN(stock_list, d, weights=weights, factor_classes=factor_classes, dividend_type=dividend_type)
        result_topns.append(topn)

    if uncached_dates:
        worker_args_list = [
            (stock_list, d, weights, factor_classes, dividend_type)
            for d in uncached_dates
        ]
        n_workers = min(_MAX_TOPN_WORKERS, max(1, (os.cpu_count() or 4) - 2), len(worker_args_list))
        uncached_topns = None
        attempt_workers = n_workers
        while uncached_topns is None:
            try:
                uncached_topns = _compute_topn_parallel(worker_args_list, attempt_workers)
            except BrokenProcessPool:
                if attempt_workers <= 2:
                    core_logger.warning(
                        "TopN 进程池异常退出，降级为当前进程串行计算剩余日期"
                    )
                    uncached_topns = _compute_topn_sequential(worker_args_list)
                    break
                next_workers = max(2, attempt_workers // 2)
                core_logger.warning(
                    f"TopN 进程池异常退出，降低并行度后重试: {attempt_workers} -> {next_workers}"
                )
                attempt_workers = next_workers
        result_topns.extend(uncached_topns)

    result_topns.sort(key=lambda x: x.base_date)

    if not testback_cache.put(cache_key, result_topns):
        core_logger.warning(f"TopN 范围缓存写入失败：{cache_key}")
    else:
        core_logger.info(f"TopN 范围计算完成并缓存：{len(result_topns)} 天")

    return result_topns
