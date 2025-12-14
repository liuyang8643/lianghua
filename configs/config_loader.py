"""
策略配置加载器

负责加载、解析和验证策略配置文件（YAML格式）
支持配置继承机制（base.yaml -> strategy_vX.yaml）
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FactorConfig:
    """因子配置"""
    name: str
    enabled: bool = True
    weight: float = 1.0
    temperature: float = 1.0
    search_range: Dict[str, List[float]] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "FactorConfig":
        """从字典创建因子配置"""
        return cls(
            name=name,
            enabled=data.get("enabled", True),
            weight=data.get("weight", 1.0),
            temperature=data.get("temperature", 1.0),
            search_range=data.get("search_range", {}),
            params=data.get("params", {})
        )


@dataclass
class SelectionConfig:
    """选股策略配置"""
    strategy: str = "TopN"
    rank_n: int = 30
    stock_pool: str = "allow_buy"


@dataclass
class PositionConfig:
    """仓位管理配置"""
    hand_size: int = 100
    min_buy_amount: int = 10000
    max_buy_amount: int = 25000
    preserve_amount: int = 0
    allocation_method: str = "equal"


@dataclass
class CapitalConfig:
    """资金配置"""
    initial_capital: int = 1000000
    max_position_ratio: float = 0.95


@dataclass
class TradingConfig:
    """交易时间配置"""
    buy_time_start: str = "14:40:00"
    buy_time_end: str = "14:55:00"
    sell_time_start: str = "09:35:00"
    sell_time_end: str = "09:45:00"


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str = "20240101"
    end_date: str = "20241231"
    commission_rate: float = 0.0003
    slippage_rate: float = 0.001
    min_commission: float = 5.0


@dataclass
class GeneticAlgorithmConfig:
    """遗传算法配置"""
    enabled: bool = False
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_ratio: float = 0.1
    objective_metric: str = "sharpe_ratio"
    objective_direction: str = "maximize"


@dataclass
class RiskControlConfig:
    """风控配置"""
    max_drawdown_ratio: float = 0.2
    stop_loss_ratio: float = 0.1
    stop_profit_ratio: float = 0.5


@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True
    cache_dir: str = "testback/.cache/.cache"
    ttl_days: int = 30


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"


@dataclass
class StrategyMeta:
    """策略元信息"""
    name: str = "Unknown Strategy"
    version: str = "unknown"
    description: str = ""
    created_at: str = ""
    author: str = ""
    base_config: Optional[str] = None
    changelog: List[str] = field(default_factory=list)


class StrategyConfig:
    """策略配置管理器"""

    def __init__(self, config_dict: Dict[str, Any]):
        """初始化配置"""
        self.raw_config = config_dict

        # 解析各个部分的配置
        self.meta = self._parse_meta(config_dict.get("meta", {}))
        self.factors = self._parse_factors(config_dict.get("factors", {}))
        self.selection = self._parse_selection(config_dict.get("selection", {}))
        self.position = self._parse_position(config_dict.get("position", {}))
        self.capital = self._parse_capital(config_dict.get("capital", {}))
        self.trading = self._parse_trading(config_dict.get("trading", {}))
        self.backtest = self._parse_backtest(config_dict.get("backtest", {}))
        self.genetic_algorithm = self._parse_ga(config_dict.get("genetic_algorithm", {}))
        self.risk_control = self._parse_risk_control(config_dict.get("risk_control", {}))
        self.cache = self._parse_cache(config_dict.get("cache", {}))
        self.logging = self._parse_logging(config_dict.get("logging", {}))

    @staticmethod
    def _parse_meta(data: Dict[str, Any]) -> StrategyMeta:
        """解析策略元信息"""
        return StrategyMeta(
            name=data.get("name", "Unknown Strategy"),
            version=data.get("version", "unknown"),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            author=data.get("author", ""),
            base_config=data.get("base_config"),
            changelog=data.get("changelog", [])
        )

    @staticmethod
    def _parse_factors(data: Dict[str, Any]) -> Dict[str, FactorConfig]:
        """解析因子配置"""
        factors = {}
        for name, config in data.items():
            factors[name] = FactorConfig.from_dict(name, config)
        return factors

    @staticmethod
    def _parse_selection(data: Dict[str, Any]) -> SelectionConfig:
        """解析选股配置"""
        return SelectionConfig(
            strategy=data.get("strategy", "TopN"),
            rank_n=data.get("rank_n", 30),
            stock_pool=data.get("stock_pool", "allow_buy")
        )

    @staticmethod
    def _parse_position(data: Dict[str, Any]) -> PositionConfig:
        """解析仓位管理配置"""
        return PositionConfig(
            hand_size=data.get("hand_size", 100),
            min_buy_amount=data.get("min_buy_amount", 10000),
            max_buy_amount=data.get("max_buy_amount", 25000),
            preserve_amount=data.get("preserve_amount", 0),
            allocation_method=data.get("allocation_method", "equal")
        )

    @staticmethod
    def _parse_capital(data: Dict[str, Any]) -> CapitalConfig:
        """解析资金配置"""
        return CapitalConfig(
            initial_capital=data.get("initial_capital", 1000000),
            max_position_ratio=data.get("max_position_ratio", 0.95)
        )

    @staticmethod
    def _parse_trading(data: Dict[str, Any]) -> TradingConfig:
        """解析交易时间配置"""
        return TradingConfig(
            buy_time_start=data.get("buy_time_start", "14:40:00"),
            buy_time_end=data.get("buy_time_end", "14:55:00"),
            sell_time_start=data.get("sell_time_start", "09:35:00"),
            sell_time_end=data.get("sell_time_end", "09:45:00")
        )

    @staticmethod
    def _parse_backtest(data: Dict[str, Any]) -> BacktestConfig:
        """解析回测配置"""
        return BacktestConfig(
            start_date=data.get("start_date", "20240101"),
            end_date=data.get("end_date", "20241231"),
            commission_rate=data.get("commission_rate", 0.0003),
            slippage_rate=data.get("slippage_rate", 0.001),
            min_commission=data.get("min_commission", 5.0)
        )

    @staticmethod
    def _parse_ga(data: Dict[str, Any]) -> GeneticAlgorithmConfig:
        """解析遗传算法配置"""
        objective = data.get("objective", {})
        return GeneticAlgorithmConfig(
            enabled=data.get("enabled", False),
            population_size=data.get("population_size", 50),
            generations=data.get("generations", 100),
            mutation_rate=data.get("mutation_rate", 0.1),
            crossover_rate=data.get("crossover_rate", 0.8),
            elite_ratio=data.get("elite_ratio", 0.1),
            objective_metric=objective.get("metric", "sharpe_ratio"),
            objective_direction=objective.get("direction", "maximize")
        )

    @staticmethod
    def _parse_risk_control(data: Dict[str, Any]) -> RiskControlConfig:
        """解析风控配置"""
        return RiskControlConfig(
            max_drawdown_ratio=data.get("max_drawdown_ratio", 0.2),
            stop_loss_ratio=data.get("stop_loss_ratio", 0.1),
            stop_profit_ratio=data.get("stop_profit_ratio", 0.5)
        )

    @staticmethod
    def _parse_cache(data: Dict[str, Any]) -> CacheConfig:
        """解析缓存配置"""
        return CacheConfig(
            enabled=data.get("enabled", True),
            cache_dir=data.get("cache_dir", "testback/.cache/.cache"),
            ttl_days=data.get("ttl_days", 30)
        )

    @staticmethod
    def _parse_logging(data: Dict[str, Any]) -> LoggingConfig:
        """解析日志配置"""
        return LoggingConfig(
            level=data.get("level", "INFO"),
            log_to_file=data.get("log_to_file", True),
            log_dir=data.get("log_dir", "logs")
        )

    def get_enabled_factors(self) -> List[FactorConfig]:
        """获取启用的因子列表"""
        return [f for f in self.factors.values() if f.enabled]

    def get_factor_weights(self) -> Dict[str, float]:
        """获取因子权重字典"""
        return {name: factor.weight for name, factor in self.factors.items() if factor.enabled}

    def get_factor_temperatures(self) -> Dict[str, float]:
        """获取因子温度参数字典"""
        return {name: factor.temperature for name, factor in self.factors.items() if factor.enabled}

    def validate(self) -> List[str]:
        """验证配置合法性，返回错误列表"""
        errors = []

        # 验证因子配置
        for name, factor in self.factors.items():
            if factor.enabled:
                if factor.weight < 0:
                    errors.append(f"因子 {name} 的权重不能为负数: {factor.weight}")
                if factor.temperature <= 0:
                    errors.append(f"因子 {name} 的温度参数必须为正数: {factor.temperature}")

        # 验证选股配置
        if self.selection.rank_n <= 0:
            errors.append(f"rank_n 必须为正数: {self.selection.rank_n}")

        # 验证仓位配置
        if self.position.min_buy_amount > self.position.max_buy_amount:
            errors.append(f"min_buy_amount ({self.position.min_buy_amount}) 不能大于 max_buy_amount ({self.position.max_buy_amount})")

        # 验证资金配置
        if self.capital.initial_capital <= 0:
            errors.append(f"initial_capital 必须为正数: {self.capital.initial_capital}")
        if not 0 < self.capital.max_position_ratio <= 1:
            errors.append(f"max_position_ratio 必须在 (0, 1] 范围内: {self.capital.max_position_ratio}")

        # 验证回测配置
        if self.backtest.commission_rate < 0:
            errors.append(f"commission_rate 不能为负数: {self.backtest.commission_rate}")
        if self.backtest.slippage_rate < 0:
            errors.append(f"slippage_rate 不能为负数: {self.backtest.slippage_rate}")

        return errors

    @classmethod
    def load(cls, config_file: str, base_dir: Optional[str] = None) -> "StrategyConfig":
        """
        加载策略配置文件

        Args:
            config_file: 配置文件名（如 "strategy_v1.yaml"）或完整路径
            base_dir: 配置文件所在目录，默认为 configs/strategies/

        Returns:
            StrategyConfig 实例
        """
        if base_dir is None:
            base_dir = Path(__file__).parent / "strategies"
        else:
            base_dir = Path(base_dir)

        config_path = base_dir / config_file if not Path(config_file).is_absolute() else Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 加载配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # 检查是否需要继承base配置
        meta = config_dict.get("meta", {})
        base_config_name = meta.get("base_config")

        if base_config_name:
            # 加载base配置
            base_path = base_dir / base_config_name
            if not base_path.exists():
                raise FileNotFoundError(f"基础配置文件不存在: {base_path}")

            with open(base_path, "r", encoding="utf-8") as f:
                base_dict = yaml.safe_load(f)

            # 深度合并配置（策略配置覆盖base配置）
            merged_dict = cls._deep_merge(base_dict, config_dict)
        else:
            merged_dict = config_dict

        # 创建配置实例
        config = cls(merged_dict)

        # 验证配置
        errors = config.validate()
        if errors:
            raise ValueError(f"配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors))

        return config

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典（override覆盖base）

        Args:
            base: 基础字典
            override: 覆盖字典

        Returns:
            合并后的字典
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并字典
                result[key] = StrategyConfig._deep_merge(result[key], value)
            else:
                # 直接覆盖
                result[key] = value

        return result

    def __repr__(self) -> str:
        """字符串表示"""
        enabled_factors = [f.name for f in self.get_enabled_factors()]
        return (
            f"StrategyConfig(\n"
            f"  name={self.meta.name}\n"
            f"  version={self.meta.version}\n"
            f"  enabled_factors={enabled_factors}\n"
            f"  rank_n={self.selection.rank_n}\n"
            f"  initial_capital={self.capital.initial_capital}\n"
            f")"
        )
