"""Unpooled, causal per-stock observations available at T-open."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from functools import cached_property
import json
from typing import Mapping, Sequence
import numpy as np
from numpy.typing import NDArray
from env.action_schema import ActionSchema, CORE_FACTOR_NAMES
from env.contracts import AccountState, Observation, PolicyMemory
from env.planner import DayMarketData
from factor.library.bilibili import calculate_lifetime_state, LIFETIME_STATE_NAMES
from offline_data.financial_versions import RAW_FINANCIAL_VALUE_NAMES, RAW_FINANCIAL_TIME_NAMES, RAW_FINANCIAL_PERIOD_NAMES, RAW_FINANCIAL_STATE_VERSION

DEFAULT_LOOKBACK = 64
OBSERVATION_SCHEMA_VERSION = "wbr-observation-v21-source-state-no-factors"
RAW_MISSING_VALUE = np.float32(-np.finfo(np.float32).max)
CURRENT_RUNTIME_FIELDS = ("open", "preClose", "st_mask")
LAGGED_RUNTIME_FIELDS = ("open", "high", "low", "close", "volume", "amount", "preClose", "total_share")
STATIC_RUNTIME_FIELDS = ("issue_price", "issue_date")
ENVIRONMENT_RUNTIME_FIELDS = ("listing_age", "delisted_mask")
QUARANTINED_RUNTIME_FIELDS = ("bps", "eps", "roe", "profit_yoy", "revenue_yoy", "operating_cf_ps", "gross_margin")
PRICE_LEGALITY_FEATURE_NAMES = ("price_buy_allowed", "price_sell_allowed")
HISTORICAL_STOCK_FEATURE_NAMES = tuple(f"{name}_lag1" for name in LAGGED_RUNTIME_FIELDS if name != "total_share") + RAW_FINANCIAL_VALUE_NAMES + RAW_FINANCIAL_PERIOD_NAMES
LATEST_STOCK_FEATURE_NAMES = ("open", "preClose", "total_share_lag1", "issue_price", "issue_age_days", "st_mask", "listing_age", *PRICE_LEGALITY_FEATURE_NAMES, *LIFETIME_STATE_NAMES, *RAW_FINANCIAL_TIME_NAMES)
STOCK_FEATURE_NAMES = HISTORICAL_STOCK_FEATURE_NAMES + LATEST_STOCK_FEATURE_NAMES
POSITION_FEATURE_NAMES = ("quantity", "average_cost", "sellable_quantity", "last_mark_price")
PORTFOLIO_FEATURE_NAMES = ("cash", "nav", "peak_nav", "max_drawdown")

def _policy_history_feature_names(schema: ActionSchema) -> tuple[str, ...]:
    return ("valid", *(f"action.{name}" for name in schema.action_names), "gross_turnover_ratio", "total_cost_ratio")

POLICY_HISTORY_FEATURE_NAMES = _policy_history_feature_names(ActionSchema())

def _canonical_hash(payload: Mapping[str, object]) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _ordered_string_hash(values: Sequence[str]) -> str:
    return sha256(json.dumps(tuple(values), separators=(",", ":")).encode()).hexdigest()

@dataclass(frozen=True)
class ObservationSchema:
    lookback: int
    stock_count: int
    stock_codes_hash: str
    runtime_schema_hash: str
    factor_schema_hash: str
    action_schema_hash: str
    stock_feature_names: tuple[str, ...] = STOCK_FEATURE_NAMES
    position_feature_names: tuple[str, ...] = POSITION_FEATURE_NAMES
    portfolio_feature_names: tuple[str, ...] = PORTFOLIO_FEATURE_NAMES
    policy_history_feature_names: tuple[str, ...] = POLICY_HISTORY_FEATURE_NAMES
    factor_names: tuple[str, ...] = CORE_FACTOR_NAMES
    version: str = OBSERVATION_SCHEMA_VERSION
    financial_state_version: str = RAW_FINANCIAL_STATE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVATION_SCHEMA_VERSION or self.lookback <= 0 or self.stock_count <= 0:
            raise ValueError("unsupported raw schema or nonpositive lookback/stock count")
        if self.financial_state_version != RAW_FINANCIAL_STATE_VERSION:
            raise ValueError("unsupported financial source state")
        for name in ("stock_feature_names", "position_feature_names", "portfolio_feature_names", "policy_history_feature_names", "factor_names"):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
            object.__setattr__(self, name, values)
        for name in ("stock_codes_hash", "runtime_schema_hash", "factor_schema_hash", "action_schema_hash"):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must be a SHA256 digest")

    @property
    def stock_feature_count(self) -> int:
        return len(self.stock_feature_names)

    def _hash_payload(self) -> dict[str, object]:
        return {name: list(value) if isinstance(value, tuple) else value for name, value in vars(self).items()}

    @property
    def schema_hash(self) -> str:
        return _canonical_hash(self._hash_payload())

    @property
    def identifier(self) -> str:
        return f"{self.version}:{self.schema_hash}"

    def to_dict(self) -> dict[str, object]:
        return {**self._hash_payload(), "schema_hash": self.schema_hash}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ObservationSchema":
        values = dict(payload)
        expected = values.pop("schema_hash")
        result = cls(**values)
        if result.schema_hash != expected:
            raise ValueError("observation schema hash mismatch")
        return result

@dataclass(frozen=True)
class StaticObservation:
    decision_index: int
    decision_date: str
    stock_panel: NDArray[np.float32]
    time_mask: NDArray[np.bool_]
    pit_universe_mask: NDArray[np.bool_]
    schema_version: str

@dataclass(frozen=True)
class StaticObservationRows:
    row_start: int
    decision_dates: tuple[str, ...]
    stock_panel: NDArray[np.float32]
    pit_universe_mask: NDArray[np.bool_]
    schema_version: str

@dataclass(frozen=True)
class AccountObservation:
    decision_index: int
    decision_date: str
    position_panel: NDArray[np.float32]
    portfolio: NDArray[np.float32]
    policy_history: NDArray[np.float32]
    pit_universe_mask: NDArray[np.bool_]
    schema_version: str

class ObservationBuilder:
    """One raw assembly; injected DayMarketData shares planner legality caches."""
    def __init__(self, runtime, factors, *, day_markets: Sequence[DayMarketData],
                 lookback: int = DEFAULT_LOOKBACK, listing_age=None,
                 action_schema: ActionSchema | None = None) -> None:
        self._data = runtime.data
        self._stock_codes = tuple(runtime.stock_codes)
        self._trade_dates = np.asarray(runtime.trade_dates, dtype="datetime64[D]")
        self._day_markets = tuple(day_markets)
        self.lookback = int(lookback)
        self.action_schema = action_schema or ActionSchema(factor_names=factors.factor_names, filter_names=factors.filter_names)
        if self.lookback <= 0 or len(self._day_markets) != len(self._trade_dates):
            raise ValueError("raw builder requires positive lookback and every market row")
        if tuple(factors.stock_codes) != self._stock_codes or not np.array_equal(factors.trade_dates, self._trade_dates):
            raise ValueError("runtime and factor axes differ")
        if factors.runtime_schema_hash != runtime.manifest.schema_hash:
            raise ValueError("factors and runtime schema differ")
        if self.action_schema.factor_names != factors.factor_names or self.action_schema.filter_names != factors.filter_names:
            raise ValueError("action and factor vocabularies differ")
        required = set(CURRENT_RUNTIME_FIELDS + LAGGED_RUNTIME_FIELDS + STATIC_RUNTIME_FIELDS + ENVIRONMENT_RUNTIME_FIELDS + RAW_FINANCIAL_VALUE_NAMES + RAW_FINANCIAL_PERIOD_NAMES + RAW_FINANCIAL_TIME_NAMES)
        if required.difference(self._data):
            raise ValueError(f"raw observation fields missing: {sorted(required.difference(self._data))}")
        self._listing_age = np.asarray(self._data["listing_age"] if listing_age is None else listing_age, dtype=np.int32)
        self._pit_universe_mask = np.ascontiguousarray((self._listing_age >= 0) & ~np.asarray(self._data["delisted_mask"], dtype=bool))
        self._pit_universe_mask.flags.writeable = False
        self._stock_to_index = {code: i for i, code in enumerate(self._stock_codes)}
        self.schema = ObservationSchema(lookback=self.lookback, stock_count=len(self._stock_codes),
            stock_codes_hash=_ordered_string_hash(self._stock_codes), runtime_schema_hash=runtime.manifest.schema_hash,
            factor_schema_hash=factors.schema_hash, action_schema_hash=self.action_schema.schema_hash,
            stock_feature_names=STOCK_FEATURE_NAMES,
            policy_history_feature_names=_policy_history_feature_names(self.action_schema), factor_names=factors.factor_names)

    @property
    def stock_codes(self):
        return self._stock_codes

    @property
    def trade_dates(self):
        return self._trade_dates

    @property
    def listing_age(self):
        return self._listing_age

    @property
    def pit_universe_mask(self):
        return self._pit_universe_mask

    @property
    def day_markets(self):
        return self._day_markets

    @cached_property
    def lifetime_state(self):
        return calculate_lifetime_state(self._trade_dates, self._data)

    def build_static_rows(self, row_start: int, row_stop: int) -> StaticObservationRows:
        if not 0 <= row_start < row_stop <= len(self._trade_dates):
            raise IndexError("raw market rows outside sealed runtime")
        rows = np.arange(row_start, row_stop)
        panel = np.zeros((len(rows), self.schema.stock_count, self.schema.stock_feature_count), dtype=np.float32)
        member = self._pit_universe_mask[rows]
        names = {name: i for i, name in enumerate(self.schema.stock_feature_names)}
        def assign(name, values, available=None):
            numeric = np.asarray(values)
            valid = member & np.isfinite(numeric) & (numeric > RAW_MISSING_VALUE) & (numeric <= np.finfo(np.float32).max)
            if available is not None:
                valid &= available
            panel[:, :, names[name]] = np.where(valid, numeric, np.where(member, RAW_MISSING_VALUE, 0.0))
        for name in ("open", "preClose", "st_mask"):
            assign(name, self._data[name][rows])
        prior = np.maximum(rows - 1, 0)
        for name in LAGGED_RUNTIME_FIELDS:
            assign(f"{name}_lag1", self._data[name][prior],
                   (rows[:, None] > 0) & self._pit_universe_mask[prior])
        issue_dates = np.asarray(self._data["issue_date"], dtype="datetime64[D]")
        assign("issue_price", np.broadcast_to(self._data["issue_price"], member.shape),
               ~np.isnat(issue_dates)[None, :] & (self._trade_dates[rows, None] >= issue_dates[None, :]))
        assign("issue_age_days", (self._trade_dates[rows, None] - issue_dates[None, :]).astype('timedelta64[D]').astype(float),
               ~np.isnat(issue_dates)[None, :] & (self._trade_dates[rows, None] >= issue_dates[None, :]))
        assign("listing_age", self._listing_age[rows])
        for name in RAW_FINANCIAL_VALUE_NAMES + RAW_FINANCIAL_PERIOD_NAMES + RAW_FINANCIAL_TIME_NAMES:
            assign(name, self._data[name][rows])
        for name in LIFETIME_STATE_NAMES:
            assign(name, self.lifetime_state[name][rows])
        for offset, row in enumerate(rows):
            legality = self._day_markets[int(row)].trade_legality(self.action_schema.fixed_limit_up_protection)
            panel[offset, :, names["price_buy_allowed"]] = legality.buy_allowed & member[offset]
            panel[offset, :, names["price_sell_allowed"]] = legality.sell_allowed & member[offset]
        return StaticObservationRows(row_start, tuple(str(d) for d in self._trade_dates[rows]), panel,
                                     np.ascontiguousarray(member), self.schema.identifier)

    def build_static(self, decision_index: int) -> StaticObservation:
        start = max(0, decision_index - self.lookback + 1)
        rows = self.build_static_rows(start, decision_index + 1)
        count = len(rows.decision_dates)
        panel = np.zeros((self.lookback, self.schema.stock_count, self.schema.stock_feature_count), dtype=np.float32)
        member = np.zeros(panel.shape[:2], dtype=bool)
        mask = np.zeros(self.lookback, dtype=bool)
        panel[-count:] = rows.stock_panel
        member[-count:] = rows.pit_universe_mask
        mask[-count:] = True
        return StaticObservation(decision_index, str(self._trade_dates[decision_index]), panel, mask, member, self.schema.identifier)

    def build_account(self, decision_index: int, account: AccountState, *, policy_memory: PolicyMemory | None = None) -> AccountObservation:
        if not 0 <= decision_index < len(self._trade_dates):
            raise IndexError("account date outside sealed runtime")
        positions = np.zeros((self.schema.stock_count, 4), dtype=np.float32)
        for code, quantity in account.positions.items():
            if quantity <= 0:
                continue
            i = self._stock_to_index[code]
            positions[i] = (quantity, account.average_costs[code], account.sellable_positions.get(code, 0), account.last_prices[code])
        portfolio = np.asarray((account.cash, account.nav, account.peak_nav, account.max_drawdown), dtype=np.float32)
        memory = policy_memory or PolicyMemory()
        self.action_schema.validate_policy_memory(memory)
        history = np.zeros((self.lookback, len(self.schema.policy_history_feature_names)), dtype=np.float32)
        if memory.history is not None:
            recorded = memory.history
            if recorded.decision_dates[-1] >= self._trade_dates[decision_index]:
                raise ValueError("policy history cannot include current or future actions")
            first = max(0, decision_index-self.lookback)
            past = self._trade_dates[first:decision_index]
            offsets = np.searchsorted(past, recorded.decision_dates)
            source = np.flatnonzero(recorded.decision_dates >= past[0]) if len(past) else np.empty(0, dtype=int)
            if len(source):
                if np.any(offsets[source]>=len(past)) or not np.array_equal(past[offsets[source]], recorded.decision_dates[source]):
                    raise ValueError("policy history dates differ from the sealed calendar")
                rows = self.lookback-len(past)+offsets[source]
                history[rows, 0] = 1
                history[rows, 1:] = recorded.values[source]
        if not all(np.isfinite(x).all() for x in (positions, portfolio, history)):
            raise ValueError("raw account must be finite")
        return AccountObservation(decision_index, str(self._trade_dates[decision_index]), positions, portfolio,
                                  history, self._pit_universe_mask[decision_index], self.schema.identifier)

    def build(self, decision_index: int, account: AccountState, *, policy_memory: PolicyMemory | None = None,
              static: StaticObservation | None = None) -> Observation:
        market = self.build_static(decision_index) if static is None else static
        dynamic = self.build_account(decision_index, account, policy_memory=policy_memory)
        if market.decision_index != decision_index or market.schema_version != self.schema.identifier:
            raise ValueError("static raw observation identity mismatch")
        return Observation(stock_panel=market.stock_panel, position_panel=dynamic.position_panel,
            portfolio=dynamic.portfolio, policy_history=dynamic.policy_history, time_mask=market.time_mask,
            pit_universe_mask=market.pit_universe_mask, schema_version=self.schema.identifier, decision_date=dynamic.decision_date)
