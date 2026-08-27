"""Gymnasium adapter over the causal single-day WBR domain engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Iterable, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

from env.action_schema import ActionSchema
from env.contracts import AccountState, DayConfig
from env.encoder import (
    ObservationEncoder,
    StaticMarketEncodingCache,
    TrainOnlyNormalizer,
)
from env.fees import DEFAULT_FEE_SCHEDULE
from env.observation import ObservationBuilder
from env.planner import DayMarketData, DayPlanner
from env.quantity import quantity_schema_manifest
from env.simulator import DaySimulator, accounting_schema_manifest
from factor import FactorBatch
from offline_data import RuntimeSlice


DEFAULT_UNIVERSE_PREFIXES = ("60", "00", "30", "688")
ENVIRONMENT_SCHEMA_VERSION = "wbr-ppo-environment-v2"


def environment_schema_manifest(action_schema: ActionSchema) -> dict[str, object]:
    """Return the frozen semantics required to run a policy bundle safely."""

    fees = DEFAULT_FEE_SCHEDULE
    payload: dict[str, object] = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "action_schema_hash": action_schema.schema_hash,
        "universe_prefixes": list(DEFAULT_UNIVERSE_PREFIXES),
        "planner": {
            "schema_version": "wbr-day-planner-v2",
            "decision_time": "T-open",
            "quantity_rules": quantity_schema_manifest(),
            "score_direction": "descending_weighted_factor_rank_best_one",
            "order_plan_execution": "sell_then_buy",
        },
        "fees": {
            "commission_rate": fees.commission_rate,
            "minimum_commission": fees.minimum_commission,
            "stamp_tax_rate": fees.stamp_tax_rate,
            "transfer_fee_rate": fees.transfer_fee_rate,
            "slippage_rate": fees.slippage_rate,
        },
        "accounting": accounting_schema_manifest(),
        "transition": {
            "action_interval": "open[T]_to_open[T+1]",
            "reward": "log(next_nav/pretrade_nav)",
            "terminated_after_final_settlement": True,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    payload["schema_hash"] = hashlib.sha256(encoded).hexdigest()
    return payload


def stock_universe_mask(
    stock_codes: Iterable[str],
    universe_prefixes: Iterable[str] = DEFAULT_UNIVERSE_PREFIXES,
) -> NDArray[np.bool_]:
    prefixes = tuple(str(value) for value in universe_prefixes)
    if not prefixes or any(not value for value in prefixes):
        raise ValueError("universe_prefixes must contain non-empty prefixes")
    codes = tuple(str(code) for code in stock_codes)
    return np.fromiter(
        (code.startswith(prefixes) for code in codes),
        dtype=np.bool_,
        count=len(codes),
    )


def _listing_age_panel(
    runtime: RuntimeSlice,
    universe_prefixes: tuple[str, ...],
) -> NDArray[np.int32]:
    """Return trading-row ages without pretending preload row zero is an IPO."""

    opens = np.asarray(runtime.field("open"))
    valid = np.isfinite(opens) & (opens > 0.0)
    allowed = stock_universe_mask(runtime.stock_codes, universe_prefixes)
    valid &= allowed[None, :]
    has_open = valid.any(axis=0)
    first = np.where(has_open, valid.argmax(axis=0), -1).astype(np.int32)
    rows = np.arange(runtime.n_dates, dtype=np.int32)[:, None]
    ages = rows - first[None, :]
    ages[(first[None, :] < 0) | (ages < 0)] = -1
    # A stock already valid at the first preload row may have been listed years
    # earlier.  Treat it as established; the 126-row preload keeps this choice
    # outside the actual decision interval in either case.
    established_at_boundary = first == 0
    ages[:, established_at_boundary] += 5
    return np.ascontiguousarray(ages, dtype=np.int32)


@dataclass(frozen=True)
class PreparedEpisode:
    """Immutable caches shared by all resets of one sealed data split."""

    runtime: RuntimeSlice
    factors: FactorBatch
    observation_builder: ObservationBuilder
    encoder: ObservationEncoder
    market_cache: StaticMarketEncodingCache
    listing_age: NDArray[np.int32]
    decision_start: int
    decision_stop: int
    code_to_index: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.runtime.stock_codes != self.factors.stock_codes:
            raise ValueError("runtime and factor stock vocabularies differ")
        if not np.array_equal(self.runtime.trade_dates, self.factors.trade_dates):
            raise ValueError("runtime and factor calendars differ")
        if not (
            self.runtime.decision_start
            <= self.decision_start
            < self.decision_stop
            <= self.runtime.decision_stop
        ):
            raise ValueError("episode interval must stay inside the sealed decision split")
        if self.decision_stop - self.decision_start < 2:
            raise ValueError("an episode needs at least two observations for one transition")
        if self.listing_age.shape != (self.runtime.n_dates, self.runtime.n_stocks):
            raise ValueError("listing_age has an incompatible shape")
        expected_indices = tuple(range(self.decision_start, self.decision_stop))
        if self.market_cache.decision_indices != expected_indices:
            raise ValueError("market cache does not cover the exact episode interval")
        object.__setattr__(
            self,
            "code_to_index",
            MappingProxyType(
                {code: index for index, code in enumerate(self.runtime.stock_codes)}
            ),
        )

    @classmethod
    def build(
        cls,
        runtime: RuntimeSlice,
        factors: FactorBatch,
        *,
        decision_start: int | None = None,
        decision_stop: int | None = None,
        lookback: int = 64,
        universe_prefixes: Iterable[str] = DEFAULT_UNIVERSE_PREFIXES,
    ) -> "PreparedEpisode":
        start = runtime.decision_start if decision_start is None else int(decision_start)
        stop = runtime.decision_stop if decision_stop is None else int(decision_stop)
        builder = ObservationBuilder(runtime, factors, lookback=lookback)
        encoder = ObservationEncoder(builder.schema)
        cache = StaticMarketEncodingCache.precompute(
            builder,
            encoder,
            range(start, stop),
        )
        prefixes = tuple(str(value) for value in universe_prefixes)
        expected_universe = stock_universe_mask(runtime.stock_codes, prefixes)
        if not np.array_equal(factors.rank_universe_mask, expected_universe):
            raise ValueError(
                "factor ranks were not computed on the episode planning universe"
            )
        return cls(
            runtime=runtime,
            factors=factors,
            observation_builder=builder,
            encoder=encoder,
            market_cache=cache,
            listing_age=_listing_age_panel(runtime, prefixes),
            decision_start=start,
            decision_stop=stop,
        )

    @property
    def observation_count(self) -> int:
        return self.decision_stop - self.decision_start

    @property
    def transition_count(self) -> int:
        return self.observation_count - 1


class WBRGymEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Single-process PPO environment using the same planner and simulator."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode: PreparedEpisode,
        *,
        action_schema: ActionSchema | None = None,
        normalizer: TrainOnlyNormalizer | None = None,
        initial_cash: float = 1_000_000.0,
        diagnostics: str = "minimal",
    ) -> None:
        super().__init__()
        if not np.isfinite(initial_cash) or initial_cash <= 0.0:
            raise ValueError("initial_cash must be finite and positive")
        self.episode = episode
        self.action_schema = action_schema or ActionSchema(
            factor_names=episode.factors.factor_names,
            filter_names=episode.factors.filter_names,
        )
        if self.action_schema.factor_names != episode.factors.factor_names:
            raise ValueError("action and factor vocabularies differ")
        if self.action_schema.filter_names != episode.factors.filter_names:
            raise ValueError("action and filter vocabularies differ")
        if normalizer is not None and (
            normalizer.encoder_schema != episode.encoder.output_schema.identifier
        ):
            raise ValueError("normalizer and encoder schemas differ")
        self.normalizer = normalizer
        self.initial_cash = float(initial_cash)
        self._planner = DayPlanner(diagnostics=diagnostics)
        self._simulator = DaySimulator()
        self.action_space = self.action_schema.action_space
        bound = normalizer.clip if normalizer is not None else np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-bound,
            high=bound,
            shape=(episode.encoder.output_dimension,),
            dtype=np.float32,
        )
        self._index: int | None = None
        self._account: AccountState | None = None
        self._terminated = False

    @property
    def current_account(self) -> AccountState:
        if self._account is None:
            raise RuntimeError("environment must be reset first")
        return self._account

    @property
    def current_index(self) -> int:
        if self._index is None:
            raise RuntimeError("environment must be reset first")
        return self._index

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        super().reset(seed=seed)
        if options:
            unknown = set(options) - {"account"}
            if unknown:
                raise ValueError(f"unsupported reset options: {sorted(unknown)}")
        supplied = None if not options else options.get("account")
        if supplied is not None and not isinstance(supplied, AccountState):
            raise TypeError("reset option 'account' must be AccountState")
        self._account = supplied or AccountState(
            cash=self.initial_cash,
            nav=self.initial_cash,
            peak_nav=self.initial_cash,
        )
        self._index = self.episode.decision_start
        self._terminated = False
        observation = self._encoded_observation()
        return observation, {
            "decision_date": self._date_text(self._index),
            "nav": self._account.nav,
        }

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        if self._terminated:
            raise RuntimeError("step called after termination; call reset")
        index = self.current_index
        account = self.current_account
        if index >= self.episode.decision_stop - 1:
            raise RuntimeError("sealed split has no next-open row for another action")

        raw_action = np.asarray(action, dtype=np.float32)
        config = self.action_schema.decode(raw_action)
        market = self._market_day(index)
        plan = self._planner.plan(market, account, config)
        next_index = index + 1
        relevant_codes = set(account.positions)
        relevant_codes.update(code for code, _ in plan.sell_orders)
        relevant_codes.update(plan.buy_orders)
        step_result = self._simulator.step(
            account,
            plan,
            self._price_mapping(index, "open", relevant_codes),
            self._price_mapping(next_index, "open", relevant_codes),
            close_prices=self._price_mapping(index, "close", relevant_codes),
            next_preclose_prices=self._price_mapping(
                next_index, "preClose", relevant_codes
            ),
            next_decision_date=self._date_text(next_index),
            terminated=next_index == self.episode.decision_stop - 1,
        )
        self._account = step_result.account_state
        self._index = next_index
        self._terminated = step_result.terminated
        observation = self._encoded_observation()
        nav = float(step_result.account_state.nav)
        exposure = 1.0 - float(step_result.account_state.cash) / nav
        info: dict[str, object] = {
            "decision_date": market.decision_date,
            "next_decision_date": self._date_text(next_index),
            "raw_action": raw_action.copy(),
            "day_config": self.action_schema.to_static_config(config),
            "nav": nav,
            "portfolio_return": float(step_result.portfolio_return),
            "exposure": exposure,
            "fill_count": len(step_result.fills),
            "total_fees": float(step_result.diagnostics["total_fees"]),
        }
        return observation, float(step_result.reward), self._terminated, False, info

    def _date_text(self, index: int) -> str:
        return np.datetime_as_string(self.episode.runtime.trade_dates[index], unit="D")

    def _encoded_observation(self) -> NDArray[np.float32]:
        account_part = self.episode.observation_builder.build_account(
            self.current_index,
            self.current_account,
        )
        return self.episode.market_cache.encode_account_state(
            account_part,
            self.episode.encoder,
            normalizer=self.normalizer,
        )

    def _market_day(self, index: int) -> DayMarketData:
        factor_day = self.episode.factors.day(index)
        return DayMarketData(
            decision_date=self._date_text(index),
            stock_codes=self.episode.runtime.stock_codes,
            factor_ranks={
                name: factor_day.ranks[position]
                for position, name in enumerate(factor_day.factor_names)
            },
            factor_validity={
                name: factor_day.validity[position]
                for position, name in enumerate(factor_day.factor_names)
            },
            filter_masks={
                name: factor_day.filters[position]
                for position, name in enumerate(factor_day.filter_names)
            },
            open_prices=self.episode.runtime.field("open")[index],
            preclose_prices=self.episode.runtime.field("preClose")[index],
            issue_prices=self.episode.runtime.field("issue_price"),
            st_mask=self.episode.runtime.field("st_mask")[index],
            listing_age=self.episode.listing_age[index],
        )

    def _price_mapping(
        self,
        index: int,
        field_name: str,
        codes: Iterable[str],
    ) -> dict[str, float]:
        row = self.episode.runtime.field(field_name)[index]
        result: dict[str, float] = {}
        for code in codes:
            stock_index = self.episode.code_to_index.get(code)
            if stock_index is not None:
                result[code] = float(row[stock_index])
        return result

    def render(self) -> None:
        return None


def fixed_config_action(schema: ActionSchema, config: DayConfig) -> NDArray[np.float32]:
    """The exact action used for static-policy baselines and normalizer fitting."""

    return schema.encode(config)


__all__ = [
    "DEFAULT_UNIVERSE_PREFIXES",
    "ENVIRONMENT_SCHEMA_VERSION",
    "PreparedEpisode",
    "WBRGymEnv",
    "fixed_config_action",
    "environment_schema_manifest",
    "stock_universe_mask",
]
