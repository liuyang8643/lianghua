"""Canonical causal episode session and deterministic policy rollouts.

This module deliberately has no Gymnasium dependency. ``EpisodeSession`` is
the single owner of the serial account timeline used by training, offline
backtests and policy replay. Adapters may decode their own action format, but
all observations, plans, fills, settlement, account memory and rewards pass
through this session.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from env.action_schema import ActionSchema
from env.contracts import (
    AccountState,
    DayConfig,
    Fill,
    Observation,
    OrderPlan,
    Policy,
    PolicyMemory,
    StepResult,
)
from env.encoder import (
    ObservationEncoder,
    RawMarketStore,
    TrainOnlyNormalizer,
)
from env.fees import DEFAULT_FEE_SCHEDULE, FeeSchedule
from env.metrics import (
    ANNUALIZATION_DAYS,
    MAX_DRAWDOWN_PENALTY_WEIGHT,
    EpisodeRewardState,
    PerformanceMetrics,
    REWARD_SCHEMA_VERSION,
    performance_from_log_rewards,
)
from env.observation import DEFAULT_LOOKBACK, ObservationBuilder, StaticObservation
from env.planner import DayMarketData, DayPlanner
from env.prefilter import (
    PrefilterUniverse,
    candidate_mask_from_previous_indices,
    candidate_mask_from_previous_ranking,
    rank_complete_universe,
)
from env.quantity import quantity_schema_manifest
from env.simulator import (
    DaySimulator,
    accounting_schema_manifest,
)
from factor import FACTOR_SCHEMA_VERSION, PRODUCTION_FACTORS, FactorBatch, precompute_factors, factor_coverage
from offline_data import RUNTIME_FIELDS, RuntimeSlice, load_runtime_slice
from offline_data.contracts import ReplayProjection


ENVIRONMENT_SCHEMA_VERSION = "wbr-ppo-environment-v37-turnover-floor"
CRITIC_CONTEXT_SCHEMA_VERSION = (
    "dense-return-incremental-drawdown-critic-context-v5"
)
CRITIC_CONTEXT_LOG_RETURN_CLIP = 4.0
CRITIC_CONTEXT_SCALAR_FEATURE_NAMES = (
    "episode_progress",
    "episode_remaining",
    "episode_horizon_saturation",
    "candidate_cumulative_net_log_return",
    "candidate_simple_return_mean",
    "candidate_simple_return_rms",
    "candidate_running_max_drawdown",
    "max_drawdown_penalty_weight",
)

PRODUCTION_FACTOR_HISTORY_ROWS = max(
    definition.metadata.hist_days for definition in PRODUCTION_FACTORS
)
RUNTIME_DECISION_LAG_ROWS = max(field.decision_lag for field in RUNTIME_FIELDS)
# The unique no-actor execution contract: decision legality, next-open
# settlement, and PIT membership. Factor source panels are consumed earlier.
EXECUTION_RUNTIME_FIELDS = frozenset({
    "open", "preClose", "close", "issue_date", "issue_price",
    "st_mask", "delisted_mask", "listing_age",
})


def required_runtime_preload_rows(lookback: int) -> int:
    """Rows required before the first decision for slice-invariant input.

    The actor window contains ``lookback - 1`` prior rows.  Its earliest row
    additionally consumes factor values with their own production history,
    plus the maximum registered
    decision lag.  These dependencies are cumulative, not alternatives.
    """

    if type(lookback) is not int or lookback <= 0:
        raise ValueError("lookback must be a positive int")
    return (
        lookback
        - 1
        + PRODUCTION_FACTOR_HISTORY_ROWS
        + RUNTIME_DECISION_LAG_ROWS
    )


def prepare_episode_from_runtime(
    runtime_path: str | Path,
    start: object,
    end: object,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    prefilter_n: int,
    encode_observations: bool = True,
    action_schema: ActionSchema | None = None,
) -> "PreparedEpisode":
    """Load, factorize and seal the sole canonical offline episode path."""

    runtime = load_runtime_slice(
        runtime_path,
        start,
        end,
        preload_rows=required_runtime_preload_rows(lookback),
    )
    factors = precompute_factors(runtime)
    episode = PreparedEpisode.build(
        runtime,
        factors,
        lookback=lookback,
        prefilter_n=prefilter_n,
        encode_observations=encode_observations,
        action_schema=action_schema,
    )
    coverage = episode.factor_coverage
    for name, item in coverage["factors"].items():
        print(json.dumps({"event": "factor_coverage", "start": coverage["start"], "end": coverage["end"], "factor": name, **item}, ensure_ascii=False), flush=True)
    return episode


def _validate_runtime_preload(runtime: RuntimeSlice, lookback: int) -> None:
    required = required_runtime_preload_rows(lookback)
    requested = int(runtime.manifest.requested_preload_rows)
    if requested < required:
        raise ValueError(
            "runtime slice preload is insufficient for invariant observations: "
            f"requested={requested}, required={required} "
            "((lookback - 1) + factor history + lag)"
        )


def critic_context_feature_names(
    encoder: ObservationEncoder,
) -> tuple[str, ...]:
    del encoder
    return CRITIC_CONTEXT_SCALAR_FEATURE_NAMES


def critic_context_dimension(encoder: ObservationEncoder) -> int:
    return len(critic_context_feature_names(encoder))


def neutral_critic_context(dimension: int) -> NDArray[np.float32]:
    """Return deployment padding that the asymmetric actor never consumes."""

    if type(dimension) is not int or dimension <= 0:
        raise ValueError("critic context dimension must be a positive int")
    return np.zeros(dimension, dtype=np.float32)


def environment_schema_manifest(
    action_schema: ActionSchema,
    encoder: ObservationEncoder,
    *,
    prefilter_n: int,
) -> dict[str, object]:
    """Return the frozen domain semantics required to run a policy bundle."""

    if type(prefilter_n) is not int or prefilter_n <= 0:
        raise ValueError("prefilter_n must be a positive int")
    fees = DEFAULT_FEE_SCHEDULE
    payload: dict[str, object] = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "action_schema_hash": action_schema.schema_hash,
        "prefilter": {
            "n": prefilter_n,
            "source": "previous_day_complete_factor_ranking_plus_holdings",
            "policy_action": False,
        },
        "stock_axis": "all_runtime_stocks_in_manifest_order",
        "actor_observation_encoding": {
            "semantics": "raw_full_stock_full_history_end_to_end_learned",
            "legacy_checkpoint_compatible": False,
            "market_normalizer_fit": "train_split_PIT_members_nonmissing_field_RMS_shared_all_stocks_dates",
            "normalizer_transform": "positive_scale_only_no_centering_no_clip_missing_minus_one",
            "account_normalizer": "initial_cash_and_train_price_scale_contract",
            "transport_reference": "explicit_readonly_store_lookup_removed_before_any_learnable_layer",
            "runtime_preload": {
                "formula": (
                    "(lookback - 1) + production_factor_history + decision_lag"
                ),
                "lookback": encoder.observation_schema.lookback,
                "production_factor_history": PRODUCTION_FACTOR_HISTORY_ROWS,
                "decision_lag": RUNTIME_DECISION_LAG_ROWS,
                "required_rows": required_runtime_preload_rows(
                    encoder.observation_schema.lookback
                ),
            },
        },
        "planner": {
            "schema_version": "wbr-day-planner-v6-tie-aware-financial-soft-scores",
            "decision_time": "T-open",
            "quantity_rules": quantity_schema_manifest(),
            "score_direction": "descending_weighted_factor_rank_best_one",
            "factor_score_semantics": "continuous_average_tied_rank_or_binary_zero_one_soft_bonus",
            "missing_factor_semantics": "available_absolute_weight_centered_rank_normalization_no_signal_stable_tail",
            "order_plan_execution": "sell_then_buy",
            "order_quantities": "explicit_positive_int_no_sell_all_sentinel",
            "rebalance": "daily_equalize_then_cash_sweep",
            "exposure_control": "none",
            "residual_cash": (
                "allowed_only_when_no_legal_target_increment_is_affordable_or_"
                "selected_target_capacity_is_exhausted"
            ),
            "diagnostics": [
                "post_fill_cash",
                "post_fill_exposure",
                "cheapest_next_legal_buy_cost",
                "residual_cash_reason",
                "full_investment_contract_satisfied",
            ],
        },
        "fees": {
            "commission_rate": fees.commission_rate,
            "minimum_commission": fees.minimum_commission,
            "stamp_tax_rate": fees.stamp_tax_rate,
            "transfer_fee_rate": fees.transfer_fee_rate,
            "slippage_rate": fees.slippage_rate,
        },
        "accounting": accounting_schema_manifest(),
        "critic_training_context": {
            "schema_version": CRITIC_CONTEXT_SCHEMA_VERSION,
            "feature_names": list(critic_context_feature_names(encoder)),
            "dimension": critic_context_dimension(encoder),
            "actor_access": False,
            "deployment_padding": "zeros_ignored_by_actor",
            "purpose": "candidate_path_dependent_return_drawdown_value_state",
            "candidate_account_state": "already_present_in_public_observation",
        },
        "actor_decision_memory": {
            "source": "previous_actual_fill_settlement",
            "config_encoding": "canonical_action_schema_encode",
            "features": list(encoder.observation_schema.policy_history_feature_names),
            "history_length": encoder.observation_schema.lookback,
            "time_alignment": "trading_dates[T-L:T]; oldest_to_newest; excludes_T",
            "cold_start": "unrecorded_rows_zero_padded_with_valid_zero",
            "live_recovery": "complete_dated_history_required_in_versioned_journal",
            "encoding": "retain_every_record_without_temporal_aggregation",
            "reward_effect": "observation_only_no_extra_cost_deduction",
        },
        "transition": {
            "action_interval": "open[T]_to_open[T+1]",
            "reward": "horizon_scaled_annualized_return_increment_minus_new_max_drawdown_increment",
            "reward_schema_version": REWARD_SCHEMA_VERSION,
            "objective": "annualized_return_minus_max_drawdown",
            "daily_component": "increment_in_episode_annualized_net_return",
            "drawdown_component": "-weight*increase_in_running_max_drawdown",
            "episode_sum_identity": "H/252*(annualized_net_return_minus_episode_max_drawdown)",
            "horizon_scale": "episode_horizon_transitions/annualization_days",
            "calmar_alignment": "positive_fixed_horizon_scaling_of_ratio_one_surrogate_changes_cross_horizon_weighting",
            "max_drawdown_penalty_weight": MAX_DRAWDOWN_PENALTY_WEIGHT,
            "annualization_days": ANNUALIZATION_DAYS,
            "fees_and_slippage": "included_once_via_net_nav",
            "full_investment": "planner_contract_fail_closed",
            "terminated_after_final_settlement": True,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    payload["schema_hash"] = hashlib.sha256(encoded).hexdigest()
    return payload


def build_day_market(
    runtime: RuntimeSlice,
    factors: FactorBatch,
    listing_age: NDArray[np.int32],
    index: int,
) -> DayMarketData:
    """Build the canonical T-open planner input for one complete stock axis.

    This is intentionally public: an offline :class:`EpisodeSession` and the
    live broker adapter must use the exact same factor/filter/legality inputs.
    It performs no execution and has no Gym or broker dependency.
    """

    if runtime.stock_codes != factors.stock_codes:
        raise ValueError("runtime and factor stock vocabularies differ")
    if not np.array_equal(runtime.trade_dates, factors.trade_dates):
        raise ValueError("runtime and factor calendars differ")
    if listing_age.shape != (runtime.n_dates, runtime.n_stocks):
        raise ValueError("listing_age has an incompatible shape")
    decision_index = int(index)
    if not 0 <= decision_index < runtime.n_dates:
        raise IndexError("decision index is outside the sealed runtime")
    factor_day = factors.day(decision_index)
    return DayMarketData(
        decision_date=np.datetime_as_string(
            runtime.trade_dates[decision_index], unit="D"
        ),
        stock_codes=runtime.stock_codes,
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
        open_prices=runtime.field("open")[decision_index],
        preclose_prices=runtime.field("preClose")[decision_index],
        issue_prices=np.where(
            runtime.field("issue_date") == runtime.trade_dates[decision_index],
            runtime.field("issue_price"),
            np.nan,
        ),
        st_mask=runtime.field("st_mask")[decision_index],
        delisted_mask=runtime.field("delisted_mask")[decision_index],
        listing_age=listing_age[decision_index],
    )


@dataclass(frozen=True)
class PreparedDecision:
    """One sealed decision snapshot shared by live and offline callers.

    Unlike :class:`PreparedEpisode`, this object needs no T+1 settlement row.
    It therefore exposes only Observation construction and canonical planning;
    execution and later settlement remain separate ports.
    """

    runtime: RuntimeSlice
    factors: FactorBatch
    observation_builder: ObservationBuilder
    listing_age: NDArray[np.int32]
    action_schema: ActionSchema
    decision_index: int
    prefilter_n: int

    def __post_init__(self) -> None:
        if self.runtime.stock_codes != self.factors.stock_codes:
            raise ValueError("runtime and factor stock vocabularies differ")
        if not np.array_equal(self.runtime.trade_dates, self.factors.trade_dates):
            raise ValueError("runtime and factor calendars differ")
        if not (
            self.runtime.decision_start
            <= self.decision_index
            < self.runtime.decision_stop
        ):
            raise ValueError("prepared decision must be inside the sealed interval")
        if self.listing_age.shape != (
            self.runtime.n_dates,
            self.runtime.n_stocks,
        ):
            raise ValueError("listing_age has an incompatible shape")
        if self.action_schema.factor_names != self.factors.factor_names:
            raise ValueError("action and factor vocabularies differ")
        if self.action_schema.filter_names != self.factors.filter_names:
            raise ValueError("action and filter vocabularies differ")
        if (
            self.observation_builder.schema.action_schema_hash
            != self.action_schema.schema_hash
        ):
            raise ValueError("action and observation policy-memory schemas differ")
        if type(self.prefilter_n) is not int or self.prefilter_n <= 0:
            raise ValueError("prefilter_n must be a positive int")

    @classmethod
    def build(
        cls,
        runtime: RuntimeSlice,
        factors: FactorBatch,
        *,
        action_schema: ActionSchema | None = None,
        decision_index: int | None = None,
        lookback: int = DEFAULT_LOOKBACK,
        prefilter_n: int,
    ) -> "PreparedDecision":
        _validate_runtime_preload(runtime, lookback)
        schema = action_schema or ActionSchema(
            factor_names=factors.factor_names,
            filter_names=factors.filter_names,
        )
        index = (
            runtime.decision_stop - 1
            if decision_index is None
            else int(decision_index)
        )
        listing_age = np.asarray(runtime.field("listing_age"), dtype=np.int32)
        builder = ObservationBuilder(
            runtime,
            factors,
            day_markets=tuple(build_day_market(runtime,factors,listing_age,i).seal(borrow_readonly=True) for i in range(runtime.n_dates)),
            lookback=lookback,
            listing_age=listing_age,
            action_schema=schema,
        )
        return cls(
            runtime=runtime,
            factors=factors,
            observation_builder=builder,
            listing_age=listing_age,
            action_schema=schema,
            decision_index=index,
            prefilter_n=prefilter_n,
        )

    @property
    def decision_date(self) -> str:
        return np.datetime_as_string(
            self.runtime.trade_dates[self.decision_index], unit="D"
        )

    @property
    def market(self) -> DayMarketData:
        return self.observation_builder.day_markets[self.decision_index]

    def build_observation(
        self,
        account: AccountState,
        policy_memory: PolicyMemory,
    ) -> Observation:
        return self.observation_builder.build(
            self.decision_index,
            account,
            policy_memory=policy_memory,
        )

    def plan(
        self,
        account: AccountState,
        config: DayConfig,
        policy_memory: PolicyMemory,
        *,
        diagnostics: str = "minimal",
    ) -> OrderPlan:
        self.action_schema.validate_day_config(config)
        previous_ranking: tuple[str, ...] | None = None
        if policy_memory.previous_day_config is not None:
            previous_index = self.decision_index - 1
            if previous_index < 0:
                raise ValueError("initialized policy memory needs a T-1 runtime row")
            previous_market = self.observation_builder.day_markets[previous_index]
            previous_ranking = rank_complete_universe(
                previous_market.stock_codes,
                previous_market.factor_ranks,
                previous_market.factor_validity,
                policy_memory.previous_day_config,
                pit_universe_mask=(previous_market.listing_age >= 0) & ~previous_market.delisted_mask,
            )
        market = self.market.with_candidate_mask(
            candidate_mask_from_previous_ranking(
                self.runtime.stock_codes,
                previous_ranking,
                self.prefilter_n,
                held_codes=tuple(account.positions),
            ),
        )
        return DayPlanner(diagnostics=diagnostics).plan(
            market,
            account,
            config,
        )


@dataclass(frozen=True)
class PreparedEpisode:
    """Immutable caches shared by all account sessions on one sealed split."""

    runtime: RuntimeSlice
    factors: FactorBatch
    observation_builder: ObservationBuilder | None
    encoder: ObservationEncoder | None
    market_store: RawMarketStore | None
    listing_age: NDArray[np.int32]
    decision_start: int
    decision_stop: int
    prefilter_n: int | None = None
    code_to_index: Mapping[str, int] = field(init=False, repr=False)
    decision_markets: tuple[DayMarketData, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.factors.runtime_schema_hash != self.runtime.manifest.schema_hash:
            raise ValueError("factor cache was built from a different runtime schema")
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
        if self.prefilter_n is not None and (
            type(self.prefilter_n) is not int or self.prefilter_n <= 0
        ):
            raise ValueError("prefilter_n must be a positive int or None")
        observation_parts = (self.observation_builder, self.encoder, self.market_store)
        if any(part is None for part in observation_parts):
            if not all(part is None for part in observation_parts):
                raise ValueError("observation components must be present or absent together")
        else:
            expected_indices = tuple(range(self.decision_start, self.decision_stop))
            if self.market_store.decision_indices != expected_indices:
                raise ValueError("raw market store does not cover the exact episode interval")
            if (self.market_store.schema.identifier != self.observation_builder.schema.identifier
                    or self.encoder.observation_schema.identifier != self.observation_builder.schema.identifier):
                raise ValueError("raw store, builder and encoder schemas differ")
            expected_dates = tuple(str(value) for value in self.runtime.trade_dates[self.decision_start:self.decision_stop])
            if self.market_store.decision_dates != expected_dates:
                raise ValueError("raw store dates differ from the sealed decision calendar")
        projection = self.runtime.manifest.replay_projection
        if projection is not None:
            if (projection.factor_schema_hash != self.factors.schema_hash
                    or projection.source_row_offset + self.runtime.n_dates > projection.source_rows
                    or projection.source_manifest.schema_hash != self.runtime.manifest.schema_hash
                    or projection.source_manifest.source_sha256 != self.runtime.manifest.source_sha256):
                raise ValueError("replay projection provenance differs from the prepared cache")
            observation_schema = None if self.observation_builder is None else self.observation_builder.schema.identifier
            encoded_schema = None if self.encoder is None else self.encoder.output_schema.identifier
            if (projection.observation_schema, projection.encoded_schema) != (observation_schema, encoded_schema):
                raise ValueError("replay projection changed the observation or encoder schema")
            required_fields = (set(item.name for item in projection.source_manifest.fields)
                               if self.observation_builder is not None else EXECUTION_RUNTIME_FIELDS)
            if set(projection.retained_fields) != required_fields:
                raise ValueError("replay retained fields differ from episode execution requirements")
        object.__setattr__(
            self,
            "code_to_index",
            MappingProxyType(
                {code: index for index, code in enumerate(self.runtime.stock_codes)}
            ),
        )
        object.__setattr__(self, "decision_markets", self.observation_builder.day_markets[self.decision_start:self.decision_stop] if self.observation_builder is not None else tuple(
            build_day_market(self.runtime, self.factors, self.listing_age, index).seal(borrow_readonly=True)
            for index in range(self.decision_start, self.decision_stop)
        ))

    def market_at(self, index: int) -> DayMarketData:
        """Return the shared immutable decision row within this sealed split."""
        if not self.decision_start <= index < self.decision_stop:
            raise IndexError("market index must remain in the sealed episode")
        return self.decision_markets[index - self.decision_start]

    @cached_property
    def factor_coverage(self) -> dict:
        """One PIT coverage scan shared by preparation logs and audit writers."""
        return factor_coverage(self.runtime, self.factors)

    def compact_for_replay(self) -> "PreparedEpisode":
        """Project completed caches to the smallest history needed by replay.

        Factor history was consumed before this operation.
        Raw Observation still needs L-1 prior rows plus the registered price
        lag; PolicyMemory needs L calendar rows and prefilter needs T-1.
        These are read-only views: shared transport materializes only retained
        rows, without allocating a second temporary copy of the large arrays.
        No-actor replay retains only explicitly declared execution fields;
        actor replay preserves every source field.
        """
        lookback = 0 if self.observation_builder is None else self.observation_builder.lookback
        retained = max(1, lookback, lookback - 1 + RUNTIME_DECISION_LAG_ROWS)
        row_start = max(0, self.decision_start - retained)
        retained_fields = tuple(name for name in self.runtime.data
                                if self.observation_builder is not None or name in EXECUTION_RUNTIME_FIELDS)
        if (row_start == 0 and self.decision_stop == self.runtime.n_dates
                and retained_fields == tuple(self.runtime.data)):
            return self
        prior = self.runtime.manifest.replay_projection
        offset = row_start + (0 if prior is None else prior.source_row_offset)
        rows = slice(row_start, self.decision_stop)
        start, stop = self.decision_start - row_start, self.decision_stop - row_start
        dates = self.runtime.trade_dates[rows]
        proof = ReplayProjection(
            source_manifest=self.runtime.manifest if prior is None else prior.source_manifest,
            source_rows=self.runtime.n_dates if prior is None else prior.source_rows,
            source_row_offset=offset, factor_schema_hash=self.factors.schema_hash,
            retained_fields=retained_fields,
            observation_schema=None if self.observation_builder is None else self.observation_builder.schema.identifier,
            encoded_schema=None if self.encoder is None else self.encoder.output_schema.identifier)
        manifest = replace(self.runtime.manifest, loaded_start=str(dates[0]), loaded_end=str(dates[-1]),
                           actual_preload_rows=start, replay_projection=proof)
        runtime = RuntimeSlice(stock_codes=self.runtime.stock_codes, trade_dates=dates,
            data={name: self.runtime.data[name][rows] if self.runtime.data[name].ndim == 2
                  else self.runtime.data[name] for name in retained_fields},
            decision_start=start, decision_stop=stop, manifest=manifest)
        factors = replace(self.factors, trade_dates=dates, decision_start=start, decision_stop=stop,
                          raw=self.factors.raw[rows], ranks=self.factors.ranks[rows],
                          validity=self.factors.validity[rows], filters=self.factors.filters[rows])
        ages = self.listing_age[rows]
        builder = encoder = cache = None
        if self.observation_builder is not None:
            builder = ObservationBuilder(runtime, factors, lookback=lookback, listing_age=ages,
                day_markets=self.observation_builder.day_markets[row_start:self.decision_stop],
                action_schema=self.observation_builder.action_schema)
            encoder = self.encoder
            cache = replace(self.market_store, row_start=self.market_store.row_start-row_start)
        return PreparedEpisode(runtime, factors, builder, encoder, cache, ages, start, stop, self.prefilter_n)

    @classmethod
    def build(
        cls,
        runtime: RuntimeSlice,
        factors: FactorBatch,
        *,
        decision_start: int | None = None,
        decision_stop: int | None = None,
        lookback: int = DEFAULT_LOOKBACK,
        prefilter_n: int | None = None,
        encode_observations: bool = True,
        action_schema: ActionSchema | None = None,
    ) -> "PreparedEpisode":
        """Seal one account episode, optionally without actor inputs.

        ``encode_observations=False`` is for fixed-config factor research.
        It omits the production-only actor encoding, never substitutes a
        different factor under a production name, and retains the same
        account timeline and execution semantics.
        """
        if type(encode_observations) is not bool:
            raise TypeError("encode_observations must be bool")
        if encode_observations:
            if factors.schema_version != FACTOR_SCHEMA_VERSION:
                raise ValueError("research factor schema cannot enter production observations")
            _validate_runtime_preload(runtime, lookback)
        else:
            required = max(
                item.hist_days + int(bool(item.lagged_fields))
                for item in (*factors.factor_metadata, *factors.filter_metadata)
            )
            if runtime.manifest.requested_preload_rows < required:
                raise ValueError("research runtime preload must cover factor history and lag")
        start = runtime.decision_start if decision_start is None else int(decision_start)
        stop = runtime.decision_stop if decision_stop is None else int(decision_stop)
        listing_age = np.asarray(runtime.field("listing_age"), dtype=np.int32)
        builder = encoder = cache = None
        if encode_observations:
            builder = ObservationBuilder(
                runtime,
                factors,
                day_markets=tuple(build_day_market(runtime,factors,listing_age,i).seal(borrow_readonly=True) for i in range(runtime.n_dates)),
                lookback=lookback,
                listing_age=listing_age,
                action_schema=action_schema,
            )
            encoder = ObservationEncoder(builder.schema)
            cache = RawMarketStore.precompute(builder,range(start,stop))
        return cls(
            runtime=runtime,
            factors=factors,
            observation_builder=builder,
            encoder=encoder,
            market_store=cache,
            listing_age=listing_age,
            decision_start=start,
            decision_stop=stop,
            prefilter_n=prefilter_n,
        )

    @property
    def observation_count(self) -> int:
        return self.decision_stop - self.decision_start

    @property
    def transition_count(self) -> int:
        return self.observation_count - 1


@dataclass(frozen=True)
class EpisodeTransition:
    """One canonical decision and its T-open to T+1-open settlement."""

    observation: NDArray[np.float32]
    day_config: DayConfig
    order_plan: OrderPlan
    step_result: StepResult
    info: Mapping[str, object]

    def __post_init__(self) -> None:
        observation = np.asarray(self.observation, dtype=np.float32)
        if observation.ndim != 1 or not np.isfinite(observation).all():
            raise ValueError("episode transition observation must be a finite vector")
        object.__setattr__(self, "observation", np.ascontiguousarray(observation))
        object.__setattr__(self, "info", MappingProxyType(dict(self.info)))


class EpisodeSession:
    """The only stateful account timeline for a sealed causal episode.

    The caller supplies a validated semantic :class:`DayConfig`; action-space
    decoding belongs to the outer adapter. Every transition then follows the
    one domain path: observation -> planner -> simulator fills/settlement ->
    account, policy memory and reward state -> next observation.
    """

    def __init__(
        self,
        episode: PreparedEpisode,
        *,
        action_schema: ActionSchema | None = None,
        normalizer: TrainOnlyNormalizer | None = None,
        initial_cash: float = 1_000_000.0,
        diagnostics: str = "minimal",
        include_critic_context: bool = False,
        fees: FeeSchedule = DEFAULT_FEE_SCHEDULE,
    ) -> None:
        if not np.isfinite(initial_cash) or initial_cash <= 0.0:
            raise ValueError("initial_cash must be finite and positive")
        self.episode = episode
        if episode.observation_builder is None and (
            normalizer is not None or include_critic_context
        ):
            raise ValueError("research episode without observations cannot use normalizer or critic context")
        self.action_schema = action_schema or ActionSchema(
            factor_names=episode.factors.factor_names,
            filter_names=episode.factors.filter_names,
        )
        if episode.factors.schema_version != FACTOR_SCHEMA_VERSION:
            if action_schema is None:
                self.action_schema = replace(
                    self.action_schema,
                    schema_version="day-config-research-v1-" + episode.factors.schema_hash,
                )
            elif action_schema.schema_version == ActionSchema().schema_version:
                raise ValueError("research factors require a distinct action schema version")
        if self.action_schema.factor_names != episode.factors.factor_names:
            raise ValueError("action and factor vocabularies differ")
        if self.action_schema.filter_names != episode.factors.filter_names:
            raise ValueError("action and filter vocabularies differ")
        if episode.observation_builder is not None and (
            self.action_schema.schema_hash
            != episode.observation_builder.schema.action_schema_hash
        ):
            raise ValueError("action and observation policy-memory schemas differ")
        if normalizer is not None and (
            normalizer.encoder_schema != episode.encoder.output_schema.identifier
        ):
            raise ValueError("normalizer and encoder schemas differ")
        self.normalizer = normalizer
        self.initial_cash = float(initial_cash)
        self.include_critic_context = bool(include_critic_context)
        self.fees = fees
        self._planner = DayPlanner(diagnostics=diagnostics, fees=fees)
        self._simulator = DaySimulator(fees=fees)
        self._index: int | None = None
        self._account: AccountState | None = None
        self._policy_memory = PolicyMemory()
        self._reward_state: EpisodeRewardState | None = None
        self._last_transition: EpisodeTransition | None = None
        self._prefilter_universe = (
            PrefilterUniverse(episode.runtime.stock_codes)
            if episode.prefilter_n is not None else None
        )
        self._previous_prefilter_indices: NDArray[np.intp] | None = None
        self._terminated = False
        self._decision_stop = episode.decision_stop

    @property
    def current_account(self) -> AccountState:
        if self._account is None:
            raise RuntimeError("episode session must be reset first")
        return self._account

    @property
    def current_index(self) -> int:
        if self._index is None:
            raise RuntimeError("episode session must be reset first")
        return self._index

    @property
    def current_policy_memory(self) -> PolicyMemory:
        return self._policy_memory

    @property
    def reward_state(self) -> EpisodeRewardState:
        if self._reward_state is None:
            raise RuntimeError("episode session has no reward state")
        return self._reward_state

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def last_transition(self) -> EpisodeTransition:
        if self._last_transition is None:
            raise RuntimeError("episode session has not completed a transition")
        return self._last_transition

    @property
    def current_observation(self) -> Observation:
        """Return the public, unencoded Policy contract at the current T-open."""

        self._require_observations()
        date = str(self.episode.runtime.trade_dates[self.current_index])
        store = self.episode.market_store
        raw, member, valid = store.window(store.row_reference(date))
        return self.episode.observation_builder.build(
            self.current_index,
            self.current_account,
            policy_memory=self.current_policy_memory,
            static=StaticObservation(self.current_index, date, raw, valid, member, store.schema.identifier),
        )

    @property
    def encoded_observation(self) -> NDArray[np.float32]:
        # Empty vectors are explicit placeholders for fixed-config research,
        # not actor tensors. Model/Gym entry points reject these episodes.
        if self.episode.observation_builder is None:
            return np.empty(0, dtype=np.float32)
        account_part = self.episode.observation_builder.build_account(
            self.current_index,
            self.current_account,
            policy_memory=self.current_policy_memory,
        )
        public_observation = self.episode.encoder.encode_account(account_part,self.episode.market_store)
        if not self.include_critic_context:
            return public_observation
        return np.ascontiguousarray(
            np.concatenate((public_observation, self._critic_context())),
            dtype=np.float32,
        )

    @property
    def observation_dimension(self) -> int:
        self._require_observations()
        return self.episode.encoder.output_dimension + (
            self.critic_context_dimension if self.include_critic_context else 0
        )

    @property
    def critic_context_dimension(self) -> int:
        self._require_observations()
        return critic_context_dimension(self.episode.encoder)

    def _require_observations(self) -> None:
        if self.episode.observation_builder is None:
            raise ValueError("research episode without observations supports fixed DayConfig replay only")

    def reset(
        self,
        *,
        account: AccountState | None = None,
        policy_memory: PolicyMemory | None = None,
        decision_start: int | None = None,
        decision_stop: int | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        if (account is None) != (policy_memory is None):
            raise ValueError("account and policy_memory must be supplied together")
        if account is not None:
            if not isinstance(account, AccountState):
                raise TypeError("account must be AccountState")
            if not isinstance(policy_memory, PolicyMemory):
                raise TypeError("policy_memory must be PolicyMemory")
            self._account = account
            self._policy_memory = policy_memory
        else:
            self._account = AccountState(
                cash=self.initial_cash,
                nav=self.initial_cash,
                peak_nav=self.initial_cash,
            )
            self._policy_memory = PolicyMemory()
        start = self.episode.decision_start if decision_start is None else int(decision_start)
        stop = self.episode.decision_stop if decision_stop is None else int(decision_stop)
        if not self.episode.decision_start <= start < stop <= self.episode.decision_stop:
            raise ValueError("reset interval must stay inside the sealed episode")
        if stop - start < 2:
            raise ValueError("reset interval needs at least one transition")
        self._index = start
        self._decision_stop = stop
        self._terminated = False
        self._last_transition = None
        self._previous_prefilter_indices = None
        horizon = self._decision_stop - self.current_index - 1
        self._reward_state = EpisodeRewardState.initial(horizon)
        observation = self.encoded_observation
        return observation, {
            "decision_date": self._date_text(self.current_index),
            "nav": self.current_account.nav,
            "episode_end": self._date_text(self._decision_stop - 1),
            "episode_transitions": horizon,
            "reward_schema_version": REWARD_SCHEMA_VERSION,
        }

    def step(self, config: DayConfig) -> EpisodeTransition:
        if self._terminated:
            raise RuntimeError("step called after termination; reset the episode session")
        if not isinstance(config, DayConfig):
            raise TypeError("EpisodeSession.step requires a DayConfig")
        self.action_schema.validate_day_config(config)
        index = self.current_index
        account = self.current_account
        if index >= self._decision_stop - 1:
            raise RuntimeError("sealed split has no next-open row for another action")

        market = self._market_day(index)
        if self.episode.prefilter_n is not None:
            market = market.with_candidate_mask(
                candidate_mask_from_previous_indices(
                    self._prefilter_universe,
                    self._previous_prefilter_indices,
                    self.episode.prefilter_n,
                    held_codes=tuple(account.positions),
                    ranking_is_prefix=True,
                ),
            )
        next_index = index + 1
        terminated = next_index == self._decision_stop - 1
        plan, step_result = self._settle_account(
            market,
            account,
            config,
            index=index,
            next_index=next_index,
            terminated=terminated,
        )
        if plan.diagnostics["full_investment_contract_satisfied"] is not True:
            raise RuntimeError(
                "planner violated the mandatory full-investment contract: "
                f"{market.decision_date}: "
                f"{plan.diagnostics['residual_cash_reason']}"
            )
        next_reward_state, episode_reward = self.reward_state.advance(
            float(step_result.diagnostics["net_log_return"])
        )
        self._reward_state = next_reward_state
        memory = step_result.policy_memory
        if self.episode.observation_builder is not None:
            memory = self.action_schema.advance_policy_memory(
                self._policy_memory, memory, decision_date=market.decision_date,
                history_length=self.episode.observation_builder.lookback,
            )
        step_result = replace(
            step_result,
            reward=episode_reward.reward,
            policy_memory=memory,
            diagnostics={
                **step_result.diagnostics,
                "reward_schema_version": REWARD_SCHEMA_VERSION,
                "episode_reward": episode_reward.as_dict(),
            },
        )
        self._account = step_result.account_state
        self._policy_memory = step_result.policy_memory
        self._index = next_index
        self._terminated = step_result.terminated
        observation = self.encoded_observation
        info: dict[str, object] = {
            "decision_date": market.decision_date,
            "next_decision_date": self._date_text(next_index),
            "day_config": self.action_schema.to_static_config(config),
            "nav": float(step_result.account_state.nav),
            "cash": float(step_result.account_state.cash),
            "portfolio_return": float(step_result.portfolio_return),
            "exposure": float(step_result.diagnostics["post_fill_exposure"]),
            "post_fill_cash": float(step_result.diagnostics["post_fill_cash"]),
            "full_investment_contract_satisfied": bool(
                plan.diagnostics["full_investment_contract_satisfied"]
            ),
            "planned_post_order_cash": float(
                plan.diagnostics["planned_post_order_cash"]
            ),
            "cheapest_next_legal_buy_cost": plan.diagnostics[
                "cheapest_next_legal_buy_cost"
            ],
            "residual_cash_reason": str(plan.diagnostics["residual_cash_reason"]),
            "fill_count": len(step_result.fills),
            "total_fees": float(step_result.diagnostics["total_fees"]),
            "gross_turnover_ratio": float(
                step_result.diagnostics["gross_turnover_ratio"]
            ),
            "total_cost_ratio": float(step_result.diagnostics["total_cost_ratio"]),
            "reward_schema_version": REWARD_SCHEMA_VERSION,
            "episode_reward": episode_reward.as_dict(),
        }
        transition = EpisodeTransition(
            observation=observation,
            day_config=config,
            order_plan=plan,
            step_result=step_result,
            info=info,
        )
        self._last_transition = transition
        return transition

    def _settle_account(
        self,
        market: DayMarketData,
        account: AccountState,
        config: DayConfig,
        *,
        index: int,
        next_index: int,
        terminated: bool,
    ) -> tuple[OrderPlan, StepResult]:
        if self.episode.prefilter_n is None:
            plan = self._planner.plan(market, account, config)
            next_prefilter_indices = None
        else:
            plan, next_prefilter_indices = self._planner.plan_and_rank(
                market, account, config, self._prefilter_universe,
                prefilter_n=self.episode.prefilter_n,
            )
        relevant_codes = set(account.positions)
        relevant_codes.update(code for code, _ in plan.sell_orders)
        relevant_codes.update(plan.buy_orders)
        known_codes = tuple(code for code in relevant_codes if code in self.episode.code_to_index)
        indices = np.asarray(
            [self.episode.code_to_index[code] for code in known_codes], dtype=np.intp,
        )
        # Settlement fields are gathered only after the decision plan is fixed.
        price_rows = tuple(
            dict(zip(known_codes, self.episode.runtime.field(name)[day, indices].tolist()))
            for day, name in (
                (index, "open"), (next_index, "open"),
                (index, "close"), (next_index, "preClose"),
            )
        )
        delisted = self.episode.runtime.field("delisted_mask")[next_index, indices]
        result = self._simulator.step(
            account,
            plan,
            price_rows[0],
            price_rows[1],
            close_prices=price_rows[2],
            next_preclose_prices=price_rows[3],
            next_delisted_codes=tuple(
                code for code, flag in zip(known_codes, delisted) if flag
            ),
            next_decision_date=self._date_text(next_index),
            terminated=terminated,
        )
        if self.episode.prefilter_n is not None:
            self._previous_prefilter_indices = next_prefilter_indices
        return plan, result

    def _critic_context(self) -> NDArray[np.float32]:
        state = self.reward_state
        candidate = state.performance_state
        horizon = float(candidate.horizon_transitions)
        elapsed = float(candidate.observed_transitions)
        scalar_context = np.asarray(
            (
                elapsed / horizon,
                (horizon - elapsed) / horizon,
                horizon / (horizon + ANNUALIZATION_DAYS),
                np.clip(
                    candidate.cumulative_log_return
                    / CRITIC_CONTEXT_LOG_RETURN_CLIP,
                    -1.0,
                    1.0,
                ),
                np.clip(candidate.simple_return_sum / horizon, -1.0, 1.0),
                np.clip(
                    np.sqrt(candidate.simple_return_square_sum / horizon),
                    0.0,
                    1.0,
                ),
                candidate.max_drawdown,
                MAX_DRAWDOWN_PENALTY_WEIGHT,
            ),
            dtype=np.float32,
        )
        result = np.ascontiguousarray(scalar_context, dtype=np.float32)
        if result.shape != (self.critic_context_dimension,):
            raise RuntimeError("critic context dimension changed")
        if not np.isfinite(result).all():
            raise ValueError("critic context must be finite")
        return result

    def _date_text(self, index: int) -> str:
        return self.episode.market_at(index).decision_date

    def _market_day(self, index: int) -> DayMarketData:
        return self.episode.market_at(index)

@dataclass(frozen=True)
class RolloutSeries:
    decision_dates: tuple[str, ...]
    next_decision_dates: tuple[str, ...]
    rewards: NDArray[np.float64]
    portfolio_returns: NDArray[np.float64]
    nav: NDArray[np.float64]
    cash: NDArray[np.float64]
    exposure: NDArray[np.float64]
    full_investment_contract: NDArray[np.bool_]

    def __post_init__(self) -> None:
        size = len(self.rewards)
        if any(len(values) != size for values in (
            self.decision_dates, self.next_decision_dates, self.portfolio_returns,
            self.cash, self.exposure, self.full_investment_contract,
        )) or self.nav.shape != (size + 1,):
            raise ValueError("rollout series have inconsistent transition lengths")
        if any(not np.isfinite(values).all() for values in (
            self.rewards, self.portfolio_returns, self.nav, self.cash, self.exposure,
        )):
            raise ValueError("rollout values must be finite")

    @property
    def metrics(self) -> PerformanceMetrics:
        return performance_from_log_rewards(self.net_log_returns)

    @property
    def net_log_returns(self) -> NDArray[np.float64]:
        return np.log1p(self.portfolio_returns)

    @property
    def average_exposure(self) -> float:
        return float(self.exposure.mean())

    @property
    def full_investment_contract_satisfied(self) -> bool:
        return bool(self.full_investment_contract.all())


@dataclass(frozen=True)
class RolloutSummary(RolloutSeries):
    """The same full account series without retaining every diagnostic object."""
    executed_sell_count: int
    total_fees: float
    sum_gross_turnover_ratio: float
    sum_total_cost_ratio: float


@dataclass(frozen=True)
class RolloutTrace(RolloutSeries):
    residual_cash_reasons: tuple[str, ...]
    order_plans: tuple[Mapping[str, object], ...]
    fills: tuple[tuple[Fill, ...], ...]
    fee_breakdowns: tuple[tuple[Mapping[str, object], ...], ...]
    account_events: tuple[Mapping[str, object], ...]
    actions: NDArray[np.float32]
    day_configs: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        size = len(self.rewards)
        if not (
            len(self.residual_cash_reasons)
            == len(self.order_plans)
            == len(self.fills)
            == len(self.fee_breakdowns)
            == len(self.actions)
            == len(self.day_configs)
            == size
        ):
            raise ValueError("rollout transition fields have inconsistent lengths")
        if any(
            len(day_fills) != len(day_breakdown)
            for day_fills, day_breakdown in zip(
                self.fills,
                self.fee_breakdowns,
                strict=True,
            )
        ):
            raise ValueError("rollout fee breakdown must match every fill")
        if any(not isinstance(event, Mapping) for event in self.account_events):
            raise ValueError("rollout account events must be mappings")
        if not np.isfinite(self.actions).all():
            raise ValueError("rollout values must be finite")

    @property
    def executed_sell_count(self) -> int:
        return sum(fill.side == 'sell' for fills in self.fills for fill in fills)

    def as_summary(self) -> dict[str, object]:
        return {
            "start": self.decision_dates[0],
            "end": self.next_decision_dates[-1],
            "average_exposure": self.average_exposure,
            "full_investment_contract_satisfied": (
                self.full_investment_contract_satisfied
            ),
            "account_event_count": len(self.account_events),
            "delist_write_off_count": sum(
                event.get("type") == "delist_write_off"
                for event in self.account_events
            ),
            "metrics": self.metrics.as_dict(),
        }


ActionProvider = Callable[[NDArray[np.float32]], NDArray[np.float32]]
DayConfigProvider = Callable[[NDArray[np.float32]], DayConfig]


def _session_from_adapter(adapter: object) -> EpisodeSession:
    if isinstance(adapter, EpisodeSession):
        return adapter
    session = getattr(adapter, "session", None)
    if not isinstance(session, EpisodeSession):
        raise TypeError("run_episode requires EpisodeSession or an adapter exposing one")
    return session


def _rollout_session(
    session: EpisodeSession,
    decision_provider: Callable[
        [NDArray[np.float32]], tuple[DayConfig, NDArray[np.float32]]
    ],
    *,
    record_details: bool = True,
) -> RolloutTrace | RolloutSummary:
    if type(record_details) is not bool:
        raise TypeError("record_details must be bool")
    observation, reset_info = session.reset()
    initial_nav = float(reset_info["nav"])
    total_fees = sum_gross_turnover = sum_total_cost = 0.0
    decision_dates: list[str] = []
    next_dates: list[str] = []
    rewards: list[float] = []
    returns: list[float] = []
    nav = [initial_nav]
    cash: list[float] = []
    exposure: list[float] = []
    full_investment_contract: list[bool] = []
    residual_cash_reasons: list[str] = []
    order_plans: list[Mapping[str, object]] = []
    fills: list[tuple[Fill, ...]] = []
    fee_breakdowns: list[tuple[Mapping[str, object], ...]] = []
    account_events: list[Mapping[str, object]] = []
    actions: list[NDArray[np.float32]] = []
    configs: list[Mapping[str, object]] = []
    sell_count = 0

    while not session.terminated:
        config, raw_action = decision_provider(observation.copy())
        action = np.asarray(raw_action, dtype=np.float32)
        transition = session.step(config)
        info = transition.info
        observation = transition.observation
        decision_dates.append(str(info["decision_date"]))
        next_dates.append(str(info["next_decision_date"]))
        rewards.append(float(transition.step_result.reward))
        returns.append(float(info["portfolio_return"]))
        nav.append(float(info["nav"]))
        cash.append(float(info["cash"]))
        exposure.append(float(info["exposure"]))
        full_investment_contract.append(
            bool(info["full_investment_contract_satisfied"])
        )
        raw_fees = transition.step_result.diagnostics["fee_breakdown"]
        if len(raw_fees) != len(transition.step_result.fills):
            raise ValueError("rollout fee breakdown must match every fill")
        if any(not isinstance(item, Mapping) for item in raw_fees):
            raise TypeError("StepResult fee breakdown must contain mappings")
        raw_account_events = transition.step_result.diagnostics.get("account_events", ())
        if not isinstance(raw_account_events, (tuple, list)):
            raise TypeError("StepResult account_events must be a sequence")
        if any(not isinstance(event, Mapping) for event in raw_account_events):
            raise TypeError("StepResult account event must be a mapping")
        if not record_details:
            sell_count += sum(fill.side == 'sell' for fill in transition.step_result.fills)
            total_fees += float(info['total_fees'])
            sum_gross_turnover += float(info['gross_turnover_ratio'])
            sum_total_cost += float(info['total_cost_ratio'])
            continue
        residual_cash_reasons.append(str(info["residual_cash_reason"]))
        plan = transition.order_plan
        order_plans.append(
            {
                "decision_date": plan.decision_date,
                "sell_orders": tuple(plan.sell_orders),
                "buy_orders": dict(plan.buy_orders),
                "day_config": dict(info["day_config"]),
                "diagnostics": dict(plan.diagnostics),
            }
        )
        fills.append(tuple(transition.step_result.fills))
        fee_breakdowns.append(
            tuple(
                dict(item)
                for item in raw_fees
            )
        )
        for event in raw_account_events:
            account_events.append(dict(event))
        actions.append(action.copy())
        configs.append(dict(info["day_config"]))

    expected_transitions = int(reset_info["episode_transitions"])
    if len(rewards) != expected_transitions:
        raise RuntimeError("rollout did not consume the exact sealed transition count")
    if not record_details:
        return RolloutSummary(
            decision_dates=tuple(decision_dates), next_decision_dates=tuple(next_dates),
            rewards=np.asarray(rewards, dtype=np.float64), portfolio_returns=np.asarray(returns, dtype=np.float64),
            nav=np.asarray(nav, dtype=np.float64), cash=np.asarray(cash, dtype=np.float64),
            exposure=np.asarray(exposure, dtype=np.float64),
            full_investment_contract=np.asarray(full_investment_contract, dtype=np.bool_),
            executed_sell_count=sell_count,
            total_fees=total_fees,
            sum_gross_turnover_ratio=sum_gross_turnover,
            sum_total_cost_ratio=sum_total_cost,
        )
    return RolloutTrace(
        decision_dates=tuple(decision_dates),
        next_decision_dates=tuple(next_dates),
        rewards=np.asarray(rewards, dtype=np.float64),
        portfolio_returns=np.asarray(returns, dtype=np.float64),
        nav=np.asarray(nav, dtype=np.float64),
        cash=np.asarray(cash, dtype=np.float64),
        exposure=np.asarray(exposure, dtype=np.float64),
        full_investment_contract=np.asarray(
            full_investment_contract, dtype=np.bool_
        ),
        residual_cash_reasons=tuple(residual_cash_reasons),
        order_plans=tuple(order_plans),
        fills=tuple(fills),
        fee_breakdowns=tuple(fee_breakdowns),
        account_events=tuple(account_events),
        actions=np.stack(actions).astype(np.float32, copy=False),
        day_configs=tuple(configs),
    )


def run_episode(
    adapter: object,
    action_provider: ActionProvider,
) -> RolloutTrace:
    """Run an encoded action provider directly on the canonical session."""

    session = _session_from_adapter(adapter)
    session._require_observations()

    def decide(observation: NDArray[np.float32]) -> tuple[DayConfig, NDArray[np.float32]]:
        action = np.asarray(action_provider(observation), dtype=np.float32)
        return session.action_schema.decode(action), action

    return _rollout_session(session, decide)


def run_day_config_episode(
    session: EpisodeSession,
    config_provider: DayConfigProvider,
    *,
    record_details: bool = True,
) -> RolloutTrace | RolloutSummary:
    """Run a fixed or dynamic DayConfig provider without importing Gym."""
    empty_action = np.empty(0, dtype=np.float32)

    def decide(observation: NDArray[np.float32]) -> tuple[DayConfig, NDArray[np.float32]]:
        config = config_provider(observation)
        return config, session.action_schema.encode(config) if record_details else empty_action

    return _rollout_session(session, decide, record_details=record_details)


def run_policy_episode(
    session: EpisodeSession,
    policy: Policy,
    *,
    deterministic: bool = True,
) -> RolloutTrace:
    """Run the public ``Policy.predict(Observation)`` contract on one session."""

    session._require_observations()
    def decide(_: NDArray[np.float32]) -> tuple[DayConfig, NDArray[np.float32]]:
        config = policy.predict(
            session.current_observation,
            deterministic=deterministic,
        )
        session.action_schema.validate_day_config(config)
        return config, session.action_schema.encode(config)

    return _rollout_session(session, decide)


def dynamic_config_summary(trace: RolloutTrace) -> dict[str, object]:
    """Describe actual continuous, binary and discrete variation in a rollout."""

    configs = trace.day_configs
    continuous: dict[str, list[float]] = {}
    categorical: dict[str, list[object]] = {}
    factor_names = tuple(configs[0]["weights"])
    filter_names = tuple(configs[0]["filter_factors"])
    for name in factor_names:
        continuous[f"factor_weight.{name}"] = [
            float(config["weights"][name]) for config in configs
        ]
        categorical[f"factor_enabled.{name}"] = [
            bool(config["factor_enabled"][name]) for config in configs
        ]
    for name in filter_names:
        categorical[f"filter_flag.{name}"] = [
            bool(config["filter_factors"][name]) for config in configs
        ]
    for name in ("turnover_rate", "rebalance_band_pct", "single_buy_pct"):
        continuous[name] = [float(config[name]) for config in configs]
    for name in ("buy_n", "limit_up_protection"):
        categorical[name] = [config[name] for config in configs]

    continuous_summary = {
        name: {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for name, values in continuous.items()
    }
    categorical_summary: dict[str, object] = {}
    for name, values in categorical.items():
        counts: dict[str, int] = {}
        for value in values:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        categorical_summary[name] = {
            "unique_count": len(counts),
            "counts": counts,
            "coverage": {key: count / len(values) for key, count in counts.items()},
        }
    return {
        "transition_count": len(configs),
        "continuous": continuous_summary,
        "categorical": categorical_summary,
    }


__all__ = [
    "ActionProvider",
    "CRITIC_CONTEXT_SCALAR_FEATURE_NAMES",
    "CRITIC_CONTEXT_SCHEMA_VERSION",
    "DayConfigProvider",
    "ENVIRONMENT_SCHEMA_VERSION",
    "EpisodeSession",
    "EpisodeTransition",
    "PreparedDecision",
    "PreparedEpisode",
    "RolloutSeries",
    "RolloutSummary",
    "RolloutTrace",
    "dynamic_config_summary",
    "build_day_market",
    "critic_context_dimension",
    "critic_context_feature_names",
    "environment_schema_manifest",
    "neutral_critic_context",
    "run_day_config_episode",
    "run_episode",
    "run_policy_episode",
]
