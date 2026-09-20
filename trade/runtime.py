"""Fail-closed assembly for one production T-open decision.

The live edge is allowed to provide a sealed local runtime snapshot and a
broker account.  It is not allowed to build a smaller candidate universe or
to fall back to a separate live scoring/planning path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from env.action_schema import ActionSchema
from env.backtest import (
    PreparedDecision,
    build_day_market,
    required_runtime_preload_rows,
)
from env.contracts import (
    AccountState,
    DayConfig,
    Observation,
    OrderPlan,
    Policy,
    PolicyMemory,
)
from env.observation import DEFAULT_LOOKBACK, ObservationSchema
from env.prefilter import (
    candidate_mask_from_previous_ranking,
    rank_complete_universe,
)
from factor import precompute_factors
from offline_data import load_runtime_slice, validate_runtime_path_identity


class SnapshotIntegrityError(ValueError):
    """The local decision snapshot is not the model's complete sealed axis."""


def live_prefilter_codes(
    runtime_path: str | Path,
    decision_date: object,
    previous_config: DayConfig | None,
    *,
    previous_decision_date: str | None,
    prefilter_n: int,
    held_codes: Iterable[str] = (),
    lookback: int = DEFAULT_LOOKBACK,
) -> tuple[str, ...] | None:
    """Rebuild the exact T candidate set from the sealed T-1 factor row.

    ``None`` is the explicit cold-start result and instructs the data edge to
    fetch the complete active universe. The returned tuple otherwise follows
    the runtime stock order and is shared with the planner mask semantics.
    """

    if previous_config is None:
        if previous_decision_date is not None:
            raise SnapshotIntegrityError(
                "cold live prefilter cannot carry a previous decision date"
            )
        return None
    if previous_decision_date is None:
        raise SnapshotIntegrityError(
            "initialized live prefilter is missing its previous decision date"
        )
    path = Path(runtime_path).resolve()
    target = np.datetime64(decision_date, "D")
    with np.load(path, allow_pickle=False) as payload:
        dates = np.asarray(payload["trade_dates"], dtype="datetime64[D]")
    prior = dates[dates < target]
    if prior.size == 0:
        raise SnapshotIntegrityError("live prefilter has no sealed T-1 runtime row")
    previous_date = prior[-1]
    if np.datetime64(previous_decision_date, "D") != previous_date:
        raise SnapshotIntegrityError(
            "live journal is not continuous with the sealed T-1 runtime row"
        )
    runtime = load_runtime_slice(
        path,
        previous_date,
        previous_date,
        preload_rows=required_runtime_preload_rows(lookback),
    )
    factors = precompute_factors(runtime)
    listing_age = np.asarray(runtime.field("listing_age"), dtype=np.int32)
    market = build_day_market(
        runtime,
        factors,
        listing_age,
        runtime.decision_start,
    )
    ranking = rank_complete_universe(
        market.stock_codes,
        market.factor_ranks,
        market.factor_validity,
        previous_config,
        pit_universe_mask=(market.listing_age >= 0) & ~market.delisted_mask,
    )
    mask = candidate_mask_from_previous_ranking(
        runtime.stock_codes,
        ranking,
        prefilter_n,
        held_codes=tuple(held_codes),
    )
    return tuple(
        code for code, selected in zip(runtime.stock_codes, mask, strict=True)
        if selected
    )


@dataclass(frozen=True)
class DecisionSnapshot:
    prepared: PreparedDecision
    source_path: Path
    source_sha256: str

    @property
    def decision_date(self) -> str:
        return self.prepared.decision_date

    @property
    def stock_codes(self) -> tuple[str, ...]:
        return self.prepared.runtime.stock_codes

    @property
    def identity(self) -> Mapping[str, object]:
        runtime = self.prepared.runtime
        return MappingProxyType(
            {
                "decision_date": self.decision_date,
                "runtime_path": str(self.source_path),
                "runtime_source_sha256": self.source_sha256,
                "runtime_schema_hash": runtime.manifest.schema_hash,
                "stock_vocabulary_sha256": (
                    runtime.manifest.stock_vocabulary_sha256
                ),
                "stock_count": runtime.n_stocks,
                "factor_schema_hash": self.prepared.factors.schema_hash,
                "observation_schema": (
                    self.prepared.observation_builder.schema.identifier
                ),
                "prefilter_n": self.prepared.prefilter_n,
            }
        )


class SealedSnapshotAdapter:
    """Load only an already-complete local runtime row.

    A 09:25 overlay is accepted only after an upstream data adapter has
    materialised it into the same versioned runtime schema and complete stock
    vocabulary as the frozen policy.  Partial code lists and quick-K-line
    dictionaries are deliberately not accepted here.
    """

    def __init__(
        self,
        *,
        action_schema: ActionSchema,
        observation_schema: ObservationSchema,
        expected_runtime: Mapping[str, object],
        expected_factors: Mapping[str, object],
        prefilter_n: int,
    ) -> None:
        self.action_schema = action_schema
        self.observation_schema = observation_schema
        self.expected_runtime = dict(expected_runtime)
        self.expected_factors = dict(expected_factors)
        self.prefilter_n = prefilter_n
        if type(prefilter_n) is not int or prefilter_n <= 0:
            raise SnapshotIntegrityError("policy prefilter_n is invalid")
        required_runtime = {
            "schema_hash",
            "source_sha256",
            "stock_vocabulary_sha256",
            "lineage",
        }
        missing_runtime = required_runtime - set(self.expected_runtime)
        if missing_runtime:
            raise SnapshotIntegrityError(
                "policy runtime identity is incomplete: "
                + ", ".join(sorted(missing_runtime))
            )
        if "schema_hash" not in self.expected_factors:
            raise SnapshotIntegrityError("policy factor identity is incomplete")
        if observation_schema.action_schema_hash != action_schema.schema_hash:
            raise SnapshotIntegrityError(
                "policy action and observation schemas do not match"
            )

    @classmethod
    def from_policy(cls, policy: object) -> "SealedSnapshotAdapter":
        manifest = getattr(policy, "manifest", None)
        action_schema = getattr(policy, "action_schema", None)
        encoder = getattr(policy, "encoder", None)
        observation_schema = getattr(encoder, "observation_schema", None)
        if manifest is None or not isinstance(action_schema, ActionSchema):
            raise SnapshotIntegrityError(
                "live policy must carry a verified deployable bundle manifest"
            )
        if not isinstance(observation_schema, ObservationSchema):
            raise SnapshotIntegrityError(
                "live policy has no frozen ObservationSchema"
            )
        return cls(
            action_schema=action_schema,
            observation_schema=observation_schema,
            expected_runtime=manifest.runtime,
            expected_factors=manifest.factors,
            prefilter_n=getattr(policy, "prefilter_n", None),
        )

    def load(
        self,
        runtime_path: str | Path,
        decision_date: object,
    ) -> DecisionSnapshot:
        path = Path(runtime_path).resolve()
        try:
            runtime = load_runtime_slice(
                path,
                decision_date,
                decision_date,
                preload_rows=required_runtime_preload_rows(
                    self.observation_schema.lookback
                ),
            )
        except (KeyError, OSError, ValueError) as exc:
            raise SnapshotIntegrityError(
                "complete sealed T-open runtime snapshot is unavailable"
            ) from exc
        expected_date = np.datetime64(decision_date, "D")
        if runtime.decision_dates.shape != (1,) or runtime.decision_dates[0] != expected_date:
            raise SnapshotIntegrityError(
                "runtime must contain exactly the requested T-open decision row"
            )
        try:
            validate_runtime_path_identity(
                runtime.manifest,
                self.expected_runtime,
                path,
            )
        except ValueError as exc:
            raise SnapshotIntegrityError(str(exc)) from exc
        factors = precompute_factors(runtime)
        if factors.schema_hash != str(self.expected_factors["schema_hash"]):
            raise SnapshotIntegrityError("factor schema differs from policy bundle")
        prepared = PreparedDecision.build(
            runtime,
            factors,
            action_schema=self.action_schema,
            lookback=self.observation_schema.lookback,
            prefilter_n=self.prefilter_n,
        )
        if prepared.observation_builder.schema.identifier != self.observation_schema.identifier:
            raise SnapshotIntegrityError(
                "complete snapshot produces a different ObservationSchema"
            )
        return DecisionSnapshot(
            prepared=prepared,
            source_path=path,
            source_sha256=runtime.manifest.source_sha256,
        )


@dataclass(frozen=True)
class LiveDecision:
    observation: Observation
    action: NDArray[np.float32]
    day_config: DayConfig
    order_plan: OrderPlan
    account_before: AccountState
    policy_memory_before: PolicyMemory
    snapshot_identity: Mapping[str, object]
    policy_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        action = np.asarray(self.action, dtype=np.float32)
        if action.ndim != 1 or not np.isfinite(action).all():
            raise ValueError("canonical action must be a finite vector")
        object.__setattr__(self, "action", np.ascontiguousarray(action))
        object.__setattr__(
            self,
            "snapshot_identity",
            MappingProxyType(dict(self.snapshot_identity)),
        )
        object.__setattr__(
            self,
            "policy_identity",
            MappingProxyType(dict(self.policy_identity)),
        )
        required_snapshot = {
            "decision_date",
            "runtime_path",
            "runtime_source_sha256",
            "runtime_schema_hash",
            "stock_vocabulary_sha256",
            "stock_count",
            "factor_schema_hash",
            "observation_schema",
            "prefilter_n",
        }
        required_policy = {
            "bundle_version",
            "created_at",
            "algorithm",
            "manifest_sha256",
            "model_sha256",
            "normalizer_sha256",
            "config_sha256",
            "source_sha256",
            "action_schema_hash",
            "observation_schema",
            "environment_schema_hash",
        }
        if set(self.snapshot_identity) != required_snapshot:
            raise ValueError("snapshot identity fields must match exactly")
        if not required_policy.issubset(self.policy_identity):
            raise ValueError("policy identity is incomplete")
        if self.snapshot_identity["decision_date"] != self.order_plan.decision_date:
            raise ValueError("snapshot identity decision date differs from OrderPlan")
        if self.observation.decision_date != self.order_plan.decision_date:
            raise ValueError("Observation decision date differs from OrderPlan")
        if (
            self.snapshot_identity["observation_schema"]
            != self.observation.schema_version
            or self.policy_identity["observation_schema"]
            != self.observation.schema_version
        ):
            raise ValueError("decision identities do not bind the ObservationSchema")
        for identity, names in (
            (
                self.snapshot_identity,
                (
                    "runtime_source_sha256",
                    "runtime_schema_hash",
                    "stock_vocabulary_sha256",
                    "factor_schema_hash",
                ),
            ),
            (
                self.policy_identity,
                (
                    "manifest_sha256",
                    "model_sha256",
                    "normalizer_sha256",
                    "config_sha256",
                    "source_sha256",
                    "action_schema_hash",
                    "environment_schema_hash",
                ),
            ),
        ):
            if any(len(str(identity[name])) != 64 for name in names):
                raise ValueError("decision identity contains an invalid SHA-256 digest")


class LiveDecisionRunner:
    """Observation -> Policy -> DayConfig -> canonical env OrderPlan."""

    def __init__(self, action_schema: ActionSchema) -> None:
        self.action_schema = action_schema

    def decide(
        self,
        snapshot: DecisionSnapshot,
        account: AccountState,
        policy_memory: PolicyMemory,
        policy: Policy,
    ) -> LiveDecision:
        if snapshot.prepared.action_schema.schema_hash != self.action_schema.schema_hash:
            raise SnapshotIntegrityError("snapshot and runner action schemas differ")
        observation = snapshot.prepared.build_observation(account, policy_memory)
        config = policy.predict(observation, deterministic=True)
        self.action_schema.validate_day_config(config)
        action = self.action_schema.encode(config)
        plan = snapshot.prepared.plan(
            account,
            config,
            policy_memory,
            diagnostics="minimal",
        )
        if plan.decision_date != snapshot.decision_date:
            raise RuntimeError("planner decision date differs from sealed snapshot")
        if not bool(plan.diagnostics["full_investment_contract_satisfied"]):
            raise RuntimeError("live planner did not prove the full-investment contract")
        manifest = getattr(policy, "manifest", None)
        if manifest is None:
            raise SnapshotIntegrityError(
                "live policy has no frozen bundle identity"
            )
        manifest_payload = manifest.to_dict()
        manifest_sha256 = hashlib.sha256(
            json.dumps(
                manifest_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        policy_identity = {
            "bundle_version": str(manifest.bundle_version),
            "created_at": str(manifest.created_at),
            "algorithm": str(manifest.algorithm),
            "manifest_sha256": manifest_sha256,
            "model_sha256": str(manifest.model_sha256),
            "normalizer_sha256": str(manifest.normalizer_sha256),
            "config_sha256": str(manifest.config_sha256),
            "source_sha256": str(manifest.source_sha256),
            "action_schema_hash": self.action_schema.schema_hash,
            "observation_schema": observation.schema_version,
            "environment_schema_hash": str(
                manifest.environment.get("schema_hash", "")
            ),
        }
        return LiveDecision(
            observation=observation,
            action=action,
            day_config=config,
            order_plan=plan,
            account_before=account,
            policy_memory_before=policy_memory,
            snapshot_identity=snapshot.identity,
            policy_identity=policy_identity,
        )


class BrokerAccountAdapter:
    """Convert a broker snapshot to the env account at the sealed T-open."""

    @staticmethod
    def _finite(value: object, *, label: str, minimum: float = 0.0) -> float:
        if isinstance(value, bool):
            raise SnapshotIntegrityError(f"{label} must be numeric")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise SnapshotIntegrityError(f"{label} must be numeric") from exc
        if not math.isfinite(result) or result < minimum:
            raise SnapshotIntegrityError(
                f"{label} must be finite and at least {minimum}"
            )
        return result

    @staticmethod
    def _positive_or_none(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) and result > 0.0 else None

    def _holding_mark(
        self,
        *,
        code: str,
        quantity: int,
        runtime_open: object,
        runtime_preclose: object,
        position: object,
        previous_account: AccountState | None,
    ) -> tuple[float, str]:
        mark = self._positive_or_none(runtime_open)
        if mark is not None:
            return mark, "runtime.open[T]"
        mark = self._positive_or_none(runtime_preclose)
        if mark is not None:
            return mark, "runtime.preClose[T]"
        if previous_account is not None:
            mark = self._positive_or_none(previous_account.last_prices.get(code))
            if mark is not None:
                return mark, "previous_account.last_prices"
        mark = self._positive_or_none(getattr(position, "last_price", None))
        if mark is not None:
            return mark, "broker.last_price"
        market_value = self._positive_or_none(
            getattr(position, "market_value", None)
        )
        if market_value is not None:
            return market_value / quantity, "broker.market_value/volume"
        raise SnapshotIntegrityError(
            f"broker holding {code!r} has no causal mark from runtime, "
            "previous account or broker snapshot"
        )

    def build(
        self,
        *,
        asset: object,
        positions: Iterable[object],
        snapshot: DecisionSnapshot,
        previous_account: AccountState | None = None,
    ) -> AccountState:
        if asset is None:
            raise SnapshotIntegrityError("broker asset snapshot is missing")
        cash = self._finite(getattr(asset, "cash", None), label="broker cash")
        market = snapshot.prepared.market
        code_to_index = {
            code: index for index, code in enumerate(market.stock_codes)
        }
        quantities: dict[str, int] = {}
        sellable: dict[str, int] = {}
        average_costs: dict[str, float] = {}
        last_prices: dict[str, float] = {}
        mark_provenance: dict[str, str] = {}
        for position in positions:
            code = str(getattr(position, "stock_code", "") or "")
            quantity = int(getattr(position, "volume", 0) or 0)
            if quantity <= 0:
                continue
            if code not in code_to_index:
                raise SnapshotIntegrityError(
                    f"broker holding {code!r} is outside the complete runtime axis"
                )
            index = code_to_index[code]
            mark, mark_source = self._holding_mark(
                code=code,
                quantity=quantity,
                runtime_open=market.open_prices[index],
                runtime_preclose=market.preclose_prices[index],
                position=position,
                previous_account=previous_account,
            )
            can_sell = int(getattr(position, "can_use_volume", 0) or 0)
            if not 0 <= can_sell <= quantity:
                raise SnapshotIntegrityError(
                    f"broker holding {code!r} has invalid sellable quantity"
                )
            average = getattr(position, "avg_price", None)
            if average is None:
                average = getattr(position, "open_price", None)
            average_value = self._finite(
                average,
                label=f"broker holding {code!r} average cost",
            )
            quantities[code] = quantity
            sellable[code] = can_sell
            average_costs[code] = average_value
            last_prices[code] = mark
            mark_provenance[code] = mark_source
        nav = cash + sum(
            quantities[code] * last_prices[code] for code in quantities
        )
        if nav <= 0.0:
            raise SnapshotIntegrityError("broker T-open NAV must be positive")
        if previous_account is None:
            peak_nav = nav
            max_drawdown = 0.0
        else:
            peak_nav = max(float(previous_account.peak_nav), nav)
            current_drawdown = max(0.0, 1.0 - nav / peak_nav)
            max_drawdown = max(float(previous_account.max_drawdown), current_drawdown)
        return AccountState(
            cash=cash,
            positions=quantities,
            sellable_positions=sellable,
            average_costs=average_costs,
            last_prices=last_prices,
            mark_provenance=mark_provenance,
            nav=nav,
            peak_nav=peak_nav,
            max_drawdown=max_drawdown,
        )


__all__ = [
    "BrokerAccountAdapter",
    "DecisionSnapshot",
    "LiveDecision",
    "LiveDecisionRunner",
    "SealedSnapshotAdapter",
    "SnapshotIntegrityError",
]
