"""Immutable live-decision journal and replay without policy inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

import numpy as np

from env.action_schema import ActionSchema
from env.fees import summarize_fill_costs
from env.contracts import (
    AccountState,
    Fill,
    Observation,
    OrderPlan,
    PolicyMemory,
    PolicyHistory,
)
from trade.executor import AcceptedBrokerOrder, validate_accepted_orders
from trade.runtime import LiveDecision
from utils.atomic_file import atomic_write_json


JOURNAL_VERSION = "wbr-live-decision-journal-v9-raw-panel"


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"journal value is not serializable: {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    atomic_write_json(path, _json_ready(payload), indent=None, separators=(",", ":"))


def _atomic_observation(path: Path, observation: Observation) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    digest = hashlib.sha256()
    arrays = (
        observation.stock_panel,
        observation.position_panel,
        observation.portfolio,
        observation.policy_history,
        observation.time_mask,
        observation.pit_universe_mask,
    )
    for values in arrays:
        digest.update(np.ascontiguousarray(values).tobytes())
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                stock_panel=observation.stock_panel,
                position_panel=observation.position_panel,
                portfolio=observation.portfolio,
                policy_history=observation.policy_history,
                time_mask=observation.time_mask,
                pit_universe_mask=observation.pit_universe_mask,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return digest.hexdigest()


def _account_payload(account: AccountState) -> dict[str, object]:
    return {
        "cash": account.cash,
        "positions": dict(account.positions),
        "sellable_positions": dict(account.sellable_positions),
        "average_costs": dict(account.average_costs),
        "last_prices": dict(account.last_prices),
        "mark_provenance": dict(account.mark_provenance),
        "nav": account.nav,
        "peak_nav": account.peak_nav,
        "max_drawdown": account.max_drawdown,
    }


def _account_from_payload(payload: Mapping[str, object]) -> AccountState:
    return AccountState(
        cash=float(payload["cash"]),
        positions={str(k): int(v) for k, v in dict(payload["positions"]).items()},
        sellable_positions={
            str(k): int(v)
            for k, v in dict(payload["sellable_positions"]).items()
        },
        average_costs={
            str(k): float(v) for k, v in dict(payload["average_costs"]).items()
        },
        last_prices={
            str(k): float(v) for k, v in dict(payload["last_prices"]).items()
        },
        mark_provenance={
            str(k): str(v)
            for k, v in dict(payload["mark_provenance"]).items()
        },
        nav=float(payload["nav"]),
        peak_nav=float(payload["peak_nav"]),
        max_drawdown=float(payload["max_drawdown"]),
    )


def _memory_payload(
    memory: PolicyMemory,
    schema: ActionSchema,
) -> dict[str, object]:
    schema.validate_policy_memory(memory)
    return {
        "previous_day_config": (
            None
            if memory.previous_day_config is None
            else schema.to_static_config(memory.previous_day_config)
        ),
        "previous_gross_turnover_ratio": memory.previous_gross_turnover_ratio,
        "previous_total_cost_ratio": memory.previous_total_cost_ratio,
        "history": None if memory.history is None else {
            "decision_dates": np.datetime_as_string(memory.history.decision_dates, unit="D").tolist(),
            "values": memory.history.values.tolist(),
            "action_schema_hash": memory.history.action_schema_hash,
        },
    }


def _memory_from_payload(
    payload: Mapping[str, object],
    schema: ActionSchema,
) -> PolicyMemory:
    config_payload = payload["previous_day_config"]
    config = (
        None
        if config_payload is None
        else schema.from_serialized_day_config(dict(config_payload))
    )
    history = payload["history"]
    memory = PolicyMemory(
        previous_day_config=config,
        previous_gross_turnover_ratio=float(
            payload["previous_gross_turnover_ratio"]
        ),
        previous_total_cost_ratio=float(payload["previous_total_cost_ratio"]),
        history=None if history is None else PolicyHistory(
            decision_dates=np.asarray(history["decision_dates"], dtype="datetime64[D]"),
            values=np.asarray(history["values"], dtype=np.float64),
            action_schema_hash=str(history["action_schema_hash"]),
        ),
    )
    schema.validate_policy_memory(memory)
    return memory


def _memory_for_fills(
    decision_payload: Mapping[str, object], fills: tuple[Fill, ...], schema: ActionSchema,
) -> PolicyMemory:
    """One journal authority for validating actual fills and extending history."""
    if decision_payload["journal_version"] != JOURNAL_VERSION:
        raise ValueError("unsupported decision journal version")
    config = schema.from_serialized_day_config(decision_payload["day_config"])
    if not np.array_equal(np.asarray(decision_payload["canonical_action"]), schema.encode(config)):
        raise ValueError("journal canonical action differs from its DayConfig")
    if decision_payload["order_plan"]["decision_date"] != decision_payload["decision_date"]:
        raise ValueError("journal OrderPlan decision date mismatch")
    fill_tuple = fills
    if any(not isinstance(fill, Fill) for fill in fill_tuple):
        raise TypeError("fills must contain env.contracts.Fill values")
    raw_plan = decision_payload["order_plan"]
    planned = {
        ("sell", str(code)): int(quantity)
        for code, quantity in raw_plan["sell_orders"]
    }
    planned.update(
        {
            ("buy", str(code)): int(quantity)
            for code, quantity in dict(raw_plan["buy_orders"]).items()
        }
    )
    actual: dict[tuple[str, str], int] = {}
    for fill in fill_tuple:
        key = (fill.side, fill.code)
        if key not in planned:
            raise ValueError("broker Fill is not part of the recorded OrderPlan")
        actual[key] = actual.get(key, 0) + fill.quantity
        if actual[key] > planned[key]:
            raise ValueError("broker Fill exceeds the recorded OrderPlan quantity")
    account = _account_from_payload(decision_payload["account_before"])
    pretrade_nav = float(account.nav)
    if not math.isfinite(pretrade_nav) or pretrade_nav <= 0.0:
        raise ValueError("journal pretrade NAV must be positive")
    gross_notional, total_cost = summarize_fill_costs(fill_tuple)
    memory = PolicyMemory(
        previous_day_config=config,
        previous_gross_turnover_ratio=gross_notional / pretrade_nav,
        previous_total_cost_ratio=total_cost / pretrade_nav,
    )
    memory = schema.advance_policy_memory(
        _memory_from_payload(decision_payload["policy_memory_before"], schema),
        memory, decision_date=str(decision_payload["decision_date"]),
        history_length=int(decision_payload["policy_history_length"]),
    )
    return memory


def _fill_payload(fill: Fill) -> dict[str, object]:
    return {
        "code": fill.code,
        "side": fill.side,
        "quantity": fill.quantity,
        "price": fill.price,
        "fee": fill.fee,
        "timestamp": fill.timestamp,
    }


def _fill_from_payload(payload: Mapping[str, object]) -> Fill:
    return Fill(
        code=str(payload["code"]),
        side=str(payload["side"]),
        quantity=int(payload["quantity"]),
        price=float(payload["price"]),
        fee=float(payload["fee"]),
        timestamp=str(payload["timestamp"]),
    )


def _accepted_order_payload(order: AcceptedBrokerOrder) -> dict[str, object]:
    return {
        "order_id": order.order_id,
        "side": order.side,
        "code": order.code,
        "planned_quantity": order.planned_quantity,
    }


def _accepted_order_from_payload(
    payload: Mapping[str, object],
) -> AcceptedBrokerOrder:
    return AcceptedBrokerOrder(
        order_id=int(payload["order_id"]),
        side=str(payload["side"]),
        code=str(payload["code"]),
        planned_quantity=int(payload["planned_quantity"]),
    )


def _validate_fills_within_accepted_orders(
    fills: Iterable[Fill],
    accepted_orders: Iterable[AcceptedBrokerOrder],
) -> tuple[Fill, ...]:
    fill_tuple = tuple(fills)
    accepted = tuple(accepted_orders)
    planned = {
        (order.side, order.code): order.planned_quantity for order in accepted
    }
    actual: dict[tuple[str, str], int] = {}
    for fill in fill_tuple:
        if not isinstance(fill, Fill):
            raise TypeError("broker fills must contain env.contracts.Fill values")
        key = (fill.side, fill.code)
        if key not in planned:
            raise ValueError("broker Fill does not belong to an accepted order")
        actual[key] = actual.get(key, 0) + fill.quantity
        if actual[key] > planned[key]:
            raise ValueError("broker Fill exceeds accepted order quantity")
    return fill_tuple


def _order_plan_from_payload(
    decision_payload: Mapping[str, object],
    schema: ActionSchema,
) -> OrderPlan:
    config = schema.from_serialized_day_config(decision_payload["day_config"])
    raw_plan = decision_payload["order_plan"]
    return OrderPlan(
        decision_date=str(raw_plan["decision_date"]),
        sell_orders=tuple(
            (str(code), int(quantity))
            for code, quantity in raw_plan["sell_orders"]
        ),
        buy_orders={
            str(code): int(quantity)
            for code, quantity in dict(raw_plan["buy_orders"]).items()
        },
        day_config=config,
        diagnostics=dict(raw_plan["diagnostics"]),
    )


@dataclass(frozen=True)
class JournalReplay:
    observation: Observation
    action: np.ndarray
    order_plan: OrderPlan
    account_before: AccountState
    policy_memory_before: PolicyMemory
    fills: tuple[Fill, ...] | None
    policy_memory_after: PolicyMemory | None
    snapshot_identity: Mapping[str, object]
    policy_identity: Mapping[str, object]


class DecisionJournal:
    """One immutable directory per decision date.

    ``decision.json`` and ``observation.npz`` are written before broker
    execution. ``fills.json`` is written once afterwards. Replay only reads
    those artifacts; it never receives a Policy object.
    """

    def __init__(self, root: str | Path, action_schema: ActionSchema) -> None:
        self.root = Path(root).resolve()
        self.action_schema = action_schema

    def _day_root(self, decision_date: str | date) -> Path:
        text = (
            decision_date.isoformat()
            if isinstance(decision_date, date)
            else str(decision_date)
        )
        date.fromisoformat(text)
        return self.root / text

    def record_decision(self, decision: LiveDecision) -> None:
        if decision.order_plan.day_config != decision.day_config:
            raise ValueError("OrderPlan and live decision DayConfig differ")
        expected_action = self.action_schema.encode(decision.day_config)
        if not np.array_equal(expected_action, decision.action):
            raise ValueError("live decision action is not the canonical DayConfig encoding")
        if (
            decision.policy_identity["action_schema_hash"]
            != self.action_schema.schema_hash
        ):
            raise ValueError("policy identity and journal ActionSchema differ")
        day_root = self._day_root(decision.order_plan.decision_date)
        try:
            day_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"decision journal already exists for {day_root.name}"
            ) from exc
        observation_file = day_root / "observation.npz"
        observation_sha256 = _atomic_observation(
            observation_file,
            decision.observation,
        )
        plan = decision.order_plan
        _atomic_json(
            day_root / "decision.json",
            {
                "journal_version": JOURNAL_VERSION,
                "decision_date": plan.decision_date,
                "observation_file": observation_file.name,
                "observation_sha256": observation_sha256,
                "observation_schema": decision.observation.schema_version,
                "policy_history_length": decision.observation.policy_history.shape[0],
                "canonical_action": decision.action.tolist(),
                "day_config": self.action_schema.to_static_config(
                    decision.day_config
                ),
                "order_plan": {
                    "decision_date": plan.decision_date,
                    "sell_orders": [list(item) for item in plan.sell_orders],
                    "buy_orders": dict(plan.buy_orders),
                    "diagnostics": dict(plan.diagnostics),
                },
                "account_before": _account_payload(decision.account_before),
                "policy_memory_before": _memory_payload(
                    decision.policy_memory_before,
                    self.action_schema,
                ),
                "snapshot_identity": dict(decision.snapshot_identity),
                "policy_identity": dict(decision.policy_identity),
            },
        )

    def record_fills(
        self,
        decision_date: str | date,
        fills: Iterable[Fill],
        *,
        _allow_pending: bool = False,
    ) -> PolicyMemory:
        day_root = self._day_root(decision_date)
        decision_payload = self._read_json(day_root / "decision.json")
        target = day_root / "fills.json"
        if target.exists():
            raise FileExistsError(f"fills journal already exists for {day_root.name}")
        if (day_root / "execution_pending.json").exists() and not _allow_pending:
            raise RuntimeError(
                "pending execution must be terminally reconciled before final Fill write"
            )
        fill_tuple = tuple(fills)
        memory = _memory_for_fills(decision_payload, fill_tuple, self.action_schema)
        _atomic_json(
            target,
            {
                "journal_version": JOURNAL_VERSION,
                "decision_date": day_root.name,
                "fills": [_fill_payload(fill) for fill in fill_tuple],
                "policy_memory_after": _memory_payload(
                    memory,
                    self.action_schema,
                ),
                "reconciled_from_pending": bool(_allow_pending),
            },
        )
        return memory

    def record_pending_execution(
        self,
        decision_date: str | date,
        *,
        accepted_orders: Iterable[AcceptedBrokerOrder],
        known_fills: Iterable[Fill],
        error: str,
    ) -> None:
        day_root = self._day_root(decision_date)
        if not (day_root / "decision.json").exists():
            raise FileNotFoundError("pending execution has no recorded decision")
        if (day_root / "fills.json").exists():
            raise RuntimeError("a finalized Fill record cannot become pending")
        target = day_root / "execution_pending.json"
        if target.exists():
            raise FileExistsError(
                f"pending execution already exists for {day_root.name}"
            )
        decision_payload = self._read_json(day_root / "decision.json")
        plan = _order_plan_from_payload(decision_payload, self.action_schema)
        accepted = validate_accepted_orders(plan, tuple(accepted_orders))
        fills = _validate_fills_within_accepted_orders(known_fills, accepted)
        _atomic_json(
            target,
            {
                "journal_version": JOURNAL_VERSION,
                "decision_date": day_root.name,
                "state": "broker_terminal_unconfirmed",
                "accepted_orders": [
                    _accepted_order_payload(order) for order in accepted
                ],
                "known_fills_are_provisional": True,
                "known_fills": [_fill_payload(fill) for fill in fills],
                "error": str(error),
            },
        )

    def load_pending_execution(
        self,
        decision_date: str | date,
    ) -> Mapping[str, object]:
        payload = self._read_json(
            self._day_root(decision_date) / "execution_pending.json"
        )
        if payload.get("journal_version") != JOURNAL_VERSION:
            raise ValueError("unsupported pending execution journal version")
        raw_orders = payload.get("accepted_orders")
        if not isinstance(raw_orders, list):
            raise ValueError("pending execution has no accepted-order mapping")
        accepted = tuple(
            _accepted_order_from_payload(item) for item in raw_orders
        )
        plan = self.load_order_plan_for_reconciliation(decision_date)
        validate_accepted_orders(plan, accepted)
        return {**payload, "accepted_orders": accepted}

    def reconcile_pending(
        self,
        decision_date: str | date,
        fills: Iterable[Fill],
    ) -> PolicyMemory:
        day_root = self._day_root(decision_date)
        if not (day_root / "execution_pending.json").exists():
            raise FileNotFoundError("decision has no pending execution to reconcile")
        pending = self.load_pending_execution(decision_date)
        final_fills = _validate_fills_within_accepted_orders(
            fills,
            pending["accepted_orders"],
        )
        return self.record_fills(
            decision_date,
            final_fills,
            _allow_pending=True,
        )

    def load_order_plan_for_reconciliation(
        self,
        decision_date: str | date,
    ) -> OrderPlan:
        payload = self._read_json(
            self._day_root(decision_date) / "decision.json"
        )
        return _order_plan_from_payload(payload, self.action_schema)

    def load_prior_state(
        self,
        decision_date: str | date,
        *,
        initialize_new_chain: bool = False,
    ) -> tuple[AccountState | None, PolicyMemory]:
        _, account, memory = self.load_prior_state_with_date(
            decision_date,
            initialize_new_chain=initialize_new_chain,
        )
        return account, memory

    def load_prior_state_with_date(
        self,
        decision_date: str | date,
        *,
        initialize_new_chain: bool = False,
    ) -> tuple[str | None, AccountState | None, PolicyMemory]:
        """Restore the latest settled state and expose its decision date."""

        current_root = self._day_root(decision_date)
        prior = [
            item
            for item in self._decision_roots()
            if item.name < current_root.name
        ]
        if not prior:
            journal_is_empty = (
                not self.root.exists()
                or not any(self.root.iterdir())
            )
            if initialize_new_chain and journal_is_empty:
                return None, None, PolicyMemory()
            raise FileNotFoundError(
                "live policy memory is missing; only an explicitly empty new "
                "journal may cold-start"
            )
        latest = prior[-1]
        decision_payload, _, memory = self._verified_history_through(latest)
        return (
            latest.name,
            _account_from_payload(decision_payload["account_before"]),
            memory,
        )

    def _verified_history_through(
        self, last_root: Path,
    ) -> tuple[Mapping[str, object], tuple[Fill, ...], PolicyMemory]:
        """Verify the serial recorded Fill chain, never infer missing history."""
        previous = PolicyMemory()
        for day_root in self._decision_roots():
            if day_root.name > last_root.name:
                break
            decision = self._read_json(day_root / "decision.json")
            if decision["journal_version"] != JOURNAL_VERSION:
                raise ValueError("unsupported decision journal version")
            if decision["decision_date"] != day_root.name:
                raise ValueError("journal decision date differs from its directory")
            before = _memory_from_payload(decision["policy_memory_before"], self.action_schema)
            if _memory_payload(before, self.action_schema) != _memory_payload(previous, self.action_schema):
                raise ValueError("journal policy history is not a complete connected account chain")
            fills_path = day_root / "fills.json"
            if not fills_path.exists():
                raise FileNotFoundError(f"previous decision {day_root.name} has no complete Fill settlement")
            payload = self._read_json(fills_path)
            if payload["journal_version"] != JOURNAL_VERSION or payload["decision_date"] != day_root.name:
                raise ValueError("journal Fill version or decision date mismatch")
            fills = tuple(_fill_from_payload(item) for item in payload["fills"])
            expected = _memory_for_fills(decision, fills, self.action_schema)
            stored = _memory_from_payload(payload["policy_memory_after"], self.action_schema)
            if _memory_payload(stored, self.action_schema) != _memory_payload(expected, self.action_schema):
                raise ValueError("journal policy history does not match its actual recorded Fill settlements")
            previous = expected
            if day_root == last_root:
                return decision, fills, expected
        raise FileNotFoundError(f"decision journal missing for {last_root.name}")

    def replay(self, decision_date: str | date) -> JournalReplay:
        day_root = self._day_root(decision_date)
        fills_path = day_root / "fills.json"
        pending_path = day_root / "execution_pending.json"
        if pending_path.exists() and not fills_path.exists():
            raise RuntimeError(
                "broker execution is pending terminal reconciliation; replay is incomplete"
            )
        if not fills_path.exists():
            raise FileNotFoundError("decision journal has no final broker Fill record")
        decision_payload, fills, memory_after = self._verified_history_through(day_root)
        observation_path = day_root / str(decision_payload["observation_file"])
        with np.load(observation_path, allow_pickle=False) as archive:
            observation = Observation(
                stock_panel=archive["stock_panel"],
                position_panel=archive["position_panel"],
                portfolio=archive["portfolio"],
                policy_history=archive["policy_history"],
                time_mask=archive["time_mask"],
                pit_universe_mask=archive["pit_universe_mask"],
                schema_version=str(decision_payload["observation_schema"]),
                decision_date=str(decision_payload["decision_date"]),
            )
        digest = hashlib.sha256()
        for values in (
            observation.stock_panel,
            observation.position_panel,
            observation.portfolio,
            observation.policy_history,
            observation.time_mask,
            observation.pit_universe_mask,
        ):
            digest.update(np.ascontiguousarray(values).tobytes())
        if digest.hexdigest() != str(decision_payload["observation_sha256"]):
            raise ValueError("journal observation hash mismatch")
        order_plan = _order_plan_from_payload(
            decision_payload,
            self.action_schema,
        )
        return JournalReplay(
            observation=observation,
            action=np.asarray(
                decision_payload["canonical_action"], dtype=np.float32
            ),
            order_plan=order_plan,
            account_before=_account_from_payload(
                decision_payload["account_before"]
            ),
            policy_memory_before=_memory_from_payload(
                decision_payload["policy_memory_before"],
                self.action_schema,
            ),
            fills=fills,
            policy_memory_after=memory_after,
            snapshot_identity=dict(decision_payload["snapshot_identity"]),
            policy_identity=dict(decision_payload["policy_identity"]),
        )

    def _decision_roots(self) -> list[Path]:
        if not self.root.exists():
            return []
        result: list[Path] = []
        for item in self.root.iterdir():
            if not item.is_dir() or not (item / "decision.json").exists():
                continue
            try:
                date.fromisoformat(item.name)
            except ValueError:
                continue
            result.append(item)
        return sorted(result, key=lambda item: item.name)

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"journal file {path.name} must contain an object")
        return payload


__all__ = ["DecisionJournal", "JOURNAL_VERSION", "JournalReplay"]
