"""Causal construction of the model input available at the T open.

The builder consumes already-loaded immutable runtime and factor caches.  It
does not perform I/O, factor calculation, filtering, or strategy decisions.
Static market data can be built once per decision date while account features
remain dynamic for each environment state.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from env.action_schema import CORE_FACTOR_NAMES
from env.contracts import AccountState, Observation


DEFAULT_LOOKBACK = 64
OBSERVATION_SCHEMA_VERSION = "wbr-observation-v1"

# The order is model structure.  Changing it requires a schema bump and a new
# model.  T-open fields are direct; post-open fields and conservatively dated
# total_share are explicitly shifted by one trading row.
CURRENT_RUNTIME_FIELDS = (
    "open",
    "preClose",
    "bps",
    "eps",
    "roe",
    "profit_yoy",
    "revenue_yoy",
    "operating_cf_ps",
    "gross_margin",
    "st_mask",
)
LAGGED_RUNTIME_FIELDS = (
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "total_share",
)
STATIC_RUNTIME_FIELDS = ("issue_price",)
OPTIONAL_CURRENT_RUNTIME_FIELDS = ("star_st_mask",)

STOCK_FEATURE_NAMES = (
    "open",
    "high_lag1",
    "low_lag1",
    "close_lag1",
    "volume_lag1",
    "amount_lag1",
    "preClose",
    "issue_price",
    "total_share_lag1",
    "bps",
    "eps",
    "roe",
    "profit_yoy",
    "revenue_yoy",
    "operating_cf_ps",
    "gross_margin",
    "st_mask",
) + tuple(f"factor_rank.{name}" for name in CORE_FACTOR_NAMES)

MARKET_STAT_NAMES = ("mean", "std", "coverage")


def _market_feature_names(stock_feature_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"{feature}__xsec_{stat}"
        for feature in stock_feature_names
        for stat in MARKET_STAT_NAMES
    )


MARKET_FEATURE_NAMES = _market_feature_names(STOCK_FEATURE_NAMES)
POSITION_FEATURE_NAMES = (
    "held",
    "quantity_share",
    "position_weight",
    "cost_return",
    "sellable_ratio",
    "last_mark_return",
)
PORTFOLIO_FEATURE_NAMES = (
    "cash",
    "nav",
    "peak_nav",
    "cash_ratio",
    "exposure",
    "drawdown",
)


class RuntimeLike(Protocol):
    """Structural subset of :class:`offline_data.RuntimeSlice` used here."""

    stock_codes: tuple[str, ...]
    trade_dates: NDArray[np.datetime64]
    data: Mapping[str, NDArray[np.generic]]
    manifest: "RuntimeManifestLike"


class RuntimeManifestLike(Protocol):
    schema_hash: str


class FactorBatchLike(Protocol):
    """Structural subset of :class:`factor.FactorBatch` used here."""

    factor_names: tuple[str, ...]
    stock_codes: tuple[str, ...]
    trade_dates: NDArray[np.datetime64]
    ranks: NDArray[np.float32]
    validity: NDArray[np.bool_]
    schema_hash: str
    runtime_schema_hash: str
    rank_universe_sha256: str


def _canonical_hash(payload: Mapping[str, object]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _ordered_string_hash(values: tuple[str, ...]) -> str:
    return sha256("\0".join(values).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ObservationSchema:
    """Versioned, ordered structure of one raw :class:`Observation`."""

    lookback: int
    stock_count: int
    stock_codes_hash: str
    runtime_schema_hash: str
    factor_schema_hash: str
    rank_universe_sha256: str
    stock_feature_names: tuple[str, ...] = STOCK_FEATURE_NAMES
    market_feature_names: tuple[str, ...] = MARKET_FEATURE_NAMES
    position_feature_names: tuple[str, ...] = POSITION_FEATURE_NAMES
    portfolio_feature_names: tuple[str, ...] = PORTFOLIO_FEATURE_NAMES
    factor_names: tuple[str, ...] = CORE_FACTOR_NAMES
    version: str = OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.lookback <= 0 or self.stock_count <= 0:
            raise ValueError("lookback and stock_count must be positive")
        for label, names in (
            ("stock_feature_names", self.stock_feature_names),
            ("market_feature_names", self.market_feature_names),
            ("position_feature_names", self.position_feature_names),
            ("portfolio_feature_names", self.portfolio_feature_names),
            ("factor_names", self.factor_names),
        ):
            if not names or len(set(names)) != len(names):
                raise ValueError(f"{label} must be non-empty and unique")
        for label, digest in (
            ("stock_codes_hash", self.stock_codes_hash),
            ("runtime_schema_hash", self.runtime_schema_hash),
            ("factor_schema_hash", self.factor_schema_hash),
            ("rank_universe_sha256", self.rank_universe_sha256),
        ):
            if len(digest) != 64:
                raise ValueError(f"{label} must be a SHA-256 hex digest")
        if not self.version:
            raise ValueError("version must not be empty")

    @property
    def stock_feature_count(self) -> int:
        return len(self.stock_feature_names)

    @property
    def market_feature_count(self) -> int:
        return len(self.market_feature_names)

    @property
    def position_feature_count(self) -> int:
        return len(self.position_feature_names)

    @property
    def portfolio_feature_count(self) -> int:
        return len(self.portfolio_feature_names)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "lookback": self.lookback,
            "stock_count": self.stock_count,
            "stock_codes_hash": self.stock_codes_hash,
            "runtime_schema_hash": self.runtime_schema_hash,
            "factor_schema_hash": self.factor_schema_hash,
            "rank_universe_sha256": self.rank_universe_sha256,
            "stock_feature_names": list(self.stock_feature_names),
            "market_feature_names": list(self.market_feature_names),
            "position_feature_names": list(self.position_feature_names),
            "portfolio_feature_names": list(self.portfolio_feature_names),
            "factor_names": list(self.factor_names),
        }

    @property
    def schema_hash(self) -> str:
        return _canonical_hash(self._hash_payload())

    @property
    def identifier(self) -> str:
        return f"{self.version}:{self.schema_hash}"

    def to_dict(self) -> dict[str, object]:
        result = self._hash_payload()
        result["schema_hash"] = self.schema_hash
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ObservationSchema":
        schema = cls(
            version=str(payload["version"]),
            lookback=int(payload["lookback"]),
            stock_count=int(payload["stock_count"]),
            stock_codes_hash=str(payload["stock_codes_hash"]),
            runtime_schema_hash=str(payload["runtime_schema_hash"]),
            factor_schema_hash=str(payload["factor_schema_hash"]),
            rank_universe_sha256=str(payload["rank_universe_sha256"]),
            stock_feature_names=tuple(str(v) for v in payload["stock_feature_names"]),
            market_feature_names=tuple(str(v) for v in payload["market_feature_names"]),
            position_feature_names=tuple(str(v) for v in payload["position_feature_names"]),
            portfolio_feature_names=tuple(str(v) for v in payload["portfolio_feature_names"]),
            factor_names=tuple(str(v) for v in payload["factor_names"]),
        )
        if str(payload["schema_hash"]) != schema.schema_hash:
            raise ValueError("observation schema hash mismatch")
        return schema


@dataclass(frozen=True)
class StaticObservation:
    """Account-independent part of an observation for one decision date."""

    decision_index: int
    decision_date: str
    stock_panel: NDArray[np.float32]
    market_panel: NDArray[np.float32]
    feature_mask: NDArray[np.bool_]
    stock_mask: NDArray[np.bool_]
    time_mask: NDArray[np.bool_]
    schema_version: str

    def __post_init__(self) -> None:
        if self.stock_panel.ndim != 3:
            raise ValueError("stock_panel must have shape [L, N, F]")
        if self.market_panel.ndim != 2 or self.market_panel.shape[0] != self.stock_panel.shape[0]:
            raise ValueError("market_panel must have shape [L, M]")
        if self.feature_mask.shape != self.stock_panel.shape:
            raise ValueError("feature_mask must match stock_panel")
        if self.stock_mask.shape != self.stock_panel.shape[:2]:
            raise ValueError("stock_mask must have shape [L, N]")
        if self.time_mask.shape != self.stock_panel.shape[:1]:
            raise ValueError("time_mask must have shape [L]")
        if not np.isfinite(self.stock_panel).all() or not np.isfinite(self.market_panel).all():
            raise ValueError("static observation values must be finite")


@dataclass(frozen=True)
class StaticObservationRows:
    """Consecutive causal market rows used for one-time cache generation."""

    row_start: int
    decision_dates: tuple[str, ...]
    stock_panel: NDArray[np.float32]
    market_panel: NDArray[np.float32]
    feature_mask: NDArray[np.bool_]
    stock_mask: NDArray[np.bool_]
    schema_version: str

    def __post_init__(self) -> None:
        row_count = len(self.decision_dates)
        if row_count == 0 or self.row_start < 0:
            raise ValueError("static rows must be non-empty and start at a valid index")
        if self.stock_panel.ndim != 3 or self.stock_panel.shape[0] != row_count:
            raise ValueError("stock_panel must have shape [D, N, F]")
        if self.market_panel.ndim != 2 or self.market_panel.shape[0] != row_count:
            raise ValueError("market_panel must have shape [D, M]")
        if self.feature_mask.shape != self.stock_panel.shape:
            raise ValueError("feature_mask must match stock_panel")
        if self.stock_mask.shape != self.stock_panel.shape[:2]:
            raise ValueError("stock_mask must have shape [D, N]")
        if not np.isfinite(self.stock_panel).all() or not np.isfinite(self.market_panel).all():
            raise ValueError("static row values must be finite")


@dataclass(frozen=True)
class AccountObservation:
    """Dynamic account part paired with one decision date."""

    decision_index: int
    decision_date: str
    position_panel: NDArray[np.float32]
    portfolio: NDArray[np.float32]
    current_stock_mask: NDArray[np.bool_]
    current_factor_ranks: NDArray[np.float32]
    current_factor_validity: NDArray[np.bool_]
    schema_version: str

    def __post_init__(self) -> None:
        if self.position_panel.ndim != 2:
            raise ValueError("position_panel must have shape [N, H]")
        if self.portfolio.ndim != 1:
            raise ValueError("portfolio must have shape [P]")
        if self.current_stock_mask.shape != self.position_panel.shape[:1]:
            raise ValueError("current_stock_mask must have shape [N]")
        if self.current_factor_ranks.ndim != 2:
            raise ValueError("current_factor_ranks must have shape [K, N]")
        if self.current_factor_ranks.shape[1] != self.position_panel.shape[0]:
            raise ValueError("current_factor_ranks stock axis must match position_panel")
        if self.current_factor_validity.shape != self.current_factor_ranks.shape:
            raise ValueError("current_factor_validity must match current_factor_ranks")
        if self.current_factor_validity.dtype != np.bool_:
            raise ValueError("current_factor_validity must be bool")
        if (
            not np.isfinite(self.position_panel).all()
            or not np.isfinite(self.portfolio).all()
            or not np.isfinite(self.current_factor_ranks).all()
        ):
            raise ValueError("account observation values must be finite")


class ObservationBuilder:
    """Build causal observations from in-memory runtime and factor caches."""

    def __init__(
        self,
        runtime: RuntimeLike,
        factors: FactorBatchLike,
        *,
        lookback: int = DEFAULT_LOOKBACK,
    ) -> None:
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        self._data = runtime.data
        self._stock_codes = tuple(str(code) for code in runtime.stock_codes)
        self._trade_dates = np.asarray(runtime.trade_dates).astype("datetime64[D]", copy=False)
        self._factor_names = tuple(str(name) for name in factors.factor_names)
        self._factor_ranks = np.asarray(factors.ranks)
        self._factor_validity = np.asarray(factors.validity, dtype=np.bool_)
        self._runtime_schema_hash = str(runtime.manifest.schema_hash)
        self._factor_schema_hash = str(factors.schema_hash)
        self.lookback = int(lookback)
        self._validate_inputs(factors)
        self._stock_to_index = {code: index for index, code in enumerate(self._stock_codes)}
        stock_feature_names = STOCK_FEATURE_NAMES
        if "star_st_mask" in self._data:
            factor_start = stock_feature_names.index(f"factor_rank.{CORE_FACTOR_NAMES[0]}")
            stock_feature_names = (
                stock_feature_names[:factor_start]
                + ("star_st_mask",)
                + stock_feature_names[factor_start:]
            )
        market_feature_names = _market_feature_names(stock_feature_names)
        self.schema = ObservationSchema(
            lookback=self.lookback,
            stock_count=len(self._stock_codes),
            stock_codes_hash=_ordered_string_hash(self._stock_codes),
            runtime_schema_hash=self._runtime_schema_hash,
            factor_schema_hash=self._factor_schema_hash,
            rank_universe_sha256=str(factors.rank_universe_sha256),
            stock_feature_names=stock_feature_names,
            market_feature_names=market_feature_names,
            factor_names=self._factor_names,
        )

    @property
    def stock_codes(self) -> tuple[str, ...]:
        return self._stock_codes

    @property
    def trade_dates(self) -> NDArray[np.datetime64]:
        return self._trade_dates

    def _validate_inputs(self, factors: FactorBatchLike) -> None:
        if not self._stock_codes or len(set(self._stock_codes)) != len(self._stock_codes):
            raise ValueError("runtime stock_codes must be non-empty and unique")
        if self._trade_dates.ndim != 1 or not len(self._trade_dates):
            raise ValueError("runtime trade_dates must be a non-empty vector")
        if np.any(self._trade_dates[1:] <= self._trade_dates[:-1]):
            raise ValueError("runtime trade_dates must be strictly increasing")

        factor_codes = tuple(str(code) for code in factors.stock_codes)
        factor_dates = np.asarray(factors.trade_dates).astype("datetime64[D]", copy=False)
        if factor_codes != self._stock_codes:
            raise ValueError("factor stock order does not match runtime stock order")
        if not np.array_equal(factor_dates, self._trade_dates):
            raise ValueError("factor dates do not match runtime dates")
        if str(factors.runtime_schema_hash) != self._runtime_schema_hash:
            raise ValueError("factor cache was built from a different runtime schema")
        if self._factor_names != tuple(CORE_FACTOR_NAMES):
            raise ValueError(
                f"factor order must be {tuple(CORE_FACTOR_NAMES)!r}, got {self._factor_names!r}"
            )

        date_count = len(self._trade_dates)
        stock_count = len(self._stock_codes)
        expected_factor_shape = (date_count, len(self._factor_names), stock_count)
        if self._factor_ranks.shape != expected_factor_shape:
            raise ValueError(f"factor ranks must have shape {expected_factor_shape}")
        if self._factor_validity.shape != expected_factor_shape:
            raise ValueError(f"factor validity must have shape {expected_factor_shape}")

        required = set(CURRENT_RUNTIME_FIELDS + LAGGED_RUNTIME_FIELDS + STATIC_RUNTIME_FIELDS)
        missing = required.difference(self._data)
        if missing:
            raise ValueError(f"runtime is missing registered fields: {sorted(missing)}")
        for field in CURRENT_RUNTIME_FIELDS + LAGGED_RUNTIME_FIELDS:
            if np.asarray(self._data[field]).shape != (date_count, stock_count):
                raise ValueError(f"runtime field {field!r} must have shape {(date_count, stock_count)}")
        for field in OPTIONAL_CURRENT_RUNTIME_FIELDS:
            if field in self._data and np.asarray(self._data[field]).shape != (
                date_count,
                stock_count,
            ):
                raise ValueError(f"runtime field {field!r} must have shape {(date_count, stock_count)}")
        if np.asarray(self._data["issue_price"]).shape != (stock_count,):
            raise ValueError(f"runtime field 'issue_price' must have shape {(stock_count,)}")

        registered = required.union(OPTIONAL_CURRENT_RUNTIME_FIELDS, {"stock_names"})
        for name, raw in self._data.items():
            values = np.asarray(raw)
            is_aligned_numeric = (
                values.shape in ((date_count, stock_count), (stock_count,))
                and (np.issubdtype(values.dtype, np.number) or values.dtype == np.bool_)
            )
            if is_aligned_numeric and name not in registered:
                raise ValueError(
                    f"unregistered runtime field {name!r}; upgrade ObservationSchema before use"
                )

    def _validate_decision_index(self, decision_index: int) -> int:
        if type(decision_index) is not int:
            raise TypeError("decision_index must be int")
        if not 0 <= decision_index < len(self._trade_dates):
            raise IndexError("decision_index outside runtime date range")
        return decision_index

    @staticmethod
    def _valid_float32(values: NDArray[np.generic]) -> NDArray[np.bool_]:
        numeric = np.asarray(values)
        limit = np.finfo(np.float32).max
        return np.isfinite(numeric) & (np.abs(numeric) <= limit)

    @staticmethod
    def _assign_feature(
        panel: NDArray[np.float32],
        feature_mask: NDArray[np.bool_],
        destination_rows: NDArray[np.int64],
        feature_index: int,
        values: NDArray[np.generic],
        stock_mask: NDArray[np.bool_],
    ) -> None:
        valid = ObservationBuilder._valid_float32(values) & stock_mask[destination_rows]
        numeric = np.zeros(valid.shape, dtype=np.float32)
        numeric[valid] = np.asarray(values)[valid].astype(np.float32, copy=False)
        panel[destination_rows, :, feature_index] = numeric
        feature_mask[destination_rows, :, feature_index] = valid

    def _build_aligned_rows(
        self,
        source_rows: NDArray[np.int64],
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
    ]:
        """Build non-padded causal rows; each runtime row is processed once."""

        if source_rows.ndim != 1 or len(source_rows) == 0:
            raise ValueError("source_rows must be a non-empty vector")
        stock_count = len(self._stock_codes)
        feature_count = self.schema.stock_feature_count
        row_count = len(source_rows)
        panel = np.zeros((row_count, stock_count, feature_count), dtype=np.float32)
        feature_mask = np.zeros_like(panel, dtype=np.bool_)
        stock_mask = np.zeros((row_count, stock_count), dtype=np.bool_)
        destination_rows = np.arange(row_count, dtype=np.int64)

        opens = np.asarray(self._data["open"])[source_rows]
        # This is an availability/tradability mask only.  ST and price filters
        # remain separate dynamic strategy controls and never enter this mask.
        stock_mask[destination_rows] = self._valid_float32(opens) & (opens > 0)

        feature_indices = {
            name: index for index, name in enumerate(self.schema.stock_feature_names)
        }
        self._assign_feature(
            panel,
            feature_mask,
            destination_rows,
            feature_indices["open"],
            opens,
            stock_mask,
        )

        direct_fields = tuple(
            field for field in CURRENT_RUNTIME_FIELDS if field != "open"
        ) + tuple(
            field for field in OPTIONAL_CURRENT_RUNTIME_FIELDS if field in self._data
        )
        for field in direct_fields:
            self._assign_feature(
                panel,
                feature_mask,
                destination_rows,
                feature_indices[field],
                np.asarray(self._data[field])[source_rows],
                stock_mask,
            )

        lag_destination = destination_rows[source_rows > 0]
        lag_sources = source_rows[source_rows > 0] - 1
        for field in LAGGED_RUNTIME_FIELDS:
            self._assign_feature(
                panel,
                feature_mask,
                lag_destination,
                feature_indices[f"{field}_lag1"],
                np.asarray(self._data[field])[lag_sources],
                stock_mask,
            )

        issue_price = np.broadcast_to(
            np.asarray(self._data["issue_price"]), (len(destination_rows), stock_count)
        )
        valid_issue = self._valid_float32(issue_price) & (issue_price > 0)
        issue_values = np.where(valid_issue, issue_price, np.nan)
        self._assign_feature(
            panel,
            feature_mask,
            destination_rows,
            feature_indices["issue_price"],
            issue_values,
            stock_mask,
        )

        factor_values = self._factor_ranks[source_rows]
        factor_validity = self._factor_validity[source_rows]
        for factor_index, factor_name in enumerate(self._factor_names):
            values = factor_values[:, factor_index, :]
            valid = factor_validity[:, factor_index, :] & self._valid_float32(values)
            masked_values = np.where(valid, values, np.nan)
            self._assign_feature(
                panel,
                feature_mask,
                destination_rows,
                feature_indices[f"factor_rank.{factor_name}"],
                masked_values,
                stock_mask,
            )

        market_panel = self._build_market_panel(panel, feature_mask, stock_mask)
        return panel, market_panel, feature_mask, stock_mask

    def build_static_rows(self, row_start: int, row_stop: int) -> StaticObservationRows:
        """Build consecutive market rows for efficient rolling cache creation."""

        if type(row_start) is not int or type(row_stop) is not int:
            raise TypeError("row_start and row_stop must be int")
        if not 0 <= row_start < row_stop <= len(self._trade_dates):
            raise IndexError("static row interval is outside runtime date range")
        source_rows = np.arange(row_start, row_stop, dtype=np.int64)
        panel, market_panel, feature_mask, stock_mask = self._build_aligned_rows(source_rows)
        dates = tuple(
            np.datetime_as_string(value, unit="D") for value in self._trade_dates[source_rows]
        )
        return StaticObservationRows(
            row_start=row_start,
            decision_dates=dates,
            stock_panel=np.ascontiguousarray(panel),
            market_panel=np.ascontiguousarray(market_panel),
            feature_mask=np.ascontiguousarray(feature_mask),
            stock_mask=np.ascontiguousarray(stock_mask),
            schema_version=self.schema.identifier,
        )

    def build_static(self, decision_index: int) -> StaticObservation:
        """Build the account-independent causal window ending at T-open."""

        decision_index = self._validate_decision_index(decision_index)
        source_start = max(0, decision_index - self.lookback + 1)
        aligned = self.build_static_rows(source_start, decision_index + 1)
        row_count = len(aligned.decision_dates)
        destination = slice(self.lookback - row_count, self.lookback)
        panel = np.zeros(
            (self.lookback, self.schema.stock_count, self.schema.stock_feature_count),
            dtype=np.float32,
        )
        market_panel = np.zeros(
            (self.lookback, self.schema.market_feature_count), dtype=np.float32
        )
        feature_mask = np.zeros_like(panel, dtype=np.bool_)
        stock_mask = np.zeros(
            (self.lookback, self.schema.stock_count), dtype=np.bool_
        )
        time_mask = np.zeros(self.lookback, dtype=np.bool_)
        panel[destination] = aligned.stock_panel
        market_panel[destination] = aligned.market_panel
        feature_mask[destination] = aligned.feature_mask
        stock_mask[destination] = aligned.stock_mask
        time_mask[destination] = True
        decision_date = np.datetime_as_string(self._trade_dates[decision_index], unit="D")
        return StaticObservation(
            decision_index=decision_index,
            decision_date=decision_date,
            stock_panel=np.ascontiguousarray(panel),
            market_panel=np.ascontiguousarray(market_panel),
            feature_mask=np.ascontiguousarray(feature_mask),
            stock_mask=np.ascontiguousarray(stock_mask),
            time_mask=np.ascontiguousarray(time_mask),
            schema_version=self.schema.identifier,
        )

    @staticmethod
    def _build_market_panel(
        stock_panel: NDArray[np.float32],
        feature_mask: NDArray[np.bool_],
        stock_mask: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        valid = feature_mask & stock_mask[:, :, None]
        values = stock_panel.astype(np.float64, copy=False)
        counts = valid.sum(axis=1, dtype=np.int64)
        sums = np.where(valid, values, 0.0).sum(axis=1, dtype=np.float64)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        centered = np.where(valid, values - means[:, None, :], 0.0)
        variances = np.divide(
            np.square(centered).sum(axis=1, dtype=np.float64),
            counts,
            out=np.zeros_like(means),
            where=counts > 0,
        )
        active = stock_mask.sum(axis=1, dtype=np.int64)[:, None]
        coverage = np.divide(
            counts,
            active,
            out=np.zeros_like(means),
            where=active > 0,
        )
        stats = np.stack((means, np.sqrt(variances), coverage), axis=2)
        return stats.reshape(stock_panel.shape[0], -1).astype(np.float32)

    def build_account(
        self,
        decision_index: int,
        account: AccountState,
    ) -> AccountObservation:
        """Build only the dynamic account features for one T-open."""

        decision_index = self._validate_decision_index(decision_index)
        cash = float(account.cash)
        if not np.isfinite(cash) or cash < 0:
            raise ValueError("account cash must be finite and non-negative")

        stock_count = len(self._stock_codes)
        position_panel = np.zeros(
            (stock_count, len(POSITION_FEATURE_NAMES)), dtype=np.float32
        )
        current_open = np.asarray(self._data["open"])[decision_index]
        current_stock_mask = self._valid_float32(current_open) & (current_open > 0)

        quantities: dict[int, int] = {}
        for code, quantity in account.positions.items():
            if code not in self._stock_to_index:
                raise ValueError(f"account position {code!r} is outside the model stock vocabulary")
            if type(quantity) is not int or quantity <= 0:
                raise ValueError(f"account position quantity for {code!r} must be a positive int")
            quantities[self._stock_to_index[code]] = quantity

        total_quantity = sum(quantities.values())
        market_value = 0.0
        marks: dict[int, float] = {}
        for stock_index, quantity in quantities.items():
            code = self._stock_codes[stock_index]
            raw_open = float(current_open[stock_index])
            if np.isfinite(raw_open) and raw_open > 0:
                mark = raw_open
            elif code in account.last_prices:
                mark = float(account.last_prices[code])
            elif code in account.average_costs:
                mark = float(account.average_costs[code])
            else:
                raise ValueError(f"held stock {code!r} has no causal mark price")
            if not np.isfinite(mark) or mark <= 0:
                raise ValueError(f"held stock {code!r} mark price must be finite and positive")
            marks[stock_index] = mark
            market_value += mark * quantity

        computed_nav = cash + market_value
        stated_nav = float(account.nav)
        if not np.isfinite(stated_nav) or stated_nav < 0:
            raise ValueError("account nav must be finite and non-negative")
        nav = stated_nav if stated_nav > 0 else computed_nav
        if nav <= 0:
            raise ValueError("account nav must be positive")
        stated_peak = float(account.peak_nav)
        if not np.isfinite(stated_peak) or stated_peak < 0:
            raise ValueError("account peak_nav must be finite and non-negative")
        peak_nav = stated_peak if stated_peak > 0 else nav
        if peak_nav + 1e-7 < nav:
            raise ValueError("account peak_nav must be greater than or equal to nav")

        for stock_index, quantity in quantities.items():
            code = self._stock_codes[stock_index]
            mark = marks[stock_index]
            average_cost = float(account.average_costs[code]) if code in account.average_costs else mark
            last_price = float(account.last_prices[code]) if code in account.last_prices else mark
            sellable = int(account.sellable_positions[code]) if code in account.sellable_positions else 0
            if not np.isfinite(average_cost) or average_cost <= 0:
                raise ValueError(f"average cost for {code!r} must be finite and positive")
            if not np.isfinite(last_price) or last_price <= 0:
                raise ValueError(f"last price for {code!r} must be finite and positive")
            if sellable < 0 or sellable > quantity:
                raise ValueError(f"sellable quantity for {code!r} must be in [0, position]")
            position_panel[stock_index] = (
                1.0,
                quantity / total_quantity,
                mark * quantity / nav,
                mark / average_cost - 1.0,
                sellable / quantity,
                mark / last_price - 1.0,
            )

        exposure = market_value / nav
        portfolio = np.asarray(
            (
                cash,
                nav,
                peak_nav,
                cash / nav,
                exposure,
                nav / peak_nav - 1.0,
            ),
            dtype=np.float32,
        )
        if not np.isfinite(position_panel).all() or not np.isfinite(portfolio).all():
            raise ValueError("account features overflowed float32")
        decision_date = np.datetime_as_string(self._trade_dates[decision_index], unit="D")
        return AccountObservation(
            decision_index=decision_index,
            decision_date=decision_date,
            position_panel=np.ascontiguousarray(position_panel),
            portfolio=np.ascontiguousarray(portfolio),
            current_stock_mask=np.ascontiguousarray(current_stock_mask),
            current_factor_ranks=np.ascontiguousarray(
                self._factor_ranks[decision_index], dtype=np.float32
            ),
            current_factor_validity=np.ascontiguousarray(
                self._factor_validity[decision_index], dtype=np.bool_
            ),
            schema_version=self.schema.identifier,
        )

    def build(
        self,
        decision_index: int,
        account: AccountState,
        *,
        static: StaticObservation | None = None,
    ) -> Observation:
        """Build a complete contract object, optionally reusing static data."""

        decision_index = self._validate_decision_index(decision_index)
        static_part = static if static is not None else self.build_static(decision_index)
        account_part = self.build_account(decision_index, account)
        self._validate_parts(decision_index, static_part, account_part)
        return Observation(
            stock_panel=static_part.stock_panel,
            market_panel=static_part.market_panel,
            position_panel=account_part.position_panel,
            portfolio=account_part.portfolio,
            feature_mask=static_part.feature_mask,
            stock_mask=static_part.stock_mask,
            time_mask=static_part.time_mask,
            schema_version=self.schema.identifier,
            decision_date=static_part.decision_date,
        )

    def _validate_parts(
        self,
        decision_index: int,
        static: StaticObservation,
        account: AccountObservation,
    ) -> None:
        expected_date = np.datetime_as_string(self._trade_dates[decision_index], unit="D")
        for label, part in (("static", static), ("account", account)):
            if part.decision_index != decision_index or part.decision_date != expected_date:
                raise ValueError(f"{label} observation belongs to a different decision date")
            if part.schema_version != self.schema.identifier:
                raise ValueError(f"{label} observation schema mismatch")


__all__ = [
    "AccountObservation",
    "DEFAULT_LOOKBACK",
    "MARKET_FEATURE_NAMES",
    "ObservationBuilder",
    "ObservationSchema",
    "PORTFOLIO_FEATURE_NAMES",
    "POSITION_FEATURE_NAMES",
    "STOCK_FEATURE_NAMES",
    "StaticObservation",
    "StaticObservationRows",
]
