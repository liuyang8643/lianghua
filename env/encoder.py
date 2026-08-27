"""Deterministic, stock-count-independent encoding for PPO.

Every registered stock feature, every stock selected by the causal masks, and
every row of the lookback window contributes through vectorised masked moments.
Factor-to-feature co-moments preserve stock-level style relationships, while
dynamic factor exposures tie current holdings back to the market panel.  The
encoder has no learned parameters.  Training and live inference therefore
share exactly the same transform, while the PPO rollout buffer stores a small
fixed vector instead of the raw ``[L, N, F]`` panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from env.contracts import Observation
from env.observation import (
    AccountObservation,
    ObservationBuilder,
    ObservationSchema,
    StaticObservation,
)


ENCODER_SCHEMA_VERSION = "wbr-deterministic-encoder-v2"
NORMALIZER_VERSION = "wbr-train-normalizer-v1"
CROSS_SECTION_STATS = (
    "mean",
    "std",
    "minimum",
    "maximum",
    "rms",
    "coverage",
    "nonzero_fraction",
)
TEMPORAL_STATS = (
    "last",
    "mean",
    "std",
    "minimum",
    "maximum",
    "rms",
    "slope",
    "delta",
    "coverage",
)
JOINT_STATS = ("product_mean", "overlap")
FACTOR_EXPOSURE_STATS = (
    "held_mean",
    "position_weighted_mean",
    "position_weighted_std",
    "sellable_weighted_mean",
    "position_minus_market",
    "position_valid_weight_fraction",
)
FACTOR_PAIR_EXPOSURE_STATS = (
    "position_weighted_product_mean",
    "position_valid_weight_fraction",
)


def _canonical_hash(payload: Mapping[str, object]) -> str:
    from hashlib import sha256

    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _as_finite_float32(values: NDArray[np.floating]) -> NDArray[np.float32]:
    limit = np.finfo(np.float32).max
    clipped = np.clip(np.asarray(values, dtype=np.float64), -limit, limit)
    result = clipped.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("encoded values must be finite")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class EncodedObservationSchema:
    """Ordered feature vocabulary of the deterministic encoder output."""

    source_observation_schema: str
    feature_names: tuple[str, ...]
    market_dimension: int
    version: str = ENCODER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.source_observation_schema:
            raise ValueError("source_observation_schema must not be empty")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("encoded feature_names must be non-empty and unique")
        if not 0 < self.market_dimension < len(self.feature_names):
            raise ValueError("market_dimension must split market and account features")
        if not self.version:
            raise ValueError("version must not be empty")

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    @property
    def account_dimension(self) -> int:
        return self.dimension - self.market_dimension

    def _hash_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source_observation_schema": self.source_observation_schema,
            "feature_names": list(self.feature_names),
            "dimension": self.dimension,
            "market_dimension": self.market_dimension,
            "account_dimension": self.account_dimension,
        }

    @property
    def schema_hash(self) -> str:
        return _canonical_hash(self._hash_payload())

    @property
    def identifier(self) -> str:
        return f"{self.version}:{self.schema_hash}"

    def to_dict(self) -> dict[str, object]:
        payload = self._hash_payload()
        payload["schema_hash"] = self.schema_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EncodedObservationSchema":
        schema = cls(
            version=str(payload["version"]),
            source_observation_schema=str(payload["source_observation_schema"]),
            feature_names=tuple(str(v) for v in payload["feature_names"]),
            market_dimension=int(payload["market_dimension"]),
        )
        if int(payload["dimension"]) != schema.dimension:
            raise ValueError("encoded schema dimension mismatch")
        if int(payload["account_dimension"]) != schema.account_dimension:
            raise ValueError("encoded schema account dimension mismatch")
        if str(payload["schema_hash"]) != schema.schema_hash:
            raise ValueError("encoded schema hash mismatch")
        return schema


def _masked_cross_section(
    values: NDArray[np.float32],
    valid: NDArray[np.bool_],
    *,
    item_axis: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Pool one item axis without ever iterating over stocks."""

    numeric = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=np.bool_)
    if numeric.shape != mask.shape:
        raise ValueError("cross-section values and mask shapes differ")
    counts = mask.sum(axis=item_axis, dtype=np.int64)
    sums = np.where(mask, numeric, 0.0).sum(axis=item_axis, dtype=np.float64)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    expanded_means = np.expand_dims(means, axis=item_axis)
    centered = np.where(mask, numeric - expanded_means, 0.0)
    variances = np.divide(
        np.square(centered).sum(axis=item_axis, dtype=np.float64),
        counts,
        out=np.zeros_like(means),
        where=counts > 0,
    )
    squares = np.where(mask, np.square(numeric), 0.0).sum(axis=item_axis, dtype=np.float64)
    rms = np.sqrt(np.divide(squares, counts, out=np.zeros_like(means), where=counts > 0))
    minima = np.where(mask, numeric, np.inf).min(axis=item_axis)
    maxima = np.where(mask, numeric, -np.inf).max(axis=item_axis)
    minima = np.where(counts > 0, minima, 0.0)
    maxima = np.where(counts > 0, maxima, 0.0)
    item_count = numeric.shape[item_axis]
    coverage = counts / float(item_count)
    nonzero = (mask & (numeric != 0.0)).sum(axis=item_axis, dtype=np.int64)
    nonzero_fraction = np.divide(
        nonzero,
        counts,
        out=np.zeros_like(means),
        where=counts > 0,
    )
    pooled = np.stack(
        (
            means,
            np.sqrt(variances),
            minima,
            maxima,
            rms,
            coverage,
            nonzero_fraction,
        ),
        axis=-1,
    )
    return pooled, counts


def _masked_temporal(
    values: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Summarise every time row of each sequence using masked moments."""

    numeric = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=np.bool_)
    if numeric.ndim != 2 or numeric.shape != mask.shape:
        raise ValueError("temporal values and mask must have identical [L, K] shape")
    length, feature_count = numeric.shape
    counts = mask.sum(axis=0, dtype=np.int64)
    sums = np.where(mask, numeric, 0.0).sum(axis=0, dtype=np.float64)
    means = np.divide(sums, counts, out=np.zeros(feature_count), where=counts > 0)
    centered = np.where(mask, numeric - means[None, :], 0.0)
    variances = np.divide(
        np.square(centered).sum(axis=0, dtype=np.float64),
        counts,
        out=np.zeros(feature_count),
        where=counts > 0,
    )
    rms = np.sqrt(
        np.divide(
            np.where(mask, np.square(numeric), 0.0).sum(axis=0, dtype=np.float64),
            counts,
            out=np.zeros(feature_count),
            where=counts > 0,
        )
    )
    minima = np.where(mask, numeric, np.inf).min(axis=0)
    maxima = np.where(mask, numeric, -np.inf).max(axis=0)
    minima = np.where(counts > 0, minima, 0.0)
    maxima = np.where(counts > 0, maxima, 0.0)

    row_index = np.arange(length, dtype=np.int64)[:, None]
    last_index = np.where(mask, row_index, -1).max(axis=0)
    first_index = np.where(mask, row_index, length).min(axis=0)
    columns = np.arange(feature_count)
    last = numeric[np.maximum(last_index, 0), columns]
    first = numeric[np.minimum(first_index, length - 1), columns]
    last = np.where(counts > 0, last, 0.0)
    first = np.where(counts > 0, first, 0.0)

    x = np.linspace(-1.0, 1.0, length, dtype=np.float64)[:, None]
    x_mean = np.divide(
        np.where(mask, x, 0.0).sum(axis=0, dtype=np.float64),
        counts,
        out=np.zeros(feature_count),
        where=counts > 0,
    )
    x_centered = np.where(mask, x - x_mean[None, :], 0.0)
    x_variance_sum = np.square(x_centered).sum(axis=0, dtype=np.float64)
    covariance_sum = (x_centered * centered).sum(axis=0, dtype=np.float64)
    slope = np.divide(
        covariance_sum,
        x_variance_sum,
        out=np.zeros(feature_count),
        where=x_variance_sum > 0,
    )
    return np.stack(
        (
            last,
            means,
            np.sqrt(variances),
            minima,
            maxima,
            rms,
            slope,
            last - first,
            counts / float(length),
        ),
        axis=1,
    )


class ObservationEncoder:
    """Encode raw observations with a deterministic, fixed-size transform."""

    def __init__(self, observation_schema: ObservationSchema) -> None:
        self.observation_schema = observation_schema
        factor_feature_names = tuple(
            f"factor_rank.{name}" for name in observation_schema.factor_names
        )
        missing_factor_features = set(factor_feature_names).difference(
            observation_schema.stock_feature_names
        )
        if missing_factor_features:
            raise ValueError(
                f"stock schema is missing factor features: {sorted(missing_factor_features)}"
            )
        self._factor_feature_indices = tuple(
            observation_schema.stock_feature_names.index(name)
            for name in factor_feature_names
        )
        non_factor_indices = tuple(
            index
            for index, name in enumerate(observation_schema.stock_feature_names)
            if name not in factor_feature_names
        )
        joint_pairs: list[tuple[int, int, str]] = []
        for factor_position, factor_name in enumerate(observation_schema.factor_names):
            for feature_index in non_factor_indices:
                feature_name = observation_schema.stock_feature_names[feature_index]
                joint_pairs.append(
                    (
                        factor_position,
                        feature_index,
                        f"factor_joint.{factor_name}__with__{feature_name}",
                    )
                )
        for left in range(len(observation_schema.factor_names)):
            for right in range(left + 1, len(observation_schema.factor_names)):
                joint_pairs.append(
                    (
                        left,
                        self._factor_feature_indices[right],
                        "factor_joint."
                        f"{observation_schema.factor_names[left]}__with__"
                        f"{observation_schema.factor_names[right]}",
                    )
                )
        self._joint_factor_positions = np.asarray(
            [item[0] for item in joint_pairs], dtype=np.intp
        )
        self._joint_feature_indices = np.asarray(
            [item[1] for item in joint_pairs], dtype=np.intp
        )
        self._factor_pair_positions = tuple(
            (left, right)
            for left in range(len(observation_schema.factor_names))
            for right in range(left + 1, len(observation_schema.factor_names))
        )

        stock_cross_names = tuple(
            f"stock.{feature}__xsec_{cross}"
            for feature in observation_schema.stock_feature_names
            for cross in CROSS_SECTION_STATS
        )
        joint_cross_names = tuple(
            f"{pair_name}__{stat}"
            for _, _, pair_name in joint_pairs
            for stat in JOINT_STATS
        )
        self._stock_sequence_names = stock_cross_names + joint_cross_names
        stock_names = tuple(
            f"{feature}__time_{temporal}"
            for feature in self._stock_sequence_names
            for temporal in TEMPORAL_STATS
        )
        market_names = tuple(
            f"market.{feature}__time_{temporal}"
            for feature in observation_schema.market_feature_names
            for temporal in TEMPORAL_STATS
        )
        position_names = tuple(
            f"position.{feature}__xsec_{cross}"
            for feature in observation_schema.position_feature_names
            for cross in CROSS_SECTION_STATS
        )
        portfolio_names = tuple(
            f"portfolio.{feature}"
            for feature in observation_schema.portfolio_feature_names
        )
        factor_exposure_names = tuple(
            f"account_factor.{factor_name}__{stat}"
            for factor_name in observation_schema.factor_names
            for stat in FACTOR_EXPOSURE_STATS
        )
        factor_pair_exposure_names = tuple(
            "account_factor_pair."
            f"{observation_schema.factor_names[left]}__with__"
            f"{observation_schema.factor_names[right]}__{stat}"
            for left, right in self._factor_pair_positions
            for stat in FACTOR_PAIR_EXPOSURE_STATS
        )
        market_dimension = len(stock_names) + len(market_names)
        self.output_schema = EncodedObservationSchema(
            source_observation_schema=observation_schema.identifier,
            feature_names=(
                stock_names
                + market_names
                + position_names
                + factor_exposure_names
                + factor_pair_exposure_names
                + portfolio_names
            ),
            market_dimension=market_dimension,
        )

    @property
    def output_dimension(self) -> int:
        return self.output_schema.dimension

    @property
    def market_dimension(self) -> int:
        return self.output_schema.market_dimension

    @property
    def account_dimension(self) -> int:
        return self.output_schema.account_dimension

    @property
    def stock_sequence_count(self) -> int:
        return len(self._stock_sequence_names)

    def _validate_static(self, static: StaticObservation | Observation) -> None:
        schema = self.observation_schema
        if static.schema_version != schema.identifier:
            raise ValueError("observation schema identifier mismatch")
        expected_stock = (schema.lookback, schema.stock_count, schema.stock_feature_count)
        expected_market = (schema.lookback, schema.market_feature_count)
        if static.stock_panel.shape != expected_stock:
            raise ValueError(f"stock_panel shape mismatch: expected {expected_stock}")
        if static.market_panel.shape != expected_market:
            raise ValueError(f"market_panel shape mismatch: expected {expected_market}")
        if static.feature_mask.shape != expected_stock:
            raise ValueError("feature_mask shape mismatch")
        if static.stock_mask.shape != expected_stock[:2]:
            raise ValueError("stock_mask shape mismatch")
        if static.time_mask.shape != expected_stock[:1]:
            raise ValueError("time_mask shape mismatch")

    def _validate_account(self, account: AccountObservation | Observation) -> None:
        schema = self.observation_schema
        if account.schema_version != schema.identifier:
            raise ValueError("observation schema identifier mismatch")
        expected_position = (schema.stock_count, schema.position_feature_count)
        if account.position_panel.shape != expected_position:
            raise ValueError(f"position_panel shape mismatch: expected {expected_position}")
        if account.portfolio.shape != (schema.portfolio_feature_count,):
            raise ValueError("portfolio shape mismatch")
        if isinstance(account, AccountObservation):
            expected_factors = (len(schema.factor_names), schema.stock_count)
            if account.current_factor_ranks.shape != expected_factors:
                raise ValueError(
                    f"current_factor_ranks shape mismatch: expected {expected_factors}"
                )
            if account.current_factor_validity.shape != expected_factors:
                raise ValueError("current_factor_validity shape mismatch")

    def _pool_factor_joint_sequences(
        self,
        stock_panel: NDArray[np.float32],
        valid_stock_features: NDArray[np.bool_],
        time_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Pair each factor rank with every other registered stock feature."""

        numeric = np.asarray(stock_panel, dtype=np.float64)
        valid = np.asarray(valid_stock_features, dtype=np.bool_)
        factor_values = numeric[:, :, self._factor_feature_indices]
        factor_valid = valid[:, :, self._factor_feature_indices]
        clean_factors = np.where(factor_valid, factor_values, 0.0)
        clean_features = np.where(valid, numeric, 0.0)
        product_sums = np.einsum(
            "dnk,dnf->dkf",
            clean_factors,
            clean_features,
            optimize=True,
        )
        overlap_counts = np.einsum(
            "dnk,dnf->dkf",
            factor_valid.astype(np.float64),
            valid.astype(np.float64),
            optimize=True,
        )
        selected_sums = product_sums[
            :, self._joint_factor_positions, self._joint_feature_indices
        ]
        selected_counts = overlap_counts[
            :, self._joint_factor_positions, self._joint_feature_indices
        ]
        product_means = np.divide(
            selected_sums,
            selected_counts,
            out=np.zeros_like(selected_sums),
            where=selected_counts > 0,
        )
        overlap = selected_counts / float(stock_panel.shape[1])
        joint_values = np.stack((product_means, overlap), axis=2)
        joint_valid = np.repeat(
            selected_counts[:, :, None] > 0,
            len(JOINT_STATS),
            axis=2,
        )
        joint_valid[:, :, JOINT_STATS.index("overlap")] = time_mask[:, None]
        row_count = stock_panel.shape[0]
        return joint_values.reshape(row_count, -1), joint_valid.reshape(row_count, -1)

    def _pool_stock_sequences(
        self,
        stock_panel: NDArray[np.float32],
        feature_mask: NDArray[np.bool_],
        stock_mask: NDArray[np.bool_],
        time_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        valid_stock_features = (
            feature_mask
            & stock_mask[:, :, None]
            & time_mask[:, None, None]
        )
        cross_values, counts = _masked_cross_section(
            stock_panel,
            valid_stock_features,
            item_axis=1,
        )
        row_count = stock_panel.shape[0]
        stock_sequences = cross_values.reshape(row_count, -1)
        sequence_valid = np.repeat(
            counts[:, :, None] > 0,
            len(CROSS_SECTION_STATS),
            axis=2,
        )
        coverage_index = CROSS_SECTION_STATS.index("coverage")
        sequence_valid[:, :, coverage_index] = time_mask[:, None]
        joint_sequences, joint_valid = self._pool_factor_joint_sequences(
            stock_panel,
            valid_stock_features,
            time_mask,
        )
        return (
            np.concatenate((stock_sequences, joint_sequences), axis=1),
            np.concatenate((sequence_valid.reshape(row_count, -1), joint_valid), axis=1),
        )

    def _encode_market_sequences(
        self,
        stock_sequences: NDArray[np.float64],
        stock_sequence_valid: NDArray[np.bool_],
        market_panel: NDArray[np.float32],
        time_mask: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        stock_encoded = _masked_temporal(
            stock_sequences,
            stock_sequence_valid,
        ).reshape(-1)
        market_valid = np.broadcast_to(time_mask[:, None], market_panel.shape)
        market_encoded = _masked_temporal(market_panel, market_valid).reshape(-1)
        result = _as_finite_float32(np.concatenate((stock_encoded, market_encoded)))
        if result.shape != (self.market_dimension,):
            raise RuntimeError("market encoder produced an unexpected dimension")
        return result

    def encode_market(self, static: StaticObservation | Observation) -> NDArray[np.float32]:
        """Encode only account-independent market data."""

        self._validate_static(static)
        stock_sequences, stock_sequence_valid = self._pool_stock_sequences(
            static.stock_panel,
            static.feature_mask,
            static.stock_mask,
            static.time_mask,
        )
        return self._encode_market_sequences(
            stock_sequences,
            stock_sequence_valid,
            static.market_panel,
            static.time_mask,
        )

    @staticmethod
    def _weighted_factor_moments(
        factor_values: NDArray[np.float64],
        factor_validity: NDArray[np.bool_],
        weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        weighted_valid = factor_validity * weights[None, :]
        denominators = weighted_valid.sum(axis=1, dtype=np.float64)
        means = np.divide(
            (factor_values * weighted_valid).sum(axis=1, dtype=np.float64),
            denominators,
            out=np.zeros(factor_values.shape[0], dtype=np.float64),
            where=denominators > 0,
        )
        centered = np.where(
            factor_validity,
            factor_values - means[:, None],
            0.0,
        )
        variances = np.divide(
            (np.square(centered) * weighted_valid).sum(axis=1, dtype=np.float64),
            denominators,
            out=np.zeros(factor_values.shape[0], dtype=np.float64),
            where=denominators > 0,
        )
        total_weight = weights.sum(dtype=np.float64)
        coverage = (
            denominators / total_weight
            if total_weight > 0
            else np.zeros_like(denominators)
        )
        return means, np.sqrt(variances), coverage

    def _factor_exposure_encoding(
        self,
        account: AccountObservation | Observation,
        current_stock_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        if isinstance(account, Observation):
            factor_ranks = np.take(
                account.stock_panel[-1],
                self._factor_feature_indices,
                axis=1,
            ).T.astype(np.float64, copy=False)
            factor_validity = np.take(
                account.feature_mask[-1],
                self._factor_feature_indices,
                axis=1,
            ).T
        else:
            factor_ranks = account.current_factor_ranks.astype(np.float64, copy=False)
            factor_validity = account.current_factor_validity
        factor_validity = factor_validity & current_stock_mask[None, :]
        factor_ranks = np.where(factor_validity, factor_ranks, 0.0)

        held_weights = (account.position_panel[:, 0] != 0.0).astype(np.float64)
        position_weights = np.maximum(
            account.position_panel[:, 2].astype(np.float64),
            0.0,
        )
        sellable_weights = position_weights * np.clip(
            account.position_panel[:, 4].astype(np.float64),
            0.0,
            1.0,
        )
        market_weights = current_stock_mask.astype(np.float64)
        held_mean, _, _ = self._weighted_factor_moments(
            factor_ranks, factor_validity, held_weights
        )
        position_mean, position_std, position_coverage = self._weighted_factor_moments(
            factor_ranks, factor_validity, position_weights
        )
        sellable_mean, _, _ = self._weighted_factor_moments(
            factor_ranks, factor_validity, sellable_weights
        )
        market_mean, _, _ = self._weighted_factor_moments(
            factor_ranks, factor_validity, market_weights
        )
        has_position_weight = position_weights.sum(dtype=np.float64) > 0
        relative = position_mean - market_mean if has_position_weight else np.zeros_like(position_mean)
        factor_exposures = np.stack(
            (
                held_mean,
                position_mean,
                position_std,
                sellable_mean,
                relative,
                position_coverage,
            ),
            axis=1,
        ).reshape(-1)

        pair_exposures: list[float] = []
        total_position_weight = position_weights.sum(dtype=np.float64)
        for left, right in self._factor_pair_positions:
            pair_valid = factor_validity[left] & factor_validity[right]
            valid_weight = position_weights * pair_valid
            denominator = valid_weight.sum(dtype=np.float64)
            product_mean = (
                float(
                    (
                        factor_ranks[left]
                        * factor_ranks[right]
                        * valid_weight
                    ).sum(dtype=np.float64)
                    / denominator
                )
                if denominator > 0
                else 0.0
            )
            coverage = (
                float(denominator / total_position_weight)
                if total_position_weight > 0
                else 0.0
            )
            pair_exposures.extend((product_mean, coverage))
        return np.concatenate(
            (factor_exposures, np.asarray(pair_exposures, dtype=np.float64))
        )

    def encode_account(
        self,
        account: AccountObservation | Observation,
    ) -> NDArray[np.float32]:
        """Encode only dynamic position and portfolio state."""

        self._validate_account(account)
        if isinstance(account, Observation):
            current_stock_mask = account.stock_mask[-1]
        else:
            current_stock_mask = account.current_stock_mask
        held = account.position_panel[:, 0] != 0.0
        item_mask = current_stock_mask | held
        valid = np.broadcast_to(item_mask[:, None], account.position_panel.shape)
        position_encoded, _ = _masked_cross_section(
            account.position_panel,
            valid,
            item_axis=0,
        )
        factor_exposures = self._factor_exposure_encoding(account, current_stock_mask)
        result = _as_finite_float32(
            np.concatenate(
                (
                    position_encoded.reshape(-1),
                    factor_exposures,
                    account.portfolio,
                )
            )
        )
        if result.shape != (self.account_dimension,):
            raise RuntimeError("account encoder produced an unexpected dimension")
        return result

    def combine(
        self,
        market_encoding: NDArray[np.float32],
        account_encoding: NDArray[np.float32],
        *,
        normalizer: "TrainOnlyNormalizer | None" = None,
    ) -> NDArray[np.float32]:
        market = np.asarray(market_encoding, dtype=np.float32)
        account = np.asarray(account_encoding, dtype=np.float32)
        if market.shape != (self.market_dimension,):
            raise ValueError("market encoding dimension mismatch")
        if account.shape != (self.account_dimension,):
            raise ValueError("account encoding dimension mismatch")
        combined = np.ascontiguousarray(np.concatenate((market, account)), dtype=np.float32)
        if not np.isfinite(combined).all():
            raise ValueError("combined encoding must be finite")
        if normalizer is not None:
            if normalizer.encoder_schema != self.output_schema.identifier:
                raise ValueError("normalizer encoder schema mismatch")
            return normalizer.transform(combined)
        return combined

    def encode(
        self,
        observation: Observation,
        *,
        normalizer: "TrainOnlyNormalizer | None" = None,
    ) -> NDArray[np.float32]:
        """Encode a complete observation; identical to cached composition."""

        return self.combine(
            self.encode_market(observation),
            self.encode_account(observation),
            normalizer=normalizer,
        )


@dataclass(frozen=True)
class StaticMarketEncodingCache:
    """Small market vectors precomputed once for a sealed date range."""

    decision_indices: tuple[int, ...]
    decision_dates: tuple[str, ...]
    market_encodings: NDArray[np.float32]
    observation_schema: str
    encoder_schema: str
    _row_by_index: Mapping[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.decision_indices or len(set(self.decision_indices)) != len(self.decision_indices):
            raise ValueError("decision_indices must be non-empty and unique")
        if len(self.decision_dates) != len(self.decision_indices):
            raise ValueError("decision_dates length mismatch")
        values = np.asarray(self.market_encodings, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(self.decision_indices):
            raise ValueError("market_encodings must have shape [D, E_market]")
        if not np.isfinite(values).all():
            raise ValueError("market cache must be finite")
        frozen_values = np.array(values, dtype=np.float32, order="C", copy=True)
        frozen_values.flags.writeable = False
        object.__setattr__(self, "market_encodings", frozen_values)
        object.__setattr__(
            self,
            "_row_by_index",
            MappingProxyType({index: row for row, index in enumerate(self.decision_indices)}),
        )

    @classmethod
    def precompute(
        cls,
        builder: ObservationBuilder,
        encoder: ObservationEncoder,
        decision_indices: Iterable[int],
        *,
        chunk_rows: int = 32,
    ) -> "StaticMarketEncodingCache":
        """Encode a date range with each underlying market row built once.

        Overlapping 64-day windows are pooled in chunks, then only their small
        daily statistics are rolled.  This preserves direct-encoding results
        without rebuilding ``[L,N,F]`` for every PPO episode or decision date.
        """

        if encoder.observation_schema.identifier != builder.schema.identifier:
            raise ValueError("builder and encoder observation schemas differ")
        indices = tuple(int(index) for index in decision_indices)
        if not indices:
            raise ValueError("decision_indices must not be empty")
        if len(set(indices)) != len(indices):
            raise ValueError("decision_indices must be unique")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        if min(indices) < 0 or max(indices) >= len(builder.trade_dates):
            raise IndexError("decision index outside runtime date range")

        daily_start = max(0, min(indices) - builder.lookback + 1)
        daily_stop = max(indices) + 1
        daily_count = daily_stop - daily_start
        stock_sequence_count = encoder.stock_sequence_count
        daily_stock = np.empty((daily_count, stock_sequence_count), dtype=np.float64)
        daily_stock_valid = np.empty((daily_count, stock_sequence_count), dtype=np.bool_)
        daily_market = np.empty(
            (daily_count, builder.schema.market_feature_count), dtype=np.float32
        )
        daily_dates: list[str] = []

        for chunk_start in range(daily_start, daily_stop, chunk_rows):
            chunk_stop = min(chunk_start + chunk_rows, daily_stop)
            rows = builder.build_static_rows(chunk_start, chunk_stop)
            if rows.schema_version != builder.schema.identifier:
                raise ValueError("static row schema mismatch")
            time_mask = np.ones(chunk_stop - chunk_start, dtype=np.bool_)
            stock_sequences, stock_valid = encoder._pool_stock_sequences(
                rows.stock_panel,
                rows.feature_mask,
                rows.stock_mask,
                time_mask,
            )
            destination = slice(chunk_start - daily_start, chunk_stop - daily_start)
            daily_stock[destination] = stock_sequences
            daily_stock_valid[destination] = stock_valid
            daily_market[destination] = rows.market_panel
            daily_dates.extend(rows.decision_dates)

        vectors: list[NDArray[np.float32]] = []
        dates: list[str] = []
        lookback = builder.lookback
        for index in indices:
            source_start = max(0, index - lookback + 1)
            source_stop = index + 1
            relative_start = source_start - daily_start
            relative_stop = source_stop - daily_start
            row_count = relative_stop - relative_start
            destination = slice(lookback - row_count, lookback)
            stock_window = np.zeros((lookback, stock_sequence_count), dtype=np.float64)
            stock_valid_window = np.zeros_like(stock_window, dtype=np.bool_)
            market_window = np.zeros(
                (lookback, builder.schema.market_feature_count), dtype=np.float32
            )
            time_mask = np.zeros(lookback, dtype=np.bool_)
            stock_window[destination] = daily_stock[relative_start:relative_stop]
            stock_valid_window[destination] = daily_stock_valid[relative_start:relative_stop]
            market_window[destination] = daily_market[relative_start:relative_stop]
            time_mask[destination] = True
            vectors.append(
                encoder._encode_market_sequences(
                    stock_window,
                    stock_valid_window,
                    market_window,
                    time_mask,
                )
            )
            dates.append(daily_dates[index - daily_start])
        return cls(
            decision_indices=indices,
            decision_dates=tuple(dates),
            market_encodings=np.stack(vectors),
            observation_schema=builder.schema.identifier,
            encoder_schema=encoder.output_schema.identifier,
        )

    def market_for(
        self,
        decision_index: int,
        *,
        decision_date: str,
        encoder: ObservationEncoder,
    ) -> NDArray[np.float32]:
        if self.observation_schema != encoder.observation_schema.identifier:
            raise ValueError("cache observation schema mismatch")
        if self.encoder_schema != encoder.output_schema.identifier:
            raise ValueError("cache encoder schema mismatch")
        if decision_index not in self._row_by_index:
            raise KeyError(f"decision index {decision_index} is not cached")
        row = self._row_by_index[decision_index]
        if self.decision_dates[row] != decision_date:
            raise ValueError("cached decision date mismatch")
        return self.market_encodings[row]

    def encode_account_state(
        self,
        account: AccountObservation,
        encoder: ObservationEncoder,
        *,
        normalizer: "TrainOnlyNormalizer | None" = None,
    ) -> NDArray[np.float32]:
        """Combine a cached market vector with the current dynamic account."""

        market = self.market_for(
            account.decision_index,
            decision_date=account.decision_date,
            encoder=encoder,
        )
        return encoder.combine(
            market,
            encoder.encode_account(account),
            normalizer=normalizer,
        )


@dataclass(frozen=True)
class TrainOnlyNormalizer:
    """Serializable z-score state that may only be fitted on training data."""

    encoder_schema: str
    mean: NDArray[np.float32]
    scale: NDArray[np.float32]
    clip: float = 10.0
    sample_count: int = 0
    version: str = NORMALIZER_VERSION

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("normalizer mean and scale must have identical vector shapes")
        if not len(mean) or not np.isfinite(mean).all():
            raise ValueError("normalizer mean must be non-empty and finite")
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("normalizer scale must be finite and positive")
        if not np.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("normalizer clip must be finite and positive")
        if self.sample_count <= 0:
            raise ValueError("normalizer sample_count must be positive")
        frozen_mean = np.array(mean, dtype=np.float32, order="C", copy=True)
        frozen_scale = np.array(scale, dtype=np.float32, order="C", copy=True)
        frozen_mean.flags.writeable = False
        frozen_scale.flags.writeable = False
        object.__setattr__(self, "mean", frozen_mean)
        object.__setattr__(self, "scale", frozen_scale)

    @classmethod
    def fit(
        cls,
        encoded_observations: NDArray[np.floating],
        schema: EncodedObservationSchema,
        *,
        dataset_role: Literal["train", "validation", "test"],
        clip: float = 10.0,
    ) -> "TrainOnlyNormalizer":
        if dataset_role != "train":
            raise ValueError("normalizer fitting is permitted only on the training split")
        values = np.asarray(encoded_observations, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != schema.dimension or values.shape[0] == 0:
            raise ValueError(f"training encodings must have shape [D, {schema.dimension}]")
        if not np.isfinite(values).all():
            raise ValueError("training encodings must be finite")
        mean = values.mean(axis=0, dtype=np.float64)
        variance = np.square(values - mean).mean(axis=0, dtype=np.float64)
        scale = np.sqrt(variance)
        scale[scale < 1e-6] = 1.0
        return cls(
            encoder_schema=schema.identifier,
            mean=_as_finite_float32(mean),
            scale=_as_finite_float32(scale),
            clip=float(clip),
            sample_count=values.shape[0],
        )

    def transform(self, values: NDArray[np.floating]) -> NDArray[np.float32]:
        encoded = np.asarray(values, dtype=np.float32)
        if encoded.shape[-1:] != self.mean.shape:
            raise ValueError("normalizer input dimension mismatch")
        if not np.isfinite(encoded).all():
            raise ValueError("normalizer input must be finite")
        result = (encoded.astype(np.float64) - self.mean) / self.scale
        result = np.clip(result, -self.clip, self.clip)
        return _as_finite_float32(result)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": self.version,
            "encoder_schema": self.encoder_schema,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "clip": self.clip,
            "sample_count": self.sample_count,
        }
        payload["state_hash"] = _canonical_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrainOnlyNormalizer":
        hash_payload = {key: value for key, value in payload.items() if key != "state_hash"}
        if str(payload["state_hash"]) != _canonical_hash(hash_payload):
            raise ValueError("normalizer state hash mismatch")
        return cls(
            version=str(payload["version"]),
            encoder_schema=str(payload["encoder_schema"]),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            clip=float(payload["clip"]),
            sample_count=int(payload["sample_count"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_schema: EncodedObservationSchema,
    ) -> "TrainOnlyNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        normalizer = cls.from_dict(payload)
        if normalizer.encoder_schema != expected_schema.identifier:
            raise ValueError("normalizer encoder schema mismatch")
        return normalizer


__all__ = [
    "CROSS_SECTION_STATS",
    "EncodedObservationSchema",
    "ObservationEncoder",
    "StaticMarketEncodingCache",
    "TEMPORAL_STATS",
    "TrainOnlyNormalizer",
]
