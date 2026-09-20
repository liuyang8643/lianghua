"""Spawn-safe shared-memory transport for immutable prepared episodes.

This transport belongs to the domain episode rather than to PPO.  GA and RL
workers attach the same read-only runtime, factor and observation caches and
never reload or recompute them in child processes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import ExitStack
import gc
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import NDArray

from env.action_schema import ActionSchema
from env.backtest import PreparedEpisode, build_day_market
from env.encoder import ObservationEncoder, RawMarketStore
from env.observation import ObservationBuilder, ObservationSchema
from factor import FactorBatch, FactorMetadata
from offline_data import RuntimeSlice
from offline_data.contracts import RuntimeManifest


@dataclass(frozen=True, slots=True)
class SharedArrayDescriptor:
    """Pickleable metadata for one immutable shared ndarray."""

    label: str
    shared_memory_name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int

    def attach(self) -> tuple[SharedMemory, NDArray[np.generic]]:
        shared = SharedMemory(name=self.shared_memory_name, create=False)
        try:
            values = np.ndarray(
                self.shape, dtype=np.dtype(self.dtype), buffer=shared.buf
            )
            values.flags.writeable = False
        except BaseException:
            shared.close()
            raise
        return shared, values


class AttachedPreparedEpisode:
    """Worker-owned shared-memory handles and reconstructed episode."""

    def __init__(
        self,
        episode: PreparedEpisode,
        handles: tuple[SharedMemory, ...],
    ) -> None:
        self._episode: PreparedEpisode | None = episode
        self._handles = handles

    @property
    def episode(self) -> PreparedEpisode:
        if self._episode is None:
            raise RuntimeError("shared prepared episode is closed")
        return self._episode

    @property
    def closed(self) -> bool:
        return self._episode is None

    def close(self) -> None:
        if self.closed:
            return
        self._episode = None
        gc.collect()
        failures: list[BaseException] = []
        for handle in reversed(self._handles):
            try:
                handle.close()
            except BaseException as error:
                failures.append(error)
        self._handles = ()
        if failures:
            raise BaseExceptionGroup(
                "worker failed to close SharedMemory handles", failures
            )

    def __enter__(self) -> "AttachedPreparedEpisode":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class SharedPreparedEpisodeDescriptor:
    """Small spawn-safe recipe for rebuilding a PreparedEpisode."""

    runtime_trade_dates: SharedArrayDescriptor
    runtime_data: tuple[tuple[str, SharedArrayDescriptor], ...]
    factor_raw: SharedArrayDescriptor
    factor_ranks: SharedArrayDescriptor
    factor_validity: SharedArrayDescriptor
    factor_filters: SharedArrayDescriptor
    listing_age: SharedArrayDescriptor
    raw_rows: SharedArrayDescriptor | None
    raw_pit_universe_mask: SharedArrayDescriptor | None
    raw_row_valid: SharedArrayDescriptor | None
    stock_codes: tuple[str, ...]
    runtime_decision_start: int
    runtime_decision_stop: int
    runtime_manifest: RuntimeManifest
    factor_schema_version: str
    factor_schema_hash: str
    factor_runtime_schema_hash: str
    factor_decision_start: int
    factor_decision_stop: int
    factor_metadata: tuple[FactorMetadata, ...]
    filter_metadata: tuple[FactorMetadata, ...]
    episode_decision_start: int
    episode_decision_stop: int
    episode_prefilter_n: int | None
    lookback: int
    raw_decision_dates: tuple[str, ...]
    raw_decision_start: int
    raw_decision_stop: int
    raw_row_start: int
    raw_observation_schema: dict | None
    raw_encoder_schema: str
    action_schema: dict | None

    @property
    def arrays(self) -> tuple[SharedArrayDescriptor, ...]:
        return tuple(dict.fromkeys(item for item in (
            self.runtime_trade_dates,
            *(descriptor for _, descriptor in self.runtime_data),
            self.factor_raw,
            self.factor_ranks,
            self.factor_validity,
            self.factor_filters,
            self.listing_age,
            self.raw_rows,
            self.raw_pit_universe_mask,
            self.raw_row_valid,
        ) if item is not None))

    @property
    def shared_memory_bytes(self) -> int:
        return sum(item.nbytes for item in self.arrays)

    def _full_episode_rows(self) -> tuple[int, int]:
        start, stop = self.episode_decision_start, self.episode_decision_stop
        if stop - start < 2:
            raise ValueError("sealed episode needs at least two observations")
        if self.raw_rows is not None and (
            self.raw_row_start + self.raw_decision_start != start
            or self.raw_row_start + self.raw_decision_stop != stop
            or len(self.raw_decision_dates) != stop - start
        ):
            raise ValueError("shared raw store does not cover its sealed interval")
        return start, stop

    def attach(self) -> AttachedPreparedEpisode:
        start, stop = self._full_episode_rows()
        handles: list[SharedMemory] = []
        arrays: dict[str, NDArray[np.generic]] = {}
        try:
            for descriptor in self.arrays:
                handle, values = descriptor.attach()
                handles.append(handle)
                arrays[descriptor.label] = values
            trade_dates = arrays[self.runtime_trade_dates.label]
            runtime = RuntimeSlice(
                stock_codes=self.stock_codes,
                trade_dates=trade_dates,
                data={
                    name: arrays[descriptor.label]
                    for name, descriptor in self.runtime_data
                },
                decision_start=self.runtime_decision_start,
                decision_stop=self.runtime_decision_stop,
                manifest=self.runtime_manifest,
            )
            factors = FactorBatch(
                schema_version=self.factor_schema_version,
                schema_hash=self.factor_schema_hash,
                runtime_schema_hash=self.factor_runtime_schema_hash,
                stock_codes=self.stock_codes,
                trade_dates=trade_dates,
                decision_start=self.factor_decision_start,
                decision_stop=self.factor_decision_stop,
                factor_metadata=self.factor_metadata,
                filter_metadata=self.filter_metadata,
                raw=arrays[self.factor_raw.label],
                ranks=arrays[self.factor_ranks.label],
                validity=arrays[self.factor_validity.label],
                filters=arrays[self.factor_filters.label],
            )
            builder = encoder = store = None
            if self.raw_rows is not None:
                builder = ObservationBuilder(
                    runtime,
                    factors,
                    lookback=self.lookback,
                    listing_age=arrays[self.listing_age.label],
                    day_markets=tuple(
                        build_day_market(runtime, factors, arrays[self.listing_age.label], index).seal(borrow_readonly=True)
                        for index in range(runtime.n_dates)
                    ),
                    action_schema=ActionSchema.from_dict(self.action_schema),
                )
                encoder = ObservationEncoder(builder.schema)
                store = RawMarketStore(
                    raw_rows=arrays[self.raw_rows.label],
                    pit_universe_mask=arrays[self.raw_pit_universe_mask.label],
                    row_valid=arrays[self.raw_row_valid.label],
                    schema=ObservationSchema.from_dict(self.raw_observation_schema),
                    decision_start=self.raw_decision_start,
                    decision_stop=self.raw_decision_stop,
                    decision_dates=self.raw_decision_dates,
                    row_start=self.raw_row_start,
                )
                if encoder.output_schema.identifier != self.raw_encoder_schema:
                    raise ValueError("shared raw encoder schema mismatch")
            episode = PreparedEpisode(
                runtime=runtime,
                factors=factors,
                observation_builder=builder,
                encoder=encoder,
                market_store=store,
                listing_age=arrays[self.listing_age.label],
                decision_start=start,
                decision_stop=stop,
                prefilter_n=self.episode_prefilter_n,
            )
            return AttachedPreparedEpisode(episode, tuple(handles))
        except BaseException as attach_error:
            gc.collect()
            failures: list[BaseException] = []
            for handle in reversed(handles):
                try:
                    handle.close()
                except BaseException as error:
                    failures.append(error)
            if failures:
                raise BaseExceptionGroup(
                    "shared episode attach and cleanup both failed",
                    [attach_error, *failures],
                )
            raise


class SharedPreparedEpisodeOwner:
    """Parent owner that creates and ultimately unlinks shared arrays."""

    def __init__(
        self,
        descriptor: SharedPreparedEpisodeDescriptor,
        segments: tuple[SharedMemory, ...],
    ) -> None:
        self.descriptor = descriptor
        self._segments = segments
        self._closed = False

    @classmethod
    def create(cls, episode: PreparedEpisode) -> "SharedPreparedEpisodeOwner":
        episode = episode.compact_for_replay()
        segments: list[SharedMemory] = []

        def share(
            label: str, source: NDArray[np.generic]
        ) -> SharedArrayDescriptor:
            values = np.ascontiguousarray(source)
            if values.nbytes <= 0:
                raise ValueError(f"shared array {label!r} must not be empty")
            segment = SharedMemory(create=True, size=values.nbytes)
            segments.append(segment)
            destination = np.ndarray(
                values.shape, dtype=values.dtype, buffer=segment.buf
            )
            destination[...] = values
            del destination
            return SharedArrayDescriptor(
                label=label,
                shared_memory_name=segment.name,
                shape=tuple(int(value) for value in values.shape),
                dtype=values.dtype.str,
                nbytes=int(values.nbytes),
            )

        try:
            trade_dates = share(
                "runtime.trade_dates", episode.runtime.trade_dates
            )
            runtime_data = tuple(
                (name, share(f"runtime.data.{name}", values))
                for name, values in episode.runtime.data.items()
            )
            age = episode.listing_age
            runtime_age = episode.runtime.field("listing_age")
            # Only alias the exact read-only view; custom episode ages keep
            # their own segment even when a separate array has equal values.
            if (
                not age.flags.writeable and not runtime_age.flags.writeable
                and age.dtype == runtime_age.dtype == np.dtype(np.int32)
                and age.shape == runtime_age.shape
                and age.strides == runtime_age.strides
                and age.ctypes.data == runtime_age.ctypes.data
            ):
                listing_age = dict(runtime_data)["listing_age"]
            else:
                listing_age = share("episode.listing_age", age)
            worker_manifest = replace(
                episode.runtime.manifest, source_path="<shared-memory>"
            )
            descriptor = SharedPreparedEpisodeDescriptor(
                runtime_trade_dates=trade_dates,
                runtime_data=runtime_data,
                factor_raw=share("factors.raw", episode.factors.raw),
                factor_ranks=share("factors.ranks", episode.factors.ranks),
                factor_validity=share(
                    "factors.validity", episode.factors.validity
                ),
                factor_filters=share(
                    "factors.filters", episode.factors.filters
                ),
                listing_age=listing_age,
                raw_rows=share("episode.raw_rows", episode.market_store.raw_rows)
                    if episode.market_store is not None else None,
                raw_pit_universe_mask=share("episode.raw_pit_universe_mask", episode.market_store.pit_universe_mask)
                    if episode.market_store is not None else None,
                raw_row_valid=share("episode.raw_row_valid", episode.market_store.row_valid)
                    if episode.market_store is not None else None,
                stock_codes=episode.runtime.stock_codes,
                runtime_decision_start=episode.runtime.decision_start,
                runtime_decision_stop=episode.runtime.decision_stop,
                runtime_manifest=worker_manifest,
                factor_schema_version=episode.factors.schema_version,
                factor_schema_hash=episode.factors.schema_hash,
                factor_runtime_schema_hash=episode.factors.runtime_schema_hash,
                factor_decision_start=episode.factors.decision_start,
                factor_decision_stop=episode.factors.decision_stop,
                factor_metadata=episode.factors.factor_metadata,
                filter_metadata=episode.factors.filter_metadata,
                episode_decision_start=episode.decision_start,
                episode_decision_stop=episode.decision_stop,
                episode_prefilter_n=episode.prefilter_n,
                lookback=episode.observation_builder.lookback if episode.observation_builder is not None else 0,
                raw_decision_dates=episode.market_store.decision_dates if episode.market_store is not None else (),
                raw_decision_start=episode.market_store.decision_start if episode.market_store is not None else 0,
                raw_decision_stop=episode.market_store.decision_stop if episode.market_store is not None else 0,
                raw_row_start=episode.market_store.row_start if episode.market_store is not None else 0,
                raw_observation_schema=episode.market_store.schema.to_dict() if episode.market_store is not None else None,
                raw_encoder_schema=episode.encoder.output_schema.identifier if episode.encoder is not None else "",
                action_schema=episode.observation_builder.action_schema.to_dict() if episode.observation_builder is not None else None,
            )
            return cls(descriptor, tuple(segments))
        except BaseException as creation_error:
            failures = cls._release_segments(tuple(segments), unlink=True)
            if failures:
                raise BaseExceptionGroup(
                    "shared episode creation and cleanup both failed",
                    [creation_error, *failures],
                )
            raise

    @staticmethod
    def _release_segments(
        segments: tuple[SharedMemory, ...], *, unlink: bool
    ) -> list[BaseException]:
        failures: list[BaseException] = []
        for segment in reversed(segments):
            try:
                segment.close()
            except BaseException as error:
                failures.append(error)
            if unlink:
                try:
                    segment.unlink()
                except FileNotFoundError:
                    pass
                except BaseException as error:
                    failures.append(error)
        return failures

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        failures = self._release_segments(self._segments, unlink=True)
        self._segments = ()
        self._closed = True
        if failures:
            raise BaseExceptionGroup(
                "failed to release episode SharedMemory", failures
            )

    def __enter__(self) -> "SharedPreparedEpisodeOwner":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class ResidentPreparedEpisode:
    """One compact shared owner, with a lazy coordinator attachment.

    Preparation arrays are never retained. Descriptor-only PPO evaluators need
    no coordinator reconstruction; GA obtains the same immutable episode lazily.
    """

    def __init__(self, episode: PreparedEpisode) -> None:
        self._resources = ExitStack()
        self._owner = self._resources.enter_context(SharedPreparedEpisodeOwner.create(episode))
        self._attachment: AttachedPreparedEpisode | None = None

    @property
    def descriptor(self) -> SharedPreparedEpisodeDescriptor:
        if self._owner.closed:
            raise RuntimeError("resident episode is closed")
        return self._owner.descriptor

    @property
    def episode(self) -> PreparedEpisode:
        if self._attachment is None:
            self._attachment = self._resources.enter_context(self.descriptor.attach())
        return self._attachment.episode

    def close(self) -> None:
        self._resources.close()

    def __enter__(self) -> "ResidentPreparedEpisode":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._resources.__exit__(*exc_info)


__all__ = [
    "ResidentPreparedEpisode",
    "AttachedPreparedEpisode",
    "SharedArrayDescriptor",
    "SharedPreparedEpisodeDescriptor",
    "SharedPreparedEpisodeOwner",
]
