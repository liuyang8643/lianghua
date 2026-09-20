"""Learned raw-panel features resolved from one explicitly bound sealed split."""
from __future__ import annotations

from functools import partial
import math
from typing import Mapping

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ai.rl.device import require_cuda_device, require_cuda_input
from env.observation import RAW_MISSING_VALUE


RAW_PANEL_NETWORK_VERSION = "wbr-raw-panel-network-v4-invertible-account-asinh"
RAW_PANEL_CONFIG = {
    "patch_size": 21,
    "d_model": 16,
    "nhead": 2,
    "dim_feedforward": 32,
    "num_layers": 1,
    "cross_stock_queries": 4,
    "patch_batch_size": 32768,
    "sequence_batch_size": 8192,
    "retained_token_input_bytes": 384 * 1024 * 1024,
}


def _masked_weights(logits: th.Tensor, valid: th.Tensor) -> th.Tensor:
    """Normalize only real members; an empty set contributes exactly zero."""
    weights = th.softmax(logits.masked_fill(~valid, th.finfo(logits.dtype).min), dim=-1)
    weights = weights * valid
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(th.finfo(weights.dtype).tiny)


class TemporalPatchEncoder(nn.Module):
    """Shared per-sequence Transformer; no stock identity or calendar input."""

    def __init__(self, feature_count: int, lookback: int, config: Mapping[str, int]) -> None:
        super().__init__()
        self.patch_size = config["patch_size"]
        self.patch_count = math.ceil(lookback / self.patch_size)
        self.padded_length = self.patch_count * self.patch_size
        width = config["d_model"]
        self.projection = nn.Linear(self.patch_size * feature_count, width)
        self.position = nn.Parameter(th.zeros(1, self.patch_count, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"], dropout=0.0,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config["num_layers"], enable_nested_tensor=False)
        self.attention = nn.Linear(width, 1, bias=False)
        self.width = width

    def forward(self, rows: th.Tensor, valid: th.Tensor) -> th.Tensor:
        batch, length, features = rows.shape
        padding = self.padded_length - length
        rows = F.pad(rows * valid.unsqueeze(-1), (0, 0, padding, 0))
        valid = F.pad(valid, (padding, 0), value=False)
        patch_valid = valid.reshape(batch, self.patch_count, self.patch_size).any(-1)
        active = patch_valid.any(-1)
        # An empty history receives one inert attention token. Multiplication
        # by its zero validity removes both output and gradient without a
        # device-to-host conditional or variable-size boolean gather.
        first_token = th.arange(self.patch_count, device=rows.device) == 0
        safe_valid = patch_valid | ((~active)[:, None] & first_token[None])
        selected = rows.reshape(batch, self.patch_count, self.patch_size * features)
        pooled = self.encode_tokens(self.projection(selected), safe_valid)
        return pooled * active[:, None]

    def encode_tokens(self, tokens: th.Tensor, valid: th.Tensor) -> th.Tensor:
        tokens = self.transformer(tokens + self.position, src_key_padding_mask=~valid)
        weights = _masked_weights(self.attention(tokens).squeeze(-1), valid)
        return (tokens * weights.unsqueeze(-1)).sum(1)



class _WindowMemberIndex(nn.Module):
    """Immutable PIT window membership shared by patch and sequence gathers."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("membership", th.empty(0, dtype=th.bool), persistent=False)
        self.register_buffer("stocks", th.empty(0, dtype=th.int64), persistent=False)
        self.offsets = np.zeros(1, dtype=np.int64)

    def bind(self, prefix: np.ndarray, span: int, device: th.device) -> None:
        rows = np.arange(len(prefix))
        member = prefix > prefix[np.maximum(rows - span, 0)]
        self.offsets = np.concatenate(([0], np.cumsum(member.sum(axis=1), dtype=np.int64)))
        self.offsets.flags.writeable = False
        self.membership = th.tensor(member, device=device)
        self.stocks = th.tensor(np.nonzero(member)[1], device=device)

    def pairs(self, ends: np.ndarray) -> tuple[th.Tensor, th.Tensor]:
        rows = np.maximum(ends + 1, 0)
        starts = self.offsets[rows]
        counts = self.offsets[rows + 1] - starts
        local_starts = np.cumsum(counts, dtype=np.int64) - counts
        total = int(counts.sum())
        device = self.stocks.device
        axis = th.repeat_interleave(
            th.arange(len(ends), device=device), th.tensor(counts, device=device), output_size=total)
        selected = th.arange(total, device=device) + th.tensor(starts - local_starts, device=device)[axis]
        return axis, self.stocks[selected]

    def mask(self, ends: np.ndarray) -> th.Tensor:
        return self.membership[th.tensor(np.maximum(ends + 1, 0), device=self.membership.device)]


class RawPanelFeatures(nn.Module):
    """One raw network for rollout, minibatches, frozen evaluation and live use.

    The compact input contains a routing row followed by dynamic account data.
    Routing is removed before any learned layer. Differentiable updates always
    rebuild learned features. Evaluation reuses a market bank only while its
    binding and every market parameter version remain unchanged.
    """

    def __init__(self, encoded_schema: Mapping[str, object], config: Mapping[str, int]) -> None:
        super().__init__()
        self.encoded_schema = dict(encoded_schema)
        self.config = dict(config)
        if set(self.config) != set(RAW_PANEL_CONFIG) or any(
            type(value) is not int or value <= 0 for value in self.config.values()
        ):
            raise ValueError("raw panel network configuration must contain positive integer fields")
        if self.config["d_model"] % self.config["nhead"]:
            raise ValueError("raw panel width must be divisible by attention heads")
        self.lookback = int(encoded_schema["lookback"])
        self.stock_count = int(encoded_schema["stock_count"])
        self.stock_features = len(encoded_schema["stock_feature_names"])
        names = list(encoded_schema["stock_feature_names"])
        history = [names.index(n) for n in encoded_schema['historical_stock_feature_names']]
        current = [names.index(n) for n in encoded_schema['latest_stock_feature_names']]
        if sorted(history + current) != list(range(self.stock_features)):
            raise ValueError('history and current fields must partition the raw state')
        self.register_buffer('historical_columns', th.tensor(history), persistent=False)
        self.register_buffer('current_columns', th.tensor(current), persistent=False)
        self.historical_features = len(history)
        self.position_features = len(encoded_schema["position_feature_names"])
        self.portfolio_features = len(encoded_schema["portfolio_feature_names"])
        self.history_features = len(encoded_schema["history_feature_names"])
        self.dimension = int(encoded_schema["dimension"])
        self.position_stop = 1 + self.stock_count * self.position_features
        self.portfolio_stop = self.position_stop + self.portfolio_features
        if self.dimension != self.portfolio_stop + self.lookback * self.history_features:
            raise ValueError("raw compact schema does not match its declared layout")
        self.width = self.config["d_model"]
        self.market_temporal = TemporalPatchEncoder(self.historical_features, self.lookback, self.config)
        self.history_temporal = TemporalPatchEncoder(self.history_features, self.lookback, self.config)
        self.stock_account = nn.Sequential(
            nn.Linear(self.width + self.position_features + len(current), self.width), nn.GELU(), nn.LayerNorm(self.width))
        self.stock_keys = nn.Linear(self.width, self.width, bias=False)
        self.stock_values = nn.Linear(self.width, self.width, bias=False)
        self.stock_queries = nn.Parameter(th.empty(self.config["cross_stock_queries"], self.width))
        nn.init.normal_(self.stock_queries, std=1 / math.sqrt(self.width))
        combined = self.config["cross_stock_queries"] * self.width + self.width + self.portfolio_features
        self.fusion = nn.Sequential(nn.Linear(combined, self.width), nn.GELU(), nn.LayerNorm(self.width))
        self.patch_batch = self.config["patch_batch_size"]
        self.sequence_batch = self.config["sequence_batch_size"]
        self.market_store = None
        self.normalizer = None
        for name in ("stock_scale", "position_scale", "portfolio_scale", "history_scale"):
            self.register_buffer(name, th.empty(0), persistent=False)
        self.register_buffer("_frozen_market", th.empty(0), persistent=False)
        self._frozen_key = None
        self._binding_version = 0
        self.register_buffer("raw_bank", th.empty(0), persistent=False)
        self.register_buffer("pit_bank", th.empty(0, dtype=th.bool), persistent=False)
        self.register_buffer("row_valid_bank", th.empty(0, dtype=th.bool), persistent=False)
        first_span = self.lookback - (self.market_temporal.patch_count - 1) * self.config["patch_size"]
        self._member_indexes = nn.ModuleDict({str(span): _WindowMemberIndex() for span in
            sorted({self.config["patch_size"], self.lookback, first_span})})

    def bind_market_store(self, store, normalizer) -> None:
        require_cuda_device(self.stock_queries.device)
        if store.schema.identifier != self.encoded_schema["source_observation_schema"]:
            raise ValueError("raw market store and encoded observation schema differ")
        expected_encoder = f"{self.encoded_schema['version']}:{self.encoded_schema['schema_hash']}"
        if normalizer.encoder_schema != expected_encoder:
            raise ValueError("raw normalizer and encoded observation schema differ")
        if store.raw_rows.shape[1:] != (self.stock_count, self.stock_features):
            raise ValueError("raw market store has an incompatible stock or field axis")
        if store.raw_rows.flags.writeable or store.pit_universe_mask.flags.writeable or store.row_valid.flags.writeable:
            raise ValueError("bound raw market arrays must be read-only")
        for name, size in (("stock_scale", self.stock_features), ("position_scale", self.position_features),
                           ("portfolio_scale", self.portfolio_features), ("history_scale", self.history_features)):
            values = np.asarray(getattr(normalizer, name))
            if values.shape != (size,) or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"invalid raw normalizer {name}")
            setattr(self, name, th.tensor(values, dtype=th.float32, device=self.stock_queries.device))
        # This device cache contains only immutable input numbers, never
        # learned embeddings. It is deliberately absent from model state_dict.
        self.raw_bank = th.tensor(store.raw_rows, device=self.stock_queries.device)
        self.pit_bank = th.tensor(store.pit_universe_mask, device=self.stock_queries.device)
        self.row_valid_bank = th.tensor(store.row_valid, device=self.stock_queries.device)
        # Fixed field scaling and PIT zeroing commute with every learned layer.
        # Store only this one normalized raw table, not one copy per window.
        missing = self.raw_bank == float(RAW_MISSING_VALUE)
        negative = (self.raw_bank < 0) & ~missing
        self.raw_bank.masked_fill_(missing, 0).div_(self.stock_scale)
        # Disjoint ranges: nonnegative values >=0, signed values <-2, missing=-1.
        # This preserves signed source values; missing is never inferred from sign.
        self.raw_bank.sub_(negative.to(self.raw_bank.dtype) * 2).masked_fill_(missing, -1)
        self.raw_bank.mul_((self.pit_bank & self.row_valid_bank[:, None]).unsqueeze(-1))
        member = store.pit_universe_mask & store.row_valid[:, None]
        prefix = np.vstack((np.zeros((1, self.stock_count), dtype=np.int32),
                            np.cumsum(member, axis=0, dtype=np.int32)))
        for span, index in self._member_indexes.items():
            index.bind(prefix, int(span), self.raw_bank.device)
        self.market_store, self.normalizer = store, normalizer
        self._binding_version += 1
        self._invalidate_frozen_market()

    def _project_chunk(self, ends, stocks, span, bank):
        days = ends[:, None] - th.arange(span - 1, -1, -1, device=bank.device)[None]
        rows = bank[days.clamp_min(0)[..., None], stocks[:, None, None], self.historical_columns[None, None, :]] * (days >= 0).unsqueeze(-1)
        weight = self.market_temporal.projection.weight.view(self.width, self.config['patch_size'], self.historical_features)[:, -span:]
        return F.linear(rows.flatten(1), weight.reshape(self.width, -1), self.market_temporal.projection.bias)

    def _patches(self, ends, span, bank):
        """Project each distinct causal patch once, keeping its gradient graph."""
        index = self._member_indexes[str(span)]
        end_axis, stock_axis = index.pairs(ends)
        all_ends = th.as_tensor(ends, device=bank.device)[end_axis]
        projected = []
        for first in range(0, len(end_axis), self.patch_batch):
            last = first + self.patch_batch
            e = all_ends[first:last]
            s = stock_axis[first:last]
            resolve = partial(self._project_chunk, e, s, span, bank)
            projected.append(checkpoint(resolve, use_reentrant=False, preserve_rng_state=False) if th.is_grad_enabled() else resolve())
        result = bank.new_zeros((len(ends) * self.stock_count, self.width))
        if projected:
            indices = end_axis * self.stock_count + stock_axis
            result = result.index_copy(0, indices, th.cat(projected))
        return result.reshape(len(ends), self.stock_count, self.width), index.mask(ends)

    def _sequence_tokens(self, date_ids, stocks, lookup, patches, valid, short_patches, short_valid):
        indices = lookup[date_ids]
        tokens = patches[indices, stocks[:, None]]
        mask = valid[indices, stocks[:, None]]
        if short_patches is not None:
            tokens = th.cat((short_patches[date_ids, stocks][:, None], tokens[:, 1:]), dim=1)
            mask = th.cat((short_valid[date_ids, stocks][:, None], mask[:, 1:]), dim=1)
        return tokens, mask

    def _market_many(self, refs):
        """Batch date/stock pairs without dropping the full stock axis.

        Non-multiple lookbacks have a left-padded first patch: only the
        projection weights for its actual suffix may read historical rows.
        """
        patch = self.config['patch_size']
        count = self.market_temporal.patch_count
        endpoints = refs[:, None] - np.arange(count - 1, -1, -1)[None] * patch
        unique, inverse = np.unique(endpoints, return_inverse=True)
        patches, valid = self._patches(unique, patch, self.raw_bank)
        span = self.lookback - (count - 1) * patch
        shorts, shortvalid = self._patches(endpoints[:, 0], span, self.raw_bank) if span < patch else (None, None)
        lookup = th.as_tensor(inverse.reshape(len(refs), count), device=self.raw_bank.device)
        dates, stocks = self._member_indexes[str(self.lookback)].pairs(refs)
        output = self.raw_bank.new_zeros((len(refs) * self.stock_count, self.width))
        if not len(dates):
            return output.reshape(len(refs), self.stock_count, self.width)
        resolve = partial(self._sequence_tokens, lookup=lookup, patches=patches,
                          valid=valid, short_patches=shorts, short_valid=shortvalid)
        if th.is_grad_enabled():
            # One gather has one backward scatter. Gathering separately for
            # every sequence chunk would allocate and add a full patch-bank
            # gradient for every chunk. SplitBackward concatenates once.
            tokens, mask = resolve(dates, stocks)
            chunks = zip(tokens.split(self.sequence_batch), mask.split(self.sequence_batch))
        else:
            # Frozen whole-split evaluation has thousands of dates. Stream
            # its input tokens; there is no gradient scatter to consolidate.
            chunks = (resolve(d, s) for d, s in zip(
                dates.split(self.sequence_batch), stocks.split(self.sequence_batch)))
        result = []
        token_input_bytes = 0
        for tokens, mask in chunks:
            # Retain a fixed prefix of sequence activations, then recompute
            # the remaining chunks. The quota measures input tokens, not
            # total activation memory, and resets for every forward.
            token_input_bytes += tokens.numel() * tokens.element_size()
            encode = partial(self.market_temporal.encode_tokens, tokens, mask)
            result.append(checkpoint(encode, use_reentrant=False, preserve_rng_state=False)
                          if th.is_grad_enabled() and token_input_bytes > self.config['retained_token_input_bytes']
                          else encode())
        indices = dates * self.stock_count + stocks
        output = output.index_copy(0, indices, th.cat(result))
        return output.reshape(len(refs), self.stock_count, self.width)

    def _invalidate_frozen_market(self) -> None:
        self._frozen_key = None
        self._frozen_market = self.raw_bank.new_empty(0)

    def train(self, mode: bool = True):
        if mode:
            self._invalidate_frozen_market()
        return super().train(mode)

    @property
    def frozen_market_token(self):
        """Identity used by a CUDA execution graph after explicit preparation."""
        return self._binding_version, self._frozen_key

    @th.no_grad()
    def prepare_frozen_market(self) -> th.Tensor:
        """Compute the complete sealed market bank for unchanged parameters.

        Callers must count this cost in inference/rollout preparation. No
        detached features are ever used by a differentiable PPO update.
        """
        if self.training or self.market_store is None:
            raise RuntimeError("frozen market preparation requires a bound evaluation-mode model")
        require_cuda_device(self.stock_queries.device)
        key = tuple((id(p), p.data_ptr(), p._version) for p in self.market_temporal.parameters())
        if key != self._frozen_key:
            store = self.market_store
            self._frozen_market = self._market_many(np.arange(store.decision_start, store.decision_stop))
            self._frozen_key = key
        return self._frozen_market

    def validate_references(self, observation: th.Tensor) -> np.ndarray:
        if self.market_store is None:
            raise RuntimeError('bind_market_store must precede raw policy inference or training')
        require_cuda_input(self, observation)
        if observation.ndim != 2 or observation.shape[1] != self.dimension:
            raise ValueError('raw policy observation does not match compact layout')
        values = observation[:, 0].detach().cpu().numpy()
        store = self.market_store
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError('raw row references must be finite integers')
        refs = values.astype(np.int64)
        if np.any(refs < store.decision_start) or np.any(refs >= store.decision_stop):
            raise ValueError('raw row reference lies outside the sealed decision interval')
        if not store.row_valid[refs].all():
            raise ValueError('raw row reference must identify a real decision date')
        return refs

    def forward_frozen_tensor(self, observation: th.Tensor) -> th.Tensor:
        """Capture-ready account path; validate/prepare precedes graph replay."""
        refs = observation[:, 0].detach().to(dtype=th.int64)
        market = self._frozen_market[refs - self.market_store.decision_start]
        return self._forward_account(observation, market, self.pit_bank[refs])

    def forward(self, observation: th.Tensor) -> th.Tensor:
        refs = self.validate_references(observation)
        if not th.is_grad_enabled() and not self.training:
            self.prepare_frozen_market()
            return self.forward_frozen_tensor(observation)
        self._invalidate_frozen_market()
        unique, inverse = np.unique(refs, return_inverse=True)
        market = self._market_many(unique)[th.as_tensor(inverse, device=observation.device)]
        member = self.pit_bank[th.as_tensor(refs, device=observation.device)]
        return self._forward_account(observation, market, member)

    def _forward_account(self, observation: th.Tensor, market: th.Tensor,
                         member: th.Tensor) -> th.Tensor:
        positions = observation[:, 1:self.position_stop].reshape(-1, self.stock_count, self.position_features)
        active = member | (positions[..., 0] > 0)
        refs = observation[:, 0].detach().to(dtype=th.int64)
        current = self.raw_bank[refs[:, None, None], th.arange(self.stock_count, device=observation.device)[None, :, None], self.current_columns[None, None, :]]
        position_values = positions / self.position_scale
        position_values = th.stack((th.asinh(position_values[..., 0]), position_values[..., 1],
                                    th.asinh(position_values[..., 2]), position_values[..., 3]), dim=-1)
        stock = self.stock_account(th.cat((market, current, position_values), dim=-1))
        scores = th.einsum('qd,bnd->bqn', self.stock_queries, self.stock_keys(stock)) / self.width ** 0.5
        weights = _masked_weights(scores, active[:, None, :])
        pooled = th.einsum('bqn,bnd->bqd', weights, self.stock_values(stock)).flatten(1)
        history = observation[:, self.portfolio_stop:].reshape(-1, self.lookback, self.history_features)
        hist = self.history_temporal(history / self.history_scale, history[..., 0] > 0)
        portfolio = observation[:, self.position_stop:self.portfolio_stop] / self.portfolio_scale
        portfolio = th.cat((th.asinh(portfolio[:, :3]), portfolio[:, 3:]), dim=1)
        return self.fusion(th.cat((pooled, hist, portfolio), dim=1))


__all__ = ["RAW_PANEL_CONFIG", "RAW_PANEL_NETWORK_VERSION", "RawPanelFeatures"]
