"""CUDA launch graph for the existing frozen actor, without a second model."""
from collections.abc import Callable

import torch as th

from ai.rl.device import require_cuda_device


class FrozenActorGraph:
    """Capture actions and validity; the caller owns invalidation and copies."""

    @th.no_grad()
    def __init__(
        self,
        actor: Callable[[th.Tensor], tuple[th.Tensor, th.Tensor]],
        observation: th.Tensor,
    ):
        require_cuda_device(observation.device)
        self.input = observation.clone()
        current = th.cuda.current_stream(observation.device)
        stream = th.cuda.Stream(device=observation.device)
        stream.wait_stream(current)
        with th.cuda.stream(stream):
            for _ in range(3):
                actor(self.input)
        current.wait_stream(stream)
        self.graph = th.cuda.CUDAGraph()
        with th.cuda.graph(self.graph, stream=stream):
            self.outputs = actor(self.input)
        current.wait_stream(stream)

    @th.no_grad()
    def __call__(self, observation: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        self.input.copy_(observation)
        self.graph.replay()
        return self.outputs


__all__ = ["FrozenActorGraph"]
