"""Depth-source attention for the stage-two Hetero-UNet experiments.

This module deliberately has no Mamba or nnU-Net imports so its axes, numerical
behavior and gradients can be checked on a CPU. It consumes a history supplied
by the caller; the network decides which group residuals belong in that history.
"""

import math
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


class DepthAttentionAggregator(nn.Module):
    """Aggregate depth sources without mixing samples or spatial tokens.

    Each history tensor has shape ``[batch, tokens, channels]``. For source ``s``:

    ``k_s = h_s / sqrt(mean_channels(h_s ** 2) + eps)``
    ``score_s = dot(query, k_s) / sqrt(channels)``

    When boundary conditioning is enabled, add
    ``boundary * dot(boundary_query, k_s) / sqrt(channels)``. The boundary map is
    ``[batch, tokens, 1]`` (or ``[batch, tokens]``) with probabilities in [0, 1].
    Its effect depends on each source's key; adding the same scalar to every
    source score would cancel in softmax and is intentionally avoided here.

    Softmax is over *sources* only. Values are the original, unnormalized
    history tensors. Both queries start at zero, so initialization is a uniform
    average, including when a boundary map is supplied. This is an average,
    not an identity mapping or an unscaled residual sum.

    Only the small ``[batch, tokens, sources]`` score/weight tensor is stacked.
    The weighted sum is streamed, avoiding a ``[sources, batch, tokens,
    channels]`` history copy. Half/bfloat16 inputs use float32 keys, softmax and
    accumulation, and return their original dtype.

    ``validate_finite`` is intended for tests and diagnostic runs. It also
    checks boundary probability bounds, but is disabled by default because
    tensor-to-bool checks synchronize the GPU on every call.
    """

    def __init__(
        self,
        hidden_size: int,
        boundary_conditioned: bool = False,
        eps: float = 1e-6,
        validate_finite: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.hidden_size = hidden_size
        self.boundary_conditioned = boundary_conditioned
        self.eps = eps
        self.validate_finite = validate_finite
        self.scale = hidden_size ** -0.5
        self.query = nn.Parameter(torch.zeros(hidden_size))
        if boundary_conditioned:
            self.boundary_query = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("boundary_query", None)

    def _validate_inputs(
        self, history: Sequence[Tensor], boundary: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if isinstance(history, Tensor) or not isinstance(history, (tuple, list)) or not history:
            raise ValueError("history must be a nonempty list or tuple of [B, L, C] tensors")
        first = history[0]
        if not isinstance(first, Tensor) or first.ndim != 3:
            raise ValueError("each history source must have shape [B, L, C]")
        if min(first.shape) < 1 or first.shape[-1] != self.hidden_size:
            raise ValueError("history must have nonempty dimensions and the configured channel count")
        if not first.is_floating_point():
            raise TypeError("history sources must have a floating-point dtype")
        if first.device != self.query.device:
            raise ValueError("history and aggregator parameters must be on the same device")
        for source in history:
            if not isinstance(source, Tensor) or source.shape != first.shape:
                raise ValueError("all history sources must have the same [B, L, C] shape")
            if source.device != first.device or source.dtype != first.dtype:
                raise ValueError("all history sources must have the same device and dtype")
            if self.validate_finite and not bool(torch.isfinite(source).all()):
                raise ValueError("history sources must contain only finite values")

        if self.boundary_conditioned:
            if boundary is None:
                raise ValueError("boundary conditioning requires a boundary probability map")
            if not isinstance(boundary, Tensor):
                raise TypeError("boundary must be a tensor")
            if boundary.shape == first.shape[:2]:
                boundary = boundary.unsqueeze(-1)
            if boundary.shape != (*first.shape[:2], 1):
                raise ValueError("boundary must have shape [B, L, 1] or [B, L]")
            if boundary.device != first.device or not boundary.is_floating_point():
                raise ValueError("boundary must be floating point on the history device")
            if self.validate_finite and (
                not bool(torch.isfinite(boundary).all())
                or bool(((boundary < 0) | (boundary > 1)).any())
            ):
                raise ValueError("boundary probabilities must be finite and in [0, 1]")
        elif boundary is not None:
            raise ValueError("boundary was supplied to an unconditioned aggregator")
        return first, boundary

    def forward(
        self,
        history: Sequence[Tensor],
        boundary: Optional[Tensor] = None,
        return_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        first, boundary = self._validate_inputs(history, boundary)
        compute_dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
        query = self.query.to(dtype=compute_dtype)
        if self.boundary_conditioned:
            boundary_query = self.boundary_query.to(dtype=compute_dtype)
            boundary = boundary.to(dtype=compute_dtype).squeeze(-1)

        scores = []
        for source in history:
            key = source.to(dtype=compute_dtype)
            key = key * torch.rsqrt(key.square().mean(dim=-1, keepdim=True) + self.eps)
            score = (key * query).sum(dim=-1) * self.scale
            if self.boundary_conditioned:
                score = score + boundary * (key * boundary_query).sum(dim=-1) * self.scale
            scores.append(score)
        logits = torch.stack(scores, dim=-1)
        if self.validate_finite and not bool(torch.isfinite(logits).all()):
            raise ValueError("depth attention logits are not finite")
        weights = torch.softmax(logits, dim=-1)
        output = torch.zeros_like(first, dtype=compute_dtype)
        for index, source in enumerate(history):
            output = output + weights[..., index:index + 1] * source.to(dtype=compute_dtype)
        output = output.to(dtype=first.dtype)
        if return_weights:
            return output, weights
        return output
