# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Independent MSLA-style encoder layer for RT-DETR-GMI.

This module is a clean-room implementation based on public algorithmic ideas:

* BCHW feature input/output.
* Four channel-split scale branches.
* Depthwise convolutions with 3x3, 5x5, 7x7, and 9x9 kernels for local
  multi-scale context extraction.
* Positive-kernel linear attention to avoid forming an HW-by-HW attention map.
* Adaptive fusion of the four scale features.

It intentionally does not import or depend on any previous MSLA implementation.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Iterable, List, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


__all__ = ("TransformerEncoderLayer_MSLA",)


def _make_four_way_channel_split(channels: int) -> List[int]:
    """Split channels into four non-empty groups while preserving the sum."""
    if channels < 4:
        raise ValueError(f"MSLA requires at least 4 channels for four scale branches, got {channels}.")
    base, remainder = divmod(channels, 4)
    return [base + (1 if i < remainder else 0) for i in range(4)]


def _activation(name: str | nn.Module | type[nn.Module]) -> nn.Module:
    """Create an activation module from a compact user-facing argument."""
    if isinstance(name, nn.Module):
        return deepcopy(name)
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name()
    key = str(name).lower()
    if key in {"gelu", "nn.gelu"}:
        return nn.GELU()
    if key in {"silu", "swish", "nn.silu"}:
        return nn.SiLU(inplace=True)
    if key in {"relu", "nn.relu"}:
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation for TransformerEncoderLayer_MSLA: {name!r}")


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of BCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class DepthwiseLocalContext(nn.Module):
    """Paper-style local branch: ReLU(depthwise_kxk(x) + x)."""

    def __init__(self, channels: int, kernel_size: int, act: nn.Module) -> None:
        super().__init__()
        del act  # Local MSLA extraction uses the paper-specified ReLU residual form.
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.depthwise(x) + x, inplace=False)


class PositiveKernelLinearAttention(nn.Module):
    """Linear spatial self-attention for BCHW tensors.

    For N = H*W positions, this computes:

        Q = phi(q), K = phi(k), phi(t)=elu(t)+1
        Y_i = Q_i (K^T V) / (Q_i sum_j K_j + eps)

    The operation avoids an explicit N x N attention matrix.
    """

    def __init__(self, channels: int, heads: int = 8, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = channels
        self.heads = max(1, math.gcd(channels, max(1, heads)))
        self.head_dim = channels // self.heads
        self.eps = eps
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x).reshape(b, 3, self.heads, self.head_dim, n)
        q, k, v = qkv.unbind(dim=1)  # each: B, heads, head_dim, N

        q = F.elu(q.transpose(-2, -1), alpha=1.0) + 1.0  # B, heads, N, head_dim
        k = F.elu(k.transpose(-2, -1), alpha=1.0) + 1.0
        v = v.transpose(-2, -1)

        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        normalizer = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)).unsqueeze(-1)
        out = torch.einsum("bhnd,bhde->bhne", q, kv) / normalizer.clamp_min(self.eps)
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        return self.proj(out)


class ScaleBranch(nn.Module):
    """One local-context plus efficient-attention scale branch."""

    def __init__(self, channels: int, kernel_size: int, heads: int, act: nn.Module, eps: float) -> None:
        super().__init__()
        self.local = DepthwiseLocalContext(channels, kernel_size, act)
        self.attention = PositiveKernelLinearAttention(channels, heads=heads, eps=eps)
        self.out_norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        local = self.local(x)
        return self.out_norm(local + self.attention(local))


class AdaptiveScaleFusion(nn.Module):
    """Softly reweight and project the four scale branches."""

    def __init__(self, channels: int, branch_channels: Sequence[int]) -> None:
        super().__init__()
        if len(branch_channels) != 4:
            raise ValueError("AdaptiveScaleFusion expects exactly four branches.")
        hidden = max(4, channels // 4)
        self.branch_channels = tuple(branch_channels)
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4, kernel_size=1, bias=True),
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, features: Iterable[Tensor]) -> Tensor:
        parts = list(features)
        if len(parts) != 4:
            raise ValueError(f"Expected four scale features, got {len(parts)}.")
        merged = torch.cat(parts, dim=1)
        weights = self.weight_net(merged).softmax(dim=1)
        weighted = [part * weights[:, i : i + 1] for i, part in enumerate(parts)]
        return self.norm(self.proj(torch.cat(weighted, dim=1)))


class MultiScaleLinearAttention(nn.Module):
    """Four-branch MSLA core for BCHW feature maps."""

    def __init__(
        self,
        channels: int,
        heads: int = 8,
        kernels: Sequence[int] = (3, 5, 7, 9),
        act: nn.Module | str | type[nn.Module] = "silu",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if tuple(kernels) != (3, 5, 7, 9):
            raise ValueError("This MSLA reimplementation uses the paper-specified kernels (3, 5, 7, 9).")
        branch_channels = _make_four_way_channel_split(channels)
        self.branch_channels = tuple(branch_channels)
        self.branches = nn.ModuleList(
            ScaleBranch(c, k, heads=heads, act=act, eps=eps) for c, k in zip(branch_channels, kernels)
        )
        self.fusion = AdaptiveScaleFusion(channels, branch_channels)

    def forward(self, x: Tensor) -> Tensor:
        parts = torch.split(x, self.branch_channels, dim=1)
        features = [branch(part) for branch, part in zip(self.branches, parts)]
        return self.fusion(features)


class ConvFeedForward(nn.Module):
    """Position-wise feed-forward network implemented with 1x1 convolutions."""

    def __init__(self, channels: int, hidden_channels: int, dropout: float, act: nn.Module | str | type[nn.Module]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            _activation(act),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerEncoderLayer_MSLA(nn.Module):
    """Ultralytics RT-DETR-compatible encoder layer using an independent MSLA core.

    Args:
        c1: Input/output channels. Ultralytics' YAML parser supplies this from
            the previous layer.
        cm: Hidden channels in the feed-forward network. RT-DETR-GMI currently
            passes 1024 for a 256-channel feature map.
        num_heads: Preferred number of linear-attention heads. Each branch uses
            the greatest divisor of its channel count not exceeding this value.
        dropout: Dropout probability for residual branches and FFN.
        act: Activation name or module.
        normalize_before: If True, use pre-norm transformer ordering; otherwise
            use post-norm ordering.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module | str | type[nn.Module] = "gelu",
        normalize_before: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.normalize_before = normalize_before
        self.norm1 = ChannelLayerNorm(c1, eps=eps)
        self.norm2 = ChannelLayerNorm(c1, eps=eps)
        self.msla = MultiScaleLinearAttention(c1, heads=num_heads, act="silu", eps=eps)
        self.ffn = ConvFeedForward(c1, cm, dropout=dropout, act=act)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _forward_post_norm(self, src: Tensor, pos: Optional[Tensor]) -> Tensor:
        q = src if pos is None else src + pos
        src = self.norm1(src + self.dropout1(self.msla(q)))
        src = self.norm2(src + self.dropout2(self.ffn(src)))
        return src

    def _forward_pre_norm(self, src: Tensor, pos: Optional[Tensor]) -> Tensor:
        q = self.norm1(src)
        q = q if pos is None else q + pos
        src = src + self.dropout1(self.msla(q))
        src = src + self.dropout2(self.ffn(self.norm2(src)))
        return src

    def forward(self, src: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        if src.ndim != 4:
            raise ValueError(f"TransformerEncoderLayer_MSLA expects BCHW input, got shape {tuple(src.shape)}.")
        if pos is not None and pos.shape != src.shape:
            raise ValueError(f"pos must have the same BCHW shape as src, got {tuple(pos.shape)} vs {tuple(src.shape)}.")
        if self.normalize_before:
            return self._forward_pre_norm(src, pos)
        return self._forward_post_norm(src, pos)
