# MSLA mechanism adopted from MSLAU-Net.
# Original repository: Monsoon49/MSLAU-Net (https://github.com/Monsoon49/MSLAU-Net).
# Commit: efdb058ebe363434376c8ef6cbf689391b60b7fc.
# Reference: Lan et al., MSLAU-Net.
# TransformerEncoderLayer_MSLA is the RT-DETR integration used in this study.

# table5_k3333: 4 branch(es), kernels=(3, 3, 3, 3).
# Provenance: reconstructed from the archived final experimental implementation
# and the recorded branch-number modification rule.

"""Experimental MSLA encoder extracted from the supplied training source.

Only unrelated experimental modules and unused imports were removed. The five
class definitions, parameter names, and forward computations are preserved.

The MSLA mechanism follows MSLAU-Net (Monsoon49/MSLAU-Net); the RT-DETR encoder
wrapper is retained from the supplied experimental source.

Compatibility notes:
* TransformerEncoderLayer_MSLA accepts BCHW feature maps.
* The original MSLA token reshape assumes square feature maps (H == W).
* c1 must be divisible by 4; c1 // 4 must be divisible by num_heads.
* The original forward always uses post-normalization. normalize_before,
  src_mask, src_key_padding_mask, and pos are retained for interface compatibility
  but are not used to change the computation.
* Direct reshape operations and shared attention are intentionally preserved
  for experimental reproducibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LayerNorm",
    "LinearAttention",
    "DepthwiseConv",
    "MSLA",
    "TransformerEncoderLayer_MSLA",
]

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.view(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)

        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        key = F.softmax(k, dim=-1)
        query = F.softmax(q, dim=-2)
        context = key.transpose(-2, -1) @ v
        x = (query @ context).reshape(b, h * w, c)

        x = self.proj(x)

        x = x.permute(0, 2, 1).view(b, c, h, w)

        return x

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = x + residual
        x = self.relu(x)
        return x

class MSLA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.dw_conv_3x3 = DepthwiseConv(dim // 4, kernel_size=3)
        self.dw_conv_5x5 = DepthwiseConv(dim // 4, kernel_size=3)
        self.dw_conv_7x7 = DepthwiseConv(dim // 4, kernel_size=3)
        self.dw_conv_9x9 = DepthwiseConv(dim // 4, kernel_size=3)

        self.linear_attention = LinearAttention(dim = dim // 4, num_heads = num_heads)

        self.final_conv = nn.Conv2d(dim, dim, 1)

        self.scale_weights = nn.Parameter(torch.ones(4), requires_grad=True)

    def forward(self, input_):
        b, n, c = input_.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)

        input_reshaped = input_.reshape([b, c, h, w])

        split_size = c // 4
        x_3x3 = input_reshaped[:, :split_size, :, :]
        x_5x5 = input_reshaped[:, split_size:2 * split_size, :, :]
        x_7x7 = input_reshaped[:, 2 * split_size:3 * split_size:, :, :]
        x_9x9 = input_reshaped[:, 3 * split_size:, :, :]

        x_3x3 = self.dw_conv_3x3(x_3x3)
        x_5x5 = self.dw_conv_5x5(x_5x5)
        x_7x7 = self.dw_conv_7x7(x_7x7)
        x_9x9 = self.dw_conv_9x9(x_9x9)

        att_3x3 = self.linear_attention(x_3x3)
        att_5x5 = self.linear_attention(x_5x5)
        att_7x7 = self.linear_attention(x_7x7)
        att_9x9 = self.linear_attention(x_9x9)

        processed_input = torch.cat([
            att_3x3 * self.scale_weights[0],
            att_5x5 * self.scale_weights[1],
            att_7x7 * self.scale_weights[2],
            att_9x9 * self.scale_weights[3]
        ], dim=1)

        final_output = self.final_conv(processed_input)

        output_reshaped = final_output.reshape(b, n, self.dim)


        return output_reshaped

class TransformerEncoderLayer_MSLA(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        self.assa = MSLA(c1, num_heads=num_heads)
        # Implementation of Feedforward model
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)

        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        BS, C, H, W = src.size()
        src2 = self.assa(src.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
