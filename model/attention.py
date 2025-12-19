import numpy as np
import torch
from torch import nn, nn as nn
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Max and average pooling along the channel dimension
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # Shape: (batch, 1, h, w)
        avg_result = torch.mean(x, dim=1, keepdim=True)    # Shape: (batch, 1, h, w)
        result = torch.cat([max_result, avg_result], dim=1)  # Shape: (batch, 2, h, w)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        out = x * self.ca(x)  # Apply Channel Attention
        out = out * self.sa(out)  # Apply Spatial Attention
        return out + residual  # Residual connection



class MultiScaleAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], reduction_ratio=8):
        super(MultiScaleAttentionBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)  # Add LeakyReLU activation
            )
            for k in kernel_sizes
        ])

        self.attention = CBAMBlock(out_channels, reduction=reduction_ratio)
        self.fusion = nn.Sequential(
            nn.Conv2d(len(kernel_sizes) * out_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)  # Add LeakyReLU to fusion
        )


    def forward(self, x):
        # Apply convolutions of different kernel sizes
        multi_scale_features = [conv(x) for conv in self.convs]
        combined = torch.cat(multi_scale_features, dim=1)  # Concatenate along channel dimension
        fused = self.fusion(combined)  # Fuse multi-scale features
        attended = self.attention(fused)  # Apply CBAM attention
        return attended


class ConvAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, reduction_ratio=8, attention_kernel=7):
        super(ConvAttentionBlock, self).__init__()

        # Single convolution with the specified kernel size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)  # Add LeakyReLU activation
        )

        # CBAM attention mechanism
        self.attention = CBAMBlock(out_channels, reduction=reduction_ratio, kernel_size=attention_kernel)

    def forward(self, x):
        # Apply convolution
        features = self.conv(x)

        # Apply CBAM attention
        attended = self.attention(features)
        return attended


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=16, num_heads=4, dropout_prob=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Input shape: (batch_size, channels=16, frames=11, height=15, width=15)
        batch_size, channels, frames, height, width = x.shape

        # Flatten spatial dimensions for attention
        x = x.view(batch_size, channels, frames * height * width).permute(0, 2, 1)  # (batch_size, seq_len, embed_dim)

        # Apply attention
        attended, _ = self.attention(x, x, x)  # Self-attention
        attended = self.norm(attended)
        attended = self.dropout(attended)

        # Reshape back to original dimensions
        attended = attended.permute(0, 2, 1).view(batch_size, channels, frames, height, width)

        return attended

class TemporalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_position=50):
        super(TemporalPositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position + 1, d_model)

    def forward(self, cumulative_positions):
        embeddings = self.position_embeddings(cumulative_positions)  # Shape: (frames, d_model)
        return embeddings