"""Local cross-scale skip fusion. Pure PyTorch; no CUDA Mamba fallback."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveMultiScaleSkip(nn.Module):
    """Decoder-conditioned, voxel-wise softmax over adjacent encoder scales.

    Project to a bounded width before upsampling; reduce spatial axes before
    projection when downsampling. This also handles anisotropic/odd shapes.
    A zero output projection makes the residual exactly zero initially; its
    first update opens gradients to the source projections and scoring paths.
    """

    def __init__(self, channels, target_index, spatial_dims=3, width=16):
        super().__init__()
        if spatial_dims not in (2, 3) or width < 1:
            raise ValueError('expected 2D/3D and positive fusion width')
        if len(channels) < 2 or not 0 <= target_index < len(channels):
            raise ValueError('need at least two non-bottleneck encoder scales')
        self.target_index = target_index
        self.source_indices = tuple(range(max(0, target_index - 1),
                                          min(len(channels), target_index + 2)))
        self.width = min(width, channels[target_index])
        self.spatial_dims = spatial_dims
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.projections = nn.ModuleList([
            conv(channels[i], self.width, 1, bias=False) for i in self.source_indices
        ])
        self.query = conv(channels[target_index], self.width, 1, bias=False)
        self.keys = nn.ModuleList([conv(self.width, self.width, 1, bias=False)
                                   for _ in self.source_indices])
        self.output = conv(self.width, channels[target_index], 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def _align(self, source, projection, size):
        # First shrink any oversized axes (including mixed anisotropic shapes).
        reduced = tuple(min(a, b) for a, b in zip(source.shape[2:], size))
        if tuple(source.shape[2:]) != reduced:
            pool = F.adaptive_avg_pool2d if self.spatial_dims == 2 else F.adaptive_avg_pool3d
            source = pool(source, reduced)
        source = projection(source)
        if tuple(source.shape[2:]) != tuple(size):
            source = F.interpolate(source, size=size,
                                   mode='bilinear' if self.spatial_dims == 2 else 'trilinear',
                                   align_corners=False)
        return source

    def forward(self, skips, decoder_context, return_weights=False):
        target = skips[self.target_index]
        if target.ndim != self.spatial_dims + 2 or decoder_context.shape != target.shape:
            raise ValueError('decoder context must match the target skip shape')
        values = [self._align(skips[i], proj, target.shape[2:])
                  for i, proj in zip(self.source_indices, self.projections)]
        # FP32 scoring/accumulation under AMP, without stacking channel volumes.
        query = self.query(decoder_context).float()
        scores = torch.cat([(query * key(value).float()).sum(1, keepdim=True)
                            / math.sqrt(self.width)
                            for key, value in zip(self.keys, values)], dim=1)
        weights = scores.softmax(dim=1)
        fused = sum(value.float() * weights[:, i:i+1] for i, value in enumerate(values))
        output = target + self.output(fused.to(target.dtype))
        return (output, weights) if return_weights else output


class MultiScaleSkipDecoder(nn.Module):
    """Reuse all original decoder modules; replace only the skip input path."""

    def __init__(self, decoder, spatial_dims=3, width=16):
        super().__init__()
        self.encoder = decoder.encoder
        self.num_classes = decoder.num_classes
        self.deep_supervision = decoder.deep_supervision
        self.stages = decoder.stages
        self.transpconvs = decoder.transpconvs
        self.seg_layers = decoder.seg_layers
        channels = self.encoder.output_channels[:-1]
        self.skip_fusions = nn.ModuleList([
            AdaptiveMultiScaleSkip(channels, i, spatial_dims, width)
            for i in range(len(channels))
        ])

    def forward(self, skips):
        current = skips[-1]
        outputs = []
        for stage in range(len(self.stages)):
            up = self.transpconvs[stage](current)
            index = len(skips) - stage - 2
            skip = self.skip_fusions[index](skips, up)
            current = self.stages[stage](torch.cat((up, skip), dim=1))
            if self.deep_supervision or stage == len(self.stages) - 1:
                outputs.append(self.seg_layers[stage](current))
        return outputs[::-1] if self.deep_supervision else outputs[-1]

    def compute_conv_feature_map_size(self, input_size):
        # Conservative accounting for added conv outputs, not a GPU memory or
        # complete operator-FLOPs estimator. Original decoder method is reused.
        from nnunetv2.nets.MambaUNet import UNetResDecoder
        baseline = UNetResDecoder.compute_conv_feature_map_size(self, input_size)
        shape = list(input_size)
        extra = 0
        for i, stride in enumerate(self.encoder.strides[:-1]):
            shape = [a // b for a, b in zip(shape, stride)]
            fusion = self.skip_fusions[i]
            extra += math.prod(shape) * (fusion.width * (2 * len(fusion.source_indices) + 1)
                                        + self.encoder.output_channels[i])
        return baseline + extra
