"""CPU integration tests for the stage-two wiring, using explicit test doubles.

The tiny encoder/decoder/residual layers exercise real PyTorch autograd, but
are not the CUDA Mamba kernels or the full nnU-Net network. Passing these tests
does not establish GPU compatibility, full-patch memory use or training quality.
"""

import ast
import io
import math
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.nets.MambaUNetAttnRes import (
    AttnResMambaUNet,
    BlockAttnResMambaEncoder,
)


class AddDeltaTestDouble(nn.Module):
    """Controlled residual increment, exposing group inputs for semantic checks."""

    def __init__(self, delta):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(float(delta)))
        self.last_input = None

    def forward(self, tokens):
        self.last_input = tokens.detach().clone()
        return tokens + self.delta


class TinyResidualTestDouble(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.transform = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, tokens):
        return tokens + self.transform(tokens)


class TinyEncoderTestDouble(nn.Module):
    def __init__(self, spatial_dims):
        super().__init__()
        self.conv_op = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.output_channels = [3, 4]
        self.shallow = self.conv_op(1, 3, 3, padding=1)
        self.bottleneck = self.conv_op(3, 4, 3, stride=2, padding=1)

    def forward(self, image):
        shallow = self.shallow(image).tanh()
        return [shallow, self.bottleneck(shallow)]

    def compute_conv_feature_map_size(self, input_size):
        return 3 * math.prod(input_size)


class TinyDecoderTestDouble(nn.Module):
    def __init__(self, spatial_dims, deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.coarse_head = conv(4, 2, 1)
        self.shallow_head = conv(3, 2, 1)

    def forward(self, skips):
        coarse = self.coarse_head(skips[-1])
        full = self.shallow_head(skips[0]) + F.interpolate(
            coarse, size=skips[0].shape[2:], mode="nearest"
        )
        return [full, coarse] if self.deep_supervision else full

    def compute_conv_feature_map_size(self, input_size):
        return 2 * math.prod(input_size)


class TinyBaseModelTestDouble(nn.Module):
    def __init__(self, spatial_dims, deep_supervision):
        super().__init__()
        self.encoder = TinyEncoderTestDouble(spatial_dims)
        self.decoder = TinyDecoderTestDouble(spatial_dims, deep_supervision)
        self.mamba_mlp = nn.Module()
        self.mamba_mlp.layers = nn.ModuleList(
            [TinyResidualTestDouble(4) for _ in range(4)]
        )
        self.mamba_mlp.last_norm = nn.LayerNorm(4)


def make_tiny_model(spatial_dims=2, deep_supervision=False,
                    boundary_guidance=False, auxiliary_boundary=False):
    base = TinyBaseModelTestDouble(spatial_dims, deep_supervision)
    return AttnResMambaUNet(
        base, hidden_size=4, spatial_dims=spatial_dims, block_size=2,
        boundary_guidance=boundary_guidance,
        auxiliary_boundary=auxiliary_boundary,
        boundary_hidden_channels=4,
    )


class TestBlockAttnResMambaEncoder(unittest.TestCase):
    def test_history_contains_group_deltas_not_cumulative_group_outputs(self):
        layers = nn.ModuleList([AddDeltaTestDouble(v) for v in (1, 2, 4, 8)])
        network = BlockAttnResMambaEncoder(layers, nn.Identity(), 3, block_size=2)
        observed_histories = []

        def record_history(module, inputs):
            observed_histories.append([source.detach().clone() for source in inputs[0]])

        handles = [aggregator.register_forward_pre_hook(record_history)
                   for aggregator in network.aggregators]
        try:
            output = network(torch.full((2, 5, 3), 10.0))
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual([len(history) for history in observed_histories], [1, 2, 3])
        expected_sources = (10.0, 3.0, 12.0)
        for source, expected in zip(observed_histories[-1], expected_sources):
            torch.testing.assert_close(source, torch.full_like(source, expected))
        torch.testing.assert_close(layers[0].last_input, torch.full_like(output, 10.0))
        torch.testing.assert_close(layers[2].last_input, torch.full_like(output, 6.5))
        torch.testing.assert_close(output, torch.full_like(output, 25.0 / 3.0))

    def test_final_short_group_is_included_and_history_resets_per_forward(self):
        layers = nn.ModuleList([AddDeltaTestDouble(v) for v in (1, 2, 4)])
        network = BlockAttnResMambaEncoder(layers, nn.Identity(), 2, block_size=2)
        self.assertEqual(len(network.aggregators), 3)
        first = network(torch.full((1, 3, 2), 10.0))
        second = network(torch.full((1, 3, 2), 20.0))
        torch.testing.assert_close(first, torch.full_like(first, 17.0 / 3.0))
        torch.testing.assert_close(second, torch.full_like(second, 9.0))

    def test_final_normalization_is_applied_after_aggregation(self):
        torch.manual_seed(81)
        layers = nn.ModuleList([AddDeltaTestDouble(v) for v in (1, 2, 3)])
        norm = nn.LayerNorm(4)
        network = BlockAttnResMambaEncoder(layers, norm, 4, block_size=2)
        tokens = torch.randn(2, 5, 4)
        expected = norm((tokens + 3.0 + 3.0) / 3.0)
        torch.testing.assert_close(network(tokens), expected)

    def test_real_gradients_reach_layers_tokens_and_depth_queries(self):
        torch.manual_seed(82)
        layers = nn.ModuleList([TinyResidualTestDouble(4) for _ in range(4)])
        network = BlockAttnResMambaEncoder(layers, nn.LayerNorm(4), 4, block_size=2)
        tokens = torch.randn(2, 7, 4, requires_grad=True)
        objective = (network(tokens) * torch.randn_like(tokens)).sum()
        objective.backward()
        for value in [tokens] + [layer.transform[-1].weight for layer in layers]:
            self.assertIsNotNone(value.grad)
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
            self.assertGreater(value.grad.abs().sum().item(), 0.0)
        # The first aggregator has only one source: its query has zero gradient
        # by construction. Later queries must receive a learning signal.
        for aggregator in network.aggregators[1:]:
            self.assertIsNotNone(aggregator.query.grad)
            self.assertGreater(aggregator.query.grad.abs().sum().item(), 0.0)

    def test_guided_encoder_rejects_missing_or_misaligned_boundary(self):
        layers = nn.ModuleList([TinyResidualTestDouble(4) for _ in range(2)])
        network = BlockAttnResMambaEncoder(
            layers, nn.Identity(), 4, block_size=1, boundary_guidance=True
        )
        tokens = torch.randn(2, 5, 4)
        with self.assertRaises(ValueError):
            network(tokens)
        with self.assertRaises(ValueError):
            network(tokens, torch.rand(2, 6, 1))


class TestAttnResMambaUNet(unittest.TestCase):
    def test_original_encoder_decoder_layers_and_normalization_are_reused(self):
        base = TinyBaseModelTestDouble(2, False)
        old_layers = list(base.mamba_mlp.layers)
        old_norm = base.mamba_mlp.last_norm
        model = AttnResMambaUNet(base, 4, 2, block_size=2)
        self.assertIs(model.encoder, base.encoder)
        self.assertIs(model.decoder, base.decoder)
        self.assertIs(model.mamba_mlp.last_norm, old_norm)
        for original, wrapped in zip(old_layers, model.mamba_mlp.layers):
            self.assertIs(original, wrapped)

    def test_tensor_and_deep_supervision_list_structure_survives_2d_and_3d(self):
        for spatial_dims, spatial in ((2, (8, 10)), (3, (4, 6, 8))):
            for deep_supervision in (False, True):
                with self.subTest(spatial_dims=spatial_dims,
                                  deep_supervision=deep_supervision):
                    model = make_tiny_model(spatial_dims, deep_supervision)
                    image = torch.randn(2, 1, *spatial, requires_grad=True)
                    output = model(image)
                    if deep_supervision:
                        self.assertIsInstance(output, list)
                        self.assertEqual(len(output), 2)
                        full, coarse = output
                        self.assertEqual(coarse.shape, (2, 2, *(size // 2 for size in spatial)))
                    else:
                        self.assertIsInstance(output, torch.Tensor)
                        full = output
                    self.assertEqual(full.shape, (2, 2, *spatial))
                    full.square().mean().backward()
                    self.assertIsNotNone(image.grad)
                    self.assertTrue(bool(torch.isfinite(image.grad).all()))
                    self.assertGreater(image.grad.abs().sum().item(), 0.0)
                    self.assertIsNone(model.boundary_logits)

    def test_guided_model_boundary_cache_is_consumed_and_eval_cleared(self):
        for spatial_dims, spatial in ((2, (8, 10)), (3, (4, 6, 8))):
            with self.subTest(spatial_dims=spatial_dims):
                model = make_tiny_model(spatial_dims, boundary_guidance=True,
                                        auxiliary_boundary=True)
                image = torch.randn(2, 1, *spatial)
                model.train()
                model(image)
                logits = model.boundary_logits
                self.assertIsNotNone(logits)
                self.assertEqual(logits.shape, (2, 1, *spatial))
                self.assertTrue(logits.requires_grad)
                self.assertIs(model.consume_boundary_logits(), logits)
                self.assertIsNone(model.boundary_logits)
                self.assertIsNone(model.consume_boundary_logits())
                model(image)
                self.assertIsNotNone(model.boundary_logits)
                model.eval()
                with torch.no_grad():
                    prediction = model(image)
                self.assertEqual(prediction.shape, (2, 2, *spatial))
                self.assertIsNone(model.boundary_logits)

    def test_auxiliary_only_model_has_boundary_logits_without_guided_queries(self):
        model = make_tiny_model(auxiliary_boundary=True)
        output = model(torch.randn(2, 1, 8, 10))
        logits = model.consume_boundary_logits()
        self.assertEqual(logits.shape, (2, 1, 8, 10))
        for aggregator in model.mamba_mlp.aggregators:
            self.assertIsNone(aggregator.boundary_query)
        (output.square().mean() + logits.square().mean()).backward()
        boundary_parameters = [parameter for name, parameter in model.named_parameters()
                               if "boundary" in name and parameter.grad is not None]
        self.assertTrue(boundary_parameters)
        self.assertTrue(any(parameter.grad.abs().sum().item() > 0
                            for parameter in boundary_parameters))

    def test_learned_boundary_query_connects_segmentation_loss_to_predictor(self):
        torch.manual_seed(83)
        model = make_tiny_model(boundary_guidance=True, auxiliary_boundary=True)
        for aggregator in model.mamba_mlp.aggregators:
            with torch.no_grad():
                aggregator.boundary_query.copy_(torch.tensor([0.3, -0.2, 0.1, 0.4]))
        output = model(torch.randn(2, 1, 8, 10))
        logits = model.consume_boundary_logits()
        logits.retain_grad()
        # Deliberately exclude the auxiliary loss: this isolates the guidance
        # path after the initially zero boundary queries have learned.
        (output * torch.randn_like(output)).sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_guided_checkpoint_round_trip_and_eval_does_not_require_labels(self):
        torch.manual_seed(84)
        original = make_tiny_model(3, True, True, True).eval()
        image = torch.randn(1, 1, 4, 6, 8)
        with torch.no_grad():
            expected = original(image)
        buffer = io.BytesIO()
        torch.save(original.state_dict(), buffer)
        buffer.seek(0)
        restored = make_tiny_model(3, True, True, True).eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        with torch.no_grad():
            actual = restored(image)
        self.assertEqual(len(actual), len(expected))
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor)
        self.assertIsNone(restored.boundary_logits)

    def test_frozen_baseline_mha_constructor_keeps_original_axis_setting(self):
        baseline_path = Path(__file__).resolve().parents[2] / "nets" / "MambaUNet.py"
        tree = ast.parse(baseline_path.read_text(encoding="utf-8-sig"))
        layer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == "MambaMLPLayer")
        constructors = [node for node in ast.walk(layer)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "MHA"]
        self.assertEqual(len(constructors), 1)
        options = {keyword.arg: keyword.value for keyword in constructors[0].keywords}
        self.assertTrue("batch_first" not in options or
                        (isinstance(options["batch_first"], ast.Constant)
                         and options["batch_first"].value is False))


if __name__ == "__main__":
    unittest.main()
