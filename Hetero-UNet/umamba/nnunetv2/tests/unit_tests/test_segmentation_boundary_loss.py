"""CPU checks for auxiliary-loss wiring, separate from CUDA/Mamba validation."""

import gc
import unittest
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.tests.unit_tests.test_attnres_network import make_tiny_model
from nnunetv2.training.loss.boundary import semantic_boundary_target
from nnunetv2.training.loss.segmentation_boundary import SegmentationBoundaryLoss


class RecordingSegmentationLoss(nn.Module):
    """Simple shape-aware stand-in for nnU-Net's segmentation loss.

    A zero scale isolates the auxiliary gradient path. Recording identities
    rather than tensors avoids keeping additional computation graphs alive.
    """

    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale
        self.last_output_id = None
        self.last_target_id = None

    def forward(self, output, target):
        self.last_output_id = id(output)
        self.last_target_id = id(target)
        predictions = output if isinstance(output, (tuple, list)) else [output]
        targets = target if isinstance(target, (tuple, list)) else [target]
        if len(predictions) != len(targets):
            raise ValueError("deep supervision predictions and targets must match")
        result = predictions[0].new_zeros(())
        for index, (prediction, reference) in enumerate(zip(predictions, targets)):
            if (prediction.shape[0] != reference.shape[0] or reference.shape[1] != 1
                    or prediction.shape[2:] != reference.shape[2:]):
                raise ValueError("segmentation target shape mismatch")
            result = result + self.scale * (.5 ** index) * prediction.square().mean()
        return result


def make_target(output):
    """Supply semantic targets at all decoder scales, finest first."""
    full = output[0] if isinstance(output, (tuple, list)) else output
    target = torch.zeros(full.shape[0], 1, *full.shape[2:])
    target[..., target.shape[-1] // 2:] = 1.
    if isinstance(output, (tuple, list)):
        return [F.interpolate(target, size=item.shape[2:], mode="nearest") for item in output]
    return target


class SegmentationBoundaryLossTests(unittest.TestCase):
    def test_tensor_list_and_tuple_contract_passes_original_objects_to_segmentation_loss(self):
        for spatial_dims, spatial in ((2, (8, 10)), (3, (4, 6, 8))):
            for structure in ("tensor", "list", "tuple"):
                with self.subTest(spatial_dims=spatial_dims, structure=structure):
                    model = make_tiny_model(
                        spatial_dims, deep_supervision=structure != "tensor",
                        boundary_guidance=True, auxiliary_boundary=True,
                    )
                    segmentation = RecordingSegmentationLoss()
                    criterion = SegmentationBoundaryLoss(segmentation, model, boundary_weight=.25)
                    output = model(torch.randn(2, 1, *spatial))
                    target = make_target(output)
                    if structure == "tuple":
                        output, target = tuple(output), tuple(target)
                    logits = model.boundary_logits
                    expected_segmentation = segmentation(output, target)
                    finest_target = target[0] if isinstance(target, (tuple, list)) else target
                    expected_boundary = criterion.boundary_loss(logits, finest_target)
                    loss = criterion(output, target)
                    torch.testing.assert_close(loss, expected_segmentation + .25 * expected_boundary)
                    self.assertEqual(segmentation.last_output_id, id(output))
                    self.assertEqual(segmentation.last_target_id, id(target))
                    self.assertIsNone(model.boundary_logits)
                    self.assertEqual(loss.ndim, 0)

    def test_auxiliary_gradients_reach_predictor_and_shallow_encoder_at_zero_guidance_queries(self):
        torch.manual_seed(101)
        model = make_tiny_model(boundary_guidance=True, auxiliary_boundary=True)
        for aggregator in model.mamba_mlp.aggregators:
            torch.testing.assert_close(
                aggregator.boundary_query, torch.zeros_like(aggregator.boundary_query)
            )
        criterion = SegmentationBoundaryLoss(RecordingSegmentationLoss(scale=0.), model)
        output = model(torch.randn(2, 1, 8, 10))
        logits = model.boundary_logits
        logits.retain_grad()
        loss = criterion(output, make_target(output))
        loss.backward()
        for gradient in (logits.grad, model.boundary_head.output.weight.grad,
                         model.encoder.shallow.weight.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_ignore_handling_is_delegated_and_masked_logits_have_zero_gradient(self):
        torch.manual_seed(102)
        model = make_tiny_model(auxiliary_boundary=True)
        criterion = SegmentationBoundaryLoss(
            RecordingSegmentationLoss(scale=0.), model, boundary_weight=.2, ignore_label=255
        )
        output = model(torch.randn(2, 1, 8, 10))
        logits = model.boundary_logits
        logits.retain_grad()
        target = make_target(output)
        target[..., 3:5] = 255.
        _, mask = semantic_boundary_target(target, ignore_label=255)
        expected = .2 * criterion.boundary_loss(logits, target)
        loss = criterion(output, target)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
        self.assertGreater(logits.grad[mask].abs().sum().item(), 0.)

    def test_all_ignored_batch_returns_finite_differentiable_zero(self):
        model = make_tiny_model(auxiliary_boundary=True)
        criterion = SegmentationBoundaryLoss(
            RecordingSegmentationLoss(scale=0.), model, ignore_label=255
        )
        output = model(torch.randn(2, 1, 8, 10))
        logits = model.boundary_logits
        logits.retain_grad()
        target = torch.full((2, 1, 8, 10), 255.)
        loss = criterion(output, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))

    def test_cache_is_consumed_and_logging_values_are_detached(self):
        model = make_tiny_model(boundary_guidance=True, auxiliary_boundary=True)
        criterion = SegmentationBoundaryLoss(RecordingSegmentationLoss(), model)
        output = model(torch.randn(2, 1, 8, 10))
        logits = model.boundary_logits
        self.assertIsNotNone(logits.grad_fn)
        loss = criterion(output, make_target(output))
        self.assertIsNone(model.boundary_logits)
        for diagnostic in (criterion.last_segmentation_loss, criterion.last_boundary_loss):
            self.assertFalse(diagnostic.requires_grad)
            self.assertIsNone(diagnostic.grad_fn)
        loss.backward()

    def test_eval_mode_uses_segmentation_only(self):
        model = make_tiny_model(boundary_guidance=True, auxiliary_boundary=True).eval()
        segmentation = RecordingSegmentationLoss()
        criterion = SegmentationBoundaryLoss(segmentation, model, boundary_weight=100.)
        with torch.no_grad():
            output = model(torch.randn(2, 1, 8, 10))
            target = make_target(output)
            expected = segmentation(output, target)
            loss = criterion(output, target)
        torch.testing.assert_close(loss, expected)
        self.assertEqual(criterion.last_boundary_loss.item(), 0.)
        self.assertIsNone(model.boundary_logits)

    def test_training_without_fresh_boundary_cache_raises(self):
        model = make_tiny_model(auxiliary_boundary=True)
        criterion = SegmentationBoundaryLoss(RecordingSegmentationLoss(), model)
        output = model(torch.randn(2, 1, 8, 10))
        target = make_target(output)
        criterion(output, target)
        with self.assertRaisesRegex(RuntimeError, "training boundary logits missing"):
            criterion(output, target)

    def test_network_is_not_registered_under_loss_or_kept_alive(self):
        model = make_tiny_model(auxiliary_boundary=True)
        criterion = SegmentationBoundaryLoss(RecordingSegmentationLoss(), model)
        self.assertIs(criterion._network_ref(), model)
        self.assertFalse(any(module is model for module in criterion.modules()))
        model_ids = {id(parameter) for parameter in model.parameters()}
        self.assertTrue(model_ids.isdisjoint(id(parameter) for parameter in criterion.parameters()))
        self.assertFalse(any("encoder" in name or "mamba_mlp" in name
                             for name in criterion.state_dict()))
        reference = weakref.ref(model)
        del model
        gc.collect()
        self.assertIsNone(reference())
        with self.assertRaisesRegex(RuntimeError, "network no longer exists"):
            criterion(torch.zeros(1, 2, 4, 4), torch.zeros(1, 1, 4, 4))

    def test_invalid_boundary_weights_are_rejected(self):
        model = make_tiny_model(auxiliary_boundary=True)
        for weight in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                SegmentationBoundaryLoss(RecordingSegmentationLoss(), model, boundary_weight=weight)


if __name__ == "__main__":
    unittest.main()
