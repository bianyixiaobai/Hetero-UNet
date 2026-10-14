"""CPU-only boundary branch tests; no Mamba/CUDA dependency is required."""

import unittest

import torch
from torch.nn import functional as F

from nnunetv2.nets.boundary_prior import BoundaryPriorPredictor, pool_boundary_prior
from nnunetv2.training.loss.boundary import BoundaryAuxiliaryLoss, semantic_boundary_target


class SemanticBoundaryTargetTests(unittest.TestCase):
    def test_organ_background_and_organ_organ_edges_mark_both_sides(self):
        for left_label, right_label in ((0, 1), (1, 2)):
            target = torch.tensor([left_label, left_label, right_label, right_label])
            target = target.view(1, 1, 1, 4).expand(1, 1, 3, 4)
            boundary, mask = semantic_boundary_target(target)
            expected = torch.tensor([0., 1., 1., 0.]).view(1, 1, 1, 4).expand_as(boundary)
            torch.testing.assert_close(boundary, expected)
            self.assertTrue(mask.all())

    def test_homogeneous_foreground_has_no_artificial_patch_border(self):
        for shape in ((2, 1, 4, 5), (1, 1, 3, 4, 5)):
            boundary, mask = semantic_boundary_target(torch.ones(shape, dtype=torch.long))
            self.assertEqual(boundary.sum().item(), 0.)
            self.assertTrue(mask.all())

    def test_three_dimensional_neighbors(self):
        target = torch.zeros(1, 1, 4, 3, 2, dtype=torch.long)
        target[:, :, 2:] = 1
        boundary, _ = semantic_boundary_target(target)
        expected = torch.zeros_like(boundary)
        expected[:, :, 1:3] = 1.
        torch.testing.assert_close(boundary, expected)

    def test_ignore_masks_both_neighbors_without_recursive_expansion(self):
        target = torch.tensor([0, 1, 1, 255, 2, 2]).view(1, 1, 1, 6)
        boundary, mask = semantic_boundary_target(target, ignore_label=255)
        torch.testing.assert_close(boundary, torch.tensor([1., 1., 0., 0., 0., 0.]).view_as(boundary))
        torch.testing.assert_close(mask, torch.tensor([True, True, False, False, False, True]).view_as(mask))

    def test_ignore_interface_is_not_a_semantic_boundary(self):
        target = torch.tensor([1, 1, 255, 2, 2]).view(1, 1, 1, 5)
        boundary, mask = semantic_boundary_target(target, ignore_label=255)
        self.assertEqual(boundary.sum().item(), 0.)
        torch.testing.assert_close(mask, torch.tensor([True, False, False, False, True]).view_as(mask))

    def test_invalid_target_shapes_are_rejected(self):
        for shape in ((1, 4, 4), (1, 2, 4, 4), (1, 1, 0, 4)):
            with self.assertRaises(ValueError):
                semantic_boundary_target(torch.zeros(shape))


class BoundaryLossTests(unittest.TestCase):
    def setUp(self):
        self.target = torch.tensor([0, 0, 1, 1]).view(1, 1, 1, 4).expand(1, 1, 3, 4)

    def test_all_ignore_is_finite_differentiable_zero(self):
        logits = torch.randn(2, 1, 3, 4, requires_grad=True)
        target = torch.full_like(logits, 255.)
        loss = BoundaryAuxiliaryLoss(ignore_label=255)(logits, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_bce_only_matches_binary_cross_entropy(self):
        logits = torch.randn(self.target.shape, requires_grad=True)
        boundary, _ = semantic_boundary_target(self.target)
        loss = BoundaryAuxiliaryLoss(dice_weight=0)(logits, self.target)
        torch.testing.assert_close(loss, F.binary_cross_entropy_with_logits(logits, boundary))

    def test_correct_boundary_logits_reduce_loss(self):
        boundary, _ = semantic_boundary_target(self.target)
        correct_logits = (2. * boundary - 1.) * 8.
        criterion = BoundaryAuxiliaryLoss()
        self.assertLess(criterion(correct_logits, self.target), criterion(torch.zeros_like(boundary), self.target))

    def test_ignored_predictions_have_zero_gradient_and_do_not_change_loss(self):
        target = torch.tensor([0, 1, 1, 255, 2, 2]).view(1, 1, 1, 6)
        logits = torch.randn(target.shape, requires_grad=True)
        criterion = BoundaryAuxiliaryLoss(ignore_label=255)
        loss = criterion(logits, target)
        loss.backward()
        _, mask = semantic_boundary_target(target, 255)
        torch.testing.assert_close(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
        changed = logits.detach().clone()
        changed[~mask] = 100.
        torch.testing.assert_close(criterion(changed, target), loss.detach())

    def test_all_ignore_sample_does_not_dilute_batch_loss(self):
        logits = torch.randn(self.target.shape)
        criterion = BoundaryAuxiliaryLoss(ignore_label=255)
        combined_logits = torch.cat((logits, torch.randn_like(logits)))
        combined_target = torch.cat((self.target, torch.full_like(self.target, 255)))
        torch.testing.assert_close(criterion(combined_logits, combined_target), criterion(logits, self.target))

    def test_logits_target_resolution_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            BoundaryAuxiliaryLoss()(torch.zeros(1, 1, 2, 2), self.target)

    def test_invalid_hyperparameters_are_rejected(self):
        for kwargs in ({"bce_weight": -1}, {"dice_weight": float("nan")}, {"smooth": 0}):
            with self.assertRaises(ValueError):
                BoundaryAuxiliaryLoss(**kwargs)


class BoundaryPredictorTests(unittest.TestCase):
    def test_2d_and_3d_shapes_and_gradients(self):
        for spatial_shape, output_size in (((6, 8), (3, 2)), ((4, 6, 8), (2, 3, 2))):
            head = BoundaryPriorPredictor(4, len(spatial_shape), hidden_channels=8)
            features = torch.randn(2, 4, *spatial_shape, requires_grad=True)
            logits = head(features)
            self.assertEqual(logits.shape, (2, 1, *spatial_shape))
            prior = pool_boundary_prior(logits, output_size)
            self.assertEqual(prior.shape, (2, 1, *output_size))
            self.assertTrue(((prior >= 0) & (prior <= 1)).all())
            target = torch.zeros(2, 1, *spatial_shape, dtype=torch.long)
            target[..., spatial_shape[-1] // 2:] = 1
            loss = BoundaryAuxiliaryLoss()(logits, target) + prior.mean()
            loss.backward()
            self.assertTrue(torch.isfinite(features.grad).all())
            for parameter in head.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(head.output.weight.grad.abs().sum().item(), 0.)

    def test_max_pool_preserves_single_boundary_response(self):
        logits = torch.full((1, 1, 4, 4), -8.)
        logits[0, 0, 1, 1] = 8.
        prior = pool_boundary_prior(logits, (2, 2))
        self.assertGreater(prior[0, 0, 0, 0].item(), .99)
        self.assertLess(prior[0, 0, 1, 1].item(), .01)

    def test_invalid_predictor_and_pool_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            BoundaryPriorPredictor(4, spatial_dims=1)
        head = BoundaryPriorPredictor(4, spatial_dims=2)
        with self.assertRaises(ValueError):
            head(torch.randn(1, 4, 3, 4, 5))
        with self.assertRaises(ValueError):
            head(torch.randn(1, 3, 4, 5))
        with self.assertRaises(ValueError):
            pool_boundary_prior(torch.randn(1, 2, 4, 4), (2, 2))
        with self.assertRaises(ValueError):
            pool_boundary_prior(torch.randn(1, 1, 4, 4), (8, 8))


if __name__ == "__main__":
    unittest.main()
