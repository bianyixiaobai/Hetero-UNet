"""CPU-only semantic tests: python -m unittest ...test_depth_aggregation."""

import unittest

import torch

from nnunetv2.nets.block_attention_residuals import DepthAttentionAggregator


class TestDepthAttentionAggregator(unittest.TestCase):
    def test_zero_queries_give_uniform_average_for_every_token(self):
        torch.manual_seed(3)
        history = [torch.randn(2, 7, 4) for _ in range(5)]
        aggregator = DepthAttentionAggregator(4, boundary_conditioned=True)
        output, weights = aggregator(history, torch.rand(2, 7, 1), return_weights=True)
        torch.testing.assert_close(output, sum(history) / len(history))
        torch.testing.assert_close(weights, torch.full((2, 7, 5), 0.2))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 7))

    def test_query_selects_a_source_not_a_batch_or_token(self):
        aggregator = DepthAttentionAggregator(2)
        with torch.no_grad():
            aggregator.query.copy_(torch.tensor([12.0, 0.0]))
        positive = torch.tensor([[[2.0, 0.0], [3.0, 0.0]], [[4.0, 0.0], [5.0, 0.0]]])
        negative = -positive
        output, weights = aggregator([positive, negative], return_weights=True)
        self.assertEqual(weights.shape, (2, 2, 2))
        self.assertTrue(bool((weights[..., 0] > 0.99999).all()))
        torch.testing.assert_close(output, positive)

    def test_boundary_changes_relative_source_scores_at_each_location(self):
        aggregator = DepthAttentionAggregator(2, boundary_conditioned=True)
        with torch.no_grad():
            aggregator.boundary_query.copy_(torch.tensor([2.0, -2.0]))
        source_a = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        source_b = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
        output, weights = aggregator(
            [source_a, source_b], torch.tensor([[[0.0], [1.0]]]), return_weights=True
        )
        torch.testing.assert_close(weights[:, 0], torch.tensor([[0.5, 0.5]]))
        self.assertGreater(weights[0, 1, 0].item(), 0.98)
        self.assertGreater(output[0, 1, 0].item(), output[0, 0, 0].item())

    def test_unrelated_batch_and_token_values_cannot_change_other_weights(self):
        torch.manual_seed(4)
        aggregator = DepthAttentionAggregator(4)
        with torch.no_grad():
            aggregator.query.copy_(torch.randn(4))
        history = [torch.randn(2, 3, 4) for _ in range(3)]
        output_before, weights_before = aggregator(history, return_weights=True)
        altered = [source.clone() for source in history]
        altered[1][1, 2] = torch.tensor([100.0, -50.0, 30.0, -20.0])
        output_after, weights_after = aggregator(altered, return_weights=True)
        torch.testing.assert_close(weights_before[0], weights_after[0])
        torch.testing.assert_close(output_before[0], output_after[0])
        torch.testing.assert_close(weights_before[1, :2], weights_after[1, :2])
        torch.testing.assert_close(output_before[1, :2], output_after[1, :2])

    def test_gradients_reach_history_and_both_queries_from_zero_initialization(self):
        aggregator = DepthAttentionAggregator(2, boundary_conditioned=True)
        history = [torch.tensor([[[1.0, 0.0]]], requires_grad=True),
                   torch.tensor([[[0.0, 2.0]]], requires_grad=True)]
        output = aggregator(history, torch.ones(1, 1, 1))
        output.square().sum().backward()
        for source in history:
            self.assertIsNotNone(source.grad)
            self.assertTrue(bool(torch.isfinite(source.grad).all()))
            self.assertGreater(source.grad.abs().sum().item(), 0)
        for query in (aggregator.query, aggregator.boundary_query):
            self.assertIsNotNone(query.grad)
            self.assertTrue(bool(torch.isfinite(query.grad).all()))
            self.assertGreater(query.grad.abs().sum().item(), 0)

    def test_learned_boundary_query_sends_gradient_to_boundary_branch(self):
        aggregator = DepthAttentionAggregator(2, boundary_conditioned=True)
        with torch.no_grad():
            aggregator.boundary_query.copy_(torch.tensor([0.3, -0.2]))
        boundary = torch.full((1, 1, 1), 0.5, requires_grad=True)
        aggregator([torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.0, 2.0]]])],
                   boundary).square().sum().backward()
        self.assertTrue(bool(torch.isfinite(boundary.grad).all()))
        self.assertGreater(boundary.grad.abs().sum().item(), 0)

    def test_half_precision_keys_do_not_square_in_half_precision(self):
        aggregator = DepthAttentionAggregator(2)
        history = [torch.full((1, 2, 2), 1000.0, dtype=torch.float16),
                   torch.full((1, 2, 2), -500.0, dtype=torch.float16)]
        output, weights = aggregator(history, return_weights=True)
        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(weights.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(output).all()))
        torch.testing.assert_close(output, torch.full_like(output, 250.0))

    def test_invalid_shapes_dtypes_and_boundary_are_rejected(self):
        aggregator = DepthAttentionAggregator(3, boundary_conditioned=True, validate_finite=True)
        source = torch.zeros(2, 4, 3)
        with self.assertRaises(ValueError):
            aggregator([], torch.zeros(2, 4, 1))
        with self.assertRaises(ValueError):
            aggregator([source, torch.zeros(2, 5, 3)], torch.zeros(2, 4, 1))
        with self.assertRaises(ValueError):
            aggregator([source], None)
        with self.assertRaises(ValueError):
            aggregator([source], torch.zeros(2, 4, 2))
        with self.assertRaises(ValueError):
            aggregator([source], torch.full((2, 4, 1), float('nan')))
        with self.assertRaises(ValueError):
            aggregator([source], torch.full((2, 4, 1), 1.01))
        with self.assertRaises(ValueError):
            aggregator([source, source.double()], torch.zeros(2, 4, 1))
        with self.assertRaises(TypeError):
            aggregator([source.long()], torch.zeros(2, 4, 1))
        with self.assertRaises(ValueError):
            DepthAttentionAggregator(3)([source], torch.zeros(2, 4, 1))
        with self.assertRaises(ValueError):
            aggregator([torch.full_like(source, float('inf'))], torch.zeros(2, 4, 1))

    def test_one_source_is_identity_and_checkpoint_round_trip(self):
        aggregator = DepthAttentionAggregator(4, boundary_conditioned=True)
        restored = DepthAttentionAggregator(4, boundary_conditioned=True)
        restored.load_state_dict(aggregator.state_dict())
        source = torch.randn(2, 3, 4)
        output, weights = restored([source], torch.rand(2, 3), return_weights=True)
        torch.testing.assert_close(output, source)
        torch.testing.assert_close(weights, torch.ones(2, 3, 1))


if __name__ == '__main__':
    unittest.main()
