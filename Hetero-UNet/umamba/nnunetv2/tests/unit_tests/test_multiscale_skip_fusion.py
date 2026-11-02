"""Real CPU autograd; CUDA Mamba is covered only by later server checks."""
import importlib.util
from pathlib import Path
import unittest

import torch

from nnunetv2.nets.multiscale_skip_fusion import AdaptiveMultiScaleSkip


class TestMultiScaleSkip(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)

    def make_case(self, dim=3):
        shapes = [(5, 9, 11), (3, 5, 6), (2, 3, 3)] if dim == 3 else [(9, 11), (5, 6), (3, 3)]
        return [torch.randn(2, c, *size, requires_grad=True)
                for c, size in zip([4, 8, 12], shapes)]

    def test_identity_and_source_softmax_all_scales_2d_3d(self):
        for dim in (2, 3):
            skips = self.make_case(dim)
            for i in range(3):
                fusion = AdaptiveMultiScaleSkip([4, 8, 12], i, dim, width=4)
                output, weights = fusion(skips, torch.randn_like(skips[i]), True)
                torch.testing.assert_close(output, skips[i], rtol=0, atol=0)
                torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]))
                self.assertEqual(weights.shape[1], 3 if i == 1 else 2)

    def test_first_update_opens_nonzero_cross_scale_and_query_gradients(self):
        skips = self.make_case()
        fusion = AdaptiveMultiScaleSkip([4, 8, 12], 1, width=4)
        context = torch.randn_like(skips[1], requires_grad=True)
        optim = torch.optim.SGD(fusion.parameters(), lr=0.1)
        target = torch.randn_like(skips[1])
        (fusion(skips, context) * target).mean().backward()
        self.assertGreater(fusion.output.weight.grad.abs().sum().item(), 0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        for source in skips:
            source.grad = None
        (fusion(skips, context) * target).mean().backward()
        for source in (skips[0], skips[2]):
            self.assertGreater(source.grad.abs().sum().item(), 0)
        for p in [fusion.query.weight] + [p.weight for p in fusion.keys]:
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_neighbor_and_decoder_context_actually_change_output(self):
        skips = self.make_case()
        fusion = AdaptiveMultiScaleSkip([4, 8, 12], 1, width=4)
        torch.nn.init.normal_(fusion.output.weight, std=0.2)
        context = torch.randn_like(skips[1])
        expected = fusion(skips, context)
        changed = list(skips)
        changed[0] = changed[0] + 2
        self.assertGreater((expected - fusion(changed, context)).abs().max().item(), 1e-5)
        self.assertGreater((expected - fusion(skips, -context)).abs().max().item(), 1e-5)
        # Fusion introduces no new cross-sample coupling.
        single = fusion([s[:1] for s in skips], context[:1])
        torch.testing.assert_close(single, expected[:1])

    def test_cpu_bfloat16_autocast_backward_is_finite(self):
        skips = self.make_case()
        fusion = AdaptiveMultiScaleSkip([4, 8, 12], 1, width=4)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = fusion(skips, torch.randn_like(skips[1]))
            loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(fusion.output.weight.grad).all())

    def test_checkpoint_roundtrip(self):
        skips = self.make_case()
        original = AdaptiveMultiScaleSkip([4, 8, 12], 1, width=4)
        torch.nn.init.normal_(original.output.weight)
        restored = AdaptiveMultiScaleSkip([4, 8, 12], 1, width=4)
        restored.load_state_dict(original.state_dict(), strict=True)
        context = torch.randn_like(skips[1])
        torch.testing.assert_close(original(skips, context), restored(skips, context))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            AdaptiveMultiScaleSkip([4], 0)
        fusion = AdaptiveMultiScaleSkip([4, 8, 12], 1)
        with self.assertRaises(ValueError):
            fusion(self.make_case(), torch.zeros(2, 1, 2, 2, 2))


class TestStage3Runner(unittest.TestCase):
    def test_isolated_names_and_protocol(self):
        path = Path(__file__).resolve().parents[4] / 'scripts' / 'run_stage3.py'
        spec = importlib.util.spec_from_file_location('stage3_runner_test', path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        for model in ('D', 'E'):
            args = runner.parse_args(['--model', model, '--results-root', '.', '--smoke'])
            self.assertEqual(args.fold, 'all')
            self.assertIn('stage3_' + model + '_seed1234_smoke', args.effective_results_root)
            self.assertFalse(args.final_validation)
            self.assertIn('MultiScaleSkip', args.trainer)


if __name__ == '__main__':
    unittest.main()
