"""CPU mechanism checks, not segmentation efficacy or real Mamba tests."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.nets.spatial_receptive_block import SpatialReceptiveBlock3D


class TestSpatialBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reference_equation(self):
        for variant, dilation in [('ND', 1), ('R0', 2)]:
            block = SpatialReceptiveBlock3D(4, variant)
            x = torch.randn(2, 4, 5, 7, 9)
            parts = [F.conv3d(x, block.branches[0].weight, groups=4),
                     F.conv3d(x, block.branches[1].weight, padding=1, groups=4),
                     F.conv3d(x, block.branches[2].weight, padding=dilation, dilation=dilation, groups=4)]
            z = F.group_norm(torch.cat(parts, 1), 4, block.norm.weight, block.norm.bias, block.norm.eps)
            expected = x + F.conv3d(F.gelu(z), block.projection.weight, block.projection.bias)
            torch.testing.assert_close(block(x), expected, rtol=0, atol=0)

    def test_sampling_support(self):
        x = torch.zeros(1, 1, 9, 9, 9)
        x[0, 0, 4, 4, 4] = 1
        for variant, offsets in [('ND', {3, 4, 5}), ('R0', {2, 4, 6})]:
            branch = SpatialReceptiveBlock3D(1, variant).branches[2]
            nn.init.ones_(branch.weight)
            support = branch(x)[0, 0].nonzero()
            self.assertEqual(len(support), 27)
            for axis in range(3):
                self.assertEqual(set(support[:, axis].tolist()), offsets)

    def test_same_initial_tensors_and_parameter_count(self):
        torch.manual_seed(1234)
        nd = SpatialReceptiveBlock3D(16, 'ND')
        rng_nd = torch.get_rng_state()
        torch.manual_seed(1234)
        r0 = SpatialReceptiveBlock3D(16, 'R0')
        self.assertTrue(torch.equal(rng_nd, torch.get_rng_state()))
        for name, value in nd.state_dict().items():
            self.assertTrue(torch.equal(value, r0.state_dict()[name]), name)
        self.assertEqual(sum(p.numel() for p in nd.parameters()), 3 * 16 ** 2 + 62 * 16)
        self.assertEqual(nd.conv_macs((5, 14, 12), 2), r0.conv_macs((5, 14, 12), 2))

    def test_shape_gradients_and_optimizer_updates(self):
        for variant in ('ND', 'R0'):
            for shape in [(1, 3, 1, 2, 3), (2, 3, 5, 7, 9)]:
                block = SpatialReceptiveBlock3D(3, variant)
                optimizer = torch.optim.SGD(block.parameters(), lr=.01, momentum=.99)
                x = torch.randn(shape, requires_grad=True)
                result = block(x)
                self.assertEqual(result.shape, x.shape)
                self.assertTrue(torch.isfinite(result).all())
                (result * torch.randn_like(result)).mean().backward()
                for name, parameter in block.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                before = [b.weight.detach().clone() for b in block.branches]
                optimizer.step()
                self.assertTrue(all(not torch.equal(a, b.weight) for a, b in zip(before, block.branches)))

    def test_invalid_arguments(self):
        for channels, variant in [(0, 'R0'), (True, 'ND'), (2, 'unknown')]:
            with self.assertRaises(ValueError):
                SpatialReceptiveBlock3D(channels, variant)
        with self.assertRaises(ValueError):
            SpatialReceptiveBlock3D(2, 'ND')(torch.ones(1, 2, 4, 4))


class TestSpatialLauncher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path(__file__).resolve().parents[4] / 'scripts/run_stage3.py'
        spec = importlib.util.spec_from_file_location('spatial_runner_test', file)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_names_isolation_and_no_trainset_export(self):
        nd = self.runner.parse_args(['--model', 'C-ND', '--results-root', '.'])
        r0 = self.runner.parse_args(['--model', 'C-R0', '--results-root', '.', '--smoke'])
        self.assertFalse(nd.final_validation)
        self.assertNotEqual(nd.effective_results_root, r0.effective_results_root)
        self.assertIn('_smoke', r0.effective_results_root)
        self.assertEqual(nd.fold, 'all')
        self.assertEqual(nd.seed, 1234)

    def test_reject_changed_protocol(self):
        for extra in [['--development'], ['--fold', '0'], ['--seed', '42'],
                      ['--configuration', '2d'], ['--dataset', '999'], ['--plans', 'OtherPlans']]:
            with self.assertRaises(SystemExit):
                self.runner.parse_args(['--model', 'C-R0', '--results-root', '.'] + extra)

    def test_old_D_E_parser_unchanged(self):
        for model in ('D', 'E'):
            args = self.runner.parse_args(['--model', model, '--results-root', '.'])
            self.assertTrue(args.final_validation)


if __name__ == '__main__':
    unittest.main()
