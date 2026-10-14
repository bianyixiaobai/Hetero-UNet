import unittest

import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


class TestPolyLRCompatibility(unittest.TestCase):
    def test_original_polynomial_learning_rates_are_preserved(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=0.01)
        scheduler = PolyLRScheduler(optimizer, initial_lr=0.01, max_steps=1000)
        for epoch in (0, 1, 499, 999, 1000):
            scheduler.step(epoch)
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'],
                                   0.01 * (1 - epoch / 1000) ** 0.9, places=14)

    def test_optimizer_state_round_trip_can_keep_scheduler_progress(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=0.01)
        scheduler = PolyLRScheduler(optimizer, initial_lr=0.01, max_steps=1000)
        scheduler.step(600)
        state = scheduler.state_dict()
        restored = PolyLRScheduler(optimizer, initial_lr=0.01, max_steps=1000)
        restored.load_state_dict(state)
        restored.step(601)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.01 * (1 - 601 / 1000) ** 0.9)


if __name__ == '__main__':
    unittest.main()
