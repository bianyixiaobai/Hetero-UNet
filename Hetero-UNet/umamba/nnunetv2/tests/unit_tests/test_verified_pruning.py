"""Deletion must reject active experiments, symlinks and stale/absent proofs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[4] / 'scripts/prune_stage2_verified_duplicates.py'
SPEC = importlib.util.spec_from_file_location('verified_pruning', MODULE)
prune = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prune)


class TestVerifiedPruning(unittest.TestCase):
    def test_live_F_and_training_data_are_protected(self):
        for name in ['stage2_results/stage2_F_seed1234/Dataset701/X/fold_all/checkpoint_best.pth',
                     'Hetero-UNet/data/nnUNet_preprocessed/Dataset701/data.npy',
                     'Hetero-UNet/data/nnUNet_raw/CT.nii.gz', 'Hetero-UNet-stage2/.git/HEAD']:
            self.assertFalse(prune.allowed(prune.ROOT / name))

    def test_only_expected_old_checkpoint_names_are_allowed(self):
        base = prune.ROOT / 'stage2_results/stage2_C_seed1234_smoke/Dataset701/Trainer/fold_all'
        self.assertTrue(prune.allowed(base / 'checkpoint_final.pth'))
        self.assertFalse(prune.allowed(base / 'plans.json'))
        self.assertFalse(prune.allowed(prune.ROOT / 'stage2_restore_archives/unknown.tar'))
        self.assertFalse(prune.allowed(prune.ROOT / 'stage2_restore_archives/../unknown.tar'))

    def test_deletion_requires_fresh_proof_and_unchanged_content(self):
        original = prune.ROOT
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            prune.ROOT = Path(directory).resolve()
            try:
                target = prune.ROOT / 'stage2_restore_archives/3d_fullres_weights.tar'
                target.parent.mkdir()
                target.write_bytes(b'preserve')
                row = {'path': str(target), 'bytes': 8, 'sha256': prune.sha(target), 'object_sha256': 'proof'}
                with self.assertRaisesRegex(RuntimeError, 'No verified OSS'):
                    prune.delete_verified(row, {})
                target.write_bytes(b'modified')
                with self.assertRaisesRegex(RuntimeError, 'changed'):
                    prune.delete_verified(row, {'proof': {}})
                self.assertTrue(target.exists())
                row['sha256'] = prune.sha(target)
                prune.delete_verified(row, {'proof': {}})
                self.assertFalse(target.exists())
            finally:
                prune.ROOT = original


if __name__ == '__main__':
    unittest.main()
