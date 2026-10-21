"""Protect active experiment paths and verify archival bytes before replacing files."""
import importlib.util
import io
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

MODULE = Path(__file__).resolve().parents[4] / 'scripts/full_project_preservation.py'
SPEC = importlib.util.spec_from_file_location('full_preservation_tests', MODULE)
backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup)


class TestFullPreservation(unittest.TestCase):
    def test_original_F_is_archived_separately(self):
        result = backup.destination_for('stage2_results/stage2_F_seed1234/fold_all/checkpoint_best.pth')
        result.relative_to(backup.ARCHIVE)

    def test_old_source_is_archived_separately(self):
        backup.destination_for('Hetero-UNet-stage2/.git/HEAD').relative_to(backup.ARCHIVE)

    def test_root_profiles_do_not_replace_live_config(self):
        backup.destination_for('_root_profiles/.bashrc').relative_to(backup.ARCHIVE)

    def test_path_escape_is_rejected(self):
        with self.assertRaises((RuntimeError, ValueError)):
            backup.destination_for('../outside')

    def test_bad_content_cannot_replace_existing_file(self):
        parent = Path(__file__).parent
        folder = parent / ('preservation_test_' + uuid4().hex)
        folder.mkdir()
        try:
            path = folder / 'protected.pth'
            path.write_bytes(b'original')
            row = {'path': 'mock.pth', 'bytes': 7, 'sha256': '0' * 64, 'mode': 0o600}
            with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                backup.materialize(row, io.BytesIO(b'corrupt'), path)
            self.assertEqual(path.read_bytes(), b'original')
        finally:
            assert folder.parent == parent and folder.name.startswith('preservation_test_')
            shutil.rmtree(folder)


if __name__ == '__main__':
    unittest.main()
