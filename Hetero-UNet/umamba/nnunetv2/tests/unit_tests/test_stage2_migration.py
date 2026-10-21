"""Archive extraction guards used before a resumed medical segmentation run."""
import importlib.util
import io
from pathlib import Path
import shutil
import tarfile
import unittest
from uuid import uuid4

MODULE = Path(__file__).resolve().parents[4] / 'scripts/restore_stage2_migration.py'
SPEC = importlib.util.spec_from_file_location('stage2_migration_restore_tests', MODULE)
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


class TestMigrationExtraction(unittest.TestCase):
    def setUp(self):
        self.parent = Path(__file__).resolve().parent
        self.folder = self.parent / ('migration_test_' + uuid4().hex)
        self.folder.mkdir()
        self.archive = self.folder / 'input.tar'
        self.destination = self.folder / 'output'

    def tearDown(self):
        assert self.folder.parent == self.parent and self.folder.name.startswith('migration_test_')
        shutil.rmtree(self.folder)

    def make_archive(self, name, link=None):
        with tarfile.open(self.archive, 'w') as archive:
            member = tarfile.TarInfo(name)
            if link is None:
                data = b'unchanged checkpoint bytes'
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            else:
                member.type = tarfile.SYMTYPE
                member.linkname = link
                archive.addfile(member)

    def test_valid_checkpoint_roundtrip_preserves_bytes(self):
        self.make_archive('stage2_F_seed1234/checkpoint_latest.pth')
        restore.extract_checked(self.archive, self.destination, {'stage2_F_seed1234'})
        self.assertEqual((self.destination / 'stage2_F_seed1234/checkpoint_latest.pth').read_bytes(),
                         b'unchanged checkpoint bytes')

    def test_parent_escape_is_rejected(self):
        self.make_archive('../outside')
        with self.assertRaises(RuntimeError):
            restore.extract_checked(self.archive, self.destination, {'stage2_F_seed1234'})
        self.assertFalse((self.folder / 'outside').exists())

    def test_absolute_path_is_rejected(self):
        self.make_archive('/outside')
        with self.assertRaises(RuntimeError):
            restore.extract_checked(self.archive, self.destination, {'stage2_F_seed1234'})

    def test_symlink_is_rejected(self):
        self.make_archive('stage2_F_seed1234/link', '../outside')
        with self.assertRaises(RuntimeError):
            restore.extract_checked(self.archive, self.destination, {'stage2_F_seed1234'})

    def test_unexpected_experiment_is_rejected(self):
        self.make_archive('stage2_C_seed1234/checkpoint_latest.pth')
        with self.assertRaises(RuntimeError):
            restore.extract_checked(self.archive, self.destination, {'stage2_F_seed1234'})

    def test_plan_hash_uses_original_manifest(self):
        descriptor = self.folder / 'plans.json'
        descriptor.write_bytes(b'{"frozen":true}')
        expected = restore.checksum(descriptor)
        manifest = {'fingerprints': {'descriptor_sha256': {'nnUNetPlans.json': expected}}}
        self.assertEqual(restore.verify_plan_descriptor(descriptor, manifest), expected)
        descriptor.write_bytes(b'{"frozen":false}')
        with self.assertRaisesRegex(RuntimeError, 'differ'):
            restore.verify_plan_descriptor(descriptor, manifest)

    def test_invalid_reference_hash_is_rejected(self):
        descriptor = self.folder / 'plans.json'
        descriptor.write_bytes(b'{}')
        manifest = {'fingerprints': {'descriptor_sha256': {'nnUNetPlans.json': 'a' * 65}}}
        with self.assertRaisesRegex(RuntimeError, 'invalid'):
            restore.verify_plan_descriptor(descriptor, manifest)


if __name__ == '__main__':
    unittest.main()
