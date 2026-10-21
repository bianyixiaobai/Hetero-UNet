"""Check the real path module in fresh processes, without CUDA dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

PATHS_FILE = Path(__file__).resolve().parents[2] / 'paths.py'
NAMES = ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')


class TestPathConfiguration(unittest.TestCase):
    def read_paths(self, overrides):
        environment = os.environ.copy()
        for name in NAMES:
            environment.pop(name, None)
        environment.update(overrides)
        code = (
            'import runpy,json; '
            f'd=runpy.run_path({str(PATHS_FILE)!r}); '
            f'print(json.dumps({{k:d[k] for k in {NAMES!r}}}))'
        )
        result = subprocess.run([sys.executable, '-S', '-c', code], env=environment,
                                capture_output=True, text=True, check=True, timeout=15)
        return json.loads(result.stdout)

    def test_explicit_environment_roots_override_project_layout(self):
        roots = {name: str(Path.cwd() / 'isolated-path-test' / name) for name in NAMES}
        self.assertEqual(self.read_paths(roots), roots)

    def test_unset_environment_preserves_original_project_layout(self):
        base = PATHS_FILE.parents[2] / 'data'
        expected = {name: str(base / name) for name in NAMES}
        self.assertEqual(self.read_paths({}), expected)

    def test_empty_environment_values_fall_back_to_usable_defaults(self):
        self.assertEqual(self.read_paths({name: '' for name in NAMES}), self.read_paths({}))


if __name__ == '__main__':
    unittest.main()
