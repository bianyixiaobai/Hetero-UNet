"""Stdlib-only CLI/deployment safeguards; no trainer or CUDA imports required."""

from contextlib import contextmanager, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from uuid import uuid4


RUNNER_PATH = Path(__file__).resolve().parents[4] / "scripts" / "run_stage2.py"
SPEC = importlib.util.spec_from_file_location("stage2_runner_for_tests", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@contextmanager
def temporary_test_folder():
    # tempfile's mode=0700 applies a restrictive Windows ACL which prevents
    # sandboxed child processes from reading their own freshly created folder.
    # Ordinary mkdir inherits the existing workspace ACL instead.
    parent = Path(__file__).resolve().parent
    folder = parent / ("stage2_test_" + uuid4().hex)
    folder.mkdir()
    try:
        yield folder
    finally:
        if folder.resolve().parent != parent or not folder.name.startswith("stage2_test_"):
            raise RuntimeError("Refusing to clean a test directory outside the owned test location")
        shutil.rmtree(folder)


class TestStage2Parser(unittest.TestCase):
    def test_frozen_defaults_and_model_mapping(self):
        args = runner.parse_args(["--model", "F", "--results-root", "."], environ={})
        self.assertEqual(args.dataset, "701")
        self.assertEqual(args.configuration, "3d_fullres")
        self.assertEqual(args.fold, "all")
        self.assertEqual(args.seed, 1234)
        self.assertEqual(args.plans, "nnUNetPlans")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.trainer, "nnUNetTrainerMambaUNetBlockAttnRes")
        self.assertTrue(args.final_validation)
        for model, trainer in runner.TRAINERS.items():
            args = runner.parse_args(["--model", model, "--results-root", "."], environ={})
            self.assertEqual(args.trainer, trainer)

    def test_model_seed_and_smoke_outputs_do_not_collide(self):
        roots = set()
        for model in ("F", "C", "F-aux"):
            for seed in (1234, 2345):
                for smoke in (False, True):
                    arguments = ["--model", model, "--seed", str(seed), "--results-root", "."]
                    if smoke:
                        arguments.append("--smoke")
                    args = runner.parse_args(arguments, environ={})
                    roots.add(args.effective_results_root)
                    self.assertEqual(Path(args.effective_results_root).parent, Path(args.results_root))
        self.assertEqual(len(roots), 12)

    def test_environment_root_and_explicit_override(self):
        environment = {"nnUNet_results": "environment_results"}
        before = environment.copy()
        args = runner.parse_args(["--model", "C"], environ=environment)
        self.assertEqual(args.results_root, str(Path("environment_results").resolve()))
        args = runner.parse_args(["--model", "C", "--results-root", "explicit_results"],
                                 environ=environment)
        self.assertEqual(args.results_root, str(Path("explicit_results").resolve()))
        self.assertEqual(environment, before)

    def test_smoke_and_skip_flags_only_disable_final_export(self):
        for option in ("--smoke", "--skip-final-validation"):
            args = runner.parse_args(["--model", "F", "--results-root", ".", option], environ={})
            self.assertFalse(args.final_validation)
        args = runner.parse_args(["--model", "F", "--results-root", ".", "--continue"], environ={})
        self.assertTrue(args.continue_training)
        self.assertTrue(args.final_validation)

    def test_dataset_names_folds_and_seed_boundaries(self):
        args = runner.parse_args([
            "--model", "C", "--results-root", ".", "--dataset", "Dataset701_AbdomenCT",
            "--fold", "4", "--development", "--seed", str(2 ** 32 - 1), "--device", "cpu",
        ], environ={})
        self.assertEqual(args.dataset, "Dataset701_AbdomenCT")
        self.assertEqual(args.fold, 4)
        self.assertEqual(args.seed, 2 ** 32 - 1)
        self.assertEqual(args.device, "cpu")

    def test_numeric_folds_require_development_and_are_isolated(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(["--model", "C", "--results-root", ".", "--fold", "0"], environ={})
        formal = runner.parse_args(["--model", "C", "--results-root", "."], environ={})
        development = runner.parse_args([
            "--model", "C", "--results-root", ".", "--fold", "0", "--development",
        ], environ={})
        self.assertNotEqual(formal.effective_results_root, development.effective_results_root)
        self.assertEqual(Path(development.effective_results_root).name, "stage2_C_seed1234_dev_fold0")

    def test_formal_ct_wrong_configuration_is_rejected_during_parsing(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args([
                "--model", "F", "--results-root", ".", "--configuration", "2d",
            ], environ={})
        development = runner.parse_args([
            "--model", "F", "--results-root", ".", "--configuration", "2d", "--development",
        ], environ={})
        self.assertTrue(development.development)

    def test_invalid_protocol_arguments_are_rejected(self):
        invalid_options = [
            ["--model", "A"], ["--fold", "-1"], ["--fold", "5"],
            ["--seed", "-1"], ["--seed", str(2 ** 32)], ["--seed", "1.5"],
            ["--dataset", "not_a_dataset"], ["--dataset", "0"], ["--device", "mps"],
            ["--batch-size", "1"], ["--epochs", "10"], ["--num-gpus", "2"],
        ]
        for options in invalid_options:
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.parse_args(["--model", "F", "--results-root", "."] + options,
                                      environ={})

    def test_missing_or_blank_results_root_is_rejected(self):
        for environment in ({}, {"nnUNet_results": ""}, {"nnUNet_results": "   "}):
            with self.subTest(environment=environment), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.parse_args(["--model", "F"], environ=environment)

    def test_dry_run_works_without_site_packages_and_creates_no_outputs(self):
        with temporary_test_folder() as temporary:
            root = Path(temporary) / "does_not_exist_yet"
            process = subprocess.run(
                [sys.executable, "-S", str(RUNNER_PATH), "--model", "C", "--seed", "2345",
                 "--results-root", str(root), "--smoke", "--dry-run"],
                capture_output=True, text=True, check=True, timeout=15,
            )
            payload = json.loads(process.stdout)
            self.assertEqual(payload["seed"], 2345)
            self.assertTrue(payload["smoke"])
            self.assertFalse(payload["final_validation"])
            self.assertEqual(Path(payload["effective_results_root"]).name, "stage2_C_seed2345_smoke")
            self.assertFalse(root.exists())


class TestCheckpointAndRecordSafeguards(unittest.TestCase):
    def test_resolved_results_folder_must_stay_inside_experiment_root(self):
        root = Path.cwd() / 'stage2_F_seed1234'
        runner.validate_results_location(root, root / 'Dataset701_CT' / 'Trainer' / 'fold_all')
        with self.assertRaises(ValueError):
            runner.validate_results_location(root, root.with_name('stage2_F_seed1234_smoke'))
        with self.assertRaises(ValueError):
            runner.validate_results_location(root, root.parent / 'unrelated_results')

    def test_resume_requires_existing_checkpoint_and_uses_upstream_precedence(self):
        with temporary_test_folder() as temporary:
            folder = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                runner.checkpoint_for_resume(folder)
            for name in ("checkpoint_best.pth", "checkpoint_latest.pth", "checkpoint_final.pth"):
                (folder / name).touch()
                self.assertEqual(runner.checkpoint_for_resume(folder), folder / name)

    def test_latest_manifest_does_not_remove_prior_invocation_record(self):
        with temporary_test_folder() as temporary:
            folder = Path(temporary)
            first = folder / "first.json"
            second = folder / "second.json"
            latest = folder / "stage2_run.json"
            runner.write_manifest({"seed": 1234, "status": "failed"}, first, latest)
            runner.write_manifest({"seed": 1234, "status": "completed"}, second, latest)
            self.assertEqual(json.loads(first.read_text())["status"], "failed")
            self.assertEqual(json.loads(second.read_text())["status"], "completed")
            self.assertEqual(json.loads(latest.read_text())["status"], "completed")

    def test_file_hashes_track_content(self):
        with temporary_test_folder() as temporary:
            path = temporary / "descriptor.json"
            path.write_bytes(b"{}")
            first = runner.sha256_file(path)
            path.write_bytes(b'{"value": 1}')
            self.assertNotEqual(first, runner.sha256_file(path))
            self.assertEqual(len(first), 64)


class TestFrozenCTProtocol(unittest.TestCase):
    def arguments(self, **changes):
        values = {"dataset": "701", "configuration": "3d_fullres", "fold": "all",
                  "development": False, "smoke": False}
        values.update(changes)
        return SimpleNamespace(**values)

    def trainer(self, batch_size=2, patch_size=(40, 224, 192)):
        return SimpleNamespace(batch_size=batch_size,
                               configuration_manager=SimpleNamespace(patch_size=patch_size))

    def test_formal_ct_id_and_dataset_name_require_frozen_plan(self):
        for dataset in ("701", 701, "Dataset701_AbdomenCT"):
            with self.subTest(dataset=dataset):
                result = runner.validate_frozen_ct_protocol(self.arguments(dataset=dataset), self.trainer())
                self.assertTrue(result["frozen_ct_enforced"])
                self.assertTrue(result["eligible_for_frozen_CT701_comparison"])

    def test_batch_patch_configuration_and_fold_mismatches_are_rejected(self):
        variants = [
            (self.arguments(), self.trainer(batch_size=1)),
            (self.arguments(), self.trainer(batch_size=2.5)),
            (self.arguments(), self.trainer(patch_size=(32, 224, 192))),
            (self.arguments(), self.trainer(patch_size=(40.5, 224, 192))),
            (self.arguments(configuration="2d"), self.trainer()),
            (self.arguments(fold=0), self.trainer()),
        ]
        for args, trainer in variants:
            with self.subTest(args=args, trainer=trainer), self.assertRaises(ValueError):
                runner.validate_frozen_ct_protocol(args, trainer)

    def test_smoke_checks_full_patch_but_is_not_a_main_result(self):
        args = self.arguments(smoke=True)
        result = runner.validate_frozen_ct_protocol(args, self.trainer())
        self.assertTrue(result["frozen_ct_enforced"])
        self.assertFalse(result["eligible_for_frozen_CT701_comparison"])
        with self.assertRaises(ValueError):
            runner.validate_frozen_ct_protocol(args, self.trainer(patch_size=(8, 16, 16)))

    def test_development_and_mr_use_their_own_explicitly_scoped_plans(self):
        result = runner.validate_frozen_ct_protocol(
            self.arguments(development=True, fold=0, configuration="2d"),
            self.trainer(batch_size=1, patch_size=(64, 64)),
        )
        self.assertEqual(result["scope"], "development")
        self.assertFalse(result["eligible_for_frozen_CT701_comparison"])
        for dataset in ("702", "Dataset702_AbdomenMR"):
            result = runner.validate_frozen_ct_protocol(
                self.arguments(dataset=dataset), self.trainer(patch_size=(24, 192, 192)),
            )
            self.assertEqual(result["scope"], "dataset_specific")
            self.assertFalse(result["frozen_ct_enforced"])


if __name__ == "__main__":
    unittest.main()
