"""Shutdown interlocks use mocks; these tests never invoke a real shutdown/OSS."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

MODULE = Path(__file__).resolve().parents[4] / "scripts/finalize_stage2.py"
SPEC = importlib.util.spec_from_file_location("stage2_finalize_tests", MODULE)
finalize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalize)


@contextmanager
def fixture():
    parent = Path(__file__).resolve().parent
    folder = parent / ("finalize_test_" + uuid4().hex)
    folder.mkdir()
    try:
        config = {"hostname": "test-instance", "cancel_file": str(folder / "CANCEL"),
                  "queue_file": str(folder / "queue.json"), "seed": 1234, "epochs": 1000,
                  "training_commit": "frozen", "jobs": [], "archive_dir": str(folder / "archives"),
                  "state_file": str(folder / "state.json"), "oss_prefix": "oss://test-only/",
                  "oss_executable": "MOCK_OSS", "shutdown_executable": str(folder / "MOCK_SHUTDOWN")}
        Path(config["shutdown_executable"]).write_bytes(b"never executed")
        config["shutdown_sha256"] = finalize.sha256(config["shutdown_executable"])
        for model in ("F", "C"):
            root = folder / model
            (root / "fold_all").mkdir(parents=True)
            job = {"model": model, "trainer": "Trainer" + model, "results_dir": str(root)}
            config["jobs"].append(job)
            manifest = {"status": "completed", "git": {"commit": "frozen"},
                        "arguments": {"model": model, "smoke": False},
                        "protocol": {"seed": 1234, "trainer": job["trainer"], "num_epochs": 1000,
                                     "train_iterations_per_epoch": 250, "validation_iterations_per_epoch": 50,
                                     "batch_size": 2, "patch_size": [40, 224, 192],
                                     "initial_lr": 0.01, "mha_batch_first": False}}
            finalize.write_json(root / "fold_all/stage2_run.json", manifest)
            for name in ("checkpoint_best.pth", "checkpoint_final.pth"):
                (root / "fold_all" / name).write_bytes(b"mock checkpoint")
        finalize.write_json(config["queue_file"], {"status": "completed", "sequence": ["F", "C"],
                                                 "seed": 1234, "epochs_each": 1000})
        path = folder / "config.json"
        finalize.write_json(path, config)
        with patch.object(finalize.socket, "gethostname", return_value="test-instance"), \
                patch.object(finalize, "active_training", return_value=False) as active:
            yield folder, config, path, active
    finally:
        assert folder.parent == parent and folder.name.startswith("finalize_test_")
        shutil.rmtree(folder)


class TestStage2Finalizer(unittest.TestCase):
    def test_F_only_queue_can_finish_without_C(self):
        with fixture() as (_, config, _, _):
            config['jobs'] = config['jobs'][:1]
            finalize.write_json(config['queue_file'], {'status': 'completed', 'sequence': ['F'],
                                                     'seed': 1234, 'epochs_each': 1000})
            self.assertTrue(finalize.readiness(config)[0])

    def test_F_only_wait_state_does_not_wait_for_C(self):
        with fixture() as (_, config, _, _):
            config['jobs'] = config['jobs'][:1]
            finalize.write_json(config['queue_file'], {'status': 'training'})
            self.assertEqual(finalize.readiness(config), (False, 'waiting_for_F'))

    def test_waits_until_queue_completed(self):
        with fixture() as (_, config, _, _):
            finalize.write_json(config["queue_file"], {"status": "training"})
            self.assertEqual(finalize.readiness(config), (False, "waiting_for_F_and_C"))

    def test_failed_training_never_ready(self):
        with fixture() as (_, config, _, _):
            finalize.write_json(config["queue_file"], {"status": "failed"})
            self.assertFalse(finalize.readiness(config)[0])

    def test_live_training_or_gpu_blocks(self):
        with fixture() as (_, config, _, active):
            active.return_value = True
            self.assertFalse(finalize.readiness(config)[0])

    def test_cancel_file_blocks(self):
        with fixture() as (_, config, _, _):
            Path(config["cancel_file"]).touch()
            self.assertEqual(finalize.readiness(config), (False, "cancelled_by_sentinel"))

    def test_missing_C_final_blocks(self):
        with fixture() as (_, config, _, _):
            (Path(config["jobs"][1]["results_dir"]) / "fold_all/checkpoint_final.pth").unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing checkpoint"):
                finalize.readiness(config)

    def test_smoke_and_wrong_protocol_block(self):
        with fixture() as (_, config, _, _):
            path = Path(config["jobs"][1]["results_dir"]) / "fold_all/stage2_run.json"
            manifest = finalize.read_json(path)
            manifest["arguments"]["smoke"] = True
            finalize.write_json(path, manifest)
            with self.assertRaises(RuntimeError):
                finalize.readiness(config)

    def test_wrong_host_blocks(self):
        with fixture() as (_, config, _, _):
            config["hostname"] = "another-instance"
            with self.assertRaises(RuntimeError):
                finalize.readiness(config)

    def test_formal_final_checkpoint_epoch_required(self):
        with fixture() as (_, config, _, _):
            def load(path, **kwargs):
                return {"current_epoch": 999, "trainer_name": "Trainer" + path.parents[1].name,
                        "network_weights": {"x": 1}}
            with patch.dict("sys.modules", {"torch": SimpleNamespace(load=load)}):
                with self.assertRaisesRegex(RuntimeError, "required epoch"):
                    finalize.verify_checkpoints(config)

    def test_roundtrip_corruption_blocks(self):
        with fixture() as (folder, config, path, _):
            worker = finalize.Finalizer(config, path, True)
            archive = folder / "result.tar"
            archive.write_bytes(b"correct")
            def run(command, **kwargs):
                if command[2].startswith("oss://"):
                    Path(command[3]).write_bytes(b"corrupt")
            with patch.object(worker, "run", side_effect=run):
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    worker.upload_verify(archive)
            self.assertEqual(worker.state["verified"], {})

    def test_upload_failure_prevents_shutdown(self):
        with fixture() as (folder, config, path, _):
            worker = finalize.Finalizer(config, path, True)
            with patch.object(finalize, "verify_checkpoints", return_value=[]), \
                    patch.object(worker, "archive_job", return_value=folder / "result.tar"), \
                    patch.object(worker, "upload_verify", side_effect=RuntimeError("HTTP401")), \
                    patch.object(finalize.subprocess, "run") as command:
                with self.assertRaisesRegex(RuntimeError, "HTTP401"):
                    worker.finish()
                command.assert_not_called()

    def test_successful_roundtrip_records_proof_and_retains_archive(self):
        with fixture() as (folder, config, path, _):
            worker = finalize.Finalizer(config, path, True)
            archive = folder / "result.tar"
            archive.write_bytes(b"verified content")
            def run(command, **kwargs):
                if command[2].startswith("oss://"):
                    Path(command[3]).write_bytes(archive.read_bytes())
            with patch.object(worker, "run", side_effect=run):
                proof = worker.upload_verify(archive)
            self.assertEqual(proof["sha256"], finalize.sha256(archive))
            self.assertTrue(archive.exists())
            self.assertFalse((worker.work / "verify_result.tar").exists())

    def test_cancel_just_before_shutdown_is_respected(self):
        with fixture() as (folder, config, path, _):
            worker = finalize.Finalizer(config, path, True)
            def upload(archive):
                if Path(archive).name == "backup_receipt.tar":
                    Path(config["cancel_file"]).touch()
            with patch.object(finalize, "verify_checkpoints", return_value=[]), \
                    patch.object(worker, "archive_job", return_value=folder / "model.tar"), \
                    patch.object(worker, "archive_audit", return_value=folder / "audit.tar"), \
                    patch.object(worker, "upload_verify", side_effect=upload), \
                    patch.object(finalize.subprocess, "run") as command:
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    worker.finish()
                command.assert_not_called()

    def test_changed_shutdown_binary_blocks(self):
        with fixture() as (folder, config, path, _):
            worker = finalize.Finalizer(config, path, True)
            Path(config["shutdown_executable"]).write_bytes(b"unexpected binary")
            with patch.object(finalize, "verify_checkpoints", return_value=[]), \
                    patch.object(worker, "archive_job", return_value=folder / "model.tar"), \
                    patch.object(worker, "archive_audit", return_value=folder / "audit.tar"), \
                    patch.object(worker, "upload_verify"), patch.object(finalize.subprocess, "run") as command:
                with self.assertRaisesRegex(RuntimeError, "executable changed"):
                    worker.finish()
                command.assert_not_called()

    def test_success_requires_all_four_uploads_before_mock_shutdown(self):
        with fixture() as (folder, config, path, _):
            worker = finalize.Finalizer(config, path, True)
            events = []
            def upload(archive):
                events.append(Path(archive).name)
            with patch.object(finalize, "verify_checkpoints", return_value=[]), \
                    patch.object(worker, "archive_job", side_effect=lambda j: folder / (j["model"] + ".tar")), \
                    patch.object(worker, "archive_audit", return_value=folder / "audit.tar"), \
                    patch.object(worker, "upload_verify", side_effect=upload), \
                    patch.object(finalize.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as command:
                worker.finish()
                self.assertEqual(events, ["F.tar", "C.tar", "audit.tar", "backup_receipt.tar"])
                command.assert_called_once()
                self.assertEqual(command.call_args.args[0], [config["shutdown_executable"]])

    def test_not_armed_never_calls_shutdown(self):
        with fixture() as (folder, config, path, _):
            worker = finalize.Finalizer(config, path, False)
            with patch.object(finalize, "verify_checkpoints", return_value=[]), \
                    patch.object(worker, "archive_job", return_value=folder / "model.tar"), \
                    patch.object(worker, "archive_audit", return_value=folder / "audit.tar"), \
                    patch.object(worker, "upload_verify"), patch.object(finalize.subprocess, "run") as command:
                worker.finish()
                command.assert_not_called()
                self.assertEqual(worker.state["status"], "verified_not_armed")
                receipt = finalize.read_json(worker.work / "backup_receipt.json")
                self.assertFalse(receipt["shutdown_authorized"])
                self.assertIn("Not authorized", receipt["shutdown"])

    def test_explicit_backup_only_policy_rejects_arming(self):
        with fixture() as (_, config, path, _):
            config["shutdown_authorized"] = False
            finalize.write_json(path, config)
            with self.assertRaisesRegex(RuntimeError, "explicitly disabled"):
                finalize.Finalizer(config, path, True)

    def test_backup_only_does_not_require_a_shutdown_binary(self):
        with fixture() as (folder, config, path, _):
            Path(config["shutdown_executable"]).unlink()
            worker = finalize.Finalizer(config, path, False)
            with patch.object(finalize, "verify_checkpoints", return_value=[]), \
                    patch.object(worker, "archive_job", return_value=folder / "model.tar"), \
                    patch.object(worker, "archive_audit", return_value=folder / "audit.tar"), \
                    patch.object(worker, "upload_verify"), patch.object(finalize.subprocess, "run") as command:
                worker.finish()
                command.assert_not_called()
                self.assertEqual(worker.state["status"], "verified_not_armed")

    def test_changed_config_rejects_saved_proofs(self):
        with fixture() as (_, config, path, _):
            worker = finalize.Finalizer(config, path)
            worker.save("waiting")
            config["epochs"] = 2
            finalize.write_json(path, config)
            with self.assertRaisesRegex(RuntimeError, "configuration changed"):
                finalize.Finalizer(config, path)


if __name__ == "__main__":
    unittest.main()
