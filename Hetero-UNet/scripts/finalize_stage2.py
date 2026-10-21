"""Wait for the configured queue, verify OSS backups, then stop if authorized.

Separate from the trainer. Default invocation is a read-only readiness check.
Only --watch --arm-shutdown permits the configured platform shutdown command.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def auth_expiry(path):
    """Report expiry only; credentials never enter state files or logs."""
    try:
        entries = dict(line.split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line)
        payload = entries["hyCloudToken"].split(".")[1]
        claim = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return datetime.fromtimestamp(claim["exp"], timezone.utc).isoformat()
    except (OSError, KeyError, ValueError, IndexError):
        return None


def active_training(config):
    """No GPU framework is imported, and no worker is signalled."""
    session = subprocess.run(["tmux", "has-session", "-t", config["training_session"]],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if session.returncode == 0:
        return True
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
    if gpu.returncode:
        raise RuntimeError("Cannot determine whether GPU work is still active")
    if gpu.stdout.strip():
        return True
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = proc.read_bytes().split(b"\0")
        except OSError:
            continue
        if any(Path(arg.decode(errors="replace")).name in
               {"run_stage2.py", "nnUNetv2_train", "run_training.py"} for arg in args):
            return True
    return False


def readiness(config):
    if socket.gethostname() != config["hostname"]:
        raise RuntimeError("Instance hostname does not match the armed configuration")
    if Path(config["cancel_file"]).exists():
        return False, "cancelled_by_sentinel"
    queue = read_json(config["queue_file"])
    if queue.get("status") == "failed":
        return False, "training_failed_requires_review"
    names = [job["model"] for job in config["jobs"]]
    if queue.get("status") != "completed":
        return False, "waiting_for_" + "_and_".join(names)
    if (queue.get("sequence") != names or queue.get("seed") != config["seed"] or
            queue.get("epochs_each") != config["epochs"]):
        raise RuntimeError("Completed queue does not match the configured frozen protocol")
    if active_training(config):
        return False, "waiting_for_training_processes_to_exit"
    for job in config["jobs"]:
        root = Path(job["results_dir"])
        manifest = read_json(root / "fold_all/stage2_run.json")
        protocol = manifest["protocol"]
        arguments = manifest["arguments"]
        if not (manifest.get("status") == "completed" and
                manifest["git"]["commit"] == config["training_commit"] and
                arguments["model"] == job["model"] and not arguments["smoke"] and
                protocol["seed"] == config["seed"] and
                protocol["trainer"] == job["trainer"] and
                protocol["num_epochs"] == config["epochs"] and
                protocol["train_iterations_per_epoch"] == 250 and
                protocol["validation_iterations_per_epoch"] == 50 and
                protocol["batch_size"] == 2 and protocol["patch_size"] == [40, 224, 192] and
                protocol["initial_lr"] == 0.01 and protocol["mha_batch_first"] is False):
            raise RuntimeError("Incomplete or inconsistent formal manifest: " + job["model"])
        for name in ("checkpoint_best.pth", "checkpoint_final.pth"):
            if not (root / "fold_all" / name).is_file():
                raise RuntimeError("Missing checkpoint: " + job["model"] + "/" + name)
    return True, "ready_for_backup"


def verify_checkpoints(config):
    # Called only after the configured trainers and queue have exited. CPU-only loads.
    import torch
    reports = []
    for job in config["jobs"]:
        for name in ("checkpoint_best.pth", "checkpoint_final.pth"):
            path = Path(job["results_dir"]) / "fold_all" / name
            checkpoint = torch.load(path, map_location="cpu")
            epoch = checkpoint["current_epoch"]
            if checkpoint["trainer_name"] != job["trainer"]:
                raise RuntimeError("Checkpoint trainer mismatch: " + str(path))
            if not (1 <= epoch <= config["epochs"]):
                raise RuntimeError("Checkpoint epoch out of range: " + str(path))
            if name == "checkpoint_final.pth" and epoch != config["epochs"]:
                raise RuntimeError("Formal final checkpoint has not reached the required epoch")
            if not checkpoint.get("network_weights"):
                raise RuntimeError("Empty network state: " + str(path))
            reports.append({"model": job["model"], "checkpoint": name,
                            "epoch": epoch, "trainer": checkpoint["trainer_name"]})
            del checkpoint
    return reports


class Finalizer:
    def __init__(self, config, config_path, armed=False):
        self.config = config
        self.config_path = Path(config_path)
        self.armed = armed
        self.work = Path(config["archive_dir"])
        self.work.mkdir(parents=True, exist_ok=True)
        self.state_path = Path(config["state_file"])
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            "created_utc": utc_now(), "verified": {}, "archives": {},
        }
        digest = sha256(self.config_path)
        if self.state.get("config_sha256", digest) != digest:
            raise RuntimeError("Watcher configuration changed; review the existing backup state")
        self.state["config_sha256"] = digest

    def save(self, status, **details):
        previous = self.state.get("status")
        self.state.update(status=status, updated_utc=utc_now(), **details)
        write_json(self.state_path, self.state)
        if previous != status:
            print(utc_now(), status, flush=True)

    def safe_to_continue(self):
        ready, reason = readiness(self.config)
        if not ready:
            raise RuntimeError("Finalization interlock: " + reason)

    def run(self, command, timeout=7200):
        # OSS diagnostics stay on the instance; no token/config is archived.
        with (self.work / "transfers.log").open("ab", buffering=0) as log:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError("Transfer or packaging command failed; inspect transfers.log")

    def archive_job(self, job):
        key = job["model"]
        previous = self.state["archives"].get(key)
        if previous:
            path = Path(previous["path"])
            if path.is_file() and sha256(path) == previous["sha256"]:
                return path
            raise RuntimeError("A previously sealed archive changed; review before proceeding")
        root = Path(job["results_dir"])
        files = sorted(p for p in root.rglob("*") if p.is_file())
        if any(p.is_symlink() for p in root.rglob("*")):
            raise RuntimeError("Refusing result symlinks")
        before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
        needed = sum(size for size, _ in before.values())
        if shutil.disk_usage(self.work).free < 2 * needed + 1024 ** 3:
            raise RuntimeError("Insufficient disk for archive and download verification")
        path = self.work / (key + "_seed1234_final.tar")
        temporary = path.with_suffix(".tar.partial")
        # A single low-priority worker packages only completed model results.
        self.run(["tar", "-cf", str(temporary), "-C", str(root.parent), root.name])
        after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                 for p in root.rglob("*") if p.is_file()}
        if before != after:
            raise RuntimeError("Results changed during packaging")
        os.replace(temporary, path)
        self.state["archives"][key] = {"path": str(path), "bytes": path.stat().st_size,
                                        "sha256": sha256(path), "files": len(files)}
        self.save("packaging")
        return path

    def upload_verify(self, path):
        path = Path(path)
        checksum = sha256(path)
        remote = self.config["oss_prefix"].rstrip("/") + "/" + path.name
        old = self.state["verified"].get(path.name)
        if old and old["sha256"] == checksum and old["remote"] == remote:
            return old
        self.safe_to_continue()
        self.run([self.config["oss_executable"], "cp", str(path), remote, "-f"])
        downloaded = self.work / ("verify_" + path.name)
        self.run([self.config["oss_executable"], "cp", remote, str(downloaded), "-f"])
        if downloaded.stat().st_size != path.stat().st_size or sha256(downloaded) != checksum:
            raise RuntimeError("OSS round-trip size/SHA256 mismatch; shutdown is blocked")
        proof = {"remote": remote, "bytes": path.stat().st_size,
                 "sha256": checksum, "roundtrip_verified_utc": utc_now()}
        self.state["verified"][path.name] = proof
        self.save("backup_verified")
        downloaded.unlink()  # Only the redundant downloaded verification copy.
        return proof

    def archive_audit(self):
        path = self.work / "final_audit.tar"
        if path.exists():
            return path
        source = self.work / "training_source.tar"
        self.run(["git", "-C", self.config["source_repo"], "archive", "--format=tar",
                  "--output=" + str(source), self.config["training_commit"]])
        # Explicit audit allowlist avoids OSS configs, passwords and unrelated data.
        with tarfile.open(path.with_suffix(".tar.partial"), "w") as archive:
            archive.add(source, arcname="training_source.tar")
            archive.add(self.config_path, arcname="autoshutdown/config.json")
            archive.add(Path(__file__), arcname="autoshutdown/finalize_stage2.py")
            archive.add(self.work / "checkpoint_validation.json", arcname="checkpoint_validation.json")
            for item in sorted(Path(self.config["audit_dir"]).iterdir()):
                if (item.is_file() and not item.is_symlink() and
                        item.suffix in {".json", ".log", ".txt", ".sh", ".py", ".md"} and
                        item.stat().st_size < 100 * 1024 ** 2):
                    archive.add(item, arcname="audit/" + item.name)
            for item in sorted((Path(self.config["audit_dir"]) / "cuda_full_patch").glob("*.json")):
                archive.add(item, arcname="audit/cuda_full_patch/" + item.name)
        os.replace(path.with_suffix(".tar.partial"), path)
        return path

    def finish(self):
        self.safe_to_continue()
        self.save("checking_completed_checkpoints")
        report = verify_checkpoints(self.config)
        write_json(self.work / "checkpoint_validation.json", report)
        for job in self.config["jobs"]:
            self.upload_verify(self.archive_job(job))
        self.upload_verify(self.archive_audit())
        receipt = self.work / "backup_receipt.tar"
        if not receipt.exists():
            write_json(self.work / "backup_receipt.json", {
                "created_utc": utc_now(), "training_commit": self.config["training_commit"],
                "checkpoints": report, "objects": self.state["verified"],
                "scope": ",".join(job["model"] for job in self.config["jobs"]) +
                         " training complete; external 50-case evaluation remains separate",
                "deferred_models": self.config.get("deferred_models", []),
                "shutdown": "Authorized only after this receipt also passes round-trip verification",
            })
            with tarfile.open(receipt, "w") as archive:
                archive.add(self.work / "backup_receipt.json", arcname="backup_receipt.json")
        self.upload_verify(receipt)
        self.safe_to_continue()
        executable = Path(self.config["shutdown_executable"])
        if sha256(executable) != self.config["shutdown_sha256"]:
            raise RuntimeError("Platform shutdown executable changed; refusing invocation")
        if not self.armed:
            self.save("verified_not_armed")
            return
        self.save("shutdown_requested", shutdown_command=str(executable))
        # Recheck the cancellation file and active queue immediately before invocation.
        self.safe_to_continue()
        result = subprocess.run([str(executable)], stdin=subprocess.DEVNULL, timeout=120,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Platform shutdown command returned an error")
        self.save("shutdown_command_returned", note="Command returned; platform power state not independently confirmed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--arm-shutdown", action="store_true")
    args = parser.parse_args(argv)
    if args.arm_shutdown and not args.watch:
        parser.error("--arm-shutdown requires --watch")
    config = read_json(args.config)
    if not args.watch:
        ready, reason = readiness(config)
        print(json.dumps({"ready": ready, "reason": reason,
                          "oss_token_expiry_utc": auth_expiry(config["oss_config"])}))
        return 0
    # A Linux file lock prevents duplicate watchers and duplicate shutdown attempts.
    import fcntl
    with open(config["lock_file"], "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finalizer = Finalizer(config, args.config, args.arm_shutdown)
        os.nice(15)
        while True:
            try:
                ready, reason = readiness(config)
                finalizer.state["oss_token_expiry_utc"] = auth_expiry(config["oss_config"])
                if reason == "cancelled_by_sentinel":
                    finalizer.save("cancelled_by_sentinel")
                    return 0
                if not ready:
                    finalizer.save(reason)
                else:
                    finalizer.finish()
                    return 0
            except Exception as error:
                finalizer.save("blocked_retrying", error=type(error).__name__ + ": " + str(error))
                print(utc_now(), type(error).__name__, str(error), flush=True)
            time.sleep(config.get("poll_seconds", 60))


if __name__ == "__main__":
    raise SystemExit(main())
