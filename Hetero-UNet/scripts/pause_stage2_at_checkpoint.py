"""Pause only a verified F trainer between checkpoint writes for migration.

External, reversible process control. No trainer/model source edits or new
checkpoint serialization are injected into the running training process.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--target-epoch', type=int, default=500)
    args = parser.parse_args()
    audit = Path(args.audit)
    spec = importlib.util.spec_from_file_location('queue_identity', audit / 'hold_stage2_queue.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    plan = json.loads((audit / 'F_only_control.json').read_text())
    expected = plan['trainer_process']
    folder = Path(plan['F_manifest']).parent
    latest = folder / 'checkpoint_latest.pth'
    log = audit / 'main_queue.log'
    report_path = audit / 'migration_pause.json'
    if report_path.exists():
        raise RuntimeError('Pause report already exists; review before repeating')
    (audit / 'DISABLE_F_ONLY_AUTOSHUTDOWN').touch()
    print('Waiting for a sealed checkpoint and the next epoch training phase', flush=True)
    while True:
        actual = helper.identity(expected['pid'])
        if not helper.same_process(actual, expected):
            raise RuntimeError('Original F process identity changed; no signal sent')
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 65536))
            tail = stream.read().decode(errors='replace')
        starts = list(re.finditer(r': Epoch (\d+)\b', tail))
        if starts:
            last = starts[-1]
            epoch = int(last.group(1))
            phase = tail[last.end():]
            # on_epoch_end logs losses before serializing best/latest. A new
            # epoch marker proves previous checkpoint writes have returned.
            if (epoch >= args.target_epoch and 'train_loss' not in phase and
                    'val_loss' not in phase and latest.is_file()):
                fd = os.pidfd_open(expected['pid'])
                try:
                    if not helper.same_process(helper.identity(expected['pid']), expected):
                        raise RuntimeError('F PID changed while opening pidfd')
                    signal.pidfd_send_signal(fd, signal.SIGSTOP)
                finally:
                    os.close(fd)
                for _ in range(50):
                    if helper.identity(expected['pid'])['state'] in ('T', 't'):
                        break
                    time.sleep(.02)
                assert helper.identity(expected['pid'])['state'] in ('T', 't')
                # These files are now stable. Load on CPU, without touching GPU.
                import torch
                reports = []
                for name in ('checkpoint_latest.pth', 'checkpoint_best.pth'):
                    path = folder / name
                    checkpoint = torch.load(path, map_location='cpu')
                    if checkpoint['trainer_name'] != 'nnUNetTrainerMambaUNetBlockAttnRes':
                        raise RuntimeError('Unexpected checkpoint Trainer; process remains paused')
                    if name == 'checkpoint_latest.pth' and checkpoint['current_epoch'] < args.target_epoch:
                        raise RuntimeError('Checkpoint is too old; process remains paused for review')
                    if not checkpoint.get('network_weights') or not checkpoint.get('optimizer_state'):
                        raise RuntimeError('Missing network/optimizer state; process remains paused')
                    reports.append({'file': name, 'epoch': checkpoint['current_epoch'],
                                    'best_ema': float(checkpoint['_best_ema']), 'bytes': path.stat().st_size,
                                    'logging_epochs': len(checkpoint['logging']['train_losses'])})
                    del checkpoint
                report = {'status': 'paused_for_migration',
                          'paused_utc': datetime.now(timezone.utc).isoformat(),
                          'trainer_process': expected, 'observed_running_epoch_zero_based': epoch,
                          'resume_from': 'checkpoint_latest.pth', 'checkpoints': reports,
                          'target_total_epochs': 1000, 'deferred_models': ['C'],
                          'original_model_source_modified': False,
                          'note': 'In-progress iterations after the saved epoch are discarded on migration; RNG not bitwise restored.'}
                helper.write_json(report_path, report)
                # Accurate metadata for the interrupted first run; its model,
                # optimizer and checkpoint contents are not modified here.
                manifest = json.loads(Path(plan['F_manifest']).read_text())
                manifest['status'] = 'paused_for_migration'
                manifest['migration_pause_report'] = str(report_path)
                helper.write_json(plan['F_manifest'], manifest)
                history = folder / 'stage2_run_manifests' / (manifest['run_id'] + '.json')
                if history.exists():
                    helper.write_json(history, manifest)
                print(json.dumps(report), flush=True)
                return
        time.sleep(1)


if __name__ == '__main__':
    main()
