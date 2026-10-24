"""Wait for completed F/C evaluation, verify full OSS delivery, then power off.

Independent of the running evaluator. Requires explicit --watch --arm-shutdown.
Cancellation: create DISABLE_EVALUATION_SHUTDOWN in the evaluation directory.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics
import subprocess
import tarfile
import time
import traceback

ORGANS = ['Liver', 'RK', 'Spleen', 'Pancreas', 'Aorta', 'IVC', 'RAG', 'LAG',
          'Gallbladder', 'Esophagus', 'Stomach', 'Duodenum', 'LK']
TERMINAL = {'completed', 'completed_backup_pending'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def ready(state):
    return (state.get('status') in TERMINAL and state.get('sequence') == ['F', 'C']
            and state.get('current_model') is None
            and all(state.get('models', {}).get(m, {}).get('evaluation') == 'completed' for m in ['F', 'C']))


def table(path, metric, names):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['Name'] + [o + '_' + metric for o in ORGANS], 'Unexpected metric columns')
        rows = list(reader)
    require(len(rows) == len(names) == 50, 'Incomplete 50-case table')
    require(len({r['Name'] for r in rows}) == 50 and {r['Name'] for r in rows} == set(names), 'Case IDs differ')
    result = {r['Name']: [float(r[o + '_' + metric]) for o in ORGANS] for r in rows}
    require(all(math.isfinite(v) and 0 <= v <= 1 for row in result.values() for v in row), 'Invalid metric value')
    return result


def summarize(work, protocol):
    """Recompute paired summaries from original CSVs, preserving the raw 50x13 data."""
    names = sorted(r['id'] + '.nii.gz' for r in protocol['cases'])
    results, organ_rows = {}, []
    for metric in ['DSC', 'NSD']:
        frames = {'A': table(work / 'baseline_A' / (metric + '.csv'), metric, names)}
        for model in ['F', 'C']:
            frames[model] = table(work / model / (model + '_' + metric + '.csv'), metric, names)
        means = {m: statistics.mean(v for row in values.values() for v in row) * 100 for m, values in frames.items()}
        rows = []
        for name in names:
            a, f, c = [statistics.mean(frames[m][name]) * 100 for m in ['A', 'F', 'C']]
            rows.append([name, a, f, c, f - a, c - a, c - f])
        with (work / ('final_paired_' + metric + '.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Name', 'A_percent', 'F_percent', 'C_percent', 'F_minus_A_pp', 'C_minus_A_pp', 'C_minus_F_pp'])
            writer.writerows(rows)
        for i, organ in enumerate(ORGANS):
            a, f, c = [statistics.mean(frames[m][name][i] for name in names) * 100 for m in ['A', 'F', 'C']]
            organ_rows.append([metric, organ, a, f, c, f - a, c - a, c - f])
        results[metric] = {'macro_percent': means, 'F_minus_A_pp': means['F'] - means['A'],
                           'C_minus_A_pp': means['C'] - means['A'], 'C_minus_F_pp': means['C'] - means['F']}
        for model in ['F', 'C']:
            previous = read(work / model / 'summary.json')[metric + '_percent']
            require(abs(previous - means[model]) < 1e-9, 'Metric summary does not match raw CSV')
    with (work / 'per_organ_comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Metric', 'Organ', 'A_percent', 'F_percent', 'C_percent', 'F_minus_A_pp', 'C_minus_A_pp', 'C_minus_F_pp'])
        writer.writerows(organ_rows)
    write(work / 'final_comparison.json', results)
    lines = ['# F/C外部50例评价结果汇总', '',
             '评价数据为同一50例CT、13器官等权宏平均；使用各自best、无TTA、滑窗0.5及相同低内存导出和评价脚本。', '',
             '| 指标 | A基线（%） | F（%） | C（%） | F−A（百分点） | C−A（百分点） | C−F（百分点） |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for metric, row in results.items():
        a, f, c = [row['macro_percent'][m] for m in ['A', 'F', 'C']]
        lines.append(f'| {metric} | {a:.4f} | {f:.4f} | {c:.4f} | {f-a:+.4f} | {c-a:+.4f} | {c-f:+.4f} |')
    lines += ['', '逐病例配对结果见final_paired_DSC.csv、final_paired_NSD.csv，逐器官结果见per_organ_comparison.csv；原始CSV不覆盖。', '',
              '本次为seed1234单种子探索。F曾从4090迁移V100续训，C在V100从头训练；不能将单次差异写成统计显著或稳定增益。',
              '模型与训练协议保持原MHA；F-aux和关键多种子确认仍按原工作计划安排。外部50例未用于调参或选择检查点。', '',
              '用户于2026-10-24授权：C评价完成、记录与数据汇总并通过OSS备份回读后关机。本报告在关机前生成，关机请求以finalization_state.json及收据为准。', '']
    (work / 'EVALUATION_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return results


class Finalizer:
    def __init__(self, config_path, armed=False):
        self.config_path = Path(config_path)
        self.config = read(config_path)
        self.work = Path(self.config['work_dir'])
        require(not armed or self.config.get('shutdown_authorized') is True, 'Shutdown not authorized')
        self.armed = armed
        self.state = {'status': 'initialized', 'verified': {}, 'shutdown_authorized': armed,
                      'config_sha256': sha(config_path), 'created_utc': datetime.now(timezone.utc).isoformat()}

    def save(self, status, **extra):
        self.state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write(self.work / 'finalization_state.json', self.state)
        print(status, flush=True)

    def guard(self):
        require(not (self.work / 'DISABLE_EVALUATION_SHUTDOWN').exists(), 'Cancelled by sentinel')
        require(socket.gethostname() == self.config['hostname'], 'Instance mismatch')
        require(sha(self.work / 'protocol.json') == self.config['protocol_sha256'], 'Frozen protocol changed')
        require(ready(read(self.work / 'state.json')), 'Evaluation not complete')
        processes = subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines()
        require(not any('run_stage2.py' in p and '--model' in p for p in processes), 'A trainer is still active')

    def upload_verify(self, path):
        remote = self.config['oss_prefix'].rstrip('/') + '/' + path.name
        returned = self.work / ('verify_final_' + path.name)
        with (self.work / 'finalization_transfers.log').open('ab', buffering=0) as log:
            for args in [['oss', 'cp', str(path), remote, '-f'], ['oss', 'cp', remote, str(returned), '-f']]:
                subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, timeout=1800)
        checksum = sha(path)
        require(returned.stat().st_size == path.stat().st_size and sha(returned) == checksum, 'OSS round-trip mismatch')
        returned.unlink()
        proof = {'remote': remote, 'bytes': path.stat().st_size, 'sha256': checksum, 'roundtrip_passed': True}
        self.state['verified'][path.name] = proof
        self.save('object_verified')
        return proof

    def finish(self):
        self.guard()
        protocol = read(self.work / 'protocol.json')
        names = {r['id'] + '.nii.gz' for r in protocol['cases']}
        # Recheck exported files against the evaluator's complete geometry/hash manifest.
        for model in ['F', 'C']:
            folder = self.work / model
            report = read(folder / 'prediction_validation.json')
            require(report['status'] == 'passed' and len(report['cases']) == 50, 'Incomplete prediction audit')
            require({p.name for p in (folder / 'predictions').glob('*.nii.gz')} == names, 'Missing predictions')
            require({r['case'] + '.nii.gz' for r in report['cases']} == names, 'Prediction audit IDs differ')
            for row in report['cases']:
                path = folder / 'predictions' / (row['case'] + '.nii.gz')
                require(row['geometry_matches_gt'] and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Prediction changed')
        results = summarize(self.work, protocol)
        self.save('summarized', comparison=results)
        # TAR is accepted by this OSS CLI; individual JSON uploads are unsupported.
        for model in ['F', 'C']:
            archive = self.work / (model + '_external50_verified.tar')
            with tarfile.open(archive, 'w') as output:
                output.add(self.work / model, arcname=model)
                for name in ['protocol.json', 'baseline_A', 'metrics_code', 'export_equivalence.log']:
                    output.add(self.work / name, arcname=name)
            self.guard()
            self.upload_verify(archive)
        package = self.work / 'F_C_evaluation_final_records.tar'
        allow = ['protocol.json', 'state.json', 'final_comparison.json', 'comparison.json',
                 'paired_DSC.csv', 'paired_NSD.csv', 'final_paired_DSC.csv', 'final_paired_NSD.csv',
                 'per_organ_comparison.csv', 'EVALUATION_RESULTS.md', 'queue.log', 'backup.log',
                 'finalization_state.json', 'finalization.log', 'finalization_transfers.log',
                 'WORK_PLAN_v1.15_20261024.md', 'WORK_PLAN_v1.16_20261024.md', 'WORK_PLAN_v1.17_20261024.md',
                 'F_C_EVALUATION_2026-10-24.md', 'EVALUATION_SHUTDOWN_2026-10-24.md',
                 'local_backup_check.json', 'export_equivalence.log', 'launch_check.json',
                 'prepare_stage2_evaluation_20261024.py', 'run_stage2_evaluation_20261024.py',
                 'finalize_stage2_evaluation_20261024.py', 'evaluation_shutdown_config.json']
        with tarfile.open(package, 'w') as output:
            for name in allow:
                path = self.work / name
                if path.is_file():
                    output.add(path, arcname=name)
            output.add(self.work / 'runtime/nnunetv2/inference/export_prediction.py', arcname='runtime/export_prediction.py')
        self.upload_verify(package)
        receipt = self.work / 'evaluation_final_receipt.json'
        write(receipt, {**self.state, 'status': 'all_results_verified',
                        'user_authorization': self.config['user_authorization'],
                        'note': 'Shutdown may only be requested after this receipt also passes round-trip verification.'})
        receipt_package = self.work / 'evaluation_final_receipt.tar'
        with tarfile.open(receipt_package, 'w') as output:
            output.add(receipt, arcname=receipt.name)
            output.add(self.config_path, arcname=self.config_path.name)
        self.upload_verify(receipt_package)
        self.guard()
        require(all(name in self.state['verified'] for name in ['F_external50_verified.tar', 'C_external50_verified.tar',
                    'F_C_evaluation_final_records.tar', 'evaluation_final_receipt.tar']), 'Incomplete delivery')
        if not self.armed:
            self.save('verified_not_armed')
            return
        executable = Path(self.config['shutdown_executable'])
        require(sha(executable) == self.config['shutdown_sha256'], 'Platform shutdown executable changed')
        self.guard()
        self.save('shutdown_requested', shutdown_executable=str(executable))
        subprocess.run([str(executable)], check=True, stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        self.save('shutdown_command_returned', note='Platform power state cannot be independently confirmed here')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--arm-shutdown', action='store_true')
    args = parser.parse_args()
    if args.arm_shutdown and not args.watch:
        parser.error('--arm-shutdown requires --watch')
    helper = Finalizer(args.config, args.arm_shutdown)
    if not args.watch:
        print(json.dumps({'evaluation_ready': ready(read(helper.work / 'state.json')),
                          'shutdown_armed': False}))
        return
    import fcntl
    import os
    os.nice(15)
    with (helper.work / 'evaluation_finalizer.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                if (helper.work / 'DISABLE_EVALUATION_SHUTDOWN').exists():
                    helper.save('cancelled')
                    return
                state = read(helper.work / 'state.json')
                if state.get('status') == 'failed':
                    helper.save('evaluation_failed_requires_review')
                    return
                if not ready(state):
                    helper.save('waiting_for_F_C_evaluation', current_model=state.get('current_model'))
                else:
                    with (helper.work / 'evaluation.lock').open('a') as evaluation:
                        try:
                            fcntl.flock(evaluation, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            helper.save('waiting_for_evaluator_exit')
                        else:
                            helper.finish()
                            return
            except Exception:
                helper.save('backup_or_validation_retry_required', error=traceback.format_exc())
            time.sleep(helper.config.get('poll_seconds', 60))


if __name__ == '__main__':
    main()
