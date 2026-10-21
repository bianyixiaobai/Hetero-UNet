"""Retire a held Bash queue after its current F child exits; never signal F.

The queue's individual PID is held with SIGSTOP. Its already running trainer
and data loaders remain runnable. Only that same stopped queue process is
retired, after the F manifest is completed and the original trainer has exited.
No shutdown command is present in this controller.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time


def identity(pid):
    try:
        root = Path('/proc') / str(pid)
        stat = (root / 'stat').read_text()
        fields = stat[stat.rindex(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'state': fields[0], 'ppid': int(fields[1]),
                'start_ticks': int(fields[19]),
                'argv': [s.decode() for s in (root / 'cmdline').read_bytes().split(b'\0') if s]}
    except FileNotFoundError:
        return None


def same_process(actual, expected):
    return actual is not None and all(actual[k] == expected[k] for k in
                                      ('pid', 'ppid', 'start_ticks', 'argv'))


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, path)


def signal_queue(plan, sig, require_stopped=False):
    """pidfd plus full identity checks prevent targeting a reused PID/group."""
    if socket.gethostname() != plan['hostname']:
        raise RuntimeError('Wrong instance')
    expected = plan['queue_process']
    with_fd = os.pidfd_open(expected['pid'])
    try:
        current = identity(expected['pid'])
        if not same_process(current, expected):
            raise RuntimeError('Queue identity changed; no signal sent')
        if current['argv'] != ['bash', plan['original_queue_script']]:
            raise RuntimeError('Only the original Bash queue may be signalled')
        if require_stopped and current['state'] not in ('T', 't'):
            raise RuntimeError('Queue is not held; no termination signal sent')
        signal.pidfd_send_signal(with_fd, sig)
    finally:
        os.close(with_fd)


def hold(plan):
    trainer = identity(plan['trainer_process']['pid'])
    if not same_process(trainer, plan['trainer_process']):
        raise RuntimeError('F trainer identity changed')
    if trainer['ppid'] != plan['queue_process']['pid']:
        raise RuntimeError('Trainer is not the direct child of the expected queue')
    args = trainer['argv']
    if not ('scripts/run_stage2.py' in args and '--model' in args and
            args[args.index('--model') + 1] == 'F'):
        raise RuntimeError('The current trainer is not F')
    signal_queue(plan, signal.SIGSTOP)
    print('Held the Bash queue only; F trainer was not signalled', flush=True)


def step(plan):
    if socket.gethostname() != plan['hostname']:
        raise RuntimeError('Wrong instance')
    manifest = json.loads(Path(plan['F_manifest']).read_text())
    status = manifest.get('status')
    queue_state = {'sequence': ['F'], 'seed': 1234, 'epochs_each': 1000,
                   'deferred_models': ['C'], 'original_queue_retired': False,
                   'models': {'F': status}, 'status': 'training', 'current_model': 'F'}
    if status == 'failed':
        queue_state['status'] = 'failed'
        write_json(plan['F_only_queue_file'], queue_state)
        return 'failed'
    if status != 'completed' or same_process(identity(plan['trainer_process']['pid']), plan['trainer_process']):
        write_json(plan['F_only_queue_file'], queue_state)
        return 'waiting_for_F'
    if (manifest['arguments']['model'] != 'F' or manifest['arguments']['smoke'] or
            manifest['protocol']['num_epochs'] != 1000 or
            manifest['git']['commit'] != plan['training_commit']):
        raise RuntimeError('Unexpected completed F manifest')
    current = identity(plan['queue_process']['pid'])
    if current is not None:
        if not same_process(current, plan['queue_process']):
            raise RuntimeError('Queue PID reused; requires review')
        # F has exited. Kill only the held queue PID so C cannot be dispatched.
        signal_queue(plan, signal.SIGKILL, require_stopped=True)
        return 'retiring_completed_queue'
    queue_state.update(status='completed', current_model='none', original_queue_retired=True)
    write_json(plan['F_only_queue_file'], queue_state)
    return 'completed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--hold', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if args.hold:
        hold(plan)
    if not args.watch:
        return
    import fcntl
    with open(args.plan + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            try:
                state = step(plan)
                if state != previous:
                    print(state, flush=True)
                    previous = state
                if state in ('completed', 'failed'):
                    return
            except Exception as error:
                print(type(error).__name__, str(error), flush=True)
            time.sleep(15)


if __name__ == '__main__':
    main()
