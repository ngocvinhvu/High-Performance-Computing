"""Shared execution, CPU core affinity, and report export helpers for ducnm benchmarks."""
import ctypes
import json
import os
import re
import statistics
import subprocess
from ctypes import wintypes
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / 'results'
EPOCH_PATTERN = re.compile(r'\[EPOCH\] ep=(\d+) loss=(\S+) val_acc=(\S+) test_acc=(\S+) elapsed_s=(\S+)')
SUMMARY_PATTERN = re.compile(r'\[SUMMARY\] total_s=(\S+) comm_s=(\S+) loss=(\S+) val_acc=(\S+) test_acc=(\S+)')

_k32 = ctypes.WinDLL('kernel32', use_last_error=True)
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]


def bind_cpu_cores(num_cores: int):
    """Bind current process and child processes to `num_cores` physical cores."""
    logical = os.cpu_count() or 8
    phys = max(1, logical // 2)
    active = min(max(1, num_cores), phys)
    mask = 0
    for c in range(active):
        mask |= (3 << (2 * c))
    _k32.SetProcessAffinityMask(_k32.GetCurrentProcess(), mask)


def bin_path(target_name: str) -> str:
    return str(ROOT_DIR / 'bin' / f'{target_name}.exe')


def invoke_binary(cores: int, cmd_list: list) -> str:
    bind_cpu_cores(cores)
    proc = subprocess.run(cmd_list, cwd=ROOT_DIR, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout


def run_training_job(runner_type: str, workers: int = 1, **cli_opts):
    """Execute one training run of cnn_cpu, cnn_dist, or cnn_gpu and return epoch list."""
    cmd = [bin_path(f'cnn_{runner_type}')]
    if runner_type == 'dist':
        cmd += ['--workers', str(workers)]
    for k, v in cli_opts.items():
        flag = k.replace('_', '-')
        if v == '' or v is True:
            cmd += [f'--{flag}']
        elif v is not False and v is not None:
            cmd += [f'--{flag}', str(v)]
    raw_out = invoke_binary(workers if runner_type == 'dist' else 1, cmd)
    return [
        dict(epoch=int(ep), loss=float(ls), val=float(va), test=float(te), time=float(dt))
        for ep, ls, va, te, dt in EPOCH_PATTERN.findall(raw_out)
    ]


def find_convergence_point(epoch_rows: list, target_acc: float):
    cum_time = 0.0
    for row in epoch_rows:
        cum_time += row['time']
        if row['val'] >= target_acc:
            return row['epoch'], cum_time
    return None


def calc_mean_std(items):
    seq = list(items)
    return statistics.fmean(seq), (statistics.stdev(seq) if len(seq) > 1 else 0.0)


def fmt_num(val: float, decimals: int = 2) -> str:
    return f'{val:.{decimals}f}'.replace('.', ',')


def export_report(stem: str, payload, md_lines: list):
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f'{stem}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8', newline='\n')
    (RESULTS_DIR / f'{stem}.md').write_text('\n'.join(md_lines) + '\n', encoding='utf-8', newline='\n')
