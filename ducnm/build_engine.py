"""Compile all C / Multi-Worker / CUDA binaries into bin/: python build_engine.py"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
CFLAGS = ['-std=gnu11', '-O3', '-ffast-math', '-Wall', '-Wextra']


def resolve_c_compiler():
    gcc_bin = shutil.which('gcc')
    if not gcc_bin:
        winget_pkgs = Path(os.environ.get('LOCALAPPDATA', '')) / 'Microsoft/WinGet/Packages'
        gcc_bin = next(iter(winget_pkgs.glob('BrechtSanders.WinLibs*/mingw64/bin/gcc.exe')), None)
    if gcc_bin:
        return [str(gcc_bin), *CFLAGS, '-static']
    return [sys.executable, '-m', 'ziglang', 'cc', *CFLAGS]


def copy_nvrtc_runtime(out_dir: Path):
    try:
        import nvidia.cuda_nvrtc as nvrtc_mod
        dll_dir = Path(nvrtc_mod.__file__).resolve().parent / 'bin'
        if dll_dir.exists():
            for f in dll_dir.glob('nvrtc*.dll'):
                dst = out_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
    except ImportError:
        pass


def main():
    cc_cmd = resolve_c_compiler()
    out_dir = WORKSPACE / 'bin'
    out_dir.mkdir(exist_ok=True)
    copy_nvrtc_runtime(out_dir)

    core_src = 'src/core/lenet_gemm.c'
    targets = [
        ('cnn_cpu',     ['src/cpu_runner.c', core_src], []),
        ('cnn_dist',    ['src/dist_worker.c', core_src], []),
        ('cnn_gpu',     ['src/gpu_runner.c', core_src], []),
        ('verify_grad', ['src/verify_grad.c', core_src], ['-DVERIFY_FP64']),
    ]

    for binary, srcs, extra_flags in targets:
        cmd = [*cc_cmd, *srcs, *extra_flags, '-o', f'bin/{binary}.exe']
        subprocess.run(cmd, cwd=WORKSPACE, check=True)
        print(f'[BUILD_OK] bin/{binary}.exe', flush=True)


if __name__ == '__main__':
    main()
