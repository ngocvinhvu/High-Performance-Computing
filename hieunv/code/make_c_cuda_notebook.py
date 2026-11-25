"""Build the Kaggle notebook for the library-free version (../c_cuda): sequential C on the CPU, hand-written CUDA on the GPU.

    python make_c_cuda_notebook.py        # -> ../kaggle/hpc_c_cuda_kaggle.ipynb

The notebook writes the sources with %%writefile, builds them with gcc and nvcc, prints which shared libraries the
two programs load (no BLAS, no cuBLAS/cuDNN, no OpenMP), runs the correctness checks, then 3 runs: GPU first with the
NVML clock log, then the CPU. Separate from the PyTorch notebooks, so none of them changes.
"""
import json
from pathlib import Path

from make_kaggle_notebook import WORK, code, md

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "c_cuda"
OUT = HERE.parent / "kaggle" / "hpc_c_cuda_kaggle.ipynb"
SOURCES = [SRC / "common.h", SRC / "bench_seq.c", SRC / "bench_cuda.cu", SRC / "Makefile", HERE / "gpu_sampler.py"]
RUNS, GPU_REPEAT, CPU_REPEAT = 3, 30, 10


def build():
    cells = [
        md(f"""
# Library-free replication: sequential C (CPU) vs hand-written CUDA (GPU), Kaggle T4

Same experiments as the PyTorch notebooks (paper section 3: conv layer vs batch, FC matvec vs n, plus the bandwidth
checks), but every kernel is written by hand: `bench_seq` is plain C on one thread (no BLAS, no OpenMP),
`bench_cuda` uses only the CUDA runtime (no cuBLAS, no cuDNN). {RUNS} runs, GPU first ({GPU_REPEAT} repeats per point),
then CPU ({CPU_REPEAT} repeats). Estimated 30–45 minutes (not measured yet).

**Before running**
1. Settings → Accelerator → **GPU T4 x2**.
2. Run All. A failing build or correctness check stops the notebook before anything is timed.
3. Download `results_c_cuda.zip` from the Output panel, then stop the session.

The plan and the predictions are in `PLAN_C_CUDA.md` in the repository, written before this notebook was run.
The cells below are generated from `c_cuda/` and `code/gpu_sampler.py` by `code/make_c_cuda_notebook.py`.
"""),
        md("## 1 · Machine and tools"),
        code(f"""
import os
os.environ["PATH"] = os.environ.get("PATH", "") + ":/usr/local/cuda/bin"
!nvidia-smi
!lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core|Socket|L1d|L2|L3|Flags' | cut -c1-300
!gcc --version | head -1
!nvcc --version | tail -2
!mkdir -p {WORK}/c_cuda
"""),
        md("## 2 · Sources (written to `/kaggle/working/c_cuda`)"),
    ]
    for path in SOURCES:
        cells.append(code(f"%%writefile {WORK}/c_cuda/{path.name}\n{path.read_text(encoding='utf-8')}"))
    cells += [
        md("## 3 · Helpers: run a command and stream its output (stop on failure); GPU clock log"),
        code(f"""
import shutil, subprocess, sys, time
from pathlib import Path
WORK = Path("{WORK}")
SRC, RES = WORK / "c_cuda", WORK / "results" / "kaggle"

def sh(*cmd, cwd=SRC):
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end="")
    if p.wait():
        raise RuntimeError(f"exit code {{p.returncode}}: {{' '.join(cmd)}}")

class ClockLog:
    \"\"\"NVML sampler (gpu_sampler.py, every 20 ms) in its own process while the GPU program runs.\"\"\"
    def __init__(self, path):
        self.path = Path(path)
    def __enter__(self):
        self.err = self.path.with_suffix(".err").open("w")
        self.proc = subprocess.Popen([sys.executable, str(SRC / "gpu_sampler.py"), str(self.path), "20"], stderr=self.err)
        time.sleep(1)
        return self
    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.err.close()
        n = max(len(self.path.read_text().splitlines()) - 1, 0) if self.path.exists() else 0
        err = self.path.with_suffix(".err").read_text().strip()
        print(f"gpu clock log: {{n}} samples" + (f" | ERROR: {{err.splitlines()[-1]}}" if err else ""))
"""),
        md("## 4 · Build, and show the linked libraries (no BLAS / OpenMP / cuBLAS / cuDNN)"),
        code("""
q = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"], capture_output=True, text=True)
first = q.stdout.strip().splitlines()[0] if q.stdout.strip() else ""
cap = first.split(",")[0].strip()
if not cap.replace(".", "").isdigit():                  # old driver without compute_cap: guess from the name
    cap = "6.0" if "P100" in first else "7.5"
ARCH = "sm_" + cap.replace(".", "")
print("GPU:", first, "-> ARCH", ARCH)
sh("make", "clean")
sh("make", "all", f"ARCH={ARCH}")
sh("ldd", "bench_seq")
sh("ldd", "bench_cuda")
print((SRC / "build_seq.txt").read_text(), (SRC / "build_cuda.txt").read_text(), sep="\\n")
"""),
        md("## 5 · Correctness checks (every kernel against float64 on the CPU)"),
        code("""
sh("./bench_seq", "--check-only")
sh("./bench_cuda", "--check-only")
"""),
        md("## 6 · Smoke test (tiny grid, kept out of the results)"),
        code("""
shutil.rmtree(WORK / "smoke", ignore_errors=True)
with ClockLog(WORK / "smoke_clocks.csv"):
    sh("./bench_cuda", "--machine", "kaggle", "--outdir", WORK / "smoke", "--quick", "--no-check")
sh("./bench_seq", "--machine", "kaggle", "--outdir", WORK / "smoke", "--quick", "--no-check")
"""),
        md(f"## 7 · {RUNS} runs: GPU first ({GPU_REPEAT} repeats, NVML clocks), then CPU ({CPU_REPEAT} repeats); shuffled sizes"),
        code(f"""
from datetime import datetime
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RES.mkdir(parents=True, exist_ok=True)
sh("bash", "-c", f"lscpu > {{RES}}/lscpu_c_cuda_{{STAMP}}.txt")
T0 = time.time()

def one_run(i):
    run = RES / f"c_cuda{{i}}_{{STAMP}}"
    run.mkdir(parents=True)
    for name in ("build_seq.txt", "build_cuda.txt"):
        shutil.copy(SRC / name, run / name)
    with ClockLog(run / "gpu_clocks.csv"):
        sh("./bench_cuda", "--machine", "kaggle", "--outdir", run, "--order-seed", i, "--repeat", {GPU_REPEAT})
    sh("./bench_seq", "--machine", "kaggle", "--outdir", run, "--order-seed", i, "--repeat", {CPU_REPEAT})
    print(f"run {{i}} done, {{(time.time() - T0) / 60:.1f}} min since the first run started")
    return run
"""),
    ]
    for i in range(1, RUNS + 1):
        cells.append(code(f"RUN{i} = one_run({i})"))
    runs = ", ".join(f"RUN{i}" for i in range(1, RUNS + 1))
    cells += [
        md("## 8 · Quick look (medians over the runs; the analysis is done offline)"),
        code(f"""
import pandas as pd
pd.set_option("display.width", 200)
def load(name):
    return pd.concat([pd.read_csv(r / name).assign(run=r.name[:7]) for r in ({runs})], ignore_index=True)
conv = load("conv.csv"); conv["ms_per_img"] = conv["mean_ms"] / conv["size"]
print("conv, ms per image"); print(conv.pivot_table(index=["device", "impl"], columns="size", values="ms_per_img", aggfunc="median").round(4).to_string())
fc = load("fc.csv")
mv, gm = fc[fc.impl.str.startswith("matvec")], fc[fc.impl.str.startswith("gemm")]
print("\\nmatvec, GB/s"); print(mv.pivot_table(index=["device", "impl"], columns="size", values="gbps", aggfunc="median").round(1).to_string())
print("\\ngemm, GFLOP/s"); print(gm.pivot_table(index=["device", "impl"], columns="size", values="gflops", aggfunc="median").round(0).to_string())
bw = load("bw.csv")
print("\\nbandwidth, GB/s"); print(bw.pivot_table(index=["device", "impl"], columns="size", values="gbps", aggfunc="median").round(1).to_string())
"""),
        md("## 9 · Package the results for download"),
        code("""
print(shutil.make_archive(str(WORK / "results_c_cuda"), "zip", WORK, "results"))
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": False,
                       "isGpuEnabled": True, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
