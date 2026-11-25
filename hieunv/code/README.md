# Benchmark code — replicating section 3 of "Parallel and Distributed Deep Learning"

| File | Purpose |
|---|---|
| `common.py` | timing (`bench`: warm-up runs, then timed runs, GPU synchronized), machine info, CSV |
| `bench_conv.py` | Experiment 1 (paper Fig 1–2): conv layer 96×11×11, stride 4, on 224×224 greyscale, vs batch size. Impls: `caffe` (per-image im2col + GEMM, main), `batched`, `conv2d` |
| `bench_fc.py` | Experiment 2 (paper Fig 3–4): `matvec` (main, O(n²)) and `gemm` (O(n³)) vs n |
| `run_all.py` | runs the full grid on every available device, saves one run folder |
| `aggregate.py` | combines several runs into one folder (mean, std between runs) |
| `gpu_sampler.py`, `gpu_diag.py` | NVML clock log during GPU work; wall vs host issue time vs SM clock table |
| `plot.py` | Fig 1–6 + `summary.md` / `summary.csv` for one run or an aggregated folder (error bars) |
| `analyze.py` | stage 5: compares the official runs with the paper's digitized figures, writes `../results/analysis` (tables, Fig 7–9) |
| `bench_mem.py` | extra checks (PREDICTIONS.md): memory bandwidth with STREAM-like kernels, FC with B vectors, conv split into im2col/GEMM |
| `make_bandwidth_notebook.py` | builds `../kaggle/hpc_bandwidth_kaggle.ipynb` for `bench_mem.py` (3 runs, about 10 minutes) |
| `make_c_cuda_notebook.py` | builds `../kaggle/hpc_c_cuda_kaggle.ipynb` for the library-free version in `../c_cuda` (sequential C + hand-written CUDA, 3 runs) |

Every benchmark first checks that its implementations agree (caffe and batched vs conv2d,
matvec/gemm vs float64), so a wrong kernel is never timed.

## Laptop (CPU only)

```powershell
D:\MASTER\HPC\.venv\Scripts\Activate.ps1
cd D:\MASTER\HPC\code
python run_all.py --machine laptop --quick      # smoke test, a few seconds
python run_all.py --machine laptop              # full grid
python plot.py ..\results\laptop\<run folder>
```

Before the full run: plug in the charger, Windows power mode "Best performance", close other apps.

## Kaggle (CPU + GPU on the same machine)

Use the self-contained notebook `../kaggle/hpc_replication_kaggle.ipynb` (it writes these
.py files itself, nothing else to upload). Regenerate it after changing any code here:
`python make_kaggle_notebook.py`.

1. kaggle.com → Create → New Notebook → File → Import Notebook → pick the .ipynb.
2. Settings → Accelerator: **GPU T4 x2** (recent PyTorch builds may no longer support the P100).
3. Run All: a cold-GPU diagnostic, then 3 full runs (GPU first, 30 repeats), about 35 minutes; a failing
   step stops the notebook. Then download `results_kaggle.zip` (Output panel) into `results\kaggle\`.

Results go to `<outdir>/<machine>/<tag>/` (`env.json`, `conv.csv`, `fc.csv`, `log.txt`);
outdir is `../results` locally and `/kaggle/working/results` on Kaggle.

## Measurement rules

- 3 warm-up + 10 timed runs per point (paper: 10; the Kaggle notebook uses 30), mean ± std reported.
- `caffe` and `batched` write into preallocated buffers. `conv2d` cannot, so on CPU it includes allocating its
  output. Runs made before 2026-11-22 also allocated in `batched`: do not use their CPU `batched`/`conv2d` series.
- GPU: 5 s busy warm-up before each implementation, shuffled sizes (`--order-seed`).
- Data already on the device; host↔GPU copies are not timed (the paper times forward only).
- float32, TF32 disabled on GPU (the paper's GPU had plain FP32 only).
- CPU baseline is 1 thread (the paper's abstract says "single core CPU"); `max` threads is an extension.

## Bandwidth checks (separate from the paper's runs)

Predictions are in `../PREDICTIONS.md`, written before any run. Laptop smoke test and one full run:

```powershell
python bench_mem.py --machine laptop --quick
python bench_mem.py --machine laptop
```

Kaggle: import `../kaggle/hpc_bandwidth_kaggle.ipynb` (Settings -> Accelerator GPU T4 x2, Run All, about 10 minutes),
download `results_bandwidth.zip` into `results\kaggle\`. A run folder has `bw.csv`, `fcb.csv`, `step.csv`, `env.json`, `gpu_clocks.csv`.

Useful flags: `--threads 1,4,max`, `--batches 1,8,64`, `--skip fc`, `--devices cpu`, `--repeat 20`.
