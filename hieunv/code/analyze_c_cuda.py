"""Analyze the library-free runs (../c_cuda: sequential C on the CPU, hand-written CUDA on the GPU) as planned in
PLAN_C_CUDA.md, sections 5 and 6.

    python analyze_c_cuda.py        # reads ../results/kaggle/c_cuda{1,2,3}_<stamp>, writes ../results/analysis/c_cuda_tables.md

Rules fixed in the plan before the run: medians over the 3 runs with the min-max of the runs; conv fit
t(B) = a + b*B on batch 16-256; matvec exponent on n >= 4096 and on n >= 1024; GPU work in cycles (us x measured MHz)
and in % of the FP32 peak at the measured clock (T4: 5.12 TFLOP/s per GHz).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STAMP = "20261125_050209"
FLOPS_PER_IMAGE = 2 * 96 * 121 * 2916
PEAK_PER_GHZ = 5.12e3                       # GFLOP/s per GHz of SM clock, Tesla T4 (2560 FP32 lanes x 2)
FIT_BATCH = (16, 256)
# PyTorch reference numbers (ANALYSIS.md, results/analysis/mem_tables.md, Kaggle 22/11/2026)
TORCH = {"conv_ratio": 27.8, "cpu_caffe_ms": 1.76, "cpu_im2col_ms": 0.80, "cpu_gemm_gflops": 88,
         "gpu_im2col_kcyc": 43.6, "gpu_gemm_peak": 25.7, "gpu_caffe_kcyc": 95.5, "matvec_ratio_16384": 21.7}


def md_table(df, fmt="{:.3g}"):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else fmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    return "\n".join([head, sep] + ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)])


def spread(values, fmt="{:.2f}"):
    v = np.asarray(values, float)
    return f"{fmt.format(v.min())}–{fmt.format(v.max())}"


def load(res):
    runs = [res / "kaggle" / f"c_cuda{i}_{STAMP}" for i in (1, 2, 3)]
    data = {}
    for name in ("conv", "fc", "bw", "fcb", "step"):
        data[name] = pd.concat([pd.read_csv(r / f"{name}.csv").assign(run=i + 1, runpath=str(r))
                                for i, r in enumerate(runs)], ignore_index=True)
    clocks = {str(r): pd.read_csv(r / "gpu_clocks.csv", on_bad_lines="skip").apply(pd.to_numeric, errors="coerce").dropna()
              for r in runs}
    return data, clocks


def add_clock(df, clocks):
    """Mean SM clock (MHz) inside each GPU row's timing window; nearest sample if the window is shorter than 20 ms."""
    mhz = []
    for x in df.itertuples():
        if x.device != "cuda":
            mhz.append(np.nan)
            continue
        log = clocks[x.runpath]
        h = log[(log["t"] >= x.t_start) & (log["t"] <= x.t_end)]
        if h.empty:
            h = log.iloc[[(log["t"] - 0.5 * (x.t_start + x.t_end)).abs().argmin()]]
        mhz.append(h["sm_mhz"].mean())
    return df.assign(mhz=mhz)


def fit_line(x, y):
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, b, r2


def exponent(n, t):
    return np.polyfit(np.log(n), np.log(t), 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE.parent / "results")
    args = ap.parse_args()
    data, clocks = load(args.results)
    conv, fc, bw, fcb, step = (add_clock(data[k], clocks) for k in ("conv", "fc", "bw", "fcb", "step"))
    L = ["# Bản C tuần tự và CUDA tự viết: bảng số (Kaggle T4, 3 lần chạy, sinh bởi code/analyze_c_cuda.py)", "",
         f"Dữ liệu: `results/kaggle/c_cuda{{1,2,3}}_{STAMP}`. Trung vị qua 3 lần chạy, kèm khoảng min–max của từng lần. "
         "CPU: C 1 luồng. GPU: kernel tự viết. Trần FP32 của T4 ở xung đo = 5,12 TFLOP/s mỗi GHz.", ""]

    # ---------------------------------------------------------------- 1. conv vs batch (Fig 1-2), E1, E4, E9
    rows, slopes = [], {}
    for (dev, impl), g in conv.groupby(["device", "impl"]):
        per = []
        for run, h in g.groupby("run"):
            h = h[(h["size"] >= FIT_BATCH[0]) & (h["size"] <= FIT_BATCH[1])]
            if len(h) >= 3:
                per.append((run,) + fit_line(h["size"].to_numpy(float), h["mean_ms"].to_numpy()))
        if not per:
            continue
        p = pd.DataFrame(per, columns=["run", "a", "b", "r2"])
        slopes[(dev, impl)] = p.set_index("run")["b"]
        rows.append({"thiết bị": dev, "cài đặt": impl, "số batch trong khoảng khớp": len(g[(g["size"] >= 16)]) // 3,
                     "a (ms)": p["a"].median(), "b (ms/ảnh)": p["b"].median(), "b từng lần": spread(p["b"], "{:.4f}"),
                     "R² nhỏ nhất": p["r2"].min()})
    L += ["## 1. Conv theo batch: khớp t(B) = a + b·B trên batch 16–256 (E1)", "", md_table(pd.DataFrame(rows), "{:.4g}"), ""]

    ratio = slopes[("cpu", "caffe")] / slopes[("cuda", "caffe")]
    L += [f"**Tỷ lệ độ dốc CPU `caffe` / GPU `caffe`** (E4): trung vị {ratio.median():.1f}×, từng lần {spread(ratio, '{:.1f}')} "
          f"(bài báo khoảng 40×, PyTorch {TORCH['conv_ratio']}×).", ""]
    for impl in ("batched", "direct"):
        r = slopes[("cpu", "caffe")] / slopes[("cuda", impl)]
        L.append(f"- CPU `caffe` / GPU `{impl}`: {r.median():.1f}× ({spread(r, '{:.1f}')}).")
    r = slopes[("cpu", "direct")] / slopes[("cpu", "caffe")]
    L += [f"- CPU `direct` / CPU `caffe` (độ dốc): {r.median():.2f}× ({spread(r)}).", ""]

    # speedup per batch, median over runs
    c = conv.assign(ms_img=conv["mean_ms"] / conv["size"])
    piv = c.pivot_table(index=["run", "size"], columns=["device", "impl"], values="ms_img")
    sp = pd.DataFrame({"batch": piv.index.get_level_values("size")})
    out = []
    for b, g in piv.groupby(level="size"):
        rr = g[("cpu", "caffe")].to_numpy() / g[("cuda", "caffe")].to_numpy()
        row = {"batch": b, "CPU caffe ms/ảnh": g[("cpu", "caffe")].median(), "GPU caffe µs/ảnh": 1e3 * g[("cuda", "caffe")].median(),
               "GPU batched µs/ảnh": 1e3 * g[("cuda", "batched")].median(), "GPU direct µs/ảnh": 1e3 * g[("cuda", "direct")].median(),
               "speedup caffe": np.median(rr), "từng lần": spread(rr, "{:.1f}")}
        if ("cpu", "direct") in g.columns and not g[("cpu", "direct")].isna().all():
            row["CPU direct ms/ảnh"] = g[("cpu", "direct")].median()
        out.append(row)
    t = pd.DataFrame(out)
    L += ["### Theo từng batch (thời gian mỗi ảnh, trung vị 3 lần)", "", md_table(t, "{:.3g}"), ""]

    # GPU cycles per image for the conv impls at batch 256
    g = c[(c.device == "cuda") & (c["size"] == 256)]
    rows = []
    for impl, h in g.groupby("impl"):
        kcyc = h["ms_img"] * 1e3 * h["mhz"] / 1e3
        gf = FLOPS_PER_IMAGE / (h["ms_img"] * 1e-3) / 1e9
        rows.append({"cài đặt (GPU, batch 256)": impl, "µs/ảnh": 1e3 * h["ms_img"].median(), "xung MHz": h["mhz"].median(),
                     "nghìn chu kỳ/ảnh": kcyc.median(), "từng lần": spread(kcyc, "{:.1f}"),
                     "% trần FP32 ở xung đo": float(np.median(100 * gf / (PEAK_PER_GHZ * h["mhz"] / 1e3)))})
    L += ["### Conv trên GPU ở batch 256, tính theo chu kỳ", "", md_table(pd.DataFrame(rows), "{:.3g}"), ""]

    # ---------------------------------------------------------------- 2. step split, E2, E3
    s = step.assign(us=step["mean_ms"] / step["size"] * 1e3)
    L += ["## 2. Tách conv kiểu Caffe (µs/ảnh, trung vị) (E2, E3)", "",
          md_table(s.pivot_table(index=["device", "impl"], columns="size", values="us", aggfunc="median").round(1).reset_index()
                   .rename(columns=str), "{:.1f}"), ""]
    rows = []
    for dev in ("cpu", "cuda"):
        for impl in ("im2col", "gemm", "caffe"):
            h = s[(s.device == dev) & (s.impl == impl) & (s["size"] == 256)]
            gf = FLOPS_PER_IMAGE / (h["us"] * 1e-6) / 1e9
            row = {"thiết bị": dev, "bước (batch 256)": impl, "µs/ảnh": h["us"].median(), "từng lần": spread(h["us"], "{:.1f}"),
                   "GFLOP/s": np.nan if impl == "im2col" else gf.median()}
            if dev == "cuda":
                kcyc = h["us"] * h["mhz"] / 1e3
                row.update({"xung MHz": h["mhz"].median(), "nghìn chu kỳ/ảnh": kcyc.median(),
                            "% trần FP32 ở xung đo": np.nan if impl == "im2col" else
                            float(np.median(100 * gf / (PEAK_PER_GHZ * h["mhz"] / 1e3)))})
            rows.append(row)
    L += [md_table(pd.DataFrame(rows), "{:.3g}"), "",
          f"PyTorch (cùng loại máy Kaggle, 22/11): CPU im2col {TORCH['cpu_im2col_ms']} ms/ảnh, GEMM MKL {TORCH['cpu_gemm_gflops']} GFLOP/s, "
          f"caffe {TORCH['cpu_caffe_ms']} ms/ảnh; GPU im2col {TORCH['gpu_im2col_kcyc']}k chu kỳ, GEMM cuBLAS {TORCH['gpu_gemm_peak']}% trần, "
          f"caffe {TORCH['gpu_caffe_kcyc']}k chu kỳ/ảnh.", ""]

    # ---------------------------------------------------------------- 3. matvec vs n (Fig 3-4), E1, E5, E6, E7
    mv = fc[fc.impl.str.startswith("matvec")]
    rows = []
    for (dev, impl), g in mv.groupby(["device", "impl"]):
        k4 = [exponent(h[h["size"] >= 4096]["size"], h[h["size"] >= 4096]["mean_ms"]) for _, h in g.groupby("run")]
        k1 = [exponent(h[h["size"] >= 1024]["size"], h[h["size"] >= 1024]["mean_ms"]) for _, h in g.groupby("run")]
        rows.append({"thiết bị": dev, "cài đặt": impl, "k (n ≥ 4096)": np.median(k4), "từng lần": spread(k4),
                     "k (n ≥ 1024)": np.median(k1), "từng lần ": spread(k1)})
    L += ["## 3. Matvec theo n: số mũ k của t ∝ nᵏ (E1)", "", md_table(pd.DataFrame(rows), "{:.3f}"), ""]
    L += ["### Băng thông của matvec (GB/s, trung vị)", "",
          md_table(mv.pivot_table(index=["device", "impl"], columns="size", values="gbps", aggfunc="median").round(1).reset_index()
                   .rename(columns=str), "{:.1f}"), ""]
    p = mv.pivot_table(index=["run", "size"], columns=["device", "impl"], values="mean_ms")
    rows = []
    for n, g in p.groupby(level="size"):
        r = g[("cpu", "matvec")].to_numpy() / g[("cuda", "matvec")].to_numpy()
        rn = g[("cpu", "matvec_naive")].to_numpy() / g[("cpu", "matvec")].to_numpy()
        gn = g[("cuda", "matvec_naive")].to_numpy() / g[("cuda", "matvec")].to_numpy()
        rows.append({"n": n, "speedup GPU/CPU (matvec)": np.median(r), "từng lần": spread(r, "{:.1f}"),
                     "CPU: naive chậm hơn (lần)": np.median(rn), "GPU: naive chậm hơn (lần)": np.median(gn)})
    L += ["### Speedup matvec GPU/CPU theo n", "", md_table(pd.DataFrame(rows), "{:.3g}"), "",
          f"PyTorch: {TORCH['matvec_ratio_16384']}× ở n = 16384.", ""]

    # ---------------------------------------------------------------- 4. bandwidth, E5-E7
    L += ["## 4. Băng thông đo độc lập (GB/s, trung vị) (E5, E6, E7)", "",
          md_table(bw.pivot_table(index=["device", "impl"], columns="size", values="gbps", aggfunc="median").round(1).reset_index()
                   .rename(columns=str), "{:.1f}"), ""]
    rows = []
    for dev in ("cpu", "cuda"):
        for n in (1024, 4096, 8192, 16384):
            m = bw[(bw.device == dev) & (bw["size"] == n)].pivot_table(index="run", columns="impl", values="gbps")
            best = m[["sum", "dot"]].max(axis=1)
            rows.append({"thiết bị": dev, "n": n, "đọc tốt nhất (max sum, dot)": best.median(), "từng lần": spread(best, "{:.1f}"),
                         "matvec": m["matvec"].median(), "matvec / đọc tốt nhất": (m["matvec"] / best).median(),
                         "từng lần ": spread(m["matvec"] / best)})
    L += ["### Matvec so với băng thông đọc tốt nhất", "", md_table(pd.DataFrame(rows), "{:.3g}"), ""]
    rows = []
    for n in (4096, 8192, 16384):
        g = bw[(bw.device == "cuda") & (bw["size"] == n)].pivot_table(index="run", columns="impl", values="gbps")
        cc = bw[(bw.device == "cpu") & (bw["size"] == n)].pivot_table(index="run", columns="impl", values="gbps")
        rb = g[["sum", "dot"]].max(axis=1) / cc[["sum", "dot"]].max(axis=1)
        rm = g["matvec"] / cc["matvec"]
        rows.append({"n": n, "tỷ lệ băng thông đọc GPU/CPU": rb.median(), "từng lần": spread(rb, "{:.1f}"),
                     "tỷ lệ matvec GPU/CPU": rm.median(), "từng lần ": spread(rm, "{:.1f}"),
                     "lệch (matvec / băng thông − 1)": (rm / rb - 1).median()})
    L += ["### Tỷ lệ matvec so với tỷ lệ băng thông, cùng lần chạy (E7)", "", md_table(pd.DataFrame(rows), "{:.3g}"), ""]

    # ---------------------------------------------------------------- 5. GEMM ladder, E8
    gm = fc[fc.impl.str.startswith("gemm")]
    L += ["## 5. Bậc thang GEMM (GFLOP/s, trung vị) (E8)", "",
          md_table(gm.pivot_table(index=["device", "impl"], columns="size", values="gflops", aggfunc="median").round(1).reset_index()
                   .rename(columns=str), "{:.1f}"), ""]
    p = gm.pivot_table(index=["run", "size"], columns=["device", "impl"], values="gflops")
    rows = []
    for n, g in p.groupby(level="size"):
        row = {"n": n}
        if not g[("cpu", "gemm")].isna().all():
            row["CPU gemm / ikj"] = (g[("cpu", "gemm")] / g[("cpu", "gemm_ikj")]).median()
        if not g[("cpu", "gemm_naive")].isna().all():
            row["CPU ikj / naive"] = (g[("cpu", "gemm_ikj")] / g[("cpu", "gemm_naive")]).median()
        row["GPU gemm / tiled"] = (g[("cuda", "gemm")] / g[("cuda", "gemm_tiled")]).median()
        row["GPU tiled / naive"] = (g[("cuda", "gemm_tiled")] / g[("cuda", "gemm_naive")]).median()
        rows.append(row)
    L += [md_table(pd.DataFrame(rows), "{:.2f}"), ""]
    h = gm[(gm.device == "cuda")]
    rows = []
    for (impl, n), x in h.groupby(["impl", "size"]):
        if n < 1024:
            continue
        pk = 100 * x["gflops"] / (PEAK_PER_GHZ * x["mhz"] / 1e3)
        rows.append({"cài đặt GPU": impl, "n": n, "GFLOP/s": x["gflops"].median(), "xung MHz": x["mhz"].median(),
                     "% trần FP32 ở xung đo": pk.median(), "từng lần": spread(pk, "{:.1f}")})
    L += ["### GEMM trên GPU theo % trần FP32 ở xung đo", "", md_table(pd.DataFrame(rows), "{:.3g}"), ""]

    # ---------------------------------------------------------------- 6. FC with B vectors
    rows = []
    for n in sorted(fcb["size"].unique()):
        for B in (1, 16, 64, 128):
            impl = f"mm{B}"
            cc = fcb[(fcb.device == "cpu") & (fcb["size"] == n) & (fcb.impl == impl)].set_index("run")
            g = fcb[(fcb.device == "cuda") & (fcb["size"] == n) & (fcb.impl == impl)].set_index("run")
            r = cc.mean_ms / g.mean_ms
            rows.append({"n": n, "B": B, "tỷ lệ CPU/GPU": r.median(), "từng lần": spread(r, "{:.1f}"),
                         "CPU GFLOP/s": cc.gflops.median(), "GPU GFLOP/s": g.gflops.median(), "GPU xung MHz": g.mhz.median()})
    L += ["## 6. FC với B vector: (n×n)·(n×B)", "", md_table(pd.DataFrame(rows), "{:.3g}"), ""]

    out = args.results / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "c_cuda_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
