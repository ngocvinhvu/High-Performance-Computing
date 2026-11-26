#!/usr/bin/env python3
"""Draw the report figures from the CSVs in runs/ into docs/figures/ (PNG, 200 dpi).

  python3 scripts/make_figures.py
Needs matplotlib. Colors: fixed categorical order, one color per method in every figure.
"""
import csv, os, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
COLOR = {"tree": "#2a78d6", "ring": "#eb6834", "rd": "#1baf7a", "host_ref": "#eda100"}
MARK = {"tree": "o", "ring": "s", "rd": "^", "host_ref": "D"}
LABEL = {"tree": "tree (flat)", "ring": "ring", "rd": "recursive doubling", "host_ref": "host_ref (via CPU)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
})

def rows(path):
    return list(csv.DictReader(open(os.path.join(ROOT, path))))

def size_label(b):
    b = int(b)
    return f"{b // 1048576} MB" if b >= 1048576 else f"{b // 1024} KB"

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join("docs/figures", name))

# ---- Fig 1: speedup vs batch size (sweep) ----------------------------------------------
# Speedups from summary.md (steady-state epoch, median of epochs 2..5 and of 3 repeats, computed
# from unrounded times), so the plotted points equal the report's table; sweep.csv only has the mean
# over all epochs, which includes the warm-up epoch.
batches = [256, 1024, 2048, 4096, 8192]
spd = {}
for line in open(os.path.join(ROOT, "runs/kaggle-sweep-2026-11-23/summary.md")):
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) == 8 and c[0].isdigit():
        spd[(int(c[0]), c[2])] = float(c[5].rstrip("×"))
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.axhline(1.0, color=NEUTRAL, lw=1.2, ls="--")
ax.text(256, 1.02, "1 GPU (break-even)", color=INK2, va="bottom", fontsize=9)
for m in ("tree", "ring", "rd"):
    ys = [spd[(b, m)] for b in batches]
    ax.plot(batches, ys, color=COLOR[m], marker=MARK[m], label=LABEL[m],
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.text(batches[-1] * 1.05, ys[-1] + {"tree": -0.045, "ring": -0.005, "rd": 0.035}[m], LABEL[m],
            color=INK, va="center", fontsize=9)
ax.set_xscale("log", base=2)
ax.set_xticks(batches, [str(b) for b in batches])
ax.set_xlim(200, 8192 * 2.4)
ax.set_ylim(0.7, 1.8)
ax.set_xlabel("Global batch size")
ax.set_ylabel("Speedup of 2 GPUs over 1 GPU\n(steady-state epoch time)")
ax.set_title("2× T4 speedup grows with batch size, levelling off near 1.6×")
ax.legend(loc="upper left")
save(fig, "fig1_speedup_vs_batch.png")

# ---- Fig 2 and 3: allreduce time and bus bandwidth vs message size (final run) ----------
ar = rows("runs/kaggle-final-2026-11-22/allreduce.csv")
by = defaultdict(list)
for r in ar:
    by[r["method"]].append((int(r["msg_bytes"]), float(r["time_ms"]), float(r["busbw_GBs"])))
order = ("tree", "ring", "rd", "host_ref")
for fig_name, idx, ylab, title, logy in (
        ("fig2_allreduce_time.png", 1, "Time per allreduce (ms, log scale)",
         "Allreduce time vs message size, 2× T4", True),
        ("fig3_allreduce_busbw.png", 2, "Bus bandwidth (GB/s)",
         "Allreduce bandwidth vs message size, 2× T4", False)):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in order:
        pts = sorted(by[m])
        ax.plot([p[0] for p in pts], [p[idx] for p in pts], color=COLOR[m], marker=MARK[m],
                markersize=5, label=LABEL[m], markeredgecolor=SURFACE, markeredgewidth=1)
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.axvline(1628200, color=NEUTRAL, lw=1.2, ls="--")
        ax.text(1628200 * 1.1, 0.02, "model gradient\n1.63 MB", color=INK2, fontsize=8, va="bottom")
    else:
        ax.set_ylim(0, 5.5)
    ticks = [4096, 65536, 1048576, 16777216, 67108864]
    ax.set_xticks(ticks, [size_label(t) for t in ticks])
    ax.set_xlabel("Message size")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(loc="upper left" if logy else "center left", bbox_to_anchor=None if logy else (0, 0.62))
    save(fig, fig_name)

# ---- Fig 4: share of GPU kernel time (profile) -----------------------------------------
ks = rows("runs/kaggle-profile-2026-11-22/kernel_share.csv")
share = defaultdict(float)
for r in ks:
    share[(int(r["hidden"]), "matmul" if "matmul" in r["kernel"] else "other")] += float(r["share_pct"])
hs = [512, 1024, 2048]
fig, ax = plt.subplots(figsize=(7, 2.8))
ylab = [f"H = {h}" for h in hs]
mm = [share[(h, "matmul")] for h in hs]
ot = [share[(h, "other")] for h in hs]
ax.barh(ylab, mm, color=COLOR["tree"], height=0.55, label="matrix multiplication",
        edgecolor=SURFACE, linewidth=2)
ax.barh(ylab, ot, left=mm, color=NEUTRAL, height=0.55, label="all other kernels",
        edgecolor=SURFACE, linewidth=2)
for y, v in zip(ylab, mm):
    ax.text(v / 2, y, f"{v:.1f} %", color="white", va="center", ha="center", fontsize=9, fontweight="bold")
ax.set_xlim(0, 100)
ax.set_xlabel("Share of GPU kernel time (%)")
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_title("Matrix multiplication dominates GPU time (nsys, 1× T4)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
save(fig, "fig4_matmul_share.png")

# ---- Fig 5: epoch time split into exposed communication and the rest (final run, batch 256) ----
# Steady-state epoch (epochs 2..10) and exposed comm, medians of 3 repeats, from summary.md, so the
# bars match the report's table (training.csv only has the mean over all epochs).
fin = {}
for line in open(os.path.join(ROOT, "runs/kaggle-final-2026-11-22/summary.md")):
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) == 9 and c[1].isdigit() and c[0] != "seq (CPU)":
        fin[c[0]] = (float(c[3].split()[0]), float(c[5].split()[0]))
names = ["1 GPU", "host_ref", "tree", "ring", "rd"]
tot = [fin[n][0] * 1000 for n in names]
com = [t * fin[n][1] / 100 for t, n in zip(tot, names)]
rest = [t - c for t, c in zip(tot, com)]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(names, rest, color=NEUTRAL, width=0.55, label="compute and other", edgecolor=SURFACE, linewidth=2)
ax.bar(names, com, bottom=rest, color=COLOR["ring"], width=0.55, label="exposed communication",
       edgecolor=SURFACE, linewidth=2)
for n, t, c in zip(names, tot, com):
    ax.text(n, t + 6, f"{t:.0f} ms" + (f"\n{100 * c / t:.0f} % comm" if c / t > 0.05 else ""),
            ha="center", va="bottom", fontsize=8.5, color=INK)
ax.set_ylim(0, max(tot) * 1.3)
ax.set_ylabel("Steady-state epoch time (ms)")
ax.grid(axis="x", visible=False)
ax.set_title("Epoch time at batch 256: half of each 2-GPU step is communication")
ax.legend(loc="upper right")
save(fig, "fig5_epoch_breakdown.png")

# ---- Fig 6: allreduce bandwidth, NVLink vs through host (4× A100 SXM) -------------------
A100 = "runs/runpod-a100-2026-11-23"
fig, axs = plt.subplots(1, 2, figsize=(9, 4.0), sharey=True)
for ax, k in zip(axs, (2, 4)):
    for mode, ls, tag in (("p2p", "-", "NVLink"), ("host", "--", "via host (PCIe)")):
        by = defaultdict(list)
        for r in rows(f"{A100}/{mode}/allreduce_k.csv"):
            if int(r["k"]) == k:
                by[r["method"]].append((int(r["msg_bytes"]), float(r["busbw_GBs"])))
        for m in ("tree", "ring", "rd"):
            pts = sorted(by[m])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR[m], ls=ls, marker=MARK[m],
                    markersize=4, markeredgecolor=SURFACE, markeredgewidth=0.8,
                    label=f"{m}, {tag}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.axvline(1628200, color=NEUTRAL, lw=1.2, ls=":")
    ticks = [4096, 65536, 1048576, 16777216, 268435456]
    ax.set_xticks(ticks, [size_label(t) for t in ticks])
    ax.set_xlabel("Message size")
    ax.set_title(f"k = {k}")
axs[0].set_ylabel("Bus bandwidth (GB/s, log scale)")
axs[0].text(1628200 * 1.15, 0.02, "gradient\n1.63 MB", color=INK2, fontsize=8)
axs[1].legend(loc="upper left", fontsize=7.5, ncol=1, handlelength=3.5)
fig.suptitle("Allreduce bandwidth on 4× A100 SXM: NVLink (solid) vs through host memory (dashed)",
             x=0.01, ha="left", fontweight="bold", fontsize=11)
save(fig, "fig6_a100_nvlink_vs_host.png")

# ---- Fig 7: training speedup vs number of GPUs, NVLink vs through host (4× A100 SXM) ----
def speedups(path):
    # speedup column of scaling_summary.md (from unrounded steady-state times), as in the report
    d = {}
    for line in open(os.path.join(ROOT, path)):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 8 and c[0].isdigit():
            d[(int(c[0]), c[2])] = float(c[5].rstrip("×"))
    return d
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, B in zip(axs, (256, 4096)):
    ax.axhline(1.0, color=NEUTRAL, lw=1.2, ls="--")
    for mode, ls, tag in (("p2p", "-", "NVLink"), ("host", "--", "via host")):
        d = speedups(f"{A100}/{mode}/scaling_summary.md")
        for m in ("tree", "ring", "rd"):
            ys = [1.0] + [d[(B, f"{m} (k={k})")] for k in (2, 4)]
            ax.plot([1, 2, 4], ys, color=COLOR[m], ls=ls, marker=MARK[m],
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"{m}, {tag}")
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel("Number of GPUs (k)")
    ax.set_title(f"Global batch {B}")
axs[0].set_ylabel("Speedup vs 1 GPU")
axs[0].set_ylim(0, 1.8)
axs[1].legend(loc="upper left", fontsize=7.5, ncol=2, handlelength=3.5)
fig.suptitle("Training speedup on 4× A100 SXM: NVLink (solid) vs through host memory (dashed)",
             x=0.01, ha="left", fontweight="bold", fontsize=11)
save(fig, "fig7_a100_speedup_vs_k.png")
