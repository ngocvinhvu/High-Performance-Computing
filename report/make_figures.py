"""Vẽ các hình chung của báo cáo nhóm từ dữ liệu gốc trong thư mục thành viên.

Chạy từ gốc repo:  python3 report/make_figures.py
Cần: python3, matplotlib. Hình ghi vào report/figures/.
"""
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Bảng màu tham chiếu (đã kiểm tra CVD và tương phản); dùng theo thứ tự cố định.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})


def vn(x, nd=1):
    """Số kiểu Việt Nam: dấu phẩy thập phân."""
    return f"{x:.{nd}f}".replace(".", ",")


def fig_conv_speedup():
    """§3: tỉ lệ thời gian CPU 1 luồng / GPU của lớp conv theo batch, bản C/CUDA tự viết (hieunv) so với bài báo."""
    rows = []
    table = (ROOT / "hieunv/results/analysis/c_cuda_tables.md").read_text(encoding="utf-8")
    sect = table.split("### Theo từng batch", 1)[1]
    for line in sect.splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|\s*([\d.]+)\s*\|\s*([\d.]+)–([\d.]+)\s*\|", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(6)), float(m.group(7)), float(m.group(8))))
        elif rows and not line.startswith("|"):
            break
    paper = []
    with open(ROOT / "hieunv/results/paper_digitized.csv", encoding="utf-8") as f:
        for r in csv.reader(line for line in f if not line.startswith("#")):
            if r and r[0] == "fig2":
                paper.append((int(r[2]), float(r[3])))

    b = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.fill_between(b, [r[2] for r in rows], [r[3] for r in rows], color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(b, [r[1] for r in rows], color=BLUE, marker="o")
    ax.plot([p[0] for p in paper], [p[1] for p in paper], color=ORANGE, marker="s", linestyle="--")
    ax.set_xscale("log", base=2)
    ticks = [1, 8, 16, 32, 64, 128, 256]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(x) for x in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 45)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("CPU 1 luồng / GPU (lần)")
    ax.set_title("Conv 96×11×11/4 trên ảnh 224×224: GPU nhanh hơn CPU bao nhiêu lần")
    last = rows[-1]
    ax.annotate(f"C/CUDA tự viết, T4: {vn(last[1])}×", (last[0], last[1]), xytext=(-6, -16),
                textcoords="offset points", ha="right", color=INK)
    ax.annotate(f"Bài báo, K520 (đọc từ Hình 2): {vn(paper[-1][1])}×", (paper[-1][0], paper[-1][1]),
                xytext=(-6, 8), textcoords="offset points", ha="right", color=INK)
    ax.legend(["khoảng của 3 lần chạy", "C/CUDA tự viết (hieunv), trung vị 3 lần", "bài báo (Hình 2)"],
              loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "h1_conv_speedup.png", dpi=200)
    plt.close(fig)


def fig_model_parallel():
    """§2: model parallelism trên 4× H100 (vinhvn): pipeline theo số micro-batch và tensor parallelism."""
    pipe, tensor, base = {}, {}, None
    with open(ROOT / "vinhvn/results/bench.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["impl"] != "cuda" or r["H"] != "4096" or r["status"] != "ok":
                continue
            g, t = int(r["gpus"]), float(r["step_ms"])
            if r["mode"] == "pipeline" and g == 1:
                base = t
            elif r["mode"] == "pipeline":
                pipe.setdefault(g, []).append((int(r["micro"]), t))
            elif r["mode"] == "tensor" and g > 1:
                tensor[g] = t

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1.5, 1]})
    for g, color in ((2, BLUE), (4, ORANGE)):
        pts = sorted(pipe[g])
        m = [p[0] for p in pts]
        a1.plot(m, [base / p[1] for p in pts], color=color, marker="o", label=f"đo được, {g} GPU")
        a1.plot(m, [g * x / (x + g - 1) for x in m], color=color, linestyle=":", linewidth=1.5,
                label=f"trần bong bóng S·M/(M+S−1), {g} GPU")
        best = max(pts, key=lambda p: base / p[1])
        a1.annotate(f"{vn(base / best[1], 2)}×", (best[0], base / best[1]), xytext=(0, 8),
                    textcoords="offset points", ha="center", color=INK)
    a1.axhline(1, color=INK2, linewidth=1)
    a1.set_xscale("log", base=2)
    a1.set_xticks([1, 2, 4, 8, 16])
    a1.set_xticklabels(["1", "2", "4", "8", "16"])
    a1.minorticks_off()
    a1.set_ylim(0, 4)
    a1.set_xlabel("Số micro-batch M")
    a1.set_ylabel("Tăng tốc so với 1 GPU (lần)")
    a1.set_title("Pipeline: M = 1 không tăng tốc")
    a1.legend(fontsize=8, loc="upper left")

    labels = ["Pipeline\nM = 1", "Pipeline\nM tốt nhất", "Tensor\nparallel"]
    x = range(len(labels))
    w = 0.36
    for i, (g, color) in enumerate(((2, BLUE), (4, ORANGE))):
        pts = pipe[g]
        vals = [base / dict(pts)[1], max(base / p[1] for p in pts), base / tensor[g]]
        xs = [k + (i - 0.5) * (w + 0.03) for k in x]
        bars = a2.bar(xs, vals, width=w, color=color, label=f"{g} GPU", edgecolor=SURFACE, linewidth=2)
        for bx, v in zip(xs, vals):
            a2.text(bx, v + 0.06, f"{vn(v, 2)}×", ha="center", fontsize=8, color=INK)
    a2.axhline(1, color=INK2, linewidth=1)
    a2.set_xticks(list(x))
    a2.set_xticklabels(labels, fontsize=8.5)
    a2.set_ylim(0, 4.3)
    a2.grid(axis="x", visible=False)
    a2.set_title("So sánh ba cách chia")
    a2.legend(fontsize=8, loc="upper left")
    fig.suptitle("Model parallelism, MLP 1,07 tỉ tham số (H = 4096, L = 8, B = 2048), 4× H100",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "h2_model_parallel.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_conv_speedup()
    fig_model_parallel()
    print("Đã ghi:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
